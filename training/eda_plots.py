"""EDA plot flow: raw distributions, regression targets, classification labels,
cleaning/transformation comparisons, engineered features, reduction ranking and
split integrity. Writes PNGs plus a manifest.

Usage:
    python eda_plots.py --config config.yaml
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from data_common import load_config

CORE_METRICS = ["blur_var", "brightness_L", "glare_pct", "glare_blob_pct", "doc_short_side_px"]
TARGET_PLOTS = [
    "target_blur_sigma",
    "target_brightness_gamma",
    "target_brightness_ev",
    "target_glare_radius_frac",
    "target_downscale_factor",
    "severity_index",
]
SEVERITY_PLOTS = [
    "blur_severity",
    "brightness_severity",
    "glare_severity",
    "resolution_severity",
    "severity_index",
]
LABEL_PLOTS = [
    "readable_blur",
    "readable_brightness",
    "readable_glare",
    "readable_resolution",
    "readable_overall",
    "readable_overall_learned",
]


class PlotFlow:
    def __init__(self, processed_dir: str, plots_dir: str, config: dict):
        self.processed_dir = processed_dir
        self.plots_dir = plots_dir
        self.config = config
        self.manifest = []
        self.dpi = config["plots"]["dpi"]

    def _save(self, fig, group: str, name: str):
        group_dir = os.path.join(self.plots_dir, group)
        os.makedirs(group_dir, exist_ok=True)
        path = os.path.join(group_dir, name)
        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi)
        plt.close(fig)
        self.manifest.append({"group": group, "name": name, "path": path, "status": "ok"})
        print("  plot:", path)

    def _run_plot(self, group: str, name: str, fn):
        try:
            fn(group, name)
        except Exception as exc:
            self.manifest.append(
                {"group": group, "name": name, "path": "", "status": f"error: {exc}"}
            )
            print(f"  plot failed ({name}): {exc}")

    def raw_histograms(self, raw: pd.DataFrame):
        def plot(group, name):
            fig, axes = plt.subplots(2, 3, figsize=(16, 9))
            for ax, column in zip(axes.ravel(), CORE_METRICS):
                data = raw[column].replace([np.inf, -np.inf], np.nan).dropna()
                ax.hist(np.log1p(data) if column in ("blur_var", "glare_pct", "glare_blob_pct") else data, bins=50)
                suffix = " (log1p)" if column in ("blur_var", "glare_pct", "glare_blob_pct") else ""
                ax.set_title(f"{column}{suffix}")
                ax.set_xlabel(column)
            axes.ravel()[-1].axis("off")
            fig.suptitle("Raw metric distributions (all ingested frames)")
            self._save(fig, group, name)

        self._run_plot("raw", "hist_raw_metrics.png", plot)

        def plot_by_type(group, name):
            top_types = raw["document_type"].value_counts().index.tolist()
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            for ax, column in zip(axes, ["blur_var", "brightness_L", "glare_pct"]):
                data = raw[raw["document_type"].isin(top_types)]
                sns.boxplot(data=data, x="document_type", y=column, ax=ax)
                ax.set_yscale("log") if column != "brightness_L" else None
                ax.tick_params(axis="x", rotation=75)
                ax.set_title(f"{column} by document type")
            self._save(fig, group, name)

        self._run_plot("raw", "hist_by_doc_type.png", plot_by_type)

        def plot_capture(group, name):
            fig, ax = plt.subplots(figsize=(10, 5))
            raw["capture_condition"].value_counts().sort_index().plot.bar(ax=ax)
            ax.set_title("Frames per capture condition (clip prefix)")
            ax.set_ylabel("count")
            self._save(fig, group, name)

        self._run_plot("raw", "capture_condition_counts.png", plot_capture)

        def plot_missing(group, name):
            missing = raw.isna().sum().sort_values(ascending=False)
            missing = missing[missing > 0]
            fig, ax = plt.subplots(figsize=(12, 6))
            if len(missing):
                missing.plot.bar(ax=ax)
                ax.set_title("Missing values per column")
            else:
                ax.text(0.5, 0.5, "No missing values", ha="center", va="center")
                ax.set_title("Missing values per column")
                ax.axis("off")
            self._save(fig, group, name)

        self._run_plot("raw", "missingness.png", plot_missing)

    def cleaning_plots(self, raw: pd.DataFrame, features: pd.DataFrame):
        def plot_transforms(group, name):
            fig, axes = plt.subplots(2, 2, figsize=(14, 9))
            axes[0, 0].hist(raw["blur_var"].clip(lower=0.01), bins=60)
            axes[0, 0].set_title("blur_var (raw)")
            axes[0, 1].hist(np.log1p(raw["blur_var"].clip(lower=0)), bins=60)
            axes[0, 1].set_title("log1p(blur_var)")
            axes[1, 0].hist(raw["glare_pct"], bins=60)
            axes[1, 0].set_title("glare_pct (raw)")
            axes[1, 1].hist(np.log1p(raw["glare_pct"]), bins=60)
            axes[1, 1].set_title("log1p(glare_pct)")
            fig.suptitle("Transformation before/after")
            self._save(fig, group, name)

        self._run_plot("cleaning", "before_after_transforms.png", plot_transforms)

        def plot_reconciliation(group, name):
            path = os.path.join(self.processed_dir, "reconciliation_dataset.json")
            if not os.path.exists(path):
                return
            with open(path, "r", encoding="utf-8") as f:
                reconciliation = json.load(f)
            counts = reconciliation["clean"]
            labels = ["input", "missing", "out_of_range", "outliers", "near_dupes", "kept"]
            values = [
                counts["input_rows"],
                counts["dropped_missing_metrics"],
                counts["dropped_out_of_range"],
                counts["outlier_flagged"],
                counts["dropped_near_duplicates"],
                counts["kept_rows"],
            ]
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(labels, values)
            ax.set_title("Cleaning reconciliation")
            ax.tick_params(axis="x", rotation=30)
            self._save(fig, group, name)

        self._run_plot("cleaning", "reconciliation.png", plot_reconciliation)

    def target_plots(self, features: pd.DataFrame):
        def plot_targets(group, name):
            fig, axes = plt.subplots(2, 3, figsize=(16, 9))
            for ax, column in zip(axes.ravel(), TARGET_PLOTS):
                ax.hist(features[column].dropna(), bins=50)
                ax.set_title(column)
            for ax in axes.ravel()[len(TARGET_PLOTS) :]:
                ax.axis("off")
            fig.suptitle("Regression target distributions")
            self._save(fig, group, name)

        self._run_plot("targets", "hist_targets.png", plot_targets)

        def plot_severities(group, name):
            fig, axes = plt.subplots(2, 3, figsize=(16, 9))
            for ax, column in zip(axes.ravel(), SEVERITY_PLOTS):
                ax.hist(features[column].dropna(), bins=40)
                ax.set_title(column)
            for ax in axes.ravel()[len(SEVERITY_PLOTS) :]:
                ax.axis("off")
            fig.suptitle("Severity target distributions")
            self._save(fig, group, name)

        self._run_plot("targets", "hist_severities.png", plot_severities)

        def plot_feature_vs_target(group, name):
            pairs = [
                ("log_blur_var", "blur_severity"),
                ("brightness_L", "brightness_severity"),
                ("log_glare_pct", "glare_severity"),
                ("doc_short_side_px", "resolution_severity"),
            ]
            sample = features.sample(min(5000, len(features)), random_state=self.config["seed"])
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            for ax, (x, y) in zip(axes.ravel(), pairs):
                ax.hexbin(sample[x], sample[y], gridsize=40, mincnt=1)
                ax.set_xlabel(x)
                ax.set_ylabel(y)
                ax.set_title(f"{x} vs {y}")
            self._save(fig, group, name)

        self._run_plot("targets", "feature_vs_target.png", plot_feature_vs_target)

    def classification_plots(self, features: pd.DataFrame, top_features: list):
        def plot_balance(group, name):
            fig, axes = plt.subplots(2, 3, figsize=(16, 9))
            for ax, column in zip(axes.ravel(), LABEL_PLOTS):
                counts = features[column].value_counts().sort_index()
                ax.bar([str(i) for i in counts.index], counts.values)
                ax.set_title(column)
                ax.set_ylabel("count")
            for ax in axes.ravel()[len(LABEL_PLOTS) :]:
                ax.axis("off")
            fig.suptitle("Classification label balance")
            self._save(fig, group, name)

        self._run_plot("classification", "class_balance.png", plot_balance)

        def plot_balance_by_type(group, name):
            table = (
                features.groupby(["document_type", self.label_column])["sample_id"]
                .count()
                .unstack(fill_value=0)
            )
            fig, ax = plt.subplots(figsize=(8, 5))
            table.plot.bar(ax=ax)
            ax.set_title(f"{self.label_column} by document type")
            ax.set_ylabel("rows")
            self._save(fig, group, name)

        self._run_plot("classification", "class_balance_by_type.png", plot_balance_by_type)

        def plot_feature_by_class(group, name):
            columns = top_features[:4] if top_features else ["log_blur_var", "brightness_L"]
            sample = features.sample(min(20000, len(features)), random_state=self.config["seed"])
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            for ax, column in zip(axes.ravel(), columns):
                for label, color in [(0, "crimson"), (1, "steelblue")]:
                    subset = sample[sample[self.label_column] == label][column].dropna()
                    if len(subset):
                        sns.kdeplot(subset, ax=ax, label=f"{self.label_column}={label}", color=color, fill=True, alpha=0.3)
                ax.set_title(column)
                ax.legend()
            self._save(fig, group, name)

        self._run_plot("classification", "feature_hist_by_class.png", plot_feature_by_class)

    def engineering_plots(self, features: pd.DataFrame, ranking: list, selected: list):
        def plot_correlation(group, name):
            columns = [column for column, _ in ranking[:15]] or selected[:15]
            columns = [c for c in columns if c in features.columns]
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(features[columns].corr(method="spearman"), cmap="coolwarm", center=0, ax=ax)
            ax.set_title("Spearman correlation (top-ranked features)")
            self._save(fig, group, name)

        self._run_plot("engineering", "correlation_heatmap.png", plot_correlation)

        def plot_ranking(group, name):
            top = ranking[:20]
            if not top:
                return
            names = [item[0] for item in top][::-1]
            values = [item[1] for item in top][::-1]
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.barh(names, values)
            ax.set_title("Mutual information vs severity_index (top 20)")
            self._save(fig, group, name)

        self._run_plot("engineering", "mi_ranking.png", plot_ranking)

        def plot_pca(group, name):
            columns = [c for c in selected if c in features.columns]
            if len(columns) < 2:
                return
            sample = features.sample(min(8000, len(features)), random_state=self.config["seed"])
            matrix = sample[columns].fillna(0.0).to_numpy(dtype=float)
            matrix = StandardScaler().fit_transform(matrix)
            coords = PCA(n_components=2, random_state=self.config["seed"]).fit_transform(matrix)
            fig, ax = plt.subplots(figsize=(9, 7))
            scatter = ax.scatter(coords[:, 0], coords[:, 1], c=sample[self.label_column], cmap="coolwarm", s=6, alpha=0.5)
            fig.colorbar(scatter, ax=ax, label=self.label_column)
            ax.set_title("PCA of selected features (colored by readability)")
            self._save(fig, group, name)

        self._run_plot("engineering", "pca_scatter.png", plot_pca)

        def plot_quality_prior(group, name):
            fig, ax = plt.subplots(figsize=(9, 6))
            sns.kdeplot(features[features[self.label_column] == 1]["quality_prior"], ax=ax, label=f"{self.label_column}=1", fill=True)
            sns.kdeplot(features[features[self.label_column] == 0]["quality_prior"], ax=ax, label=f"{self.label_column}=0", fill=True)
            ax.set_title("quality_prior by readability")
            ax.legend()
            self._save(fig, group, name)

        self._run_plot("engineering", "quality_prior_by_class.png", plot_quality_prior)

    def split_plots(self, features: pd.DataFrame):
        def plot_counts(group, name):
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
            features["split"].value_counts().plot.bar(ax=axes[0])
            axes[0].set_title("rows per split")
            features.groupby("split")["group"].nunique().plot.bar(ax=axes[1])
            axes[1].set_title("groups per split")
            features.groupby("split")["readable_overall"].mean().plot.bar(ax=axes[2])
            axes[2].set_title("readable_overall rate per split")
            self._save(fig, group, name)

        self._run_plot("splits", "split_counts.png", plot_counts)

        def plot_targets_by_split(group, name):
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
            for ax, column in zip(axes, ["severity_index", "blur_severity", "resolution_severity"]):
                for split in ["train", "val", "test"]:
                    subset = features[features["split"] == split][column]
                    if len(subset):
                        sns.kdeplot(subset, ax=ax, label=split, fill=True, alpha=0.25)
                ax.set_title(column)
                ax.legend()
            self._save(fig, group, name)

        self._run_plot("splits", "target_dist_by_split.png", plot_targets_by_split)

    def run(self):
        os.makedirs(self.plots_dir, exist_ok=True)
        raw = pd.read_parquet(os.path.join(self.processed_dir, "raw_metrics.parquet"))
        features = pd.read_parquet(os.path.join(self.processed_dir, "features.parquet"))
        self.label_column = (
            "readable_overall_learned" if "readable_overall_learned" in features.columns else "readable_overall"
        )

        ranking_path = os.path.join(self.processed_dir, "feature_ranking.json")
        ranking = []
        if os.path.exists(ranking_path):
            with open(ranking_path, "r", encoding="utf-8") as f:
                ranking = json.load(f)

        spec_path = os.path.join(self.processed_dir, "feature_spec.json")
        top_features = []
        if os.path.exists(spec_path):
            with open(spec_path, "r", encoding="utf-8") as f:
                spec = json.load(f)
            top_features = spec.get("top_k_features", [])

        print("Generating plots...")
        self.raw_histograms(raw)
        self.cleaning_plots(raw, features)
        self.target_plots(features)
        self.classification_plots(features, top_features)
        self.engineering_plots(features, ranking, top_features)
        self.split_plots(features)

        manifest_path = os.path.join(self.plots_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2)
        ok = sum(1 for item in self.manifest if item["status"] == "ok")
        print(f"Wrote {ok}/{len(self.manifest)} plots. Manifest: {manifest_path}")
        return manifest_path


def run(processed_dir: str, plots_dir: str, config: dict):
    return PlotFlow(processed_dir, plots_dir, config).run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate EDA plots for the quality dataset.")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    run(config["paths"]["processed_dir"], config["paths"]["plots_dir"], config)
