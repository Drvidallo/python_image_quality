"""Train the learned quality models on the built dataset.

Learned heads:
- Classification: readable_overall (blur AND brightness AND glare), per-dimension
  readability for blur, brightness and glare.
- Regression: severity per learned dimension plus the learned composite severity.

Resolution is deterministic (no learned model): the rule from feature_spec.json is
applied to the measured warped-document short side. The report also evaluates the
combined system (learned overall AND resolution rule) against the 4-dimension label.

Operating thresholds are auto-selected from the validation precision-recall curve
for a per-class target recall, with a max-F1 fallback.

Usage:
    python train_models.py --config config.yaml
    python train_models.py --config config.yaml --force
"""

import argparse
import json
import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_sample_weight

from data_common import load_config, resolution_readable

CLASSIFICATION_HEADS = [
    ("readable_overall", "readable_overall_learned"),
    ("readable_blur", "readable_blur"),
    ("readable_brightness", "readable_brightness"),
    ("readable_glare", "readable_glare"),
]

REGRESSION_HEADS = [
    ("blur_severity", "blur_severity"),
    ("brightness_severity", "brightness_severity"),
    ("glare_severity", "glare_severity"),
    ("severity_index", "severity_index_learned"),
]


def _feature_list(spec: dict) -> list:
    features = spec.get("selected_features") or spec.get("top_k_features") or []
    return list(features)


def _prepare(features_frame: pd.DataFrame, feature_columns: list):
    data = features_frame.copy()
    for column in feature_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def _select_threshold(y_true, probabilities, target_recall: float) -> float:
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(np.unique(y_true)) < 2 or len(probabilities) < 3:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return 0.5
    precision = precision[:-1]
    recall = recall[:-1]
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    if target_recall and target_recall > 0:
        meets = recall >= target_recall
        if meets.any():
            masked = np.where(meets, f1, -1.0)
            return float(thresholds[int(np.argmax(masked))])
    return float(thresholds[int(np.argmax(f1))])


def _classification_metrics(y_true, predictions, probabilities=None) -> dict:
    y_true = np.asarray(y_true).astype(int)
    predictions = np.asarray(predictions).astype(int)
    metrics = {
        "n": int(len(y_true)),
        "positives": int(np.sum(y_true)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
    }
    if probabilities is not None and len(np.unique(y_true)) > 1:
        probabilities = np.asarray(probabilities, dtype=float)
        metrics["roc_auc"] = float(roc_auc_score(y_true, probabilities))
        metrics["average_precision"] = float(average_precision_score(y_true, probabilities))
    else:
        metrics["roc_auc"] = None
        metrics["average_precision"] = None
    return metrics


def _binary_metrics(y_true, probabilities, threshold: float) -> dict:
    predictions = (np.asarray(probabilities, dtype=float) >= threshold).astype(int)
    return _classification_metrics(y_true, predictions, probabilities)


def _regression_metrics(y_true, y_pred) -> dict:
    mse = float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))
    return {
        "n": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else None,
    }


def _plot_confusion_matrices(report: dict, plots_dir: str):
    targets = [name for name in report if "val" in report[name]]
    if not targets:
        return None
    fig, axes = plt.subplots(1, len(targets), figsize=(4 * len(targets), 4))
    if len(targets) == 1:
        axes = [axes]
    for ax, target in zip(axes, targets):
        matrix = np.array(report[target]["val"]["confusion_matrix"])
        ax.imshow(matrix, cmap="Blues")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
        threshold = report[target].get("threshold")
        ax.set_title(f"{target}\n(thr={threshold:.3f})" if threshold is not None else target)
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
    fig.suptitle("Confusion matrices (validation, auto-selected thresholds)")
    fig.tight_layout()
    path = os.path.join(plots_dir, "confusion_matrices.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _plot_roc_curves(models: dict, label_map: dict, data: pd.DataFrame, feature_columns: list, plots_dir: str):
    from sklearn.metrics import roc_curve

    val = data[data["split"] == "val"]
    if not len(val):
        return None
    fig, ax = plt.subplots(figsize=(7, 6))
    plotted = False
    for target, model in models.items():
        label_column = label_map.get(target, target)
        if label_column not in val.columns:
            continue
        y_true = val[label_column].astype(int)
        if len(np.unique(y_true)) < 2:
            continue
        probabilities = model.predict_proba(val[feature_columns])[:, 1]
        fpr, tpr, _ = roc_curve(y_true, probabilities)
        auc = roc_auc_score(y_true, probabilities)
        ax.plot(fpr, tpr, label=f"{target} (AUC={auc:.3f})")
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC curves (validation)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(plots_dir, "roc_curves.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _plot_regression(models: dict, label_map: dict, data: pd.DataFrame, feature_columns: list, plots_dir: str):
    val = data[data["split"] == "val"]
    targets = [name for name in models if name in label_map and label_map[name] in val.columns]
    if not len(val) or not targets:
        return None
    fig, axes = plt.subplots(1, len(targets), figsize=(4 * len(targets), 4))
    if len(targets) == 1:
        axes = [axes]
    for ax, target in zip(axes, targets):
        y_true = val[label_map[target]].to_numpy()
        y_pred = models[target].predict(val[feature_columns])
        ax.scatter(y_true, y_pred, s=6, alpha=0.4)
        limit = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(limit, limit, "k--", linewidth=1)
        ax.set_title(f"{target}\n(label: {label_map[target]})")
        ax.set_xlabel("actual")
        ax.set_ylabel("predicted")
    fig.suptitle("Regression predicted vs actual (validation)")
    fig.tight_layout()
    path = os.path.join(plots_dir, "regression_pred_vs_actual.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _plot_importance(importances: dict, plots_dir: str):
    names = list(importances.keys())
    if not names:
        return None
    fig, axes = plt.subplots(1, len(names), figsize=(7 * len(names), 6))
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        values = importances[name]
        ordered = sorted(values.items(), key=lambda item: item[1])
        labels = [item[0] for item in ordered]
        scores = [item[1] for item in ordered]
        ax.barh(labels, scores)
        ax.set_title(f"Permutation importance: {name}")
    fig.tight_layout()
    path = os.path.join(plots_dir, "permutation_importance.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _resolution_rule_report(data: pd.DataFrame, config: dict) -> dict:
    min_side = config["labels"]["min_effective_short_side_px"]
    report = {
        "rule": "effective_short_side_px >= min_effective_short_side_px",
        "min_effective_short_side_px": min_side,
        "note": (
            "Deterministic rule; in production apply to the measured warped document short "
            "side. Synthetic rows are upscaled back to native size, so this prototype "
            "evaluation uses the parameter-derived effective size."
        ),
    }
    for split in ["val", "test"]:
        subset = data[data["split"] == split]
        if not len(subset):
            continue
        predictions = np.array(
            [resolution_readable(value, min_side) for value in subset["effective_short_side_px"]],
            dtype=int,
        )
        report[split] = _classification_metrics(subset["readable_resolution"].astype(int), predictions)
    return report


def _combined_system_report(data: pd.DataFrame, models: dict, thresholds: dict, feature_columns: list, config: dict) -> dict:
    model = models.get("readable_overall")
    if model is None:
        return {}
    min_side = config["labels"]["min_effective_short_side_px"]
    threshold = thresholds.get("readable_overall", 0.5)
    report = {
        "description": "learned_overall AND resolution_rule vs 4-dimension readable_overall",
        "threshold": threshold,
    }
    for split in ["val", "test"]:
        subset = data[data["split"] == split]
        if not len(subset):
            continue
        learned = model.predict_proba(subset[feature_columns])[:, 1] >= threshold
        rule = np.array(
            [resolution_readable(value, min_side) for value in subset["effective_short_side_px"]],
            dtype=int,
        ).astype(bool)
        combined = (learned & rule).astype(int)
        report[split] = _classification_metrics(subset["readable_overall"].astype(int), combined)
    return report


def run(config: dict, force=False):
    processed_dir = config["paths"]["processed_dir"]
    models_dir = config["paths"].get("models_dir", os.path.join(os.path.dirname(processed_dir), "models"))
    plots_dir = os.path.join(config["paths"]["plots_dir"], "model")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    report_path = os.path.join(models_dir, "model_report.json")
    if os.path.exists(report_path) and not force:
        print(f"model report already exists: {report_path} (use --force to overwrite)")
        return report_path

    features_path = os.path.join(processed_dir, "features.parquet")
    spec_path = os.path.join(processed_dir, "feature_spec.json")
    data = pd.read_parquet(features_path)
    with open(spec_path, "r", encoding="utf-8") as f:
        spec = json.load(f)
    feature_columns = _feature_list(spec)
    data = _prepare(data, feature_columns)

    train = data[data["split"] == "train"]
    val = data[data["split"] == "val"]
    test = data[data["split"] == "test"]
    print(
        f"Rows train/val/test: {len(train)}/{len(val)}/{len(test)} | features: {len(feature_columns)}"
    )
    print(
        f"Resolution head: {config['training'].get('resolution_head', 'deterministic')} "
        f"(min short side {config['labels']['min_effective_short_side_px']} px)"
    )

    seed = config["seed"]
    X_train = train[feature_columns]
    X_val = val[feature_columns]
    X_test = test[feature_columns]
    target_recalls = config.get("training", {}).get("target_recall", {})

    classification_report = {}
    classification_models = {}
    classification_label_map = {}
    thresholds = {}
    for report_name, label_column in CLASSIFICATION_HEADS:
        if label_column not in data.columns or not len(train):
            continue
        y_train = train[label_column].astype(int)
        weights = compute_sample_weight("balanced", y_train)
        model = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_depth=6, random_state=seed
        )
        model.fit(X_train, y_train, sample_weight=weights)
        val_probabilities = model.predict_proba(X_val)[:, 1]
        threshold = _select_threshold(
            val[label_column].astype(int), val_probabilities, target_recalls.get(report_name, 0.0)
        )
        entry = {
            "model_file": f"clf_{report_name}.joblib",
            "label_column": label_column,
            "threshold": threshold,
            "val": _binary_metrics(val[label_column].astype(int), val_probabilities, threshold),
        }
        if len(test):
            entry["test"] = _binary_metrics(
                test[label_column].astype(int), model.predict_proba(X_test)[:, 1], threshold
            )
        classification_report[report_name] = entry
        classification_models[report_name] = model
        classification_label_map[report_name] = label_column
        thresholds[report_name] = threshold
        joblib.dump(model, os.path.join(models_dir, entry["model_file"]))
        print(
            f"[clf] {report_name:<22} thr={threshold:.3f} "
            f"val f1={entry['val']['f1']:.3f} "
            f"auc={entry['val']['roc_auc'] if entry['val']['roc_auc'] is None else round(entry['val']['roc_auc'], 3)} "
            f"recall={entry['val']['recall']:.3f} precision={entry['val']['precision']:.3f}"
        )

    regression_report = {}
    regression_models = {}
    regression_label_map = {}
    for report_name, label_column in REGRESSION_HEADS:
        if label_column not in data.columns or not len(train):
            continue
        model = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.06, max_depth=6, random_state=seed
        )
        model.fit(X_train, train[label_column])
        entry = {
            "model_file": f"reg_{report_name}.joblib",
            "label_column": label_column,
            "val": _regression_metrics(val[label_column], model.predict(X_val)),
        }
        if len(test):
            entry["test"] = _regression_metrics(test[label_column], model.predict(X_test))
        regression_report[report_name] = entry
        regression_models[report_name] = model
        regression_label_map[report_name] = label_column
        joblib.dump(model, os.path.join(models_dir, entry["model_file"]))
        print(
            f"[reg] {report_name:<22} val mae={entry['val']['mae']:.3f} "
            f"r2={entry['val']['r2'] if entry['val']['r2'] is None else round(entry['val']['r2'], 3)}"
        )

    resolution_rule = _resolution_rule_report(data, config)
    combined_system = _combined_system_report(data, classification_models, thresholds, feature_columns, config)

    importances = {}
    if "readable_overall" in classification_models and len(val):
        importance = permutation_importance(
            classification_models["readable_overall"],
            X_val,
            val["readable_overall_learned"].astype(int),
            n_repeats=5,
            random_state=seed,
            scoring="roc_auc",
        )
        importances["readable_overall"] = dict(zip(feature_columns, importance.importances_mean.tolist()))
    if "severity_index" in regression_models and len(val):
        importance = permutation_importance(
            regression_models["severity_index"],
            X_val,
            val["severity_index_learned"],
            n_repeats=5,
            random_state=seed,
            scoring="neg_mean_absolute_error",
        )
        importances["severity_index"] = dict(zip(feature_columns, importance.importances_mean.tolist()))

    plot_paths = [
        _plot_confusion_matrices(classification_report, plots_dir),
        _plot_roc_curves(classification_models, classification_label_map, data, feature_columns, plots_dir),
        _plot_regression(regression_models, regression_label_map, data, feature_columns, plots_dir),
        _plot_importance(importances, plots_dir),
    ]
    plot_paths = [p for p in plot_paths if p]

    manifest_path = os.path.join(config["paths"]["plots_dir"], "manifest.json")
    manifest = []
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:
            manifest = []
    manifest = [item for item in manifest if item.get("group") != "model"]
    for path in plot_paths:
        manifest.append(
            {"group": "model", "name": os.path.basename(path), "path": path, "status": "ok"}
        )
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    with open(os.path.join(models_dir, "thresholds.json"), "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)

    report = {
        "seed": seed,
        "feature_columns": feature_columns,
        "rows": {"train": len(train), "val": len(val), "test": len(test)},
        "target_recall": target_recalls,
        "thresholds": thresholds,
        "label_definition": spec.get("label_definition", {}),
        "resolution_rule": resolution_rule,
        "combined_system": combined_system,
        "classification": classification_report,
        "regression": regression_report,
        "permutation_importance": importances,
        "plots": plot_paths,
        "models_dir": models_dir,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {report_path}")
    print(f"Thresholds: {thresholds}")
    if resolution_rule.get("val"):
        print(
            f"[rule] resolution        val f1={resolution_rule['val']['f1']:.3f} "
            f"accuracy={resolution_rule['val']['accuracy']:.3f}"
        )
    if combined_system.get("val"):
        print(
            f"[sys] combined overall   val f1={combined_system['val']['f1']:.3f} "
            f"recall={combined_system['val']['recall']:.3f} precision={combined_system['val']['precision']:.3f}"
        )
    print(f"Models saved to {models_dir}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="Train quality classification and regression models.")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    run(config, args.force)


if __name__ == "__main__":
    main()