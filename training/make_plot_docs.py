"""Generate README documentation for every plots subfolder.

Each graph is explained in technical, layman and ELI5 terms, with what to look
for, what it gives you, and live numbers from the current pipeline run.

Usage:
    python make_plot_docs.py --config config.yaml
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from data_common import load_config

GROUPS = ["raw", "cleaning", "targets", "classification", "engineering", "splits", "model"]

GROUP_INFO = {
    "raw": {
        "title": "Raw metric distributions",
        "intro": (
            "These plots describe the ingested frames BEFORE cleaning. They answer: "
            "do the four quality measurements vary enough to learn from, and are the "
            "document types and capture setups represented fairly?"
        ),
    },
    "cleaning": {
        "title": "Cleaning and transformation",
        "intro": (
            "These plots document what was removed, flagged or transformed on the way "
            "from raw frames to the model table, so every dropped row has a reason."
        ),
    },
    "targets": {
        "title": "Regression targets",
        "intro": (
            "These plots describe the synthetic degradation severity targets used for "
            "the regression head: how strongly each flaw (blur, brightness, glare, "
            "resolution) was applied and whether features track them."
        ),
    },
    "classification": {
        "title": "Classification labels",
        "intro": (
            "These plots describe the readability pass/fail labels (overall and per "
            "dimension) and how well the engineered features separate the classes."
        ),
    },
    "engineering": {
        "title": "Feature engineering and reduction",
        "intro": (
            "These plots justify the final feature set: which features carry signal, "
            "which are redundant, and how separable the classes look in feature space."
        ),
    },
    "splits": {
        "title": "Train / validation / test splits",
        "intro": (
            "These plots verify that the splits are leak-free (groups never shared), "
            "balanced, and that the held-out test documents are genuinely different."
        ),
    },
    "model": {
        "title": "Learned model evaluation",
        "intro": (
            "These plots evaluate the trained classifiers and regressors on validation "
            "and held-out test data: where they are right, where they fail, and which "
            "features they rely on."
        ),
    },
}

GRAPH_NOTES = {
    "raw": {
        "hist_raw_metrics.png": {
            "technical": (
                "2x3 grid of 50-bin histograms over all ingested frames: blur_var, "
                "brightness_L, glare_pct, glare_blob_pct and doc_short_side_px. blur_var, "
                "glare_pct and glare_blob_pct are shown as log1p because they are strongly "
                "right-skewed; brightness is Lab L-channel mean; short side is the minimum "
                "warped document dimension in pixels."
            ),
            "layman": (
                "A bird's-eye view of how sharp, bright, glary and large the documents are "
                "across the whole dataset, before any filtering."
            ),
            "eli5": (
                "Imagine measuring every photo's sharpness, brightness and size, then drawing "
                "bars showing how many photos fell into each range."
            ),
            "look": (
                "blur_var should span from very low (blurry) to high (sharp) with a long right "
                "tail; brightness of warped document crops should sit in a broad mid-to-high "
                "band (roughly 120-220 Lab L) and not pile up at 0 or 255; glare should be "
                "mostly near zero with a few high-glare outliers; short side should cluster in "
                "the 400-800 px range for warped documents."
            ),
            "gives": (
                "Baseline sanity: if a metric is constant or has no spread, no model can learn "
                "it. This is the first go/no-go check for each dimension."
            ),
        },
        "hist_by_doc_type.png": {
            "technical": (
                "Boxplots of blur_var (log scale), brightness_L and glare_pct split by "
                "document_type (id_card vs passport), one panel per metric."
            ),
            "layman": (
                "Compares the quality ranges of ID cards versus passports so you can see "
                "whether one document style is systematically different."
            ),
            "eli5": (
                "Like comparing test scores between two classrooms to see if one class always "
                "scores higher, regardless of the students."
            ),
            "look": (
                "Overlapping medians and ranges mean the two types behave similarly and one "
                "model can serve both. One type consistently darker, glarier or sharper means "
                "the feature reflects the design, not the capture quality."
            ),
            "gives": (
                "Evidence for the single-model vs per-type decision and for keeping "
                "document_type as an input feature."
            ),
        },
        "capture_condition_counts.png": {
            "technical": (
                "Bar chart of frame counts per capture_condition, which is the raw MIDV-500 "
                "clip prefix (CA/CS/HA/HS/KA/KS/PA/PS/TA/TS)."
            ),
            "layman": "How many photos came from each capture setup or device condition.",
            "eli5": "Counting how many kids arrived on each school bus.",
            "look": (
                "Roughly equal bars are healthy. A missing prefix means that capture setup was "
                "not ingested; a dominant prefix can bias splits and evaluation."
            ),
            "gives": "Sampling balance across capture setups, which feeds stratified split choices.",
        },
        "missingness.png": {
            "technical": (
                "Bar chart of NaN counts per column of raw_metrics.parquet. Empty chart with a "
                "'No missing values' label means every metric computed for every frame."
            ),
            "layman": "Which measurements failed to compute for some photos.",
            "eli5": "A checklist showing which fields of each photo's report card were left blank.",
            "look": (
                "Ideally no bars at all. Bars over quad or metric columns mean broken "
                "annotations or decode failures upstream."
            ),
            "gives": "Ingest integrity confirmation before cleaning even runs.",
        },
    },
    "cleaning": {
        "before_after_transforms.png": {
            "technical": (
                "2x2 histograms comparing raw blur_var vs log1p(blur_var) and raw glare_pct vs "
                "log1p(glare_pct), 60 bins each."
            ),
            "layman": "Before/after views showing how skewed measurements were straightened out.",
            "eli5": "Like squashing a stretched-out spring back into an even spring.",
            "look": (
                "The raw plots are long right tails; the log1p plots should look much closer "
                "to a bell shape, which is what linear models and distance metrics prefer."
            ),
            "gives": "Justification for the log columns recorded in feature_spec.json.",
        },
        "reconciliation.png": {
            "technical": (
                "Bar chart of cleaning counts from reconciliation_dataset.json: input_rows, "
                "dropped_missing_metrics, dropped_out_of_range, outlier_flagged, "
                "dropped_near_duplicates and kept_rows."
            ),
            "layman": "How many images went in, how many were kept, and why the rest were dropped.",
            "eli5": "How many crayons you kept, threw away because they were broken, or removed "
            "because they were duplicates of ones you already had.",
            "look": (
                "Drops should be small and explainable. Outliers are flagged and kept for "
                "analysis (not silently deleted); near-duplicate drops should be modest because "
                "frames were already subsampled every N-th frame."
            ),
            "gives": "The audit trail that makes the dataset trustworthy and reproducible.",
        },
    },
    "targets": {
        "hist_targets.png": {
            "technical": (
                "Histograms (50 bins) of the continuous regression targets on degraded rows: "
                "target_blur_sigma, target_brightness_gamma, target_brightness_ev, "
                "target_glare_radius_frac, target_downscale_factor and the composite "
                "severity_index (0-1). brightness_ev is the photographic exposure shift in "
                "stops (2^EV brightness multiplier)."
            ),
            "layman": "How strongly each simulated flaw was applied across the training images.",
            "eli5": "How much rain, sun, blur and dirt we added to each photo on purpose, and "
            "how many photos got each amount.",
            "look": (
                "Coverage across the whole grid: spikes only at the neutral values (sigma=0, "
                "gamma=1, factor=1) mean too few degraded examples; severity_index should be "
                "spread between 0 and 1 rather than piled at the extremes."
            ),
            "gives": "Regression target coverage; under-sampled severities will make the model "
            "weak exactly where it matters (moderate degradation).",
        },
        "hist_severities.png": {
            "technical": (
                "Histograms of the per-dimension severities (blur, brightness, glare, "
                "resolution) normalized to 0-1 plus severity_index. Severity is derived from "
                "degradation parameters, never from measured metrics."
            ),
            "layman": "Each flaw rescaled so 0 means perfect and 1 means worst, so they can be "
            "compared on one scale.",
            "eli5": "Giving each problem a grade from 0 (none) to 1 (very bad) so blur and glare "
            "can be compared fairly.",
            "look": (
                "Mass should exist near the label cutoffs, not only at 0 and 1. If all mass sits "
                "at 0, the classifier sees only easy negatives and will disappoint in the real "
                "moderate-degradation range."
            ),
            "gives": "Realism of the class boundary and difficulty of the learning problem.",
        },
        "feature_vs_target.png": {
            "technical": (
                "Hexbin density plots (40 bins, 5,000-row sample) of log_blur_var vs "
                "blur_severity, brightness_L vs brightness_severity, log_glare_pct vs "
                "glare_severity and doc_short_side_px vs resolution_severity."
            ),
            "layman": "Do the measured numbers actually move together with how degraded the "
            "image was made?",
            "eli5": "Does the thermometer reading go up when the soup gets hotter? If not, the "
            "thermometer is not useful.",
            "look": (
                "Monotone bands or clouds mean the feature is learnable. A diffuse blob (often "
                "resolution and brightness) means the degradation is hard to see from that "
                "feature, and the model will score lower on that dimension."
            ),
            "gives": "Which dimensions are learnable from image evidence and where to invest in "
            "better feature engineering.",
        },
    },
    "classification": {
        "class_balance.png": {
            "technical": (
                "Bar charts of 0/1 counts for readable_overall (4-dimension physical label), "
                "readable_overall_learned (blur AND brightness AND glare, the label the learned "
                "overall classifier is trained on) and the four per-dimension labels. "
                "resolution is deterministic via the pixel-size rule."
            ),
            "layman": "How many images pass and fail the readability check, overall and for each "
            "individual flaw.",
            "eli5": "A tally of green check marks and red crosses for the whole set and for each "
            "type of problem.",
            "look": (
                "Severe imbalance (e.g. 95/5) needs class weights or threshold tuning. Per-"
                "dimension rates differ a lot: glare failures are rare because the glare radius "
                "grid is mostly mild."
            ),
            "gives": "Class weighting strategy and the baseline positive rate each F1 score must "
            "beat.",
        },
        "class_balance_by_type.png": {
            "technical": "Grouped bar chart of readable_overall counts split by document_type.",
            "layman": "Pass/fail rates for ID cards versus passports.",
            "eli5": "Pass rate for each classroom, to check one classroom is not treated unfairly.",
            "look": (
                "Similar pass rates across types are healthy. A large gap means the model (or "
                "the label rule) behaves differently per document design, so always report "
                "per-type metrics too."
            ),
            "gives": "The single-model vs per-type decision and fairness checks for evaluation.",
        },
        "feature_hist_by_class.png": {
            "technical": (
                "Kernel density estimates of the top-k selected features, split by "
                "readable_overall_learned (0 red, 1 blue), on a 20,000-row sample. This is the "
                "label the learned overall classifier predicts."
            ),
            "layman": "Do the numbers look different for readable versus unreadable images?",
            "eli5": "Two bell curves: if they sit far apart, the clue is strong; if they overlap "
            "heavily, the clue is weak.",
            "look": (
                "Well-separated peaks (blur, glare features) versus heavy overlap (some "
                "brightness/resolution features). Overlap on a top feature predicts a weak "
                "classifier for that dimension."
            ),
            "gives": "A quick separability read before training and a sanity check on the top-k "
            "feature ranking.",
        },
    },
    "engineering": {
        "correlation_heatmap.png": {
            "technical": (
                "Spearman correlation heatmap (coolwarm, centered at 0) among the top 15 "
                "MI-ranked features."
            ),
            "layman": "Which measurements move together, meaning they carry the same information.",
            "eli5": "Finding out which crayons always break at the same time, so you don't need "
            "to pack both.",
            "look": (
                "Off-diagonal blocks above |0.95| indicate redundant features that reduction "
                "removed. A mostly pale matrix means the kept features add distinct information."
            ),
            "gives": "Evidence that the correlation-pruned feature set is not wasteful.",
        },
        "mi_ranking.png": {
            "technical": (
                "Horizontal bars of mutual information between each candidate feature and "
                "severity_index (top 20), computed on the training split."
            ),
            "layman": "Which measurements tell you the most about overall image badness.",
            "eli5": "Ranking detective clues by how much each one helps predict the answer.",
            "look": (
                "Blur and edge-energy features usually rank first because severity_index is "
                "blur-dominated. Dimension-specific features (glare, brightness) appear lower "
                "but must stay non-zero, otherwise that dimension needs new features."
            ),
            "gives": "The ranking used to build feature_spec.json and to choose the compact "
            "top-k list for the TypeScript port.",
        },
        "pca_scatter.png": {
            "technical": (
                "2-D PCA projection of standardized selected features (8,000-row sample), "
                "points colored by readable_overall_learned."
            ),
            "layman": "A flat map of all images, colored by whether they are readable.",
            "eli5": "Squashing a 3-D toy box onto paper to see whether red and yellow balls "
            "separate into different areas.",
            "look": (
                "Partial color separation is good; total overlap means the features cannot "
                "easily separate classes. Isolated clusters often point to a specific document "
                "type or capture condition."
            ),
            "gives": "Holistic view of separability, cluster structure and potential outliers.",
        },
        "quality_prior_by_class.png": {
            "technical": (
                "KDE of the hand-weighted quality_prior score (mean of fixed-range normalized "
                "blur, brightness, glare, resolution terms) split by readable_overall_learned."
            ),
            "layman": "Does a simple homemade quality score already separate good from bad "
            "images?",
            "eli5": "Comparing a quick-and-dirty grade against the official pass/fail stamp.",
            "look": (
                "Shifted densities with limited overlap mean the simple prior works; heavy "
                "overlap means the learned model must (and does) add value."
            ),
            "gives": "A transparent baseline the learned classifier should beat.",
        },
    },
    "splits": {
        "split_counts.png": {
            "technical": (
                "Three bars: rows per split, unique groups (doc_folder/clip) per split, and "
                "readable_overall rate per split. Test is a leave-doc-types-out set."
            ),
            "layman": "Are the train, validation and test buckets fair and are test documents "
            "completely unseen?",
            "eli5": "Three buckets of candy: do they have similar amounts and kinds, and is the "
            "test bucket made of candies the model never tasted?",
            "look": (
                "Class rates should be roughly similar across splits. The test bucket contains "
                "whole document designs never seen in training, so some metric drop is expected "
                "and is the honest measure of generalization."
            ),
            "gives": "Trust in the evaluation: no frame, clip or document design leaks across "
            "splits.",
        },
        "target_dist_by_split.png": {
            "technical": (
                "KDE of severity_index, blur_severity and resolution_severity per split."
            ),
            "layman": "Do the splits contain a similar mix of easy and hard images?",
            "eli5": "Checking each bucket got the same recipe portions.",
            "look": (
                "Overlapping curves mean consistent difficulty. A shifted test curve warns that "
                "test scores are not directly comparable to validation scores."
            ),
            "gives": "Fairness check for comparing validation and held-out test metrics.",
        },
    },
    "model": {
        "confusion_matrices.png": {
            "technical": (
                "One confusion matrix per learned classifier on the validation split (rows "
                "actual, columns predicted), computed at the auto-selected operating threshold "
                "from val precision-recall targeting the per-class recall in config "
                "(training.target_recall); thresholds are saved to models/thresholds.json. "
                "Resolution has no matrix because it uses the deterministic pixel-size rule."
            ),
            "layman": "A right/wrong scorecard: how many pass/fail calls were correct and what "
            "kind of mistakes were made.",
            "eli5": "Marking a quiz: how many answers were right, and did the mistakes mark good "
            "answers wrong or bad answers right?",
            "look": (
                "False negatives (bad image called readable) are the costly mistakes for OCR "
                "gating. resolution and brightness show the most errors; if false negatives "
                "concentrate there, raise that threshold or add features."
            ),
            "gives": "Per-dimension error profile and threshold-tuning direction.",
        },
        "roc_curves.png": {
            "technical": (
                "ROC curves with AUC for each learned classifier on validation, one curve per "
                "label (overall is trained on readable_overall_learned). Resolution is excluded: "
                "it is deterministic via the document pixel-size rule and is reported separately "
                "in model_report.json as resolution_rule and combined_system."
            ),
            "layman": "How well each model ranks bad images above good ones across all possible "
            "thresholds.",
            "eli5": "A graph of how many bad ones you catch versus how many false alarms you "
            "raise as you tighten the rule.",
            "look": (
                "Curves hugging the top-left corner are strong (glare, blur). A curve close to "
                "the diagonal (resolution) means the model barely separates classes."
            ),
            "gives": "Model comparison independent of threshold and the operating-point choice "
            "for production.",
        },
        "regression_pred_vs_actual.png": {
            "technical": (
                "Scatter of predicted versus actual severity for blur, brightness, glare and "
                "severity_index_learned (mean of the three learned dimensions) on validation, "
                "with a dashed identity line. Resolution severity is not regressed because "
                "resolution is deterministic in production."
            ),
            "layman": "How close the predicted severity is to the true severity.",
            "eli5": "Guessing someone's age and plotting the guess against the real age: perfect "
            "guesses sit on the diagonal.",
            "look": (
                "Points should cluster on the diagonal. Systematic flattening at the extremes is "
                "regression to the mean; a wide vertical spread at a given actual value means "
                "that dimension is under-determined by the features."
            ),
            "gives": "Reliability of each severity score the Angular app would display.",
        },
        "permutation_importance.png": {
            "technical": (
                "Permutation importance on validation: ROC-AUC drop for readable_overall and "
                "MAE increase for severity_index when each feature is shuffled (5 repeats)."
            ),
            "layman": "Which measurements the model actually relies on to make decisions.",
            "eli5": "Figuring out which clues the detective really used by hiding each clue and "
            "seeing if the case falls apart.",
            "look": (
                "Blur and edge-energy features (blur_var, tenengrad, fft_hf_ratio, "
                "quality_prior) should dominate. Near-zero bars are candidates for removal and "
                "shrink the TypeScript port."
            ),
            "gives": "Final feature pruning evidence and the shortlist of features to implement "
            "in pure TypeScript.",
        },
    },
}


def _safe_load_json(path):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _fmt(value, digits=3):
    if value is None:
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return f"{int(value)}"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _load_context(config: dict) -> dict:
    processed = config["paths"]["processed_dir"]
    models = config["paths"].get("models_dir", os.path.join(os.path.dirname(processed), "models"))
    context = {
        "config": config,
        "raw": None,
        "features": None,
        "ingest": _safe_load_json(os.path.join(processed, "reconciliation_ingest.json")),
        "dataset": _safe_load_json(os.path.join(processed, "reconciliation_dataset.json")),
        "validation": _safe_load_json(os.path.join(processed, "validation.json")),
        "ranking": _safe_load_json(os.path.join(processed, "feature_ranking.json")),
        "spec": _safe_load_json(os.path.join(processed, "feature_spec.json")),
        "splits": _safe_load_json(os.path.join(processed, "splits.json")),
        "model_report": _safe_load_json(os.path.join(models, "model_report.json")),
        "plot_manifest": _safe_load_json(os.path.join(config["paths"]["plots_dir"], "manifest.json")),
    }
    raw_path = os.path.join(processed, "raw_metrics.parquet")
    features_path = os.path.join(processed, "features.parquet")
    if os.path.exists(raw_path):
        context["raw"] = pd.read_parquet(raw_path)
    if os.path.exists(features_path):
        context["features"] = pd.read_parquet(features_path)
    return context


def _observed(group: str, name: str, ctx: dict) -> str:
    raw = ctx.get("raw")
    features = ctx.get("features")
    ingest = ctx.get("ingest") or {}
    dataset = ctx.get("dataset") or {}
    ranking = ctx.get("ranking") or []
    spec = ctx.get("spec") or {}
    model_report = ctx.get("model_report") or {}

    try:
        if name == "hist_raw_metrics.png" and raw is not None:
            return (
                f"{len(raw)} frames; median blur_var={_fmt(raw['blur_var'].median())}, "
                f"brightness_L={_fmt(raw['brightness_L'].median(), 1)}, "
                f"glare_pct={_fmt(raw['glare_pct'].median())}; "
                f"doc short side p10/p50/p90 = "
                f"{_fmt(raw['doc_short_side_px'].quantile(0.1), 0)}/"
                f"{_fmt(raw['doc_short_side_px'].median(), 0)}/"
                f"{_fmt(raw['doc_short_side_px'].quantile(0.9), 0)} px."
            )
        if name == "hist_by_doc_type.png" and raw is not None:
            counts = raw["document_type"].value_counts().to_dict()
            return f"Frames per type: {counts}."
        if name == "capture_condition_counts.png" and raw is not None:
            counts = raw["capture_condition"].value_counts()
            top = ", ".join(f"{k}={v}" for k, v in counts.head(4).items())
            return f"{counts.shape[0]} capture conditions; largest: {top}."
        if name == "missingness.png" and raw is not None:
            missing_columns = int((raw.isna().sum() > 0).sum())
            return f"Columns with any missing value: {missing_columns} of {raw.shape[1]}."
        if name == "before_after_transforms.png" and raw is not None:
            from scipy.stats import skew

            raw_skew = skew(raw["blur_var"].dropna())
            log_skew = skew(np.log1p(raw["blur_var"].dropna()))
            return f"blur_var skew raw={_fmt(raw_skew, 2)} vs log1p={_fmt(log_skew, 2)}."
        if name == "reconciliation.png":
            clean = dataset.get("clean") or {}
            return (
                f"input={clean.get('input_rows', 'n/a')}, kept={clean.get('kept_rows', 'n/a')}, "
                f"missing={clean.get('dropped_missing_metrics', 'n/a')}, "
                f"out_of_range={clean.get('dropped_out_of_range', 'n/a')}, "
                f"outliers_flagged={clean.get('outlier_flagged', 'n/a')}, "
                f"near_duplicates={clean.get('dropped_near_duplicates', 'n/a')}."
            )
        if name == "hist_targets.png" and features is not None:
            degraded = features[features["source"] == "degraded"]
            return (
                f"{len(degraded)} degraded rows; severity_index min/median/max = "
                f"{_fmt(degraded['severity_index'].min(), 2)}/"
                f"{_fmt(degraded['severity_index'].median(), 2)}/"
                f"{_fmt(degraded['severity_index'].max(), 2)}."
            )
        if name == "hist_severities.png" and features is not None:
            degraded = features[features["source"] == "degraded"]
            means = {
                column: _fmt(degraded[column].mean(), 2)
                for column in ["blur_severity", "brightness_severity", "glare_severity", "resolution_severity"]
            }
            return f"Mean severity per dimension: {means}."
        if name == "feature_vs_target.png" and features is not None:
            pairs = [
                ("log_blur_var", "blur_severity"),
                ("brightness_L", "brightness_severity"),
                ("log_glare_pct", "glare_severity"),
                ("doc_short_side_px", "resolution_severity"),
            ]
            correlations = {
                f"{x}~{y}": _fmt(features[x].corr(features[y], method="spearman"), 2) for x, y in pairs
            }
            return f"Spearman correlations: {correlations}."
        if name == "class_balance.png" and features is not None:
            rates = {
                column: _fmt(float(features[column].mean()), 3)
                for column in [
                    "readable_overall",
                    "readable_overall_learned",
                    "readable_blur",
                    "readable_brightness",
                    "readable_glare",
                    "readable_resolution",
                ]
                if column in features.columns
            }
            return f"Positive rates: {rates}."
        if name == "class_balance_by_type.png" and features is not None:
            rates = (
                features.groupby("document_type")["readable_overall"].mean().round(3).to_dict()
            )
            return f"readable_overall rate by type: {rates}."
        if name == "feature_hist_by_class.png":
            top = spec.get("top_k_features", [])
            return f"Top-k features shown: {top[:6]}."
        if name == "correlation_heatmap.png":
            return (
                f"{len(spec.get('selected_features', []))} features survived pruning; "
                f"correlation threshold={ctx['config']['reduction']['corr_threshold']}; "
                f"low-variance dropped={len(spec.get('dropped_low_variance', []))}."
            )
        if name == "mi_ranking.png" and ranking:
            top = ", ".join(f"{item[0]}={_fmt(item[1], 3)}" for item in ranking[:5])
            return f"Top MI features: {top}."
        if name == "pca_scatter.png" and features is not None:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler

            columns = [c for c in spec.get("selected_features", []) if c in features.columns]
            if len(columns) >= 2:
                sample = features.sample(min(4000, len(features)), random_state=ctx["config"]["seed"])
                matrix = StandardScaler().fit_transform(sample[columns].fillna(0.0).to_numpy(dtype=float))
                pca = PCA(n_components=2, random_state=ctx["config"]["seed"]).fit(matrix)
                return (
                    f"Explained variance PC1+PC2 = "
                    f"{_fmt(pca.explained_variance_ratio_.sum() * 100, 1)}% over {len(columns)} features."
                )
        if name == "quality_prior_by_class.png" and features is not None:
            means = features.groupby("readable_overall")["quality_prior"].mean().round(3).to_dict()
            return f"Mean quality_prior by class: {means}."
        if name == "split_counts.png" and ctx.get("splits"):
            summary = ctx["splits"].get("summary", {})
            return f"Rows: {summary.get('rows', {})}; groups: {summary.get('groups', {})}."
        if name == "target_dist_by_split.png" and features is not None:
            medians = features.groupby("split")["severity_index"].median().round(3).to_dict()
            return f"Median severity_index per split: {medians}."
        if name == "confusion_matrices.png" and model_report:
            items = {
                target: _fmt(entry.get("val", {}).get("f1"), 3)
                for target, entry in model_report.get("classification", {}).items()
            }
            thresholds = model_report.get("thresholds", {})
            return f"Thresholds: {thresholds}; validation F1: {items}."
        if name == "roc_curves.png" and model_report:
            items = {
                target: _fmt(entry.get("val", {}).get("roc_auc"), 3)
                for target, entry in model_report.get("classification", {}).items()
            }
            rule = (model_report.get("resolution_rule") or {}).get("val")
            suffix = f" Resolution rule val F1 = {_fmt(rule.get('f1'))}." if rule else ""
            combined = (model_report.get("combined_system") or {}).get("val")
            suffix += (
                f" Combined system val F1 = {_fmt(combined.get('f1'))}." if combined else ""
            )
            return f"Validation AUC per classifier: {items}.{suffix}"
        if name == "regression_pred_vs_actual.png" and model_report:
            items = {
                target: _fmt(entry.get("val", {}).get("r2"), 3)
                for target, entry in model_report.get("regression", {}).items()
            }
            return f"Validation R2 per regressor: {items}."
        if name == "permutation_importance.png" and model_report:
            importances = model_report.get("permutation_importance", {})
            parts = []
            for target, values in importances.items():
                top = sorted(values.items(), key=lambda item: item[1], reverse=True)[:3]
                parts.append(f"{target}: " + ", ".join(f"{k}={_fmt(v, 3)}" for k, v in top))
            return "; ".join(parts) if parts else "n/a"
    except Exception as exc:
        return f"(could not compute: {exc})"
    return "n/a"


def _write_group_readmes(plots_dir: str, ctx: dict) -> list:
    manifest = ctx.get("plot_manifest") or []
    generated = {}
    for item in manifest:
        if item.get("status") == "ok":
            generated.setdefault(item["group"], set()).add(item["name"])

    written = []
    for group in GROUPS:
        info = GROUP_INFO[group]
        notes = GRAPH_NOTES[group]
        group_dir = os.path.join(plots_dir, group)
        disk_files = (
            {f for f in os.listdir(group_dir) if f.endswith(".png")}
            if os.path.isdir(group_dir)
            else set()
        )
        generated_files = generated.get(group, set()) | disk_files
        lines = [f"# {info['title']}", "", info["intro"], ""]
        if not generated_files:
            lines.append("_No plots were generated for this group in the current run._")
        for filename, note in notes.items():
            present = filename in generated_files
            status = "" if present else " _(not generated in the current run)_"
            lines.extend(
                [
                    f"## {filename}{status}",
                    "",
                    f"**Technical** — {note['technical']}",
                    "",
                    f"**Layman** — {note['layman']}",
                    "",
                    f"**ELI5** — {note['eli5']}",
                    "",
                    f"**What to look for** — {note['look']}",
                    "",
                    f"**What it gives you** — {note['gives']}",
                    "",
                    f"**Observed in the current run** — {_observed(group, filename, ctx)}",
                    "",
                ]
            )
        if present_extra := generated_files - set(notes.keys()):
            lines.append(f"Undocumented plots generated: {sorted(present_extra)}")
            lines.append("")
        os.makedirs(group_dir, exist_ok=True)
        path = os.path.join(group_dir, "README.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        written.append(path)
    return written


def _write_index(plots_dir: str, ctx: dict) -> str:
    features = ctx.get("features")
    model_report = ctx.get("model_report") or {}
    validation = ctx.get("validation") or {}
    spec = ctx.get("spec") or {}
    dataset = ctx.get("dataset") or {}

    lines = [
        "# Plots guide",
        "",
        "Every folder under `training/plots/` has a README explaining each graph in technical,",
        "layman and ELI5 terms, with what to look for, what it gives you, and live numbers from",
        "the latest run.",
        "",
        "## How to read each entry",
        "",
        "- **Technical** — what is plotted, axes, formulas, thresholds, source columns.",
        "- **Layman** — plain-English meaning with no jargon.",
        "- **ELI5** — one concrete analogy.",
        "- **What to look for** — expected shapes and red flags.",
        "- **What it gives you** — the decision or action this plot informs.",
        "- **Observed in the current run** — actual numbers from the latest artifacts.",
        "",
        "## Current run at a glance",
        "",
    ]
    if features is not None:
        splits = features["split"].value_counts().to_dict()
        lines.extend(
            [
                f"- Model rows: **{len(features)}** (splits: {splits})",
                f"- Features used by the model: **{len(spec.get('selected_features', []))}**",
                f"- Validation checks passed: **{validation.get('validated', 'n/a')}**",
            ]
        )
    if dataset:
        rows = dataset.get("rows", {})
        lines.append(f"- Dataset composition: {rows}")
    overall = (model_report.get("classification", {}) or {}).get("readable_overall", {})
    severity = (model_report.get("regression", {}) or {}).get("severity_index", {})
    if overall:
        val = overall.get("val", {})
        test = overall.get("test", {})
        lines.append(
            f"- readable_overall classifier: val F1 = **{_fmt(val.get('f1'))}**, "
            f"AUC = **{_fmt(val.get('roc_auc'))}**; test F1 = **{_fmt(test.get('f1'))}**, "
            f"AUC = **{_fmt(test.get('roc_auc'))}**"
        )
    if severity:
        val = severity.get("val", {})
        test = severity.get("test", {})
        lines.append(
            f"- severity_index regressor: val MAE = **{_fmt(val.get('mae'))}**, "
            f"R2 = **{_fmt(val.get('r2'))}**; test MAE = **{_fmt(test.get('mae'))}**, "
            f"R2 = **{_fmt(test.get('r2'))}**"
        )
    thresholds = model_report.get("thresholds", {})
    if thresholds:
        lines.append(f"- Auto-selected operating thresholds: {thresholds}")
    resolution_rule = (model_report.get("resolution_rule") or {}).get("val")
    if resolution_rule:
        lines.append(
            f"- Deterministic resolution rule: val F1 = **{_fmt(resolution_rule.get('f1'))}**, "
            f"accuracy = **{_fmt(resolution_rule.get('accuracy'))}**"
        )
    combined = (model_report.get("combined_system") or {}).get("val")
    if combined:
        lines.append(
            f"- Combined system (learned overall AND resolution rule): "
            f"val F1 = **{_fmt(combined.get('f1'))}**, recall = **{_fmt(combined.get('recall'))}**, "
            f"precision = **{_fmt(combined.get('precision'))}**"
        )
    lines.extend(["", "## Folders", ""])
    for group in GROUPS:
        info = GROUP_INFO[group]
        lines.append(f"- [`{group}/`]({group}/README.md) — {info['title']}: {info['intro']}")
    lines.extend(
        [
            "",
            "## Notes and caveats",
            "",
            "- Labels are provisional synthetic-degradation proxies (see `labels` in `config.yaml`).",
            "  Human review is required before treating model metrics as production-ready.",
            "- Brightness uses the exposure-adequacy label (gamma range AND |EV| <= "
            "`brightness_ev_max`); this change is flagged for human review.",
            "- Resolution is deterministic: apply the pixel-size rule from `feature_spec.json`",
            "  (`resolution_rule`) to the measured warped document short side; no learned model.",
            "- Operating thresholds are auto-selected per classifier from the validation",
            "  precision-recall curve for the recall targets in `training.target_recall`,",
            "  with a max-F1 fallback; see `models/thresholds.json`.",
            "- Test metrics come from whole document designs held out of training",
            "  (`splits.test_doc_folders` in `config.yaml`), so they measure generalization.",
            "- Docs are generated by `training/make_plot_docs.py`; regenerate after any pipeline run:",
            "  `uv run python training/make_plot_docs.py`.",
            "",
            "`manifest.json` lists every generated plot with its status.",
        ]
    )
    path = os.path.join(plots_dir, "README.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def run(config: dict):
    plots_dir = config["paths"]["plots_dir"]
    os.makedirs(plots_dir, exist_ok=True)
    ctx = _load_context(config)
    written = _write_group_readmes(plots_dir, ctx)
    index = _write_index(plots_dir, ctx)
    written.append(index)
    for path in written:
        print("  docs:", path)
    return written


def main():
    parser = argparse.ArgumentParser(description="Generate README docs for the plot folders.")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()