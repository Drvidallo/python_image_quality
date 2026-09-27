"""Build the model-ready dataset from raw metrics plus synthetic degradations.

Stages: clean -> degrade/label -> transform -> engineer -> reduce -> split -> export.

Usage:
    python build_dataset.py --config config.yaml
    python build_dataset.py --config config.yaml --max-variants 4 --workers 4
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

from data_common import (
    engineer_features,
    hamming_distance,
    load_config,
    metrics_on,
    order_corners,
    warp_native,
)
from degradations import apply_variant, sample_variants, targets_and_labels

CORE_METRICS = [
    "blur_var",
    "brightness_L",
    "glare_pct",
    "glare_blob_pct",
    "doc_short_side_px",
    "doc_area_ratio",
]

TARGET_COLUMNS = [
    "target_blur_sigma",
    "target_motion_length",
    "target_motion_angle",
    "target_brightness_gamma",
    "target_brightness_ev",
    "target_glare_radius_frac",
    "target_downscale_factor",
    "effective_short_side_px",
    "blur_severity",
    "brightness_severity",
    "glare_severity",
    "resolution_severity",
    "severity_index",
    "severity_index_learned",
]

LABEL_COLUMNS = [
    "readable_blur",
    "readable_brightness",
    "readable_glare",
    "readable_resolution",
    "readable_overall",
    "readable_overall_learned",
]

IDENTITY_COLUMNS = [
    "sample_id",
    "filepath",
    "rel_path",
    "doc_folder",
    "document_type",
    "capture_condition",
    "clip",
    "frame_index",
    "source",
    "variant_index",
    "dhash",
]


PARAM_COLUMNS = [
    "blur_sigma",
    "motion_length",
    "motion_angle",
    "brightness_gamma",
    "brightness_ev",
    "glare_radius_frac",
    "downscale_factor",
]


def _degrade_worker(job: dict) -> list:
    config = job["config"]
    row = job["row"]
    seed = int(row["sample_id"], 16) % (2**32)
    rng = np.random.default_rng(seed)

    image = cv2.imread(row["filepath"], cv2.IMREAD_COLOR)
    if image is None:
        return []

    quad = np.array(
        [
            [row["quad_tl_x"], row["quad_tl_y"]],
            [row["quad_tr_x"], row["quad_tr_y"]],
            [row["quad_br_x"], row["quad_br_y"]],
            [row["quad_bl_x"], row["quad_bl_y"]],
        ],
        dtype=np.float64,
    )
    crop = warp_native(image, order_corners(quad))
    if crop is None:
        return []

    native_short_side = float(min(crop.shape[1], crop.shape[0]))
    max_variants = job["max_variants"] or config["degradations"]["max_variants_per_crop"]
    variants = sample_variants(config, rng, max_variants)
    glare_threshold = config["ingest"]["glare_v_threshold"]

    rows = []
    for index, params in enumerate(variants[1:], start=1):
        degraded = apply_variant(crop, params, rng)
        metrics = metrics_on(degraded, glare_threshold)
        targets = targets_and_labels(params, native_short_side, config)
        row_out = {
            "sample_id": row["sample_id"],
            "filepath": row["filepath"],
            "rel_path": row["rel_path"],
            "doc_folder": row["doc_folder"],
            "document_type": row["document_type"],
            "capture_condition": row["capture_condition"],
            "clip": row["clip"],
            "frame_index": row["frame_index"],
            "dhash": row["dhash"],
            "source": "degraded",
            "variant_index": index,
            "doc_width_px": row["doc_width_px"],
            "doc_height_px": row["doc_height_px"],
            "doc_short_side_px": row["doc_short_side_px"],
            "doc_area_ratio": row["doc_area_ratio"],
            "aspect_ratio": row["aspect_ratio"],
            "aspect_deviation": row["aspect_deviation"],
            "edge_cv": row["edge_cv"],
            "est_dpi": row["est_dpi"],
            "image_height": row["image_height"],
            "image_width": row["image_width"],
            "blur_sigma": params["blur_sigma"],
            "motion_length": params["motion_length"],
            "motion_angle": params["motion_angle"],
            "brightness_gamma": params["brightness_gamma"],
            "brightness_ev": params.get("brightness_ev", 0.0),
            "glare_radius_frac": params["glare_radius_frac"],
            "downscale_factor": params["downscale_factor"],
        }
        row_out.update(metrics)
        row_out.update(targets)
        rows.append(row_out)
    return rows


def _iqr_bounds(series: pd.Series, k: float):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def _clean(raw: pd.DataFrame, config: dict):
    cleaning = config["cleaning"]
    reconciliation = {}
    frame = raw.copy()
    reconciliation["input_rows"] = len(frame)

    for column in CORE_METRICS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan)

    before = len(frame)
    frame = frame.dropna(subset=CORE_METRICS + ["dhash"])
    reconciliation["dropped_missing_metrics"] = before - len(frame)

    before = len(frame)
    frame = frame[
        (frame["blur_var"] > 0)
        & (frame["brightness_L"].between(0, 255))
        & (frame["glare_pct"] >= 0)
        & (frame["glare_blob_pct"] >= 0)
    ]
    reconciliation["dropped_out_of_range"] = before - len(frame)

    frame["log_blur_var"] = np.log1p(frame["blur_var"])
    frame["log_glare_pct"] = np.log1p(frame["glare_pct"])
    k = cleaning["outlier_iqr_k"]
    blur_lo, blur_hi = _iqr_bounds(frame["log_blur_var"], k)
    glare_lo, glare_hi = _iqr_bounds(frame["log_glare_pct"], k)
    frame["is_outlier"] = (
        (frame["log_blur_var"] < blur_lo)
        | (frame["log_blur_var"] > blur_hi)
        | (frame["log_glare_pct"] < glare_lo)
        | (frame["log_glare_pct"] > glare_hi)
    )
    reconciliation["outlier_flagged"] = int(frame["is_outlier"].sum())

    frame = frame.sort_values(["doc_folder", "clip", "frame_index"]).reset_index(drop=True)
    threshold = cleaning["near_duplicate_hamming"]
    kept_hashes = {}
    drop_indices = []
    for idx, row in frame.iterrows():
        key = (row["doc_folder"], row["clip"])
        hashes = kept_hashes.setdefault(key, [])
        if any(hamming_distance(int(row["dhash"]), int(h)) <= threshold for h in hashes):
            drop_indices.append(idx)
        else:
            hashes.append(int(row["dhash"]))
    frame = frame.drop(index=drop_indices).reset_index(drop=True)
    reconciliation["dropped_near_duplicates"] = len(drop_indices)
    reconciliation["kept_rows"] = len(frame)
    return frame, reconciliation


def _source_gate(frame: pd.DataFrame, config: dict):
    gate = config["source_gate"]
    labels_config = config["labels"]
    min_side = max(gate["min_short_side_px"], labels_config["min_effective_short_side_px"])
    brightness_lo, brightness_hi = gate["brightness_range"]
    mask = (
        (~frame["is_outlier"])
        & (frame["blur_var"] >= gate["min_blur_var"])
        & (frame["brightness_L"].between(brightness_lo, brightness_hi))
        & (frame["glare_pct"] <= gate["max_glare_pct"])
        & (frame["doc_short_side_px"] >= min_side)
    )
    return frame[mask].reset_index(drop=True)


def _clean_training_rows(base: pd.DataFrame, config: dict) -> pd.DataFrame:
    neutral = {
        "blur_sigma": 0.0,
        "motion_length": 0.0,
        "motion_angle": 0.0,
        "brightness_gamma": 1.0,
        "brightness_ev": 0.0,
        "glare_radius_frac": 0.0,
        "downscale_factor": 1.0,
    }
    rows = []
    for _, row in base.iterrows():
        targets = targets_and_labels(neutral, float(row["doc_short_side_px"]), config)
        record = row.to_dict()
        record.update(
            {
                "source": "clean",
                "variant_index": 0,
                "blur_sigma": neutral["blur_sigma"],
                "motion_length": neutral["motion_length"],
                "motion_angle": neutral["motion_angle"],
                "brightness_gamma": neutral["brightness_gamma"],
                "brightness_ev": neutral["brightness_ev"],
                "glare_radius_frac": neutral["glare_radius_frac"],
                "downscale_factor": neutral["downscale_factor"],
                "image_height": row["image_height"],
                "image_width": row["image_width"],
            }
        )
        record.update(targets)
        rows.append(record)
    return pd.DataFrame(rows)


def _split(frame: pd.DataFrame, config: dict):
    splits_config = config["splits"]
    seed = config["seed"]
    present_test_folders = [f for f in splits_config["test_doc_folders"] if f in set(frame["doc_folder"])]
    frame = frame.copy()
    frame["split"] = "train"

    test_mask = frame["doc_folder"].isin(present_test_folders)
    frame.loc[test_mask, "split"] = "test"

    remaining = frame.loc[~test_mask]
    if remaining["group"].nunique() > 2:
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=splits_config["val_fraction"], random_state=seed
        )
        train_idx, val_idx = next(splitter.split(remaining, groups=remaining["group"]))
        frame.loc[remaining.index[train_idx], "split"] = "train"
        frame.loc[remaining.index[val_idx], "split"] = "val"

    train_groups = set(frame.loc[frame["split"] == "train", "group"])
    val_groups = set(frame.loc[frame["split"] == "val", "group"])
    test_groups = set(frame.loc[frame["split"] == "test", "group"])
    assert not (train_groups & val_groups)
    assert not (train_groups & test_groups)
    assert not (val_groups & test_groups)

    summary = {
        "test_doc_folders": present_test_folders,
        "rows": frame["split"].value_counts().to_dict(),
        "groups": {
            "train": len(train_groups),
            "val": len(val_groups),
            "test": len(test_groups),
        },
    }
    return frame, summary


def _reduce(frame: pd.DataFrame, feature_columns: list, config: dict):
    reduction = config["reduction"]
    train = frame[frame["split"] == "train"]
    numeric = train[feature_columns].select_dtypes(include=[np.number])

    variances = numeric.var()
    low_variance = variances[variances <= reduction["variance_threshold"]].index.tolist()
    candidates = [c for c in numeric.columns if c not in low_variance]

    target_column = (
        "severity_index_learned" if "severity_index_learned" in train.columns else "severity_index"
    )
    target = train[target_column].to_numpy()
    mi_scores = mutual_info_regression(
        numeric[candidates].fillna(0.0), target, random_state=config["seed"]
    )
    ranking = sorted(
        zip(candidates, [float(v) for v in mi_scores]), key=lambda item: item[1], reverse=True
    )

    corr = numeric[candidates].corr(method="spearman").abs()
    threshold = reduction["corr_threshold"]
    selected = []
    for name, _ in ranking:
        if all(corr.loc[name, kept] <= threshold for kept in selected):
            selected.append(name)
    top_k = selected[: reduction["top_k"]]

    return {
        "dropped_low_variance": low_variance,
        "ranking": ranking,
        "selected_features": selected,
        "top_k_features": top_k,
    }


def _winsorize(frame: pd.DataFrame, config: dict):
    train = frame[frame["split"] == "train"]
    skip = {
        "target_motion_angle",
        "blur_severity",
        "brightness_severity",
        "glare_severity",
        "resolution_severity",
        "severity_index",
        "severity_index_learned",
    }
    bounds = {}
    result = frame.copy()
    for column in TARGET_COLUMNS:
        if column in skip:
            continue
        lo, hi = train[column].quantile([0.01, 0.99])
        bounds[column] = [float(lo), float(hi)]
        result[column] = result[column].clip(lo, hi)
    return result, bounds


def _validate(frame: pd.DataFrame, feature_columns: list, config: dict) -> dict:
    checks = {}
    checks["no_param_columns_in_features"] = not (set(PARAM_COLUMNS) & set(feature_columns))
    checks["no_quad_columns_in_features"] = not any(c.startswith("quad_") for c in feature_columns)

    clean = frame[frame["source"] == "clean"]
    checks["clean_targets_neutral"] = bool((clean["severity_index"] == 0).all()) if len(clean) else True
    checks["clean_is_readable"] = bool((clean["readable_overall"] == 1).all()) if len(clean) else True

    degraded = frame[frame["source"] == "degraded"]
    per_dimension = ["readable_blur", "readable_brightness", "readable_glare", "readable_resolution"]
    if len(degraded):
        all_dims = degraded[per_dimension].all(axis=1).astype(int)
        checks["overall_equals_all_dimensions"] = bool((degraded["readable_overall"] == all_dims).all())
    else:
        checks["overall_equals_all_dimensions"] = True

    severity_columns = ["blur_severity", "brightness_severity", "glare_severity", "resolution_severity"]
    checks["severity_index_consistent"] = bool(
        np.allclose(frame["severity_index"], frame[severity_columns].mean(axis=1))
    )

    learned_severity_columns = ["blur_severity", "brightness_severity", "glare_severity"]
    checks["severity_index_learned_consistent"] = bool(
        np.allclose(frame["severity_index_learned"], frame[learned_severity_columns].mean(axis=1))
    )
    checks["overall_learned_equals_dimensions"] = bool(
        (
            frame["readable_overall_learned"]
            == frame[["readable_blur", "readable_brightness", "readable_glare"]]
            .all(axis=1)
            .astype(int)
        ).all()
    )

    group_sets = frame.groupby("split")["group"].apply(set)
    keys = list(group_sets.index)
    disjoint = True
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if group_sets[keys[i]] & group_sets[keys[j]]:
                disjoint = False
    checks["group_disjoint"] = disjoint
    checks["all_splits_non_empty"] = set(frame["split"].unique()) >= {"train", "val"}

    checks["validated"] = all(checks.values())
    return checks


def run(config: dict, max_variants=None, workers=None, force=False):
    processed_dir = config["paths"]["processed_dir"]
    raw_path = os.path.join(processed_dir, "raw_metrics.parquet")
    out_path = os.path.join(processed_dir, "features.parquet")
    if os.path.exists(out_path) and not force:
        print(f"features already exists: {out_path} (use --force to overwrite)")
        return out_path
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"raw metrics not found: {raw_path}. Run preprocess_midv500.py first.")

    workers = workers or config["ingest"]["workers"]
    raw = pd.read_parquet(raw_path)
    cleaned, clean_reconciliation = _clean(raw, config)
    base = _source_gate(cleaned, config)
    print(f"Clean rows: {len(cleaned)} | source-gated base crops: {len(base)}")

    clean_rows = _clean_training_rows(base, config)

    jobs = [{"row": record, "config": config, "max_variants": max_variants} for record in base.to_dict("records")]
    degraded_rows = []
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for rows in tqdm(
                executor.map(_degrade_worker, jobs, chunksize=4), total=len(jobs), desc="Degrading"
            ):
                degraded_rows.extend(rows)
    else:
        for job in tqdm(jobs, desc="Degrading"):
            degraded_rows.extend(_degrade_worker(job))

    degraded = pd.DataFrame(degraded_rows)
    combined = pd.concat([clean_rows, degraded], ignore_index=True)
    combined = engineer_features(combined)
    combined["severity_index_learned"] = combined[
        ["blur_severity", "brightness_severity", "glare_severity"]
    ].mean(axis=1)
    combined["readable_overall_learned"] = (
        combined[["readable_blur", "readable_brightness", "readable_glare"]]
        .all(axis=1)
        .astype(int)
    )
    combined["group"] = combined["doc_folder"] + "/" + combined["clip"]
    combined, split_summary = _split(combined, config)
    combined, winsor_bounds = _winsorize(combined, config)

    identity = [c for c in IDENTITY_COLUMNS if c in combined.columns]
    non_feature = (
        set(identity)
        | set(TARGET_COLUMNS)
        | set(LABEL_COLUMNS)
        | set(PARAM_COLUMNS)
        | {
            "split",
            "group",
            "is_outlier",
            "log_blur_var",
            "log_glare_pct",
            "variant_index",
            "image_width",
            "image_height",
            "frame_index",
        }
    )
    feature_columns = [
        c
        for c in combined.columns
        if c not in non_feature
        and not c.startswith("quad_")
        and pd.api.types.is_numeric_dtype(combined[c])
    ]
    reduction = _reduce(combined, feature_columns, config)

    validation = _validate(combined, feature_columns, config)
    with open(os.path.join(processed_dir, "validation.json"), "w", encoding="utf-8") as f:
        json.dump(validation, f, indent=2)
    if not validation["validated"]:
        failed = [k for k, v in validation.items() if v is False]
        raise AssertionError(f"Dataset validation failed: {failed}")

    combined.to_parquet(out_path, index=False)

    splits = {
        "summary": split_summary,
        "test_doc_folders": split_summary["test_doc_folders"],
        "val_fraction": config["splits"]["val_fraction"],
        "seed": config["seed"],
    }
    with open(os.path.join(processed_dir, "splits.json"), "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    feature_spec = {
        "seed": config["seed"],
        "feature_columns": feature_columns,
        "selected_features": reduction["selected_features"],
        "top_k_features": reduction["top_k_features"],
        "dropped_low_variance": reduction["dropped_low_variance"],
        "categorical_columns": ["document_type", "capture_condition"],
        "target_columns": TARGET_COLUMNS,
        "label_columns": LABEL_COLUMNS,
        "winsor_bounds": winsor_bounds,
        "resolution_head": config["training"].get("resolution_head", "deterministic"),
        "resolution_rule": {
            "metric": "doc_short_side_px",
            "min_effective_short_side_px": config["labels"]["min_effective_short_side_px"],
            "note": (
                "apply to the measured warped document short side at inference; "
                "deterministic, no learned model"
            ),
        },
        "label_definition": {
            "brightness_label": config["labels"].get("brightness_label", "exposure_adequacy"),
            "brightness_gamma_range": config["labels"]["brightness_gamma_range"],
            "brightness_ev_max": config["labels"].get("brightness_ev_max", 1.0),
            "readable_overall_learned": "readable_blur AND readable_brightness AND readable_glare",
            "review_required": True,
        },
        "ts_portable": True,
    }
    with open(os.path.join(processed_dir, "feature_spec.json"), "w", encoding="utf-8") as f:
        json.dump(feature_spec, f, indent=2)
    with open(os.path.join(processed_dir, "feature_ranking.json"), "w", encoding="utf-8") as f:
        json.dump(reduction["ranking"], f, indent=2)

    reconciliation = {
        "clean": clean_reconciliation,
        "source_gate": {
            "base_crops": len(base),
            "min_blur_var": config["source_gate"]["min_blur_var"],
            "brightness_range": config["source_gate"]["brightness_range"],
            "max_glare_pct": config["source_gate"]["max_glare_pct"],
        },
        "rows": {
            "clean": len(clean_rows),
            "degraded": len(degraded),
            "total": len(combined),
        },
        "split": split_summary,
        "labels": {
            column: int(combined[column].sum()) for column in LABEL_COLUMNS
        },
    }
    with open(os.path.join(processed_dir, "reconciliation_dataset.json"), "w", encoding="utf-8") as f:
        json.dump(reconciliation, f, indent=2)

    print(f"Wrote {len(combined)} rows to {out_path}")
    print("Split:", split_summary["rows"], "Groups:", split_summary["groups"])
    print("Label positives:", reconciliation["labels"])
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Build the model-ready dataset from raw metrics and degradations.")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    parser.add_argument("--max-variants", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    run(config, args.max_variants, args.workers, args.force)


if __name__ == "__main__":
    main()