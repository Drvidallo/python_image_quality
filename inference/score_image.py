"""Score ID images (e.g. driver licences) with the learned quality models.

Usage:
    uv run python inference/score_image.py
    uv run python inference/score_image.py --input-folder "C:/path/to/images"
    uv run python inference/score_image.py --document-type passport --no-detect

Scores are 0-100 (higher is better). Each dimension passes when the score
reaches its threshold; thresholds are editable in inference_config.json.
Resolution is deterministic: score = 100 * min(short_side_px / reference_px, 1).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
EDA_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(EDA_ROOT / "training"))

from data_common import (  # noqa: E402
    SUPPORTED_EXTENSIONS,
    detect_document_quad,
    engineer_features,
    metrics_on,
    order_corners,
    quad_geometry,
    warp_native,
)

DEFAULT_THRESHOLDS = {"blur": 55, "brightness": 35, "glare": 35, "resolution": 50}
DEFAULT_REFERENCE_PX = 1000.0
GLARE_V_THRESHOLD = 250
MODEL_FILES = {
    "blur": "clf_readable_blur.joblib",
    "brightness": "clf_readable_brightness.joblib",
    "glare": "clf_readable_glare.joblib",
}
DIMENSIONS = ["blur", "brightness", "glare", "resolution"]


def _friendly(value):
    value = float(value)
    return int(value) if value.is_integer() else value


def load_inference_config(path: Path):
    thresholds = {key: float(value) for key, value in DEFAULT_THRESHOLDS.items()}
    reference_px = DEFAULT_REFERENCE_PX
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
        thresholds.update({key: float(value) for key, value in config.get("thresholds", {}).items()})
        reference_px = float(config.get("resolution_reference_px", reference_px))
    return thresholds, reference_px


def load_models(models_dir: Path):
    models = {}
    for name, filename in MODEL_FILES.items():
        path = models_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"model not found: {path}")
        models[name] = joblib.load(path)
    return models


def extract_features(img_bgr, quad, document_type: str, capture_condition_code: int) -> dict:
    ordered = order_corners(quad)
    warped = warp_native(img_bgr, ordered)
    if warped is None:
        raise ValueError("warp produced an empty document image")
    row = {}
    row.update(quad_geometry(ordered, img_bgr.shape, document_type))
    row.update(metrics_on(warped, GLARE_V_THRESHOLD))
    row["document_type"] = document_type
    row["capture_condition"] = "unknown"
    frame = engineer_features(pd.DataFrame([row]))
    frame["capture_condition_code"] = int(capture_condition_code)
    return frame.iloc[0].to_dict()


def score_features(features: dict, spec: dict, models: dict, thresholds: dict, reference_px: float):
    feature_columns = spec["selected_features"]
    values = {}
    warnings = []
    for name in feature_columns:
        value = features.get(name)
        if value is None or not np.isfinite(float(value)):
            values[name] = 0.0
            warnings.append(f"missing_feature:{name}")
        else:
            values[name] = float(value)
    if values.get("blur_var", 0.0) <= 0.0 and values.get("tenengrad", 0.0) <= 0.0:
        warnings.append("degenerate_document_crop: image appears blank or black")
    matrix = pd.DataFrame([values])[feature_columns]

    scores = {}
    for name in ["blur", "brightness", "glare"]:
        probability = float(models[name].predict_proba(matrix)[:, 1][0])
        score = round(100.0 * probability, 1)
        scores[name] = {
            "score": score,
            "threshold": _friendly(thresholds[name]),
            "is_pass": bool(score >= thresholds[name]),
        }

    short_side_px = float(features.get("doc_short_side_px", 0.0))
    resolution_score = round(100.0 * min(short_side_px / reference_px, 1.0), 1)
    resolution = {
        "score": resolution_score,
        "threshold": _friendly(thresholds["resolution"]),
        "is_pass": bool(resolution_score >= thresholds["resolution"]),
        "short_side_px": round(short_side_px, 1),
    }
    est_dpi = features.get("est_dpi")
    if est_dpi is not None and np.isfinite(float(est_dpi)):
        resolution["est_dpi"] = round(float(est_dpi), 1)
    scores["resolution"] = resolution
    return scores, warnings


def score_image(
    path: str,
    models: dict,
    spec: dict,
    thresholds: dict,
    reference_px: float,
    document_type: str,
    detect: bool,
    capture_condition_code: int,
) -> dict:
    name = os.path.basename(path)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return {"image": name, "error": "unreadable"}

    warnings = []
    quad = detect_document_quad(img) if detect else None
    detected = quad is not None
    if not detected:
        if detect:
            warnings.append(
                "document_not_detected: scored on the full image; brightness, glare and "
                "resolution may be less accurate"
            )
        height, width = img.shape[:2]
        quad = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float64
        )

    features = extract_features(img, quad, document_type, capture_condition_code)
    scores, feature_warnings = score_features(features, spec, models, thresholds, reference_px)
    warnings.extend(feature_warnings)

    return {
        "image": name,
        "document_detected": detected,
        "document_type": document_type,
        "blur": scores["blur"],
        "brightness": scores["brightness"],
        "glare": scores["glare"],
        "resolution": scores["resolution"],
        "overall_pass": all(scores[dimension]["is_pass"] for dimension in DIMENSIONS),
        "raw_metrics": {
            "blur_var": round(float(features["blur_var"]), 3),
            "brightness_L": round(float(features["brightness_L"]), 3),
            "glare_pct": round(float(features["glare_pct"]), 3),
            "glare_blob_pct": round(float(features["glare_blob_pct"]), 3),
        },
        "warnings": warnings,
    }


def main():
    parser = argparse.ArgumentParser(description="Score ID images with the learned quality models.")
    parser.add_argument("--input-folder", default=str(SCRIPT_DIR / "test_images"))
    parser.add_argument("--output", default=str(SCRIPT_DIR / "results.json"))
    parser.add_argument("--models-dir", default=str(EDA_ROOT / "models"))
    parser.add_argument(
        "--spec",
        default=str(EDA_ROOT / "data" / "midv500" / "processed" / "feature_spec.json"),
    )
    parser.add_argument("--config", default=str(SCRIPT_DIR / "inference_config.json"))
    parser.add_argument("--document-type", choices=["id_card", "passport"], default="id_card")
    parser.add_argument("--capture-condition-code", type=int, default=0)
    parser.add_argument("--no-detect", action="store_true")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--blur-threshold", type=float, default=None)
    parser.add_argument("--brightness-threshold", type=float, default=None)
    parser.add_argument("--glare-threshold", type=float, default=None)
    parser.add_argument("--resolution-threshold", type=float, default=None)
    args = parser.parse_args()

    thresholds, reference_px = load_inference_config(Path(args.config))
    for name in DIMENSIONS:
        override = getattr(args, f"{name}_threshold")
        if override is not None:
            thresholds[name] = float(override)

    input_folder = Path(args.input_folder)
    input_folder.mkdir(parents=True, exist_ok=True)
    pattern = "**/*" if args.recursive else "*"
    files = sorted(
        path
        for path in input_folder.glob(pattern)
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        print(f"No supported images found in {input_folder}.")
        print("Put your image there and re-run, or pass --input-folder <path>.")
        return

    with open(args.spec, "r", encoding="utf-8") as f:
        spec = json.load(f)
    models = load_models(Path(args.models_dir))

    results = [
        score_image(
            str(path),
            models,
            spec,
            thresholds,
            reference_px,
            args.document_type,
            not args.no_detect,
            args.capture_condition_code,
        )
        for path in files
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "document_type": args.document_type,
        "thresholds": {key: _friendly(value) for key, value in thresholds.items()},
        "resolution_reference_px": reference_px,
        "results": results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nScored {len(results)} image(s). Results: {output_path}\n")
    header = f"{'image':<30} {'blur':>7} {'bright':>7} {'glare':>7} {'res':>7}  overall"
    print(header)
    print("-" * len(header))
    for result in results:
        if "error" in result:
            print(f"{result['image']:<30} {result['error']}")
            continue
        print(
            f"{result['image']:<30} {result['blur']['score']:>7.1f} "
            f"{result['brightness']['score']:>7.1f} {result['glare']['score']:>7.1f} "
            f"{result['resolution']['score']:>7.1f}  "
            f"{'PASS' if result['overall_pass'] else 'FAIL'}"
        )
    print(f"\nPass when score >= threshold: {payload['thresholds']}")


if __name__ == "__main__":
    main()