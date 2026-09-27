"""Score blur, brightness, glare and resolution of images placed in a folder.

Usage:
    python score_images.py --input-folder "path/to/my/images"
    python score_images.py --input-folder "path" --output-dir "path/out" --document-type id_card --detect-document

Output is a CSV with one row per image: raw metrics, 0-100 scores, pass/warn/fail
status per metric, and an overall readability flag.

The score mappings and thresholds in DEFAULT_THRESHOLDS are PRELIMINARY
baselines (tuned roughly from MIDV-500 measurements). They will be replaced by
the EDA-learned model once training data exists.
"""

import argparse
import csv
import os
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm

from data_common import (
    DEFAULT_THRESHOLDS,
    PHYSICAL_SIZES_IN,
    SUPPORTED_EXTENSIONS,
    compute_scores,
    detect_document_quad,
    metrics_on,
    warp_native,
)


def score_image(path: str, document_type, detect_document: bool, thresholds: dict) -> dict:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return {"filepath": path, "error": "unreadable"}
    img_h, img_w = img.shape[:2]

    doc_detected = False
    doc_short_side_px = min(img_w, img_h)
    est_dpi = None
    doc_metrics = metrics_on(img, thresholds["glare_v_threshold"])

    if detect_document:
        quad = detect_document_quad(img)
        if quad is not None:
            warped = warp_native(img, quad)
            if warped is not None:
                doc_metrics = metrics_on(warped, thresholds["glare_v_threshold"])
                doc_short_side_px = min(warped.shape[1], warped.shape[0])
                doc_detected = True
                if document_type in PHYSICAL_SIZES_IN:
                    w_in, h_in = PHYSICAL_SIZES_IN[document_type]
                    est_dpi = min(warped.shape[1] / w_in, warped.shape[0] / h_in)

    scores = compute_scores(doc_metrics, doc_short_side_px, thresholds)

    row = {
        "filepath": path,
        "image_width": img_w,
        "image_height": img_h,
        "megapixels": round(img_w * img_h / 1e6, 3),
        "document_type": document_type or "",
        "doc_detected": doc_detected,
        "doc_short_side_px": doc_short_side_px,
        "est_dpi": "" if est_dpi is None else round(est_dpi, 1),
        "blur_lapvar": round(doc_metrics["blur_var"], 3),
        "brightness_L": round(doc_metrics["brightness_L"], 3),
        "glare_pct": round(doc_metrics["glare_pct"], 3),
        "glare_blob_pct": round(doc_metrics["glare_blob_pct"], 3),
    }
    row.update({k: (round(v, 1) if isinstance(v, float) else v) for k, v in scores.items()})
    return row


def collect_images(input_folder: str, recursive: bool) -> list:
    pattern = os.path.join(input_folder, "**", "*") if recursive else os.path.join(input_folder, "*")
    files = []
    for p in glob(pattern, recursive=recursive):
        if os.path.isfile(p) and os.path.splitext(p)[1].lower() in SUPPORTED_EXTENSIONS:
            files.append(p)
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description="Score blur/brightness/glare/resolution of images in a folder.")
    parser.add_argument("--input-folder", default="data/input_images", help="Folder containing the images to score.")
    parser.add_argument("--output-dir", default="data/scores", help="Folder where results.csv is written.")
    parser.add_argument("--document-type", default=None, choices=[None, "id_card", "passport"], help="Known physical document format, used for DPI estimation.")
    parser.add_argument("--detect-document", action="store_true", help="Auto-detect and crop the document before scoring.")
    parser.add_argument("--glare-v-threshold", type=int, default=DEFAULT_THRESHOLDS["glare_v_threshold"])
    parser.add_argument("--recursive", action="store_true", help="Also scan subfolders of --input-folder.")
    args = parser.parse_args()

    os.makedirs(args.input_folder, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    files = collect_images(args.input_folder, args.recursive)
    if not files:
        print(f"No supported images found in {args.input_folder}.")
        print(f"Put your images there and re-run, or pass a different --input-folder.")
        return

    thresholds = dict(DEFAULT_THRESHOLDS)
    thresholds["glare_v_threshold"] = args.glare_v_threshold

    rows = []
    failed = 0
    for path in tqdm(files, desc="Scoring"):
        row = score_image(path, args.document_type, args.detect_document, thresholds)
        if "error" in row:
            failed += 1
            print("  failed:", path)
            continue
        rows.append(row)

    out_path = os.path.join(args.output_dir, "results.csv")
    fields = [
        "filepath",
        "image_width",
        "image_height",
        "megapixels",
        "document_type",
        "doc_detected",
        "doc_short_side_px",
        "est_dpi",
        "blur_lapvar",
        "brightness_L",
        "glare_pct",
        "glare_blob_pct",
        "blur_score",
        "blur_status",
        "brightness_score",
        "brightness_status",
        "glare_score",
        "glare_status",
        "resolution_score",
        "resolution_status",
        "overall_score",
        "readable",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nScored {len(rows)} image(s), {failed} failed.")
    print(f"Results written to: {out_path}")
    for key, label in [
        ("blur_score", "blur"),
        ("brightness_score", "brightness"),
        ("glare_score", "glare"),
        ("resolution_score", "resolution"),
        ("overall_score", "overall"),
    ]:
        values = [r[key] for r in rows]
        print(f"  mean {label:<12} score: {np.mean(values):6.1f}   min {np.min(values):.1f}  max {np.max(values):.1f}")
    print(
        f"  readable counts: pass={sum(1 for r in rows if r['readable']=='pass')} "
        f"warn={sum(1 for r in rows if r['readable']=='warn')} "
        f"fail={sum(1 for r in rows if r['readable']=='fail')}"
    )


if __name__ == "__main__":
    main()