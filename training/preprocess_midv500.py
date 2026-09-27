"""Ingest MIDV-500 ID/passport frames: pair images with ground-truth quads, warp,
compute raw quality metrics and write the raw metrics table.

Usage:
    python preprocess_midv500.py --config config.yaml
    python preprocess_midv500.py --config config.yaml --every-n 20 --limit-folders 3
"""

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from data_common import (
    PHYSICAL_SIZES_IN,
    dhash,
    document_type_from_folder,
    load_config,
    metrics_on,
    order_corners,
    quad_geometry,
    warp_native,
)

IMAGE_EXTENSIONS = {".tif", ".tiff"}


def _sample_id(rel_path: str) -> str:
    return hashlib.md5(rel_path.encode("utf-8")).hexdigest()[:12]


def _worker(job: dict) -> dict:
    img_path = job["image_path"]
    annotation_path = job["annotation_path"]
    glare_threshold = job["glare_v_threshold"]
    min_quad_area_ratio = job["min_quad_area_ratio"]
    min_edge_px = job["min_edge_px"]
    base = dict(job)
    for key in (
        "image_path",
        "annotation_path",
        "glare_v_threshold",
        "min_quad_area_ratio",
        "min_edge_px",
    ):
        base.pop(key)

    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if image is None:
        return {**base, "error": "unreadable_image"}

    try:
        with open(annotation_path, "r", encoding="utf-8") as f:
            quad_raw = json.load(f)["quad"]
        quad = np.asarray(quad_raw, dtype=np.float64)
    except Exception:
        return {**base, "error": "invalid_annotation"}

    if quad.shape != (4, 2) or not np.isfinite(quad).all():
        return {**base, "error": "invalid_quad_shape"}

    ordered = order_corners(quad)
    geometry = quad_geometry(ordered, image.shape, base["document_type"])
    if geometry["doc_area_ratio"] < min_quad_area_ratio:
        return {**base, "error": "quad_too_small"}
    if geometry["doc_short_side_px"] < min_edge_px:
        return {**base, "error": "quad_edge_too_short"}

    warped = warp_native(image, ordered)
    if warped is None:
        return {**base, "error": "warp_failed"}

    metrics = metrics_on(warped, glare_threshold)
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    dhash_value = dhash(gray)

    row = {
        **base,
        "image_height": image.shape[0],
        "image_width": image.shape[1],
        "quad_tl_x": float(ordered[0][0]),
        "quad_tl_y": float(ordered[0][1]),
        "quad_tr_x": float(ordered[1][0]),
        "quad_tr_y": float(ordered[1][1]),
        "quad_br_x": float(ordered[2][0]),
        "quad_br_y": float(ordered[2][1]),
        "quad_bl_x": float(ordered[3][0]),
        "quad_bl_y": float(ordered[3][1]),
        "dhash": dhash_value,
    }
    row.update(geometry)
    row.update(metrics)
    return row


def discover_jobs(config: dict, every_n: int, limit_folders=None) -> tuple:
    root = config["paths"]["midv500_root"]
    ingest = config["ingest"]
    require_types = set(ingest["require_doc_types"])
    excluded = set(ingest["excluded_folders"])

    doc_folders = sorted(
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
    )
    selected = []
    discovered = []
    for folder in doc_folders:
        doc_type = document_type_from_folder(folder)
        included = doc_type in require_types and folder not in excluded
        discovered.append({"folder": folder, "document_type": doc_type, "included": included})
        if included:
            selected.append((folder, doc_type))
    if limit_folders:
        selected = selected[:limit_folders]

    jobs = []
    for folder, doc_type in selected:
        images_dir = os.path.join(root, folder, "images")
        truth_dir = os.path.join(root, folder, "ground_truth")
        if not os.path.isdir(images_dir):
            continue
        for clip in sorted(os.listdir(images_dir)):
            clip_dir = os.path.join(images_dir, clip)
            if not os.path.isdir(clip_dir):
                continue
            truth_clip_dir = os.path.join(truth_dir, clip)
            frames = sorted(
                f for f in os.listdir(clip_dir) if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
            )
            frames = frames[::every_n]
            for position, frame in enumerate(frames):
                stem = os.path.splitext(frame)[0]
                annotation_path = os.path.join(truth_clip_dir, stem + ".json")
                if not os.path.isfile(annotation_path):
                    continue
                image_path = os.path.join(clip_dir, frame)
                rel_path = os.path.relpath(image_path, root).replace("\\", "/")
                jobs.append(
                    {
                        "sample_id": _sample_id(rel_path),
                        "image_path": image_path,
                        "annotation_path": annotation_path,
                        "rel_path": rel_path,
                        "filepath": image_path.replace("\\", "/"),
                        "doc_folder": folder,
                        "document_type": doc_type,
                        "capture_condition": clip,
                        "clip": clip,
                        "frame_index": position,
                        "glare_v_threshold": ingest["glare_v_threshold"],
                        "min_quad_area_ratio": ingest["min_quad_area_ratio"],
                        "min_edge_px": ingest["min_edge_px"],
                    }
                )
    return jobs, discovered


def write_mapping_doc(discovered: list, config: dict, path: str):
    lines = [
        "# MIDV-500 Document Type Mapping (auto-generated draft)",
        "",
        "REVIEW REQUIRED: verify the id_card / passport assignments and the physical",
        "sizes below before using any labels derived from this mapping.",
        "",
        "| Folder | Type | Physical format | Included |",
        "|---|---|---|---|",
    ]
    for entry in discovered:
        doc_type = entry["document_type"] or "excluded (drvlic/other)"
        if entry["document_type"] in PHYSICAL_SIZES_IN:
            w_in, h_in = PHYSICAL_SIZES_IN[entry["document_type"]]
            size = f"{w_in * 25.4:.1f} x {h_in * 25.4:.1f} mm"
        else:
            size = "-"
        lines.append(
            f"| {entry['folder']} | {doc_type} | {size} | {'yes' if entry['included'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- 47_usa_bordercrossing and 48_usa_passportcard are ID-1 format but excluded by config decision.",
            "- capture_condition values are raw MIDV-500 clip prefixes (CA/CS/HA/HS/KA/KS/PA/PS/TA/TS).",
        ]
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run(config: dict, every_n=None, limit_folders=None, workers=None, force=False):
    paths = config["paths"]
    processed_dir = paths["processed_dir"]
    os.makedirs(processed_dir, exist_ok=True)
    out_path = os.path.join(processed_dir, "raw_metrics.parquet")
    if os.path.exists(out_path) and not force:
        print(f"raw_metrics already exists: {out_path} (use --force to overwrite)")
        return out_path

    every_n = every_n or config["ingest"]["every_n"]
    workers = workers or config["ingest"]["workers"]

    jobs, discovered = discover_jobs(config, every_n, limit_folders)
    selected_count = sum(1 for d in discovered if d["included"]) if not limit_folders else min(
        limit_folders, sum(1 for d in discovered if d["included"])
    )
    print(
        f"Discovered {len(jobs)} frames to process from {selected_count} selected folders "
        f"({len(discovered)} total, every_n={every_n})."
    )
    write_mapping_doc(discovered, config, os.path.join(processed_dir, "DOCUMENT_TYPE_MAPPING.md"))

    rows = []
    failures = {}
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for result in tqdm(
                executor.map(_worker, jobs, chunksize=8), total=len(jobs), desc="Ingesting"
            ):
                if "error" in result:
                    failures[result["error"]] = failures.get(result["error"], 0) + 1
                else:
                    rows.append(result)
    else:
        for job in tqdm(jobs, desc="Ingesting"):
            result = _worker(job)
            if "error" in result:
                failures[result["error"]] = failures.get(result["error"], 0) + 1
            else:
                rows.append(result)

    frame = pd.DataFrame(rows)
    frame.to_parquet(out_path, index=False)
    print(f"Wrote {len(frame)} rows to {out_path}")
    if failures:
        print("Skipped:", failures)

    reconciliation = {
        "stage": "ingest",
        "jobs": len(jobs),
        "kept": len(rows),
        "failures": failures,
        "every_n": every_n,
        "folders_selected": sum(1 for d in discovered if d["included"]),
    }
    pd.Series(reconciliation).to_json(
        os.path.join(processed_dir, "reconciliation_ingest.json"), indent=2
    )
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Ingest MIDV-500 ID/passport frames into a raw metrics table.")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    parser.add_argument("--every-n", type=int, default=None)
    parser.add_argument("--limit-folders", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    run(config, args.every_n, args.limit_folders, args.workers, args.force)


if __name__ == "__main__":
    main()