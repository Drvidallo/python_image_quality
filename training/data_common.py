"""Shared document geometry, quality metrics and config utilities for the EDA pipeline."""

import os

import cv2
import numpy as np
import pandas as pd
import yaml

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

PHYSICAL_SIZES_IN = {
    "id_card": (85.6 / 25.4, 53.98 / 25.4),
    "passport": (125.0 / 25.4, 88.0 / 25.4),
}

EXPECTED_ASPECT = {
    "id_card": 85.6 / 53.98,
    "passport": 125.0 / 88.0,
}

DEFAULT_THRESHOLDS = {
    "blur_var_fail": 30.0,
    "blur_var_warn": 60.0,
    "brightness_dark_fail": 50.0,
    "brightness_dark_warn": 70.0,
    "brightness_over_fail": 230.0,
    "brightness_over_warn": 205.0,
    "glare_pct_fail": 8.0,
    "glare_pct_warn": 3.0,
    "glare_blob_fail": 3.0,
    "glare_blob_warn": 1.0,
    "resolution_px_fail": 500,
    "resolution_px_warn": 800,
    "glare_v_threshold": 250,
}


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def document_type_from_folder(folder_name: str):
    if folder_name.endswith("_passport") or "_passport_" in folder_name:
        return "passport"
    if folder_name.endswith("_id") or "_id_" in folder_name:
        return "id_card"
    return None


def _linear(value: float, lo: float, hi: float) -> float:
    return float(max(0.0, min(1.0, (value - lo) / (hi - lo))))


def order_corners(quad) -> np.ndarray:
    pts = np.asarray(quad, dtype=np.float32).reshape(-1, 2)
    s = pts.sum(axis=1)
    d = pts[:, 0] - pts[:, 1]
    tl = pts[int(np.argmin(s))]
    br = pts[int(np.argmax(s))]
    tr = pts[int(np.argmax(d))]
    bl = pts[int(np.argmin(d))]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _largest_quad_contour(binary, h, w):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0.0
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            area = cv2.contourArea(approx)
            if area > best_area and area > 0.05 * h * w:
                best = approx.reshape(4, 2)
                best_area = area
    if best is None:
        return None
    return order_corners(best)


def detect_document_quad(img_bgr):
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    quad = _largest_quad_contour(cv2.Canny(blurred, 50, 150), h, w)
    if quad is not None:
        return quad

    threshold = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return _largest_quad_contour(threshold, h, w)


def warp_native(img_bgr, quad):
    tl, tr, br, bl = order_corners(quad)
    top = float(np.linalg.norm(tr - tl))
    bottom = float(np.linalg.norm(br - bl))
    left = float(np.linalg.norm(bl - tl))
    right = float(np.linalg.norm(br - tr))
    out_w = int(round((top + bottom) / 2.0))
    out_h = int(round((left + right) / 2.0))
    if out_w < 1 or out_h < 1:
        return None
    src = np.array([tl, tr, br, bl], dtype=np.float32)
    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img_bgr, matrix, (out_w, out_h))


def _glare_stats(img_bgr, glare_v_threshold: int) -> dict:
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    saturation = hsv[:, :, 1]
    mask = (value >= glare_v_threshold).astype(np.uint8)
    glare_pct = float(mask.mean() * 100.0)

    result = {
        "glare_pct": glare_pct,
        "glare_blob_pct": 0.0,
        "glare_blob_aspect": 0.0,
        "glare_center_dist": 0.0,
        "glare_sat_mean": 0.0,
    }
    if glare_pct <= 0.0:
        return result

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return result

    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = int(np.argmax(areas)) + 1
    blob_area = int(stats[idx, cv2.CC_STAT_AREA])
    bw = int(stats[idx, cv2.CC_STAT_WIDTH])
    bh = int(stats[idx, cv2.CC_STAT_HEIGHT])
    cx, cy = centroids[idx]

    result["glare_blob_pct"] = float(blob_area) / float(h * w) * 100.0
    result["glare_blob_aspect"] = float(max(bw, bh)) / float(max(1, min(bw, bh)))
    result["glare_center_dist"] = float(
        np.hypot(cx - w / 2.0, cy - h / 2.0) / np.hypot(w / 2.0, h / 2.0)
    )
    blob_pixels = labels == idx
    if blob_pixels.any():
        result["glare_sat_mean"] = float(saturation[blob_pixels].mean())
    return result


def metrics_on(img_bgr, glare_v_threshold: int = 250) -> dict:
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    gray_half = cv2.resize(gray, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_AREA)
    blur_var_half = float(cv2.Laplacian(gray_half, cv2.CV_64F).var())
    gray_big = cv2.resize(
        gray, (max(1, int(w * 1.5)), max(1, int(h * 1.5))), interpolation=cv2.INTER_LINEAR
    )
    blur_var_big = float(cv2.Laplacian(gray_big, cv2.CV_64F).var())

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = float((gx * gx + gy * gy).mean())

    small = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32)
    if float(small.std()) < 1e-6:
        fft_hf_ratio = 0.0
    else:
        spectrum = np.abs(np.fft.fftshift(np.fft.fft2(small)))
        cy, cx = 128, 128
        low_energy = float(spectrum[cy - 32 : cy + 32, cx - 32 : cx + 32].sum())
        total_energy = float(spectrum.sum()) + 1e-9
        fft_hf_ratio = 1.0 - low_energy / total_energy

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    lightness = lab[:, :, 0].astype(np.float32)
    brightness_L = float(lightness.mean())
    brightness_std = float(lightness.std())
    brightness_p05, brightness_p50, brightness_p95 = (
        float(v) for v in np.percentile(lightness, [5, 50, 95])
    )
    overexposed_pct = float((lightness >= 250).mean() * 100.0)
    underexposed_pct = float((lightness <= 5).mean() * 100.0)

    sampled = lightness[::4, ::4]
    sampled_mean = float(sampled.mean())
    sampled_std = float(sampled.std())
    if sampled_std > 1e-6:
        z = (sampled - sampled_mean) / sampled_std
        brightness_skew = float((z**3).mean())
        brightness_kurtosis = float((z**4).mean() - 3.0)
    else:
        brightness_skew = 0.0
        brightness_kurtosis = 0.0
    brightness_q_ratio = (brightness_p95 - brightness_p50) / (
        brightness_p50 - brightness_p05 + 1e-6
    )
    local_contrast = brightness_std / (brightness_L + 1e-6)

    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    probabilities = histogram / max(1.0, histogram.sum())
    positive = probabilities[probabilities > 0]
    entropy = float(-(positive * np.log2(positive)).sum())

    result = {
        "blur_var": blur_var,
        "blur_var_half": blur_var_half,
        "blur_var_big": blur_var_big,
        "blur_scale_ratio": blur_var_big / (blur_var_half + 1e-9),
        "tenengrad": tenengrad,
        "fft_hf_ratio": fft_hf_ratio,
        "brightness_L": brightness_L,
        "brightness_std": brightness_std,
        "brightness_p05": brightness_p05,
        "brightness_p50": brightness_p50,
        "brightness_p95": brightness_p95,
        "brightness_q_ratio": brightness_q_ratio,
        "brightness_skew": brightness_skew,
        "brightness_kurtosis": brightness_kurtosis,
        "local_contrast": local_contrast,
        "overexposed_pct": overexposed_pct,
        "underexposed_pct": underexposed_pct,
        "entropy": entropy,
    }
    result.update(_glare_stats(img_bgr, glare_v_threshold))
    return result


def resolution_readable(short_side_px: float, min_effective_short_side_px: float) -> int:
    """Deterministic resolution readability rule.

    In production this is applied to the measured short side (in pixels) of the
    warped document; no learned model is used for the resolution dimension.
    """
    return int(short_side_px >= min_effective_short_side_px)


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Derived model features computed identically in training and inference."""
    result = frame.copy()
    result["log_blur_var"] = np.log1p(result["blur_var"])
    result["log_glare_pct"] = np.log1p(result["glare_pct"])
    result["log_glare_blob_pct"] = np.log1p(result["glare_blob_pct"])
    result["log_doc_area_ratio"] = np.log1p(result["doc_area_ratio"])
    result["brightness_dev"] = result["brightness_L"] - 128.0
    result["brightness_balance"] = 1.0 - (result["brightness_dev"].abs() / 128.0)
    result["log_blur_scale_gap"] = np.log1p(result["blur_var_big"]) - np.log1p(
        result["blur_var_half"]
    )
    blur_norm = np.clip(np.log1p(result["blur_var"]) / np.log1p(500.0), 0, 1)
    bright_norm = np.clip(1.0 - result["brightness_dev"].abs() / 128.0, 0, 1)
    glare_norm = np.clip(1.0 - result["glare_pct"] / 10.0, 0, 1)
    resolution_norm = np.clip(result["doc_short_side_px"] / 1200.0, 0, 1)
    result["quality_prior"] = (blur_norm + bright_norm + glare_norm + resolution_norm) / 4.0
    result["log_blur_x_brightness_dev"] = result["log_blur_var"] * result["brightness_dev"]
    result["glare_ratio"] = result["glare_blob_pct"] / (result["glare_pct"] + 1e-6)
    result["doc_type_id_card"] = (result["document_type"] == "id_card").astype(int)
    result["doc_type_passport"] = (result["document_type"] == "passport").astype(int)
    result["capture_condition_code"] = pd.Categorical(result["capture_condition"]).codes
    return result


def quad_geometry(quad, img_shape, document_type=None) -> dict:
    ordered = order_corners(quad)
    tl, tr, br, bl = ordered
    top = float(np.linalg.norm(tr - tl))
    bottom = float(np.linalg.norm(br - bl))
    left = float(np.linalg.norm(bl - tl))
    right = float(np.linalg.norm(br - tr))
    edges = np.array([top, bottom, left, right], dtype=np.float64)
    doc_w = float((top + bottom) / 2.0)
    doc_h = float((left + right) / 2.0)
    short_side = float(min(doc_w, doc_h))
    h, w = img_shape[:2]

    polygon = ordered.astype(np.float64)
    x = polygon[:, 0]
    y = polygon[:, 1]
    polygon_area = float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2.0)

    aspect = float(max(doc_w, doc_h) / max(1.0, min(doc_w, doc_h)))
    expected = EXPECTED_ASPECT.get(document_type)
    aspect_deviation = abs(aspect - expected) / expected if expected else np.nan

    est_dpi = np.nan
    if document_type in PHYSICAL_SIZES_IN:
        phys_w, phys_h = PHYSICAL_SIZES_IN[document_type]
        est_dpi = float(min(doc_w / phys_w, doc_h / phys_h))

    return {
        "doc_width_px": doc_w,
        "doc_height_px": doc_h,
        "doc_short_side_px": short_side,
        "doc_area_ratio": polygon_area / float(h * w),
        "aspect_ratio": aspect,
        "aspect_deviation": aspect_deviation,
        "edge_cv": float(edges.std() / (edges.mean() + 1e-9)),
        "est_dpi": est_dpi,
    }


def dhash(gray, hash_size: int = 8) -> int:
    small = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    packed = np.packbits(bits)
    return int.from_bytes(packed.tobytes(), "big")


def hamming_distance(a: int, b: int) -> int:
    return int(bin(int(a) ^ int(b)).count("1"))


def compute_scores(metrics: dict, short_side_px: int, thresholds: dict) -> dict:
    t = thresholds
    blur = metrics["blur_var"]
    lum = metrics["brightness_L"]
    glare = metrics["glare_pct"]
    blob = metrics["glare_blob_pct"]

    blur_score = _linear(blur, 15.0, 150.0) * 100.0
    if lum <= 60.0:
        brightness_score = _linear(lum, 20.0, 60.0) * 100.0
    elif lum >= 190.0:
        brightness_score = _linear(250.0 - lum, 0.0, 60.0) * 100.0
    else:
        brightness_score = 100.0
    glare_score = _linear(10.0 - glare, 0.0, 10.0) * 100.0
    resolution_score = _linear(short_side_px, 300.0, 1200.0) * 100.0

    blur_status = _status_low(blur, t["blur_var_warn"], t["blur_var_fail"])
    brightness_status = "pass"
    if lum <= t["brightness_dark_fail"] or lum >= t["brightness_over_fail"]:
        brightness_status = "fail"
    elif lum <= t["brightness_dark_warn"] or lum >= t["brightness_over_warn"]:
        brightness_status = "warn"
    glare_status = _status_high(
        max(glare, 2.0 * blob), t["glare_pct_warn"], t["glare_pct_fail"]
    )
    resolution_status = _status_low(short_side_px, t["resolution_px_warn"], t["resolution_px_fail"])

    overall = (blur_score + brightness_score + glare_score + resolution_score) / 4.0

    return {
        "blur_score": blur_score,
        "blur_status": blur_status,
        "brightness_score": brightness_score,
        "brightness_status": brightness_status,
        "glare_score": glare_score,
        "glare_status": glare_status,
        "resolution_score": resolution_score,
        "resolution_status": resolution_status,
        "overall_score": overall,
        "readable": _readable([blur_status, brightness_status, glare_status, resolution_status]),
    }


def _status_low(value, warn_under, fail_under) -> str:
    if value < fail_under:
        return "fail"
    if value < warn_under:
        return "warn"
    return "pass"


def _status_high(value, warn_over, fail_over) -> str:
    if value > fail_over:
        return "fail"
    if value > warn_over:
        return "warn"
    return "pass"


def _readable(statuses) -> str:
    if "fail" in statuses:
        return "fail"
    if "warn" in statuses:
        return "warn"
    return "pass"
