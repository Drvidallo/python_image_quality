"""Synthetic degradation engine.

Generates controlled blur, motion blur, brightness, glare and resolution
degradations on document crops. Regression targets and readability labels are
derived from the degradation parameters only, never from computed metrics, so
the downstream model cannot learn to invert its own feature computation.

Usage:
    from degradations import sample_variants, apply_variant, targets_and_labels
"""

import numpy as np
import cv2


def _neutral_params() -> dict:
    return {
        "blur_sigma": 0.0,
        "motion_length": 0.0,
        "motion_angle": 0.0,
        "brightness_gamma": 1.0,
        "brightness_ev": 0.0,
        "glare_radius_frac": 0.0,
        "downscale_factor": 1.0,
    }


def sample_variants(config: dict, rng: np.random.Generator, count: int) -> list:
    grid = config["degradations"]
    variants = [_neutral_params()]
    for _ in range(max(0, count - 1)):
        variants.append(
            {
                "blur_sigma": float(rng.choice(grid["blur_sigmas"])),
                "motion_length": float(rng.choice(grid["motion_lengths"])),
                "motion_angle": float(rng.uniform(0.0, 180.0)),
                "brightness_gamma": float(rng.choice(grid["brightness_gammas"])),
                "brightness_ev": float(rng.choice(grid["brightness_evs"])),
                "glare_radius_frac": float(rng.choice(grid["glare_radius_fracs"])),
                "downscale_factor": float(rng.choice(grid["downscale_factors"])),
            }
        )
    return variants


def apply_gaussian_blur(img, sigma: float):
    if sigma <= 0:
        return img
    return cv2.GaussianBlur(img, (0, 0), sigma)


def apply_motion_blur(img, length: float, angle: float):
    size = int(round(length))
    if size < 3:
        return img
    if size % 2 == 0:
        size += 1
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[size // 2, :] = 1.0
    matrix = cv2.getRotationMatrix2D((size / 2.0 - 0.5, size / 2.0 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, matrix, (size, size))
    total = kernel.sum()
    if total <= 0:
        return img
    kernel /= total
    return cv2.filter2D(img, -1, kernel)


def apply_brightness_gamma(img, gamma: float):
    if abs(gamma - 1.0) < 1e-6:
        return img
    table = np.array(
        [((value / 255.0) ** gamma) * 255.0 for value in range(256)], dtype=np.uint8
    )
    return cv2.LUT(img, table)


def apply_exposure(img, ev: float):
    if abs(ev) < 1e-6:
        return img
    scale = 2.0**ev
    table = np.array(
        [min(255.0, max(0.0, value * scale)) for value in range(256)], dtype=np.uint8
    )
    return cv2.LUT(img, table)


def apply_glare(img, radius_frac: float, rng: np.random.Generator):
    if radius_frac <= 0:
        return img
    h, w = img.shape[:2]
    radius = max(2, int(round(radius_frac * min(h, w))))
    center = (
        int(rng.uniform(0.3 * w, 0.7 * w)),
        int(rng.uniform(0.3 * h, 0.7 * h)),
    )
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.circle(mask, center, radius, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), max(1.0, radius * 0.35))
    mask = np.clip(mask, 0.0, 1.0)[..., None]
    blended = img.astype(np.float32) * (1.0 - mask) + 255.0 * mask
    return np.clip(blended, 0, 255).astype(np.uint8)


def apply_downscale(img, factor: float):
    if factor >= 1.0:
        return img
    h, w = img.shape[:2]
    small = cv2.resize(
        img, (max(1, int(round(w * factor))), max(1, int(round(h * factor)))),
        interpolation=cv2.INTER_AREA,
    )
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def apply_variant(img, params: dict, rng: np.random.Generator):
    result = apply_gaussian_blur(img, params["blur_sigma"])
    result = apply_motion_blur(result, params["motion_length"], params["motion_angle"])
    result = apply_brightness_gamma(result, params["brightness_gamma"])
    result = apply_exposure(result, params.get("brightness_ev", 0.0))
    result = apply_glare(result, params["glare_radius_frac"], rng)
    result = apply_downscale(result, params["downscale_factor"])
    return result


def severity_vector(params: dict) -> dict:
    blur_severity = max(params["blur_sigma"] / 8.0, params["motion_length"] / 35.0)
    brightness_severity = max(
        abs(params["brightness_gamma"] - 1.0) / 0.5,
        abs(params.get("brightness_ev", 0.0)) / 1.5,
    )
    glare_severity = params["glare_radius_frac"] / 0.14
    resolution_severity = (1.0 - params["downscale_factor"]) / 0.85
    vector = {
        "blur_severity": float(min(1.0, blur_severity)),
        "brightness_severity": float(min(1.0, brightness_severity)),
        "glare_severity": float(min(1.0, glare_severity)),
        "resolution_severity": float(min(1.0, resolution_severity)),
    }
    vector["severity_index"] = float(np.mean(list(vector.values())))
    return vector


def targets_and_labels(params: dict, native_short_side_px: float, config: dict) -> dict:
    labels_config = config["labels"]
    gamma_lo, gamma_hi = labels_config["brightness_gamma_range"]
    ev_max = labels_config.get("brightness_ev_max", 1.0)
    brightness_mode = labels_config.get("brightness_label", "exposure_adequacy")
    effective_short_side = native_short_side_px * params["downscale_factor"]

    readable_blur = (
        params["blur_sigma"] <= labels_config["blur_sigma_max"]
        and params["motion_length"] <= labels_config["motion_length_max"]
    )
    gamma_ok = gamma_lo <= params["brightness_gamma"] <= gamma_hi
    if brightness_mode == "exposure_adequacy":
        readable_brightness = gamma_ok and abs(params.get("brightness_ev", 0.0)) <= ev_max
    else:
        readable_brightness = gamma_ok
    readable_glare = params["glare_radius_frac"] <= labels_config["glare_radius_frac_max"]
    readable_resolution = effective_short_side >= labels_config["min_effective_short_side_px"]

    targets = {
        "target_blur_sigma": params["blur_sigma"],
        "target_motion_length": params["motion_length"],
        "target_motion_angle": params["motion_angle"],
        "target_brightness_gamma": params["brightness_gamma"],
        "target_brightness_ev": params.get("brightness_ev", 0.0),
        "target_glare_radius_frac": params["glare_radius_frac"],
        "target_downscale_factor": params["downscale_factor"],
        "effective_short_side_px": float(effective_short_side),
    }
    targets.update(severity_vector(params))

    labels = {
        "readable_blur": int(readable_blur),
        "readable_brightness": int(readable_brightness),
        "readable_glare": int(readable_glare),
        "readable_resolution": int(readable_resolution),
    }
    labels["readable_overall"] = int(all(labels.values()))
    return {**targets, **labels}
