"""
Post-inference quality validation for try-on results.

Validates garment geometry, upper-body preservation, fabric realism,
and overall output quality.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("idm-vton.worker.quality")


# ── SCHP labels ──────────────────────────────────────────────────────────────

_LABEL_SKIRT = 5
_LABEL_PANTS = 6
_LABEL_UPPER_CLOTHES = 4
_LABEL_LEFT_LEG = 12
_LABEL_RIGHT_LEG = 13

# ── Source label map: labels associated with each cloth_type in source ────────

source_label_map: dict[str, set[int]] = {
    "upper_body": {_LABEL_UPPER_CLOTHES},
    "lower_body": {_LABEL_PANTS, _LABEL_SKIRT},
    "dresses": {_LABEL_UPPER_CLOTHES, _LABEL_PANTS, _LABEL_SKIRT},
}


def _garment_geometry_score(
    result: Image.Image,
    mask_np: np.ndarray,
    target_cloth_type: str,
) -> float:
    """
    Verify generated garment has correct geometry for its type.

    Returns score 0..1 (higher is better).
    """
    out = np.array(result.convert("RGB"), dtype=np.float32)
    h, w = out.shape[:2]

    # Find the inpainted region
    if mask_np.shape[:2] != (h, w):
        mask_resized = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        mask_resized = mask_np

    inpaint_region = mask_resized > 127
    if not np.any(inpaint_region):
        return 0.0

    # Get bounding box of inpainted region
    ys, xs = np.where(inpaint_region)
    y_min, y_max = ys.min() / h, ys.max() / h
    x_min, x_max = xs.min() / w, xs.max() / w

    centroid_y = (y_min + y_max) / 2
    vertical_span = y_max - y_min

    score = 0.0

    if target_cloth_type == "lower_body":
        # Lower body: centroid should be in bottom 50%, span moderate
        if 0.4 < centroid_y < 0.9:
            score += 0.2
        if 0.2 < vertical_span < 0.7:
            score += 0.15
        # Check it doesn't extend too far into upper body
        if y_min > 0.15:
            score += 0.15

    elif target_cloth_type == "upper_body":
        # Upper body: centroid should be in top 60%
        if 0.1 < centroid_y < 0.6:
            score += 0.2
        if 0.1 < vertical_span < 0.5:
            score += 0.15
        # Should not extend below waist
        if y_max < 0.8:
            score += 0.15

    elif target_cloth_type == "dresses":
        # Dresses: should span > 50% vertically
        if vertical_span > 0.5:
            score += 0.2
        if 0.2 < centroid_y < 0.8:
            score += 0.15
        if x_max - x_min > 0.15:
            score += 0.15

    return min(score, 1.0)


def _upper_body_preservation_score(
    person: Image.Image,
    result: Image.Image,
    cloth_type: str,
) -> float:
    """
    Verify the upper body is preserved during lower-body try-on.

    Compares the top 40% of the image (person vs result) — for lower-body
    try-on, this region should be nearly identical.

    Returns score 0..100 (higher = more preserved).
    """
    if cloth_type != "lower_body":
        return 100.0

    orig = np.array(person.convert("RGB"), dtype=np.float32)
    out = np.array(result.convert("RGB"), dtype=np.float32)

    if orig.shape != out.shape:
        out = np.array(result.convert("RGB").resize(person.size, Image.LANCZOS), dtype=np.float32)

    h = orig.shape[0]
    # Upper body band: top 40% — covers face, shoulders, chest, arms
    upper_band = slice(0, int(0.40 * h), None)

    orig_upper = orig[upper_band]
    out_upper = out[upper_band]

    # Mean absolute per-pixel difference across RGB channels
    diff = np.mean(np.abs(orig_upper - out_upper), axis=2)
    mean_diff = float(np.mean(diff))

    # Score: 0 drift = 100, 30+ drift = 0
    score = max(0.0, 100.0 - (mean_diff / 30.0) * 100.0)
    return score


def _face_quality_score(
    person: Image.Image,
    result: Image.Image,
) -> float:
    """
    Verify face identity is preserved.

    Compares the top 22% of the image (face band) between person and result.
    Any significant drift indicates face regeneration or identity loss.

    Returns score 0..100 (higher = better face preservation).
    """
    orig = np.array(person.convert("RGB"), dtype=np.float32)
    out = np.array(result.convert("RGB"), dtype=np.float32)

    if orig.shape != out.shape:
        out = np.array(result.convert("RGB").resize(person.size, Image.LANCZOS), dtype=np.float32)

    h = orig.shape[0]
    face_band = slice(0, int(0.22 * h), None)

    orig_face = orig[face_band]
    out_face = out[face_band]

    diff = np.mean(np.abs(orig_face - out_face), axis=2)
    mean_diff = float(np.mean(diff))

    # Score: 0 drift = 100, 25+ drift = 0
    score = max(0.0, 100.0 - (mean_diff / 25.0) * 100.0)
    return score


def _fabric_texture_score(
    result: Image.Image,
    mask_np: np.ndarray,
) -> float:
    """
    Evaluate fabric realism via local texture variance.

    Real fabric has visible texture (folds, wrinkles, weave). Overly smooth
    outputs score low. The metric computes local standard deviation in the
    inpainted region — a proxy for texture richness.

    Returns score 0..100 (higher = more realistic texture).
    """
    out = np.array(result.convert("L"), dtype=np.float32)

    if mask_np.shape[:2] != out.shape:
        mask_resized = cv2.resize(mask_np, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        mask_resized = mask_np

    inpaint_region = mask_resized > 127
    if not np.any(inpaint_region):
        return 50.0  # neutral when no inpaint region

    # Local variance via Laplacian — captures edge/texture density
    laplacian = cv2.Laplacian(out, cv2.CV_32F)
    lap_inpaint = laplacian[inpaint_region]

    # Mean absolute Laplacian — higher means more texture
    texture_metric = float(np.mean(np.abs(lap_inpaint)))

    # Thresholds: < 2.0 = too smooth, > 8.0 = good texture
    if texture_metric < 2.0:
        score = texture_metric / 2.0 * 40.0  # 0..40
    elif texture_metric < 8.0:
        score = 40.0 + (texture_metric - 2.0) / 6.0 * 40.0  # 40..80
    else:
        score = min(100.0, 80.0 + (texture_metric - 8.0) / 4.0 * 20.0)  # 80..100

    return score


def validate_output_quality(
    person: Image.Image,
    result: Image.Image,
    inpaint_mask: Image.Image,
    cloth_type: str,
    garment_ref: Image.Image | None = None,
) -> dict[str, object]:
    """
    Comprehensive post-inference quality check.

    Returns dict with:
        - geometry_score: float 0..1
        - upper_body_preservation: float 0..100 (lower_body only)
        - fabric_texture: float 0..100
        - passed: bool
        - reasons: list[str]
    """
    reasons: list[str] = []

    mask_np = np.array(inpaint_mask.convert("L"), dtype=np.uint8)
    geometry_score = _garment_geometry_score(result, mask_np, cloth_type)

    if geometry_score < 0.2:
        reasons.append(f"geometry_score_low:{geometry_score:.2f}")

    # Upper-body preservation (only relevant for lower_body)
    upper_preservation = _upper_body_preservation_score(person, result, cloth_type)
    if cloth_type == "lower_body" and upper_preservation < 60.0:
        reasons.append(f"upper_body_drift:{upper_preservation:.0f}")

    # Face identity preservation
    face_score = _face_quality_score(person, result)
    if face_score < 50.0:
        reasons.append(f"face_identity_drift:{face_score:.0f}")

    # Fabric texture realism
    fabric_texture = _fabric_texture_score(result, mask_np)
    if fabric_texture < 30.0:
        reasons.append(f"fabric_too_smooth:{fabric_texture:.0f}")

    return {
        "geometry_score": round(geometry_score, 2),
        "upper_body_preservation": round(upper_preservation, 1),
        "face_quality": round(face_score, 1),
        "fabric_texture": round(fabric_texture, 1),
        "passed": len(reasons) == 0,
        "reasons": reasons,
    }
