"""
GPU worker mask pipeline — retry, hybrid fusion, failure detection.

Runs on RunPod where SCHP, OpenPose, and DensePose are available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("idm-vton.worker.mask")


class WorkerMaskStrategy(str, Enum):
    EXTERNAL = "external"
    AUTOMASKER = "automasker"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class WorkerMaskAttempt:
    strategy: WorkerMaskStrategy
    mask: Image.Image
    score: float


@dataclass(frozen=True)
class InferenceQualityReport:
    passed: bool
    identity_drift_score: float
    floating_garment_score: float
    missing_arms_score: float
    visible_original_clothing_score: float
    color_fidelity_score: float
    color_drift_mean_rgb: float
    failure_reasons: tuple[str, ...]


def fuse_hybrid_mask(
    external: Image.Image | None,
    automasker: Image.Image,
    cloth_type: str,
) -> Image.Image:
    """
    GPU hybrid: union external rembg mask with AutoMasker semantic mask,
    then keep AutoMasker-fixed regions (head, shoes, hands).
    """
    auto_np = np.array(automasker.convert("L"), dtype=np.uint8)
    if external is None:
        return automasker

    ext_np = np.array(external.convert("L"), dtype=np.uint8)
    if ext_np.shape != auto_np.shape:
        ext_np = np.array(
            external.convert("L").resize(automasker.size, Image.NEAREST),
            dtype=np.uint8,
        )

    # Union inpaint regions — external often covers arms rembg caught
    fused = np.maximum(
        (ext_np > 127).astype(np.uint8) * 255,
        (auto_np > 127).astype(np.uint8) * 255,
    )

    # Shrink union slightly to avoid background bleed from rembg
    erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fused = cv2.erode(fused, erode_k, iterations=1)
    dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    fused = cv2.dilate(fused, dilate_k, iterations=1)

    return Image.fromarray(fused, mode="L")


def apply_protected_mask(inpaint_mask: Image.Image, protected: Image.Image | None) -> Image.Image:
    if protected is None:
        return inpaint_mask
    inp = np.array(inpaint_mask.convert("L"), dtype=np.uint8)
    prot = np.array(protected.convert("L"), dtype=np.uint8)
    if prot.shape != inp.shape:
        prot = np.array(protected.convert("L").resize(inpaint_mask.size, Image.NEAREST))
    inp[prot > 127] = 0
    return Image.fromarray(inp, mode="L")


def detect_inference_failures(
    original: Image.Image,
    result: Image.Image,
    inpaint_mask: Image.Image,
    protected: Image.Image | None = None,
    *,
    identity_threshold: float = 28.0,
    garment_ref: Image.Image | None = None,
) -> InferenceQualityReport:
    """
    Heuristic post-inference QA — triggers retry with alternate mask.

    Args:
        original: Person image BEFORE try-on (used for identity/shape checks).
        result: Try-on output image.
        inpaint_mask: Binary mask of the garment region to evaluate.
        protected: Optional mask of regions to exclude from evaluation.
        identity_threshold: Sensitivity for face-region drift detection.
        garment_ref: Source garment image (used for TRUE color fidelity
            comparison — compares garment source vs output garment color.
            If omitted, a heuristic fallback is used (less accurate).
    """
    orig = np.array(original.convert("RGB"), dtype=np.float32)
    out = np.array(result.convert("RGB"), dtype=np.float32)
    if orig.shape != out.shape:
        out_img = result.convert("RGB").resize(original.size, Image.LANCZOS)
        out = np.array(out_img, dtype=np.float32)

    mask_np = np.array(inpaint_mask.convert("L"), dtype=np.uint8)
    if mask_np.shape[:2] != orig.shape[:2]:
        mask_np = np.array(inpaint_mask.convert("L").resize(original.size, Image.NEAREST))

    reasons: list[str] = []

    # Identity drift in face band (top 22% of image)
    h = orig.shape[0]
    face_band = slice(0, int(0.22 * h), None)
    face_diff = np.mean(np.abs(orig[face_band] - out[face_band]))
    identity_drift = float(face_diff)
    if identity_drift > identity_threshold:
        reasons.append(f"identity_drift:{identity_drift:.1f}")

    # Missing arms — inpaint region at elbow height has low change (original leaked)
    elbow_y = int(0.38 * h)
    band = slice(max(0, elbow_y - 20), min(h, elbow_y + 20), None)
    inpaint_band = mask_np[band] > 127
    if np.any(inpaint_band):
        diff_band = np.mean(np.abs(orig[band] - out[band]), axis=2)
        change_ratio = float(np.mean(diff_band[inpaint_band] < 8.0))
        missing_arms = change_ratio * 100.0
        if change_ratio > 0.55:
            reasons.append(f"missing_inpaint_at_arms:{change_ratio:.2f}")
    else:
        missing_arms = 0.0

    # Floating garment — high variance at mask edge but flat interior
    edges = cv2.Canny((mask_np > 127).astype(np.uint8) * 255, 50, 150)
    edge_pixels = edges > 0
    if np.any(edge_pixels):
        edge_diff = np.mean(np.abs(orig - out)[edge_pixels])
        floating = float(edge_diff)
        if edge_diff < 6.0:
            reasons.append(f"floating_garment:{edge_diff:.1f}")
    else:
        floating = 0.0

    # Original clothing visible — inpaint area barely changed
    inpaint_region = mask_np > 127
    if np.any(inpaint_region):
        diff_inpaint = np.mean(np.abs(orig - out), axis=2)
        unchanged = float(np.mean(diff_inpaint[inpaint_region] < 10.0))
        visible_original = unchanged * 100.0
        if unchanged > 0.45:
            reasons.append(f"original_clothing_visible:{unchanged:.2f}")
    else:
        visible_original = 0.0

    # ── Color fidelity: compare GARMENT SOURCE vs OUTPUT in inpaint region ──
    # CRITICAL: We compare garment_ref (source product image) against the
    #           output garment, NOT the original person against the output.
    #           Comparing person vs output is backwards — a SUCCESSFUL try-on
    #           SHOULD have a large color change (new garment != old clothing),
    #           so it would ALWAYS register as "drift" — the exact opposite of
    #           what the metric should detect.
    color_drift_mean_rgb = 0.0
    color_fidelity_score = 100.0
    inpaint_region = mask_np > 127
    if np.any(inpaint_region) and garment_ref is not None:
        # Extract mean RGB from source garment (non-white pixels)
        garm = np.array(garment_ref.convert("RGB").resize(out.shape[1::-1], Image.LANCZOS), dtype=np.float32)
        garm_mask = (garm[:, :, 0] < 240) | (garm[:, :, 1] < 240) | (garm[:, :, 2] < 240)
        if np.any(garm_mask):
            garm_mean = np.mean(garm[garm_mask], axis=0)
        else:
            garm_mean = np.mean(garm, axis=(0, 1))

        # Extract mean RGB from output in inpaint region
        out_garm = out[inpaint_region]
        if len(out_garm) > 100:
            out_mean = np.mean(out_garm, axis=0)
            # Per-channel absolute difference between source garment and output
            drift = float(np.mean(np.abs(garm_mean - out_mean)))
            color_drift_mean_rgb = drift

            # Threshold: 50 means ~20% per-channel drift
            # Dark garments get tighter threshold (mean < 80)
            garm_luminance = float(np.mean(garm_mean))
            color_threshold = 30.0 if garm_luminance < 80.0 else 50.0
            color_fidelity_score = max(0.0, 100.0 - (drift / color_threshold) * 100.0)
            if color_fidelity_score < 50.0:
                reasons.append(f"color_drift:{drift:.1f} garm_mean={garm_luminance:.0f}")
    elif np.any(inpaint_region):
        # Fallback: no garment_ref provided — use heuristic
        inpaint_flat = orig[inpaint_region] - out[inpaint_region]
        color_drift_mean_rgb = float(np.mean(np.abs(inpaint_flat)))
        orig_mean = float(np.mean(orig[inpaint_region]))
        color_threshold = 25.0 if orig_mean < 80.0 else 35.0
        color_fidelity_score = max(0.0, 100.0 - (color_drift_mean_rgb / color_threshold) * 100.0)
        if color_fidelity_score < 50.0:
            reasons.append(f"color_drift_fallback:{color_drift_mean_rgb:.1f}")

    passed = len(reasons) == 0
    return InferenceQualityReport(
        passed=passed,
        identity_drift_score=identity_drift,
        floating_garment_score=floating,
        missing_arms_score=missing_arms,
        visible_original_clothing_score=visible_original,
        color_fidelity_score=round(color_fidelity_score, 1),
        color_drift_mean_rgb=round(color_drift_mean_rgb, 1),
        failure_reasons=tuple(reasons),
    )


def select_worker_mask_strategy(
    external_mask: Image.Image | None,
    mask_quality_score: float | None,
    min_quality: float = 62.0,
) -> WorkerMaskStrategy:
    """
    Decide initial mask strategy on GPU.

    Low-quality external masks are ignored in favour of AutoMasker.
    """
    if external_mask is None:
        return WorkerMaskStrategy.AUTOMASKER
    if mask_quality_score is not None and mask_quality_score < min_quality:
        logger.info(
            "external_mask_rejected score=%.1f min=%.1f",
            mask_quality_score,
            min_quality,
        )
        return WorkerMaskStrategy.AUTOMASKER
    return WorkerMaskStrategy.EXTERNAL
