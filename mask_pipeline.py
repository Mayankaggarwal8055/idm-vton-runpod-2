"""
GPU worker mask pipeline — retry, hybrid fusion, failure detection.

Runs on RunPod where SCHP, OpenPose, and DensePose are available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import FrozenSet, Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("idm-vton.worker.mask")


# ── SCHP Label Constants ─────────────────────────────────────────────────────

_LABEL_BG = 0
_LABEL_HAT = 1
_LABEL_HAIR = 2
_LABEL_SUNGLASSES = 3
_LABEL_UPPER_CLOTHES = 4
_LABEL_SKIRT = 5
_LABEL_PANTS = 6
_LABEL_DRESS = 7
_LABEL_BELT = 8
_LABEL_LEFT_SHOE = 9
_LABEL_RIGHT_SHOE = 10
_LABEL_FACE = 11
_LABEL_LEFT_LEG = 12
_LABEL_RIGHT_LEG = 13
_LABEL_LEFT_ARM = 14
_LABEL_RIGHT_ARM = 15
_LABEL_BAG = 16
_LABEL_SCARF = 17
_LABEL_NECK = 18

# ── Clothing label sets per cloth_type ────────────────────────────────────────

_CLOTHING_LABELS: dict[str, frozenset[int]] = {
    "upper_body": frozenset({_LABEL_UPPER_CLOTHES}),
    "lower_body": frozenset({_LABEL_PANTS, _LABEL_SKIRT}),
    "dresses": frozenset({_LABEL_UPPER_CLOTHES, _LABEL_PANTS, _LABEL_SKIRT,
                          _LABEL_DRESS, _LABEL_SCARF}),
}


# ── GarmentProfile dataclass ─────────────────────────────────────────────────

@dataclass(frozen=True)
class GarmentProfile:
    """Structured understanding of a target garment for mask building."""
    family: str  # "upper", "lower", "full"
    cloth_type: str  # "upper_body", "lower_body", "dresses"
    covers_upper: bool
    covers_lower: bool
    covers_arms: bool
    covers_hands: bool
    covers_torso_full: bool
    has_sleeves: bool
    is_fitted: bool


# ── GARMENT_PROFILES: canonical garment subtypes ─────────────────────────────

GARMENT_PROFILES: dict[str, GarmentProfile] = {
    # Upper body
    "shirt": GarmentProfile("upper", "upper_body", True, False, True, True, False, True, False),
    "tshirt": GarmentProfile("upper", "upper_body", True, False, True, True, False, True, False),
    "t_shirt": GarmentProfile("upper", "upper_body", True, False, True, True, False, True, False),
    "hoodie": GarmentProfile("upper", "upper_body", True, False, True, True, False, True, False),
    "jacket": GarmentProfile("upper", "upper_body", True, False, True, True, False, True, False),
    "kurta": GarmentProfile("upper", "upper_body", True, False, True, True, False, True, False),
    "blazer": GarmentProfile("upper", "upper_body", True, False, True, True, False, True, False),
    "sweater": GarmentProfile("upper", "upper_body", True, False, True, True, False, True, False),
    "tank_top": GarmentProfile("upper", "upper_body", True, False, False, True, False, False, False),
    # Lower body
    "jeans": GarmentProfile("lower", "lower_body", False, True, False, False, False, False, True),
    "trousers": GarmentProfile("lower", "lower_body", False, True, False, False, False, False, True),
    "pants": GarmentProfile("lower", "lower_body", False, True, False, False, False, False, True),
    "shorts": GarmentProfile("lower", "lower_body", False, True, False, False, False, False, False),
    "skirt": GarmentProfile("lower", "lower_body", False, True, False, False, False, False, True),
    "mini_skirt": GarmentProfile("lower", "lower_body", False, True, False, False, False, False, True),
    "long_skirt": GarmentProfile("lower", "lower_body", False, True, False, False, False, False, True),
    "leggings": GarmentProfile("lower", "lower_body", False, True, False, False, False, False, True),
    "joggers": GarmentProfile("lower", "lower_body", False, True, False, False, False, False, False),
    "wide_leg": GarmentProfile("lower", "lower_body", False, True, False, False, False, False, False),
    "palazzo": GarmentProfile("lower", "lower_body", False, True, False, False, False, False, False),
    "cargo_pants": GarmentProfile("lower", "lower_body", False, True, False, False, False, False, False),
    "chinos": GarmentProfile("lower", "lower_body", False, True, False, False, False, False, True),
    "bermuda": GarmentProfile("lower", "lower_body", False, True, False, False, False, False, False),
    "maxi_skirt": GarmentProfile("lower", "lower_body", False, True, False, False, False, False, False),
    "dhoti_pants": GarmentProfile("lower", "lower_body", False, True, False, False, False, False, False),
    # Dresses / full body
    "dress": GarmentProfile("full", "dresses", True, True, True, True, True, True, False),
    "gown": GarmentProfile("full", "dresses", True, True, True, True, True, True, False),
    "saree": GarmentProfile("full", "dresses", True, True, True, True, True, True, False),
    "lehenga": GarmentProfile("full", "dresses", True, True, True, True, True, True, False),
    "dupatta": GarmentProfile("full", "dresses", True, True, True, True, True, True, False),
    "anarkali": GarmentProfile("full", "dresses", True, True, True, True, True, True, False),
    "abaya": GarmentProfile("full", "dresses", True, True, True, True, True, True, False),
    "kaftan": GarmentProfile("full", "dresses", True, True, True, True, True, True, False),
    "kimono": GarmentProfile("full", "dresses", True, True, True, True, True, True, False),
    "thobe": GarmentProfile("full", "dresses", True, True, True, True, True, True, False),
    "sherwani": GarmentProfile("full", "dresses", True, True, True, True, True, True, False),
    "salwar_suit": GarmentProfile("full", "dresses", True, True, True, True, True, True, False),
    "sharara": GarmentProfile("full", "dresses", True, True, True, True, True, True, False),
    "kurti": GarmentProfile("full", "dresses", True, True, True, True, True, True, False),
    "jumpsuit": GarmentProfile("full", "dresses", True, True, True, True, True, True, False),
    "kurta_set": GarmentProfile("full", "dresses", True, True, True, True, True, True, False),
    "coord": GarmentProfile("full", "dresses", True, True, True, True, True, True, False),
    "tracksuit": GarmentProfile("full", "dresses", True, True, True, True, True, True, False),
    "overall": GarmentProfile("full", "dresses", True, True, True, True, True, True, False),
    "bodysuit": GarmentProfile("full", "dresses", True, True, True, True, True, True, True),
}


# ── GarmentGeometry: mask expansion properties ────────────────────────────────

@dataclass(frozen=True)
class GarmentGeometry:
    """Geometric expansion properties for mask building."""
    body_region: str  # "upper", "lower", "full"
    protect_upper: bool = False
    protect_lower: bool = False
    expansion_width: int = 10
    expansion_down: int = 20


GARMENT_GEOMETRY: dict[str, GarmentGeometry] = {
    # Upper body
    "shirt": GarmentGeometry("upper"),
    "tshirt": GarmentGeometry("upper"),
    "t_shirt": GarmentGeometry("upper"),
    "hoodie": GarmentGeometry("upper", expansion_width=15),
    "jacket": GarmentGeometry("upper", expansion_width=20),
    "kurta": GarmentGeometry("upper"),
    # Lower body
    "jeans": GarmentGeometry("lower", protect_upper=True),
    "trousers": GarmentGeometry("lower", protect_upper=True),
    "pants": GarmentGeometry("lower", protect_upper=True),
    "shorts": GarmentGeometry("lower", protect_upper=True, expansion_down=10),
    "skirt": GarmentGeometry("lower", protect_upper=True),
    "mini_skirt": GarmentGeometry("lower", protect_upper=True, expansion_down=10),
    "long_skirt": GarmentGeometry("lower", protect_upper=True, expansion_down=30),
    "leggings": GarmentGeometry("lower", protect_upper=True, expansion_width=5),
    "joggers": GarmentGeometry("lower", protect_upper=True, expansion_width=15),
    "wide_leg": GarmentGeometry("lower", protect_upper=True, expansion_width=30),
    "palazzo": GarmentGeometry("lower", protect_upper=True, expansion_width=60),
    "cargo_pants": GarmentGeometry("lower", protect_upper=True, expansion_width=20),
    "chinos": GarmentGeometry("lower", protect_upper=True),
    "bermuda": GarmentGeometry("lower", protect_upper=True, expansion_down=10),
    "maxi_skirt": GarmentGeometry("lower", protect_upper=True, expansion_down=120),
    "dhoti_pants": GarmentGeometry("lower", protect_upper=True, expansion_width=35, expansion_down=40),
    # Dresses
    "dress": GarmentGeometry("full"),
    "gown": GarmentGeometry("full", expansion_down=60),
    "saree": GarmentGeometry("full", expansion_width=35, expansion_down=80),
    "lehenga": GarmentGeometry("full", expansion_width=40, expansion_down=80),
    "dupatta": GarmentGeometry("full", expansion_width=30, expansion_down=40),
    "anarkali": GarmentGeometry("full", expansion_width=25, expansion_down=70),
    "abaya": GarmentGeometry("full", expansion_width=35, expansion_down=70),
    "kaftan": GarmentGeometry("full", expansion_width=45, expansion_down=60),
    "kimono": GarmentGeometry("full", expansion_width=35, expansion_down=50),
    "thobe": GarmentGeometry("full", expansion_width=25, expansion_down=60),
    "sherwani": GarmentGeometry("full", expansion_width=20, expansion_down=30),
    "salwar_suit": GarmentGeometry("full", expansion_width=25, expansion_down=55),
    "sharara": GarmentGeometry("full", expansion_width=45, expansion_down=80),
    "kurti": GarmentGeometry("full", expansion_width=20, expansion_down=45),
    "kurta_set": GarmentGeometry("full", expansion_width=20, expansion_down=45),
    "jumpsuit": GarmentGeometry("full", expansion_width=15, expansion_down=35),
    "coord": GarmentGeometry("full", expansion_width=15, expansion_down=35),
    "tracksuit": GarmentGeometry("full", expansion_width=15, expansion_down=35),
}


# ── EDITABLE_BODY_REGIONS: which SCHP labels are inpaintable ─────────────────

EDITABLE_BODY_REGIONS: dict[str, frozenset[int]] = {
    "upper_body": frozenset({_LABEL_UPPER_CLOTHES, _LABEL_LEFT_ARM, _LABEL_RIGHT_ARM}),
    "lower_body": frozenset({_LABEL_PANTS, _LABEL_SKIRT}),
    "dresses": frozenset({
        _LABEL_UPPER_CLOTHES, _LABEL_PANTS, _LABEL_SKIRT,
        _LABEL_LEFT_ARM, _LABEL_RIGHT_ARM,
        _LABEL_LEFT_LEG, _LABEL_RIGHT_LEG,
    }),
}


# ── Helpers ──────────────────────────────────────────────────────────────────

_FAMILY_LOWER: frozenset[str] = frozenset({
    "jeans", "trousers", "pants", "shorts", "skirt", "mini_skirt",
    "long_skirt", "leggings", "joggers", "wide_leg", "palazzo",
    "cargo_pants", "chinos", "bermuda", "maxi_skirt", "dhoti_pants",
})


def get_garment_family(subtype: str) -> str:
    """Classify a garment subtype into its family."""
    if subtype in _FAMILY_LOWER:
        return "lower"
    profile = GARMENT_PROFILES.get(subtype)
    if profile:
        return profile.family
    return "upper"


def build_garment_profile(
    garment_subtype: str,
    cloth_type: str,
    garment_img_info: dict | None = None,
) -> GarmentProfile:
    """Build a structured understanding of the target garment."""
    # Try exact match first
    profile = GARMENT_PROFILES.get(garment_subtype)
    if profile:
        return profile

    # Fallback: build from cloth_type
    if cloth_type == "lower_body":
        return GarmentProfile(
            family="lower", cloth_type="lower_body",
            covers_upper=False, covers_lower=True, covers_arms=False,
            covers_hands=False, covers_torso_full=False,
            has_sleeves=False, is_fitted=True,
        )
    if cloth_type == "dresses":
        return GarmentProfile(
            family="full", cloth_type="dresses",
            covers_upper=True, covers_lower=True, covers_arms=True,
            covers_hands=True, covers_torso_full=True,
            has_sleeves=True, is_fitted=False,
        )
    return GarmentProfile(
        family="upper", cloth_type="upper_body",
        covers_upper=True, covers_lower=False, covers_arms=True,
        covers_hands=True, covers_torso_full=False,
        has_sleeves=True, is_fitted=False,
    )


def get_garment_geometry(subtype: str) -> GarmentGeometry:
    """Look up geometric expansion properties for mask building."""
    geo = GARMENT_GEOMETRY.get(subtype)
    if geo:
        return geo
    # Fallback based on family
    family = get_garment_family(subtype)
    if family == "lower":
        return GarmentGeometry("lower", protect_upper=True)
    if family == "full":
        return GarmentGeometry("full")
    return GarmentGeometry("upper")


def get_profile_editable_labels(profile: GarmentProfile) -> frozenset[int]:
    """Get the set of SCHP labels that are inpaintable for this profile."""
    # Full-body garments (dresses, jumpsuits, co-ord sets): everything
    # clothing-related is editable — upper clothes, lower clothes, arms, legs
    if profile.covers_upper and profile.covers_lower:
        return frozenset({
            _LABEL_UPPER_CLOTHES, _LABEL_PANTS, _LABEL_SKIRT,
            _LABEL_LEFT_ARM, _LABEL_RIGHT_ARM,
            _LABEL_LEFT_LEG, _LABEL_RIGHT_LEG,
        })
    if profile.covers_lower:
        return frozenset({_LABEL_PANTS, _LABEL_SKIRT})
    if profile.covers_upper:
        return frozenset({_LABEL_UPPER_CLOTHES, _LABEL_LEFT_ARM, _LABEL_RIGHT_ARM})
    return frozenset()


def detect_source_cloth_type(schp_np: np.ndarray) -> str:
    """Detect what the person is currently wearing from SCHP pixel ratios."""
    h, w = schp_np.shape[:2]
    total_px = h * w

    garment_labels = {
        _LABEL_UPPER_CLOTHES, _LABEL_PANTS, _LABEL_SKIRT,
        _LABEL_LEFT_ARM, _LABEL_RIGHT_ARM,
        _LABEL_LEFT_LEG, _LABEL_RIGHT_LEG,
    }
    garment_px = int(np.sum(np.isin(schp_np, list(garment_labels))))
    if garment_px == 0:
        return "unknown"

    garment_ratio = garment_px / total_px

    upper_px = int(np.sum(schp_np == _LABEL_UPPER_CLOTHES))
    pants_px = int(np.sum(schp_np == _LABEL_PANTS))
    skirt_px = int(np.sum(schp_np == _LABEL_SKIRT))
    lower_px = pants_px + skirt_px

    # Detection priority: Lower body if pants/skirt dominate
    if lower_px / garment_ratio > 0.40:
        return "lower_body"

    # Pre-upper catch: pants/skirt >= upper
    if lower_px >= upper_px:
        return "lower_body"

    if upper_px > 0:
        return "upper_body"

    return "unknown"


def build_schp_inpaint_mask(
    schp_labels: np.ndarray,
    cloth_type: str,
    garment_subtype: str,
    source_cloth_type: str,
    profile: GarmentProfile,
) -> np.ndarray:
    """Build binary inpaint mask from SCHP labels + GarmentProfile."""
    target_labels = EDITABLE_BODY_REGIONS.get(cloth_type, set())

    same_category = (
        (source_cloth_type == cloth_type)
        or (source_cloth_type == "unknown")
    )

    if same_category:
        source_labels = _CLOTHING_LABELS.get(source_cloth_type, set())
        mask = np.isin(schp_labels, list(source_labels)).astype(np.uint8) * 255
        mask = np.maximum(
            mask,
            np.isin(schp_labels, list(target_labels)).astype(np.uint8) * 255,
        )
    else:
        # Cross-category: target body region + source garment for erasure
        mask = np.isin(schp_labels, list(target_labels)).astype(np.uint8) * 255
        source_labels = _CLOTHING_LABELS.get(source_cloth_type, set())
        if source_labels:
            mask = np.maximum(
                mask,
                np.isin(schp_labels, list(source_labels)).astype(np.uint8) * 255,
            )

    return mask


def build_schp_protect_mask(
    schp_labels: np.ndarray,
    cloth_type: str,
    profile: GarmentProfile,
) -> np.ndarray:
    """Build protect mask — regions that must NOT be inpainted."""
    editable = get_profile_editable_labels(profile)

    # Identity labels that are ALWAYS protected (face, head, background)
    identity_labels = {_LABEL_BG, _LABEL_HAT, _LABEL_HAIR, _LABEL_SUNGLASSES,
                       _LABEL_FACE, _LABEL_NECK}

    # For full-body garments, socks/shoes/belt/scarf are part of the outfit
    # and should NOT be identity-protected. For upper/lower-only, protect them.
    if not (profile.covers_upper and profile.covers_lower):
        identity_labels |= {_LABEL_BELT, _LABEL_LEFT_SHOE, _LABEL_RIGHT_SHOE,
                            _LABEL_BAG, _LABEL_SCARF}

    protect_labels = identity_labels | editable
    # Protect everything NOT in the editable set
    all_labels = set(range(20))
    non_editable = all_labels - editable - identity_labels
    protect_mask = np.isin(schp_labels, list(non_editable)).astype(np.uint8) * 255
    return protect_mask


def build_final_inpaint_mask(
    schp_labels: np.ndarray,
    cloth_type: str,
    garment_subtype: str,
    source_cloth_type: str,
    garment_img_info: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build final mask triplet: (final_mask, inpaint_mask, protect_mask).

    Returns:
        final_mask_np: Combined mask for inference (uint8, 768x1024)
        inpaint_mask_np: Regions to inpaint (uint8, 768x1024)
        protect_mask_np: Regions to protect (uint8, 768x1024)
    """
    profile = build_garment_profile(garment_subtype, cloth_type, garment_img_info)

    inpaint_mask = build_schp_inpaint_mask(
        schp_labels, cloth_type, garment_subtype, source_cloth_type, profile,
    )

    protect_mask = build_schp_protect_mask(schp_labels, cloth_type, profile)

    # Apply protect: zero out protected regions from inpaint
    final_mask = inpaint_mask.copy()
    final_mask[protect_mask > 127] = 0

    # Boundary smoothing: cloth-type-specific dilation kernel
    if cloth_type in ("dresses", "full_body"):
        # Dresses/full-body: larger vertical kernel to cover full garment
        # boundary from neckline to hem. Two iterations for sufficient
        # coverage of flowing silhouettes (gowns, kurtis, jumpsuits).
        dress_kw = max(3, int(19 * 1.0))
        dress_kh = max(5, int(29 * 1.0))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dress_kw, dress_kh))
        inpaint_dilated = cv2.dilate(final_mask, kernel, iterations=2)
    elif cloth_type == "lower_body":
        # Lower body: moderate vertical kernel — covers leg edges without
        # bleeding into the torso. Single iteration keeps the mask tight.
        leg_kw = max(3, int(4 * 2.0))
        leg_kh = max(5, int(6 * 2.0))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (leg_kw, leg_kh))
        inpaint_dilated = cv2.dilate(final_mask, kernel, iterations=1)

        # Hard upper-body exclusion: nothing above the waist should be
        # inpainted. Use the top row of inpaint pixels as the boundary.
        rows_with_mask = np.where(inpaint_dilated.any(axis=1))[0]
        if len(rows_with_mask) > 0:
            mask_top = int(rows_with_mask[0])
            # 20px margin preserves the waistband region for natural fit.
            exclude_top = max(0, mask_top - 20)
            inpaint_dilated[:exclude_top, :] = 0
    else:
        # upper_body: wider horizontal kernel
        up_kw = max(7, int(10 * 2.5))
        up_kh = max(5, int(6 * 2.5))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (up_kw, up_kh))
        inpaint_dilated = cv2.dilate(final_mask, kernel, iterations=1)

    return inpaint_dilated, inpaint_mask, protect_mask


def validate_mask_coverage(
    mask: Image.Image,
    cloth_type: str,
) -> dict[str, object]:
    """Pre-inference mask sanity check."""
    binary = np.array(mask.convert("L"), dtype=np.uint8)
    h, w = binary.shape[:2]
    total = h * w
    if total == 0:
        return {"valid": False, "reason": "empty_mask"}

    coverage = float(np.sum(binary > 127)) / total

    if cloth_type in ("lower_body", "dresses", "full_body"):
        lower_zone = binary[h * 3 // 5:, :]
        lower_coverage = float(np.sum(lower_zone > 127)) / lower_zone.size
        if lower_coverage < 0.03:
            return {
                "valid": False,
                "coverage_percent": round(coverage * 100, 1),
                "reason": f"lower_body_too_sparse:{lower_coverage*100:.1f}%",
            }

    return {"valid": True, "coverage_percent": round(coverage * 100, 1)}


# ── Existing pipeline code ───────────────────────────────────────────────────


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
    cloth_type: str = "upper_body",
) -> WorkerMaskStrategy:
    """
    Decide initial mask strategy on GPU.

    Low-quality external masks are ignored in favour of AutoMasker.
    """
    if external_mask is None:
        return WorkerMaskStrategy.AUTOMASKER
    mask_np = np.array(external_mask.convert("L"), dtype=np.uint8)
    coverage = float(np.mean(mask_np > 127)) * 100.0
    coverage_upper_bounds = {
        "upper_body": 55.0,
        "lower_body": 65.0,
        "dresses": 75.0,
        "full_body": 75.0,
    }
    upper_bound = coverage_upper_bounds.get(cloth_type, 60.0)
    if coverage <= 1.0 or coverage > upper_bound:
        logger.info(
            "external_mask_rejected coverage=%.1f cloth_type=%s upper_bound=%.1f",
            coverage,
            cloth_type,
            upper_bound,
        )
        return WorkerMaskStrategy.AUTOMASKER
    if mask_quality_score is None and cloth_type in ("lower_body", "dresses", "full_body"):
        logger.info(
            "external_mask_rejected missing_quality_score cloth_type=%s coverage=%.1f",
            cloth_type,
            coverage,
        )
        return WorkerMaskStrategy.AUTOMASKER
    if mask_quality_score is not None and mask_quality_score < min_quality:
        logger.info(
            "external_mask_rejected score=%.1f min=%.1f",
            mask_quality_score,
            min_quality,
        )
        return WorkerMaskStrategy.AUTOMASKER
    return WorkerMaskStrategy.EXTERNAL
