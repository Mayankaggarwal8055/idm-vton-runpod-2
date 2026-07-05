from __future__ import annotations

import io
import os
import sys
import time
import logging
import random
import base64
import threading
import traceback
from pathlib import Path
from typing import Any

import runpod
import requests
import numpy as np
import torch
from PIL import Image
import cloudinary
import cloudinary.uploader
from requests.adapters import HTTPAdapter


# =============================================================================
# Logging
# =============================================================================

logger = logging.getLogger("idm-vton.worker")
_handler_configured = False


def _ensure_logging():
    global _handler_configured
    if not _handler_configured:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        _handler_configured = True


# =============================================================================
# Env / Constants
# =============================================================================

TARGET_SIZE = (768, 1024)
TARGET_W, TARGET_H = TARGET_SIZE

IDM_VTON_DIR = os.environ.get("IDM_VTON_DIR", "/workspace/IDM-VTON")
IDM_VTON_MODEL = os.environ.get("IDM_VTON_MODEL", "/workspace/models/idm-vton")
DENSEPOSE_WEIGHTS = os.environ.get(
    "DENSEPOSE_WEIGHTS",
    "/workspace/models/densepose/model_final_162be9.pkl",
)

CLOUDINARY_FOLDER = os.environ.get("CLOUDINARY_FOLDER", "trylix/tryon/results")

DENOISE_STEPS = int(os.environ.get("IDM_VTON_STEPS", "30"))
GUIDANCE_SCALE = float(os.environ.get("IDM_VTON_GUIDANCE", "2.0"))

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16

# Memory/perf knobs
ENABLE_XFORMERS = os.environ.get("ENABLE_XFORMERS", "1") == "1"
ENABLE_TORCH_COMPILE = os.environ.get("ENABLE_TORCH_COMPILE", "0") == "1"
ENABLE_MODEL_CPU_OFFLOAD = os.environ.get("ENABLE_MODEL_CPU_OFFLOAD", "0") == "1"
ALLOW_TF32 = os.environ.get("ALLOW_TF32", "1") == "1"

# =============================================================================
# Global state
# =============================================================================

pipe = None
parsing_model = None
openpose_model = None
densepose_predictor = None
densepose_cfg = None
tensor_transform = None
get_mask_location_fn = None

_WARM = threading.Event()
_STARTUP_TIME = time.perf_counter()
_REUSE_COUNT: int = 0

_SESSION: requests.Session | None = None
_SESSION_LOCK = threading.Lock()


# =============================================================================
# Helpers
# =============================================================================

def _require_path(path: str | Path, label: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing {label}: {p}")
    return p


def _ensure_dir_layout():
    _require_path(IDM_VTON_DIR, "IDM_VTON_DIR")

    needed = [
        Path(IDM_VTON_MODEL) / "unet",
        Path(IDM_VTON_MODEL) / "vae",
        Path(IDM_VTON_MODEL) / "scheduler",
        Path(IDM_VTON_MODEL) / "text_encoder",
        Path(IDM_VTON_MODEL) / "text_encoder_2",
        Path(IDM_VTON_MODEL) / "image_encoder",
        Path(IDM_VTON_MODEL) / "tokenizer",
        Path(IDM_VTON_MODEL) / "tokenizer_2",
        Path(IDM_VTON_MODEL) / "unet_encoder",
        Path(IDM_VTON_DIR) / "configs" / "densepose_rcnn_R_50_FPN_s1x.yaml",
        Path(DENSEPOSE_WEIGHTS),
    ]
    for p in needed:
        _require_path(p, f"required path {p}")

    parsing_paths = [
        Path(IDM_VTON_DIR) / "ckpt" / "humanparsing" / "parsing_atr.onnx",
        Path(IDM_VTON_DIR) / "ckpt" / "humanparsing" / "parsing_lip.onnx",
        Path(IDM_VTON_DIR) / "ckpt" / "openpose" / "body_pose_model.pth",
        Path(IDM_VTON_DIR) / "ckpt" / "image_encoder",
        Path(IDM_VTON_DIR) / "ckpt" / "ip_adapter",
    ]
    for p in parsing_paths:
        _require_path(p, f"required path {p}")


def _get_session() -> requests.Session:
    global _SESSION
    if _SESSION is not None:
        return _SESSION
    with _SESSION_LOCK:
        if _SESSION is not None:
            return _SESSION
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": "TryLix-Worker/1.0",
                "Accept": "image/webp,image/jpeg,image/png,*/*",
            }
        )
        adapter = HTTPAdapter(pool_connections=8, pool_maxsize=16, max_retries=2)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        _SESSION = session
        logger.info("http_session_created pool_maxsize=16")
        return session


def _configure_cloudinary() -> bool:
    cloud_name = os.environ.get("CLOUDINARY_CLOUD_NAME")
    api_key = os.environ.get("CLOUDINARY_API_KEY")
    api_secret = os.environ.get("CLOUDINARY_API_SECRET")
    if not all([cloud_name, api_key, api_secret]):
        logger.warning("Cloudinary not configured - cannot upload results")
        return False
    cloudinary.config(
        cloud_name=cloud_name,
        api_key=api_key,
        api_secret=api_secret,
        secure=True,
    )
    return True


def _upload_to_cloudinary(image: Image.Image, job_id: str) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    buffer.seek(0)

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            result = cloudinary.uploader.upload(
                buffer,
                folder=CLOUDINARY_FOLDER,
                public_id=f"result_{job_id}",
                resource_type="image",
                overwrite=True,  # Must be True so retried jobs upload fresh results
            )
            url = str(result["secure_url"])
            logger.info("cloudinary_upload_complete result_url=%s", url)
            return url
        except Exception as exc:
            last_error = exc
            logger.warning("cloudinary_upload_failed attempt=%s error=%s", attempt + 1, exc)
            if attempt < 2:
                buffer.seek(0)
                time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"Cloudinary upload failed after 3 attempts: {last_error}")


def download_image(url: str, timeout: int = 60) -> Image.Image:
    session = _get_session()
    resp = session.get(url, timeout=timeout, stream=True)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def _is_url_reference(value: str) -> bool:
    normalized = value.strip().lower()
    return normalized.startswith("http://") or normalized.startswith("https://")


def _decode_base64_image(value: str) -> Image.Image:
    payload = value.strip()
    if payload.startswith("data:"):
        _, payload = payload.split(",", 1)

    payload = "".join(payload.split())
    padding = (-len(payload)) % 4
    if padding:
        payload += "=" * padding

    raw = base64.b64decode(payload)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def load_image_reference(value: str, timeout: int = 60) -> Image.Image:
    """Load an image from either an http(s) URL or a base64/data URL payload."""
    if _is_url_reference(value):
        return download_image(value, timeout=timeout)
    return _decode_base64_image(value)


def _set_torch_perf_flags():
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = ALLOW_TF32
            torch.backends.cudnn.allow_tf32 = ALLOW_TF32
        except Exception:
            pass
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


# =============================================================================
# Model loading
# =============================================================================

def load_models():
    global pipe, parsing_model, openpose_model
    global densepose_predictor, densepose_cfg, tensor_transform, get_mask_location_fn

    if pipe is not None:
        logger.info("Models already loaded — skipping")
        return

    logger.info("=" * 60)
    logger.info("MODEL LOADING BEGIN")
    logger.info("=" * 60)

    _ensure_dir_layout()
    _set_torch_perf_flags()

    load_start = time.perf_counter()

    logger.info("torch_version=%s", torch.__version__)
    logger.info("cuda_available=%s", torch.cuda.is_available())
    logger.info("device=%s", DEVICE)

    if torch.cuda.is_available():
        logger.info("cuda_version=%s", torch.version.cuda)
        logger.info("gpu_name=%s", torch.cuda.get_device_name(0))

        try:
            torch.cuda.empty_cache()
            logger.info("cuda_cache_cleared=True")
        except Exception as exc:
            logger.warning("cuda_cache_clear_failed error=%s", exc)

    if IDM_VTON_DIR not in sys.path:
        sys.path.insert(0, IDM_VTON_DIR)

    gradio_demo_dir = os.path.join(IDM_VTON_DIR, "gradio_demo")

    if gradio_demo_dir not in sys.path:
        sys.path.insert(0, gradio_demo_dir)

    logger.info("python_paths_configured=True")

    from torchvision import transforms

    tensor_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    logger.info("Importing custom IDM-VTON modules...")

    from src.unet_hacked_garmnet import (
        UNet2DConditionModel as UNet2DConditionModel_ref
    )

    from src.unet_hacked_tryon import (
        UNet2DConditionModel as UNet2DConditionModel_tryon
    )

    from src.tryon_pipeline import (
        StableDiffusionXLInpaintPipeline as TryonPipeline
    )

    logger.info("Custom modules imported")

    from transformers import (
        CLIPImageProcessor,
        CLIPVisionModelWithProjection,
        CLIPTextModel,
        CLIPTextModelWithProjection,
        AutoTokenizer,
    )

    from diffusers import (
        DDPMScheduler,
        AutoencoderKL,
    )

    logger.info("Loading IDM-VTON model from %s", IDM_VTON_MODEL)

    logger.info("Loading UNet...")
    unet = UNet2DConditionModel_tryon.from_pretrained(
        IDM_VTON_MODEL,
        subfolder="unet",
        torch_dtype=TORCH_DTYPE,
    ).requires_grad_(False)

    logger.info("Loading tokenizer_one...")
    tokenizer_one = AutoTokenizer.from_pretrained(
        IDM_VTON_MODEL,
        subfolder="tokenizer",
        use_fast=False,
    )

    logger.info("Loading tokenizer_two...")
    tokenizer_two = AutoTokenizer.from_pretrained(
        IDM_VTON_MODEL,
        subfolder="tokenizer_2",
        use_fast=False,
    )

    logger.info("Loading scheduler...")
    noise_scheduler = DDPMScheduler.from_pretrained(
        IDM_VTON_MODEL,
        subfolder="scheduler",
    )

    logger.info("Loading text_encoder_one...")
    text_encoder_one = CLIPTextModel.from_pretrained(
        IDM_VTON_MODEL,
        subfolder="text_encoder",
        torch_dtype=TORCH_DTYPE,
    ).requires_grad_(False)

    logger.info("Loading text_encoder_two...")
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        IDM_VTON_MODEL,
        subfolder="text_encoder_2",
        torch_dtype=TORCH_DTYPE,
    ).requires_grad_(False)

    logger.info("Loading image_encoder...")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        IDM_VTON_MODEL,
        subfolder="image_encoder",
        torch_dtype=TORCH_DTYPE,
    ).requires_grad_(False)

    logger.info("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        IDM_VTON_MODEL,
        subfolder="vae",
        torch_dtype=TORCH_DTYPE,
    ).requires_grad_(False)

    logger.info("Loading UNet encoder...")
    unet_encoder = UNet2DConditionModel_ref.from_pretrained(
        IDM_VTON_MODEL,
        subfolder="unet_encoder",
        torch_dtype=TORCH_DTYPE,
    ).requires_grad_(False)

    logger.info("Building SDXL tryon pipeline...")

    pipe = TryonPipeline.from_pretrained(
        IDM_VTON_MODEL,
        unet=unet,
        vae=vae,
        feature_extractor=CLIPImageProcessor(),
        text_encoder=text_encoder_one,
        text_encoder_2=text_encoder_two,
        tokenizer=tokenizer_one,
        tokenizer_2=tokenizer_two,
        scheduler=noise_scheduler,
        image_encoder=image_encoder,
        torch_dtype=TORCH_DTYPE,
    )

    logger.info("Assigning UNet encoder...")
    pipe.unet_encoder = unet_encoder

    logger.info("Moving pipeline to device=%s", DEVICE)
    pipe = pipe.to(DEVICE)

    if ENABLE_XFORMERS:

        logger.info("Attempting xformers enable...")

        try:
            pipe.enable_xformers_memory_efficient_attention()
            logger.info("xformers_enabled=True")

        except Exception as exc:
            logger.warning(
                "xformers_enable_failed error=%s",
                exc,
            )

    if ENABLE_MODEL_CPU_OFFLOAD:

        logger.info("Attempting model CPU offload...")

        try:
            pipe.enable_model_cpu_offload()
            logger.info("model_cpu_offload_enabled=True")

        except Exception as exc:
            logger.warning(
                "cpu_offload_enable_failed error=%s",
                exc,
            )

    if ENABLE_TORCH_COMPILE and hasattr(torch, "compile"):

        logger.info("Attempting torch.compile...")

        try:
            pipe.unet = torch.compile(
                pipe.unet,
                mode="reduce-overhead",
            )

            logger.info("torch_compile_enabled=True")

        except Exception as exc:
            logger.warning(
                "torch_compile_failed error=%s",
                exc,
            )

    logger.info("Pipeline fully initialized")

    logger.info("Loading Parsing model...")
    from preprocess.humanparsing.run_parsing import Parsing
    parsing_model = Parsing(0)

    logger.info("Loading OpenPose model...")
    from preprocess.openpose.run_openpose import OpenPose
    openpose_model = OpenPose(0)

    logger.info("Parsing + OpenPose ready")

    logger.info("Loading DensePose config...")

    from detectron2.config import get_cfg
    from densepose import add_densepose_config
    from detectron2.engine.defaults import DefaultPredictor

    densepose_cfg = get_cfg()

    add_densepose_config(densepose_cfg)

    config_path = os.path.join(
        IDM_VTON_DIR,
        "configs",
        "densepose_rcnn_R_50_FPN_s1x.yaml",
    )

    logger.info("DensePose config path=%s", config_path)

    densepose_cfg.merge_from_file(config_path)

    densepose_cfg.MODEL.WEIGHTS = DENSEPOSE_WEIGHTS

    logger.info("DensePose weights=%s", DENSEPOSE_WEIGHTS)

    densepose_cfg.MODEL.DEVICE = DEVICE

    densepose_cfg.freeze()

    logger.info("Creating DensePose predictor...")

    densepose_predictor = DefaultPredictor(densepose_cfg)

    logger.info("DensePose predictor ready")

    logger.info("Loading mask utility...")

    from utils_mask import get_mask_location as _get_mask_location

    get_mask_location_fn = _get_mask_location

    load_ms = (time.perf_counter() - load_start) * 1000

    logger.info("=" * 60)
    logger.info("MODELS READY")
    logger.info("model_load_ms=%.0f", load_ms)
    logger.info("=" * 60)

# =============================================================================
# Warmup
# =============================================================================

def warmup():
    global _REUSE_COUNT
    if _WARM.is_set():
        return

    logger.info("=" * 60)
    logger.info("COLD START BEGIN")
    logger.info("=" * 60)

    load_models()

    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        logger.info("gpu_warmup_ready=True")
    except Exception as exc:
        logger.warning("gpu_warmup_skipped error=%s", exc)

    cloudinary_ok = _configure_cloudinary()

    startup_total_ms = (time.perf_counter() - _STARTUP_TIME) * 1000
    logger.info("=" * 60)
    logger.info("COLD START COMPLETE")
    logger.info("  startup_total_ms=%.0f", startup_total_ms)
    logger.info("  cloudinary_configured=%s", cloudinary_ok)
    logger.info("=" * 60)

    _WARM.set()
    _REUSE_COUNT = 0


# =============================================================================
# Inference
# =============================================================================

def _maybe_autocast():
    if torch.cuda.is_available():
        return torch.cuda.amp.autocast(dtype=TORCH_DTYPE)
    class _NullCtx:
        def __enter__(self): return None
        def __exit__(self, exc_type, exc, tb): return False
    return _NullCtx()


# =============================================================================
# Subtype-aware prompt attributes (lower-body only)
# =============================================================================

_GARMENT_PROMPT_ATTRS: dict[str, dict[str, str]] = {
    "jeans": {
        "coverage": "lower body garment",
        "fit": "regular fit",
        "silhouette": "straight leg",
        "sleeves": "n/a",
        "neckline": "n/a",
        "collar": "n/a",
        "waist_position": "natural waist or hip",
        "garment_length": "full length to ankle",
        "layering": "single layer",
        "structure": "denim two legs button fly",
        "drape": "minimal drape",
        "material": "denim cotton twill",
        "fabric_behavior": "stiff structured",
    },
    "trousers": {
        "coverage": "lower body garment",
        "fit": "regular fit",
        "silhouette": "straight leg",
        "sleeves": "n/a",
        "neckline": "n/a",
        "collar": "n/a",
        "waist_position": "natural waist",
        "garment_length": "full length to ankle",
        "layering": "single layer",
        "structure": "two legs zip fly",
        "drape": "moderate drape",
        "material": "woven cotton polyester",
        "fabric_behavior": "smooth structured",
    },
    "pants": {
        "coverage": "lower body garment",
        "fit": "regular fit",
        "silhouette": "straight leg",
        "sleeves": "n/a",
        "neckline": "n/a",
        "collar": "n/a",
        "waist_position": "natural waist or hip",
        "garment_length": "full length to ankle",
        "layering": "single layer",
        "structure": "two legs",
        "drape": "moderate drape",
        "material": "woven cotton",
        "fabric_behavior": "smooth structured",
    },
    "shorts": {
        "coverage": "lower body garment",
        "fit": "regular fit",
        "silhouette": "straight leg",
        "sleeves": "n/a",
        "neckline": "n/a",
        "collar": "n/a",
        "waist_position": "natural waist or hip",
        "garment_length": "above knee",
        "layering": "single layer",
        "structure": "two legs",
        "drape": "minimal drape",
        "material": "cotton twill",
        "fabric_behavior": "casual structured",
    },
    "skirt": {
        "coverage": "lower body garment",
        "fit": "regular fit",
        "silhouette": "A-line",
        "sleeves": "n/a",
        "neckline": "n/a",
        "collar": "n/a",
        "waist_position": "natural waist",
        "garment_length": "varies",
        "layering": "single layer",
        "structure": "no leg separation",
        "drape": "flowing drape",
        "material": "woven cotton",
        "fabric_behavior": "soft flowing",
    },
    "joggers": {
        "coverage": "lower body garment",
        "fit": "relaxed fit",
        "silhouette": "tapered leg",
        "sleeves": "n/a",
        "neckline": "n/a",
        "collar": "n/a",
        "waist_position": "elastic waist",
        "garment_length": "full length to ankle",
        "layering": "single layer",
        "structure": "two legs elastic cuff",
        "drape": "soft drape",
        "material": "fleece cotton jersey",
        "fabric_behavior": "soft relaxed",
    },
    "leggings": {
        "coverage": "lower body garment",
        "fit": "tight fitted",
        "silhouette": "body-hugging",
        "sleeves": "n/a",
        "neckline": "n/a",
        "collar": "n/a",
        "waist_position": "high waist",
        "garment_length": "full length to ankle",
        "layering": "single layer",
        "structure": "two legs no fly",
        "drape": "no drape skin-tight",
        "material": "stretch jersey",
        "fabric_behavior": "stretch conforming",
    },
    "cargo_pants": {
        "coverage": "lower body garment",
        "fit": "relaxed fit",
        "silhouette": "straight leg",
        "sleeves": "n/a",
        "neckline": "n/a",
        "collar": "n/a",
        "waist_position": "natural waist or hip",
        "garment_length": "full length to ankle",
        "layering": "single layer",
        "structure": "pocketed utility",
        "drape": "moderate drape",
        "material": "cotton twill",
        "fabric_behavior": "rugged structured",
    },
    "chinos": {
        "coverage": "lower body garment",
        "fit": "regular fit",
        "silhouette": "straight leg",
        "sleeves": "n/a",
        "neckline": "n/a",
        "collar": "n/a",
        "waist_position": "natural waist",
        "garment_length": "full length to ankle",
        "layering": "single layer",
        "structure": "two legs zip fly",
        "drape": "moderate drape",
        "material": "cotton chino",
        "fabric_behavior": "smooth structured",
    },
    "wide_leg": {
        "coverage": "lower body garment",
        "fit": "relaxed fit",
        "silhouette": "wide leg",
        "sleeves": "n/a",
        "neckline": "n/a",
        "collar": "n/a",
        "waist_position": "natural waist",
        "garment_length": "full length to ankle",
        "layering": "single layer",
        "structure": "two legs wide",
        "drape": "flowing drape",
        "material": "woven cotton",
        "fabric_behavior": "flowing structured",
    },
    "palazzo": {
        "coverage": "lower body garment",
        "fit": "relaxed fit",
        "silhouette": "extremely wide leg",
        "sleeves": "n/a",
        "neckline": "n/a",
        "collar": "n/a",
        "waist_position": "natural waist",
        "garment_length": "full length to ankle",
        "layering": "single layer",
        "structure": "two legs very wide",
        "drape": "heavy flowing drape",
        "material": "flowing woven",
        "fabric_behavior": "flowing soft",
    },
    "bermuda": {
        "coverage": "lower body garment",
        "fit": "regular fit",
        "silhouette": "straight leg",
        "sleeves": "n/a",
        "neckline": "n/a",
        "collar": "n/a",
        "waist_position": "natural waist or hip",
        "garment_length": "above knee to knee",
        "layering": "single layer",
        "structure": "two legs",
        "drape": "minimal drape",
        "material": "cotton twill",
        "fabric_behavior": "casual structured",
    },
    # ── Dress / full-body garments ──────────────────────────────────
    "dress": {
        "coverage": "full body garment",
        "fit": "regular fit",
        "silhouette": "fitted bodice flared skirt",
        "sleeves": "varies",
        "neckline": "varies",
        "collar": "n/a",
        "waist_position": "natural waist",
        "garment_length": "knee length or longer",
        "layering": "single layer",
        "structure": "one-piece bodice and skirt",
        "drape": "moderate drape",
        "material": "woven cotton polyester",
        "fabric_behavior": "structured flowing",
    },
    "gown": {
        "coverage": "full body garment",
        "fit": "regular fit",
        "silhouette": "floor length elegant",
        "sleeves": "varies",
        "neckline": "varies",
        "collar": "n/a",
        "waist_position": "natural waist",
        "garment_length": "floor length",
        "layering": "single layer",
        "structure": "one-piece full length",
        "drape": "heavy flowing drape",
        "material": "silk satin chiffon",
        "fabric_behavior": "flowing elegant",
    },
    "jumpsuit": {
        "coverage": "full body garment",
        "fit": "regular fit",
        "silhouette": "continuous torso to legs",
        "sleeves": "varies",
        "neckline": "varies",
        "collar": "n/a",
        "waist_position": "natural waist",
        "garment_length": "full length to ankle",
        "layering": "single layer",
        "structure": "one-piece with leg separation",
        "drape": "moderate drape",
        "material": "woven cotton polyester",
        "fabric_behavior": "structured fitted",
    },
    "kurti": {
        "coverage": "full body garment",
        "fit": "regular fit",
        "silhouette": "tunic over pants",
        "sleeves": "varies",
        "neckline": "round or v-neck",
        "collar": "n/a",
        "waist_position": "natural waist",
        "garment_length": "hip to knee length",
        "layering": "layered with bottoms",
        "structure": "tunic top with separate bottoms",
        "drape": "moderate drape",
        "material": "cotton silk",
        "fabric_behavior": "soft flowing",
    },
    "coord": {
        "coverage": "full body outfit",
        "fit": "regular fit",
        "silhouette": "matching top and bottom",
        "sleeves": "varies",
        "neckline": "varies",
        "collar": "varies",
        "waist_position": "natural waist",
        "garment_length": "varies by set",
        "layering": "coordinated set",
        "structure": "matching top and bottom pieces",
        "drape": "moderate drape",
        "material": "matching fabric set",
        "fabric_behavior": "coordinated structured",
    },
    "overall": {
        "coverage": "full body garment",
        "fit": "relaxed fit",
        "silhouette": "one-piece bib front",
        "sleeves": "sleeveless or with shirt",
        "neckline": "bib front",
        "collar": "n/a",
        "waist_position": "natural waist",
        "garment_length": "full length to ankle",
        "layering": "over shirt",
        "structure": "bib and brace with legs",
        "drape": "minimal drape",
        "material": "denim cotton twill",
        "fabric_behavior": "rugged structured",
    },
}


def _build_subtype_aware_prompt(garment_desc: str, garment_subtype: str = "") -> str:
    """
    Build a detailed prompt enriched with subtype-specific garment attributes.

    For lower-body garments, appends fabric, fit, silhouette, and structure
    details that guide the diffusion model toward realistic generation.

    For upper-body or unknown subtypes, returns the basic prompt unchanged.
    """
    key = (garment_subtype or "").strip().lower().replace(" ", "_").replace("-", "_")
    attrs = _GARMENT_PROMPT_ATTRS.get(key)
    if not attrs:
        return "model is wearing " + garment_desc

    parts = ["model wearing " + garment_desc]
    for attr_key in (
        "coverage", "fit", "silhouette", "sleeves", "neckline",
        "collar", "waist_position", "garment_length", "layering",
        "structure", "drape", "material", "fabric_behavior",
    ):
        if attr_key in attrs:
            parts.append(attrs[attr_key])
    return ", ".join(parts)


def _build_source_specific_negative(source_cloth_type: str = "", target_subtype: str = "") -> str:
    """
    Build a negative prompt that suppresses accessories, artifacts, and
    unrealistic rendering styles.

    Does NOT include task-specific terms (e.g., no "shirt, top" for lower-body)
    — those are handled by the positive prompt's preservation instructions.
    """
    return (
        "monochrome, lowres, bad anatomy, worst quality, low quality, "
        "deformed, distorted, disfigured, bad proportions, "
        "extra limbs, missing limbs, cloned head, body out of frame, "
        "poorly drawn face, mutation, mutated, extra fingers, "
        "ugly, blurry, watermark, signature, text, logo, "
        "smooth plastic, airbrushed, cg render, 3d render, "
        "flat lighting, "
        "bag, purse, handbag, clutch, tote, backpack, "
        "headphones, earphones, headset, "
        "necklace, chain, pendant, choker, "
        "watch, wristwatch, bracelet, "
        "sunglasses, eyewear, glasses, "
        "phone, smartphone, mobile, "
        "strap, belt, waist belt, "
        "accessory, accessories, "
        "extra object, held item, carrying"
    )


def _restore_person_identity(
    result: Image.Image,
    original: Image.Image,
    cloth_type: str,
) -> Image.Image:
    """
    Hard-composite the person's face and upper identity from the original
    onto the diffusion result.

    The IDM-VTON model with strength=1.0 denoises the entire image,
    regenerating the face even though it's not in the inpaint mask.
    This function restores the person's identity by detecting the face
    in the original and blending it back with a soft mask.

    For dresses/full-body: restore face + hair + neck + upper chest.
    For lower_body: restore face + hair + neck + shoulders + upper torso.
    """
    import cv2

    orig_np = np.array(original.convert("RGB"), dtype=np.float32)
    result_np = np.array(result.convert("RGB"), dtype=np.float32)
    h, w = orig_np.shape[:2]

    if orig_np.shape != result_np.shape:
        result_np = np.array(
            result.convert("RGB").resize(original.size, Image.LANCZOS),
            dtype=np.float32,
        )

    # Detect face in original using OpenCV Haar cascade
    orig_uint8 = np.array(original.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(orig_uint8, cv2.COLOR_RGB2GRAY)
    cascade_path = os.path.join(
        cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
    )
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        return result

    min_dim = max(30, int(min(h, w) * 0.04))
    faces = detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_dim, min_dim)
    )
    if len(faces) == 0:
        return result

    # Take the largest face
    (fx, fy, fw, fh) = max(faces, key=lambda r: r[2] * r[3])

    # Build identity region mask: face + generous margins for hair, jawline, neck
    # Top: extends into hair (40% above face box)
    # Bottom: extends into neck/upper chest (60% below face box)
    # Sides: extends to include ears (15% wider each side)
    pad_x = int(fw * 0.15)
    pad_y_top = int(fh * 0.40)  # hair
    pad_y_bottom = int(fh * 0.60)  # neck + upper chest

    face_x1 = max(0, fx - pad_x)
    face_y1 = max(0, fy - pad_y_top)
    face_x2 = min(w, fx + fw + pad_x)
    face_y2 = min(h, fy + fh + pad_y_bottom)

    # Create soft-edged mask using distance transform for smooth blending
    mask_region = np.zeros((h, w), dtype=np.uint8)
    mask_region[face_y1:face_y2, face_x1:face_x2] = 255

    # Erode then blur for soft edges (feather ~15px)
    erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_eroded = cv2.erode(mask_region, erode_k, iterations=2)
    blur_size = max(15, min(fw, fh) // 8)
    if blur_size % 2 == 0:
        blur_size += 1
    mask_soft = cv2.GaussianBlur(mask_eroded.astype(np.float32), (blur_size, blur_size), 0)
    mask_3d = mask_soft[:, :, np.newaxis] / 255.0

    # Composite: result * (1 - mask) + original * mask
    restored = (result_np * (1.0 - mask_3d) + orig_np * mask_3d).astype(np.uint8)

    logger.info(
        "face_identity_restored face_box=(%d,%d,%d,%d) region=(%d,%d,%d,%d) blur=%d",
        fx, fy, fw, fh, face_x1, face_y1, face_x2, face_y2, blur_size,
    )

    return Image.fromarray(restored, mode="RGB")


def run_idm_vton_inference(
    person_img: Image.Image,
    garment_img: Image.Image,
    garment_desc: str,
    cloth_type: str,
    garment_subtype: str = "",
    steps: int = 30,
    seed: int = 42,
    auto_crop: bool = True,
    external_mask: Image.Image | None = None,
    protected_mask: Image.Image | None = None,
    mask_strategy: str = "external",
    mask_quality_score: float | None = None,
    guidance_scale: float | None = None,
) -> tuple[Image.Image, dict[str, object]]:
    global pipe, parsing_model, openpose_model
    global densepose_predictor, densepose_cfg, tensor_transform, get_mask_location_fn

    import cv2

    device = DEVICE

    if torch.cuda.is_available():
        openpose_model.preprocessor.body_estimation.model.to(device)
        pipe.to(device)
        pipe.unet_encoder.to(device)

    from mask_pipeline import (
        WorkerMaskStrategy,
        apply_protected_mask,
        fuse_hybrid_mask,
        select_worker_mask_strategy,
    )

    garm_img = garment_img.convert("RGB").resize(TARGET_SIZE)
    human_img_orig = person_img.convert("RGB")

    width, height = human_img_orig.size
    left, top, crop_size = 0.0, 0.0, None

    if auto_crop:
        target_width = int(min(width, height * (TARGET_W / TARGET_H)))
        target_height = int(min(height, width * (TARGET_H / TARGET_W)))

        is_full_body = cloth_type in ("dresses", "lower_body", "full_body")
        if is_full_body:
            # Bottom-anchored crop: keep the bottom portion, sacrifice top
            left = (width - target_width) / 2
            bottom = height
            top = height - target_height
            right = (width + target_width) / 2
        else:
            # Center-anchored crop (default for upper_body)
            left = (width - target_width) / 2
            top = (height - target_height) / 2
            right = (width + target_width) / 2
            bottom = (height + target_height) / 2

        cropped_img = human_img_orig.crop((left, top, right, bottom))
        crop_size = cropped_img.size
        human_img = cropped_img.resize(TARGET_SIZE)
    else:
        human_img = human_img_orig.resize(TARGET_SIZE)

    # Always compute AutoMasker mask (SCHP + OpenPose) for routing / hybrid
    keypoints = openpose_model(human_img.resize((384, 512)))
    model_parse, _ = parsing_model(human_img.resize((384, 512)))
    automasker_mask, _ = get_mask_location_fn("hd", cloth_type, model_parse, keypoints)
    automasker_mask = automasker_mask.resize(TARGET_SIZE)

    min_quality = float(os.environ.get("MASK_MIN_QUALITY_SCORE", "62.0"))
    strategy = select_worker_mask_strategy(
        external_mask,
        mask_quality_score,
        min_quality=min_quality,
    )
    if mask_strategy == "automasker":
        strategy = WorkerMaskStrategy.AUTOMASKER
    elif mask_strategy == "hybrid":
        strategy = WorkerMaskStrategy.HYBRID

    mask_meta: dict[str, object] = {
        "mask_type_used": strategy.value,
        "mask_quality_score": mask_quality_score,
    }

    if strategy == WorkerMaskStrategy.EXTERNAL and external_mask is not None:
        mask = external_mask.convert("L").resize(TARGET_SIZE)
    elif strategy == WorkerMaskStrategy.HYBRID:
        mask = fuse_hybrid_mask(external_mask, automasker_mask, cloth_type)
        mask_meta["mask_type_used"] = "hybrid"
    else:
        mask = automasker_mask
        mask_meta["mask_type_used"] = "automasker"

    mask = apply_protected_mask(mask, protected_mask)

    # Lower-body silhouette enhancement: AND mask with garment silhouette
    # clipped to editable body labels to prevent mask bleeding into upper body
    if cloth_type == "lower_body":
        from mask_pipeline import (
            EDITABLE_BODY_REGIONS,
        )
        _target_labels = EDITABLE_BODY_REGIONS.get("lower_body", set())

        # Get garment silhouette from garment image
        garm_gray = np.array(garm_img.convert("L"), dtype=np.uint8)
        _, garm_silhouette_mask = cv2.threshold(garm_gray, 240, 255, cv2.THRESH_BINARY_INV)

        # Exclude face/hair from garment silhouette to prevent identity bleed
        _face_hair_labels = {2, 11}  # _LABEL_HAIR, _LABEL_FACE
        _schp_exclude = np.isin(model_parse, list(_face_hair_labels)).astype(np.uint8) * 255
        _schp_exclude = cv2.resize(_schp_exclude, (garm_silhouette_mask.shape[1], garm_silhouette_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        garm_silhouette_mask[_schp_exclude > 127] = 0

        # Get SCHP body region labels — model_parse is at half-res (512x384),
        # upsample to match garm_silhouette_mask (1024x768)
        _schp_body = np.isin(model_parse, list(_target_labels)).astype(np.uint8) * 255
        _schp_body = cv2.resize(_schp_body, (garm_silhouette_mask.shape[1], garm_silhouette_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

        # AND: only keep silhouette pixels that overlap with body region
        _clipped = np.minimum(garm_silhouette_mask, _schp_body)

        # Check if AND removed too much (>40% of silhouette pixels)
        silhouette_px = int(np.sum(garm_silhouette_mask > 127))
        clipped_px = int(np.sum(_clipped > 127))
        if silhouette_px > 0 and clipped_px < silhouette_px * 0.60:
            # Geometric fallback: SCHP misclassified pants as upper_clothes
            # Use all body labels EXCLUDING legs to preserve leg geometry
            from mask_pipeline import _LABEL_LEFT_LEG, _LABEL_RIGHT_LEG
            _fallback_labels = [l for l in range(1, 17) if l not in (_LABEL_LEFT_LEG, _LABEL_RIGHT_LEG)]
            _all_body = np.isin(model_parse, _fallback_labels).astype(np.uint8) * 255
            _all_body = cv2.resize(_all_body, (garm_silhouette_mask.shape[1], garm_silhouette_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            _clipped = np.minimum(garm_silhouette_mask, _all_body)
            # Morphological closing to fill holes from jeans
            close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            _clipped = cv2.morphologyEx(_clipped, cv2.MORPH_CLOSE, close_kernel)
            logger.info("lower_body_geometric_fallback silhouette_px=%d clipped_px=%d", silhouette_px, clipped_px)

        # Light dilation to preserve garment shape and prevent mask erosion
        _dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        _clipped = cv2.dilate(_clipped, _dilate_k, iterations=1)

        # Merge enhanced silhouette into mask
        mask_np = np.array(mask.convert("L"), dtype=np.uint8)
        mask_np = np.maximum(mask_np, _clipped)

        # Hard upper-body exclusion: zero out anything above the mask's
        # topmost row to prevent silhouette bleed into the torso
        rows_with_mask = np.where(mask_np.any(axis=1))[0]
        if len(rows_with_mask) > 0:
            mask_top = int(rows_with_mask[0])
            # 20px margin preserves the waistband region that the diffusion
            # model needs for natural lower-body garment placement.
            exclude_top = max(0, mask_top - 20)
            mask_np[:exclude_top, :] = 0

        mask = Image.fromarray(mask_np, mode="L")

    # Dress/full-body silhouette enhancement: AND mask with garment silhouette
    # clipped to ALL editable body labels (clothing + arms + legs)
    if cloth_type in ("dresses", "full_body"):
        from mask_pipeline import EDITABLE_BODY_REGIONS
        _target_labels = EDITABLE_BODY_REGIONS.get("dresses", set())

        garm_gray = np.array(garm_img.convert("L"), dtype=np.uint8)
        _, garm_silhouette_mask = cv2.threshold(garm_gray, 240, 255, cv2.THRESH_BINARY_INV)

        _schp_body = np.isin(model_parse, list(_target_labels)).astype(np.uint8) * 255
        _schp_body = cv2.resize(_schp_body, (garm_silhouette_mask.shape[1], garm_silhouette_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

        # CRITICAL: Exclude the face region from the garment silhouette.
        # The garment image contains the garment MODEL's body, including their
        # face. If we don't exclude it, the model's face bleeds into the mask
        # and the diffusion model regenerates the garment model's face instead
        # of preserving the person's identity.
        _face_labels = {11}  # _LABEL_FACE
        _hair_labels = {2}   # _LABEL_HAIR
        _exclude_labels = _face_labels | _hair_labels
        _schp_face_hair = np.isin(model_parse, list(_exclude_labels)).astype(np.uint8) * 255
        _schp_face_hair = cv2.resize(_schp_face_hair, (garm_silhouette_mask.shape[1], garm_silhouette_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        # Zero out face/hair from garment silhouette
        garm_silhouette_mask[_schp_face_hair > 127] = 0

        _clipped = np.minimum(garm_silhouette_mask, _schp_body)

        # Geometric fallback: if AND removed too much, use all clothing labels
        silhouette_px = int(np.sum(garm_silhouette_mask > 127))
        clipped_px = int(np.sum(_clipped > 127))
        if silhouette_px > 0 and clipped_px < silhouette_px * 0.60:
            _all_body = np.isin(model_parse, list(range(1, 17))).astype(np.uint8) * 255
            _all_body = cv2.resize(_all_body, (garm_silhouette_mask.shape[1], garm_silhouette_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            _clipped = np.minimum(garm_silhouette_mask, _all_body)
            close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            _clipped = cv2.morphologyEx(_clipped, cv2.MORPH_CLOSE, close_kernel)
            logger.info("dress_geometric_fallback silhouette_px=%d clipped_px=%d", silhouette_px, clipped_px)

        mask_np = np.array(mask.convert("L"), dtype=np.uint8)
        mask_np = np.maximum(mask_np, _clipped)
        mask = Image.fromarray(mask_np, mode="L")

    logger.info(
        "mask_selected strategy=%s mask_size=%s quality_score=%s",
        mask_meta["mask_type_used"],
        mask.size,
        mask_quality_score,
    )

    from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation
    human_img_arg = _apply_exif_orientation(human_img.resize((384, 512)))
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")

    with torch.no_grad():
        densepose_outputs = densepose_predictor(human_img_arg)["instances"]

    from densepose.vis.densepose_results import DensePoseResultsFineSegmentationVisualizer
    from densepose.vis.extractor import create_extractor

    vis = DensePoseResultsFineSegmentationVisualizer(cfg=densepose_cfg)
    extractor = create_extractor(vis)
    data = extractor(densepose_outputs)

    gray_img = cv2.cvtColor(human_img_arg, cv2.COLOR_BGR2GRAY)
    gray_img = np.tile(gray_img[:, :, np.newaxis], [1, 1, 3])
    pose_img = vis.visualize(gray_img, data)
    pose_img = pose_img[:, :, ::-1]
    pose_img = Image.fromarray(pose_img).resize(TARGET_SIZE)

    effective_guidance = guidance_scale if guidance_scale is not None else GUIDANCE_SCALE

    if cloth_type in ("lower_body", "dresses", "full_body"):
        prompt = _build_subtype_aware_prompt(garment_desc, garment_subtype)
        if cloth_type == "lower_body":
            negative_prompt = _build_source_specific_negative() + (
                ", changed shirt, new shirt, different top, altered torso, "
                "regenerated upper body, different arms, moved hands, "
                "changed shoulders, modified chest, new upper garment, "
                "generic pants, plain pants, lost pockets, missing seams, "
                "lost belt loops, lost fabric detail, smooth texture, "
                "lost garment structure, changed silhouette"
            )
        else:
            # dresses / full_body: suppress common failure modes + identity bleed
            negative_prompt = _build_source_specific_negative() + (
                ", changed garment category, wrong outfit type, "
                "mini dress, different silhouette, wrong length, "
                "missing sleeves, changed sleeve style, wrong neckline, "
                "different face, new face, changed facial features, "
                "different hair, changed hair color, different skin tone, "
                "regenerated face, altered identity, different person"
            )
    else:
        prompt = "model is wearing " + garment_desc
        negative_prompt = _build_source_specific_negative()

    with torch.inference_mode():
        with _maybe_autocast():
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
                prompt,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )

            prompt_c = "a photo of " + garment_desc
            prompt_embeds_c, _, _, _ = pipe.encode_prompt(
                prompt_c,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
                negative_prompt=negative_prompt,
            )

    pose_tensor = tensor_transform(pose_img).unsqueeze(0).to(device, TORCH_DTYPE)
    garm_tensor = tensor_transform(garm_img).unsqueeze(0).to(device, TORCH_DTYPE)
    generator = torch.Generator(device).manual_seed(seed) if seed is not None and torch.cuda.is_available() else None

    with torch.inference_mode():
        with _maybe_autocast():
            images = pipe(
                prompt_embeds=prompt_embeds.to(device, TORCH_DTYPE),
                negative_prompt_embeds=negative_prompt_embeds.to(device, TORCH_DTYPE),
                pooled_prompt_embeds=pooled_prompt_embeds.to(device, TORCH_DTYPE),
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device, TORCH_DTYPE),
                num_inference_steps=steps,
                generator=generator,
                strength=1.0,
                pose_img=pose_tensor.to(device, TORCH_DTYPE),
                text_embeds_cloth=prompt_embeds_c.to(device, TORCH_DTYPE),
                cloth=garm_tensor.to(device, TORCH_DTYPE),
                mask_image=mask,
                image=human_img,
                height=TARGET_H,
                width=TARGET_W,
                ip_adapter_image=garm_img.resize(TARGET_SIZE),
                guidance_scale=effective_guidance,
            )[0]

    if auto_crop and crop_size is not None:
        out_img = images[0].resize(crop_size)
        final_img = human_img_orig.copy()

        # Edge-aware feathering: blend the crop output with the original
        # at the crop boundary to avoid visible seams
        crop_w, crop_h = crop_size
        feather_px = max(12, min(crop_w, crop_h) // 20)

        # Build 1D feathering ramps for each edge
        alpha = np.ones((crop_h, crop_w), dtype=np.float32)

        # Bottom edge fade (only if not at image bottom)
        orig_bottom = int(top) + crop_h
        if orig_bottom < height:
            ramp = np.linspace(1.0, 0.0, feather_px)
            alpha[-feather_px:, :] *= ramp[:, np.newaxis]

        # Left edge fade (only if not at image left)
        if int(left) > 0:
            ramp = np.linspace(0.0, 1.0, feather_px)
            alpha[:, :feather_px] *= ramp[np.newaxis, :]

        # Right edge fade (only if not at image right)
        orig_right = int(left) + crop_w
        if orig_right < width:
            ramp = np.linspace(1.0, 0.0, feather_px)
            alpha[:, -feather_px:] *= ramp[np.newaxis, :]

        # Alpha composite: output * alpha + original * (1 - alpha)
        out_np = np.array(out_img.convert("RGB"), dtype=np.float32)
        orig_crop = np.array(
            human_img_orig.crop((int(left), int(top), orig_right, orig_bottom))
            .resize((crop_w, crop_h)),
            dtype=np.float32,
        )
        blended = (out_np * alpha[..., np.newaxis] + orig_crop * (1.0 - alpha[..., np.newaxis])).astype(np.uint8)
        final_img.paste(Image.fromarray(blended), (int(left), int(top)))

        # ── Face identity restoration ──────────────────────────────────
        # The IDM-VTON model with strength=1.0 denoises the ENTIRE image,
        # regenerating the face even though it's not in the inpaint mask.
        # We MUST hard-composite the person's face from the original to
        # preserve identity. This is not a heuristic — it's required for
        # any diffusion-based try-on with strength=1.0.
        if cloth_type in ("dresses", "full_body", "lower_body"):
            final_img = _restore_person_identity(
                final_img, human_img_orig, cloth_type,
            )

        return final_img, mask_meta

    return images[0], mask_meta


# =============================================================================
# Per-job
# =============================================================================

def run_inference(job_input: dict[str, Any], job_id: str) -> dict[str, Any]:
    """
    Direct inference — no preprocessing service.

    Downloads raw person + garment images, resizes to target size,
    and runs IDM-VTON with the model's built-in AutoMasker for mask
    generation. No external masks, no protected regions, no preprocessing
    pipeline.
    """
    from quality_validation import validate_output_quality

    job_start = time.perf_counter()

    person_url = job_input.get("person_image_url") or job_input.get("person_image")
    garment_url = job_input.get("garment_image_url") or job_input.get("garment_image")
    garment_desc = job_input.get("garment_desc") or job_input.get("garment_description") or "garment"
    garment_subtype = job_input.get("garment_subtype", "")
    cloth_type = job_input.get("cloth_type", "upper_body")
    steps = int(job_input.get("steps", DENOISE_STEPS))
    seed = int(job_input.get("seed", random.randint(0, 2**31 - 1)))
    trace_id = job_input.get("trace_id", "")

    if not person_url or not garment_url:
        raise ValueError("Missing required inputs: person_image_url and garment_image_url")

    cloth_type_map = {
        "upper": "upper_body",
        "upper_body": "upper_body",
        "lower": "lower_body",
        "lower_body": "lower_body",
        "dress": "dresses",
        "dresses": "dresses",
        "overall": "dresses",
        "full_body": "dresses",
    }
    vton_type = cloth_type_map.get(cloth_type, "upper_body")

    garment_desc = garment_desc.strip()
    if garment_desc.lower().startswith(("a ", "an ", "the ")):
        garment_desc = garment_desc[garment_desc.index(" ") + 1:].strip()

    logger.info(
        "inference_start cloth_type=%s steps=%s seed=%s garment_desc=%s trace_id=%s",
        vton_type, steps, seed, garment_desc, trace_id,
    )

    # ── Download raw images ──
    download_start = time.perf_counter()
    person_img = download_image(person_url)
    garment_img = download_image(garment_url)
    download_ms = (time.perf_counter() - download_start) * 1000

    # ── Garment RGB diagnostics ──
    garm_np = np.array(garment_img.convert("RGB"), dtype=np.float32)
    garm_mean_all = float(np.mean(garm_np))
    logger.info(
        "garment_rgb_stats mean_all=%.1f is_dark=%s",
        garm_mean_all, garm_mean_all < 80.0,
    )

    # ── Direct inference — always use AutoMasker, no preprocessing ──
    effective_guidance = GUIDANCE_SCALE
    # Only reduce guidance for dark UPPER-BODY garments where over-saturation
    # is the primary failure mode. For lower-body and dresses, keep full
    # guidance to preserve garment detail fidelity (pockets, seams, texture).
    if garm_mean_all < 80.0 and vton_type == "upper_body":
        effective_guidance = GUIDANCE_SCALE * 0.75

    inference_start = time.perf_counter()
    result, mask_meta = run_idm_vton_inference(
        person_img=person_img,
        garment_img=garment_img,
        garment_desc=garment_desc,
        cloth_type=vton_type,
        garment_subtype=garment_subtype,
        steps=steps,
        seed=seed,
        auto_crop=True,
        external_mask=None,
        protected_mask=None,
        mask_strategy="automasker",
        mask_quality_score=None,
        guidance_scale=effective_guidance,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_ms = (time.perf_counter() - inference_start) * 1000

    # ── Quality validation ──
    geometry_report = None
    if result is not None:
        geometry_report = validate_output_quality(
            person_img,
            result,
            Image.fromarray(np.zeros((TARGET_H, TARGET_W), dtype=np.uint8), mode="L"),
            vton_type,
            garment_img,
        )
        if not geometry_report["passed"]:
            logger.warning("quality_check_failed reasons=%s", geometry_report["reasons"])

    upload_start = time.perf_counter()
    result_url = _upload_to_cloudinary(result, job_id)
    upload_ms = (time.perf_counter() - upload_start) * 1000

    total_ms = (time.perf_counter() - job_start) * 1000

    logger.info(
        "job_complete total_ms=%.0f download_ms=%.0f inference_ms=%.0f upload_ms=%.0f "
        "mask_type=%s trace_id=%s",
        total_ms, download_ms, inference_ms, upload_ms,
        mask_meta.get("mask_type_used"),
        trace_id,
    )

    return {
        "status": "success",
        "result_url": result_url,
        "cloth_type_used": vton_type,
        "steps_used": steps,
        "seed": seed,
        "inference_ms": round(inference_ms, 2),
        "upload_ms": round(upload_ms, 2),
        "download_ms": round(download_ms, 2),
        "total_ms": round(total_ms, 2),
        "mask_type_used": mask_meta.get("mask_type_used"),
        "trace_id": trace_id,
    }


# =============================================================================
# RunPod handler
# =============================================================================

def handler(job: dict[str, Any]) -> dict[str, Any]:
    job_start = time.time()

    if not _WARM.is_set():
        warmup()
        cold_start = True
    else:
        cold_start = False

    global _REUSE_COUNT
    _REUSE_COUNT += 1

    logger.info(
        "handler_invoked cold_start=%s reuse_count=%s job_id=%s",
        cold_start, _REUSE_COUNT, job.get("id", "unknown"),
    )

    user_input = job.get("input", {})
    job_id = str(job.get("id", "unknown"))

    try:
        output = run_inference(user_input, job_id)
        output["cold_start"] = cold_start
        return output
    except Exception as exc:
        total_ms = (time.time() - job_start) * 1000
        logger.error("job_failed total_ms=%.0f error=%s", total_ms, exc, exc_info=True)
        return {
            "status": "error",
            "error": str(exc),
            "error_code": "worker_inference_failed",
            "total_ms": round(total_ms, 2),
            "cold_start": cold_start,
        }


# =============================================================================
# Startup
# =============================================================================

_ensure_logging()

# ── Startup diagnostics: verify mask_pipeline import ──────────────────
def _startup_diagnostics():
    """
    Verify that mask_pipeline.py is available and importable at runtime.

    Checks:
      1. /workspace is in sys.path (or adds it)
      2. /workspace/mask_pipeline.py exists on disk
      3. The module imports correctly

    This runs once at worker startup, before any job arrives, so the
    ModuleNotFoundError that previously only appeared during jobs
    is caught early.
    """
    logger.info("STARTUP_DIAG: cwd=%s", os.getcwd())
    logger.info("STARTUP_DIAG: sys.path=%s", sys.path)
    logger.info("STARTUP_DIAG: handler_location=%s", os.path.abspath(__file__))

    # Belt-and-suspenders: ensure /workspace is on sys.path
    ws = "/workspace"
    if ws not in sys.path:
        sys.path.insert(0, ws)
        logger.info("STARTUP_DIAG: added %s to sys.path", ws)

    # Check file exists on disk
    mp_path = os.path.join(ws, "mask_pipeline.py")
    if not os.path.isfile(mp_path):
        logger.error(
            "STARTUP_DIAG: mask_pipeline.py NOT FOUND at %s — "
            "Dockerfile must have COPY mask_pipeline.py /workspace/mask_pipeline.py",
            mp_path,
        )
        return False

    logger.info("STARTUP_DIAG: mask_pipeline.py found at %s (%d bytes)", mp_path, os.path.getsize(mp_path))

    # Actual import test — catches ModuleNotFoundError at startup, not during a job
    try:
        from mask_pipeline import (
            WorkerMaskStrategy,
            apply_protected_mask,
            fuse_hybrid_mask,
            detect_inference_failures,
            select_worker_mask_strategy,
        )
        logger.info("STARTUP_DIAG: import mask_pipeline OK")
        return True
    except Exception as exc:
        logger.error(
            "STARTUP_DIAG: import mask_pipeline FAILED — %s: %s",
            type(exc).__name__, exc,
        )
        return False

_startup_diagnostics_result = _startup_diagnostics()
if not _startup_diagnostics_result:
    logger.warning(
        "STARTUP_DIAG: mask_pipeline is unavailable — inference retry "
        "and hybrid mask features will fail when a job arrives"
    )

logger.info("=" * 60)
logger.info("IDM-VTON Worker v2.0.0 — loading")
logger.info("target_size=%s", TARGET_SIZE)
logger.info("device=%s", DEVICE)
logger.info("gpu_available=%s", torch.cuda.is_available())
if torch.cuda.is_available():
    dev = torch.cuda.get_device_properties(0)
    logger.info("gpu_device=%s", dev.name)
    logger.info("vram_total_gb=%.1f", dev.total_memory / (1024**3))
logger.info("=" * 60)

if __name__ == "__main__":
    try:
        if not os.environ.get("RUNPOD_WARMUP_DISABLE"):
            warmup()
        runpod.serverless.start({"handler": handler})
    except Exception:
        logger.error("Worker startup failed")
        traceback.print_exc()
        sys.stdout.flush()
        raise
