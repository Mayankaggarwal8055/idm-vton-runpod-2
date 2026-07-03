# syntax=docker/dockerfile:1.4

# =============================================================================
# IDM-VTON RunPod Inference Image
# =============================================================================

FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel AS build

# =============================================================================
# Environment
# =============================================================================

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    PYTHONPATH=/workspace \
    IDM_VTON_DIR=/workspace/IDM-VTON \
    IDM_VTON_MODEL=/workspace/models/yisol/IDM-VTON \
    DENSEPOSE_WEIGHTS=/workspace/IDM-VTON/ckpt/densepose/model_final_162be9.pkl \
    CLOUDINARY_FOLDER=trylix/tryon/results \
    ENABLE_XFORMERS=0 \
    ALLOW_TF32=1

WORKDIR /workspace

# =============================================================================
# Layer 1 — OS dependencies
# =============================================================================

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Layer 2 — Python dependencies
# =============================================================================

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
        diffusers==0.25.1 \
        transformers==4.41.1 \
        accelerate==0.25.0 \
        peft==0.11.1 \
        safetensors==0.4.3 \
        tokenizers==0.19.1 \
        huggingface_hub==0.25.2 \
        einops==0.7.0 \
        scipy==1.10.1 \
        scikit-image==0.21.0 \
        opencv-python-headless==4.7.0.72 \
        Pillow==9.4.0 \
        onnxruntime-gpu==1.16.2 \
        av==12.3.0 \
        "protobuf<5" \
        fvcore \
        cloudpickle \
        omegaconf \
        pycocotools \
        tqdm \
        requests==2.32.3 \
        runpod==1.9.1 \
        cloudinary==1.41.0

# =============================================================================
# Layer 3 — Detectron2
# =============================================================================

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
        "detectron2@git+https://github.com/facebookresearch/detectron2.git@main"

# =============================================================================
# Layer 4 — Clone IDM-VTON repo + download ALL binary checkpoints
# =============================================================================

# 4a — Clone the repository and download OpenPose checkpoint
# The GIT_REVISION build arg acts as a cache buster — pass the latest commit SHA
# at build time to invalidate this layer when the repo changes.
#   docker build --build-arg GIT_REVISION=$(git rev-parse HEAD) ...
ARG GIT_REVISION=unknown
RUN git lfs install && \
    echo "Cloning IDM-VTON at revision: ${GIT_REVISION}" && \
    git clone --depth 1 https://github.com/Mayankaggarwal8055/IDM-VTON.git $IDM_VTON_DIR && \
    mkdir -p $IDM_VTON_DIR/ckpt/openpose/ckpts && \
    curl -L \
        -o $IDM_VTON_DIR/ckpt/openpose/ckpts/body_pose_model.pth \
        https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/openpose/ckpts/body_pose_model.pth && \
    ln -sf \
        $IDM_VTON_DIR/ckpt/openpose/ckpts/body_pose_model.pth \
        $IDM_VTON_DIR/ckpt/openpose/body_pose_model.pth

# 4b — Download ONNX humanparsing models (bypasses git LFS issues)
RUN python - <<'PY'
from huggingface_hub import hf_hub_download
import os, shutil, sys

dest = os.environ["IDM_VTON_DIR"] + "/ckpt/humanparsing"
os.makedirs(dest, exist_ok=True)

for fname in ["parsing_atr.onnx", "parsing_lip.onnx"]:
    print(f"Downloading {fname}...")
    cached = hf_hub_download(
        repo_id="levihsu/OOTDiffusion",
        filename=f"checkpoints/humanparsing/{fname}",
        cache_dir="/tmp/hf_onnx_cache",
    )
    final = os.path.join(dest, fname)
    shutil.copy2(cached, final)
    size_mb = os.path.getsize(final) / 1024 / 1024
    if size_mb < 50:
        print(f"FATAL: {fname} is {size_mb:.2f} MB — still a git LFS pointer", flush=True)
        sys.exit(1)
    print(f"  OK: {fname} = {size_mb:.1f} MB")

shutil.rmtree("/tmp/hf_onnx_cache", ignore_errors=True)
print("humanparsing ONNX downloads complete")
PY

# 4c — Download DensePose checkpoint (model_final_162be9.pkl)
RUN python3 - <<'PY'
from huggingface_hub import hf_hub_download
import os, shutil, sys

dest = os.environ["IDM_VTON_DIR"] + "/ckpt/densepose"
os.makedirs(dest, exist_ok=True)

fname = "model_final_162be9.pkl"
print(f"Downloading {fname}...")
cached = hf_hub_download(
    repo_id="yisol/IDM-VTON",
    filename=f"densepose/{fname}",
    cache_dir="/tmp/hf_densepose_cache",
)
final = os.path.join(dest, fname)
shutil.copy2(cached, final)

size_mb = os.path.getsize(final) / 1024 / 1024
print(f"{fname} = {size_mb:.1f} MB")
if size_mb < 100:
    sys.exit("DensePose checkpoint looks truncated or is a pointer file")

shutil.rmtree("/tmp/hf_densepose_cache", ignore_errors=True)
print("DensePose download complete")
PY

# 4d — Verify OpenPose checkpoint size
RUN python - <<'PY'
import os, sys
p = "/workspace/IDM-VTON/ckpt/openpose/ckpts/body_pose_model.pth"
if not os.path.exists(p):
    sys.exit(f"Missing: {p}")
size_mb = os.path.getsize(p) / 1024 / 1024
print("body_pose_model.pth size_mb =", size_mb)
if size_mb < 10:
    sys.exit("body_pose_model.pth looks corrupted or incomplete")
PY

# =============================================================================
# Layer 5 — Download full IDM-VTON SDXL weights
# =============================================================================

RUN python - <<'PY'
from huggingface_hub import snapshot_download

target_dir = "/workspace/models/yisol/IDM-VTON"

print("Downloading IDM-VTON weights...")

snapshot_download(
    repo_id="yisol/IDM-VTON",
    local_dir=target_dir,
    local_dir_use_symlinks=False,
)

print("Download complete")
print("Saved to:", target_dir)
PY

# =============================================================================
# Layer 6 — Build validation (IDM-VTON pipeline)
# =============================================================================

RUN python - <<'PY'
import os
import sys

root = os.environ["IDM_VTON_DIR"]
demo = os.path.join(root, "gradio_demo")

sys.path.insert(0, root)
sys.path.insert(0, demo)

required_files = {
    os.path.join(root, "ckpt/humanparsing/parsing_atr.onnx"): 50,
    os.path.join(root, "ckpt/humanparsing/parsing_lip.onnx"): 50,
    os.path.join(root, "ckpt/openpose/ckpts/body_pose_model.pth"): 10,
    os.path.join(root, "ckpt/densepose/model_final_162be9.pkl"): 100,
}
for path, min_mb in required_files.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    size_mb = os.path.getsize(path) / 1024 / 1024
    if size_mb < min_mb:
        raise RuntimeError(
            f"File too small ({size_mb:.2f} MB < {min_mb} MB) — "
            f"likely corrupted: {path}"
        )
    print(f"Size OK: {os.path.basename(path)} = {size_mb:.1f} MB")

import diffusers, transformers, torch, cv2, onnxruntime, detectron2
print(f"Core imports OK (diffusers={diffusers.__version__} torch={torch.__version__})")

from src.tryon_pipeline import StableDiffusionXLInpaintPipeline
print("Pipeline import OK")

from utils_mask import get_mask_location
print("Mask utils import OK")

from preprocess.openpose.run_openpose import OpenPose
print("OpenPose import OK")

from densepose import add_densepose_config
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
print("DensePose + Detectron2 imports OK")

print("All validation passed")
PY

# =============================================================================
# Copy RunPod handler + mask pipeline
# =============================================================================

COPY handler.py /workspace/handler.py
COPY mask_pipeline.py /workspace/mask_pipeline.py
COPY quality_validation.py /workspace/quality_validation.py

# =============================================================================
# Layer 7 — Validate worker module (mask_pipeline.py)
# =============================================================================
# NOTE: This runs AFTER the COPY layer above, so /workspace/mask_pipeline.py exists.

RUN python - <<'PY'
import os
import sys

# Ensure /workspace is on the path — this is the target for COPY mask_pipeline.py
_ws = "/workspace"
sys.path.insert(0, _ws)

_mp = os.path.join(_ws, "mask_pipeline.py")

if not os.path.exists(_mp):
    raise FileNotFoundError(
        f"MASK_PIPELINE NOT FOUND at {_mp}. "
        "This should never happen because the COPY layer above places it there."
    )

print(f"mask_pipeline.py exists at {_mp} ({os.path.getsize(_mp)} bytes)")

try:
    from mask_pipeline import (
        WorkerMaskStrategy,
        apply_protected_mask,
        fuse_hybrid_mask,
        detect_inference_failures,
        select_worker_mask_strategy,
    )
    print("import mask_pipeline OK")
    print(f"  WorkerMaskStrategy: {list(WorkerMaskStrategy)}")
    print(f"  apply_protected_mask: {callable(apply_protected_mask)}")
    print(f"  fuse_hybrid_mask: {callable(fuse_hybrid_mask)}")
    print(f"  detect_inference_failures: {callable(detect_inference_failures)}")
    print(f"  select_worker_mask_strategy: {callable(select_worker_mask_strategy)}")
except Exception as exc:
    raise RuntimeError(f"Failed to import mask_pipeline: {exc}") from exc

print("Mask pipeline validation passed")
PY

# =============================================================================
# Runtime
# =============================================================================

WORKDIR $IDM_VTON_DIR

CMD ["python", "-u", "/workspace/handler.py"]
