#!/bin/bash
# =====================================================================
# Streaming Pruned-Transducer-Stateless7 — Environment Setup
#
# HF repo mapping (verified 2026-04-07):
#   libri_giga:   marcoyang/icefall-libri-giga-pruned-transducer-stateless7-streaming-2023-04-04
#                 files: exp/pretrained.pt (epoch-20-avg-4)
#                        data/lang_bpe_500/bpe.model
#   librispeech:  Zengwei/icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29
#                 files: exp/pretrained.pt (epoch-30-avg-9)
#                        data/lang_bpe_500/bpe.model
#
# Run this once before using the model.
# Re-running is safe (idempotent): install steps are skipped if packages
# already exist, and download is skipped if weights are already present.
# =====================================================================
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"

# =============================================================
# Stage 1: Read variant from config.yaml
# =============================================================
echo "=== Stage 1: Read variant ==="
VARIANT=$(python3 -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('$DIR/config.yaml')
print(cfg.model.variant)
")
echo "Variant: $VARIANT"

if [ "$VARIANT" = "libri_giga" ]; then
    HF_REPO="marcoyang/icefall-libri-giga-pruned-transducer-stateless7-streaming-2023-04-04"
elif [ "$VARIANT" = "librispeech" ]; then
    HF_REPO="Zengwei/icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29"
else
    echo "ERROR: unknown variant '$VARIANT'. Must be 'libri_giga' or 'librispeech'."
    exit 1
fi
echo "HF repo: $HF_REPO"

# =============================================================
# Stage 2: Detect environment
# =============================================================
echo "=== Stage 2: Detect environment ==="
TORCH_VER=$(python3 -c "import torch; print(torch.__version__.split('+')[0])")
PY_TAG="cp$(python3 -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')")"
echo "PyTorch version: $TORCH_VER | Python tag: $PY_TAG"

# Select kaldifeat wheel for platform
PLATFORM=$(python3 -c "import platform; print(platform.system())")
ARCH=$(python3 -c "import platform; print(platform.machine())")

if [ "$PLATFORM" = "Darwin" ] && [ "$ARCH" = "arm64" ]; then
    KALDIFEAT_WHEEL="https://huggingface.co/csukuangfj/kaldifeat/resolve/main/cpu/1.25.5.dev20241029/macos/kaldifeat-1.25.5.dev20250630+cpu.torch${TORCH_VER}-${PY_TAG}-${PY_TAG}-macosx_11_0_arm64.whl"
else
    KALDIFEAT_WHEEL="https://huggingface.co/csukuangfj/kaldifeat/resolve/main/cpu/1.25.5.dev20250307/linux-x64/kaldifeat-1.25.5.dev20250630+cpu.torch2.5.0-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.whl"
fi
echo "Platform: $PLATFORM/$ARCH"

# =============================================================
# Stage 3: Install packages (skip if already present)
# =============================================================
echo "=== Stage 3: Install packages ==="

pip install -q numpy sentencepiece tqdm huggingface_hub omegaconf

# kaldifeat: for streaming online Fbank feature extraction
if python3 -c "import kaldifeat" 2>/dev/null; then
    echo "kaldifeat already installed, skipping."
else
    pip install --no-deps "$KALDIFEAT_WHEEL"
fi

# NOTE: k2 and icefall are NOT required — model.py uses JIT-traced models directly.

# =============================================================
# Stage 4: Download model weights (skip if already present)
# =============================================================
echo "=== Stage 4: Download model weights (variant=$VARIANT) ==="
WEIGHTS_DIR="$DIR/weights/$VARIANT"
PRETRAINED="$WEIGHTS_DIR/exp/pretrained.pt"
BPE_MODEL="$WEIGHTS_DIR/data/lang_bpe_500/bpe.model"

# Files to download depend on variant:
#   libri_giga   -> exp/cpu_jit.pt (scripted model, single file)
#   librispeech  -> exp/encoder_jit_trace.pt + decoder_jit_trace.pt + joiner_jit_trace.pt
if [ "$VARIANT" = "libri_giga" ]; then
    JIT_FILES="exp/cpu_jit.pt"
else
    JIT_FILES="exp/encoder_jit_trace.pt exp/decoder_jit_trace.pt exp/joiner_jit_trace.pt"
fi

# Check if already downloaded (use bpe.model as sentinel)
if [ -f "$BPE_MODEL" ]; then
    ALL_PRESENT=true
    for f in $JIT_FILES; do
        if [ ! -f "$WEIGHTS_DIR/$f" ]; then
            ALL_PRESENT=false
        fi
    done
    if [ "$ALL_PRESENT" = "true" ]; then
        echo "Weights already present at $WEIGHTS_DIR, skipping download."
    else
        echo "Some JIT files missing, re-downloading..."
    fi
fi

if [ ! -f "$BPE_MODEL" ] || [ "$ALL_PRESENT" = "false" ]; then
    FILES_STR="${JIT_FILES} data/lang_bpe_500/bpe.model"
    python3 << PYEOF
from huggingface_hub import hf_hub_download
import os

repo = "$HF_REPO"
wd = "$WEIGHTS_DIR"
os.makedirs(wd + '/exp', exist_ok=True)
os.makedirs(wd + '/data/lang_bpe_500', exist_ok=True)

files = "$FILES_STR".split()
for f in files:
    print(f"Downloading {f} ...")
    hf_hub_download(repo_id=repo, filename=f, local_dir=wd)
    print(f"  -> {wd}/{f}")
PYEOF
fi

echo "=== setup.sh complete (variant=$VARIANT) ==="
