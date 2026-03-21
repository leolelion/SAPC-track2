#!/bin/bash
# NeMo FastConformer: install nemo_toolkit[asr] and pre-download model weights.
#
# The RunPod competition image (xiuwenz2/sapc2-runtime) already has PyTorch 2.5.0+cu124,
# torchaudio, torchvision, and numpy preinstalled. The venv is created with
# --system-site-packages so it inherits those, and only NeMo (plus its
# non-torch extras) is installed on top.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="${DIR}/venv"

# NeMo 2.x requires Python 3.10 or 3.11.
# kaldialign and other NeMo deps have no wheels for Python 3.12+.
PYTHON=""
for candidate in python3.11 python3.10; do
    if command -v "$candidate" &>/dev/null; then
        PYTHON="$candidate"
        break
    fi
done
if [ -z "$PYTHON" ]; then
    PY_VER="$(python3 -c 'import sys; print(sys.version_info[:2])')"
    echo "ERROR: NeMo requires Python 3.10 or 3.11 (found ${PY_VER})."
    echo "       Install with:  brew install python@3.11"
    echo "       Then re-run this script."
    exit 1
fi
echo "Using Python: $($PYTHON --version)"

echo "=== NeMo FastConformer: creating venv (--system-site-packages) at ${VENV} ==="
# --system-site-packages: inherit system torch/torchaudio/numpy — avoids ABI mismatches.
rm -rf "${VENV}"
"$PYTHON" -m venv --system-site-packages "${VENV}"
"${VENV}/bin/pip" install --upgrade pip -q

echo "=== NeMo FastConformer: installing nemo_toolkit[asr] ==="
# kaldialign (a NeMo dep) needs C++/cmake to build; no macOS arm64 wheel.
# On macOS it may fail — that's OK, inference still works without it.
"${VENV}/bin/pip" install --no-cache-dir --prefer-binary \
    "omegaconf>=2.3" \
    "huggingface_hub>=0.24" \
    sentencepiece \
    "nemo_toolkit[asr]>=2.5.0" \
    || echo "WARNING: nemo_toolkit[asr] did not fully install (some build deps may be missing). Trying base package ..."

# Install torchvision inside the venv so it matches the venv's torch version.
# Without this, model.py's sys.path injection picks up the system torchvision
# (built against system torch), causing an ABI/op mismatch at import time.
"${VENV}/bin/pip" install --no-cache-dir --prefer-binary torchvision \
    || echo "WARNING: torchvision install failed (inference may still work if torchvision is unused at runtime)."

# If the full [asr] extras failed, ensure the base nemo_toolkit is present.
"${VENV}/bin/python3" -c "import nemo" 2>/dev/null \
    || "${VENV}/bin/pip" install --no-cache-dir --prefer-binary "nemo_toolkit>=2.5.0"

echo "=== NeMo FastConformer: verifying installation ==="
"${VENV}/bin/python3" -c "
import numpy, torch
import nemo.collections.asr as nemo_asr
print('NumPy:', numpy.__version__)
print('Torch:', torch.__version__)
print('NeMo ASR: OK')
"

# Read model name from config using system python3 (omegaconf is available there)
MODEL_NAME="$(python3 -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('${DIR}/config.yaml')
print(cfg.model.name)
")"

echo "=== NeMo FastConformer: pre-downloading model (${MODEL_NAME}) ==="
"${VENV}/bin/python3" -c "
import nemo.collections.asr as nemo_asr
model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(
    model_name='${MODEL_NAME}'
)
params = sum(p.numel() for p in model.parameters()) / 1e6
print(f'Model loaded: {params:.1f}M params')
print('Weights cached in HuggingFace cache.')
"

echo "=== NeMo FastConformer: setup complete ==="
