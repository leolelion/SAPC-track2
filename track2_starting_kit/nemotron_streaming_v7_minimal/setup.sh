#!/bin/bash
# =====================================================================
# Nemotron Streaming (ONNX, int8) — Environment Setup (v7, minimal)
#
# Mirrors track2_starting_kit/streaming_zipformer/setup.sh shape (the
# one known-working Track 2 Codabench baseline). Three stages, no
# smoke test, no final verification block. Installs into base Python.
#
#   Stage 1: Detect environment
#   Stage 2: Install packages
#   Stage 3: Download model weights
#
# No NeMo, no librosa, no numba, no sentencepiece — mel preprocessor
# is reimplemented in pure torch + numpy inside model.py.
# =====================================================================
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"

# =============================================================
# Stage 1: Detect environment
# =============================================================
echo "=== Stage 1: Detect environment ==="
TORCH_VER=$(python3 -c "import torch; print(torch.__version__.split('+')[0])")
PY_TAG="cp$(python3 -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')")"
echo "PyTorch version: $TORCH_VER | Python tag: $PY_TAG"

# =============================================================
# Stage 2: Install packages
# =============================================================
echo "=== Stage 2: Install packages ==="
# Pin torch to whatever's already in the runtime image so nothing
# can replace it.
SYS_TORCH=$(python3 -c "import torch; print(torch.__version__)")
CONSTRAINT="/tmp/torch_pin_$$.txt"
echo "torch==${SYS_TORCH}" > "$CONSTRAINT"

pip install -q --no-cache-dir --prefer-binary -c "$CONSTRAINT" \
    omegaconf huggingface_hub onnxruntime onnx

rm -f "$CONSTRAINT"

# =============================================================
# Stage 3: Download model weights
# =============================================================
echo "=== Stage 3: Download model weights ==="
python3 -c "
from huggingface_hub import hf_hub_download
import shutil, os

repo = 'danielbodart/nemotron-speech-600m-onnx'
rev = '03c03d7b9fd16e3cdd5c147f1bd4ad2242aafa75'
variant = 'int8-static'
weights = '$DIR/weights'
os.makedirs(weights, exist_ok=True)

files = [
    (f'{variant}/encoder_model.onnx', 'encoder_model.onnx'),
    (f'{variant}/encoder_model.onnx.data', 'encoder_model.onnx.data'),
    (f'{variant}/decoder_model.onnx', 'decoder_model.onnx'),
    (f'{variant}/decoder_model.onnx.data', 'decoder_model.onnx.data'),
    ('shared/tokens.txt', 'tokens.txt'),
]
for src, dst in files:
    p = hf_hub_download(repo_id=repo, revision=rev, filename=src)
    shutil.copy(p, os.path.join(weights, dst))
"

echo "=== setup.sh complete ==="
