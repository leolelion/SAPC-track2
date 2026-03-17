#!/usr/bin/env bash
# =====================================================================
#  Qwen3-ASR Streaming — Dependency Setup
#
#  Creates a clean venv at /opt/qwen3_asr_venv and installs the
#  qwen-asr package (transformers backend) and omegaconf.
#
#  The venv path is hard-coded so model.py can inject it into
#  sys.path at runtime, isolating dependencies from the base conda
#  environment and avoiding CUDA / numpy .so conflicts.
# =====================================================================
set -euo pipefail

VENV=/opt/qwen3_asr_venv
MODEL_NAME="Qwen/Qwen3-ASR-1.7B"

echo "=== Qwen3-ASR: creating clean venv at ${VENV} ==="
python3 -m venv "${VENV}"

echo "=== Qwen3-ASR: upgrading pip ==="
"${VENV}/bin/pip" install --upgrade pip -q

echo "=== Qwen3-ASR: installing qwen-asr (transformers backend) ==="
"${VENV}/bin/pip" install --no-cache-dir \
    "qwen-asr>=0.0.6" \
    "omegaconf>=2.3"

echo "=== Qwen3-ASR: verifying installation ==="
"${VENV}/bin/python3" -c "
import torch, transformers, omegaconf
from qwen_asr import Qwen3ASRModel
print('torch:', torch.__version__)
print('transformers:', transformers.__version__)
print('omegaconf:', omegaconf.__version__)
print('qwen-asr: OK')
"

echo "=== Qwen3-ASR: pre-downloading model weights (${MODEL_NAME}) ==="
"${VENV}/bin/python3" -c "
import torch
from qwen_asr import Qwen3ASRModel
print('Downloading ${MODEL_NAME} ...')
Qwen3ASRModel.from_pretrained(
    '${MODEL_NAME}',
    dtype=torch.float32,
    device_map='cpu',
    max_new_tokens=512,
)
print('Weights cached.')
"

echo "=== Qwen3-ASR: setup complete ==="
