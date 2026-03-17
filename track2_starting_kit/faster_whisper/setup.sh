#!/bin/bash
# =====================================================================
#  Faster-Whisper Streaming — Dependency Setup
#
#  Creates a clean venv at /opt/faster_whisper_venv and installs the
#  faster-whisper package (CTranslate2 CPU backend) and omegaconf.
#
#  The venv path is hard-coded so model.py can inject it into
#  sys.path at runtime, isolating dependencies from the base conda
#  environment and avoiding CUDA / numpy .so conflicts.
# =====================================================================
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
VENV=/opt/faster_whisper_venv
MODEL_NAME=$(python3 -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('${DIR}/config.yaml')
print(cfg.model.name)
" 2>/dev/null || echo "small")

echo "=== Faster-Whisper: creating clean venv at ${VENV} ==="
python3 -m venv "${VENV}"

echo "=== Faster-Whisper: upgrading pip ==="
"${VENV}/bin/pip" install --upgrade pip -q

echo "=== Faster-Whisper: installing faster-whisper and omegaconf ==="
"${VENV}/bin/pip" install --no-cache-dir \
    "faster-whisper>=1.0.0" \
    "omegaconf>=2.3"

echo "=== Faster-Whisper: verifying installation ==="
"${VENV}/bin/python3" -c "
from faster_whisper import WhisperModel
import omegaconf
print('omegaconf:', omegaconf.__version__)
print('faster-whisper: OK')
"

echo "=== Faster-Whisper: pre-downloading model weights (${MODEL_NAME}) ==="
"${VENV}/bin/python3" -c "
from faster_whisper import WhisperModel
print('Downloading ${MODEL_NAME} ...')
WhisperModel('${MODEL_NAME}', device='cpu', compute_type='int8')
print('Weights cached.')
"

echo "=== Faster-Whisper: setup complete ==="
