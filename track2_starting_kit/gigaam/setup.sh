#!/bin/bash
# =====================================================================
#  GigaAM Streaming — Dependency Setup
#
#  Creates a clean venv at /opt/gigaam_venv and installs the gigaam
#  package from Sber (salute-developers/GigaAM on GitHub/PyPI).
#
#  ⚠️  Before investing benchmarking time, verify English language
#  support — GigaAM v2 was trained on Russian speech. If English
#  accuracy is poor (CER > 60% on clean English), skip this model.
#
#  GigaAM package sources:
#    PyPI:   pip install gigaam
#    GitHub: https://github.com/salute-developers/GigaAM
#    HF:     https://huggingface.co/salute-developers/GigaAM
# =====================================================================
set -euo pipefail

VENV=/opt/gigaam_venv
DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_NAME=$(python3 -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('${DIR}/config.yaml')
print(cfg.model.name)
" 2>/dev/null || echo "v2_rnnt")

echo "=== GigaAM: creating clean venv at ${VENV} ==="
python3 -m venv "${VENV}"

echo "=== GigaAM: upgrading pip ==="
"${VENV}/bin/pip" install --upgrade pip -q

echo "=== GigaAM: installing gigaam, omegaconf, and dependencies ==="
# gigaam requires torch; install CPU-only to avoid the 2GB+ CUDA build
"${VENV}/bin/pip" install --no-cache-dir \
    "gigaam" \
    "omegaconf>=2.3" \
    "torch" \
    "torchaudio"

echo "=== GigaAM: verifying installation ==="
"${VENV}/bin/python3" -c "
import gigaam
import omegaconf
print('omegaconf:', omegaconf.__version__)
print('gigaam: OK (version:', getattr(gigaam, '__version__', 'unknown'), ')')
"

echo "=== GigaAM: pre-downloading model weights (${MODEL_NAME}) ==="
"${VENV}/bin/python3" -c "
import gigaam
print('Downloading ${MODEL_NAME} weights from HuggingFace ...')
model = gigaam.load_model('${MODEL_NAME}')
print('GigaAM ${MODEL_NAME} loaded.')

# Quick English smoke test — warns if output is empty/garbage
import numpy as np
audio = np.random.default_rng(42).standard_normal(16000).astype(np.float32) * 0.01
try:
    result = model.transcribe(audio, sr=16000)
except TypeError:
    result = model.transcribe(audio)
print(f'Smoke test (1s noise): {result!r}')
print()
print('⚠️  IMPORTANT: Test with real English speech to verify language support.')
print('   If CER > 60% on clean English, skip this model.')
"

echo "=== GigaAM: setup complete ==="
echo "    Model:  ${MODEL_NAME}"
echo "    Venv:   ${VENV}"
echo "    Run:    python3 test_streaming.py --mock  (interface check)"
echo "            python3 test_streaming.py          (real model)"
echo "            python3 test_streaming.py --audio /path/to/english.wav"
