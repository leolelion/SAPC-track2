#!/bin/bash
# =====================================================================
#  Moonshine Streaming — Dependency Setup
#
#  Creates a clean venv at <model_dir>/venv and installs the
#  useful-moonshine-onnx package (ONNX CPU backend) and omegaconf.
#
#  model.py globs for venv/lib/python3.*/site-packages at runtime and
#  injects it into sys.path, isolating moonshine-onnx from the base
#  conda environment and avoiding CUDA / numpy .so conflicts.
# =====================================================================
set -eu
set -o pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="${DIR}/venv"
MODEL_NAME="$(python3 -c "
from omegaconf import OmegaConf
import os
cfg = OmegaConf.load('${DIR}/config.yaml')
print(cfg.model.name)
" 2>/dev/null || echo "moonshine/tiny")"

if [ -x "${VENV}/bin/python3" ] && "${VENV}/bin/python3" -c "from moonshine_onnx import MoonshineOnnxModel" 2>/dev/null; then
  echo "=== Moonshine: venv already set up — skipping install ==="
else
  echo "=== Moonshine: creating clean venv at ${VENV} ==="
  python3 -m venv "${VENV}"

  echo "=== Moonshine: upgrading pip ==="
  "${VENV}/bin/pip" install --upgrade pip -q

  echo "=== Moonshine: installing useful-moonshine-onnx and omegaconf ==="
  "${VENV}/bin/pip" install --no-cache-dir \
      "useful-moonshine-onnx>=0.1.0" \
      "omegaconf>=2.3"
fi

echo "=== Moonshine: verifying installation ==="
"${VENV}/bin/python3" -c "
from moonshine_onnx import MoonshineOnnxModel
import omegaconf
print('omegaconf:', omegaconf.__version__)
print('moonshine-onnx: OK')
"

echo "=== Moonshine: pre-downloading model weights (${MODEL_NAME}) ==="
"${VENV}/bin/python3" -c "
from moonshine_onnx import MoonshineOnnxModel
print('Downloading ${MODEL_NAME} ...')
MoonshineOnnxModel(model_name='${MODEL_NAME}')
print('Weights cached.')
"

echo "=== Moonshine: setup complete ==="
