#!/bin/bash
# =====================================================================
#  Moonshine Streaming — Dependency Setup
#
#  Creates a clean venv at /opt/moonshine_venv and installs the
#  useful-moonshine-onnx package (ONNX CPU backend) and omegaconf.
#
#  The venv path is hard-coded so model.py can inject it into
#  sys.path at runtime, isolating dependencies from the base conda
#  environment and avoiding CUDA / numpy .so conflicts.
# =====================================================================
set -euo pipefail

VENV=/opt/moonshine_venv
MODEL_NAME="$(python3 -c "
from omegaconf import OmegaConf
import os
cfg = OmegaConf.load(os.path.join(os.path.dirname('$0'), 'config.yaml'))
print(cfg.model.name)
" 2>/dev/null || echo "moonshine/tiny")"

echo "=== Moonshine: creating clean venv at ${VENV} ==="
python3 -m venv "${VENV}"

echo "=== Moonshine: upgrading pip ==="
"${VENV}/bin/pip" install --upgrade pip -q

echo "=== Moonshine: installing useful-moonshine-onnx and omegaconf ==="
"${VENV}/bin/pip" install --no-cache-dir \
    "useful-moonshine-onnx>=0.1.0" \
    "omegaconf>=2.3"

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
