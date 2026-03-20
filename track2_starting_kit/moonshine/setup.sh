#!/bin/bash
# =====================================================================
#  Moonshine Streaming — Dependency Setup
#
#  Reads model.version from config.yaml and installs the appropriate
#  backend into a clean venv at <model_dir>/venv:
#
#    version: "v2" (default) → moonshine-voice  (C library + ONNX Runtime)
#    version: "v1"           → useful-moonshine-onnx (ONNX Python bindings)
#
#  model.py globs for venv/lib/python3.*/site-packages at runtime and
#  injects it into sys.path, isolating the package from the base conda
#  environment and avoiding CUDA / numpy .so conflicts.
#
#  Re-running this script is safe: the install step is skipped if the
#  expected package is already importable from the venv.
# =====================================================================
set -eu
set -o pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="${DIR}/venv"

# ---------------------------------------------------------------------------
# Read model.version and model.v2_arch from config.yaml.
# Try the base-env python3 with omegaconf first; fall back to grep+awk.
# ---------------------------------------------------------------------------
MODEL_VERSION="$(python3 -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('${DIR}/config.yaml')
print(OmegaConf.select(cfg, 'model.version', default='v2'))
" 2>/dev/null \
  || grep -m1 'version:' "${DIR}/config.yaml" | awk '{print $2}' | tr -d '"' 2>/dev/null \
  || echo "v2")"

V2_ARCH="$(python3 -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('${DIR}/config.yaml')
print(OmegaConf.select(cfg, 'model.v2_arch', default='tiny_streaming'))
" 2>/dev/null \
  || grep -m1 'v2_arch:' "${DIR}/config.yaml" | awk '{print $2}' | tr -d '"' 2>/dev/null \
  || echo "tiny_streaming")"

MODEL_NAME="$(python3 -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('${DIR}/config.yaml')
print(cfg.model.name)
" 2>/dev/null || echo "moonshine/tiny")"

echo "=== Moonshine: backend=${MODEL_VERSION}  v2_arch=${V2_ARCH}  v1_name=${MODEL_NAME} ==="

# ---------------------------------------------------------------------------
# Choose which package to check / install based on model.version
# ---------------------------------------------------------------------------
if [ "${MODEL_VERSION}" = "v2" ]; then
  IMPORT_CHECK="from moonshine_voice import Transcriber, ModelArch"
  PACKAGES='"moonshine-voice>=0.0.51" "omegaconf>=2.3"'
else
  IMPORT_CHECK="from moonshine_onnx import MoonshineOnnxModel"
  PACKAGES='"useful-moonshine-onnx>=0.1.0" "omegaconf>=2.3"'
fi

# ---------------------------------------------------------------------------
# Idempotency check: skip install if the expected package is already present
# ---------------------------------------------------------------------------
if [ -x "${VENV}/bin/python3" ] && "${VENV}/bin/python3" -c "${IMPORT_CHECK}" 2>/dev/null; then
  echo "=== Moonshine: venv already set up — skipping install ==="
else
  echo "=== Moonshine: creating clean venv at ${VENV} ==="
  # Remove stale venv (e.g. switching from v1 to v2 or vice versa)
  rm -rf "${VENV}"
  python3 -m venv "${VENV}"

  echo "=== Moonshine: upgrading pip ==="
  "${VENV}/bin/pip" install --upgrade pip -q

  echo "=== Moonshine: installing ${PACKAGES} ==="
  eval "${VENV}/bin/pip" install --no-cache-dir "${PACKAGES}"
fi

# ---------------------------------------------------------------------------
# Verify installation
# ---------------------------------------------------------------------------
echo "=== Moonshine: verifying installation ==="
if [ "${MODEL_VERSION}" = "v2" ]; then
  "${VENV}/bin/python3" -c "
from moonshine_voice import Transcriber, ModelArch, get_model_for_language
import moonshine_voice
import omegaconf
print('omegaconf:', omegaconf.__version__)
print('moonshine-voice: OK  (available archs:', [a.name for a in ModelArch], ')')
"
else
  "${VENV}/bin/python3" -c "
from moonshine_onnx import MoonshineOnnxModel
import omegaconf
print('omegaconf:', omegaconf.__version__)
print('moonshine-onnx: OK')
"
fi

# ---------------------------------------------------------------------------
# Pre-download model weights
# ---------------------------------------------------------------------------
echo "=== Moonshine: pre-downloading model weights ==="
if [ "${MODEL_VERSION}" = "v2" ]; then
  "${VENV}/bin/python3" -c "
from moonshine_voice import get_model_for_language, ModelArch
arch_map = {
    'tiny':             ModelArch.TINY,
    'base':             ModelArch.BASE,
    'tiny_streaming':   ModelArch.TINY_STREAMING,
    'base_streaming':   ModelArch.BASE_STREAMING,
    'small_streaming':  ModelArch.SMALL_STREAMING,
    'medium_streaming': ModelArch.MEDIUM_STREAMING,
}
arch = arch_map.get('${V2_ARCH}', ModelArch.TINY_STREAMING)
print(f'Downloading {arch.name} ...')
model_path, _ = get_model_for_language('en', arch)
print(f'Weights cached at {model_path}')
"
else
  "${VENV}/bin/python3" -c "
from moonshine_onnx import MoonshineOnnxModel
print('Downloading ${MODEL_NAME} ...')
MoonshineOnnxModel(model_name='${MODEL_NAME}')
print('Weights cached.')
"
fi

echo "=== Moonshine: setup complete (version=${MODEL_VERSION}) ==="
