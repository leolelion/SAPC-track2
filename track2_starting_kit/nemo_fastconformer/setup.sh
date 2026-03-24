#!/bin/bash
# NeMo FastConformer: install nemo_toolkit[asr] and pre-download model weights.
#
# Speed tiers (fastest to slowest):
#   1. wheels/ + model.nemo present  →  ~1-2 min  (offline install + local load)
#   2. wheels/ present only          →  ~3-5 min  (offline install + HF download)
#   3. model.nemo present only       →  ~12-20 min (PyPI install + local load)
#   4. Neither present               →  ~20-30 min (PyPI install + HF download)
#
# To build wheels/ for fast installs:
#   bash build_wheels.sh        (requires Docker, run once on Linux x86_64)
#
# To create model.nemo for local loading:
#   bash download_model.sh      (run after setup.sh completes)
#
# The venv is created with --system-site-packages to inherit the pre-installed
# PyTorch 2.5.0+cu124, torchaudio, and numpy from the submission container.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="${DIR}/venv"
WHEELS="${DIR}/wheels"
MODEL_FILE="${DIR}/model.nemo"

# NeMo 2.x requires Python 3.10 or 3.11.
PYTHON=""
for candidate in python3.11 python3.10; do
    if command -v "$candidate" &>/dev/null; then
        PYTHON="$candidate"
        break
    fi
done
if [ -z "$PYTHON" ]; then
    echo "ERROR: NeMo requires Python 3.10 or 3.11."
    exit 1
fi
echo "Using Python: $($PYTHON --version)"

# ── Step 1: Create venv ──────────────────────────────────────────────────────
# Always recreate to avoid stale state from previous submissions.
# --system-site-packages inherits system torch/torchaudio/numpy (avoids ABI issues).
echo "=== Creating venv (--system-site-packages) at ${VENV} ==="
rm -rf "${VENV}"
"$PYTHON" -m venv --system-site-packages "${VENV}"
"${VENV}/bin/pip" install --upgrade pip -q

# ── Step 2: Install NeMo ─────────────────────────────────────────────────────
if [ -d "${WHEELS}" ] && compgen -G "${WHEELS}/*.whl" > /dev/null 2>&1; then
    echo "=== Installing from pre-built wheels (offline, fast) ==="
    # --no-index: never go to PyPI; --find-links: use only our wheel directory.
    # Torch/numpy/etc. are already satisfied via --system-site-packages.
    "${VENV}/bin/pip" install \
        --no-index \
        --find-links="${WHEELS}" \
        omegaconf sentencepiece huggingface_hub \
        "nemo_toolkit[asr]" \
        || echo "WARNING: Some packages missing from wheels/. Re-run build_wheels.sh."
else
    echo "=== No pre-built wheels found — installing from PyPI (slow) ==="
    echo "    Tip: run build_wheels.sh once to pre-build wheels and save 10-15 min."
    "${VENV}/bin/pip" install --no-cache-dir --prefer-binary \
        "omegaconf>=2.3" \
        "huggingface_hub>=0.24" \
        sentencepiece \
        "nemo_toolkit[asr]>=2.5.0" \
        || echo "WARNING: nemo_toolkit[asr] did not fully install (some build deps may be missing)."
fi

# If the [asr] extras failed, ensure the base nemo_toolkit is present.
"${VENV}/bin/python3" -c "import nemo" 2>/dev/null \
    || "${VENV}/bin/pip" install --no-cache-dir --prefer-binary "nemo_toolkit>=2.5.0"

# ── Step 3: Verify NeMo ──────────────────────────────────────────────────────
echo "=== Verifying NeMo installation ==="
"${VENV}/bin/python3" -c "
import numpy, torch
import nemo.collections.asr as nemo_asr
print('NumPy:', numpy.__version__)
print('Torch:', torch.__version__)
print('NeMo ASR: OK')
"

# ── Step 4: Model weights ─────────────────────────────────────────────────────
# If model.nemo is bundled in the submission zip, skip the HuggingFace download.
if [ -f "${MODEL_FILE}" ]; then
    echo "=== Found model.nemo ($(du -sh "${MODEL_FILE}" | cut -f1)) — skipping download ==="
    # Quick sanity check that the file loads
    "${VENV}/bin/python3" -c "
import nemo.collections.asr as nemo_asr
model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(
    restore_path='${MODEL_FILE}',
    map_location='cpu',
)
params = sum(p.numel() for p in model.parameters()) / 1e6
print(f'model.nemo verified: {params:.1f}M params')
del model
"
else
    MODEL_NAME="$("${VENV}/bin/python3" -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('${DIR}/config.yaml')
print(cfg.model.name)
")"
    echo "=== Downloading model from HuggingFace: ${MODEL_NAME} ==="
    echo "    Tip: run download_model.sh after setup to bundle model.nemo and save ~5 min."
    "${VENV}/bin/python3" -c "
import nemo.collections.asr as nemo_asr
model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(
    model_name='${MODEL_NAME}',
    map_location='cpu',
)
params = sum(p.numel() for p in model.parameters()) / 1e6
print(f'Model downloaded: {params:.1f}M params')
print('Weights cached in HuggingFace cache (~/.cache/huggingface).')
del model
"
fi

echo "=== NeMo FastConformer: setup complete ==="
