#!/bin/bash
# NeMo FastConformer Hybrid Large Multi: install NeMo and pre-download weights.
#
# Identical speed-tier strategy to nemo_fastconformer/setup.sh.
# See that file's header for full documentation on wheels/ and model.nemo.
#
# Speed tiers:
#   1. wheels/ + model.nemo present  →  ~1-2 min
#   2. wheels/ present only          →  ~5-8 min  (HF download is larger: ~430MB)
#   3. model.nemo present only       →  ~12-20 min
#   4. Neither present               →  ~25-35 min
#
# To build wheels/ (shared with nemo_fastconformer/ — same deps):
#   bash ../nemo_fastconformer/build_wheels.sh    (or build_wheels.sh in this dir)
#
# To create model.nemo:
#   bash download_model.sh
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
echo "=== FastConformer Multi: creating venv (--system-site-packages) ==="
rm -rf "${VENV}"
"$PYTHON" -m venv --system-site-packages "${VENV}"
"${VENV}/bin/pip" install --upgrade pip -q

# ── Step 2: Install NeMo ─────────────────────────────────────────────────────
if [ -d "${WHEELS}" ] && compgen -G "${WHEELS}/*.whl" > /dev/null 2>&1; then
    echo "=== Installing from pre-built wheels (offline, fast) ==="
    "${VENV}/bin/pip" install \
        --no-index \
        --find-links="${WHEELS}" \
        omegaconf sentencepiece huggingface_hub \
        "nemo_toolkit[asr]" \
        || echo "WARNING: Some packages missing from wheels/. Re-run build_wheels.sh."
else
    echo "=== No pre-built wheels — installing from PyPI (slow) ==="
    echo "    Tip: run build_wheels.sh once to pre-build wheels and save 10-15 min."
    "${VENV}/bin/pip" install --no-cache-dir --prefer-binary \
        "omegaconf>=2.3" \
        "huggingface_hub>=0.24" \
        sentencepiece \
        "nemo_toolkit[asr]>=2.5.0" \
        || echo "WARNING: nemo_toolkit[asr] did not fully install."
fi

"${VENV}/bin/python3" -c "import nemo" 2>/dev/null \
    || "${VENV}/bin/pip" install --no-cache-dir --prefer-binary "nemo_toolkit>=2.5.0"

# ── Step 3: Verify NeMo ──────────────────────────────────────────────────────
echo "=== FastConformer Multi: verifying NeMo ==="
"${VENV}/bin/python3" -c "
import nemo.collections.asr as nemo_asr
import torch
print('NeMo ASR: OK | Torch:', torch.__version__)
"

# ── Step 4: Model weights ─────────────────────────────────────────────────────
if [ -f "${MODEL_FILE}" ]; then
    echo "=== Found model.nemo ($(du -sh "${MODEL_FILE}" | cut -f1)) — skipping download ==="
    "${VENV}/bin/python3" -c "
import nemo.collections.asr as nemo_asr
model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(
    restore_path='${MODEL_FILE}',
    map_location='cpu',
)
params = sum(p.numel() for p in model.parameters()) / 1e6
print(f'model.nemo verified: {params:.1f}M params')

# Verify streaming setup works
model.encoder.setup_streaming_params()
cfg = model.encoder.streaming_cfg
print(f'Default chunk_size: {cfg.chunk_size}')
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
from omegaconf import OmegaConf

cfg = OmegaConf.load('${DIR}/config.yaml')
model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(
    model_name=cfg.model.name,
    map_location='cpu',
)
params = sum(p.numel() for p in model.parameters()) / 1e6
print(f'Model downloaded: {params:.1f}M params')

model.encoder.setup_streaming_params()
cfg_enc = model.encoder.streaming_cfg
print(f'Default chunk_size: {cfg_enc.chunk_size}')
del model
"
fi

echo ""
echo "=== FastConformer Multi: setup complete ==="
echo ""
echo "Available latency modes in config.yaml: 80ms, 480ms, 1040ms"
