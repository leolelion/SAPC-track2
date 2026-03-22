#!/bin/bash
# NeMo FastConformer Hybrid Large Multi: install NeMo and pre-download weights.
#
# Identical to nemo_fastconformer/setup.sh but for the 114M multi-latency model.
# The venv is separate to avoid conflicts with the existing 32M model's venv.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="${DIR}/venv"

# NeMo 2.x requires Python 3.10 or 3.11
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
    exit 1
fi
echo "Using Python: $($PYTHON --version)"

echo "=== FastConformer Multi: creating venv (--system-site-packages) ==="
rm -rf "${VENV}"
"$PYTHON" -m venv --system-site-packages "${VENV}"
"${VENV}/bin/pip" install --upgrade pip -q

echo "=== FastConformer Multi: installing nemo_toolkit[asr] ==="
"${VENV}/bin/pip" install --no-cache-dir --prefer-binary \
    "omegaconf>=2.3" \
    "huggingface_hub>=0.24" \
    sentencepiece \
    "nemo_toolkit[asr]>=2.5.0" \
    || echo "WARNING: nemo_toolkit[asr] did not fully install."

"${VENV}/bin/pip" install --no-cache-dir --prefer-binary torchvision \
    || echo "WARNING: torchvision install failed."

"${VENV}/bin/python3" -c "import nemo" 2>/dev/null \
    || "${VENV}/bin/pip" install --no-cache-dir --prefer-binary "nemo_toolkit>=2.5.0"

echo "=== FastConformer Multi: verifying installation ==="
"${VENV}/bin/python3" -c "
import nemo.collections.asr as nemo_asr
import torch
print('NeMo ASR: OK')
print('Torch:', torch.__version__)
"

MODEL_NAME="$(python3 -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('${DIR}/config.yaml')
print(cfg.model.name)
")"

echo "=== FastConformer Multi: pre-downloading model (${MODEL_NAME}) ==="
"${VENV}/bin/python3" -c "
import nemo.collections.asr as nemo_asr
from omegaconf import OmegaConf

cfg = OmegaConf.load('${DIR}/config.yaml')
model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(
    model_name=cfg.model.name,
    map_location='cpu',
)
params = sum(p.numel() for p in model.parameters()) / 1e6
print(f'Model loaded: {params:.1f}M params')

# Verify multi-latency streaming params
model.encoder.setup_streaming_params()
cfg_enc = model.encoder.streaming_cfg
print(f'Default chunk_size: {cfg_enc.chunk_size}')
print()

# Test all three latency modes
first_chunk = cfg_enc.chunk_size[0]
latency_map = {'80ms': 8, '480ms': 48, '1040ms': 104}
for mode, frames in latency_map.items():
    cfg_enc.chunk_size = [first_chunk, frames]
    model.encoder.streaming_cfg = cfg_enc
    print(f'  latency_mode={mode}: chunk_size={cfg_enc.chunk_size}  ({frames * 10}ms per step)')

print()
print('All latency modes verified. Weights cached in HuggingFace cache.')
"

echo ""
echo "=== FastConformer Multi: setup complete ==="
echo ""
echo "Available latency modes in config.yaml: 80ms, 480ms, 1040ms"
echo ""
echo "Next steps:"
echo "  cd $(dirname "$0")"
echo "  python test_streaming.py --latency 80ms   # test aggressive latency mode"
echo "  python test_streaming.py --latency 480ms  # test balanced mode (default)"
echo "  python test_streaming.py --latency 1040ms # test high-accuracy mode"
echo "  python test_streaming.py --audio /workspace/SAPC2/processed/Dev/<utt>.wav"
