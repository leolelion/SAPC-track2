#!/bin/bash
# Parakeet RNNT 120M: install NeMo and pre-download model weights.
#
# IMPORTANT: Run this script BEFORE running test_streaming.py.
# This script also performs the model investigation described in the plan
# (Step 1: confirm model name, class, and streaming capability).
#
# The RunPod competition image (xiuwenz2/sapc2-runtime) already has
# PyTorch 2.5.0+cu124, torchaudio, torchvision pre-installed. The venv
# inherits these via --system-site-packages.
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

echo "=== Parakeet RNNT: creating venv (--system-site-packages) ==="
rm -rf "${VENV}"
"$PYTHON" -m venv --system-site-packages "${VENV}"
"${VENV}/bin/pip" install --upgrade pip -q

echo "=== Parakeet RNNT: installing nemo_toolkit[asr] ==="
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

echo "=== Parakeet RNNT: verifying NeMo installation ==="
"${VENV}/bin/python3" -c "
import nemo.collections.asr as nemo_asr
import torch
print('NeMo ASR: OK')
print('Torch:', torch.__version__)
"

# ─── STEP 1: Model investigation ────────────────────────────────────────────
echo ""
echo "=== Parakeet RNNT: Step 1 — investigating available models ==="
"${VENV}/bin/python3" -c "
import nemo.collections.asr as nemo_asr

# Known Parakeet candidates to try (ordered by suitability for CPU streaming)
candidates = [
    ('nvidia/parakeet-rnnt-0.12b', 'EncDecRNNTBPEModel'),
    ('nvidia/stt_en_fastconformer_transducer_large', 'EncDecRNNTBPEModel'),
    ('nvidia/parakeet-ctc-0.6b', 'EncDecCTCModelBPE'),
]

for model_name, class_hint in candidates:
    print(f'\\nTrying: {model_name} ...')
    try:
        model_class = getattr(nemo_asr.models, class_hint, nemo_asr.models.ASRModel)
        model = model_class.from_pretrained(model_name=model_name, map_location='cpu')
        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f'  FOUND: {model_name}')
        print(f'  Class: {type(model).__name__}')
        print(f'  Params: {params:.1f}M')

        # Check streaming support
        has_streaming = (
            hasattr(model, 'encoder') and
            hasattr(model.encoder, 'setup_streaming_params')
        )
        print(f'  Streaming (setup_streaming_params): {has_streaming}')
        if has_streaming:
            try:
                model.encoder.setup_streaming_params()
                cfg = model.encoder.streaming_cfg
                print(f'  chunk_size: {cfg.chunk_size}')
                print(f'  look_ahead: {getattr(cfg, \"look_ahead\", \"N/A\")}')
            except Exception as e:
                print(f'  setup_streaming_params() error: {e}')

        # Check for CTC head
        has_ctc = hasattr(model, 'ctc_decoder') and model.ctc_decoder is not None
        print(f'  Has CTC head: {has_ctc}')
        print()
        print(f'  => RECOMMENDED config.yaml settings:')
        print(f'     model.name: \"{model_name}\"')
        print(f'     model.model_class: \"{type(model).__name__}\"')
        print(f'     model.partial_decoder: {\"ctc\" if has_ctc else \"rnnt\"}')
        break  # Use the first model that loads successfully
    except Exception as e:
        print(f'  NOT FOUND or error: {e}')
"

MODEL_NAME="$(python3 -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('${DIR}/config.yaml')
print(cfg.model.name)
")"

echo ""
echo "=== Parakeet RNNT: pre-downloading model (${MODEL_NAME}) ==="
"${VENV}/bin/python3" -c "
import nemo.collections.asr as nemo_asr
from omegaconf import OmegaConf
cfg = OmegaConf.load('${DIR}/config.yaml')

model_class = getattr(nemo_asr.models, cfg.model.model_class, nemo_asr.models.ASRModel)
model = model_class.from_pretrained(model_name=cfg.model.name, map_location='cpu')
params = sum(p.numel() for p in model.parameters()) / 1e6
print(f'Model {cfg.model.name} loaded: {params:.1f}M params')
print('Weights cached in HuggingFace cache.')
"

echo ""
echo "=== Parakeet RNNT: setup complete ==="
echo ""
echo "IMPORTANT: After reviewing the Step 1 output above, update config.yaml:"
echo "  model.name: <confirmed model name>"
echo "  model.model_class: <confirmed class>"
echo "  model.partial_decoder: ctc (if has_ctc=True) or rnnt"
echo ""
echo "Then run:"
echo "  python test_streaming.py --mock   # interface check"
echo "  python test_streaming.py          # real model test"
echo "  python test_streaming.py --audio /workspace/SAPC2/processed/Dev/<utt>.wav"
