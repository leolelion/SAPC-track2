#!/bin/bash
# =====================================================================
# Nemotron Streaming 0.6B — Environment Setup
#
# Installs NeMo ASR into a local venv (inheriting system PyTorch 2.5.0)
# and pre-downloads model weights from HuggingFace with a pinned revision.
#
# Runtime: xiuwenz2/sapc2-runtime:latest (PyTorch 2.5.0+cu124, Python 3.11)
# =====================================================================
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="${DIR}/venv"

HF_MODEL="nvidia/nemotron-speech-streaming-en-0.6b"
HF_REVISION="ac0580bb7d3d6e39c4361db6afe28db9211793e4"

# ── Find Python 3.10 or 3.11 (NeMo requirement) ────────────────────
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

# ── Helper: find the active python inside the venv ──────────────────
_find_venv_python() {
    if [ -f "${VENV}/bin/python3" ]; then
        echo "${VENV}/bin/python3"
        return
    fi
    for nested in "${VENV}"/*/bin/python3; do
        if [ -f "$nested" ]; then
            echo "$nested"
            return
        fi
    done
}

VENV_PYTHON="$(_find_venv_python)"

# ── Step 1: Create venv and install NeMo (skip if already done) ─────
if [ -n "$VENV_PYTHON" ] && "$VENV_PYTHON" -c "import nemo.collections.asr" &>/dev/null 2>&1; then
    echo "=== NeMo already installed at ${VENV_PYTHON}, skipping venv setup ==="
else
    echo "=== Creating venv (--system-site-packages) at ${VENV} ==="
    rm -rf "${VENV}" 2>/dev/null || true
    "$PYTHON" -m venv --system-site-packages "${VENV}"

    # Pin torch to the system version so NeMo doesn't upgrade it
    SYS_TORCH="$("${VENV}/bin/python3" -c 'import torch; print(torch.__version__.split("+")[0])' 2>/dev/null || echo '')"
    CONSTRAINT_FILE=""
    if [ -n "$SYS_TORCH" ]; then
        CONSTRAINT_FILE="/tmp/torch_constraint_$$.txt"
        echo "torch==${SYS_TORCH}" > "$CONSTRAINT_FILE"
        echo "=== Pinning torch to system version: ${SYS_TORCH} ==="
    fi

    "${VENV}/bin/pip" install --upgrade pip -q

    echo "=== Installing NeMo ASR ==="
    INSTALL_CMD=("${VENV}/bin/pip" install --no-cache-dir --prefer-binary
        "omegaconf>=2.3"
        "huggingface_hub>=0.24"
        sentencepiece
        "numpy<2"
        "nemo_toolkit[asr]>=2.5.0")
    if [ -n "$CONSTRAINT_FILE" ]; then
        INSTALL_CMD+=(-c "$CONSTRAINT_FILE")
    fi
    "${INSTALL_CMD[@]}" || echo "WARNING: nemo_toolkit[asr] install had issues."

    [ -n "$CONSTRAINT_FILE" ] && rm -f "$CONSTRAINT_FILE"

    # Fallback: ensure base nemo is available
    "${VENV}/bin/python3" -c "import nemo" 2>/dev/null \
        || "${VENV}/bin/pip" install --no-cache-dir --prefer-binary "nemo_toolkit>=2.5.0"

    VENV_PYTHON="$(_find_venv_python)"
fi

# ── Step 2: Verify NeMo installation ────────────────────────────────
echo "=== Verifying NeMo ==="
"$VENV_PYTHON" -c "
import numpy, torch
import nemo.collections.asr as nemo_asr
print('NumPy:', numpy.__version__)
print('Torch:', torch.__version__)
print('NeMo ASR: OK')
"

# ── Step 3: Pre-download model weights (pinned revision) ────────────
echo "=== Downloading model: ${HF_MODEL} @ ${HF_REVISION} ==="
"$VENV_PYTHON" -c "
from huggingface_hub import snapshot_download
path = snapshot_download(
    '${HF_MODEL}',
    revision='${HF_REVISION}',
)
print(f'Cached at: {path}')
"

# ── Step 4: Verify model loads correctly ─────────────────────────────
echo "=== Verifying model loads ==="
"$VENV_PYTHON" -c "
import torch
torch.set_num_threads(1)
from nemo.collections.asr.models import ASRModel

model = ASRModel.from_pretrained('${HF_MODEL}', map_location='cpu')
model.encoder.set_default_att_context_size([70, 1])

params = sum(p.numel() for p in model.parameters()) / 1e6
cfg = model.encoder.streaming_cfg
print(f'Model: {params:.1f}M params')
print(f'att_context_size: {model.encoder.att_context_size}')
print(f'streaming_cfg.chunk_size: {cfg.chunk_size}')
print(f'streaming_cfg.shift_size: {cfg.shift_size}')
print(f'streaming_cfg.pre_encode_cache_size: {cfg.pre_encode_cache_size}')
print(f'streaming_cfg.drop_extra_pre_encoded: {cfg.drop_extra_pre_encoded}')
del model
import gc; gc.collect()
"

echo "=== setup.sh complete ==="
