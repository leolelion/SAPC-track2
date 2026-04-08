#!/bin/bash
# Parakeet CTC (fine-tuned SAP): install NeMo ASR and verify model weights.
#
# The model weights (weights/final.nemo, ~4GB) must be present in the submission zip.
# This script installs NeMo and verifies the weights load correctly.
#
# The venv is created with --system-site-packages to inherit pre-installed
# PyTorch 2.5.0+cu124, torchaudio, and numpy from the submission container.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="${DIR}/venv"
MODEL_FILE="${DIR}/weights/final.nemo"

# NeMo 2.x requires Python 3.10 or 3.11
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

# ── Find the active python in the venv (handles nested layout) ───────────────
_find_venv_python() {
    # Check direct layout: venv/bin/python3
    if [ -f "${VENV}/bin/python3" ]; then
        echo "${VENV}/bin/python3"
        return
    fi
    # Check nested layout: venv/<name>/bin/python3
    for nested in "${VENV}"/*/bin/python3; do
        if [ -f "$nested" ]; then
            echo "$nested"
            return
        fi
    done
}

VENV_PYTHON="$(_find_venv_python)"

# ── Step 1: Create venv (skip if NeMo already importable) ────────────────────
if [ -n "$VENV_PYTHON" ] && "$VENV_PYTHON" -c "import nemo.collections.asr" &>/dev/null 2>&1; then
    echo "=== NeMo already installed at ${VENV_PYTHON}, skipping venv setup ==="
else
    echo "=== Creating venv (--system-site-packages) at ${VENV} ==="
    rm -rf "${VENV}" 2>/dev/null || true
    "$PYTHON" -m venv --system-site-packages "${VENV}"

    # Write a torch constraint file to prevent NeMo from upgrading torch
    SYS_TORCH="$("${VENV}/bin/python3" -c 'import torch; print(torch.__version__.split("+")[0])' 2>/dev/null || echo '')"
    CONSTRAINT_FILE=""
    if [ -n "$SYS_TORCH" ]; then
        CONSTRAINT_FILE="/tmp/torch_constraint_$$.txt"
        echo "torch==${SYS_TORCH}" > "$CONSTRAINT_FILE"
        echo "=== Pinning torch to system version: ${SYS_TORCH} ==="
    fi

    "${VENV}/bin/pip" install --upgrade pip -q

    # ── Step 2: Install NeMo ─────────────────────────────────────────────────
    echo "=== Installing NeMo ASR from PyPI ==="
    INSTALL_CMD=("${VENV}/bin/pip" install --no-cache-dir --prefer-binary
        "omegaconf>=2.3"
        "huggingface_hub>=0.24"
        sentencepiece
        "nemo_toolkit[asr]>=2.5.0")
    if [ -n "$CONSTRAINT_FILE" ]; then
        INSTALL_CMD+=(-c "$CONSTRAINT_FILE")
    fi
    "${INSTALL_CMD[@]}" || echo "WARNING: nemo_toolkit[asr] did not fully install."

    [ -n "$CONSTRAINT_FILE" ] && rm -f "$CONSTRAINT_FILE"

    # Ensure base nemo is available even if extras failed
    "${VENV}/bin/python3" -c "import nemo" 2>/dev/null \
        || "${VENV}/bin/pip" install --no-cache-dir --prefer-binary "nemo_toolkit>=2.5.0"

    VENV_PYTHON="$(_find_venv_python)"
fi

# ── Step 3: Verify NeMo ──────────────────────────────────────────────────────
echo "=== Verifying NeMo installation ==="
"$VENV_PYTHON" -c "
import numpy, torch
import nemo.collections.asr as nemo_asr
print('NumPy:', numpy.__version__)
print('Torch:', torch.__version__)
print('NeMo ASR: OK')
"

# ── Step 4: Verify model weights ─────────────────────────────────────────────
if [ -f "${MODEL_FILE}" ]; then
    echo "=== Found ${MODEL_FILE} ($(du -sh "${MODEL_FILE}" | cut -f1)) ==="
    echo "=== Verifying weights load correctly ==="
    "$VENV_PYTHON" -c "
import nemo.collections.asr as nemo_asr
model = nemo_asr.models.EncDecCTCModelBPE.restore_from(
    restore_path='${MODEL_FILE}',
    map_location='cpu',
)
params = sum(p.numel() for p in model.parameters()) / 1e6
print(f'Model verified: {params:.1f}M params, vocab={len(model.decoder.vocabulary)}')
del model
"
else
    echo "WARNING: ${MODEL_FILE} not found!"
    echo "Download it with:"
    echo "  huggingface-cli download scott-morgan-foundation/sapc2-track1-parakeet-ctc final.nemo --local-dir weights/"
    exit 1
fi

echo "=== Parakeet CTC: setup complete ==="
