#!/usr/bin/env bash
# =====================================================================
# Nemotron Streaming (ONNX, int8) — Environment Setup
#
# Installs ORT + NeMo (preprocessor only) into a local venv and downloads
# danielbodart's int8-static ONNX bundle from HuggingFace.
#
# Runtime target: xiuwenz2/sapc2-runtime:latest (PyTorch 2.5.0+cu124,
# Python 3.11).
# =====================================================================
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="${DIR}/venv"
WEIGHTS="${DIR}/weights"

HF_REPO="danielbodart/nemotron-speech-600m-onnx"
HF_REVISION="03c03d7b9fd16e3cdd5c147f1bd4ad2242aafa75"
VARIANT="int8-static"     # int8-static | int8-dynamic | fp32 | fp16

mkdir -p "$WEIGHTS"

# ── Find Python 3.10 or 3.11 (NeMo requirement) ─────────────────────
PYTHON=""
for candidate in python3.11 python3.10; do
    if command -v "$candidate" &>/dev/null; then PYTHON="$candidate"; break; fi
done
if [ -z "$PYTHON" ]; then
    echo "ERROR: NeMo requires Python 3.10 or 3.11."
    exit 1
fi
echo "Using Python: $($PYTHON --version)"

# ── Helper: find the active python inside the venv ──────────────────
_find_venv_python() {
    if [ -f "${VENV}/bin/python3" ]; then echo "${VENV}/bin/python3"; return 0; fi
    # Use nullglob so a non-matching pattern expands to nothing (under set -e).
    local _shopt; _shopt=$(shopt -p nullglob 2>/dev/null || true)
    shopt -s nullglob
    local nested
    for nested in "${VENV}"/*/bin/python3; do
        if [ -f "$nested" ]; then echo "$nested"; eval "$_shopt"; return 0; fi
    done
    eval "$_shopt"
    return 0
}

VENV_PYTHON="$(_find_venv_python || true)"

# ── Step 1: Create venv (skip if already populated) ────────────────
if [ -n "$VENV_PYTHON" ] && \
   "$VENV_PYTHON" -c "import onnxruntime, nemo.collections.asr" &>/dev/null 2>&1; then
    echo "=== venv already populated, skipping install ==="
else
    echo "=== Creating venv (--system-site-packages) at ${VENV} ==="
    rm -rf "${VENV}" 2>/dev/null || true
    "$PYTHON" -m venv --system-site-packages "${VENV}"

    SYS_TORCH="$("${VENV}/bin/python3" -c 'import torch; print(torch.__version__.split("+")[0])' 2>/dev/null || echo '')"
    CONSTRAINT=""
    if [ -n "$SYS_TORCH" ]; then
        CONSTRAINT="/tmp/torch_constraint_$$.txt"
        echo "torch==${SYS_TORCH}" > "$CONSTRAINT"
        echo "=== Pinning torch to system version: ${SYS_TORCH} ==="
    fi

    "${VENV}/bin/pip" install --upgrade pip -q

    echo "=== Installing ORT + NeMo (preprocessor only) ==="
    INSTALL_CMD=("${VENV}/bin/pip" install --no-cache-dir --prefer-binary
        "numpy<2"
        "omegaconf>=2.3"
        "huggingface_hub>=0.24"
        sentencepiece
        onnxruntime
        onnx
        "nemo_toolkit[asr]>=2.5.0,<2.6")
    [ -n "$CONSTRAINT" ] && INSTALL_CMD+=(-c "$CONSTRAINT")
    "${INSTALL_CMD[@]}" || echo "WARNING: nemo_toolkit[asr] install had issues; trying base"
    [ -n "$CONSTRAINT" ] && rm -f "$CONSTRAINT"

    "${VENV}/bin/python3" -c "import onnxruntime, nemo.collections.asr" \
        || { echo "ERROR: env didn't install correctly"; exit 1; }
    VENV_PYTHON="$(_find_venv_python)"
fi

# ── Step 2: Download danielbodart prebuilt ONNX (${VARIANT}) ─────────
NEED_DOWNLOAD=0
for f in encoder_model.onnx encoder_model.onnx.data \
         decoder_model.onnx decoder_model.onnx.data tokens.txt; do
    [ -f "${WEIGHTS}/${f}" ] || NEED_DOWNLOAD=1
done

if [ "$NEED_DOWNLOAD" -eq 0 ]; then
    echo "=== Weights already present in ${WEIGHTS}, skipping download ==="
else
    echo "=== Downloading ${HF_REPO} @ ${HF_REVISION} (${VARIANT}) ==="
    "$VENV_PYTHON" - <<PYEOF
from huggingface_hub import hf_hub_download
import shutil, os

repo = "${HF_REPO}"
rev = "${HF_REVISION}"
variant = "${VARIANT}"
weights = "${WEIGHTS}"

files = [
    (f"{variant}/encoder_model.onnx", "encoder_model.onnx"),
    (f"{variant}/encoder_model.onnx.data", "encoder_model.onnx.data"),
    (f"{variant}/decoder_model.onnx", "decoder_model.onnx"),
    (f"{variant}/decoder_model.onnx.data", "decoder_model.onnx.data"),
    ("shared/tokens.txt", "tokens.txt"),
]
for src, dst in files:
    print(f"  fetching {src} -> {dst}")
    p = hf_hub_download(repo_id=repo, revision=rev, filename=src)
    out = os.path.join(weights, dst)
    # Copy through symlink (HF cache uses symlinks) so onnx loader accepts it.
    shutil.copy(p, out)
print(f"All weights placed in {weights}")
PYEOF
fi

# ── Step 3: Verify model loads + smoke test on silence ──────────────
#
# Defensive: the smoke test is best-effort. If it fails (e.g., transient
# resource constraint), we still let setup.sh succeed so the ingestion
# stage gets a chance to run. The actual Model instantiation will be
# retried by the SAPC2 ingestion program with whatever resources it has.
echo "=== Verifying model loads (best-effort) ==="
set +e
"$VENV_PYTHON" -c "
import os, sys, numpy as np
sys.path.insert(0, '${DIR}')
from model import Model
m = Model()
m.set_partial_callback(lambda t: None)
m.reset()
chunk = np.zeros(1600, dtype=np.float32)
for _ in range(6):
    m.accept_chunk(chunk)
result = m.input_finished()
print(f'Smoke test passed. Output on silence: {result!r}')
print(f'compute_time_sec: {m.compute_time_sec:.3f}s')
"
SMOKE_RC=$?
set -e
if [ "$SMOKE_RC" -ne 0 ]; then
    echo "WARNING: smoke test exited $SMOKE_RC (e.g., OOM during install-time validation). "\
         "Continuing — ingestion will retry Model() with its own resources."
fi

echo "=== setup.sh complete ==="
