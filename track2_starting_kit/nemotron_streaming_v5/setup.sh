#!/usr/bin/env bash
# =====================================================================
# Nemotron Streaming (ONNX, int8) — Environment Setup (NO-VENV PROBE)
#
# H1b probe: this variant installs all Python deps into the base
# interpreter rather than into a submission-local venv. Goal is to
# test whether the venv mechanism is the cause of repeated Codabench
# ingestion failures (see scripts/audit/reference_submission_diff.md).
#
# Mirrors the upstream `streaming_zipformer` baseline's install
# pattern: pip install globally, pin torch to whatever's already in
# the runtime image.
#
# Runtime target: xiuwenz2/sapc2-runtime:latest (PyTorch 2.5.0+cu124,
# Python 3.11).
# =====================================================================
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
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

# ── Step 1: Install deps globally, pinning torch to the system version ─
# The runtime image ships PyTorch 2.5.0+cu124; we MUST NOT let NeMo's
# install upgrade torch out from under us. A constraint file pins it.

SKIP_INSTALL=0
if "$PYTHON" -c "import onnxruntime, nemo.collections.asr" &>/dev/null; then
    echo "=== Deps already importable in base Python, skipping install ==="
    SKIP_INSTALL=1
fi

if [ "$SKIP_INSTALL" -eq 0 ]; then
    SYS_TORCH="$("$PYTHON" -c 'import torch; print(torch.__version__)')"
    echo "=== System torch detected: ${SYS_TORCH} (will pin) ==="
    CONSTRAINT="/tmp/torch_pin_$$.txt"
    echo "torch==${SYS_TORCH}" > "$CONSTRAINT"

    "$PYTHON" -m pip install --upgrade pip -q

    echo "=== Installing deps into base Python (constraint: torch==${SYS_TORCH}) ==="
    "$PYTHON" -m pip install --no-cache-dir --prefer-binary \
        --upgrade-strategy only-if-needed \
        -c "$CONSTRAINT" \
        "numpy>=1.26,<3" \
        "omegaconf>=2.3" \
        "huggingface_hub>=0.24" \
        sentencepiece \
        onnxruntime \
        onnx \
        "nemo_toolkit[asr]>=2.5.0,<2.6"

    rm -f "$CONSTRAINT"
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
    "$PYTHON" - <<PYEOF
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
    shutil.copy(p, out)
print(f"All weights placed in {weights}")
PYEOF
fi

# ── Step 3: Environment state — log everything that matters ──────────
# Pipefail-fatal. We WANT this to fail loudly if anything is off so we
# don't ship a probe with hidden breakage.
echo "=== Final pip list (torch / onnx / nemo / numpy) ==="
"$PYTHON" -m pip list 2>&1 | grep -iE "torch|onnx|nemo|numpy" || true

echo "=== Final import check ==="
"$PYTHON" -c "import torch, onnxruntime; import nemo.collections.asr as _; print('torch', torch.__version__, '/ ort', onnxruntime.__version__, '/ nemo OK')"

# ── Step 4: Smoke test (NO set +e wrap — fail loud) ──────────────────
echo "=== Smoke test: Model() + 6 silence chunks + input_finished() ==="
"$PYTHON" -c "
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

echo "=== setup.sh complete ==="
