#!/bin/bash
# =====================================================================
# parakeet_realtime (120M) Arm A, ONNX — Codabench setup
#
# PROVABLY OFFLINE: no PyPI, no HuggingFace, no NGC. onnxruntime (+ its deps) is
# installed from the wheels bundled in this zip; the model weights are bundled as
# real files under weights/. This is the recipe that every one of our submissions
# that actually scored used (A1 greedy, beam-4, Nemotron int8) — see memory
# `submission-offline-packaging`. The NeMo-backed sibling submission dir needs the
# network at setup time and has never scored.
#
# numpy + torch are already in the runtime image (torch is used only by the STFT in
# localmel.py). Do not add them here.
# =====================================================================
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"

if ! python3 -c "import onnxruntime" 2>/dev/null; then
  python3 -m pip install --no-index --find-links "$DIR/wheels" onnxruntime
fi

python3 - <<'PY'
import onnxruntime, numpy, torch
print("deps ready: ort", onnxruntime.__version__, "| numpy", numpy.__version__, "| torch", torch.__version__)
PY
