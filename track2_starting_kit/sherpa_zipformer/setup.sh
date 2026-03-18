#!/bin/bash
# =====================================================================
#  Sherpa-ONNX Zipformer — Dependency Setup
#
#  Installs sherpa-onnx and downloads ONNX model weights for the
#  selected variant (standard | kroko | small) from HuggingFace.
#
#  Model zoo reference:
#    https://github.com/k2-fsa/sherpa-onnx/blob/master/docs/source/
#            onnx/pretrained_models/online-transducer/
#            zipformer-transducer-models.rst
#
#  Run once before using model.py:
#    bash setup.sh
# =====================================================================
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"

# -----------------------------------------------------------------------
# Read variant from config.yaml
# -----------------------------------------------------------------------
VARIANT=$(python3 -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('${DIR}/config.yaml')
print(cfg.model.variant)
" 2>/dev/null || echo "standard")

echo "=== Sherpa-ONNX Zipformer setup (variant: ${VARIANT}) ==="

# -----------------------------------------------------------------------
# Stage 1: Install sherpa-onnx and omegaconf
# -----------------------------------------------------------------------
echo "--- Stage 1: pip install sherpa-onnx omegaconf ---"
if python3 -c "import sherpa_onnx, omegaconf" 2>/dev/null; then
  python3 -c "import sherpa_onnx; print('  Already installed — sherpa-onnx version:', sherpa_onnx.__version__)"
else
  pip install --no-cache-dir "sherpa-onnx>=1.10.0" "omegaconf>=2.3"
  python3 -c "import sherpa_onnx; print('sherpa-onnx version:', sherpa_onnx.__version__)"
fi

# -----------------------------------------------------------------------
# Stage 2: Download ONNX model weights
#
# Each variant maps to a HuggingFace repo. The repo contains:
#   encoder.onnx, decoder.onnx, joiner.onnx, tokens.txt
#
# ⚠️  Model repo IDs — verify against the sherpa-onnx model zoo if
#     downloads fail (URLs may change as new models are released):
#     https://github.com/k2-fsa/sherpa-onnx
# -----------------------------------------------------------------------
echo "--- Stage 2: Downloading weights for variant '${VARIANT}' ---"

WEIGHTS_DIR="${DIR}/weights/${VARIANT}"
mkdir -p "${WEIGHTS_DIR}"

if [ -f "${WEIGHTS_DIR}/encoder.onnx" ] && [ -f "${WEIGHTS_DIR}/decoder.onnx" ] && [ -f "${WEIGHTS_DIR}/joiner.onnx" ]; then
  echo "  Weights already present — skipping download."
  # skip to stage 3
else

# Map variant → HuggingFace repo and file list
case "${VARIANT}" in

  standard)
    # ~70M params — LibriSpeech streaming Zipformer (official baseline architecture)
    # Corresponds to: icefall-asr-librispeech-streaming-zipformer-2023-05-17
    HF_REPO="csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26"
    FILES=(
      "encoder-epoch-99-avg-1-chunk-16-left-128.onnx"
      "decoder-epoch-99-avg-1-chunk-16-left-128.onnx"
      "joiner-epoch-99-avg-1-chunk-16-left-128.onnx"
      "tokens.txt"
      "bpe.model"
    )
    ;;

  kroko)
    # Zipformer2 architecture — edge-optimised, newer than standard
    # Zipformer2 improves accuracy at similar or lower compute cost
    # ⚠️  Filenames not verified — check actual repo contents before running
    HF_REPO="csukuangfj/sherpa-onnx-streaming-zipformer2-en-2024-09-18"
    FILES=(
      "encoder-epoch-99-avg-1.onnx"
      "decoder-epoch-99-avg-1.onnx"
      "joiner-epoch-99-avg-1.onnx"
      "tokens.txt"
    )
    ;;

  small)
    # ~20M params — smallest Zipformer, targeting embedded/Cortex-A7 devices
    # Much lower RTF than standard; accuracy recovers with fine-tuning on SAP data
    # ⚠️  Filenames not verified — check actual repo contents before running
    HF_REPO="csukuangfj/sherpa-onnx-streaming-zipformer-small-en-2023-06-26"
    FILES=(
      "encoder-epoch-99-avg-1.onnx"
      "decoder-epoch-99-avg-1.onnx"
      "joiner-epoch-99-avg-1.onnx"
      "tokens.txt"
    )
    ;;

  *)
    echo "ERROR: unknown variant '${VARIANT}'. Expected: standard | kroko | small"
    exit 1
    ;;
esac

echo "  HuggingFace repo : ${HF_REPO}"
echo "  Local weights dir: ${WEIGHTS_DIR}"

python3 - <<PYEOF
from huggingface_hub import hf_hub_download
import os, shutil

repo = "${HF_REPO}"
weights_dir = "${WEIGHTS_DIR}"
files = """${FILES[*]}""".split()

# Download each ONNX file
for filename in files:
    local_path = hf_hub_download(
        repo_id=repo,
        filename=filename,
        local_dir=weights_dir,
    )
    print(f"  Downloaded: {filename}")

print("All weight files downloaded.")
PYEOF

# Rename files to canonical names expected by model.py config
(
  cd "${WEIGHTS_DIR}"
  case "${VARIANT}" in
    standard)
      mv "encoder-epoch-99-avg-1-chunk-16-left-128.onnx" encoder.onnx 2>/dev/null && echo "  Renamed encoder → encoder.onnx" || true
      mv "decoder-epoch-99-avg-1-chunk-16-left-128.onnx" decoder.onnx 2>/dev/null && echo "  Renamed decoder → decoder.onnx" || true
      mv "joiner-epoch-99-avg-1-chunk-16-left-128.onnx"  joiner.onnx  2>/dev/null && echo "  Renamed joiner  → joiner.onnx"  || true
      ;;
    *)
      # kroko/small: filenames not verified, fall back to glob rename
      for src in encoder-epoch-*.onnx; do [ -f "$src" ] && mv "$src" encoder.onnx && echo "  Renamed $src → encoder.onnx"; done || true
      for src in decoder-epoch-*.onnx; do [ -f "$src" ] && mv "$src" decoder.onnx && echo "  Renamed $src → decoder.onnx"; done || true
      for src in joiner-epoch-*.onnx;  do [ -f "$src" ] && mv "$src" joiner.onnx  && echo "  Renamed $src → joiner.onnx";  done || true
      ;;
  esac
)
fi  # end weights download block

# -----------------------------------------------------------------------
# Stage 3: Smoke test — build recognizer from the downloaded weights
# -----------------------------------------------------------------------
echo "--- Stage 3: Smoke test ---"
python3 - <<PYEOF
import sys
sys.path.insert(0, "${DIR}")
from omegaconf import OmegaConf
import sherpa_onnx
from pathlib import Path

cfg = OmegaConf.load("${DIR}/config.yaml")
w = Path("${WEIGHTS_DIR}")

rec = sherpa_onnx.OnlineRecognizer.from_transducer(
    encoder=str(w / cfg.model.encoder_file),
    decoder=str(w / cfg.model.decoder_file),
    joiner=str(w / cfg.model.joiner_file),
    tokens=str(w / cfg.model.tokens_file),
    num_threads=1,
    provider="cpu",
    sample_rate=cfg.audio.sample_rate,
    feature_dim=cfg.audio.feature_dim,
    decoding_method="greedy_search",
)
stream = rec.create_stream()

import numpy as np
# 0.5 s of silence — just testing the pipeline, not accuracy
chunk = np.zeros(8000, dtype=np.float32)
stream.accept_waveform(sample_rate=16000, waveform=chunk)
stream.input_finished()
while rec.is_ready(stream):
    rec.decode_stream(stream)
result = rec.get_result(stream)
print(f"Smoke test passed. Output on silence: {result!r}")
PYEOF

echo "=== Setup complete for variant '${VARIANT}' ==="
echo "    Weights: ${WEIGHTS_DIR}"
echo "    Run:     python3 test_streaming.py --mock  (interface check)"
echo "             python3 test_streaming.py          (real model)"
