#!/usr/bin/env bash
# export_onnx.sh — Export finetuned Zipformer checkpoint to ONNX for sherpa-onnx.
#
# Usage:
#   bash finetuning/export_onnx.sh --epoch 20 --avg 5
#   bash finetuning/export_onnx.sh --epoch 20 --avg 5 --variant kroko
#
# Outputs ONNX files to:
#   finetuning/exp/encoder-epoch-<E>-avg-<A>.onnx
#   finetuning/exp/decoder-epoch-<E>-avg-<A>.onnx
#   finetuning/exp/joiner-epoch-<E>-avg-<A>.onnx
#
# Then copies them to track2_starting_kit/sherpa_zipformer/weights/<variant>/
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ── Defaults ──────────────────────────────────────────────────────────────────
EPOCH=20
AVG=5
VARIANT="standard"
EXP_DIR="finetuning/exp"
BPE_MODEL="finetuning/pretrained/data/lang_bpe_500/bpe.model"

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --epoch)   EPOCH="$2";   shift 2 ;;
        --avg)     AVG="$2";     shift 2 ;;
        --variant) VARIANT="$2"; shift 2 ;;
        --exp-dir) EXP_DIR="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

EXPORT_SCRIPT="finetuning/icefall/egs/librispeech/ASR/zipformer/export-onnx-streaming.py"
WEIGHTS_DIR="track2_starting_kit/sherpa_zipformer/weights/${VARIANT}"

echo "=== Exporting epoch-${EPOCH} avg-${AVG} to ONNX ==="
echo "  exp_dir:  ${EXP_DIR}"
echo "  bpe:      ${BPE_MODEL}"
echo "  target:   ${WEIGHTS_DIR}"
echo ""

if [ ! -f "$EXPORT_SCRIPT" ]; then
    echo "ERROR: $EXPORT_SCRIPT not found."
    echo "Run bash finetuning/setup_finetune.sh first."
    exit 1
fi

if [ ! -d "$EXP_DIR" ]; then
    echo "ERROR: $EXP_DIR does not exist. Did training complete?"
    exit 1
fi

# ── Run icefall's export script ───────────────────────────────────────────────
python3 "$EXPORT_SCRIPT" \
    --epoch          "$EPOCH" \
    --avg            "$AVG" \
    --exp-dir        "$EXP_DIR" \
    --bpe-model      "$BPE_MODEL" \
    --causal         True \
    --chunk-size     16 \
    --left-context-frames 128

# ── Locate exported files ─────────────────────────────────────────────────────
ENCODER="${EXP_DIR}/encoder-epoch-${EPOCH}-avg-${AVG}.onnx"
DECODER="${EXP_DIR}/decoder-epoch-${EPOCH}-avg-${AVG}.onnx"
JOINER="${EXP_DIR}/joiner-epoch-${EPOCH}-avg-${AVG}.onnx"

for f in "$ENCODER" "$DECODER" "$JOINER"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Expected ONNX file not found: $f"
        exit 1
    fi
done

echo ""
echo "=== Copying to ${WEIGHTS_DIR} ==="
mkdir -p "$WEIGHTS_DIR"
cp "$ENCODER" "${WEIGHTS_DIR}/encoder.onnx"
cp "$DECODER" "${WEIGHTS_DIR}/decoder.onnx"
cp "$JOINER"  "${WEIGHTS_DIR}/joiner.onnx"

# Copy bpe.model if not already there (tokens.txt is already in the repo)
if [ ! -f "${WEIGHTS_DIR}/bpe.model" ]; then
    cp "$BPE_MODEL" "${WEIGHTS_DIR}/bpe.model"
fi

echo ""
echo "=== Verifying exported ONNX with sherpa-onnx ==="
python3 - <<PYEOF
import sys
try:
    import sherpa_onnx
except ImportError:
    print("  sherpa-onnx not installed — skipping verification")
    sys.exit(0)

import os
wd = "${WEIGHTS_DIR}"
recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
    encoder="${WEIGHTS_DIR}/encoder.onnx",
    decoder="${WEIGHTS_DIR}/decoder.onnx",
    joiner="${WEIGHTS_DIR}/joiner.onnx",
    tokens=os.path.join(wd, "tokens.txt"),
    num_threads=2,
    sample_rate=16000,
    feature_dim=80,
    enable_endpoint_detection=False,
    rule1_min_trailing_silence=2.4,
    rule2_min_trailing_silence=1.2,
    rule3_min_utterance_length=300,
    decoding_method="greedy_search",
    chunk_size=16,
    left_context=128,
)
print("  sherpa-onnx recognizer loaded successfully!")
PYEOF

echo ""
echo "=== Export complete ==="
echo ""
echo "ONNX files are in: ${WEIGHTS_DIR}"
echo ""
echo "Next steps:"
echo "  1. Validate:"
echo "     cd track2_starting_kit"
echo "     python3 local_decode.py \\"
echo "       --submission-dir ./sherpa_zipformer \\"
echo "       --manifest-csv /workspace/data/manifest/Dev.csv \\"
echo "       --data-root /workspace/data \\"
echo "       --out-csv ./Dev.predict.csv"
echo ""
echo "  2. Evaluate:"
echo "     cd .."
echo "     ./evaluate.sh --start_stage 2 --stop_stage 2 \\"
echo "       --split Dev --hyp-csv track2_starting_kit/Dev.predict.csv"
echo ""
echo "  3. Submit:"
echo "     cd track2_starting_kit/sherpa_zipformer"
echo "     zip -r submission.zip model.py config.yaml setup.sh weights/"
