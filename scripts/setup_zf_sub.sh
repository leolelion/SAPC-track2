#!/usr/bin/env bash
# Build a zipformer-A1 submission dir on the pod from the int8 ONNX in onnx/a1,
# so local_decode.py can run the finetuned zipformer on the same diagnostic utts (exp A).
# Prereq: scp the A1 greedy model.py and tokens.txt into $SUBDIR first (see RUN_PLAN).
set -uo pipefail
SUBDIR=/workspace/finetune/zf_a1_sub
A1=/workspace/finetune/onnx/a1
mkdir -p "$SUBDIR/wheels"
ln -sf "$A1/encoder.int8.onnx" "$SUBDIR/encoder.onnx"
ln -sf "$A1/decoder.int8.onnx" "$SUBDIR/decoder.onnx"
ln -sf "$A1/joiner.int8.onnx"  "$SUBDIR/joiner.onnx"
echo "zf submission dir:"; ls -la "$SUBDIR"
[ -f "$SUBDIR/model.py" ]   || echo "WARN: scp the A1 model.py into $SUBDIR"
[ -f "$SUBDIR/tokens.txt" ] || echo "WARN: scp tokens.txt into $SUBDIR"
