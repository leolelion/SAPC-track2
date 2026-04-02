#!/bin/bash
# Scan the entire Zipformer training path for k2 usages.
# Run this BEFORE patching to understand the full scope of k2 dependency.
#
# Usage:
#   bash check_k2_usage.sh
#
ICEFALL_DIR="/workspace/finetune/icefall"
ZIPFORMER_DIR="${ICEFALL_DIR}/egs/librispeech/ASR/zipformer"

echo "============================================"
echo "  k2 usage scan in Zipformer training path"
echo "============================================"
echo ""

# 1. All k2 calls (non-comment lines) in training-relevant files
echo "=== k2. references in training scripts ==="
grep -n "k2\." \
    "${ZIPFORMER_DIR}/train.py" \
    "${ZIPFORMER_DIR}/scaling.py" \
    "${ZIPFORMER_DIR}/model.py" \
    "${ZIPFORMER_DIR}/zipformer.py" \
    "${ZIPFORMER_DIR}/optim.py" \
    2>/dev/null \
    | grep -v "^.*:#" \
    | grep -v "^\s*#"

echo ""
echo "=== k2 imports ==="
grep -rn "import k2\|from k2" \
    "${ZIPFORMER_DIR}/train.py" \
    "${ZIPFORMER_DIR}/scaling.py" \
    "${ZIPFORMER_DIR}/model.py" \
    "${ZIPFORMER_DIR}/zipformer.py" \
    2>/dev/null

echo ""
echo "=== rnnt_loss / transducer loss ==="
grep -rn "rnnt_loss\|pruned.*loss\|k2.*loss" \
    "${ZIPFORMER_DIR}/train.py" \
    "${ZIPFORMER_DIR}/model.py" \
    2>/dev/null | grep -v "^\s*#"

echo ""
echo "============================================"
echo "  Interpretation:"
echo "  - k2.swoosh_l/r        → CUDA-only kernel (segfaults with CPU wheel)"
echo "  - k2.rnnt_loss_pruned  → CUDA-only kernel (segfaults with CPU wheel)"
echo ""
echo "  BOTH require a CUDA k2 build. No pure-Python replacement for rnnt_loss."
echo ""
echo "  REQUIRED: bash build_k2_cuda.sh"
echo "  (builds k2 for sm_90 / H200; takes ~15-20 min)"
echo "============================================"
