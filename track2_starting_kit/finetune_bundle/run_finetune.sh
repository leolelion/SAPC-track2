#!/bin/bash
# Fine-tune streaming Zipformer on SAPC2 dysarthric speech data.
#
# Usage:
#   bash run_finetune.sh [standard|kroko]
#
# IMPORTANT: The argument names below are TEMPLATES.
# After cloning Icefall on the pod, run:
#   python3 ${ICEFALL_DIR}/egs/librispeech/ASR/zipformer/train.py --help
# and adjust argument names accordingly.
#
set -euo pipefail

VARIANT="${1:-standard}"
ICEFALL_DIR="/workspace/finetune/icefall"
DATA_DIR="/workspace/finetune/data"
EXP_DIR="/workspace/finetune/exp/${VARIANT}"
WEIGHTS_DIR="/workspace/finetune/weights/${VARIANT}"

# ── Hyperparameters ──
# Bernard's Track 1 used lr=2e-5 for 1.7B params.
# For ~70M params: scale up. Original Zipformer peak LR is ~0.045.
# Conservative fine-tune starting point: 2e-4 (≈1/200 of original).
LR="2e-4"
MAX_DURATION=300    # seconds of audio per batch (adjust for GPU VRAM)
NUM_EPOCHS=1        # Bernard found 1 epoch sufficient
WARMUP_FRAC="0.02"  # 2% warmup steps

# ── Checkpoint ──
# Standard: epoch-30.pt from LibriSpeech streaming Zipformer training
PRETRAINED_CKPT="${WEIGHTS_DIR}/exp/epoch-30.pt"

if [ ! -f "${PRETRAINED_CKPT}" ]; then
    echo "ERROR: Pre-trained checkpoint not found: ${PRETRAINED_CKPT}"
    echo "       Run setup_pod.sh first to download it."
    exit 1
fi

mkdir -p "${EXP_DIR}"

echo "============================================"
echo "  Fine-tuning ${VARIANT} Zipformer"
echo "  LR:           ${LR}"
echo "  Max duration: ${MAX_DURATION}s/batch"
echo "  Epochs:       ${NUM_EPOCHS}"
echo "  Checkpoint:   ${PRETRAINED_CKPT}"
echo "  Exp dir:      ${EXP_DIR}"
echo "============================================"

cd "${ICEFALL_DIR}"

# ── Verify manifest files exist ──
for f in "${DATA_DIR}/sapc2_train_cuts.jsonl.gz" "${DATA_DIR}/sapc2_dev_cuts.jsonl.gz"; do
    if [ ! -f "${f}" ]; then
        echo "ERROR: Manifest not found: ${f}"
        echo "       Run prepare_lhotse_manifests.py first."
        exit 1
    fi
done

# ── Discover actual argument names from the training script ──
echo ""
echo "=== Training script help (first 80 lines) ==="
python3 egs/librispeech/ASR/zipformer/train.py --help 2>&1 | head -80 || true
echo "..."
echo ""
echo "=== Checking for --checkpoint / --pretrained-model / --start-epoch flags ==="
python3 egs/librispeech/ASR/zipformer/train.py --help 2>&1 | grep -iE "checkpoint|pretrain|finetune|resume|epoch" | head -20 || true
echo ""
echo "=== Checking for manifest / cuts flags ==="
python3 egs/librispeech/ASR/zipformer/train.py --help 2>&1 | grep -iE "manifest|cuts|lhotse|train.*cut|data" | head -20 || true
echo ""

# ── Launch training ──
# ⚠️  UPDATE ARGUMENT NAMES based on the --help output above.
# Common variants seen in Icefall scripts:
#   --checkpoint          (load pre-trained weights)
#   --pretrained-model    (alternative checkpoint flag)
#   --start-epoch 1       (epoch counter starts at 1 for fine-tune)
#   --lr-factor           (scales the configured LR schedule)
#   --manifest-dir        (directory containing CutSet files)
#   --train-cuts-filename / --valid-cuts-filename
#   --max-duration        (max seconds of audio per GPU batch)
#   --use-fp16            (half-precision training)
#
# Gradient accumulation may not be a native argument — if not, multiply
# max-duration by the desired accumulation steps instead.

echo "=== Starting training ==="
python3 egs/librispeech/ASR/zipformer/train.py \
    --world-size 1 \
    --num-epochs "${NUM_EPOCHS}" \
    --start-epoch 1 \
    --use-fp16 1 \
    --exp-dir "${EXP_DIR}" \
    --max-duration "${MAX_DURATION}" \
    --manifest-dir "${DATA_DIR}" \
    --checkpoint "${PRETRAINED_CKPT}" \
    2>&1 | tee "${EXP_DIR}/train.log"

# ⚠️  The above will likely FAIL on first run because argument names differ.
# The --help output printed above will tell you the correct flags.
# Edit this script with the real flag names, then re-run.

echo ""
echo "============================================"
echo "  Training complete!"
echo "  Checkpoints: ${EXP_DIR}"
echo "  Log:         ${EXP_DIR}/train.log"
echo ""
echo "  Next: bash export_to_onnx.sh ${VARIANT}"
echo "============================================"
