#!/bin/bash
# Fine-tune streaming Zipformer on SAPC2 dysarthric speech data.
#
# Usage:
#   bash run_finetune.sh [standard|kroko]
#
# Prerequisites (run once per pod):
#   1. bash setup_pod.sh
#   2. bash build_k2_cuda.sh                 ← builds k2 with CUDA (REQUIRED)
#      Reason: k2.rnnt_loss_pruned AND k2.swoosh_l/r are both CUDA-only kernels.
#      The CPU wheel causes segfaults in both the loss AND the activation functions.
#   3. python3 prepare_lhotse_manifests.py ...
#
set -euo pipefail

VARIANT="${1:-standard}"
ICEFALL_DIR="/workspace/finetune/icefall"
DATA_DIR="/workspace/finetune/data"
EXP_DIR="/workspace/finetune/exp/${VARIANT}"
WEIGHTS_DIR="/workspace/finetune/weights/${VARIANT}"

# ── Verify prerequisites ──
if [ ! -d "${ICEFALL_DIR}/.git" ]; then
    echo "ERROR: Icefall not cloned. Run setup_pod.sh first."
    exit 1
fi

for f in "${DATA_DIR}/sapc2_train_cuts.jsonl.gz" "${DATA_DIR}/sapc2_dev_cuts.jsonl.gz"; do
    if [ ! -f "${f}" ]; then
        echo "ERROR: Manifest not found: ${f}"
        echo "       Run prepare_lhotse_manifests.py first."
        exit 1
    fi
done

mkdir -p "${EXP_DIR}"

# ── Copy pre-trained checkpoint to epoch-0.pt ──
# train.py with --start-epoch 1 loads exp_dir/epoch-0.pt as the warm start.
PRETRAINED_CKPT="${WEIGHTS_DIR}/exp/epoch-30.pt"
EPOCH0="${EXP_DIR}/epoch-0.pt"
if [ ! -f "${EPOCH0}" ]; then
    if [ ! -f "${PRETRAINED_CKPT}" ]; then
        echo "ERROR: Pre-trained checkpoint not found: ${PRETRAINED_CKPT}"
        echo "       Run setup_pod.sh first."
        exit 1
    fi
    echo "Copying pre-trained checkpoint → ${EPOCH0} ..."
    cp "${PRETRAINED_CKPT}" "${EPOCH0}"
fi

# ── Verify k2 has CUDA support ──
python3 -c "
import k2
assert k2.with_cuda, (
    'k2 was built WITHOUT CUDA support. '
    'Run: bash build_k2_cuda.sh'
)
print(f'k2 CUDA: OK (version {k2.__version__})')
" || {
    echo ""
    echo "ERROR: k2 CUDA build required. Run: bash build_k2_cuda.sh"
    exit 1
}

# ── Symlink manifests to LibriSpeech names expected by the datamodule ──
cd "${DATA_DIR}"
for src_name dest_name in \
    "sapc2_train_cuts.jsonl.gz" "librispeech_cuts_train-all-shuf.jsonl.gz" \
    "sapc2_dev_cuts.jsonl.gz"   "librispeech_cuts_dev-clean.jsonl.gz" \
    "sapc2_dev_cuts.jsonl.gz"   "librispeech_cuts_dev-other.jsonl.gz"
do
    if [ ! -e "${dest_name}" ]; then
        ln -s "${src_name}" "${dest_name}"
        echo "  Symlinked: ${dest_name} → ${src_name}"
    fi
done
cd "${ICEFALL_DIR}"

echo ""
echo "============================================"
echo "  Fine-tuning ${VARIANT} Zipformer"
echo "  Exp dir: ${EXP_DIR}"
echo "  Data:    ${DATA_DIR}"
echo "============================================"
echo ""

# ── Architecture params per variant ──
if [ "${VARIANT}" = "standard" ]; then
    ENCODER_DIM="192,256,384,512,384,256"
    FEEDFORWARD_DIM="512,768,1024,1536,1024,768"
    NUM_ENCODER_LAYERS="2,2,3,4,3,2"
    EXTRA_ARCH_ARGS=""
elif [ "${VARIANT}" = "kroko" ]; then
    # Kroko is a 6-stage Zipformer2 — verify these dims by inspecting extracted weights
    ENCODER_DIM="192,256,384,512,384,256"
    FEEDFORWARD_DIM="384,576,768,1152,768,576"
    NUM_ENCODER_LAYERS="2,2,2,2,2,2"
    EXTRA_ARCH_ARGS="--encoder-dim ${ENCODER_DIM} --feedforward-dim ${FEEDFORWARD_DIM} --num-encoder-layers ${NUM_ENCODER_LAYERS}"
else
    echo "ERROR: Unknown variant '${VARIANT}'. Use 'standard' or 'kroko'."
    exit 1
fi

python3 egs/librispeech/ASR/zipformer/train.py \
    --world-size 1 \
    --num-epochs 1 \
    --start-epoch 1 \
    --exp-dir "${EXP_DIR}" \
    --bpe-model "${WEIGHTS_DIR}/data/lang_bpe_500/bpe.model" \
    --base-lr 0.0045 \
    --use-fp16 1 \
    --causal 1 \
    --chunk-size "16,32,64,-1" \
    --left-context-frames "64,128,256,-1" \
    --manifest-dir "${DATA_DIR}" \
    --max-duration 300 \
    --full-libri 1 \
    --on-the-fly-feats 1 \
    --enable-musan 0 \
    --num-workers 2 \
    ${EXTRA_ARCH_ARGS} \
    2>&1 | tee "${EXP_DIR}/train.log"

echo ""
echo "============================================"
echo "  Training complete!"
echo "  Checkpoint: ${EXP_DIR}/epoch-1.pt"
echo "  Log:        ${EXP_DIR}/train.log"
echo ""
echo "  Next: bash export_to_onnx.sh ${VARIANT} 1"
echo "============================================"
