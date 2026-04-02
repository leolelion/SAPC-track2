#!/bin/bash
# Export a fine-tuned Zipformer checkpoint to ONNX for sherpa-onnx deployment.
#
# Usage:
#   bash export_to_onnx.sh [standard|kroko] [epoch_num]
#
set -euo pipefail

VARIANT="${1:-standard}"
EPOCH="${2:-1}"
ICEFALL_DIR="/workspace/finetune/icefall"
EXP_DIR="/workspace/finetune/exp/${VARIANT}"
ONNX_DIR="/workspace/finetune/onnx/${VARIANT}"
WEIGHTS_DIR="/workspace/finetune/weights/${VARIANT}"

mkdir -p "${ONNX_DIR}"

echo "============================================"
echo "  Export ${VARIANT} epoch-${EPOCH} → ONNX"
echo "  Source:  ${EXP_DIR}/epoch-${EPOCH}.pt"
echo "  Output:  ${ONNX_DIR}"
echo "============================================"

if [ ! -f "${EXP_DIR}/epoch-${EPOCH}.pt" ]; then
    echo "ERROR: Checkpoint not found: ${EXP_DIR}/epoch-${EPOCH}.pt"
    ls "${EXP_DIR}/"*.pt 2>/dev/null || echo "(no .pt files in ${EXP_DIR})"
    exit 1
fi

cd "${ICEFALL_DIR}"

EXPORT_SCRIPT="egs/librispeech/ASR/zipformer/export-onnx-streaming.py"
if [ ! -f "${EXPORT_SCRIPT}" ]; then
    echo "Searching for export script ..."
    for candidate in \
        "egs/librispeech/ASR/zipformer/export_onnx_streaming.py" \
        "egs/librispeech/ASR/zipformer/export-onnx.py" \
        "egs/librispeech/ASR/zipformer/export.py"; do
        if [ -f "${candidate}" ]; then
            EXPORT_SCRIPT="${candidate}"
            break
        fi
    done
fi

if [ ! -f "${EXPORT_SCRIPT}" ]; then
    echo "ERROR: ONNX export script not found."
    find egs/librispeech/ASR -name '*onnx*' -o -name '*export*' 2>/dev/null
    exit 1
fi

echo "Using export script: ${EXPORT_SCRIPT}"

# ── Architecture params per variant ──
if [ "${VARIANT}" = "standard" ]; then
    ENCODER_DIM="192,256,384,512,384,256"
    FEEDFORWARD_DIM="512,768,1024,1536,1024,768"
    NUM_ENCODER_LAYERS="2,2,3,4,3,2"
elif [ "${VARIANT}" = "kroko" ]; then
    ENCODER_DIM="192,256,384,512,384,256"
    FEEDFORWARD_DIM="384,576,768,1152,768,576"
    NUM_ENCODER_LAYERS="2,2,2,2,2,2"
else
    echo "ERROR: Unknown variant '${VARIANT}'."
    exit 1
fi

python3 "${EXPORT_SCRIPT}" \
    --epoch "${EPOCH}" \
    --avg 1 \
    --use-averaged-model 0 \
    --exp-dir "${EXP_DIR}" \
    --tokens "${WEIGHTS_DIR}/data/lang_bpe_500/tokens.txt" \
    --causal 1 \
    --chunk-size 16 \
    --left-context-frames 128 \
    --encoder-dim "${ENCODER_DIM}" \
    --feedforward-dim "${FEEDFORWARD_DIM}" \
    --num-encoder-layers "${NUM_ENCODER_LAYERS}" \
    2>&1 | tee "${EXP_DIR}/export.log"

# ── Move ONNX files to output dir if export script put them in exp_dir ──
echo ""
echo "=== Collecting ONNX files ==="
for component in encoder decoder joiner; do
    # Export script may place files in exp_dir with epoch/chunk suffix
    for candidate in \
        "${EXP_DIR}/${component}-epoch-${EPOCH}-avg-1.onnx" \
        "${EXP_DIR}/${component}.onnx" \
        "${EXP_DIR}/${component}-chunk-16-left-128.onnx"; do
        if [ -f "${candidate}" ]; then
            cp "${candidate}" "${ONNX_DIR}/${component}.onnx"
            echo "  ${component}.onnx ← $(basename ${candidate})"
            break
        fi
    done
done

# ── Copy tokenizer files ──
for f in bpe.model tokens.txt; do
    src="${WEIGHTS_DIR}/data/lang_bpe_500/${f}"
    if [ -f "${src}" ]; then
        cp "${src}" "${ONNX_DIR}/"
        echo "  Copied ${f}"
    fi
done

echo ""
echo "============================================"
echo "  Export complete!"
echo ""
ls -lh "${ONNX_DIR}/" 2>/dev/null
echo ""
echo "  Validate with:"
echo "    python3 /workspace/SAPC-track2/track2_starting_kit/finetune_bundle/evaluate_finetuned.py \\"
echo "        --onnx-dir ${ONNX_DIR} \\"
echo "        --manifest-csv /workspace/SAPC2/manifest/Dev.csv \\"
echo "        --data-root /workspace/SAPC2 \\"
echo "        --out-csv ${EXP_DIR}/dev_hyp.csv"
echo "============================================"
