#!/usr/bin/env bash
# Run our newly quantized int8 k-quant encoder through the standard
# danielbodart-style harness on Dev_streaming. Decoder stays FP32
# (paper §6.2).
set -euo pipefail

DB=/workspace/.cache/hf/models--danielbodart--nemotron-speech-600m-onnx/snapshots/03c03d7b9fd16e3cdd5c147f1bd4ad2242aafa75
ENC=/workspace/phase2_quant/encoder_int8_kquant.onnx
DEC="$DB/fp32/decoder_model.onnx"
TOKENS="$DB/shared/tokens.txt"
OUT=/workspace/phase2_quant/dev_streaming.int8_kquant

echo "[$(date -Iseconds)] starting int8 k-quant sweep"
echo "  encoder: $ENC"
echo "  decoder: $DEC (FP32)"
echo "  tokens : $TOKENS"

cd /workspace/SAPC-template && \
/root/venvs/nemotron_bench/bin/python run_onnx_streaming_latency.py \
  --encoder "$ENC" \
  --decoder "$DEC" \
  --tokens "$TOKENS" \
  --preprocessor-source nvidia/nemotron-speech-streaming-en-0.6b \
  --manifest /workspace/SAPC2/manifest/Dev_streaming.csv \
  --audio-root /workspace/SAPC2 \
  --out-csv "${OUT}.csv" \
  --out-partial-json "${OUT}.partial.json" \
  --realtime --threads 4 2>&1 | grep -vE "NeMo W|RuntimeWarning|OneLogger|telemetry"

echo "[$(date -Iseconds)] sweep complete"
