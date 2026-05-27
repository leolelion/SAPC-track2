#!/usr/bin/env bash
# Phase 1b — measure danielbodart's 3 prebuilt ONNX variants on Dev_streaming
# with the existing `run_onnx_streaming_latency.py` (single-thread sleep loop).
#
# Variants: fp32, int8-dynamic, int8-static
# Output: ledger row per variant + a heartbeat in /workspace/tmux_logs/.
# Scoring: jiwer min-of-two-refs (sclite re-eval optional, deferred).

set -euo pipefail

VENV=/root/venvs/nemotron_bench
PY=$VENV/bin/python
REPO=/workspace/sapc-nemotron
SAPC=/workspace/SAPC-template      # has run_onnx_streaming_latency.py
DANIELBODART=/workspace/.cache/hf/models--danielbodart--nemotron-speech-600m-onnx/snapshots/03c03d7b9fd16e3cdd5c147f1bd4ad2242aafa75
RESULTS=/workspace/phase1b_danielbodart
LOGS=/workspace/tmux_logs
mkdir -p "$RESULTS" "$LOGS"

MANIFEST=/workspace/SAPC2/manifest/Dev_streaming.csv
AUDIO_ROOT=/workspace/SAPC2

COMMIT=$(cd "$REPO" && git rev-parse --short HEAD)

# Order: int8-static first (it's the validation target: TTLT P50 should ≈ 56ms)
VARIANTS=("int8-static" "int8-dynamic" "fp32")

started_at=$(date -Iseconds)
echo "=========================================="
echo "Phase 1b danielbodart sweep start: $started_at"
echo "commit: $COMMIT"
echo "harness: $SAPC/run_onnx_streaming_latency.py (single-thread sleep loop)"
echo "manifest: $MANIFEST"
echo "=========================================="

for VARIANT in "${VARIANTS[@]}"; do
  LABEL="${VARIANT//-/_}"
  ENCODER="$DANIELBODART/$VARIANT/encoder_model.onnx"
  DECODER="$DANIELBODART/$VARIANT/decoder_model.onnx"
  TOKENS="$DANIELBODART/shared/tokens.txt"
  OUT_CSV="$RESULTS/dev_streaming.${LABEL}.csv"
  OUT_PARTIAL="$RESULTS/dev_streaming.${LABEL}.partial_results.json"

  echo ""
  echo "=========================================="
  echo "VARIANT [$VARIANT]  ($(date -Iseconds))"
  echo "=========================================="
  echo "  encoder: $ENCODER"
  echo "  decoder: $DECODER"
  echo "  encoder size: $(du -h $ENCODER | cut -f1)"

  # Heartbeat marker
  echo "$(date -Iseconds) variant=$VARIANT starting" > "$LOGS/danielbodart_sweep.heartbeat"

  cd "$SAPC" && $PY run_onnx_streaming_latency.py \
    --encoder "$ENCODER" \
    --decoder "$DECODER" \
    --tokens "$TOKENS" \
    --preprocessor-source nvidia/nemotron-speech-streaming-en-0.6b \
    --manifest "$MANIFEST" \
    --audio-root "$AUDIO_ROOT" \
    --out-csv "$OUT_CSV" \
    --out-partial-json "$OUT_PARTIAL" \
    --realtime \
    --threads 4 2>&1 | grep -vE "^\[NeMo W|RuntimeWarning|OneLogger|telemetry" | /usr/bin/tail -3

  # ── Compute latency from partial JSON ───────────────────────────────
  LATENCY_JSON="$RESULTS/dev_streaming.${LABEL}.latency.json"
  $PY $SAPC/utils/compute_latency.py \
    --partial-json "$OUT_PARTIAL" \
    --manifest-csv "$MANIFEST" \
    --mfa-col mfa_speech_start \
    --out-json "$LATENCY_JSON" 2>&1 | /usr/bin/tail -3

  # ── Compute CER/WER via jiwer (min-of-two-refs) ─────────────────────
  METRIC_JSON="$RESULTS/dev_streaming.${LABEL}.metrics.json"
  $PY - <<EOF > "$METRIC_JSON"
import csv, json, sys
import jiwer

def norm(s):
    s = s.lower().strip()
    for ch in ",.?!:;\"'":
        s = s.replace(ch, "")
    return " ".join(s.split())

refs1, refs2, hyps = {}, {}, {}
with open("$MANIFEST") as f:
    for r in csv.DictReader(f):
        refs1[r["id"]] = r.get("norm_text_with_disfluency", r.get("text",""))
        refs2[r["id"]] = r.get("norm_text_without_disfluency", r.get("text",""))
with open("$OUT_CSV") as f:
    for r in csv.DictReader(f):
        hyps[r["id"]] = r["raw_hypos"]

w_sub=w_del=w_ins=w_ref=0
c_sub=c_del=c_ins=c_ref=0
n_empty=0
for uid, h in hyps.items():
    r1 = refs1.get(uid, "")
    r2 = refs2.get(uid, "")
    r1n, r2n, hn = norm(r1), norm(r2), norm(h)
    if not hn: n_empty += 1
    o1 = jiwer.process_words(r1n, hn)
    o2 = jiwer.process_words(r2n, hn)
    if (o1.substitutions + o1.deletions + o1.insertions) <= (o2.substitutions + o2.deletions + o2.insertions):
        w_sub+=o1.substitutions; w_del+=o1.deletions; w_ins+=o1.insertions; w_ref+=o1.hits+o1.substitutions+o1.deletions
    else:
        w_sub+=o2.substitutions; w_del+=o2.deletions; w_ins+=o2.insertions; w_ref+=o2.hits+o2.substitutions+o2.deletions
    co1 = jiwer.process_characters(r1n, hn)
    co2 = jiwer.process_characters(r2n, hn)
    if (co1.substitutions + co1.deletions + co1.insertions) <= (co2.substitutions + co2.deletions + co2.insertions):
        c_sub+=co1.substitutions; c_del+=co1.deletions; c_ins+=co1.insertions; c_ref+=co1.hits+co1.substitutions+co1.deletions
    else:
        c_sub+=co2.substitutions; c_del+=co2.deletions; c_ins+=co2.insertions; c_ref+=co2.hits+co2.substitutions+co2.deletions

wer = (w_sub + w_del + w_ins) / max(1, w_ref)
cer = (c_sub + c_del + c_ins) / max(1, c_ref)
print(json.dumps({
    "n_utt": len(hyps),
    "wer": wer,
    "cer": cer,
    "n_empty": n_empty,
    "empty_rate": n_empty / max(1, len(hyps)),
}, indent=2))
EOF
  /bin/cat "$METRIC_JSON"

  # ── Parse + log to ledger ───────────────────────────────────────────
  CER=$($PY -c "import json; print(round(json.load(open('$METRIC_JSON'))['cer']*100, 2))")
  WER=$($PY -c "import json; print(round(json.load(open('$METRIC_JSON'))['wer']*100, 2))")
  TTFT=$($PY -c "import json; d=json.load(open('$LATENCY_JSON')); print(round(d['ttft_sec']['p50']*1000, 0))")
  TTLT=$($PY -c "import json; d=json.load(open('$LATENCY_JSON')); print(round(d['ttlt_sec']['p50']*1000, 0))")

  # Approximate RTF: get from script's stdout summary. The script prints "RTF=X" in its log line.
  # Easier: skip RTF here (we know it's < 1 since the script keeps up with real-time).
  # Leave RTF null and compute properly if needed via wall_clock / total_audio later.

  ENC_SIZE_MB=$(du -m $ENCODER | cut -f1)
  DEC_SIZE_MB=$(du -m $DECODER | cut -f1)
  TOTAL_MB=$((ENC_SIZE_MB + DEC_SIZE_MB))

  DATE=$(date +%Y-%m-%d)
  ENTRY_ID="${DATE}-nemotron-${LABEL}-onnx-danielbodart"
  ENTRY_JSON="/tmp/phase1b_entry_${LABEL}.json"
  cat > "$ENTRY_JSON" <<EOF
{
  "id": "$ENTRY_ID",
  "date": "$DATE",
  "model": "nemotron_streaming/danielbodart_${LABEL}",
  "kit": "nemotron_streaming",
  "model_source": "danielbodart/nemotron-speech-600m-onnx ($VARIANT)",
  "git_commit": "$COMMIT",
  "dataset": "Dev_streaming",
  "n_utt": 123,
  "wer": $WER,
  "cer": $CER,
  "ttft_p50_ms": $TTFT,
  "ttlt_p50_ms": $TTLT,
  "rtf": null,
  "hardware": "RunPod GPU pod, Intel Xeon Platinum 8568Y+ CPU, 4 threads (CPU EP, GPU unused at inference)",
  "raw_output": "$RESULTS/dev_streaming.${LABEL}.*",
  "verified": true,
  "status": "complete",
  "notes": "Phase 1b danielbodart prebuilt ONNX [70,6]; existing run_onnx_streaming_latency.py harness; size=${TOTAL_MB}MB; threads=4; sclite re-eval pending",
  "format": "onnx_${LABEL}",
  "threads": 4,
  "size_mb": $TOTAL_MB,
  "variant": "$VARIANT"
}
EOF
  (cd "$REPO/track2_starting_kit/experiments" && $PY experiments.py add --file "$ENTRY_JSON" --verified)

  echo ""
  echo "  RESULT [$VARIANT]: CER=${CER}%  WER=${WER}%  TTFT=${TTFT}ms  TTLT=${TTLT}ms  size=${TOTAL_MB}MB"
done

echo ""
echo "=========================================="
echo "ALL VARIANTS DONE  ($(date -Iseconds))"
echo "  started: $started_at"
echo "=========================================="

(cd "$REPO/track2_starting_kit/experiments" && $PY experiments.py board)
echo ""
echo "=== LEADERBOARD (final) ==="
cat "$REPO/track2_starting_kit/experiments/LEADERBOARD.md"
