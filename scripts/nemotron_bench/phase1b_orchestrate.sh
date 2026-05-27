#!/usr/bin/env bash
# Phase 1b — orchestrate 4-config FP32 ONNX sweep on Dev_streaming.
#
# For each att_context_size:
#   1. Export ONNX (encoder + decoder_joint, cache-aware streaming)
#   2. Run sweep on Dev_streaming, 4 threads, real-time pacing
#   3. Append result row to results.jsonl via experiments.py
#   4. Verify the row landed
#   5. Delete ONNX bundle
#
# Sequence: [70,6] FIRST as smoke test vs PyTorch baseline.
#   If ONNX RTF >= 1.07 or ONNX TTFT P50 > 1609ms → STOP before other configs.
#
# Heartbeat to /workspace/tmux_logs/phase1b.heartbeat (updated per-utt by sweep).

set -euo pipefail

VENV=/root/venvs/nemotron_bench
PY=$VENV/bin/python
REPO=/workspace/sapc-nemotron
RESULTS=/workspace/phase1b_results
EXPORT_BASE=/root/nemotron_exports
LOGS=/workspace/tmux_logs
mkdir -p "$RESULTS" "$LOGS" "$EXPORT_BASE"

# Order: [70,6] first (smoke test), then descending complexity.
CONFIGS=("70 6" "70 13" "70 1" "70 0")
COMMIT=$(cd "$REPO" && git rev-parse --short HEAD)

# PyTorch baselines on this same set (from plan v2 table)
PT_RTF_70_6=1.07
PT_TTFT_70_6=1609

started_at=$(date -Iseconds)
echo "=========================================="
echo "Phase 1b sweep start: $started_at"
echo "commit: $COMMIT"
echo "venv: $VENV"
echo "manifest: /workspace/SAPC2/manifest/Dev_streaming.csv (123 utts)"
echo "threads: 4"
echo "=========================================="

for cfg in "${CONFIGS[@]}"; do
  LEFT=$(echo "$cfg" | cut -d' ' -f1)
  RIGHT=$(echo "$cfg" | cut -d' ' -f2)
  LABEL="${LEFT}_${RIGHT}"
  EXPORT_DIR="$EXPORT_BASE/fp32_$LABEL"
  OUT_PREFIX="$RESULTS/dev_streaming.fp32_$LABEL"

  echo ""
  echo "=========================================="
  echo "CONFIG [$LEFT,$RIGHT]  ($(date -Iseconds))"
  echo "=========================================="
  echo "disk before:"
  df -h /

  # ── Export ──────────────────────────────────────────────────────────
  echo "[$(date -Iseconds)] exporting to $EXPORT_DIR ..."
  rm -rf "$EXPORT_DIR"
  $PY "$REPO/scripts/nemotron_bench/phase1a_export_streaming.py" \
    --att-context $LEFT $RIGHT \
    --out-dir "$EXPORT_DIR" \
    --device cuda \
    --no-parity 2>&1 | grep -vE "^\[NeMo W|RuntimeWarning|OneLogger|telemetry"

  ENCODER=$(ls "$EXPORT_DIR"/encoder*.onnx | head -1)
  DECODER=$(ls "$EXPORT_DIR"/decoder*.onnx | head -1)
  SIZE_GB=$($PY -c "import os; t=sum(os.path.getsize(os.path.join('$EXPORT_DIR',f)) for f in os.listdir('$EXPORT_DIR')); print(round(t/1e9,2))")
  echo "  encoder: $ENCODER"
  echo "  decoder: $DECODER"
  echo "  bundle size: ${SIZE_GB} GB"

  # ── Sweep ───────────────────────────────────────────────────────────
  echo "[$(date -Iseconds)] sweeping ..."
  $PY "$REPO/scripts/nemotron_bench/phase1b_sweep.py" \
    --encoder "$ENCODER" --decoder "$DECODER" \
    --att-context $LEFT $RIGHT \
    --out-prefix "$OUT_PREFIX" \
    --threads 4 \
    --config-label "$LABEL" \
    --heartbeat "$LOGS/phase1b.heartbeat" 2>&1 | grep -vE "^\[NeMo W|RuntimeWarning|OneLogger|telemetry"

  # ── Parse summary ───────────────────────────────────────────────────
  SUM_JSON="$OUT_PREFIX.summary.json"
  WER=$($PY -c "import json; print(round(json.load(open('$SUM_JSON'))['wer']*100,2))")
  CER=$($PY -c "import json; print(round(json.load(open('$SUM_JSON'))['cer']*100,2))")
  TTFT=$($PY -c "import json; print(round(json.load(open('$SUM_JSON'))['ttft_p50_ms'],0))")
  TTLT=$($PY -c "import json; print(round(json.load(open('$SUM_JSON'))['ttlt_p50_ms'],0))")
  RTF=$($PY -c "import json; print(round(json.load(open('$SUM_JSON'))['rtf'],3))")
  GATE=$($PY -c "import json; print('PASS' if json.load(open('$SUM_JSON'))['rtf_gate_pass'] else 'FAIL')")

  echo ""
  echo "  RESULT [$LABEL]: CER=$CER%  WER=$WER%  TTFT=${TTFT}ms  TTLT=${TTLT}ms  RTF=$RTF  GATE=$GATE"

  # ── Append to ledger ────────────────────────────────────────────────
  DATE=$(date +%Y-%m-%d)
  ENTRY_JSON="/tmp/phase1b_entry_$LABEL.json"
  ENTRY_ID="${DATE}-nemotron-fp32-onnx-${LABEL}"
  cat > "$ENTRY_JSON" <<EOF
{
  "id": "$ENTRY_ID",
  "date": "$DATE",
  "model": "nemotron_streaming/fp32_onnx_${LABEL}",
  "kit": "nemotron_streaming",
  "model_source": "nvidia/nemotron-speech-streaming-en-0.6b",
  "git_commit": "$COMMIT",
  "dataset": "Dev_streaming",
  "n_utt": 123,
  "wer": $WER,
  "cer": $CER,
  "ttft_p50_ms": $TTFT,
  "ttlt_p50_ms": $TTLT,
  "rtf": $RTF,
  "hardware": "RunPod GPU pod, Intel Xeon Platinum 8568Y+ CPU, 4 threads (CPU EP, GPU unused at inference)",
  "raw_output": "$OUT_PREFIX",
  "verified": true,
  "status": "complete",
  "notes": "Phase 1b ONNX FP32 sweep; att_context_size=[$LEFT,$RIGHT]; size=${SIZE_GB}GB; threads=4; min-of-two-refs jiwer scoring (sclite re-eval pending)",
  "format": "onnx_fp32",
  "threads": 4,
  "size_gb": $SIZE_GB,
  "config_label": "$LABEL"
}
EOF
  (cd "$REPO/track2_starting_kit/experiments" && $PY experiments.py add --file "$ENTRY_JSON" --verified)

  # ── Verify the row landed before deleting the ONNX ──────────────────
  LEDGER="$REPO/track2_starting_kit/experiments/results.jsonl"
  if ! grep -q "\"$ENTRY_ID\"" "$LEDGER"; then
    echo "ERROR: ledger row missing for $LABEL — refusing to delete ONNX"
    exit 1
  fi
  echo "  ledger row appended ✓ ($ENTRY_ID)"

  # ── Smoke check after [70,6] ────────────────────────────────────────
  if [ "$LABEL" = "70_6" ]; then
    echo ""
    echo "=========================================="
    echo "SMOKE CHECK [70,6] vs PyTorch baseline"
    echo "=========================================="
    echo "  PyTorch [70,6]:  CER=18.2  WER=24.9  TTFT=${PT_TTFT_70_6}ms  RTF=$PT_RTF_70_6"
    echo "  ONNX FP32 [70,6]: CER=$CER   WER=$WER   TTFT=${TTFT}ms  RTF=$RTF"
    SANITY=$($PY -c "
import json
s=json.load(open('$SUM_JSON'))
rtf=s['rtf']; ttft=s['ttft_p50_ms']
ok = (rtf < $PT_RTF_70_6) and (ttft < $PT_TTFT_70_6)
print('OK' if ok else 'BAD')")
    if [ "$SANITY" = "BAD" ]; then
      echo ""
      echo "  SANITY FAIL: ONNX should beat PyTorch on BOTH RTF and TTFT."
      echo "  RTF threshold:  ONNX < $PT_RTF_70_6   (got $RTF)"
      echo "  TTFT threshold: ONNX < ${PT_TTFT_70_6}ms (got ${TTFT}ms)"
      echo "  Stopping. Investigate before running remaining configs."
      exit 1
    fi
    echo "  SANITY OK — proceeding to remaining configs"
  fi

  # ── Cleanup ONNX ────────────────────────────────────────────────────
  echo "  cleaning $EXPORT_DIR ..."
  rm -rf "$EXPORT_DIR"

  # ── Disk safety abort ───────────────────────────────────────────────
  FREE_GB=$(df -BG / | awk 'NR==2 {gsub("G",""); print $4}')
  echo "  overlay free: ${FREE_GB} GB"
  if [ "$FREE_GB" -lt 2 ]; then
    echo "ERROR: overlay free < 2 GB after cleanup. Aborting."
    exit 1
  fi
done

echo ""
echo "=========================================="
echo "ALL CONFIGS DONE  ($(date -Iseconds))"
echo "  started: $started_at"
echo "=========================================="
echo ""
echo "Regenerating LEADERBOARD.md ..."
(cd "$REPO/track2_starting_kit/experiments" && $PY experiments.py board)
echo ""
echo "=== LEADERBOARD ==="
cat "$REPO/track2_starting_kit/experiments/LEADERBOARD.md"
