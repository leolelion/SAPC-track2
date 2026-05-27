#!/usr/bin/env bash
# Phase 4 validation: drive /root/sub (fresh setup.sh install) through
# Dev_streaming via the standard SAPC2 ingestion design.
# Threads pinned to 4 via SAPC2_THREADS + OMP/MKL.
set -euo pipefail

KIT=/root/sub
OUT=/workspace/phase4/${1:-validate}
mkdir -p "$(dirname "$OUT")"

# Find the venv python (matches setup.sh logic)
VENV_PY=""
if [ -f "$KIT/venv/bin/python3" ]; then VENV_PY="$KIT/venv/bin/python3"; fi
for nested in "$KIT/venv"/*/bin/python3; do
    [ -f "$nested" ] && VENV_PY="$nested" && break
done
[ -z "$VENV_PY" ] && { echo "ERROR: venv python not found"; exit 1; }
echo "using venv python: $VENV_PY"

export SAPC2_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

LIMIT_ARG=""
if [ -n "${2:-}" ]; then LIMIT_ARG="--limit $2"; fi

cd "$KIT" && $VENV_PY /workspace/sapc-nemotron/scripts/nemotron_bench/phase3_runner.py \
    --submission-dir "$KIT" \
    --manifest /workspace/SAPC2/manifest/Dev_streaming.csv \
    --audio-root /workspace/SAPC2 \
    --out-prefix "$OUT" \
    $LIMIT_ARG 2>&1 | grep -vE "NeMo W|RuntimeWarning|OneLogger|telemetry"

echo "=== compute-RTF ==="
$VENV_PY -c "import json; d=json.load(open('$OUT.compute_rtf.json'))['summary']; print(json.dumps(d, indent=2))"

echo "=== sclite scoring (full set only) ==="
if [ -z "${2:-}" ]; then
    cd /workspace/SAPC-template && PATH="$(dirname "$VENV_PY"):$PATH" \
        bash evaluate.sh --start_stage 2 --stop_stage 2 --split Dev_streaming --hyp-csv "$OUT.csv" 2>&1 | /usr/bin/tail -5
fi
echo "=== TTFT/TTLT ==="
$VENV_PY /workspace/SAPC-template/utils/compute_latency.py \
    --partial-json "$OUT.partial_results.json" \
    --manifest-csv /workspace/SAPC2/manifest/Dev_streaming.csv \
    --mfa-col mfa_speech_start \
    --out-json "$OUT.latency.json" 2>&1 | /usr/bin/tail -3
/bin/cat "$OUT.latency.json"
