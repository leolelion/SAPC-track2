#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Pod-side TIMING GATE for the threads=1 fix:
#   nemotron_encoder_sos_int8_submission_t1.zip
#
# Why this exists: the threads=4 zip TIMED OUT on Codabench (20 batch workers x
# 4 threads on a 24-CPU EPYC-Milan worker -> ~3x oversubscription -> Test1
# accuracy pass exceeded the 15000s budget -> killed -> no predict.csv).
# The fix sets SAPC2_THREADS default to 1. This gate proves the fixed zip
# finishes Test1 UNDER budget on the real worker *before* we spend a slot.
#
# It reproduces the exact Codabench batch config:
#   20 workers  x  1 thread/worker  x  pinned to 24 CPUs
# on a Dev subset, measures the 20-worker pool decode throughput, then projects
# the full Test1 wall (132399.2 audio-sec, from the failed run's preload line)
# and applies a per-core SPEED CORRECTION (this pod's matmul vs the Codabench
# worker's 142.11 ms) so the number is comparable to the 15000s budget.
#
# Usage (on the pod, after uploading the _t1 zip):
#   bash gate_timing_t1.sh [/path/to/..._t1.zip]
#
# Tunables via env: DATA, BENCH, MANIFEST, MAXROWS, CPUS, WORKERS, THREADS,
#   TEST1_AUDIO_SEC, CODABENCH_MATMUL_MS, BUDGET
# ---------------------------------------------------------------------------
set -euo pipefail

ZIP="${1:-/workspace/nemotron_encoder_sos_int8_submission_t1.zip}"
DATA="${DATA:-/workspace/SAPC2}"
BENCH="${BENCH:-/workspace/bench_nemotron_multiproc.py}"
MANIFEST="${MANIFEST:-$DATA/manifest/Dev.csv}"
MAXROWS="${MAXROWS:-1000}"                          # subset size for a stable rate
CPUS="${CPUS:-0-23}"                                # match the 24-CPU Codabench worker
WORKERS="${WORKERS:-20}"                            # Codabench batch worker count
THREADS="${THREADS:-1}"                             # the fix under test
TEST1_AUDIO_SEC="${TEST1_AUDIO_SEC:-132399.2}"      # ingestion log: "132399.2s audio"
CODABENCH_MATMUL_MS="${CODABENCH_MATMUL_MS:-142.11}" # failed run CPU_DIAGNOSTIC
BUDGET="${BUDGET:-15000}"

SUB=/dev/shm/sub_t1                                 # local extract -> avoids MooseFS load skew
OUT=/workspace/gate_timing_t1
mkdir -p "$OUT"
JSONL="$OUT/bench_w${WORKERS}_t${THREADS}_${MAXROWS}.jsonl"
LOG="$OUT/bench_w${WORKERS}_t${THREADS}_${MAXROWS}.log"

for f in "$ZIP" "$BENCH" "$MANIFEST"; do
  [ -e "$f" ] || { echo "FATAL: missing $f"; exit 1; }
done
command -v taskset >/dev/null 2>&1 || { echo "FATAL: taskset not found (needed to pin 24 CPUs)"; exit 1; }

echo "== 1/4 extract submission to /dev/shm (local disk, like Codabench) =="
rm -rf "$SUB"; mkdir -p "$SUB"
if command -v unzip >/dev/null 2>&1; then
  unzip -o -q "$ZIP" -d "$SUB"
else
  python3 -m zipfile -e "$ZIP" "$SUB"   # portable fallback (no unzip on minimal images)
fi
[ -f "$SUB/model.py" ] || { echo "FATAL: model.py missing in zip"; exit 1; }
echo -n "  threads default in zip: "; grep -m1 'SAPC2_THREADS' "$SUB/model.py"
echo

echo "== 1b/4 run the zip's own setup.sh (installs onnxruntime from bundled wheels, like Codabench) =="
( cd "$SUB" && bash setup.sh ) || { echo "FATAL: setup.sh failed (offline dep install)"; exit 1; }
echo

echo "== 2/4 calibrate this pod's per-core speed (matmul_1024_20iter_ms, pinned, 1 core) =="
POD_MATMUL_MS=$(OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c "${CPUS%%-*}" python3 - <<'PY'
import time, numpy as np
a=np.random.rand(1024,1024).astype('float32'); b=np.random.rand(1024,1024).astype('float32')
for _ in range(3): _=a@b
t=time.perf_counter()
for _ in range(20): _=a@b
print((time.perf_counter()-t)/20*1000.0)
PY
)
echo "  pod matmul_1024_20iter_ms = $POD_MATMUL_MS  (Codabench worker = $CODABENCH_MATMUL_MS)"
echo

echo "== 3/4 bench: ${WORKERS} workers x ${THREADS} thread, pinned to CPUs ${CPUS}, ${MAXROWS} utts =="
taskset -c "$CPUS" python3 "$BENCH" --submission-dir "$SUB" \
  --manifest-csv "$MANIFEST" --data-root "$DATA" --out-jsonl "$JSONL" \
  --workers "$WORKERS" --threads "$THREADS" --max-rows "$MAXROWS" 2>&1 | tee "$LOG"
echo

echo "== 4/4 projection + verdict =="
LOG="$LOG" POD_MATMUL_MS="$POD_MATMUL_MS" TEST1_AUDIO_SEC="$TEST1_AUDIO_SEC" \
CODABENCH_MATMUL_MS="$CODABENCH_MATMUL_MS" BUDGET="$BUDGET" python3 - <<'PY'
import json, os, sys
log=open(os.environ["LOG"]).read().splitlines()
line=[l for l in log if l.startswith("SUMMARY ")]
if not line:
    print("FATAL: no SUMMARY line in bench output (bench failed?)"); sys.exit(1)
summ=json.loads(line[-1][len("SUMMARY "):])
pod_mm=float(os.environ["POD_MATMUL_MS"]); cb_mm=float(os.environ["CODABENCH_MATMUL_MS"])
test1=float(os.environ["TEST1_AUDIO_SEC"]); budget=float(os.environ["BUDGET"])

audio=float(summ["total_audio_sec"]); wall=float(summ["total_wall_sec"])
loads=[float(w["load_sec"]) for w in summ.get("worker_loads",[]) if "load_sec" in w]
max_load=max(loads) if loads else 0.0
decode_wall=max(wall-max_load, 1e-6)          # decode window after all workers loaded
rate=audio/decode_wall                        # audio-sec processed per wall-sec by the pool
speed=cb_mm/pod_mm                            # >1 => Codabench core is slower than this pod

test1_decode_raw=test1/rate
test1_wall_raw=max_load+test1_decode_raw
test1_wall_cb=test1_wall_raw*speed            # conservative: scale whole wall by core-speed gap
margin=(budget-test1_wall_cb)/budget*100.0
submit_ceiling=0.85*budget                    # require >=15% headroom to SUBMIT

print(f"  subset      : {audio:.0f}s audio | wall {wall:.0f}s | max_load {max_load:.0f}s | decode_wall {decode_wall:.0f}s")
print(f"  pool rate   : {rate:.2f} audio-sec / wall-sec  (20 workers x 1 thread, {os.environ.get('CPUS','0-23')} CPUs)")
print(f"  speed corr  : pod {pod_mm:.1f}ms vs Codabench {cb_mm:.1f}ms -> x{speed:.3f}")
print(f"  Test1 wall  : {test1_wall_raw:.0f}s (this pod)  ->  {test1_wall_cb:.0f}s (Codabench-corrected)")
print(f"  budget      : {budget:.0f}s  | projected margin {margin:.0f}%  | SUBMIT needs < {submit_ceiling:.0f}s")
if test1_wall_cb < submit_ceiling:
    print("  VERDICT: ✅ SUBMIT  — comfortable margin, upload the _t1 zip.")
elif test1_wall_cb < budget:
    print("  VERDICT: ⚠️  MARGINAL — fits on paper but <15% headroom. Add headroom before submitting")
    print("           (e.g. raise CHUNK_NEW on the batch path, or enable full ORT graph opt). Do NOT submit yet.")
else:
    print("  VERDICT: ❌ DO NOT SUBMIT — projected to exceed the 15000s budget. Needs a real speedup.")
PY
echo
echo "artifacts: $LOG  $JSONL"
