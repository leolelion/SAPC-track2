#!/usr/bin/env bash
# =====================================================================
# Step 0 + Step 1 in one CPU-only pod session. NO GPU. NO retraining.
#
#   Step 0 — official-scorer error decomposition (scripts/error_decomposition.py)
#   Step 1 — blank-logit penalty sweep on the SHIPPED fp32 artifact
#
# Rationale, gates and stop condition: investigations/step01_runbook.md.
# Read that file before running this one. The pre-registered adopt/reject rule lives
# there and in the header below; do not renegotiate it after seeing the numbers.
#
#   STAGES="decode_grid score_grid decode_final score_final latency" \
#   REPO=/workspace/SAPC-template DATA=/workspace/data/SAP \
#   WORK=/workspace/parakeet_onnx ART=/workspace/artifacts/blank_penalty \
#   bash scripts/run_blank_penalty_sweep.sh
#
# PRE-REGISTERED DECISION RULE (Pareto ranking; write it before the run):
#   Baseline = the SHIPPED artifact at beta=0: Dev_diag severe CER 18.733%,
#   mean(TTFT p50, TTLT p50) 375.7 ms, ~4620 s projected Test wall-clock.
#   ADOPT a beta IFF, through the organizers' evaluate.sh on BOTH Dev slices:
#     A1. Dev_diag severe CER <= 18.43%  (>= 0.30 pt better; Dev->Test transfer was
#         +0.28 pt last time, so anything smaller is not distinguishable from transfer)
#     A2. Dev_clean2k CER does not regress by more than 0.20 pt vs the beta=0 run
#         measured in the SAME session (guards against tuning the severe tail only)
#     A3. mean(TTFT p50, TTLT p50) <= 420 ms
#     A4. projected Test wall-clock <= 7500 s (50% margin; more emissions = more
#         prediction-net calls = slower)
#     A5. the winning beta sits INSIDE a plateau — its two grid neighbours also beat
#         baseline. A single-point spike is noise, not a mechanism.
#   OR, as a pure corner extension (we are ranked on Pareto and own the low-latency
#   corner): mean latency improves by >= 50 ms with Dev_diag CER within +0.10 pt and
#   A2/A4/A5 holding.
#   Otherwise: keep beta=0.0 and report the curve. Never submit to test a hypothesis.
# =====================================================================
set -euo pipefail

REPO=${REPO:-/workspace/SAPC-template}
DATA=${DATA:?set DATA to the SAP data root (must contain manifest/)}
WORK=${WORK:-/workspace/parakeet_onnx}
ART=${ART:-/workspace/artifacts/blank_penalty}
EXTRACT=${EXTRACT:-$WORK/extract}          # the EXTRACTED shipped zip, not the build dir
GATEPY=${GATEPY:-$WORK/offlinevenv/bin/python}
ORT_VERSION=${ORT_VERSION:-1.27.0}
# Provisional. The `probe` stage measures the real blank-margin distribution and prints
# the grid to use; replace this with what it says. A synthetic-audio probe on 2026-07-30
# put blank margins at 6.7-15.9 (median 14.6), so the original 0-4 guess was entirely
# dead -- every beta below ~6 flipped nothing. Do not sweep before probing.
GRID=${GRID:-"0.0 2.0 4.0 6.0 8.0 10.0 12.0"}
FINAL=${FINAL:-}                            # betas to confirm on Dev_clean2k; set after the grid
PROBE_UTTS=${PROBE_UTTS:-60}
STAGES=${STAGES:-"probe"}
GRID_SPLIT=${GRID_SPLIT:-Dev_diag}
FINAL_SPLIT=${FINAL_SPLIT:-Dev_clean2k}

mkdir -p "$ART"
step() { echo -e "\n=========== $* ($(date -u +%H:%M:%S)) ==========="; }
has()  { [[ " $STAGES " == *" $1 "* ]]; }
tag()  { echo "b${1/./p}"; }                # 1.5 -> b1p5, usable in filenames

# --------------------------------------------------------------- preflight
# Same lesson as ORT_VERSION in run_parakeet_onnx_pod.sh: gate the runtime you ship.
[[ -x "$GATEPY" ]] || { echo "preflight: $GATEPY missing — extract the zip and build the offline venv first"; exit 1; }
GATE_ORT=$("$GATEPY" -c "import onnxruntime;print(onnxruntime.__version__)")
[[ "$GATE_ORT" == "$ORT_VERSION" ]] || {
  echo "preflight: interpreter has ort $GATE_ORT but the shipped bundle pins $ORT_VERSION — refusing"; exit 1; }
[[ -f "$EXTRACT/model.py" ]] || { echo "preflight: $EXTRACT/model.py missing"; exit 1; }
grep -q "SAPC2_BLANK_PENALTY" "$EXTRACT/model.py" || {
  echo "preflight: the extracted model.py has no blank-penalty hook — re-package before sweeping"; exit 1; }
echo "[preflight] $GATEPY ort=$GATE_ORT · artifact=$EXTRACT · grid='$GRID'"

# --------------------------------------------------------------- probe
# Minutes, and it can kill the whole sweep before it costs anything. Two questions:
#   (1) what beta range is even responsive (logits are unnormalised -- we have no prior);
#   (2) are the empties' blank decisions SEPARABLE from ordinary blanks? If the empties
#       are more confidently blank than everyone else, no constant shift can flip them
#       without flooding every other utterance with insertions -> report and STOP.
if has probe; then
  step "blank-margin probe on $GRID_SPLIT ($PROBE_UTTS utts)"
  "$GATEPY" "$REPO/scripts/probe_blank_margin.py" \
    --submission-dir "$EXTRACT" \
    --manifest-csv "$DATA/manifest/$GRID_SPLIT.csv" --data-root "$DATA" \
    --max-utts "$PROBE_UTTS" \
    --out-json "$ART/probe_margin.json" 2>&1 | tee "$ART/probe_margin.log"
  echo
  echo "STOP AND READ. Set GRID from the flip-fraction table (target 0.5%-20% flipped),"
  echo "and check the empty-vs-non-empty separation before spending the grid."
fi

# --------------------------------------------------------------- decode grid
# One process per beta, in parallel, each pinned to 1 intra-op thread. local_decode.py is
# single-process (AudioSender + Decoder threads), so K betas = K cores, not K*24.
if has decode_grid; then
  step "decode grid on $GRID_SPLIT: $GRID"
  pids=()
  for B in $GRID; do
    T=$(tag "$B")
    SAPC2_BLANK_PENALTY="$B" SAPC2_THREADS=1 "$GATEPY" \
      "$REPO/track2_starting_kit/local_decode.py" \
      --submission-dir "$EXTRACT" \
      --manifest-csv "$DATA/manifest/$GRID_SPLIT.csv" \
      --streaming-interval 0 --data-root "$DATA" \
      --out-csv "$ART/$GRID_SPLIT.$T.predict.csv" \
      --out-partial-json "$ART/$GRID_SPLIT.$T.partial.json" \
      > "$ART/decode_$GRID_SPLIT.$T.log" 2>&1 &
    pids+=($!)
    echo "  beta=$B -> $T (pid ${pids[-1]})"
  done
  fail=0
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  [[ $fail -eq 0 ]] || { echo "decode_grid: at least one decode FAILED — see $ART/decode_*.log"; exit 1; }
  # Prove the value we intended actually reached the wrapper. The gate-what-you-ship rule.
  grep -h "blank_penalty=" "$ART"/decode_$GRID_SPLIT.*.log | sort -u
fi

# --------------------------------------------------------------- score grid
# evaluate.sh writes into the SHARED $DATA/eval tree (hyp trn + sctk sgml), so scoring
# MUST be serial, and the SGML must be copied aside before the next beta overwrites it.
if has score_grid; then
  step "score grid on $GRID_SPLIT (serial; shared \$DATA/eval)"
  for B in $GRID; do
    T=$(tag "$B")
    ( cd "$REPO" && ./evaluate.sh --split "$GRID_SPLIT" \
        --hyp-csv "$ART/$GRID_SPLIT.$T.predict.csv" \
        --start_stage 2 --stop_stage 2 ) 2>&1 | tee "$ART/eval_$GRID_SPLIT.$T.log"
    for R in ref1 ref2; do
      cp "$DATA/eval/sctk/$GRID_SPLIT.$R.sgml" "$ART/$GRID_SPLIT.$T.$R.sgml"
    done
    "$GATEPY" "$REPO/scripts/error_decomposition.py" \
      --sgml-ref1 "$ART/$GRID_SPLIT.$T.ref1.sgml" \
      --sgml-ref2 "$ART/$GRID_SPLIT.$T.ref2.sgml" \
      --manifest-csv "$DATA/manifest/$GRID_SPLIT.csv" \
      --hyp-csv "$ART/$GRID_SPLIT.$T.predict.csv" \
      --out-json "$ART/decomp_$GRID_SPLIT.$T.json" \
      2>&1 | tee "$ART/decomp_$GRID_SPLIT.$T.log"
  done
  step "grid summary"
  "$GATEPY" - "$ART" "$GRID_SPLIT" <<'PY'
import glob, json, os, re, sys
art, split = sys.argv[1], sys.argv[2]
rows = []
for p in sorted(glob.glob(os.path.join(art, f"decomp_{split}.b*.json"))):
    d = json.load(open(p))
    t = d["totals"]; e = d["empties"]
    beta = float(re.search(r"\.b([0-9p]+)\.json$", p).group(1).replace("p", "."))
    rows.append((beta, t["cer"] * 100, t["wer"] * 100, e["n"], e["cer_points"],
                 t["char_ops"]["D"], t["char_ops"]["S"], t["char_ops"]["I"]))
rows.sort()
print(f"{'beta':>6}{'CER%':>9}{'WER%':>9}{'empty':>7}{'empt_pts':>10}{'charD':>9}{'charS':>9}{'charI':>9}")
for r in rows:
    print(f"{r[0]:>6.2f}{r[1]:>9.3f}{r[2]:>9.3f}{r[3]:>7d}{r[4]:>10.2f}{r[5]:>9d}{r[6]:>9d}{r[7]:>9d}")
base = next((r for r in rows if r[0] == 0.0), None)
if base:
    print(f"\ndelta vs beta=0 (CER {base[1]:.3f}%):")
    for r in rows:
        if r[0]:
            print(f"  beta={r[0]:<5.2f} dCER={r[1]-base[1]:+.3f}  dEmpty={r[3]-base[3]:+d}  dIns={r[7]-base[7]:+d}")
print("\nA5 reminder: adopt only a beta whose NEIGHBOURS also beat baseline. A lone spike is noise.")
PY
fi

# --------------------------------------------------------------- final confirm
# Run beta=0 here too: fp32's Dev_clean2k was never measured (the one open criterion from
# exp_parakeet_onnx_fp32), so this closes it at zero marginal cost.
if has decode_final; then
  [[ -n "$FINAL" ]] || { echo "decode_final: set FINAL='0.0 <best> <runner-up>' from the grid"; exit 1; }
  step "decode $FINAL_SPLIT for: $FINAL"
  pids=()
  for B in $FINAL; do
    T=$(tag "$B")
    SAPC2_BLANK_PENALTY="$B" SAPC2_THREADS=1 "$GATEPY" \
      "$REPO/track2_starting_kit/local_decode.py" \
      --submission-dir "$EXTRACT" \
      --manifest-csv "$DATA/manifest/$FINAL_SPLIT.csv" \
      --streaming-interval 0 --data-root "$DATA" \
      --out-csv "$ART/$FINAL_SPLIT.$T.predict.csv" \
      --out-partial-json "$ART/$FINAL_SPLIT.$T.partial.json" \
      > "$ART/decode_$FINAL_SPLIT.$T.log" 2>&1 &
    pids+=($!)
  done
  fail=0
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  [[ $fail -eq 0 ]] || { echo "decode_final: a decode FAILED"; exit 1; }
fi

if has score_final; then
  [[ -n "$FINAL" ]] || { echo "score_final: set FINAL"; exit 1; }
  step "score $FINAL_SPLIT (serial)"
  for B in $FINAL; do
    T=$(tag "$B")
    ( cd "$REPO" && ./evaluate.sh --split "$FINAL_SPLIT" \
        --hyp-csv "$ART/$FINAL_SPLIT.$T.predict.csv" \
        --start_stage 2 --stop_stage 2 ) 2>&1 | tee "$ART/eval_$FINAL_SPLIT.$T.log"
    for R in ref1 ref2; do
      cp "$DATA/eval/sctk/$FINAL_SPLIT.$R.sgml" "$ART/$FINAL_SPLIT.$T.$R.sgml"
    done
    "$GATEPY" "$REPO/scripts/error_decomposition.py" \
      --sgml-ref1 "$ART/$FINAL_SPLIT.$T.ref1.sgml" \
      --sgml-ref2 "$ART/$FINAL_SPLIT.$T.ref2.sgml" \
      --manifest-csv "$DATA/manifest/$FINAL_SPLIT.csv" \
      --out-json "$ART/decomp_$FINAL_SPLIT.$T.json" \
      2>&1 | tee "$ART/decomp_$FINAL_SPLIT.$T.log"
  done
fi

# --------------------------------------------------------------- latency
# A3/A4. The blank penalty changes WHEN the first token appears and how many
# prediction-net calls each chunk costs, so both the latency and the wall clock have to be
# re-measured, never inferred from the accuracy pass.
if has latency; then
  [[ -n "$FINAL" ]] || { echo "latency: set FINAL"; exit 1; }
  step "streaming pass (real 100 ms pacing) for: $FINAL"
  for B in $FINAL; do
    T=$(tag "$B")
    SAPC2_BLANK_PENALTY="$B" SAPC2_THREADS=1 "$GATEPY" \
      "$REPO/track2_starting_kit/local_decode.py" \
      --submission-dir "$EXTRACT" \
      --manifest-csv "$DATA/manifest/Dev_streaming.csv" \
      --streaming-manifest-csv "$DATA/manifest/Dev_streaming.csv" \
      --data-root "$DATA" \
      --out-csv "$ART/stream.$T.csv" --out-partial-json "$ART/stream.$T.json" \
      2>&1 | tee "$ART/stream.$T.log"
    ( cd "$REPO" && ./evaluate.sh --start_stage 3 --stop_stage 3 \
        --partial-json "$ART/stream.$T.json" \
        --manifest-csv "$DATA/manifest/Dev_streaming.csv" ) 2>&1 | tee "$ART/latency.$T.log"
  done
  echo "Check n_utts_with_timing == n_utts_total in each latency log: an all-empty TTFT"
  echo "fallback looks like a latency number but is a decode failure (the int8 lesson)."
fi

step "done — copy back $ART, then STOP THE POD"
