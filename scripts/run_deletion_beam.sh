#!/usr/bin/env bash
# =====================================================================
# Step 2 — one CPU-only pod session. NO GPU. NO retraining.
#
#   S2a  localise the deletion mass (free: re-reads alignments we already have)
#   S2b  prove the beam patch is INERT at beam=1 (byte-identical to the shipped decode)
#   S2c  beam-search grid on the SHIPPED fp32 artifact
#
# Why beam and not another blank penalty: exp_s0_s1_probe measured 3780 char deletions
# against 1294 substitutions, and 3780 char-deletions / 760 word-deletions = 4.97 chars
# each against a 4.91-char mean reference word. The model drops ~13% of reference words
# WHOLE. Greedy takes blank at a frame and can never revisit it; a beam keeps the emitting
# continuation alive so a later frame can pay for it. The same probe showed a constant
# blank shift cannot separate empties from ordinary blanks (p10 margin 6.35 vs 2.61), so
# the crude version of this lever is already dead.
#
# Rationale, gates and stop condition: investigations/step02_runbook.md. Read it first.
#
#   STAGES="decomp_banked parity" REPO=/workspace/SAPC-step01 DATA=/workspace/SAPC2 \
#   EXTRACT=/workspace/onnx_ship/extract32 GATEPY=/workspace/onnx_ship/offvenv32/bin/python \
#   ART=/workspace/artifacts/step02 BANKED_CSV=/workspace/onnx_ship/art32/Dev_diag.fp32.predict.csv \
#   bash scripts/run_deletion_beam.sh
#
# PRE-REGISTERED DECISION RULE (Pareto ranking; written before the run):
#   Baseline = the SHIPPED artifact at beam=1: Dev_diag severe CER 18.733%,
#   mean(TTFT p50, TTLT p50) 375.7 ms on Dev, ~4620 s projected Test wall-clock.
#
#   HARD DISQUALIFIER (any branch): projected Test wall-clock > 12000 s. The 15000 s
#   budget has already cost us one submission to a timeout; 80% of it is the ceiling.
#
#   B1 — REPLACE the shipped point. All of:
#     B1a. Dev_diag severe CER <= 18.43%   (>= 0.30 pt; Dev->Test transfer was +0.28 pt)
#     B1b. Dev_clean2k CER does not regress > 0.20 pt vs the beam=1 run in the SAME session
#     B1c. mean(TTFT p50, TTLT p50) <= 420 ms
#     B1d. projected Test wall-clock <= 7500 s
#     B1e. the winner sits in a PLATEAU: the next-smaller beam also beats baseline.
#          A single-point spike is noise.
#
#   B2 — EXTEND the frontier with a SECOND submission (we are ranked on the Pareto
#   frontier, so a higher-latency/lower-CER point is worth owning even when it does not
#   dominate the greedy one). All of:
#     B2a. Dev_diag severe CER <= 17.73%   (>= 1.00 pt — must clearly buy its latency)
#     B2b. mean(TTFT p50, TTLT p50) <= 1000 ms
#     B2c. B1b and the hard disqualifier hold
#   NB B2b/B2a are a judgement call, not a measurement: they say what latency we are
#   willing to pay for a CER point. Q owns that number — change it BEFORE the run, not after.
#
#   Otherwise: keep beam=1 and report the curve. Never submit to test a hypothesis.
# =====================================================================
set -euo pipefail

REPO=${REPO:-/workspace/SAPC-step01}
DATA=${DATA:?set DATA to the SAP data root (must contain manifest/)}
ART=${ART:-/workspace/artifacts/step02}
EXTRACT=${EXTRACT:?set EXTRACT to the EXTRACTED shipped zip}
GATEPY=${GATEPY:?set GATEPY to the offline venv python that matches the shipped ort}
DECOMPPY=${DECOMPPY:-python3}               # scoring/analysis: the interpreter evaluate.sh uses
ORT_VERSION=${ORT_VERSION:-1.27.0}

# config := beam:beam_max_symbols:length_norm
# beam=1 is the shipped greedy path and MUST be in the grid: it is the in-session control
# for both CER and wall clock, and every delta below is measured against it.
GRID=${GRID:-"1:2:none 2:1:none 2:2:none 2:2:mean 4:2:none 4:2:mean"}
FINAL=${FINAL:-}                            # configs to confirm on Dev_clean2k; set after the grid
STAGES=${STAGES:-"decomp_banked parity"}
GRID_SPLIT=${GRID_SPLIT:-Dev_diag}
FINAL_SPLIT=${FINAL_SPLIT:-Dev_clean2k}
BASE_PROJ_S=${BASE_PROJ_S:-4620}            # the beam=1 projected Test wall-clock we ship against

mkdir -p "$ART"
step() { echo -e "\n=========== $* ($(date -u +%H:%M:%S)) ==========="; }
has()  { [[ " $STAGES " == *" $1 "* ]]; }
tag()  { local IFS=:; read -r b m n <<<"$1"; echo "b${b}m${m}${n}"; }
setenv() { local IFS=:; read -r b m n <<<"$1"; \
           echo "SAPC2_BEAM=$b SAPC2_BEAM_MAX_SYMBOLS=$m SAPC2_BEAM_LENGTH_NORM=$n"; }

# --------------------------------------------------------------- preflight
# Gate the runtime you ship (the int8/VNNI lesson: numerics follow the binary).
[[ -x "$GATEPY" ]] || { echo "preflight: $GATEPY missing"; exit 1; }
GATE_ORT=$("$GATEPY" -c "import onnxruntime;print(onnxruntime.__version__)")
[[ "$GATE_ORT" == "$ORT_VERSION" ]] || {
  echo "preflight: interpreter has ort $GATE_ORT but the bundle pins $ORT_VERSION — refusing"; exit 1; }
[[ -f "$EXTRACT/model.py" ]] || { echo "preflight: $EXTRACT/model.py missing"; exit 1; }
if has parity || has decode_grid || has decode_final || has latency; then
  grep -q "SAPC2_BEAM" "$EXTRACT/model.py" || {
    echo "preflight: extracted model.py has no beam hook — copy the patched wrapper in"; exit 1; }
fi
echo "[preflight] $GATEPY ort=$GATE_ORT · artifact=$EXTRACT · grid='$GRID'"

# --------------------------------------------------------------- S2a: localise the deletions
# No decode. Re-scores a banked hypothesis CSV to regenerate the SGML, then reads the
# alignments a second way. Minutes. This is what decides whether the next spend is an
# export change (M1), a decode change (M2) or a GPU session (M3).
if has decomp_banked; then
  : "${BANKED_CSV:?set BANKED_CSV to a banked hypothesis CSV}"
  BANKED_SPLIT=${BANKED_SPLIT:-Dev_diag}
  BANKED_TAG=${BANKED_TAG:-banked}
  step "S2a: rescore + localise deletions in $BANKED_CSV on $BANKED_SPLIT"
  ( cd "$REPO" && ./evaluate.sh --split "$BANKED_SPLIT" --hyp-csv "$BANKED_CSV" \
      --start_stage 2 --stop_stage 2 ) 2>&1 | tee "$ART/eval_${BANKED_SPLIT}.${BANKED_TAG}.log"
  for R in ref1 ref2; do
    cp "$DATA/eval/sctk/$BANKED_SPLIT.$R.sgml" "$ART/$BANKED_SPLIT.$BANKED_TAG.$R.sgml"
  done
  "$DECOMPPY" "$REPO/scripts/error_decomposition.py" \
    --sgml-ref1 "$ART/$BANKED_SPLIT.$BANKED_TAG.ref1.sgml" \
    --sgml-ref2 "$ART/$BANKED_SPLIT.$BANKED_TAG.ref2.sgml" \
    --manifest-csv "$DATA/manifest/$BANKED_SPLIT.csv" \
    --hyp-csv "$BANKED_CSV" \
    --out-json "$ART/decomp_${BANKED_SPLIT}.${BANKED_TAG}.json" \
    2>&1 | tee "$ART/decomp_${BANKED_SPLIT}.${BANKED_TAG}.log"
  echo
  echo "STOP AND READ the Step 2a/2b/2c tables before spending the grid:"
  echo "  position RISING + runs LONG  -> M1/M3, a beam may not be the right spend"
  echo "  position FLAT  + runs SHORT  -> M2, the beam is aimed correctly"
fi

# --------------------------------------------------------------- S2b: parity at beam=1
# The patch must be INERT at the shipped default. Decode a slice at beam=1 and require a
# byte-identical hypothesis against the banked CSV. If this fails, nothing below means
# anything, because the control is not the shipped model.
if has parity; then
  : "${BANKED_CSV:?set BANKED_CSV for the parity comparison}"
  PARITY_N=${PARITY_N:-40}
  step "S2b: beam=1 parity vs the banked shipped decode (first $PARITY_N utts)"
  head -n $((PARITY_N + 1)) "$DATA/manifest/$GRID_SPLIT.csv" > "$ART/parity_manifest.csv"
  SAPC2_THREADS=1 "$GATEPY" "$REPO/track2_starting_kit/local_decode.py" \
    --submission-dir "$EXTRACT" \
    --manifest-csv "$ART/parity_manifest.csv" \
    --streaming-interval 0 --data-root "$DATA" \
    --out-csv "$ART/parity.predict.csv" \
    --out-partial-json "$ART/parity.partial.json" 2>&1 | tee "$ART/parity.log"
  "$DECOMPPY" - "$ART/parity.predict.csv" "$BANKED_CSV" <<'PY'
import csv, sys
new = {r["id"]: r["raw_hypos"] for r in csv.DictReader(open(sys.argv[1], newline=""))}
old = {r["id"]: r["raw_hypos"] for r in csv.DictReader(open(sys.argv[2], newline=""))}
shared = [k for k in new if k in old]
if not shared:
    sys.exit("parity: no shared ids — wrong banked CSV for this split")
diff = [k for k in shared if new[k] != old[k]]
print(f"parity: {len(shared)} shared utterances, {len(diff)} differ")
if diff:
    print("FAILED — the beam patch is NOT inert at beam=1. Ids:", diff[:10])
    sys.exit(1)
print("PASS — beam=1 reproduces the shipped decode byte-for-byte")
PY
fi

# --------------------------------------------------------------- S2c: decode grid
# One process per config, in parallel, each pinned to 1 intra-op thread. Wall clock is
# recorded per config because the beam multiplies decoder calls and the 15000 s budget is
# a live constraint, not a formality.
if has decode_grid; then
  step "decode grid on $GRID_SPLIT: $GRID"
  pids=(); names=()
  for C in $GRID; do
    T=$(tag "$C"); IFS=: read -r B M N <<<"$C"
    ( start=$(date +%s)
      SAPC2_BEAM="$B" SAPC2_BEAM_MAX_SYMBOLS="$M" SAPC2_BEAM_LENGTH_NORM="$N" \
      SAPC2_THREADS=1 "$GATEPY" "$REPO/track2_starting_kit/local_decode.py" \
        --submission-dir "$EXTRACT" \
        --manifest-csv "$DATA/manifest/$GRID_SPLIT.csv" \
        --streaming-interval 0 --data-root "$DATA" \
        --out-csv "$ART/$GRID_SPLIT.$T.predict.csv" \
        --out-partial-json "$ART/$GRID_SPLIT.$T.partial.json" \
        > "$ART/decode_$GRID_SPLIT.$T.log" 2>&1
      echo $(( $(date +%s) - start )) > "$ART/walltime_$GRID_SPLIT.$T.txt" ) &
    pids+=($!); names+=("$T")
    echo "  $C -> $T (pid ${pids[-1]})"
  done
  fail=0
  for i in "${!pids[@]}"; do wait "${pids[$i]}" || { echo "FAILED: ${names[$i]}"; fail=1; }; done
  [[ $fail -eq 0 ]] || { echo "decode_grid: a decode FAILED — see $ART/decode_*.log"; exit 1; }
  # Prove the values we intended actually reached the wrapper.
  grep -h "beam=" "$ART"/decode_$GRID_SPLIT.*.log | sort -u
  echo "NOTE: these ran in PARALLEL on shared cores — wall times are comparable to each"
  echo "other but are NOT a clean projection. The timing stage re-runs the winner alone."
fi

# --------------------------------------------------------------- score grid
# evaluate.sh writes into the SHARED $DATA/eval tree, so scoring MUST be serial and each
# SGML copied aside before the next config overwrites it.
if has score_grid; then
  step "score grid on $GRID_SPLIT (serial; shared \$DATA/eval)"
  for C in $GRID; do
    T=$(tag "$C")
    ( cd "$REPO" && ./evaluate.sh --split "$GRID_SPLIT" \
        --hyp-csv "$ART/$GRID_SPLIT.$T.predict.csv" \
        --start_stage 2 --stop_stage 2 ) 2>&1 | tee "$ART/eval_$GRID_SPLIT.$T.log"
    for R in ref1 ref2; do
      cp "$DATA/eval/sctk/$GRID_SPLIT.$R.sgml" "$ART/$GRID_SPLIT.$T.$R.sgml"
    done
    "$DECOMPPY" "$REPO/scripts/error_decomposition.py" \
      --sgml-ref1 "$ART/$GRID_SPLIT.$T.ref1.sgml" \
      --sgml-ref2 "$ART/$GRID_SPLIT.$T.ref2.sgml" \
      --manifest-csv "$DATA/manifest/$GRID_SPLIT.csv" \
      --hyp-csv "$ART/$GRID_SPLIT.$T.predict.csv" \
      --out-json "$ART/decomp_$GRID_SPLIT.$T.json" \
      2>&1 | tee "$ART/decomp_$GRID_SPLIT.$T.log"
  done
  step "grid summary"
  "$DECOMPPY" - "$ART" "$GRID_SPLIT" "$BASE_PROJ_S" <<'PY'
import glob, json, os, re, sys
art, split, base_proj = sys.argv[1], sys.argv[2], float(sys.argv[3])
rows = []
for p in sorted(glob.glob(os.path.join(art, f"decomp_{split}.b*.json"))):
    d = json.load(open(p)); t, e = d["totals"], d["empties"]
    tag = re.search(rf"decomp_{re.escape(split)}\.(b.+)\.json$", p).group(1)
    wt = os.path.join(art, f"walltime_{split}.{tag}.txt")
    secs = float(open(wt).read().strip()) if os.path.exists(wt) else None
    rows.append([tag, t["cer"]*100, t["wer"]*100, e["n"], t["char_ops"]["D"],
                 t["char_ops"]["S"], t["char_ops"]["I"], secs])
base = next((r for r in rows if r[0].startswith("b1m")), None)  # b1m..., never b12m...
rows.sort(key=lambda r: r[1])
print(f"{'config':<12}{'CER%':>9}{'WER%':>9}{'empty':>7}{'charD':>8}{'charS':>8}{'charI':>8}{'wall_s':>9}{'proj_s':>9}")
for r in rows:
    proj = base_proj * r[7] / base[7] if (base and base[7] and r[7]) else float("nan")
    print(f"{r[0]:<12}{r[1]:>9.3f}{r[2]:>9.3f}{r[3]:>7d}{r[4]:>8d}{r[5]:>8d}{r[6]:>8d}"
          f"{(r[7] if r[7] else 0):>9.0f}{proj:>9.0f}")
if base:
    print(f"\ndelta vs {base[0]} (CER {base[1]:.3f}%, charD {base[4]}):")
    for r in rows:
        if r[0] != base[0]:
            print(f"  {r[0]:<12} dCER={r[1]-base[1]:+7.3f}  dDel={r[4]-base[4]:+6d}  "
                  f"dIns={r[6]-base[6]:+6d}  dEmpty={r[3]-base[3]:+4d}")
    print("\nThe mechanism check: a beam that is WORKING must move dDel strongly negative.")
    print("If dCER improves mainly by dropping dIns or dEmpty, it is not the deletion fix")
    print("we predicted — say so rather than adopting a win for the wrong reason.")
print("\nB1e reminder: adopt only a config whose next-smaller beam also beats baseline.")
print("Hard disqualifier: proj_s > 12000 regardless of CER.")
PY
fi

# --------------------------------------------------------------- final confirm
if has decode_final; then
  [[ -n "$FINAL" ]] || { echo "decode_final: set FINAL='1:2:none <best> ...' from the grid"; exit 1; }
  step "decode $FINAL_SPLIT for: $FINAL"
  pids=()
  for C in $FINAL; do
    T=$(tag "$C"); IFS=: read -r B M N <<<"$C"
    SAPC2_BEAM="$B" SAPC2_BEAM_MAX_SYMBOLS="$M" SAPC2_BEAM_LENGTH_NORM="$N" \
    SAPC2_THREADS=1 "$GATEPY" "$REPO/track2_starting_kit/local_decode.py" \
      --submission-dir "$EXTRACT" \
      --manifest-csv "$DATA/manifest/$FINAL_SPLIT.csv" \
      --streaming-interval 0 --data-root "$DATA" \
      --out-csv "$ART/$FINAL_SPLIT.$T.predict.csv" \
      --out-partial-json "$ART/$FINAL_SPLIT.$T.partial.json" \
      > "$ART/decode_$FINAL_SPLIT.$T.log" 2>&1 &
    pids+=($!)
  done
  fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
  [[ $fail -eq 0 ]] || { echo "decode_final: a decode FAILED"; exit 1; }
fi

if has score_final; then
  [[ -n "$FINAL" ]] || { echo "score_final: set FINAL"; exit 1; }
  step "score $FINAL_SPLIT (serial)"
  for C in $FINAL; do
    T=$(tag "$C")
    ( cd "$REPO" && ./evaluate.sh --split "$FINAL_SPLIT" \
        --hyp-csv "$ART/$FINAL_SPLIT.$T.predict.csv" \
        --start_stage 2 --stop_stage 2 ) 2>&1 | tee "$ART/eval_$FINAL_SPLIT.$T.log"
    for R in ref1 ref2; do
      cp "$DATA/eval/sctk/$FINAL_SPLIT.$R.sgml" "$ART/$FINAL_SPLIT.$T.$R.sgml"
    done
    "$DECOMPPY" "$REPO/scripts/error_decomposition.py" \
      --sgml-ref1 "$ART/$FINAL_SPLIT.$T.ref1.sgml" \
      --sgml-ref2 "$ART/$FINAL_SPLIT.$T.ref2.sgml" \
      --manifest-csv "$DATA/manifest/$FINAL_SPLIT.csv" \
      --out-json "$ART/decomp_$FINAL_SPLIT.$T.json" \
      2>&1 | tee "$ART/decomp_$FINAL_SPLIT.$T.log"
  done
fi

# --------------------------------------------------------------- latency + honest wall clock
# B1c/B1d. A beam changes WHEN the first token appears and how many decoder calls each
# chunk costs. Run SERIALLY here: the grid's parallel wall times share cores and would
# flatter or punish a config for reasons that have nothing to do with the beam.
if has latency; then
  [[ -n "$FINAL" ]] || { echo "latency: set FINAL"; exit 1; }
  step "streaming pass (real 100 ms pacing, serial) for: $FINAL"
  for C in $FINAL; do
    T=$(tag "$C"); IFS=: read -r B M N <<<"$C"
    start=$(date +%s)
    SAPC2_BEAM="$B" SAPC2_BEAM_MAX_SYMBOLS="$M" SAPC2_BEAM_LENGTH_NORM="$N" \
    SAPC2_THREADS=1 "$GATEPY" "$REPO/track2_starting_kit/local_decode.py" \
      --submission-dir "$EXTRACT" \
      --manifest-csv "$DATA/manifest/Dev_streaming.csv" \
      --streaming-manifest-csv "$DATA/manifest/Dev_streaming.csv" \
      --data-root "$DATA" \
      --out-csv "$ART/stream.$T.csv" --out-partial-json "$ART/stream.$T.json" \
      2>&1 | tee "$ART/stream.$T.log"
    echo "$(( $(date +%s) - start ))" | tee "$ART/stream_walltime.$T.txt"
    ( cd "$REPO" && ./evaluate.sh --start_stage 3 --stop_stage 3 \
        --partial-json "$ART/stream.$T.json" \
        --manifest-csv "$DATA/manifest/Dev_streaming.csv" ) 2>&1 | tee "$ART/latency.$T.log"
  done
  echo
  echo "Check n_utts_with_timing == n_utts_total in each latency log: an all-empty TTFT"
  echo "fallback looks like a latency number but is a decode failure (the int8 lesson)."
  echo "Also check the streaming log for chunks taking > 100 ms — past that the stream"
  echo "falls behind real time and TTLT degrades for a reason the beam cannot fix."
fi

step "done — copy back $ART, then STOP THE POD"
