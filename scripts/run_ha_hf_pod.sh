#!/usr/bin/env bash
# =====================================================================
# Step 3 pod session — H-F (speaker forensics) + H-A (lookahead re-export).
#
# WHY THESE TWO, IN THIS ORDER
# `scripts/analyze_rate_and_speakers.py` on the banked Dev_diag run (local, $0) found:
#   * speaking-rate quartiles CER 39.70 / 31.03 / 18.19 / 10.31, but error MASS is flat
#     (19.7 / 27.4 / 27.9 / 25.0 %) and the slow quartile holds only 9.3% of reference
#     characters. Any rate-only fix is capped at ~3.7 CER points. 74% of the characters
#     are in Q3+Q4, and LOOKAHEAD is the only lever we have that can reach them.
#   * speaker 55c1784a: CER 86.65%, 21 of 28 utterances empty, 14.1% of all error mass,
#     leave-one-out shift -2.14 points. Duration, rate and word count are all ruled out
#     locally -- it is not severity and not the rate axis. The cause is in the waveform.
#   * Dev_diag's unpaired 95% CI is +-2.7 CER points. Only PAIRED comparisons on the
#     identical utterance set can resolve what we care about. Every gate below is paired.
#
# H-F runs FIRST: it is minutes, needs no export, and is independent of H-A's outcome.
# If the session dies after stage 1 we still bought 2.14 points' worth of diagnosis.
#
# ---------------------------------------------------------------------
# PRE-REGISTERED GATES (written before the run; see investigations/step03_runbook.md)
#
#   G-HA0  SCREEN, KILL-ONLY. Offline proxy CER at a larger context must be within
#          +3.0 points of [70,1] on the 30-utt screen. Proxy numbers can only VETO here;
#          they never approve (an 11-point proxy error is on this repo's record).
#          MISS -> our single-context finetune did not leave the other contexts usable;
#          H-A becomes a GPU arm. STOP the session, do not export.
#
#   G-HA1  ONNX PARITY at the new context: >=90% exact-match vs the NeMo wrapper set to
#          the SAME context, on >=20 utterances. This is the stage that caught four
#          export/wrapper bugs last time. MISS -> the export is wrong, not the model.
#          Fix or stop; a decode on an unfaithful graph measures nothing.
#
#   G-HA2  ACCURACY, official scorer, Dev_diag severe, PAIRED vs the banked greedy
#          baseline (18.733%): delta CER <= -1.00 points AND the 95% paired CI excludes 0.
#          1.00 is not arbitrary: the point must clearly buy the latency it costs, and it
#          matches the B2a threshold used for the beam grid.
#
#   G-HA3  ON-TARGET. The gain must appear in Q3+Q4, not only Q1+Q2. H-A's whole premise
#          is that lookahead reaches the 74% of characters that rate-based levers cannot.
#          If the gain is confined to the slow quartiles, H-A is a duplicate of the
#          cheaper front-end lever and should not be paid for in latency.
#
#   G-HA4  PARETO. Shipped point is Test1 19.01% CER @ 416.9 ms mean(TTFT,TTLT).
#          Dev->Test transfer measured at +0.28 CER. A larger-lookahead point is only
#          worth submitting if it is NOT dominated on the live board:
#            [70,3] (~240 ms lookahead, projected ~537 ms mean): needs Dev_diag <= 17.8
#            [70,6] (~480 ms lookahead, projected ~630 ms mean): needs Dev_diag <= 17.8
#                   AND must beat yac3xn's 18.10 on CER, or it is dominated outright.
#          [70,3] is the PRIMARY target: it can dominate yac3xn on both axes.
#
#   H-F has no pass/fail gate -- it is forensics. Its stop condition is "report written".
#
# STOP CONDITION FOR THE WHOLE SESSION
#   Stop the pod as soon as (a) G-HA0 kills, or (b) the score_ctx table and the forensics
#   JSON are copied back. Do not leave it running to "look at" anything.
#
# BUDGET
#   forensics ~3 min · probe ~8 min · export ~10 min/context · parity ~5 min/context ·
#   decode Dev_diag ~65 min/context (parallel, 1 thread each, 24 vCPU) · scoring ~10 min.
#   Two contexts end-to-end ~= 1 h 45 m. On the $2.99/h pod ~= $5.50.
#
# USAGE
#   DATA=/workspace/SAPC2 \
#   STAGES="forensics probe" bash scripts/run_ha_hf_pod.sh
#   # then, only if G-HA0 passes:
#   STAGES="export parity decode score" CONTEXTS="70,3 70,6" bash scripts/run_ha_hf_pod.sh
# =====================================================================
set -euo pipefail

REPO=${REPO:-/workspace/SAPC-step01}
DATA=${DATA:?set DATA to the SAP data root (must contain manifest/)}
ART=${ART:-/workspace/artifacts/step03}
CKPT=${CKPT:-/workspace/parakeet_ft/armA_full/ft_smoke_encoder_only.nemo}
VENV=${VENV:-/workspace/nemoenv}
GATEPY=${GATEPY:-/workspace/onnx_ship/offvenv32/bin/python}   # ships ort 1.27.0
DECOMPPY=${DECOMPPY:-python3}                                  # the interpreter evaluate.sh uses
ORT_VERSION=${ORT_VERSION:-1.27.0}
WORK=${WORK:-/workspace/step03}

BANKED_CSV=${BANKED_CSV:-/workspace/onnx_ship/art32/Dev_diag.fp32.predict.csv}
BANKED_SUMMARY=${BANKED_SUMMARY:-/workspace/artifacts/step02/decomp_Dev_diag.banked.summary.json}
TARGET_SPEAKER=${TARGET_SPEAKER:-55c1784a}
CONTEXTS=${CONTEXTS:-"70,3 70,6"}          # candidates; [70,1] is always the reference
SCREEN_N=${SCREEN_N:-30}
PARITY_WAVS=${PARITY_WAVS:-}
SPLIT=${SPLIT:-Dev_diag}
STAGES=${STAGES:-"forensics probe"}

mkdir -p "$ART" "$WORK"
step() { echo -e "\n=========== $* ($(date -u +%H:%M:%S)) ==========="; }
has()  { [[ " $STAGES " == *" $1 "* ]]; }
tag()  { echo "ctx${1/,/_}"; }

# --------------------------------------------------------------- preflight
[[ -d "$DATA/manifest" ]] || { echo "preflight: $DATA/manifest missing"; exit 1; }
[[ -f "$BANKED_SUMMARY" ]] || { echo "preflight: $BANKED_SUMMARY missing (step02 artifact)"; exit 1; }
if has probe || has export; then
  [[ -f "$CKPT" ]] || { echo "preflight: checkpoint $CKPT missing"; exit 1; }
fi
if has parity || has decode; then
  [[ -x "$GATEPY" ]] || { echo "preflight: $GATEPY missing"; exit 1; }
  GATE_ORT=$("$GATEPY" -c "import onnxruntime;print(onnxruntime.__version__)")
  [[ "$GATE_ORT" == "$ORT_VERSION" ]] || {
    echo "preflight: interpreter has ort $GATE_ORT but we ship $ORT_VERSION — refusing"; exit 1; }
fi
echo "[preflight] repo=$REPO data=$DATA art=$ART contexts='$CONTEXTS' stages='$STAGES'"

# =============================================================== H-F
# Minutes, no model, no export. Answers whether 2.14 CER points are a recording-session
# artifact (data-side fix), a front-end gap shared with the corpus-wide empties (one fix,
# two problems), or an in-family talker the model simply fails on (training only).
if has forensics; then
  step "H-F: acoustic forensics on speaker $TARGET_SPEAKER"
  "$DECOMPPY" "$REPO/scripts/probe_speaker_audio.py" \
    --manifest-csv "$DATA/manifest/$SPLIT.csv" \
    --data-root "$DATA" \
    --decomp-summary "$BANKED_SUMMARY" \
    --target-speaker "$TARGET_SPEAKER" \
    --out-json "$ART/speaker_forensics.json" 2>&1 | tee "$ART/speaker_forensics.log"
  echo "[H-F] done -> $ART/speaker_forensics.json  (read the READING GUIDE block)"
fi

# =============================================================== H-A stage 1
# The kill gate. Costs ~8 minutes and can save the whole export + decode spend.
if has probe; then
  step "H-A/G-HA0: structural + functional lookahead probe"
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  python3 "$REPO/scripts/probe_lookahead_support.py" \
    --nemo "$CKPT" \
    --contexts 70,1 $CONTEXTS \
    --screen-manifest "$DATA/manifest/$SPLIT.csv" \
    --screen-n "$SCREEN_N" --data-root "$DATA" \
    --out-json "$ART/lookahead_probe.json" 2>&1 | tee "$ART/lookahead_probe.log"
  echo
  echo "STOP AND READ before running 'export':"
  echo "  G-HA0 KILL  -> H-A is a GPU arm. Stop the pod now."
  echo "  G-HA0 PASS  -> rerun with STAGES=\"export parity decode score\" and only the"
  echo "                 surviving contexts in CONTEXTS."
fi

# =============================================================== H-A stage 2
if has export; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  for C in $CONTEXTS; do
    T=$(tag "$C"); L=${C%%,*}; R=${C##*,}
    step "export ONNX at att_context=[$L,$R] -> $WORK/$T"
    python3 "$REPO/scripts/export_parakeet_onnx.py" \
      --nemo "$CKPT" --src-dir "$REPO/track2_starting_kit/parakeet_onnx" \
      --out-dir "$WORK/$T" --work-dir "$WORK/raw_$T" \
      --att-context "$L" "$R" 2>&1 | tee "$ART/export_$T.log"
    # the wrapper is meta-driven, but drop/trim/first-step policies were verified at
    # [70,1] ONLY. Record the geometry so a policy change is visible, not silent.
    "$DECOMPPY" -c "
import json,sys
m=json.load(open('$WORK/$T/weights/streaming_meta.json'))
print('[geometry]', {k: m.get('streaming_cfg',{}).get(k) for k in
      ('chunk_size','pre_encode_cache_size','valid_out_len','drop_extra_pre_encoded')},
      'att_context=', m.get('att_context_size'))
" | tee -a "$ART/export_$T.log"
  done
fi

# --------------------------------------------------------------- G-HA1 parity
# The NeMo side must be pinned to the SAME context or the comparison is meaningless.
# A patched COPY of the wrapper dir is used; the pristine one is never edited.
if has parity; then
  [[ -n "$PARITY_WAVS" ]] || { echo "set PARITY_WAVS to a wav glob (>=20 utts)"; exit 1; }
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  for C in $CONTEXTS; do
    T=$(tag "$C"); L=${C%%,*}; R=${C##*,}
    step "G-HA1: ONNX vs NeMo text parity at [$L,$R]"
    NEMODIR="$WORK/nemo_$T"
    rm -rf "$NEMODIR"; cp -r "$REPO/track2_starting_kit/parakeet_realtime_ft" "$NEMODIR"
    sed -i "s/^  att_context_size: .*/  att_context_size: [$L, $R]/" "$NEMODIR/config.yaml"
    grep -E "^  att_context_size" "$NEMODIR/config.yaml"
    # shellcheck disable=SC2086
    python3 "$REPO/scripts/parity_parakeet_onnx.py" \
      --onnx-dir "$WORK/$T" --nemo-dir "$NEMODIR" \
      --wavs $PARITY_WAVS --stages text --min-exact-rate 0.90 \
      --out-json "$ART/parity_$T.json" 2>&1 | tee "$ART/parity_$T.log"
  done
fi

# --------------------------------------------------------------- decode
# Real harness, accuracy pass (--streaming-interval 0), 1 intra-op thread per process,
# contexts in parallel. Hand-rolled decode never signs off (house rule).
if has decode; then
  step "decode $SPLIT through the REAL local_decode.py"
  pids=(); names=()
  for C in $CONTEXTS; do
    T=$(tag "$C")
    ( start=$(date +%s)
      SAPC2_THREADS=1 "$GATEPY" "$REPO/track2_starting_kit/local_decode.py" \
        --submission-dir "$WORK/$T" \
        --manifest-csv "$DATA/manifest/$SPLIT.csv" \
        --streaming-interval 0 --data-root "$DATA" \
        --out-csv "$ART/$SPLIT.$T.predict.csv" \
        --out-partial-json "$ART/$SPLIT.$T.partial.json" \
        > "$ART/decode_$SPLIT.$T.log" 2>&1
      echo $(( $(date +%s) - start )) > "$ART/walltime_$SPLIT.$T.txt" ) &
    pids+=($!); names+=("$T"); echo "  [$C] -> $T (pid ${pids[-1]})"
  done
  fail=0
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then echo "  ${names[$i]} ok ($(cat "$ART/walltime_$SPLIT.${names[$i]}.txt")s)"
    else echo "  ${names[$i]} FAILED — see $ART/decode_$SPLIT.${names[$i]}.log"; fail=1; fi
  done
  [[ $fail -eq 0 ]] || { echo "decode stage had failures; not scoring a partial grid"; exit 1; }
fi

# --------------------------------------------------------------- G-HA2/3 score
# evaluate.sh writes into the SHARED $DATA/eval tree, so scoring is SERIAL and each
# split's SGML is copied aside immediately (stage 2 overwrites per split).
if has score; then
  for C in $CONTEXTS; do
    T=$(tag "$C")
    step "score $T through evaluate.sh + official decomposition"
    ( cd "$REPO" && ./evaluate.sh --split "$SPLIT" --hyp-csv "$ART/$SPLIT.$T.predict.csv" \
        --start_stage 2 --stop_stage 2 ) 2>&1 | tee "$ART/eval_$SPLIT.$T.log"
    for R in ref1 ref2; do cp "$DATA/eval/sctk/$SPLIT.$R.sgml" "$ART/$SPLIT.$T.$R.sgml"; done
    "$DECOMPPY" "$REPO/scripts/error_decomposition.py" \
      --sgml-ref1 "$ART/$SPLIT.$T.ref1.sgml" --sgml-ref2 "$ART/$SPLIT.$T.ref2.sgml" \
      --manifest-csv "$DATA/manifest/$SPLIT.csv" --hyp-csv "$ART/$SPLIT.$T.predict.csv" \
      --out-json "$ART/decomp_$SPLIT.$T.json" 2>&1 | tee "$ART/decomp_$SPLIT.$T.log"
    step "G-HA2/G-HA3: paired vs banked greedy baseline"
    "$DECOMPPY" "$REPO/scripts/analyze_rate_and_speakers.py" \
      "$BANKED_SUMMARY" --compare "$ART/decomp_$SPLIT.$T.summary.json" \
      2>&1 | tee "$ART/paired_$SPLIT.$T.log"
  done
  echo
  echo "READ THE GATES:"
  echo "  G-HA2 needs delta CER <= -1.00 pts with a 95% paired CI that excludes 0."
  echo "  G-HA3 needs the gain to show up in Q3+Q4, not only Q1+Q2."
  echo "  Both pass -> run the latency pass. Either misses -> stop the pod and report."
fi

# --------------------------------------------------------------- G-HA4 latency
# Only for a context that already cleared G-HA2 and G-HA3. Lookahead is bought with
# latency; this is where we find out the price.
if has latency; then
  for C in $CONTEXTS; do
    T=$(tag "$C")
    step "G-HA4: streaming latency pass for $T"
    SAPC2_THREADS=1 "$GATEPY" "$REPO/track2_starting_kit/local_decode.py" \
      --submission-dir "$WORK/$T" \
      --manifest-csv "$DATA/manifest/${SPLIT}.csv" \
      --streaming-manifest-csv "$DATA/manifest/${SPLIT}_streaming.csv" \
      --data-root "$DATA" \
      --out-csv "$ART/stream.$T.csv" --out-partial-json "$ART/stream.$T.json" \
      2>&1 | tee "$ART/stream_$T.log"
    ( cd "$REPO" && ./evaluate.sh --start_stage 3 --stop_stage 3 \
        --partial-json "$ART/stream.$T.json" \
        --manifest-csv "$DATA/manifest/${SPLIT}_streaming.csv" ) 2>&1 | tee "$ART/latency_$T.log"
  done
  echo "  Shipped reference: Test1 19.01% CER @ 416.9 ms mean. Dev->Test CER transfer +0.28."
fi

echo
echo "=========== SESSION DONE ($(date -u +%H:%M:%S)) ==========="
echo "Copy back: $ART  then STOP THE POD."
