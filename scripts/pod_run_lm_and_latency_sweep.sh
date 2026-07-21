#!/usr/bin/env bash
# pod_run_lm_and_latency_sweep.sh — SINGLE cost-gated pod session for SAPC2 Track 2.
#
# Purpose: resolve the highest-EV Zipformer experiments in ONE paid run, using ONLY
# the organizers' faithful harness (local_decode.py + evaluate.sh). Wrap, never modify
# scoring (AGENTS.md house rules). Each block writes an experiments/exp_XXX/ artifact.
#
# PRECONDITIONS (do all locally BEFORE starting a pod — AGENTS.md cost gate):
#   - SAP data present at $DATA_ROOT with manifest/Dev.csv + manifest/Dev_streaming.csv
#   - Speaker-disjoint gate built:  scripts/make_speaker_disjoint_dev.py (see below)
#   - A Linux x86_64 CPU host / the Docker runtime image xiuwenz2/sapc2-runtime:latest
#   - Sibling submission dirs prepared (streaming_zipformer_lm/, *_beam8/, latency variants)
#   - **evaluate.sh lines 36-38 EDITED on the pod** — DATA_ROOT / PROJ_ROOT / SCTK_DIR are
#     UNCONDITIONAL assignments in evaluate.sh (they clobber any exported env var), so the
#     exports below do NOT reach evaluate.sh's internal scoring. You MUST set them there too:
#         DATA_ROOT="$DATA_ROOT"   PROJ_ROOT="$PROJ_ROOT"   SCTK_DIR=/workspace/SCTK
#     (Filling path placeholders is config, NOT a scoring-semantics change — house rule intact.)
#   - Each submission dir must have run its setup.sh (downloads icefall + weights/epoch-30.pt).
#
# VARIANTS: streaming_zipformer (required) is always run. streaming_zipformer_lm,
#   _beam8, and the latency-sweep dirs are OPTIONAL — this script SKIPS any that are
#   absent (prints [SKIP]) instead of failing, so it is safe to run with a partial set.
#   NOTE: no earlyemit variant exists — model.py:377 already emits a partial after every
#   decoded chunk with no suppression policy, so early-emit is not applicable (research/43 Task C).
#
# STOP CONDITION: once Dev_heldout CER (beam4, beam4+LM, beam8) and the latency-sweep
# JSON are known, scp artifacts back and STOP the pod immediately.
set -euo pipefail

: "${DATA_ROOT:?set DATA_ROOT to the SAP data root}"
: "${PROJ_ROOT:=$(pwd)}"
KIT="$PROJ_ROOT/track2_starting_kit"
MAN="$DATA_ROOT/manifest"

# ---------------------------------------------------------------------------
# 0) Build the speaker-disjoint gate (free; safe to re-run). This is THE control
#    that makes Dev an honest Test proxy (research/37: Dev->Test unreliable).
# ---------------------------------------------------------------------------
python3 "$PROJ_ROOT/scripts/make_speaker_disjoint_dev.py" \
  --manifest-csv  "$MAN/Dev.csv" \
  --streaming-csv "$MAN/Dev_streaming.csv" \
  --out-prefix    "$MAN/Dev" \
  --heldout-frac 0.3 --seed 0
HELD="$MAN/Dev_heldout.csv"
HELD_STREAM="$MAN/Dev_heldout_streaming.csv"

# Helper: decode a submission dir on the heldout split, then score CER + latency.
run_and_score () {
  local sub="$1" tag="$2"
  local outdir="$PROJ_ROOT/experiments/$tag"; mkdir -p "$outdir"
  ( cd "$KIT" && python3 local_decode.py \
      --submission-dir "./$sub" \
      --manifest-csv "$HELD" \
      --streaming-manifest-csv "$HELD_STREAM" \
      --data-root "$DATA_ROOT" \
      --out-csv "$outdir/Dev_heldout.predict.csv" \
      --out-partial-json "$outdir/Dev_heldout.partial.json" )
  ./evaluate.sh --split Dev --hyp-csv "$outdir/Dev_heldout.predict.csv" \
      --start_stage 0 --stop_stage 2 | tee "$outdir/accuracy.log"
  ./evaluate.sh --start_stage 3 --stop_stage 3 \
      --partial-json "$outdir/Dev_heldout.partial.json" \
      --manifest-csv "$HELD_STREAM" | tee "$outdir/latency.log"
  # snapshot config + git hash for reproducibility (experiments/ convention)
  cp "$KIT/$sub/config.yaml" "$outdir/config.snapshot.yaml" 2>/dev/null || true
  git -C "$PROJ_ROOT" rev-parse HEAD > "$outdir/git_commit.txt"
}

# Optional-variant wrapper: run only if the submission dir exists, else [SKIP].
maybe_run_and_score () {
  local sub="$1" tag="$2"
  if [ -d "$KIT/$sub" ]; then
    run_and_score "$sub" "$tag"
  else
    echo "[SKIP] $sub not present — skipping $tag"
  fi
}

# ---------------------------------------------------------------------------
# E-E: official exp_000 anchor (pristine baseline, beam-4 as shipped)
# ---------------------------------------------------------------------------
run_and_score streaming_zipformer exp_000_zipformer_beam4_heldout

# ---------------------------------------------------------------------------
# E-B: RNN-LM shallow fusion on beam-4  (needs streaming_zipformer_lm/ prebuilt:
#      icefall rnn_lm trained on $MAN transcripts; config.yaml -> decoding.lm_scale.
#      SKELETON as of local prep: model.py does not yet read lm_scale/lm_path — this
#      block runs only after the sherpa online-LM wiring is added on the pod.)
# ---------------------------------------------------------------------------
maybe_run_and_score streaming_zipformer_lm exp_lm_shallowfusion_heldout

# ---------------------------------------------------------------------------
# E-D: beam-8 accuracy + LATENCY confirmation (config: num_active_paths=8)
# ---------------------------------------------------------------------------
maybe_run_and_score streaming_zipformer_beam8 exp_beam8_heldout

# ---------------------------------------------------------------------------
# E-C: latency Pareto sweep (config-only clones; no retrain). Each dir differs
#      only in encoder.chunk_size / encoder.left_context_frames. (No earlyemit:
#      model.py already emits after every chunk — research/43 Task C.)
# ---------------------------------------------------------------------------
for v in cs8_lc128 cs16_lc64 cs16_lc256 cs32_lc128; do
  maybe_run_and_score "streaming_zipformer_$v" "exp_latsweep_$v"
done

echo "[DONE] All faithful-harness results under $PROJ_ROOT/experiments/."
echo "[NEXT] Pick the config that Pareto-dominates beam-4 on the SPEAKER-DISJOINT"
echo "       gate (CER and/or latency); scp experiments/ back; STOP THE POD NOW."
