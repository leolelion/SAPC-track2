#!/usr/bin/env bash
# pod_train_rnnlm.sh — train a small RNN-LM for shallow fusion (research/42 R1).
#
# THIS IS A SEPARATE, GPU-REQUIRING POD SESSION — not the cheap CPU latency sweep
# (pod_run_lm_and_latency_sweep.sh). Run it ONLY after the FEASIBILITY GATE below
# passes, i.e. after confirming icefall's STREAMING decode path can actually consume
# an external LM. Do not burn GPU training an LM that cannot be wired in.
#
# PRECONDITIONS:
#   - streaming_zipformer/ setup.sh has run on this pod (icefall cloned, bpe.model present).
#   - SAP data at $DATA_ROOT with manifest/Train.csv (+ optionally Dev_pool.csv from the
#     speaker-disjoint split). NEVER feed Dev_heldout/Test text to the LM (eval leak).
#
# NOTE ON REPRODUCIBILITY: streaming_zipformer/setup.sh clones icefall at UNPINNED HEAD
#   (`git clone --depth 1 …`). Exact recipe paths/flags below are the standard icefall
#   rnn_lm layout but MUST be verified against the actual checkout — hence Stage 0.
set -euo pipefail

: "${DATA_ROOT:?set DATA_ROOT to the SAP data root}"
: "${PROJ_ROOT:=$(pwd)}"
KIT="$PROJ_ROOT/track2_starting_kit"
ICEFALL="$KIT/streaming_zipformer/icefall"
BPE="$KIT/streaming_zipformer/weights/data/lang_bpe_500/bpe.model"
LM_DIR="$KIT/streaming_zipformer_lm/lm"          # artifact lands here (config.yaml lm_path)
WORK="$PROJ_ROOT/experiments/lm_train"
mkdir -p "$WORK" "$LM_DIR"

# ===========================================================================
# STAGE 0 — FEASIBILITY GATE (do this FIRST; costs nothing).
#   Determine whether the STREAMING zipformer decode path exposes an LM hook.
#   In icefall, `modified_beam_search_lm_shallow_fusion` lives in the OFFLINE
#   beam_search.py; the streaming path (streaming_decode.py / streaming_beam_search.py)
#   historically does NOT take an LM. If the grep below finds no streaming LM hook,
#   shallow fusion needs novel code in the sibling model.py — which the house rules
#   say NOT to hack blindly. STOP and write research/44 instead of training.
# ===========================================================================
echo "=== STAGE 0: feasibility gate ==="
ZIP="$ICEFALL/egs/librispeech/ASR/zipformer"
echo "[probe] streaming decode files:"; ls "$ZIP"/streaming_*.py 2>/dev/null || true
echo "[probe] LM hooks in the STREAMING path:"
if grep -RnE "lm_scale|shallow_fusion|LmScorer|\blm\b" \
      "$ZIP/streaming_decode.py" "$ZIP/streaming_beam_search.py" 2>/dev/null; then
  echo "[GATE] streaming path references an LM — inspect the exact signature above and"
  echo "       wire lm_scale/lm_path into streaming_zipformer_lm/model.py (SIBLING ONLY)."
else
  echo "[GATE][BLOCKER] no LM hook found in the streaming decode path."
  echo "  => Shallow fusion is NOT available out of the box. Do NOT train yet."
  echo "  => Options: (a) implement modified_beam_search_lm_shallow_fusion in the SIBLING"
  echo "     streaming_zipformer_lm/model.py's decode loop (novel, gate carefully), or"
  echo "     (b) drop R1 and let the latency sweep carry the session. Write research/44."
  echo "  Exiting without training (feasibility not confirmed)."
  exit 3
fi
echo "[probe] rnn_lm recipe location (verify before Stage 2):"
find "$ICEFALL" -path '*rnn_lm*train.py' -print 2>/dev/null || true
find "$ICEFALL" -path '*prepare_lm_training_data*' -print 2>/dev/null || true

# ===========================================================================
# STAGE 1 — build the in-domain LM corpus (pod-independent; script self-tested).
# ===========================================================================
echo "=== STAGE 1: build LM corpus ==="
MAN="$DATA_ROOT/manifest"
CORPUS_ARGS=(--manifest-csv "$MAN/Train.csv")
[ -f "$MAN/Dev_pool.csv" ] && CORPUS_ARGS+=(--manifest-csv "$MAN/Dev_pool.csv")
python3 "$PROJ_ROOT/scripts/prepare_lm_text.py" \
  "${CORPUS_ARGS[@]}" \
  --out "$WORK/corpus.txt" --stats-json "$WORK/corpus.stats.json"
echo "[reminder] Dev_heldout/Test text is intentionally NOT included (eval leak)."

# ===========================================================================
# STAGE 2 — tokenize corpus with the SHARED ASR bpe.model + train rnn_lm.
#   The LM tokenizer MUST equal the ASR tokenizer ($BPE) or fusion is meaningless.
#   Paths/flags below follow the standard icefall rnn_lm recipe — VERIFY against the
#   Stage-0 `find` output before running; adjust if the unpinned checkout differs.
# ===========================================================================
echo "=== STAGE 2: prepare LM data + train (VERIFY PATHS FIRST) ==="
cat <<EOF
[MANUAL/VERIFY] Standard icefall rnn_lm flow (edit to match the actual checkout):

  # 2a) sentence-piece encode the corpus into token ids using the ASR bpe.model:
  python3 $ICEFALL/egs/librispeech/ASR/local/prepare_lm_training_data.py \\
      --bpe-model "$BPE" \\
      --lm-data "$WORK/corpus.txt" \\
      --lm-archive "$WORK/lm_data.pt"

  # 2b) train a small LSTM LM (keep it small — CPU inference budget at decode time):
  python3 $ICEFALL/rnn_lm/train.py \\
      --start-epoch 0 --num-epochs 10 \\
      --use-fp16 false --tie-weights true \\
      --embedding-dim 512 --hidden-dim 512 --num-layers 2 \\
      --lm-data "$WORK/lm_data.pt" \\
      --lm-data-valid "$WORK/lm_data.pt" \\
      --exp-dir "$WORK/rnn_lm_exp"

  # 2c) publish the chosen checkpoint where config.yaml expects it:
  cp "$WORK/rnn_lm_exp/epoch-best.pt" "$LM_DIR/epoch-best.pt"
EOF
echo "[STOP] Stage 2 is manual/verify-gated — no blind training from this script."

# ===========================================================================
# STAGE 3 — wire + sweep (only after Stage 0 confirmed a hook AND Stage 2 produced a ckpt)
# ===========================================================================
echo "=== STAGE 3: wiring + lm_scale sweep (manual) ==="
cat <<'EOF'
[NEXT]
  1. Wire the LM into streaming_zipformer_lm/model.py (SIBLING ONLY) using the exact
     Stage-0 signature; set decoding.lm_path -> lm/epoch-best.pt.
  2. Sweep decoding.lm_scale in {0.05,0.1,0.2,0.3} via config clones, then run the
     faithful gate (scripts/pod_run_lm_and_latency_sweep.sh picks up streaming_zipformer_lm).
  3. Gate on Dev_heldout CER AND per-etiology latency; ship only if non-dominated vs beam-4.
     If the ALS/DS severe tail regresses, stop (density-ratio/ILME is the fallback, R5).
EOF
echo "[DONE] LM prep staged. Feasibility gate is the load-bearing step — respect its exit 3."
