#!/usr/bin/env bash
# D2 launcher — Arm B recipe + leakage-filtered, capped kNN-VC. DO NOT RUN until D1 clears
# GATE-TRAIN (PLANNED.md: D1 gates D2-D6; a frozen-joint fix cannot be bought with data).
#
# Three stages, each fails loud:
#   1. build_knnvc_train_manifest.py  — Train-provenance ONLY (drops 18,797 Dev-derived wavs and
#      the 40,339 `unknown`), hour-capped, ALS/Down-weighted.
#   2. build_d2_mixed_manifest.py     — merge with real SAP, hard-assert synth <= 25% of steps.
#   3. nemo_finetune_v2.py            — identical flags to D1 rung l0 EXCEPT --train-json, so any
#      delta is attributable to the data, not the recipe.
#
# val.json is untouched (it is Dev-drawn, and is the same val used by Arm A and D1) so the
# GATE-TRAIN comparison stays apples-to-apples.
set -euo pipefail
HOURS="${1:-60}"          # target synth hours before the step cap trims further
FE="${2:-0.03}"           # match whichever rung cleared D1
# armb_post.sh resolves its weights as /workspace/parakeet_ft/armB_<tag>/, so the out dir must
# follow that naming to reuse the post-chain unmodified.
OUT=/workspace/parakeet_ft/armB_d2knnvc
mkdir -p "$OUT"

python3 /workspace/build_knnvc_train_manifest.py \
  --out /workspace/nemo_ft/knnvc_train.json --hours "$HOURS" 2>&1 | tee "$OUT/build_knnvc.log"
grep -q KNNVC_TRAIN_MANIFEST_DONE "$OUT/build_knnvc.log" || { echo "!! knnvc manifest failed"; exit 1; }

python3 /workspace/build_d2_mixed_manifest.py \
  --real /workspace/nemo_ft/train.json \
  --synth /workspace/nemo_ft/knnvc_train.json \
  --out /workspace/nemo_ft/train_d2_knnvc.json \
  --max-frac 0.25 2>&1 | tee "$OUT/build_mixed.log"
grep -q D2_MIXED_MANIFEST_DONE "$OUT/build_mixed.log" || { echo "!! mixed manifest failed"; exit 1; }

python3 /workspace/nemo_finetune_v2.py --mode smoke \
  --base-nemo /workspace/sweep/parakeet120.nemo \
  --train-json /workspace/nemo_ft/train_d2_knnvc.json \
  --val-json /workspace/nemo_ft/val.json \
  --out-dir "$OUT" --freeze joint_unfreeze --train-ctx "[70,1]" \
  --lr 1e-4 --joint-lr-mult 0.1 --epochs 4 --bs 16 --fastemit-lambda "$FE" \
  > "$OUT/train.log" 2>&1

bash /workspace/armb_post.sh d2knnvc > /workspace/parakeet_ft/gate_armB_d2knnvc_post.log 2>&1
echo "D2_DONE $(date -u +%FT%TZ)"
