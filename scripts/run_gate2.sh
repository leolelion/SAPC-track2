#!/usr/bin/env bash
# Gate 2: clean overfit confirm + smoke (full vs encoder-only) + held-out per-etiology eval.
set -uo pipefail
OUT=/workspace/finetune/nemo_ft; M=$OUT/manifests
LOG="$OUT/gate2.log"; : > "$LOG"; exec > >(tee -a "$LOG") 2>&1
source /workspace/nemoenv/bin/activate
DIAG=/workspace/SAPC2/manifest/Dev_diag.csv
[ -f "$DIAG" ] || python3 /workspace/build_diag_manifest.py /workspace/SAPC2/manifest/Dev.csv "$DIAG"
step(){ echo; echo "############### $* ###############"; }

step "0. CLEAN OVERFIT (all 30 utts in one batch -> CER should -> ~0)"
python3 /workspace/nemo_finetune.py --mode overfit --freeze full --bs 32 \
  --train-json $M/overfit.json --val-json $M/dev_internal.json --out-dir $OUT/gate1_clean --train-ctx "[70,1]" 2>&1 | tail -20

step "1. SMOKE arm = FULL unfrozen (~4k utts, 3000 steps)"
python3 /workspace/nemo_finetune.py --mode smoke --freeze full \
  --train-json $M/smoke_train.json --val-json $M/dev_internal.json --out-dir $OUT/gate2_full --train-ctx "[70,1]" 2>&1 | tail -25

step "2. SMOKE arm = ENCODER-ONLY (anti-forgetting)"
python3 /workspace/nemo_finetune.py --mode smoke --freeze encoder_only \
  --train-json $M/smoke_train.json --val-json $M/dev_internal.json --out-dir $OUT/gate2_enc --train-ctx "[70,1]" 2>&1 | tail -25

step "3. HELD-OUT EVAL on Dev_diag-425 (zero-shot baseline = 47.5% CER / 25% empty, research/15)"
echo "--- FULL arm ---"
python3 /workspace/nemo_eval_diag.py $OUT/gate2_full/ft_smoke_full.nemo "$DIAG" /workspace/SAPC2 "[70,1]" 2>&1 | tail -12
echo "--- ENCODER-ONLY arm ---"
python3 /workspace/nemo_eval_diag.py $OUT/gate2_enc/ft_smoke_encoder_only.nemo "$DIAG" /workspace/SAPC2 "[70,1]" 2>&1 | tail -12

echo "GATE2_DONE"
