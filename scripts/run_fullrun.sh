#!/usr/bin/env bash
# Full finetune: cheap val/ckpt WIRING GATE -> two arms (encoder-only + full-unfrozen) on all 309k utts
# (cased targets), pinned [70,1], with validation + top-5 ckpt avg + early-stop. Eval each on Dev_diag.
set -uo pipefail
OUT=/workspace/finetune/nemo_ft; M=$OUT/manifests_cased
LOG="$OUT/fullrun.log"; : > "$LOG"; exec > >(tee -a "$LOG") 2>&1
source /workspace/nemoenv/bin/activate
DIAG=/workspace/SAPC2/manifest/Dev_diag.csv
[ -f "$DIAG" ] || python3 /workspace/build_diag_manifest.py /workspace/SAPC2/manifest/Dev.csv "$DIAG"
[ -f "$M/train.json" ] || python3 /workspace/prep_nemo_manifest.py --train-csv /workspace/SAPC2/manifest/Train.csv \
  --data-root /workspace/SAPC2 --out-dir "$M" --target text 2>&1 | tail -6
step(){ echo; echo "############### $* ###############"; }

step "0. VAL/CKPT WIRING GATE (4k smoke, 2 epochs, enc-only) — abort the expensive run if broken"
python3 /workspace/nemo_finetune.py --mode smoke --epochs 2 --freeze encoder_only \
  --train-json "$M/smoke_train.json" --val-json "$M/dev_internal.json" --out-dir "$OUT/wiring_gate" --train-ctx "[70,1]" 2>&1 | tail -35
if [ ! -f "$OUT/wiring_gate/ft_smoke_encoder_only.nemo" ]; then echo "WIRING_GATE_FAILED -> abort"; echo "FULLRUN_DONE"; exit 0; fi
echo "WIRING_OK"

step "1. FULL arm A: ENCODER-ONLY (all 309k cased, 4 epochs, early-stop, top-5 avg)"
python3 /workspace/nemo_finetune.py --mode smoke --epochs 4 --freeze encoder_only \
  --train-json "$M/train.json" --val-json "$M/dev_internal.json" --out-dir "$OUT/full_enc" --train-ctx "[70,1]" 2>&1 | tail -30
echo "--- EVAL enc-only on Dev_diag (vs zero-shot 43.4%/18% empty) ---"
python3 /workspace/nemo_eval_diag.py "$OUT/full_enc/ft_smoke_encoder_only.nemo" "$DIAG" /workspace/SAPC2 "[70,1]" 2>&1 | tail -12

step "2. FULL arm B: FULL-UNFROZEN (all 309k cased, 4 epochs, early-stop, top-5 avg)"
python3 /workspace/nemo_finetune.py --mode smoke --epochs 4 --freeze full \
  --train-json "$M/train.json" --val-json "$M/dev_internal.json" --out-dir "$OUT/full_unfrozen" --train-ctx "[70,1]" 2>&1 | tail -30
echo "--- EVAL full-unfrozen on Dev_diag ---"
python3 /workspace/nemo_eval_diag.py "$OUT/full_unfrozen/ft_smoke_full.nemo" "$DIAG" /workspace/SAPC2 "[70,1]" 2>&1 | tail -12

echo "FULLRUN_DONE"
