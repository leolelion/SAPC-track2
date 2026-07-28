#!/usr/bin/env bash
# D1 Arm B rung launcher. Usage: run_armb_rung.sh <tag> <fastemit_lambda> [extra args...]
set -uo pipefail
TAG="$1"; FE="$2"; shift 2
OUT=/workspace/parakeet_ft/armB_${TAG}
mkdir -p "$OUT"
source /workspace/nemoenv/bin/activate
cd /workspace
echo "=== ARMB rung ${TAG} fastemit=${FE} start $(date -u +%FT%TZ) ==="
python3 /workspace/nemo_finetune_v2.py \
  --mode smoke \
  --base-nemo /workspace/sweep/parakeet120.nemo \
  --train-json /workspace/nemo_ft/train.json \
  --val-json   /workspace/nemo_ft/val.json \
  --out-dir "$OUT" \
  --freeze joint_unfreeze \
  --train-ctx "[70,1]" \
  --lr 1e-4 --joint-lr-mult 0.1 \
  --epochs 4 --bs 16 \
  --fastemit-lambda "$FE" \
  "$@" > "$OUT/train.log" 2>&1
RC=$?
echo "ARMB_${TAG}_EXIT=$RC $(date -u +%FT%TZ)"
