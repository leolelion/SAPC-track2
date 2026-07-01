#!/usr/bin/env bash
# Phase-0 (v2 workup) pod orchestration: free drop-audit + fast-but-accurate convergence curve.
# Expects this dir uploaded to $WORK with normalizer/ package alongside the .py scripts.
set -uo pipefail
WORK=/workspace/finetune/nemo_ft/phase0
NEMO_DIR=/workspace/finetune/nemo_ft
DATA=/workspace/SAPC2
TRAIN_CSV=$DATA/manifest/Train.csv
LOG=$WORK/phase0.log; : > "$LOG"; exec > >(tee -a "$LOG") 2>&1
source /workspace/nemoenv/bin/activate
touch "$WORK/normalizer/__init__.py"

echo "############### preflight ###############"
ls -la "$NEMO_DIR/full_enc"/ft-*.ckpt 2>&1 || { echo "NO_CKPTS -> abort"; exit 0; }
ls -la "$NEMO_DIR"/nemotron-speech-streaming-en-0.6b.nemo 2>&1 || { echo "NO_BASE_NEMO -> abort"; exit 0; }
ls -la "$TRAIN_CSV" 2>&1 || { echo "NO_TRAIN_CSV -> abort"; exit 0; }

echo "############### 1. free drop-audit (>40s by etiology) ###############"
python3 "$WORK/phase0_dropped_by_etiology.py" --train-csv "$TRAIN_CSV" --max-duration 40 \
  --out-json "$NEMO_DIR/phase0_drop_audit.json" 2>&1 | tail -40

echo "############### 2. convergence curve (3k speaker-stratified, official metric) ###############"
python3 "$WORK/phase0_eval_curve.py" \
  --train-csv "$TRAIN_CSV" --data-root "$DATA" \
  --base-nemo "$NEMO_DIR/nemotron-speech-streaming-en-0.6b.nemo" \
  --ckpt-dir  "$NEMO_DIR/full_enc" \
  --subset-utts 3000 \
  --out-json  "$NEMO_DIR/phase0_curve.json" 2>&1 | tail -60

echo "PHASE0_ALL_DONE"
