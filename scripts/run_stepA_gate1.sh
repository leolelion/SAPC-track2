#!/usr/bin/env bash
# Step A (persistent venv + data prep) + Gate 1 (overfit). Robust + discovers the train data source.
set -uo pipefail
OUT=/workspace/finetune/nemo_ft; mkdir -p "$OUT"
LOG="$OUT/stepA_gate1.log"; : > "$LOG"; exec > >(tee -a "$LOG") 2>&1
VENV=/workspace/nemoenv
NEMO="$OUT/nemotron-speech-streaming-en-0.6b.nemo"
step(){ echo; echo "############### $* ###############"; }

step "0. PERSISTENT venv on /workspace (idempotent)"
if [ -x "$VENV/bin/python" ] && "$VENV/bin/python" -c "import nemo.collections.asr" 2>/dev/null; then
  echo "reuse existing venv"
else
  echo "building persistent venv (one-time ~20 min)"
  rm -rf "$VENV"; python3 -m venv "$VENV"; source "$VENV/bin/activate"
  python -m pip install -q --upgrade pip
  pip install -q "nemo_toolkit[asr]" huggingface_hub 2>&1 | tail -8
fi
source "$VENV/bin/activate"
python -c "import nemo; print('nemo', nemo.__version__)" || { echo VENV_BROKEN; exit 1; }
[ -f "$NEMO" ] || python -c "from huggingface_hub import hf_hub_download as d; d('nvidia/nemotron-speech-streaming-en-0.6b', filename='$(basename $NEMO)', local_dir='$OUT')" 2>&1 | tail -2

step "1. DISCOVER training data source"
echo "--- /workspace/SAPC2/manifest ---"; ls -la /workspace/SAPC2/manifest/ 2>/dev/null | grep -iE 'train|\.csv'
echo "--- lhotse cuts ---"; ls -la /workspace/finetune/data/*train* 2>/dev/null
TRAIN_CSV=""
for c in /workspace/SAPC2/manifest/Train.csv /workspace/SAPC2/manifest/train.csv; do [ -f "$c" ] && TRAIN_CSV="$c" && break; done
[ -z "$TRAIN_CSV" ] && TRAIN_CSV=$(ls /workspace/SAPC2/manifest/Train*.csv 2>/dev/null | head -1 || true)
echo "TRAIN_CSV=${TRAIN_CSV:-<none>}"
if [ -n "$TRAIN_CSV" ]; then echo "--- header ---"; head -1 "$TRAIN_CSV"; echo "rows: $(($(wc -l < "$TRAIN_CSV")-1))"; fi

if [ -z "$TRAIN_CSV" ]; then
  echo "NO Train CSV in SAP schema -> data prep needs a lhotse-cuts->manifest converter (next step)."
  echo "STEPA_GATE1_DONE (data-discovery only)"; exit 0
fi

step "2. DATA PREP (speaker-disjoint NeMo manifests)"
python3 /workspace/prep_nemo_manifest.py --train-csv "$TRAIN_CSV" --data-root /workspace/SAPC2 \
  --out-dir "$OUT/manifests" --target norm_text_without_disfluency 2>&1 | tail -20

OVF="$OUT/manifests/overfit.json"; DEV="$OUT/manifests/dev_internal.json"
if [ ! -s "$OVF" ] || [ ! -s "$DEV" ]; then echo "manifests empty -> STOP"; echo STEPA_GATE1_DONE; exit 0; fi

step "3. GATE 1 — overfit a single batch (preflight prints first; aug off; CER->~0)"
python3 /workspace/nemo_finetune.py --mode overfit --freeze full \
  --train-json "$OVF" --val-json "$DEV" --out-dir "$OUT/gate1" --train-ctx "[70,1]" 2>&1 | tail -70

echo "STEPA_GATE1_DONE"
