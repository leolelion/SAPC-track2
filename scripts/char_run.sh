#!/usr/bin/env bash
# Pod-side Gate-0: ensure a NeMo env, then characterize the .nemo. Robust; writes characterize.log.
set -uo pipefail
OUT=/workspace/finetune/nemo_ft; mkdir -p "$OUT"
LOG="$OUT/characterize.log"; : > "$LOG"; exec > >(tee -a "$LOG") 2>&1
echo "=== $(date) check NeMo ==="
if ! python3 -c "import nemo.collections.asr" 2>/dev/null; then
  echo "nemo not importable -> installing nemo_toolkit[asr] (slow, may take 10-20 min) ..."
  pip install -q "nemo_toolkit[asr]" huggingface_hub 2>&1 | tail -8
fi
python3 -c "import nemo; print('nemo', nemo.__version__)" 2>&1 || echo NEMO_IMPORT_FAIL
python3 -c "import huggingface_hub" 2>/dev/null || pip install -q huggingface_hub 2>&1 | tail -3
echo "=== run characterization ==="
python3 /workspace/nemo_characterize.py 2>&1 || echo CHARACTERIZE_PY_FAIL
echo "CHARACTERIZE_DONE $(date)"
