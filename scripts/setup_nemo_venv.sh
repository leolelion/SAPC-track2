#!/usr/bin/env bash
# Build an ISOLATED NeMo venv (no system site-packages) so NeMo pulls a self-consistent
# torch+torchvision+torchaudio set -> avoids the ABI break we hit in the system env.
# Then VERIFY it actually loads our .nemo (the real env test) + characterize. Pod-side.
set -uo pipefail
OUT=/workspace/finetune/nemo_ft; mkdir -p "$OUT"
LOG="$OUT/venv_setup.log"; : > "$LOG"; exec > >(tee -a "$LOG") 2>&1
NEMO="$OUT/nemotron-speech-streaming-en-0.6b.nemo"

echo "=== $(date) ensure isolated NeMo venv ==="
# MUST live on /workspace (MooseFS network volume) -> persists across pod stop/start/recreate.
# The container disk resets each cycle, so a venv there would need rebuilding every time.
VENV=/workspace/nemoenv
echo "VENV=$VENV"
if [ -x "$VENV/bin/python" ] && "$VENV/bin/python" -c "import nemo.collections.asr" 2>/dev/null; then
  echo "=== existing working NeMo venv found -> REUSE (no rebuild) ==="
  source "$VENV/bin/activate"
else
  echo "=== building NeMo venv (one-time, ~15-30 min) ==="
  rm -rf "$VENV"
  python3 -m venv "$VENV"
  source "$VENV/bin/activate"
  python -m pip install -q --upgrade pip
  pip install -q "nemo_toolkit[asr]" huggingface_hub 2>&1 | tail -12
fi

echo "=== verify consistent stack ==="
python -c "import torch,torchvision,torchaudio,nemo; print('torch',torch.__version__,'| tv',torchvision.__version__,'| ta',torchaudio.__version__,'| nemo',nemo.__version__)" 2>&1 | tail -3 || echo STACK_IMPORT_FAIL
python -c "import nemo.collections.asr as a; import torch; print('asr import OK; cuda',torch.cuda.is_available())" 2>&1 | tail -3 || echo ASR_IMPORT_FAIL

if [ ! -f "$NEMO" ]; then
  echo "=== .nemo missing -> download ==="
  python -c "from huggingface_hub import hf_hub_download as d; d('nvidia/nemotron-speech-streaming-en-0.6b', filename='nemotron-speech-streaming-en-0.6b.nemo', local_dir='$OUT')" 2>&1 | tail -3
fi

echo "=== CRITICAL: load the .nemo (does the env work for OUR model?) ==="
python - "$NEMO" <<'PY' 2>&1 | tail -25
import sys
import nemo.collections.asr as nemo_asr
m = nemo_asr.models.ASRModel.restore_from(sys.argv[1], map_location="cpu")
print("LOADED_OK:", type(m).__name__)
c = m.cfg
g = lambda o,k: (o.get(k) if hasattr(o,"get") else getattr(o,k,"<?>"))
print("decoder._target_:", g(c.decoder,"_target_"))
print("att_context_size:", g(c.encoder,"att_context_size"))
print("att_context_probs:", g(c.encoder,"att_context_probs"))
print("att_context_style:", g(c.encoder,"att_context_style"))
print("d_model:", g(c.encoder,"d_model"), "n_layers:", g(c.encoder,"n_layers"))
try: print("vocab_size:", m.tokenizer.vocab_size)
except Exception as e: print("vocab_size?:", e)
PY

echo "=== dependency-free tar config dump (backup) ==="
python3 /workspace/nemo_characterize.py "$NEMO" 2>&1 | tail -30 || true
echo "NEMO_VENV_DONE $(date)"
