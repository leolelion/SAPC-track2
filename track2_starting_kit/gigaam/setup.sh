#!/bin/bash
# =====================================================================
#  GigaAM Streaming — Dependency Setup
#
#  Installs gigaam into the base Python environment using --no-deps
#  to bypass the torch<=2.5.1 constraint (gigaam works with torch 2.6+).
#  torch is expected to already be present in the base env.
#
#  GigaAM package sources:
#    PyPI:   pip install gigaam
#    GitHub: https://github.com/salute-developers/GigaAM
#    HF:     https://huggingface.co/salute-developers/GigaAM
#
#  ⚠️  GigaAM v2 was trained on Russian speech — verify English CER
#      before using in competition. Run test_streaming.py with real audio.
# =====================================================================
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_NAME=$(python3 -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('${DIR}/config.yaml')
print(cfg.model.name)
" 2>/dev/null || echo "v2_rnnt")

echo "=== GigaAM: installing into base env (torch constraint bypassed) ==="
if python3 -c "import gigaam" 2>/dev/null; then
  echo "  gigaam already installed — skipping."
else
  # Install gigaam without pulling in torch (already present in base env)
  pip install --no-cache-dir --no-deps "gigaam"
  # Install gigaam's non-torch dependencies
  pip install --no-cache-dir "hydra-core<=1.3.2" "pydub<=0.25.1" "sentencepiece" "omegaconf>=2.3"
fi

echo "=== GigaAM: verifying installation ==="
python3 -c "
import gigaam, omegaconf
print('omegaconf:', omegaconf.__version__)
print('gigaam: OK (version:', getattr(gigaam, '__version__', 'unknown'), ')')
import torch
print('torch:', torch.__version__)
"

echo "=== GigaAM: pre-downloading model weights (${MODEL_NAME}) ==="
python3 -c "
import torch
# torch 2.6+ changed torch.load default to weights_only=True, breaking gigaam.
_orig_load = torch.load
torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, 'weights_only': False})
import gigaam
print('Downloading ${MODEL_NAME} weights from HuggingFace ...')
model = gigaam.load_model('${MODEL_NAME}')
print('GigaAM ${MODEL_NAME} loaded.')

# Quick smoke test — write 1s of silence to a temp wav and transcribe
import numpy as np, tempfile, wave, struct, os
audio = np.zeros(16000, dtype=np.float32)
with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
    tmp_path = f.name
with wave.open(tmp_path, 'w') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(16000)
    wf.writeframes(struct.pack('<' + 'h' * len(audio), *np.round(audio * 32767).astype(np.int16)))
result = model.transcribe(tmp_path)
os.unlink(tmp_path)
print(f'Smoke test (1s silence): {result!r}')
print()
print('⚠️  IMPORTANT: Test with real English speech to verify language support.')
print('   If CER > 60% on clean English, skip this model.')
"

echo "=== GigaAM: setup complete ==="
echo "    Model: ${MODEL_NAME}"
echo "    Run:   python3 test_streaming.py --mock  (interface check)"
echo "           python3 test_streaming.py          (real model)"
echo "           python3 test_streaming.py --audio /path/to/english.wav"
