#!/usr/bin/env bash
# setup_finetune.sh — Install dependencies and download pretrained checkpoint.
# Targets PyTorch 2.5.0 + CUDA 12.4 + Python 3.11 (SAPC2 runtime).
# Run once on the RunPod GPU before training.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ── Step 0: Upgrade PyTorch to 2.5.0 ──────────────────────────────────────────
echo "=== Upgrading PyTorch to 2.5.0 (CUDA 12.4) ==="
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 \
    --index-url https://download.pytorch.org/whl/cu124 --quiet
python3 -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
"

# ── Step 1: Install k2 for torch 2.5.0 ────────────────────────────────────────
echo ""
echo "=== Installing k2 (torch 2.5.0, CUDA 12.4, cp311) ==="
K2_WHEEL="https://huggingface.co/csukuangfj/k2/resolve/main/cuda/1.24.4.dev20250307/linux-x64/k2-1.24.4.dev20250714+cu124.torch2.5.0-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.whl"
pip install --no-deps "$K2_WHEEL"
python3 -c "import k2; print(f'k2 {k2.__dev_version__} installed')"

# ── Step 2: Install kaldifeat for torch 2.5.0 ─────────────────────────────────
echo ""
echo "=== Installing kaldifeat (torch 2.5.0, CUDA 12.4, cp311) ==="
KALDIFEAT_WHEEL="https://huggingface.co/csukuangfj/kaldifeat/resolve/main/cuda/1.25.5.dev20250307/linux-x64/kaldifeat-1.25.5.dev20250630+cu124.torch2.5.0-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.whl"
pip install --no-deps "$KALDIFEAT_WHEEL"
python3 -c "import kaldifeat; print('kaldifeat installed')"

# ── Step 3: Clone / update icefall ────────────────────────────────────────────
echo ""
echo "=== Setting up icefall ==="
ICEFALL_TARGET="${ICEFALL_TARGET:-/workspace/finetune/icefall}"
if [ ! -d "$ICEFALL_TARGET" ]; then
    git clone --depth 1 https://github.com/k2-fsa/icefall.git "$ICEFALL_TARGET"
fi
pip install -e "$ICEFALL_TARGET" --quiet
python3 -c "import icefall; print('icefall installed')"

# ── Step 4: Remaining deps ────────────────────────────────────────────────────
echo ""
echo "=== Installing remaining deps ==="
pip install lhotse omegaconf sentencepiece tqdm huggingface_hub tensorboard \
    soundfile pandas --quiet

# ── Step 5: Download pretrained checkpoint ────────────────────────────────────
WEIGHTS_TARGET="${WEIGHTS_TARGET:-/workspace/finetune/weights/standard}"
echo ""
echo "=== Downloading pretrained checkpoint to ${WEIGHTS_TARGET} ==="
python3 - <<PYEOF
from huggingface_hub import hf_hub_download
import os, pathlib

repo = "Zengwei/icefall-asr-librispeech-streaming-zipformer-2023-05-17"
wd   = "${WEIGHTS_TARGET}"
os.makedirs(f"{wd}/exp",               exist_ok=True)
os.makedirs(f"{wd}/data/lang_bpe_500", exist_ok=True)

for f in ["exp/epoch-30.pt", "data/lang_bpe_500/bpe.model"]:
    dest = pathlib.Path(wd) / f
    if dest.exists():
        print(f"  Already exists: {dest}")
    else:
        result = hf_hub_download(repo_id=repo, filename=f, local_dir=wd)
        print(f"  Downloaded: {result}")
PYEOF

echo ""
echo "=== Setup complete! ==="
echo "Next: python3 finetuning/prepare_sapc_lhotse.py --data-root /workspace/SAPC2"
echo "Then: bash finetuning/tests/run_tests.sh"
