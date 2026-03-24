#!/bin/bash
# Master setup script for Zipformer fine-tuning on RunPod.
# Run this FIRST when the pod starts.
#
# Usage:
#   bash setup_pod.sh
#
set -euo pipefail

echo "============================================"
echo "  SAPC2 Zipformer Fine-tuning — Pod Setup"
echo "============================================"

WORK="/workspace/finetune"
mkdir -p "${WORK}"/{data,exp,onnx,weights}

# ══════════════════════════════════════════════
# Step 1: GPU & CUDA verification
# ══════════════════════════════════════════════
echo ""
echo "=== Step 1: GPU & CUDA check ==="
nvidia-smi
python3 -c "
import torch
print(f'PyTorch:  {torch.__version__}')
print(f'CUDA:     {torch.version.cuda}')
print(f'GPU:      {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NOT FOUND\"}')
print(f'cuDNN:    {torch.backends.cudnn.version()}')
"

# ══════════════════════════════════════════════
# Step 2: Verify SAPC2 data is accessible
# ══════════════════════════════════════════════
echo ""
echo "=== Step 2: Data check ==="
SAPC2="/workspace/SAPC2/processed"

if [ -d "${SAPC2}/Train" ]; then
    SAMPLE=$(find "${SAPC2}/Train" -name "*.wav" | head -3 | wc -l)
    echo "  Train audio dir: OK (found ${SAMPLE} sample WAVs in first scan)"
else
    echo "  ERROR: ${SAPC2}/Train not found!"
    echo ""
    echo "  If Train.tar.partXX files exist, extract with:"
    echo "    cd /workspace/SAPC2/processed && cat Train.tar.part* > Train.tar && tar xf Train.tar"
    exit 1
fi

for csv_name in Train.csv Dev.csv; do
    csv_path="${SAPC2}/manifest/${csv_name}"
    if [ -f "${csv_path}" ]; then
        ROWS=$(wc -l < "${csv_path}")
        echo "  ${csv_name}: ${ROWS} rows"
    else
        echo "  WARNING: ${csv_path} not found — check manifest location"
    fi
done

# ══════════════════════════════════════════════
# Step 3: Install Python dependencies
# ══════════════════════════════════════════════
echo ""
echo "=== Step 3: Install Python dependencies ==="

# Basic deps for data prep and training
pip install --quiet lhotse omegaconf sentencepiece huggingface_hub

# Detect PyTorch/CUDA/Python versions for wheel selection
TORCH_VER=$(python3 -c "import torch; print(torch.__version__.split('+')[0])")
CUDA_VER=$(python3 -c "import torch; v=torch.version.cuda; print(v.replace('.','') if v else 'cpu')")
PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')")

echo ""
echo "  Detected: PyTorch=${TORCH_VER}, CUDA=${CUDA_VER}, Python=cp${PY_VER}"
echo ""
echo "  ┌─────────────────────────────────────────────────────────────┐"
echo "  │  MANUAL STEP: Install k2 and kaldifeat with GPU support     │"
echo "  │                                                             │"
echo "  │  Browse available wheels:                                   │"
echo "  │    https://huggingface.co/csukuangfj/k2                     │"
echo "  │    https://huggingface.co/csukuangfj/kaldifeat              │"
echo "  │                                                             │"
echo "  │  Match: torch${TORCH_VER}, cuda${CUDA_VER}, cp${PY_VER}                │"
echo "  │                                                             │"
echo "  │  Example (update URL for your exact combo):                 │"
echo "  │    pip install https://huggingface.co/csukuangfj/k2/...    │"
echo "  │    pip install https://huggingface.co/csukuangfj/           │"
echo "  │                kaldifeat/...                                │"
echo "  └─────────────────────────────────────────────────────────────┘"
echo ""

# Attempt auto-install for common combos (torch 2.5.0, CUDA 12.4, cp311)
# Uncomment and adjust if you know the exact wheel URL:
# K2_WHEEL="https://huggingface.co/csukuangfj/k2/resolve/main/cuda/..."
# pip install "${K2_WHEEL}"

echo "  After manual install, verify with:"
echo "    python3 -c \"import k2; import kaldifeat; print('k2 + kaldifeat OK')\""

# ══════════════════════════════════════════════
# Step 4: Clone Icefall
# ══════════════════════════════════════════════
echo ""
echo "=== Step 4: Clone Icefall ==="
if [ -d "${WORK}/icefall/.git" ]; then
    echo "  Icefall already cloned at ${WORK}/icefall"
    echo "  Updating..."
    git -C "${WORK}/icefall" pull --ff-only || echo "  (pull skipped)"
else
    git clone --depth 1 https://github.com/k2-fsa/icefall.git "${WORK}/icefall"
    pip install --quiet -e "${WORK}/icefall"
    echo "  Icefall cloned and installed."
fi

# ══════════════════════════════════════════════
# Step 5: Download pre-trained checkpoints
# ══════════════════════════════════════════════
echo ""
echo "=== Step 5: Download pre-trained checkpoints ==="

# Standard streaming Zipformer — LibriSpeech, ~70M params
STANDARD_DIR="${WORK}/weights/standard"
mkdir -p "${STANDARD_DIR}/exp" "${STANDARD_DIR}/data/lang_bpe_500"

STANDARD_CKPT="${STANDARD_DIR}/exp/epoch-30.pt"
if [ -f "${STANDARD_CKPT}" ]; then
    echo "  Standard checkpoint already present: ${STANDARD_CKPT}"
else
    echo "  Downloading Standard Zipformer checkpoint..."
    python3 - <<'PYEOF'
from huggingface_hub import hf_hub_download
import os

repo = "Zengwei/icefall-asr-librispeech-streaming-zipformer-2023-05-17"
local_dir = "/workspace/finetune/weights/standard"

files = [
    "exp/epoch-30.pt",
    "data/lang_bpe_500/bpe.model",
    "data/lang_bpe_500/tokens.txt",
]
for f in files:
    dest = os.path.join(local_dir, f)
    if not os.path.exists(dest):
        print(f"  Downloading {f} ...")
        hf_hub_download(repo_id=repo, filename=f, local_dir=local_dir)
    else:
        print(f"  Already exists: {f}")

print("Standard checkpoint ready.")
PYEOF
fi

# Kroko Zipformer2 — check if PyTorch checkpoint is available
echo ""
echo "  Checking for Kroko PyTorch checkpoint..."
python3 - <<'PYEOF'
try:
    from huggingface_hub import list_repo_tree
    repo = "csukuangfj/sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06"
    pt_files = [
        item.path for item in list_repo_tree(repo)
        if item.path.endswith(".pt") or item.path.endswith(".ckpt")
    ]
    if pt_files:
        print(f"  Found PyTorch files in Kroko repo: {pt_files}")
    else:
        print("  No .pt/.ckpt files found in Kroko ONNX repo.")
        print("  Kroko will likely be submitted zero-shot (ONNX files already exist).")
except Exception as e:
    print(f"  Could not check Kroko repo: {e}")
PYEOF

# ══════════════════════════════════════════════
# Step 6: Inspect Icefall training & export scripts
# ══════════════════════════════════════════════
echo ""
echo "=== Step 6: Inspect Icefall training script ==="

TRAIN_SCRIPT="${WORK}/icefall/egs/librispeech/ASR/zipformer/train.py"
if [ -f "${TRAIN_SCRIPT}" ]; then
    echo "  Found training script: ${TRAIN_SCRIPT}"
    echo ""
    echo "  --- Key flags (checkpoint / manifest / LR) ---"
    python3 "${TRAIN_SCRIPT}" --help 2>&1 | \
        grep -iE "checkpoint|pretrain|finetune|resume|manifest|cuts|lhotse|lr|learning.rate|epoch" | \
        head -30 || true
    echo ""
else
    echo "  WARNING: Expected training script not found!"
    echo "  Browse: ls ${WORK}/icefall/egs/librispeech/ASR/zipformer/"
    ls "${WORK}/icefall/egs/librispeech/ASR/zipformer/" 2>/dev/null || true
fi

echo ""
echo "  --- Fine-tuning recipes in Icefall ---"
find "${WORK}/icefall/egs" -name "*finetune*" -o -name "*fine_tune*" 2>/dev/null | head -20 || echo "  (none found by name)"
grep -rl "finetune\|fine_tune\|pretrained_model" "${WORK}/icefall/egs/librispeech/ASR/" \
    --include="*.py" --include="*.sh" 2>/dev/null | head -10 || echo "  (no fine-tuning refs found)"

echo ""
echo "  --- ONNX export scripts ---"
find "${WORK}/icefall/egs/librispeech/ASR" -name "*export*onnx*" -o -name "*onnx*export*" 2>/dev/null

# ══════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════
echo ""
echo "============================================"
echo "  Setup complete."
echo ""
echo "  MANUAL STEPS REMAINING:"
echo "  1. Install k2 GPU wheel (see Step 3 above)"
echo "  2. Install kaldifeat GPU wheel"
echo "  3. Verify: python3 -c \"import k2, kaldifeat; print('OK')\""
echo "  4. Review Step 6 output to update run_finetune.sh argument names"
echo ""
echo "  AUTOMATED NEXT STEPS:"
echo "  python3 prepare_lhotse_manifests.py \\"
echo "      --data-root /workspace/SAPC2/processed \\"
echo "      --train-csv /workspace/SAPC2/processed/manifest/Train.csv \\"
echo "      --dev-csv /workspace/SAPC2/processed/manifest/Dev.csv \\"
echo "      --output-dir /workspace/finetune/data"
echo ""
echo "  bash run_finetune.sh standard"
echo "============================================"
