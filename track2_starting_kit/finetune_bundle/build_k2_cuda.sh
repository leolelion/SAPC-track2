#!/bin/bash
# Build k2 from source with CUDA support on H200 (sm_90).
#
# Required because:
#   - No pre-built CUDA wheels exist for PyTorch >= 2.1
#   - k2.rnnt_loss_pruned AND k2.swoosh_l/r are CUDA-only kernels
#   - The CPU wheel causes segfaults during both forward pass AND loss computation
#
# Usage:
#   bash build_k2_cuda.sh
#
# Time: ~15-20 minutes on H200 with 100 vCPUs
#
set -euo pipefail

echo "============================================"
echo "  Build k2 from source (CUDA sm_90 / H200)"
echo "============================================"

# ── Verify CUDA is available ──
echo ""
echo "=== CUDA check ==="
if ! command -v nvcc &>/dev/null; then
    # Try common locations
    for candidate in /usr/local/cuda/bin/nvcc /usr/bin/nvcc; do
        if [ -f "${candidate}" ]; then
            export PATH="$(dirname ${candidate}):${PATH}"
            echo "  Found nvcc at ${candidate}"
            break
        fi
    done
fi
nvcc --version || { echo "ERROR: nvcc not found. Check CUDA installation."; exit 1; }

# Set CUDA_HOME if not set
if [ -z "${CUDA_HOME:-}" ]; then
    if [ -d /usr/local/cuda ]; then
        export CUDA_HOME=/usr/local/cuda
    else
        export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
    fi
fi
export CUDACXX="${CUDA_HOME}/bin/nvcc"
echo "  CUDA_HOME: ${CUDA_HOME}"
echo "  CUDACXX:   ${CUDACXX}"

# ── Verify PyTorch CUDA ──
python3 -c "
import torch
print(f'  PyTorch:  {torch.__version__}')
print(f'  CUDA:     {torch.version.cuda}')
print(f'  GPU:      {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NOT FOUND\"}')
assert torch.cuda.is_available(), 'CUDA not available in PyTorch'
"

# ── Uninstall existing k2 ──
echo ""
echo "=== Removing existing k2 ==="
pip uninstall -y k2 2>/dev/null || echo "  (k2 was not installed)"

# ── Clone k2 ──
K2_SRC="/workspace/k2_source"
if [ -d "${K2_SRC}/.git" ]; then
    echo ""
    echo "=== k2 source already exists at ${K2_SRC} ==="
    echo "  To rebuild from scratch: rm -rf ${K2_SRC}"
else
    echo ""
    echo "=== Cloning k2 ==="
    git clone --depth 1 https://github.com/k2-fsa/k2.git "${K2_SRC}"
fi

cd "${K2_SRC}"

# ── Build ──
# sm_90 = H200 (and H100)
# -j$(nproc) uses all available CPUs (RunPod H200 pods have 40-100 vCPUs)
echo ""
echo "=== Building k2 (this takes ~15-20 minutes) ==="
echo "  Architecture: sm_90 (H200/H100)"
echo "  Parallelism:  $(nproc) cores"
echo "  Started at:   $(date)"
echo ""

export K2_CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=90"
export K2_MAKE_ARGS="-j$(nproc)"

# Install cmake/ninja if not present
pip install --quiet cmake ninja

pip install . 2>&1 | tee /workspace/k2_build.log

echo ""
echo "  Build finished at: $(date)"

# ── Verify ──
echo ""
echo "=== Verifying k2 CUDA build ==="
python3 -c "
import k2
print(f'k2 version: {k2.__version__}')
print(f'k2 CUDA:    {k2.with_cuda}')
assert k2.with_cuda, 'Build succeeded but CUDA not enabled in k2 — check /workspace/k2_build.log'

import torch
# Quick functional test of RNN-T loss
logits = torch.randn(2, 10, 5, 20, requires_grad=True).cuda()
targets = torch.randint(1, 20, (2, 4)).cuda()
logit_lengths = torch.tensor([10, 8]).cuda()
target_lengths = torch.tensor([4, 3]).cuda()
loss = k2.rnnt_loss_simple(
    lm=logits[:,:,:,:1].squeeze(-1),
    am=logits[:,:,:,1:].reshape(2, 10, 5, -1),
    symbols=targets,
    termination_symbol=0,
    boundary=None,
    reduction='sum',
) if hasattr(k2, 'rnnt_loss_simple') else None
print('k2.rnnt_loss_simple: available' if loss is not None else 'k2.rnnt_loss_simple: not tested (API may differ)')

# Test swoosh functions
from k2 import swoosh_l, swoosh_r
x = torch.randn(4, 8).cuda()
print(f'swoosh_l output shape: {swoosh_l(x).shape}')
print(f'swoosh_r output shape: {swoosh_r(x).shape}')
print()
print('=== k2 CUDA build: VERIFIED OK ===')
"

echo ""
echo "============================================"
echo "  k2 CUDA build complete!"
echo ""
echo "  Next: bash run_finetune.sh standard"
echo "============================================"
