#!/bin/bash
# =====================================================================
# parakeet_realtime (120M) streaming — Environment Setup
#
#   Stage 1: Install NeMo (ASR) + deps
#   Stage 2: Download the base .nemo checkpoint into weights/
#
# Runs once before model load (Codabench runs setup.sh before importing model.py).
# For the FINE-TUNED deploy, replace weights/parakeet_realtime.nemo with the FT
# checkpoint produced by scripts/nemo_finetune_v2.py before packaging.
# =====================================================================
set -e
export DIR="$(cd "$(dirname "$0")" && pwd)"

# =============================================================
# Stage 1: Install packages
# =============================================================
echo "=== Stage 1: Install NeMo + deps ==="
# >>> VERIFY-ON-POD <<< NeMo version. 2.7.2 is what our 114M/parakeet sweep used
# (the list-return quirk is patched for it — docs/results/parakeet_comparison.md).
# Codabench runtime is torch 2.5.0+cu124; confirm nemo_toolkit==2.7.2 resolves
# against that torch on the pod and bump only if the resolver complains.
pip install -q "nemo_toolkit[asr]==2.7.2" omegaconf numpy

# =============================================================
# Stage 2: Download checkpoint (into weights/, loaded locally at runtime)
# =============================================================
echo "=== Stage 2: Download parakeet_realtime base checkpoint ==="
mkdir -p "$DIR/weights"
python3 - <<'PY'
import os
from omegaconf import OmegaConf
d = os.environ["DIR"]
cfg = OmegaConf.load(os.path.join(d, "config.yaml"))
name = cfg.weights.model_name
dest = os.path.join(d, cfg.weights.nemo_file)
os.makedirs(os.path.dirname(dest), exist_ok=True)
if os.path.exists(dest):
    print(f"[setup] {dest} already present (fine-tuned ckpt?) — not overwriting.")
else:
    import nemo.collections.asr as nemo_asr
    # generic ASRModel auto-dispatches the subclass (RNNT/TDT/Hybrid)
    m = nemo_asr.models.ASRModel.from_pretrained(model_name=name, map_location="cpu")
    m.save_to(dest)
    print(f"[setup] saved {name} -> {dest}")
PY

echo "=== setup.sh complete ==="
