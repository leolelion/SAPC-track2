#!/bin/bash
# =====================================================================
# FastConformer-medium (32M) streaming — Environment Setup
#
#   Stage 1: Install NeMo (ASR) + deps
#   Stage 2: Download the .nemo checkpoint into weights/
#
# Runs once before model load (Codabench runs setup.sh before importing model.py).
# =====================================================================
set -e
export DIR="$(cd "$(dirname "$0")" && pwd)"

# =============================================================
# Stage 1: Install packages
# =============================================================
echo "=== Stage 1: Install NeMo + deps ==="
# NOTE (verify on pod): NeMo 2.7.2 is the version our 114M streaming run used
# (the `_rnnt_greedy_decode` list-return quirk was patched for it — see
# docs/results/parakeet_comparison.md). Codabench runtime is torch 2.5.0+cu124;
# confirm nemo_toolkit==2.7.2 is compatible with that torch on the pod, and
# bump if the resolver complains. Pin what we can reproduce, not "latest".
pip install -q "nemo_toolkit[asr]==2.7.2" omegaconf numpy

# =============================================================
# Stage 2: Download checkpoint (into weights/, loaded locally at runtime)
# =============================================================
echo "=== Stage 2: Download FastConformer-medium streaming checkpoint ==="
mkdir -p "$DIR/weights"
python3 - <<'PY'
import os
from omegaconf import OmegaConf
d = os.environ["DIR"]
cfg = OmegaConf.load(os.path.join(d, "config.yaml"))
name = cfg.weights.model_name
dest = os.path.join(d, cfg.weights.nemo_file)
os.makedirs(os.path.dirname(dest), exist_ok=True)
# NeMo resolves the model from NGC/HF and caches a .nemo; copy it to weights/.
import nemo.collections.asr as nemo_asr
m = nemo_asr.models.ASRModel.from_pretrained(model_name=name, map_location="cpu")
m.save_to(dest)
print(f"[setup] saved {name} -> {dest}")
PY

echo "=== setup.sh complete ==="
