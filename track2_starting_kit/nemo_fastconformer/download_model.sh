#!/bin/bash
# Pre-download model weights as model.nemo for bundling in the submission zip.
#
# Run this ONCE after setup.sh has completed (requires the venv + NeMo installed).
# Include the resulting model.nemo in your Codabench submission zip.
# setup.sh and model.py will detect it and skip the HuggingFace download.
#
# File size: ~130MB for nvidia/stt_en_fastconformer_hybrid_medium_streaming_80ms
#
# Usage:
#   bash setup.sh         # install NeMo first
#   bash download_model.sh

set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="${DIR}/venv"
OUT="${DIR}/model.nemo"

if [ ! -f "${VENV}/bin/python3" ]; then
    echo "ERROR: venv not found at ${VENV}. Run setup.sh first."
    exit 1
fi

if [ -f "${OUT}" ]; then
    echo "model.nemo already exists at ${OUT} ($(du -sh "${OUT}" | cut -f1))."
    echo "Delete it and re-run if you want to re-download."
    exit 0
fi

MODEL_NAME="$("${VENV}/bin/python3" -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('${DIR}/config.yaml')
print(cfg.model.name)
")"

echo "=== Downloading model: ${MODEL_NAME} ==="
echo "    Output: ${OUT}"

"${VENV}/bin/python3" - <<PYEOF
import nemo.collections.asr as nemo_asr

model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(
    model_name="${MODEL_NAME}",
    map_location="cpu",
)
params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Downloaded: {params:.1f}M params")
print("Saving to ${OUT} ...")
model.save_to("${OUT}")
print("Saved.")
PYEOF

SIZE=$(du -sh "${OUT}" | cut -f1)
echo "=== Done. model.nemo saved: ${OUT} (${SIZE}) ==="
echo ""
echo "Include this file in your Codabench submission zip."
echo "setup.sh and model.py will load it directly (skipping HuggingFace download)."
