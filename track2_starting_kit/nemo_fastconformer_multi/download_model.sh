#!/bin/bash
# Pre-download the 114M multi-latency model as model.nemo for bundling in the submission zip.
#
# Run after setup.sh completes. Include the resulting model.nemo in your Codabench zip.
# File size: ~430MB for nvidia/stt_en_fastconformer_hybrid_large_streaming_multi
#
# Usage:
#   bash setup.sh
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
echo "    Output: ${OUT}  (~430MB)"

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
