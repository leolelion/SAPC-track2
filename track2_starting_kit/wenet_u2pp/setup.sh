#!/bin/bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="${DIR}/venv"
WEIGHTS_DIR="${DIR}/weights"

echo "=== WeNet U2++: creating clean venv at ${VENV} ==="
python3 -m venv "${VENV}"
"${VENV}/bin/pip" install --upgrade pip -q

echo "=== WeNet U2++: installing dependencies ==="
# WeNet is not on PyPI. Inference uses torch.jit.load directly on the
# exported final.zip — no wenet Python package needed at inference time.
"${VENV}/bin/pip" install --no-cache-dir \
    "omegaconf>=2.3" \
    torchaudio

echo "=== WeNet U2++: verifying installation ==="
"${VENV}/bin/python3" -c "
import torch, torchaudio, omegaconf
print('torch:', torch.__version__)
print('torchaudio:', torchaudio.__version__)
print('omegaconf:', omegaconf.__version__)
print('All inference deps OK (wenet not needed — uses torch.jit.load)')
"

echo "=== WeNet U2++: downloading pretrained LibriSpeech U2++ Conformer ==="
mkdir -p "${WEIGHTS_DIR}"

"${VENV}/bin/python3" - "${WEIGHTS_DIR}" << 'PYEOF'
import os
import sys
import tarfile
import urllib.request

weights_dir = sys.argv[1]

# Check if already downloaded
if os.path.exists(os.path.join(weights_dir, "final.zip")):
    print("Model already downloaded, skipping.")
    sys.exit(0)

# WeNet model zoo URL for LibriSpeech U2++ Conformer
# See: https://github.com/wenet-e2e/wenet/blob/main/docs/pretrained_models.en.md
url = "https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/librispeech/20210610_u2pp_conformer_libtorch.tar.gz"

tarpath = os.path.join(weights_dir, "model.tar.gz")
print(f"Downloading from {url} ...")
try:
    urllib.request.urlretrieve(url, tarpath)
    print("Extracting ...")
    with tarfile.open(tarpath, "r:gz") as tar:
        # Extract directly into weights_dir (strip leading path components)
        for member in tar.getmembers():
            # Strip the top-level directory from the tar archive
            parts = member.name.split("/", 1)
            if len(parts) > 1 and parts[1]:
                member.name = parts[1]
                tar.extract(member, weights_dir)
    os.remove(tarpath)
    print("Download complete.")
    print("Files in weights/:")
    for f in sorted(os.listdir(weights_dir)):
        print(f"  {f}")
except Exception as e:
    print(f"ERROR: Download failed: {e}")
    print()
    print("Manual download instructions:")
    print("  1. Visit: https://github.com/wenet-e2e/wenet/blob/main/docs/pretrained_models.en.md")
    print("  2. Download the LibriSpeech U2++ Conformer libtorch model")
    print(f"  3. Extract final.zip, train.yaml, words.txt to: {weights_dir}/")
    sys.exit(1)
PYEOF

echo "=== WeNet U2++: listing weights ==="
ls -la "${WEIGHTS_DIR}/"

echo "=== WeNet U2++: setup complete ==="
