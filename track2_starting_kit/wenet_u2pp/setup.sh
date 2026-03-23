#!/bin/bash
# WeNet U2++: download pretrained LibriSpeech U2++ Conformer weights.
set -euo pipefail

# Install omegaconf if missing (not present in all competition images)
python3 -c "import omegaconf" 2>/dev/null || pip install -q omegaconf

DIR="$(cd "$(dirname "$0")" && pwd)"
WEIGHTS_DIR="${DIR}/weights"

echo "=== WeNet U2++: downloading pretrained LibriSpeech U2++ Conformer ==="
mkdir -p "${WEIGHTS_DIR}"

python3 - "${WEIGHTS_DIR}" << 'PYEOF'
import os
import sys
import tarfile
import urllib.request

weights_dir = sys.argv[1]

# Check if already downloaded
if os.path.exists(os.path.join(weights_dir, "final.zip")):
    print("Model already downloaded, skipping.")
    sys.exit(0)

# WeNet model zoo URL for LibriSpeech U2++ Conformer (2022-05-06 release)
# Contains: final.zip, train.yaml, units.txt, global_cmvn
url = "https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/librispeech/20220506_u2pp_conformer_libtorch.tar.gz"

tarpath = os.path.join(weights_dir, "model.tar.gz")
print(f"Downloading from {url} ...")
try:
    urllib.request.urlretrieve(url, tarpath)
    print("Extracting ...")
    with tarfile.open(tarpath, "r:gz") as tar:
        # Strip the top-level directory from the tar archive
        for member in tar.getmembers():
            parts = member.name.split("/", 1)
            if len(parts) > 1 and parts[1]:
                member.name = parts[1]
                tar.extract(member, weights_dir)
    os.remove(tarpath)
    print("Download complete.")
    print("Files in weights/:")
    for f in sorted(os.listdir(weights_dir)):
        print(f"  {f}")
    # Warn if global_cmvn is missing (causes all-blank CTC outputs)
    if not os.path.exists(os.path.join(weights_dir, "global_cmvn")):
        print("WARNING: global_cmvn not found in tarball — CTC outputs may be all-blank.")
except Exception as e:
    print(f"ERROR: Download failed: {e}")
    print()
    print("Manual download instructions:")
    print("  1. Visit: https://github.com/wenet-e2e/wenet/blob/main/docs/pretrained_models.en.md")
    print("  2. Download the LibriSpeech U2++ Conformer libtorch model")
    print(f"  3. Extract final.zip, train.yaml, units.txt, global_cmvn to: {weights_dir}/")
    sys.exit(1)
PYEOF

echo "=== WeNet U2++: listing weights ==="
ls -la "${WEIGHTS_DIR}/"

echo "=== WeNet U2++: setup complete ==="
