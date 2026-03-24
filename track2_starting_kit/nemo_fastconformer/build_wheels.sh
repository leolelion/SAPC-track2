#!/bin/bash
# Build a pre-built wheel bundle for the submission container.
#
# Run this ONCE on a Linux x86_64 machine (or in Docker) to create ./wheels/.
# Then include ./wheels/ in your Codabench submission zip.
# setup.sh will use --no-index --find-links=./wheels/ for an offline install.
#
# Usage (recommended — uses the exact submission container):
#   bash build_wheels.sh
#
# Requirements: Docker must be installed and running.
#
# Output: ./wheels/*.whl  (~150-250MB uncompressed, ~120-200MB compressed)
# These wheels are built for: manylinux + Python 3.11 + x86_64 (matching the container).

set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
WHEELS_DIR="${DIR}/wheels"

echo "=== Building NeMo wheel bundle for xiuwenz2/sapc2-runtime ==="
echo "    Output: ${WHEELS_DIR}"
echo ""

mkdir -p "${WHEELS_DIR}"

# Run inside the exact submission container so platform/Python version matches.
# We create a venv (--system-site-packages to inherit system torch/numpy),
# install NeMo, then collect wheels only for packages NOT already in the system.
docker run --rm \
    -v "${WHEELS_DIR}:/wheels_out" \
    xiuwenz2/sapc2-runtime:latest \
    bash -c '
set -euo pipefail

# Find Python 3.11 or 3.10 (same logic as setup.sh)
PYTHON=""
for candidate in python3.11 python3.10; do
    if command -v "$candidate" &>/dev/null; then
        PYTHON="$candidate"
        break
    fi
done
if [ -z "$PYTHON" ]; then
    echo "ERROR: Could not find python3.11 or python3.10 in container."
    exit 1
fi
echo "Container Python: $($PYTHON --version)"

# Step 1: Record system-installed packages (these come from --system-site-packages)
# We exclude these from the wheel bundle to avoid bundling torch (2GB+), numpy, etc.
$PYTHON -c "
import pkg_resources, json
pkgs = {d.project_name.lower().replace(\"-\",\"_\"): d.version for d in pkg_resources.working_set}
with open(\"/tmp/system_pkgs.json\", \"w\") as f:
    json.dump(pkgs, f)
print(f\"System packages recorded: {len(pkgs)}\")
"

# Step 2: Create a venv with system site-packages (mirrors submission environment)
$PYTHON -m venv --system-site-packages /tmp/nemo_wheel_build_venv
/tmp/nemo_wheel_build_venv/bin/pip install --upgrade pip -q

# Step 3: Install NeMo (this is the slow step — only done once during build)
echo "Installing nemo_toolkit[asr] inside container venv..."
/tmp/nemo_wheel_build_venv/bin/pip install --no-cache-dir --prefer-binary \
    "omegaconf>=2.3" \
    "huggingface_hub>=0.24" \
    sentencepiece \
    "nemo_toolkit[asr]>=2.5.0"

# Step 4: Identify newly installed packages (in venv, not system)
VENV_SITE=/tmp/nemo_wheel_build_venv/lib/$($PYTHON -c "import sys; print(f\"python{sys.version_info.major}.{sys.version_info.minor}\")")/site-packages

$PYTHON -c "
import pkg_resources, json, sys

with open(\"/tmp/system_pkgs.json\") as f:
    system_pkgs = json.load(f)

# Packages installed in venv site-packages (not inherited from system)
sys.path.insert(0, \"$VENV_SITE\")
venv_pkgs = {}
for d in pkg_resources.find_distributions(\"$VENV_SITE\"):
    name = d.project_name.lower().replace(\"-\", \"_\")
    venv_pkgs[name] = d.version

new_pkgs = {k: v for k, v in venv_pkgs.items() if k not in system_pkgs}
print(f\"Packages to bundle: {len(new_pkgs)}\")
for k, v in sorted(new_pkgs.items()):
    print(f\"  {k}=={v}\")

with open(\"/tmp/new_pkgs.txt\", \"w\") as f:
    for k, v in new_pkgs.items():
        f.write(f\"{k}=={v}\n\")
"

# Step 5: Download wheels for new packages (no-deps since we resolved above)
echo ""
echo "Downloading wheels for new packages..."
/tmp/nemo_wheel_build_venv/bin/pip download \
    --dest /wheels_out \
    --no-deps \
    --prefer-binary \
    -r /tmp/new_pkgs.txt

echo ""
echo "Wheel bundle built successfully."
echo "Wheel count: $(ls /wheels_out/*.whl 2>/dev/null | wc -l)"
echo "Total size:  $(du -sh /wheels_out/*.whl 2>/dev/null | tail -1 | cut -f2 || echo unknown)"
'

echo ""
echo "=== Done. Wheel bundle at: ${WHEELS_DIR} ==="
echo ""
WCOUNT=$(ls "${WHEELS_DIR}"/*.whl 2>/dev/null | wc -l | tr -d ' ')
WSIZE=$(du -sh "${WHEELS_DIR}" 2>/dev/null | cut -f1)
echo "  Wheel count: ${WCOUNT}"
echo "  Bundle size: ${WSIZE}"
echo ""
echo "Next steps:"
echo "  1. Run download_model.sh to pre-download model.nemo (optional, saves ~5 min)"
echo "  2. Include wheels/ and (optionally) model.nemo in your Codabench zip"
echo "  3. Expected setup.sh time: ~1-2 min (vs 15-30 min without wheels)"
