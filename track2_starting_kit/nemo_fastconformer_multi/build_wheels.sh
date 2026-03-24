#!/bin/bash
# Build pre-built wheel bundle for the submission container.
#
# NeMo deps are identical between nemo_fastconformer/ and nemo_fastconformer_multi/.
# You can share a single build: run this script in either directory and copy the
# resulting wheels/ to the other, or just run it once per submission dir.
#
# Usage: bash build_wheels.sh   (requires Docker)
# Output: ./wheels/*.whl  (~150-250MB)

set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
WHEELS_DIR="${DIR}/wheels"

echo "=== Building NeMo wheel bundle for xiuwenz2/sapc2-runtime ==="
echo "    Output: ${WHEELS_DIR}"
echo ""

mkdir -p "${WHEELS_DIR}"

docker run --rm \
    -v "${WHEELS_DIR}:/wheels_out" \
    xiuwenz2/sapc2-runtime:latest \
    bash -c '
set -euo pipefail

PYTHON=""
for candidate in python3.11 python3.10; do
    if command -v "$candidate" &>/dev/null; then
        PYTHON="$candidate"; break
    fi
done
[ -z "$PYTHON" ] && { echo "ERROR: python3.11/3.10 not found"; exit 1; }
echo "Container Python: $($PYTHON --version)"

# Record system packages so we can exclude them from the bundle
$PYTHON -c "
import pkg_resources, json
pkgs = {d.project_name.lower().replace(\"-\",\"_\"): d.version for d in pkg_resources.working_set}
with open(\"/tmp/system_pkgs.json\", \"w\") as f: json.dump(pkgs, f)
print(f\"System packages: {len(pkgs)}\")
"

$PYTHON -m venv --system-site-packages /tmp/nemo_wheel_venv
/tmp/nemo_wheel_venv/bin/pip install --upgrade pip -q

echo "Installing nemo_toolkit[asr] (slow, but only once during build)..."
/tmp/nemo_wheel_venv/bin/pip install --no-cache-dir --prefer-binary \
    "omegaconf>=2.3" "huggingface_hub>=0.24" sentencepiece "nemo_toolkit[asr]>=2.5.0"

PY_TAG=$($PYTHON -c "import sys; print(f\"python{sys.version_info.major}.{sys.version_info.minor}\")")
VENV_SITE=/tmp/nemo_wheel_venv/lib/$PY_TAG/site-packages

# Find packages newly installed in venv (not inherited from system)
$PYTHON -c "
import pkg_resources, json, sys
with open(\"/tmp/system_pkgs.json\") as f: system_pkgs = json.load(f)
new_pkgs = {}
for d in pkg_resources.find_distributions(\"$VENV_SITE\"):
    name = d.project_name.lower().replace(\"-\", \"_\")
    if name not in system_pkgs:
        new_pkgs[name] = d.version
print(f\"Packages to bundle: {len(new_pkgs)}\")
for k, v in sorted(new_pkgs.items()): print(f\"  {k}=={v}\")
with open(\"/tmp/new_pkgs.txt\", \"w\") as f:
    f.writelines(f\"{k}=={v}\n\" for k, v in new_pkgs.items())
"

echo ""
echo "Downloading wheels..."
/tmp/nemo_wheel_venv/bin/pip download \
    --dest /wheels_out \
    --no-deps \
    --prefer-binary \
    -r /tmp/new_pkgs.txt

echo "Done. Wheels: $(ls /wheels_out/*.whl 2>/dev/null | wc -l)"
'

echo ""
echo "=== Done. Wheel bundle: ${WHEELS_DIR} ==="
WCOUNT=$(ls "${WHEELS_DIR}"/*.whl 2>/dev/null | wc -l | tr -d ' ')
WSIZE=$(du -sh "${WHEELS_DIR}" 2>/dev/null | cut -f1)
echo "  Wheels: ${WCOUNT}  |  Size: ${WSIZE}"
echo ""
echo "Next: run download_model.sh, then include wheels/ and model.nemo in your zip."
