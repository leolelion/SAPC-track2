#!/usr/bin/env bash
# =====================================================================
# Hello-world diagnostic setup.sh.
#
# Installs NOTHING. Downloads NOTHING. Exits 0.
#
# Purpose: dump enough environment facts that, even if Model never
# loads, the ingestion stdout tells us what the Codabench eval VM
# looks like (cwd, python, filesystem, memory, hostname, env vars).
# =====================================================================
set -e

echo "[SETUP_DIAGNOSTIC] ----- BEGIN -----"
echo "[SETUP_DIAGNOSTIC] timestamp=$(date -Iseconds 2>/dev/null || date)"
echo "[SETUP_DIAGNOSTIC] cwd=$(pwd)"
echo "[SETUP_DIAGNOSTIC] script_dir=$(cd "$(dirname "$0")" && pwd)"

echo "[SETUP_DIAGNOSTIC] listing script_dir:"
ls -la "$(cd "$(dirname "$0")" && pwd)" || true

echo "[SETUP_DIAGNOSTIC] listing cwd (first 50):"
ls -la | head -50 || true

echo "[SETUP_DIAGNOSTIC] /tmp writable probe:"
if touch /tmp/hello_world_setup.probe 2>/dev/null; then
    echo "[SETUP_DIAGNOSTIC] /tmp writable=YES"
    rm -f /tmp/hello_world_setup.probe
else
    echo "[SETUP_DIAGNOSTIC] /tmp writable=NO"
fi

echo "[SETUP_DIAGNOSTIC] python: $(command -v python3 2>&1) // $(python3 --version 2>&1)"
echo "[SETUP_DIAGNOSTIC] which pip: $(command -v pip 2>&1)"
echo "[SETUP_DIAGNOSTIC] hostname=$(hostname 2>&1)"
echo "[SETUP_DIAGNOSTIC] uname=$(uname -a 2>&1)"
echo "[SETUP_DIAGNOSTIC] nproc=$(nproc 2>&1)"

echo "[SETUP_DIAGNOSTIC] free -h:"
free -h 2>&1 || true

echo "[SETUP_DIAGNOSTIC] /proc/meminfo head:"
sed -n '1,5p' /proc/meminfo 2>&1 || true

echo "[SETUP_DIAGNOSTIC] disk df -h .:"
df -h . 2>&1 || true

echo "[SETUP_DIAGNOSTIC] env subset (PATH/HOME/USER/CODA*/SAPC*/EVAL*):"
env | grep -E "^(PATH|HOME|USER|CODA|SAPC|EVAL)" | head -30 2>&1 || true

echo "[SETUP_DIAGNOSTIC] outbound network probe (huggingface.co):"
if command -v curl >/dev/null 2>&1; then
    curl -s -o /dev/null -w "[SETUP_DIAGNOSTIC] curl_status=%{http_code} time=%{time_total}\n" \
        --max-time 5 https://huggingface.co/ 2>&1 || \
        echo "[SETUP_DIAGNOSTIC] curl_failed"
else
    echo "[SETUP_DIAGNOSTIC] curl_unavailable"
fi

echo "[SETUP_DIAGNOSTIC] ----- END (no installs performed) -----"
exit 0
