#!/usr/bin/env bash
# run_tests.sh — Run all finetuning tests on the pod.
#
# Usage:
#   bash finetuning/tests/run_tests.sh
#   bash finetuning/tests/run_tests.sh --data-root /workspace/SAPC2
#   bash finetuning/tests/run_tests.sh --suite env      # just environment
#   bash finetuning/tests/run_tests.sh --suite data     # just data
#   bash finetuning/tests/run_tests.sh --suite model    # just model/onnx
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

DATA_ROOT="/workspace/SAPC2"
CUTS_DIR="/workspace/finetune/data"
SUITE="all"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        --cuts-dir)  CUTS_DIR="$2";  shift 2 ;;
        --suite)     SUITE="$2";     shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

PASS=0
FAIL=0
ERRORS=()

run_suite() {
    local name="$1"
    local script="$2"
    shift 2
    echo ""
    echo "════════════════════════════════════════"
    echo "  Suite: $name"
    echo "════════════════════════════════════════"
    if python3 "$script" "$@" -v 2>&1; then
        PASS=$((PASS + 1))
        echo "  ✓ $name PASSED"
    else
        FAIL=$((FAIL + 1))
        ERRORS+=("$name")
        echo "  ✗ $name FAILED"
    fi
}

if [[ "$SUITE" == "all" || "$SUITE" == "env" ]]; then
    run_suite "Environment" "finetuning/tests/test_env.py"
fi

if [[ "$SUITE" == "all" || "$SUITE" == "data" ]]; then
    run_suite "Data" "finetuning/tests/test_data.py" \
        --data-root "$DATA_ROOT" --cuts-dir "$CUTS_DIR"
fi

if [[ "$SUITE" == "all" || "$SUITE" == "model" ]]; then
    run_suite "Model/ONNX" "finetuning/tests/test_model.py"
fi

echo ""
echo "════════════════════════════════════════"
echo "  Results: ${PASS} passed, ${FAIL} failed"
if [[ ${#ERRORS[@]} -gt 0 ]]; then
    echo "  Failed suites: ${ERRORS[*]}"
    exit 1
fi
echo "  All tests passed!"
echo "════════════════════════════════════════"
