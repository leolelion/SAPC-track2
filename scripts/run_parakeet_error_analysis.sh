#!/usr/bin/env bash
# Parakeet Arm A — empty-utterance ROOT-CAUSE battery (run on the pod, BEFORE Arm B).
# ---------------------------------------------------------------------------------------
# Purpose: answer "why do ~11% of severe utts come back empty, and would Arm B fix it?"
#   with CHEAP evidence, so we do not spend ~2.5h GPU on joint-unfreeze on faith.
#   All three steps are read-only w.r.t. training; total runtime is minutes (offline
#   transcribe of ~48 short utts on CPU).
#
# ORDER MATTERS — do STEP 0 (data rescue) first: the Arm A .nemo lives on a volumeInGb=0
#   pod that can be reclaimed. Copy it OFF-POD before anything else (memory
#   parakeet-ft-wrapper-streaming "POD STORAGE / DATA-LOSS RISK").
#
# This script FAILS LOUD on any missing input (no silent fallback). Override paths via env:
#   NEMO=...  MANIFEST=...  HYP_CSV=...  UTILS_DIR=...  DATA_ROOT=...  OUT_DIR=...
#
# Usage:  bash scripts/run_parakeet_error_analysis.sh
# ---------------------------------------------------------------------------------------
set -euo pipefail

# ---- paths (pod defaults; override via env) -------------------------------------------
NEMO="${NEMO:-/workspace/parakeet_ft/armA_full/ft_smoke_encoder_only.nemo}"
MANIFEST="${MANIFEST:-/workspace/SAPC2/manifest/Dev_diag.csv}"
HYP_CSV="${HYP_CSV:-/workspace/parakeet_ft/Dev_diag.armA.predict.csv}"   # <-- the ARM A Dev_diag accuracy-pass csv
UTILS_DIR="${UTILS_DIR:-/workspace/SAPC-template/utils}"
DATA_ROOT="${DATA_ROOT:-/workspace/SAPC2}"
OUT_DIR="${OUT_DIR:-/workspace/parakeet_ft/error_analysis}"
ATT="${ATT:-70 1}"
HERE="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$OUT_DIR"
EMPTY_IDS="$OUT_DIR/empty_ids.txt"

echo "==================================================================================="
echo "STEP 0 (DO BY HAND, NOT AUTOMATED): rescue the Arm A checkpoint OFF-POD first."
echo "  This pod has NO network volume — a stopped no-volume pod can be reclaimed & lost."
echo "  From your laptop:  scp -P <port> root@<host>:$NEMO  ./artifacts/"
echo "  Do NOT proceed to Arm B training until this .nemo is safely copied."
echo "==================================================================================="
echo

# ---- fail-loud input checks -----------------------------------------------------------
missing=0
for f in "$MANIFEST" "$HYP_CSV" "$NEMO"; do
  if [[ ! -e "$f" ]]; then echo "[MISSING] $f"; missing=1; fi
done
if [[ ! -d "$UTILS_DIR" ]]; then echo "[MISSING dir] $UTILS_DIR (official normalizer -> fallback will be used)"; fi
if [[ "$missing" == "1" ]]; then
  echo
  echo "One or more inputs are missing. If the HYP csv name differs, find it with:"
  echo "  ls -la /workspace/parakeet_ft/ | grep -i diag"
  echo "and re-run with  HYP_CSV=/full/path bash $0"
  exit 2
fi

# ---- STEP 1: empty decomposition + empty-id list (reuse the proven scorer) -------------
echo "### STEP 1: empties_analysis.py — how much CER is 'emitted nothing' + which utts"
python3 "$HERE/empties_analysis.py" \
  --manifest-csv "$MANIFEST" \
  --hyp-csv      "$HYP_CSV" \
  --utils-dir    "$UTILS_DIR" \
  --out-json     "$OUT_DIR/empties_analysis.json" \
  --empty-ids-out "$EMPTY_IDS"
echo

# ---- STEP 2: T1/T2/T4 probe (NeMo-free) -----------------------------------------------
echo "### STEP 2: parakeet_empty_probe.py — T1 error-mix / T2 energy / T4 speaker-spread"
python3 "$HERE/parakeet_empty_probe.py" \
  --manifest-csv "$MANIFEST" \
  --hyp-csv      "$HYP_CSV" \
  --utils-dir    "$UTILS_DIR" \
  --data-root    "$DATA_ROOT" \
  --out-json     "$OUT_DIR/empty_probe.json" \
  --empty-ids-out "$OUT_DIR/empty_ids_probe.txt"
echo

# ---- STEP 3: T3/T5 offline recovery (THE decisive test; needs NeMo) --------------------
echo "### STEP 3: parakeet_offline_recovery.py — do the empties come back OFFLINE?"
# shellcheck disable=SC2086
python3 "$HERE/parakeet_offline_recovery.py" \
  --nemo       "$NEMO" \
  --manifest   "$MANIFEST" \
  --empty-ids  "$EMPTY_IDS" \
  --data-root  "$DATA_ROOT" \
  --att        $ATT \
  --out-json   "$OUT_DIR/offline_recovery.json"
echo

# ---- decision rule --------------------------------------------------------------------
cat <<'RULE'
=======================================================================================
DECISION RULE (combine the three read-outs before committing GPU to Arm B):

  A. offline_recovery recovered >~60%
       -> STREAMING-path fault (H7 train->stream gap), NOT the joint.
       -> Arm B will NOT help the empties. Instead: raise lookahead/left-context or fix
          the online emission policy. VETO Arm B-for-empties.

  B. offline still_empty >~60%  AND  empty_probe T1 del:sub > ~1.3  (deletion-heavy)
       -> true blank given full context, frozen-joint blank bias.
       -> Arm B (joint-unfreeze) is on-target. Proceed (drop severity/aug per v2 plan).

  C. empty_probe T2: empties markedly QUIETER (RMS) / shorter than non-empties
       -> encoder-energy sensitivity. Try the FREE Pass-1 RMS-normalize first (no GPU);
          re-measure empties before any Arm B.

  D. offline eou_only high
       -> T5 EOU-misfire (this is the _eou variant). Fix EOU handling, not the joint.

  E. empty_probe T4: empties spread across many speakers with high error everywhere
       -> data-bound / genuinely unintelligible; no cheap lever recovers them. Re-scope.

Artifacts: $OUT_DIR/{empties_analysis,empty_probe,offline_recovery}.json
=======================================================================================
RULE
echo "PARAKEET_ERROR_ANALYSIS_DONE"
