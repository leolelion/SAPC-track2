#!/usr/bin/env bash
# =====================================================================
# nemotron_repro_preflight.sh  (READ-ONLY — discovers + asserts, runs nothing heavy)
# Purpose: gather everything needed to run the FAITHFUL reproduction of the
# Nemotron Test1 result on Dev via the organizers' REAL local_decode.py +
# evaluate.sh, so we compose the exact run command from real output instead of
# guessing paths. Per CLAUDE.md "faithful validation gate" + "cost gate".
#
# Usage (on the pod):  bash nemotron_repro_preflight.sh
# It only prints. No decode, no scoring, no writes outside /tmp.
# =====================================================================
set -uo pipefail
ROOT="${ROOT:-/workspace}"
hr(){ printf '\n========== %s ==========\n' "$1"; }

hr "HOST / RESOURCES"
echo "date: $(date -u)"; echo "cwd: $(pwd)"
nproc 2>/dev/null | sed 's/^/nproc: /'
df -h /workspace /dev/shm 2>/dev/null

hr "REAL HARNESS: local_decode.py (organizers')"
# There may be more than one copy; show all, prefer track2_starting_kit / sapc.
mapfile -t LD < <(find "$ROOT" -name local_decode.py -not -path '*/.git/*' 2>/dev/null)
printf 'found %s copy/copies:\n' "${#LD[@]}"; printf '  %s\n' "${LD[@]:-<none>}"
for f in "${LD[@]}"; do
  echo "--- argparse for: $f ---"
  grep -nE 'add_argument|def main|accept_chunk|input_finished|read|load|wav|float32|int16|sample' "$f" 2>/dev/null | sed -n '1,60p'
done

hr "EVALUATE + SCORER"
find "$ROOT" -name evaluate.sh -not -path '*/.git/*' 2>/dev/null | sed 's/^/evaluate.sh: /'
find "$ROOT" -path '*utils/compute_metrics.py' -not -path '*/.git/*' 2>/dev/null | sed 's/^/compute_metrics: /'
find "$ROOT" -path '*normalize_hyp.py' -not -path '*/.git/*' 2>/dev/null | sed 's/^/normalize_hyp: /'

hr "NEMOTRON SUBMISSION DIR (model.py + weights/)"
mapfile -t SUB < <(find "$ROOT" -name model.py -path '*nemo*' -not -path '*/.git/*' 2>/dev/null)
printf '  %s\n' "${SUB[@]:-<none>}"
for d in $(printf '%s\n' "${SUB[@]}" | xargs -r -n1 dirname | sort -u); do
  echo "--- $d ---"; ls -la "$d" 2>/dev/null
  echo "  weights:"; ls -la "$d/weights" 2>/dev/null | sed -n '1,20p'
done

hr "DEV MANIFESTS + DATA ROOT (where the wavs live)"
find "$ROOT" -name 'Dev*.csv' -not -path '*/.git/*' 2>/dev/null | sed 's/^/manifest: /'
echo "--- header + 2 rows of each Dev manifest ---"
for m in $(find "$ROOT" -name 'Dev*.csv' -not -path '*/.git/*' 2>/dev/null); do
  echo ">> $m"; sed -n '1,3p' "$m" 2>/dev/null
done
echo "--- sample audio_filepath resolves? (first Dev.csv) ---"
M1="$(find "$ROOT" -name 'Dev.csv' -not -path '*/.git/*' 2>/dev/null | head -n1)"
echo "Dev.csv = ${M1:-<none>}"
[ -n "${M1:-}" ] && awk -F, 'NR==2{print "row2 audio_filepath field(s):"; for(i=1;i<=NF;i++) if($i ~ /\.wav|\.flac/) print "  col"i"="$i}' "$M1"

hr "PRIOR RESULTS (persisted, survives restart)"
find "$ROOT" -path '*e1_results*' -name '*nemotron*' 2>/dev/null | sed 's/^/  /'
cat "$ROOT/finetune/eval/e1_results/nemotron_dev_diagnostic.txt" 2>/dev/null

hr "RUNTIMEFIX ZIP (Codex's, on pod)"
find "$ROOT" -name 'nemo_submission*worker*runtimefix*.zip' 2>/dev/null | sed 's/^/  /'

hr "DONE"
echo "Next: compose the faithful subset run from the paths above:"
echo "  local_decode.py --submission-dir <SUB> --manifest-csv <subset Dev.csv> \\"
echo "    --streaming-manifest-csv <Dev_streaming.csv> --data-root <DATA_ROOT> \\"
echo "    --out-csv /dev/shm/dev_repro.csv --out-partial-json /dev/shm/dev_repro.json"
echo "  then evaluate.sh --split Dev --hyp-csv /dev/shm/dev_repro.csv --start_stage 1 --stop_stage 2"
