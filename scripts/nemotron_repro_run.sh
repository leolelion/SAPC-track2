#!/usr/bin/env bash
# Faithful Nemotron reproduction via the organizers' REAL local_decode.py.
# Usage (on pod): bash nemotron_repro_run.sh <ACC_manifest> <STREAM_manifest>
#   e.g. bash nemotron_repro_run.sh Dev_10 Dev_10      (smoke)
#        bash nemotron_repro_run.sh Dev_100 Dev_10     (CER number; Pass2 bounded to 10)
# Prints hyp-vs-ref for the first rows so we can eyeball truncation/garbling.
set -uo pipefail
ACC="${1:-Dev_10}"; STR="${2:-Dev_10}"
SUB=/workspace/finetune/nemo_submission
LD=/workspace/sapc-nemotron/track2_starting_kit/local_decode.py
MANDIR=/workspace/SAPC2/manifest
MAN="$MANDIR/$ACC.csv"; SMAN="$MANDIR/$STR.csv"
[ -f "$MAN" ]  || { echo "no manifest $MAN"; exit 2; }
REL=$(sed -n '2p' "$MAN" | cut -d, -f4)
DR=""
for c in /workspace/SAPC2 /workspace/finetune/data /workspace/finetune /workspace; do
  [ -e "$c/$REL" ] && { DR="$c"; break; }
done
[ -n "$DR" ] || { echo "DATA_ROOT not found for rel=$REL"; exit 3; }
echo "ACC=$MAN  STR=$SMAN  DATA_ROOT=$DR"
OUT=/dev/shm/${ACC}_repro
cd "$SUB"
SAPC2_THREADS=4 python3 "$LD" --submission-dir "$SUB" --manifest-csv "$MAN" \
  --streaming-manifest-csv "$SMAN" --data-root "$DR" \
  --out-csv "${OUT}.csv" --out-partial-json "${OUT}.json"
echo "DECODE_EXIT=$?"
echo "=== hyp vs ref (first 12) ==="
python3 - "$MAN" "${OUT}.csv" <<'PY'
import csv,sys
man,hyp=sys.argv[1],sys.argv[2]
refs={}
with open(man) as f:
    for r in csv.DictReader(f): refs[r['id']]=r.get('norm_text_without_disfluency','')
with open(hyp) as f:
    for i,r in enumerate(csv.DictReader(f)):
        if i>=12: break
        print("ID ", r['id'][:20])
        print("  HYP:", r['raw_hypos'])
        print("  REF:", refs.get(r['id'],'<no-ref>'))
PY
echo REPRO_DONE
