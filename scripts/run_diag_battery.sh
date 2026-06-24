#!/usr/bin/env bash
# Autonomous full-picture Nemotron diagnostic. Robust: a failing step logs and continues.
# Writes everything to RESULTS.txt. Run on pod: nohup bash run_diag_battery.sh &
set -uo pipefail
OUT=/workspace/finetune/eval/diag_battery; mkdir -p "$OUT"
RES="$OUT/RESULTS.txt"; : > "$RES"
exec > >(tee -a "$RES") 2>&1
step(){ echo; echo "############### $* ###############"; }

MANDIR=/workspace/SAPC2/manifest
LD=/workspace/sapc-nemotron/track2_starting_kit/local_decode.py
NEMO=/workspace/finetune/nemo_submission
ZF=/workspace/finetune/zf_a1_sub
UT=/workspace/sapc-nemotron/utils

step "1. BUILD STRATIFIED MANIFEST"
python3 /workspace/build_diag_manifest.py "$MANDIR/Dev.csv" "$MANDIR/Dev_diag.csv" || echo "STEP1_FAIL"

REL=$(sed -n '2p' "$MANDIR/Dev_diag.csv" | cut -d, -f4); DR=""
for c in /workspace/SAPC2 /workspace/finetune/data /workspace/finetune /workspace; do
  [ -e "$c/$REL" ] && { DR="$c"; break; }; done
echo "DATA_ROOT=$DR"

step "2. DECODE NEMOTRON (int8) on diag set"
( cd "$NEMO" && SAPC2_THREADS=8 python3 "$LD" --submission-dir "$NEMO" \
  --manifest-csv "$MANDIR/Dev_diag.csv" --streaming-manifest-csv "$MANDIR/Dev_10.csv" \
  --data-root "$DR" --out-csv /dev/shm/diag_nemo.csv --out-partial-json /dev/shm/_n.json ) \
  && echo NEMO_DECODE_OK || echo NEMO_DECODE_FAIL

step "3. SETUP + DECODE ZIPFORMER A1 (exp A)"
bash /workspace/setup_zf_sub.sh || true
if [ -f "$ZF/model.py" ] && [ -f "$ZF/tokens.txt" ]; then
  ( cd "$ZF" && python3 "$LD" --submission-dir "$ZF" \
    --manifest-csv "$MANDIR/Dev_diag.csv" --streaming-manifest-csv "$MANDIR/Dev_10.csv" \
    --data-root "$DR" --out-csv /dev/shm/diag_zf.csv --out-partial-json /dev/shm/_z.json ) \
    && echo ZF_DECODE_OK || echo ZF_DECODE_FAIL
else
  echo "ZF_SKIP: model.py/tokens.txt not staged in $ZF"
fi

step "4. CROSS-TAB (length x severity + head-to-head rescue)"
ZFARG=""; [ -f /dev/shm/diag_zf.csv ] && ZFARG="zf=/dev/shm/diag_zf.csv"
( cd "$UT" && python3 /workspace/diag_crosstab.py "$MANDIR/Dev_diag.csv" \
  nemo=/dev/shm/diag_nemo.csv $ZFARG ) || echo CROSSTAB_FAIL

# choose probe ids: shortest empty, longest empty, median empty, + a good control
IDS=$(python3 - <<'PY'
import csv
man={r['id']:r for r in csv.DictReader(open('/workspace/SAPC2/manifest/Dev_diag.csv'))}
hyp={r['id']:r['raw_hypos'] for r in csv.DictReader(open('/dev/shm/diag_nemo.csv'))}
emp=sorted((float(man[u]['duration']),u) for u in hyp if u in man and not hyp[u].strip())
good=sorted((float(man[u]['duration']),u) for u in hyp if u in man and hyp[u].strip())
out=[]
if emp: out+=[emp[0][1], emp[-1][1]]
if len(emp)>2: out.append(emp[len(emp)//2][1])
if good: out.append(good[len(good)//2][1])
print(' '.join(dict.fromkeys(out)))
PY
)
echo "PROBE_IDS=$IDS"

step "5. (B) BLANK-PROBABILITY PROBE on empties (+ control)"
( cd "$NEMO" && python3 /workspace/nemotron_blank_probe.py "$MANDIR/Dev_diag.csv" "$DR" $IDS ) || echo PROBE_FAIL

step "5b. BEAM/BLANK-PENALTY RECOVERY on empties"
( cd "$NEMO" && python3 /workspace/nemotron_altdecode_empties.py "$MANDIR/Dev_diag.csv" "$DR" $IDS ) || echo ALTDECODE_FAIL

step "6. (C) fp32 vs int8 availability"
ls -la "$NEMO/weights/"*.onnx* 2>&1
echo "(C requires an fp32 encoder; if only int8 present, fetch danielbodart fp32 variant later.)"

step "BATTERY_DONE"
