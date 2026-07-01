#!/usr/bin/env bash
# Test a no-empty blank-penalty fallback on EXACTLY the Dev_diag utts that decoded empty.
# Builds a 31-utt manifest from the saved empty-id list, patches a COPY of the submission,
# decodes those utts with (a) original model (baseline: empty => 100% CER each) and
# (b) fallback model, then scores both. If fallback CER << 100%, the empties are recoverable.
set -uo pipefail

OUT=/workspace/finetune/nemo_ft
ART="$OUT/artifacts"
LOG="$OUT/fallback_test.log"; : > "$LOG"; exec > >(tee -a "$LOG") 2>&1
source /workspace/nemoenv/bin/activate

PROJ=/workspace/sapc-nemotron
LD="$PROJ/track2_starting_kit/local_decode.py"
DATA=/workspace/SAPC2
ORIG="$OUT/submission_encoder_sos_int8"
FB="$OUT/submission_encoder_sos_int8_fallback"
BOOT=/workspace/streaming_cer_bootstrap.py
EMP=/workspace/empties_analysis.py
DEVDIAG="$DATA/manifest/Dev_diag.csv"
EMPTY_IDS="$ART/devdiag_int8_empty_ids.txt"
SUBMAN=/dev/shm/devdiag_empty_subset.csv
STREAM_SMOKE=/dev/shm/Dev_streaming_1.csv

step(){ echo; echo "############### $* ###############"; }

step "Preflight"
for f in "$LD" "$ORIG/model.py" "$DEVDIAG" "$EMPTY_IDS" /workspace/patch_fallback.py; do
  test -e "$f" || { echo "MISSING $f"; exit 1; }
done
echo "empty ids: $(wc -l < "$EMPTY_IDS")"

step "Build 31-empty subset manifest"
python3 - "$DEVDIAG" "$EMPTY_IDS" "$SUBMAN" <<'PY'
import csv, sys
man, ids_f, out = sys.argv[1], sys.argv[2], sys.argv[3]
ids = {l.strip() for l in open(ids_f) if l.strip()}
rows = [r for r in csv.DictReader(open(man, newline="")) if r["id"] in ids]
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print(f"subset rows={len(rows)} (want {len(ids)})")
assert len(rows) == len(ids), "ID MISMATCH"
PY

step "Build patched fallback submission (copy + patch)"
rm -rf "$FB"; cp -r "$ORIG" "$FB"
python3 /workspace/patch_fallback.py "$FB/model.py"
python3 -m py_compile "$FB/model.py" && echo "py_compile OK"
grep -q "_fallback_decode" "$FB/model.py" && echo "fallback present"

python3 - "$DATA/manifest/Dev_streaming.csv" "$STREAM_SMOKE" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1], newline="")))
with open(sys.argv[2], "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerow(rows[0])
PY

decode () {  # $1=tag $2=sub-dir $3=out-csv
  ( cd "$2" && SAPC2_THREADS=8 python3 "$LD" --submission-dir "$2" \
      --manifest-csv "$SUBMAN" --streaming-manifest-csv "$STREAM_SMOKE" \
      --streaming-interval 0 --data-root "$DATA" \
      --out-csv "$3" --out-partial-json "/dev/shm/${1}_part.json" ) 2>&1 | tail -4
}

step "Decode 31 empties with ORIGINAL model (baseline)"
decode orig "$ORIG" /dev/shm/fb_orig.csv
step "Decode 31 empties with FALLBACK model"
decode fb "$FB" /dev/shm/fb_patched.csv

step "Score ORIGINAL on the 31 (expect ~empty/100%)"
python3 "$BOOT" --manifest-csv "$SUBMAN" --hyp-csv /dev/shm/fb_orig.csv --utils-dir "$PROJ/utils" \
  --out-json "$ART/fb_orig_31.json"
step "Score FALLBACK on the 31"
python3 "$BOOT" --manifest-csv "$SUBMAN" --hyp-csv /dev/shm/fb_patched.csv --utils-dir "$PROJ/utils" \
  --out-json "$ART/fb_patched_31.json"
python3 "$EMP" --manifest-csv "$SUBMAN" --hyp-csv /dev/shm/fb_patched.csv --utils-dir "$PROJ/utils" \
  --out-json "$ART/fb_patched_31_empties.json"

step "Sample fallback hypotheses vs refs (first 8)"
python3 - "$SUBMAN" /dev/shm/fb_patched.csv <<'PY'
import csv, sys
man = {r["id"]: r for r in csv.DictReader(open(sys.argv[1], newline=""))}
hyp = {r["id"]: r.get("raw_hypos","") for r in csv.DictReader(open(sys.argv[2], newline=""))}
for i, uid in enumerate(list(hyp)[:8]):
    ref = man[uid].get("norm_text_without_disfluency") or man[uid].get("text","")
    print(f"[{man[uid].get('etiology','?')}] REF: {ref[:80]}")
    print(f"        HYP: {hyp[uid][:80]}")
PY

echo "FALLBACK_TEST_DONE"
