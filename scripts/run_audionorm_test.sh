#!/usr/bin/env bash
# Test input gain normalization (rms/peak) on the 31 Dev_diag empties. If quiet-audio is the
# cause, normalization should drop empties and recover real text. CPU-only, shipped weights.
set -uo pipefail
OUT=/workspace/finetune/nemo_ft; ART="$OUT/artifacts"
LOG="$OUT/audionorm_test.log"; : > "$LOG"; exec > >(tee -a "$LOG") 2>&1
source /workspace/nemoenv/bin/activate
PROJ=/workspace/sapc-nemotron; LD="$PROJ/track2_starting_kit/local_decode.py"; DATA=/workspace/SAPC2
ORIG="$OUT/submission_encoder_sos_int8"; NORM="$OUT/submission_encoder_sos_int8_norm"
BOOT=/workspace/streaming_cer_bootstrap.py
SUBMAN=/dev/shm/devdiag_empty_subset.csv   # built by run_fallback_test.sh
STREAM_SMOKE=/dev/shm/Dev_streaming_1.csv
step(){ echo; echo "############### $* ###############"; }

test -f "$SUBMAN" || { echo "MISSING $SUBMAN (run fallback test first)"; exit 1; }
test -f "$STREAM_SMOKE" || python3 - "$DATA/manifest/Dev_streaming.csv" "$STREAM_SMOKE" <<'PY'
import csv,sys
r=list(csv.DictReader(open(sys.argv[1],newline="")))
import csv as c
w=c.DictWriter(open(sys.argv[2],"w",newline=""),fieldnames=r[0].keys());w.writeheader();w.writerow(r[0])
PY

step "Build normalization submission (copy + patch)"
rm -rf "$NORM"; cp -r "$ORIG" "$NORM"
python3 /workspace/patch_audionorm.py "$NORM/model.py"
python3 -m py_compile "$NORM/model.py" && echo "py_compile OK"

decode () {  # $1=tag $2=NORMMODE $3=out
  ( cd "$NORM" && SAPC2_THREADS=8 SAPC2_AUDIO_NORM="$2" python3 "$LD" --submission-dir "$NORM" \
      --manifest-csv "$SUBMAN" --streaming-manifest-csv "$STREAM_SMOKE" --streaming-interval 0 \
      --data-root "$DATA" --out-csv "$3" --out-partial-json "/dev/shm/${1}_p.json" ) 2>&1 | tail -3
}
score () { python3 "$BOOT" --manifest-csv "$SUBMAN" --hyp-csv "$1" --utils-dir "$PROJ/utils" --out-json "$2"; }

step "Decode 31 empties: NORM=none (control, expect 31 empty/100%)"; decode none none /dev/shm/an_none.csv; score /dev/shm/an_none.csv "$ART/an_none.json"
step "Decode 31 empties: NORM=peak"; decode peak peak /dev/shm/an_peak.csv; score /dev/shm/an_peak.csv "$ART/an_peak.json"
step "Decode 31 empties: NORM=rms";  decode rms  rms  /dev/shm/an_rms.csv;  score /dev/shm/an_rms.csv  "$ART/an_rms.json"

step "Sample RMS-norm hyps vs refs (first 10)"
python3 - "$SUBMAN" /dev/shm/an_rms.csv <<'PY'
import csv,sys
man={r["id"]:r for r in csv.DictReader(open(sys.argv[1],newline=""))}
hyp={r["id"]:r.get("raw_hypos","") for r in csv.DictReader(open(sys.argv[2],newline=""))}
for uid in list(hyp)[:10]:
    ref=man[uid].get("norm_text_without_disfluency") or man[uid].get("text","")
    print(f"[{man[uid].get('etiology','?')[:3]}] REF: {ref[:70]}")
    print(f"      HYP: {hyp[uid][:70]}")
PY
echo "AUDIONORM_TEST_DONE"
