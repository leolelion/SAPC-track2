#!/usr/bin/env bash
# Apples-to-apples Dev->Test disambiguation for the SHIPPED Nemotron encoder-SOS int8 model.
# Decodes the etiology-balanced 2k Dev ruler (dev_eval_2k.csv: 400 each ALS/CP/DS/PD/Stroke)
# through the REAL local_decode.py batch pass, then scores with the OFFICIAL evaluate.sh
# (same scorer family as Codabench Test1) + a bootstrap cross-check + per-etiology CER.
# No model changes, no training. Reuses the known-good faithful+official pipeline verbatim.
set -uo pipefail

OUT=/workspace/finetune/nemo_ft
ART="$OUT/artifacts"
LOG="$OUT/dev2k_int8_eval.log"; mkdir -p "$ART"; : > "$LOG"; exec > >(tee -a "$LOG") 2>&1
source /workspace/nemoenv/bin/activate
export PATH="/workspace/SCTK/bin:$PATH"

PROJ=/workspace/sapc-nemotron
LD="$PROJ/track2_starting_kit/local_decode.py"
DATA=/workspace/SAPC2
SUB="$OUT/submission_encoder_sos_int8"     # the shipped int8 build tree
BOOT=/workspace/streaming_cer_bootstrap.py
MAN=/workspace/finetune/eval/dev_eval_2k.csv
STREAM_SMOKE=/dev/shm/Dev_streaming_1.csv

HYP=/dev/shm/int8_dev2k.csv
PJSON=/dev/shm/int8_dev2k_partial.json

step(){ echo; echo "############### $* ###############"; }

step "Preflight"
for f in "$LD" "$DATA" "$SUB/model.py" "$SUB/weights/encoder_model.onnx" "$MAN" "$BOOT" \
         "$PROJ/steps/eval/prepare_ref_trn.sh" "$PROJ/steps/eval/evaluate.sh"; do
  test -e "$f" || { echo "MISSING $f"; exit 1; }
done
grep -q 'np.array(\[\[BLANK_ID\]\], dtype=np.int32)' "$SUB/model.py" || { echo "BAD_SOS"; exit 1; }
grep -q '^CHUNK_NEW = 16' "$SUB/model.py" || { echo "BAD_CHUNK"; exit 1; }
echo "manifest rows: $(tail -n +2 "$MAN" | wc -l)"

# streaming smoke (one row, interval 0 -> latency pass is a no-op; we only want accuracy)
python3 - "$DATA/manifest/Dev_streaming.csv" "$STREAM_SMOKE" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1], newline="")))
with open(sys.argv[2], "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerow(rows[0])
print("stream smoke:", sys.argv[2])
PY

step "Decode dev_eval_2k via real local_decode.py (int8)"
( cd "$SUB" && SAPC2_THREADS=4 python3 "$LD" --submission-dir "$SUB" \
    --manifest-csv "$MAN" --streaming-manifest-csv "$STREAM_SMOKE" \
    --streaming-interval 0 --data-root "$DATA" \
    --out-csv "$HYP" --out-partial-json "$PJSON" ) 2>&1 | tail -16
test -s "$HYP" || { echo "NO_HYP"; exit 1; }
echo "hyp rows: $(tail -n +2 "$HYP" | wc -l)"

step "OFFICIAL evaluate.sh (sclite SGML, min-over-two-refs) -- headline, Test1-comparable"
REF_DIR="$ART/official_refs_dev2k"; mkdir -p "$REF_DIR"
bash "$PROJ/steps/eval/prepare_ref_trn.sh" "$PROJ" \
  --split dev2k_int8 --csv "$MAN" \
  --ref1-col norm_text_with_disfluency --ref2-col norm_text_without_disfluency \
  --out-dir "$REF_DIR"
bash "$PROJ/steps/eval/evaluate.sh" "$PROJ" "$DATA" \
  --split dev2k_int8 --hyp-csv "$HYP" --hyp-col raw_hypos \
  --manifest-csv "$MAN" --ref-dir "$REF_DIR" \
  --out-json "$ART/dev2k_int8_official_metrics.json"
echo "OFFICIAL_JSON:"; cat "$ART/dev2k_int8_official_metrics.json" 2>/dev/null; echo

step "Bootstrap CER cross-check (overall)"
python3 "$BOOT" --manifest-csv "$MAN" --hyp-csv "$HYP" \
  --utils-dir "$PROJ/utils" --out-json "$ART/dev2k_int8_overall_bootstrap.json"
cat "$ART/dev2k_int8_overall_bootstrap.json" 2>/dev/null; echo

step "Per-etiology CER (bootstrap scorer on each etiology subset)"
for E in "ALS" "Cerebral Palsy" "Down Syndrome" "Parkinson's Disease" "Stroke"; do
  SAFE=$(echo "$E" | tr " '" "__")
  SUBMAN="/dev/shm/dev2k_${SAFE}.csv"
  python3 - "$MAN" "$SUBMAN" "$E" <<'PY'
import csv, sys
src, out, et = sys.argv[1], sys.argv[2], sys.argv[3]
rows = list(csv.DictReader(open(src, newline="")))
sel = [r for r in rows if r["etiology"] == et]
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(sel)
print(f"{et}: {len(sel)} rows -> {out}")
PY
  python3 "$BOOT" --manifest-csv "$SUBMAN" --hyp-csv "$HYP" \
    --utils-dir "$PROJ/utils" --out-json "$ART/dev2k_int8_${SAFE}_bootstrap.json" 2>/dev/null
  echo -n "  $E -> "; python3 - "$ART/dev2k_int8_${SAFE}_bootstrap.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
cer = d.get("cer", d.get("CER", d.get("cer_mean")))
print(f"CER={cer}")
PY
done

echo "DEV2K_INT8_EVAL_DONE"
