#!/usr/bin/env bash
# Severe + representative CER gate for the chosen encoder-only Nemotron model with SOS=BLANK_ID.
# This measures CER via the real local_decode.py batch pass. The streaming pass is set to a
# one-row smoke manifest with interval 0 because latency was already measured in run_sos_fix.sh.
set -uo pipefail

OUT=/workspace/finetune/nemo_ft
LOG="$OUT/encoder_sos_eval.log"; : > "$LOG"; exec > >(tee -a "$LOG") 2>&1
source /workspace/nemoenv/bin/activate

LD=/workspace/sapc-nemotron/track2_starting_kit/local_decode.py
DATA=/workspace/SAPC2
MAN_DIR="$DATA/manifest"
BOOT=/workspace/streaming_cer_bootstrap.py
STREAM_SMOKE=/dev/shm/Dev_streaming_1.csv
REP_N=${REP_N:-500}
REP_MAN="$OUT/Dev_rep${REP_N}_seed23.csv"

step(){ echo; echo "############### $* ###############"; }

make_stream_smoke () {
  python3 - "$MAN_DIR/Dev_streaming.csv" "$STREAM_SMOKE" <<'PY'
import csv, sys
src, out = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(src, newline="")))
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerow(rows[0])
print(f"stream smoke manifest: {out}")
PY
}

make_rep_manifest () {
  python3 - "$MAN_DIR/Dev.csv" "$REP_MAN" "$REP_N" <<'PY'
import csv, random, sys
src, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
rng = random.Random(23)
rows = list(csv.DictReader(open(src, newline="")))
rng.shuffle(rows)
sel = rows[:min(n, len(rows))]
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(sel)
print(f"representative manifest: {out} rows={len(sel)}")
PY
}

build_encoder_sos () {
  SUB="$OUT/deploy_encoder_sos"
  rm -rf "$SUB"; cp -r /workspace/finetune/nemo_submission "$SUB"
  rm -f "$SUB"/weights/encoder_model.onnx* "$SUB"/weights/decoder_model.onnx* 2>/dev/null
  cp "$OUT/export_full70_1"/* "$SUB"/weights/ 2>/dev/null
  mv "$SUB"/weights/encoder-model.onnx "$SUB"/weights/encoder_model.onnx
  mv "$SUB"/weights/decoder_joint-model.onnx "$SUB"/weights/decoder_model.onnx
  sed -i 's/^CHUNK_NEW = 56/CHUNK_NEW = 16/' "$SUB"/model.py
  before=$(grep -c 'np.array(\[\[0\]\], dtype=np.int32)' "$SUB"/model.py)
  sed -i 's/np.array(\[\[0\]\], dtype=np.int32)/np.array([[BLANK_ID]], dtype=np.int32)/g' "$SUB"/model.py
  after=$(grep -c 'np.array(\[\[BLANK_ID\]\], dtype=np.int32)' "$SUB"/model.py)
  echo "SOS-fix deploy dir: $SUB (replaced $before, blank occurrences now $after)" >&2
  echo "$SUB"
}

eval_manifest () {  # $1=tag $2=manifest
  tag="$1"; manifest="$2"; csv_out="/dev/shm/enc_sos_${tag}.csv"; json_out="/dev/shm/enc_sos_${tag}.json"
  step "Decode $tag: $manifest"
  ( cd "$SUB" && SAPC2_THREADS=4 python3 "$LD" --submission-dir "$SUB" \
      --manifest-csv "$manifest" --streaming-manifest-csv "$STREAM_SMOKE" \
      --streaming-interval 0 --data-root "$DATA" \
      --out-csv "$csv_out" --out-partial-json "$json_out" ) 2>&1 | tail -8
  step "CER + speaker bootstrap for $tag"
  python3 "$BOOT" --manifest-csv "$manifest" --hyp-csv "$csv_out" \
    --utils-dir /workspace/sapc-nemotron/utils \
    --out-json "$OUT/enc_sos_${tag}_cer_bootstrap.json"
}

step "Prepare manifests"
test -f "$BOOT" || { echo "missing $BOOT"; exit 2; }
make_stream_smoke
make_rep_manifest

step "Build encoder-only SOS-fixed submission dir"
SUB=$(build_encoder_sos)

eval_manifest devdiag "$MAN_DIR/Dev_diag.csv"
eval_manifest "devrep${REP_N}" "$REP_MAN"

echo "ENCODER_SOS_EVAL_DONE"
