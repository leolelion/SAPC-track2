#!/usr/bin/env bash
# Evaluate the built encoder-only + SOS package zip itself on Dev_diag and representative Dev-500.
set -euo pipefail

OUT=/workspace/finetune/nemo_ft
ART="$OUT/artifacts"
LOG="$OUT/package_encoder_sos_eval.log"; : > "$LOG"; exec > >(tee -a "$LOG") 2>&1
source /workspace/nemoenv/bin/activate

ZIP="$ART/nemotron_encoder_sos_submission.zip"
EXTRACT=/dev/shm/nemotron_encoder_sos_eval_extract
LD=/workspace/sapc-nemotron/track2_starting_kit/local_decode.py
BOOT=/workspace/streaming_cer_bootstrap.py
DATA=/workspace/SAPC2
MAN_DIR="$DATA/manifest"
STREAM_SMOKE=/dev/shm/Dev_streaming_1.csv
REP_N=${REP_N:-500}
REP_MAN="$OUT/Dev_rep${REP_N}_seed23.csv"

step(){ echo; echo "############### $* ###############"; }

extract_package () {
  test -f "$ZIP"
  rm -rf "$EXTRACT"; mkdir -p "$EXTRACT"
  python3 - "$ZIP" "$EXTRACT" <<'PY'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as zf:
    zf.extractall(sys.argv[2])
print(f"extracted {sys.argv[1]} -> {sys.argv[2]}")
PY
  test -f "$EXTRACT/model.py"
  test -f "$EXTRACT/setup.sh"
  test -f "$EXTRACT/weights/encoder_model.onnx"
  test -f "$EXTRACT/weights/decoder_model.onnx"
  grep -q 'np.array(\[\[BLANK_ID\]\], dtype=np.int32)' "$EXTRACT/model.py"
}

make_manifests () {
  python3 - "$MAN_DIR/Dev_streaming.csv" "$STREAM_SMOKE" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1], newline="")))
with open(sys.argv[2], "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerow(rows[0])
print(f"stream smoke manifest: {sys.argv[2]}")
PY
  if [ ! -f "$REP_MAN" ]; then
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
  else
    echo "reusing representative manifest: $REP_MAN"
  fi
}

eval_manifest () {  # $1=tag $2=manifest
  tag="$1"; manifest="$2"; csv_out="/dev/shm/pkg_enc_sos_${tag}.csv"; json_out="/dev/shm/pkg_enc_sos_${tag}.json"
  step "Decode extracted package $tag: $manifest"
  ( cd "$EXTRACT" && SAPC2_THREADS=4 python3 "$LD" --submission-dir "$EXTRACT" \
      --manifest-csv "$manifest" --streaming-manifest-csv "$STREAM_SMOKE" \
      --streaming-interval 0 --data-root "$DATA" \
      --out-csv "$csv_out" --out-partial-json "$json_out" ) 2>&1 | tail -12
  step "CER + speaker bootstrap for package $tag"
  python3 "$BOOT" --manifest-csv "$manifest" --hyp-csv "$csv_out" \
    --utils-dir /workspace/sapc-nemotron/utils \
    --out-json "$ART/pkg_enc_sos_${tag}_cer_bootstrap.json"
}

step "Extract package"
extract_package
make_manifests

eval_manifest devdiag "$MAN_DIR/Dev_diag.csv"
eval_manifest "devrep${REP_N}" "$REP_MAN"

echo "PACKAGE_ENCODER_SOS_EVAL_DONE"
