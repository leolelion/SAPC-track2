#!/usr/bin/env bash
# Build and gate an encoder-dynamic-int8 variant of the encoder-only SOS package.
set -euo pipefail

OUT=/workspace/finetune/nemo_ft
ART="$OUT/artifacts"
LOG="$OUT/encoder_sos_int8.log"; mkdir -p "$ART"; : > "$LOG"; exec > >(tee -a "$LOG") 2>&1
source /workspace/nemoenv/bin/activate

FP32_SUB="$OUT/submission_encoder_sos"
FP32_ZIP="$ART/nemotron_encoder_sos_submission.zip"
INT8_SUB="$OUT/submission_encoder_sos_int8"
INT8_ZIP="$ART/nemotron_encoder_sos_int8_submission.zip"
INT8_SHA="$INT8_ZIP.sha256"
INT8_EXTRACT=/dev/shm/nemotron_encoder_sos_int8_extract

LD=/workspace/sapc-nemotron/track2_starting_kit/local_decode.py
PROJ=/workspace/sapc-nemotron
DATA=/workspace/SAPC2
MAN_DIR="$DATA/manifest"
BOOT=/workspace/streaming_cer_bootstrap.py
PAIR=/workspace/compare_cer_pairs.py
QUANT=/workspace/quantize_nemotron_encoder.py
BENCH=/workspace/bench_nemotron_multiproc.py
SGML_DEBUG=/workspace/debug_sgml_pred_mismatch.py

SMOKE_N=${SMOKE_N:-20}
REP_N=${REP_N:-500}
REP_MAN="$OUT/Dev_rep${REP_N}_seed23.csv"
SMOKE_MAN="/dev/shm/Dev_int8_smoke_${SMOKE_N}.csv"
STREAM_SMOKE=/dev/shm/Dev_streaming_1.csv
RUN_THREAD_BENCH=${RUN_THREAD_BENCH:-1}
RUN_FULL_LATENCY=${RUN_FULL_LATENCY:-1}
RUN_OFFICIAL_EVAL=${RUN_OFFICIAL_EVAL:-1}
FORCE_REBUILD_INT8=${FORCE_REBUILD_INT8:-1}
OFFICIAL_FAILURES=()

step(){ echo; echo "############### $* ###############"; }
require_file(){ test -f "$1" || { echo "MISSING_FILE $1"; exit 1; }; }
require_dir(){ test -d "$1" || { echo "MISSING_DIR $1"; exit 1; }; }

ensure_fp32_source () {
  if [ -f "$FP32_SUB/model.py" ] && [ -f "$FP32_SUB/weights/encoder_model.onnx" ]; then
    echo "using existing FP32 SOS submission dir: $FP32_SUB"
  elif [ -f /workspace/run_package_encoder_sos.sh ]; then
    echo "FP32 SOS submission dir missing; rebuilding via /workspace/run_package_encoder_sos.sh"
    FORCE_REBUILD=0 bash /workspace/run_package_encoder_sos.sh
  else
    echo "FP32 SOS submission dir missing and no /workspace/run_package_encoder_sos.sh available"
    exit 1
  fi
  require_file "$FP32_SUB/model.py"
  require_file "$FP32_SUB/weights/encoder_model.onnx"
  require_file "$FP32_SUB/weights/decoder_model.onnx"
  grep -q 'np.array(\[\[BLANK_ID\]\], dtype=np.int32)' "$FP32_SUB/model.py"
  grep -q '^CHUNK_NEW = 16' "$FP32_SUB/model.py"
}

make_manifests () {
  python3 - "$MAN_DIR/Dev.csv" "$SMOKE_MAN" "$SMOKE_N" <<'PY'
import csv, sys
src, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = list(csv.DictReader(open(src, newline="")))[:n]
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
print(f"smoke manifest: {out} rows={len(rows)}")
PY
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

build_int8_tree () {
  require_file "$QUANT"
  if [ "$FORCE_REBUILD_INT8" = "1" ] || [ ! -f "$INT8_SUB/weights/encoder_model.onnx" ]; then
    python3 "$QUANT" \
      --source-submission "$FP32_SUB" \
      --output-submission "$INT8_SUB" \
      --work-dir "$ART/int8_quant_work" \
      --out-json "$ART/enc_sos_int8_preflight.json"
  else
    echo "reusing existing INT8 submission dir: $INT8_SUB"
  fi
  require_file "$INT8_SUB/model.py"
  require_file "$INT8_SUB/weights/encoder_model.onnx"
  require_file "$INT8_SUB/weights/decoder_model.onnx"
  grep -q 'np.array(\[\[BLANK_ID\]\], dtype=np.int32)' "$INT8_SUB/model.py"
  grep -q '^CHUNK_NEW = 16' "$INT8_SUB/model.py"
  if grep -q 'np.array(\[\[0\]\], dtype=np.int32)' "$INT8_SUB/model.py"; then
    echo "FOUND_OLD_SOS_ZERO"
    exit 1
  fi
}

build_zip () {
  step "Build int8 zip"
  rm -f "$INT8_ZIP" "$INT8_SHA" "$INT8_ZIP.tmp"
  python3 - "$INT8_SUB" "$INT8_ZIP" <<'PY'
import sys, zipfile
from pathlib import Path
src = Path(sys.argv[1])
out = Path(sys.argv[2])
tmp = out.with_suffix(out.suffix + ".tmp")
if tmp.exists():
    tmp.unlink()
with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for path in sorted(src.rglob("*")):
        if not path.is_file():
            continue
        arc = path.relative_to(src).as_posix()
        zf.write(path, arc)
        if arc == "setup.sh":
            zf.getinfo(arc).external_attr = (0o755 & 0xFFFF) << 16
tmp.replace(out)
print(f"wrote {out}")
PY
  sha256sum "$INT8_ZIP" | tee "$INT8_SHA"
  ls -lh "$INT8_ZIP" "$INT8_SHA"
  python3 - "$INT8_ZIP" <<'PY'
import sys, zipfile
zf = zipfile.ZipFile(sys.argv[1])
names = set(zf.namelist())
required = {
    "model.py", "localmel.py", "config.yaml", "setup.sh",
    "weights/encoder_model.onnx", "weights/decoder_model.onnx", "weights/tokens.txt",
}
missing = sorted(required - names)
bad_nested = [n for n in names if n.startswith("submission_encoder_sos_int8/")]
weight_files = [n for n in names if n.startswith("weights/") and not n.endswith("/")]
print(f"zip entries={len(names)} weight_files={len(weight_files)} missing={missing} nested_root={bool(bad_nested)}")
assert not missing
assert not bad_nested
PY
}

extract_int8_zip () {
  step "Extract int8 package"
  rm -rf "$INT8_EXTRACT"; mkdir -p "$INT8_EXTRACT"
  python3 - "$INT8_ZIP" "$INT8_EXTRACT" <<'PY'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as zf:
    zf.extractall(sys.argv[2])
print(f"extracted {sys.argv[1]} -> {sys.argv[2]}")
PY
  require_file "$INT8_EXTRACT/model.py"
  require_file "$INT8_EXTRACT/weights/encoder_model.onnx"
  require_file "$INT8_EXTRACT/weights/decoder_model.onnx"
  grep -q 'np.array(\[\[BLANK_ID\]\], dtype=np.int32)' "$INT8_EXTRACT/model.py"
}

decode_manifest () {  # $1=tag $2=submission-dir $3=manifest $4=out-csv $5=out-json $6=stream-interval
  tag="$1"; sub="$2"; manifest="$3"; csv_out="$4"; json_out="$5"; interval="$6"
  step "Decode $tag: $manifest"
  ( cd "$sub" && SAPC2_THREADS=4 python3 "$LD" --submission-dir "$sub" \
      --manifest-csv "$manifest" --streaming-manifest-csv "$STREAM_SMOKE" \
      --streaming-interval "$interval" --data-root "$DATA" \
      --out-csv "$csv_out" --out-partial-json "$json_out" ) 2>&1 | tail -16
}

ensure_sclite () {
  if command -v sclite >/dev/null 2>&1; then
    echo "sclite: $(command -v sclite)"
    return
  fi
  if [ -x /workspace/SCTK/bin/sclite ]; then
    export PATH="/workspace/SCTK/bin:$PATH"
    echo "sclite: $(command -v sclite)"
    return
  fi
  step "Install SCTK for official evaluate.sh"
  rm -rf /workspace/SCTK
  git clone https://github.com/usnistgov/SCTK.git /workspace/SCTK
  ( cd /workspace/SCTK && make config && make all && make install )
  export PATH="/workspace/SCTK/bin:$PATH"
  sclite -V || true
}

official_score () {  # $1=split $2=manifest $3=hyp-csv $4=out-json
  split="$1"; manifest="$2"; hyp_csv="$3"; out_json="$4"
  [ "$RUN_OFFICIAL_EVAL" = "1" ] || { echo "official score skipped by RUN_OFFICIAL_EVAL=0"; return; }
  ensure_sclite
  ref_dir="$ART/official_refs"
  mkdir -p "$ref_dir"
  bash "$PROJ/steps/eval/prepare_ref_trn.sh" "$PROJ" \
    --split "$split" --csv "$manifest" \
    --ref1-col norm_text_with_disfluency --ref2-col norm_text_without_disfluency \
    --out-dir "$ref_dir"

  set +e
  bash "$PROJ/steps/eval/evaluate.sh" "$PROJ" "$DATA" \
    --split "$split" --hyp-csv "$hyp_csv" --hyp-col raw_hypos \
    --manifest-csv "$manifest" --ref-dir "$ref_dir" --out-json "$out_json"
  rc=$?
  set -e
  if [ "$rc" -eq 0 ]; then
    return 0
  fi

  echo "OFFICIAL_EVAL_FAILED split=$split rc=$rc"
  OFFICIAL_FAILURES+=("$split:$rc")

  dbg="$ART/official_debug_${split}"
  rm -rf "$dbg"; mkdir -p "$dbg"
  cp -f "$hyp_csv" "$dbg/hyp.csv" 2>/dev/null || true
  cp -f "$manifest" "$dbg/manifest.csv" 2>/dev/null || true
  cp -f "$ref_dir/ref1.${split}.norm.trn" "$dbg/ref1.norm.trn" 2>/dev/null || true
  cp -f "$ref_dir/ref2.${split}.norm.trn" "$dbg/ref2.norm.trn" 2>/dev/null || true
  cp -f "$DATA/eval/hyp.${split}.norm.trn" "$dbg/hyp.norm.trn" 2>/dev/null || true
  cp -f "$DATA/eval/hyp.${split}.ref1.norm.trn" "$dbg/hyp.ref1.norm.trn" 2>/dev/null || true
  cp -f "$DATA/eval/hyp.${split}.ref2.norm.trn" "$dbg/hyp.ref2.norm.trn" 2>/dev/null || true
  cp -f "$DATA/eval/sctk/${split}.ref1.sgml" "$dbg/ref1.sgml" 2>/dev/null || true
  cp -f "$DATA/eval/sctk/${split}.ref2.sgml" "$dbg/ref2.sgml" 2>/dev/null || true

  if [ -f "$SGML_DEBUG" ] && [ -f "$dbg/ref1.sgml" ] && [ -f "$dbg/ref2.sgml" ]; then
    set +e
    python3 "$SGML_DEBUG" \
      --sgml-ref1 "$dbg/ref1.sgml" \
      --sgml-ref2 "$dbg/ref2.sgml" \
      --hyp-trn "$dbg/hyp.norm.trn" \
      --ref1-trn "$dbg/ref1.norm.trn" \
      --ref2-trn "$dbg/ref2.norm.trn" \
      --manifest-csv "$dbg/manifest.csv" \
      --out-json "$dbg/sgml_pred_mismatch.json" \
      > "$dbg/sgml_pred_mismatch.txt" 2>&1
    dbg_rc=$?
    set -e
    echo "SGML_DEBUG_DONE split=$split rc=$dbg_rc"
    cat "$dbg/sgml_pred_mismatch.txt"
  else
    echo "SGML_DEBUG_SKIPPED split=$split script_or_sgml_missing"
  fi
  tar -C "$dbg" -czf "$ART/official_debug_${split}.tgz" .
  echo "OFFICIAL_DEBUG_TGZ $ART/official_debug_${split}.tgz"
  return 0
}

eval_gate () {  # $1=tag $2=manifest
  tag="$1"; manifest="$2"
  fp32_csv="/dev/shm/fp32_${tag}.csv"
  fp32_json="/dev/shm/fp32_${tag}.json"
  int8_csv="/dev/shm/int8_${tag}.csv"
  int8_json="/dev/shm/int8_${tag}.json"
  decode_manifest "fp32_$tag" "$FP32_SUB" "$manifest" "$fp32_csv" "$fp32_json" 0
  decode_manifest "int8_$tag" "$INT8_EXTRACT" "$manifest" "$int8_csv" "$int8_json" 0

  step "Custom CER for int8 $tag"
  python3 "$BOOT" --manifest-csv "$manifest" --hyp-csv "$int8_csv" \
    --utils-dir "$PROJ/utils" --out-json "$ART/enc_sos_int8_${tag}_cer_bootstrap.json"

  step "Paired CER delta fp32 -> int8 $tag"
  python3 "$PAIR" --manifest-csv "$manifest" --base-hyp-csv "$fp32_csv" \
    --candidate-hyp-csv "$int8_csv" --utils-dir "$PROJ/utils" \
    --out-json "$ART/enc_sos_int8_${tag}_paired_delta.json"

  step "Official evaluate.sh gate for int8 $tag"
  official_score "enc_sos_int8_${tag}" "$manifest" "$int8_csv" \
    "$ART/enc_sos_int8_${tag}_official_metrics.json"
}

thread_bench () {
  [ "$RUN_THREAD_BENCH" = "1" ] || { echo "thread bench skipped by RUN_THREAD_BENCH=0"; return; }
  require_file "$BENCH"
  for threads in 1 4; do
    step "20-worker batch throughput int8 threads=$threads"
    out_jsonl="$ART/enc_sos_int8_w20_t${threads}_120.jsonl"
    out_log="$ART/enc_sos_int8_w20_t${threads}_120.log"
    if command -v taskset >/dev/null 2>&1; then
      taskset -c 0-19 python3 "$BENCH" --submission-dir "$INT8_EXTRACT" \
        --manifest-csv "$REP_MAN" --data-root "$DATA" --out-jsonl "$out_jsonl" \
        --workers 20 --threads "$threads" --max-rows 120 2>&1 | tee "$out_log"
    else
      python3 "$BENCH" --submission-dir "$INT8_EXTRACT" \
        --manifest-csv "$REP_MAN" --data-root "$DATA" --out-jsonl "$out_jsonl" \
        --workers 20 --threads "$threads" --max-rows 120 2>&1 | tee "$out_log"
    fi
  done
}

full_latency () {
  [ "$RUN_FULL_LATENCY" = "1" ] || { echo "full latency skipped by RUN_FULL_LATENCY=0"; return; }
  step "Full Dev_streaming latency for int8 package"
  lat_csv=/dev/shm/int8_latency_smoke_batch.csv
  lat_json=/dev/shm/int8_devstream_latency.json
  ( cd "$INT8_EXTRACT" && SAPC2_THREADS=4 python3 "$LD" --submission-dir "$INT8_EXTRACT" \
      --manifest-csv "$SMOKE_MAN" --streaming-manifest-csv "$MAN_DIR/Dev_streaming.csv" \
      --data-root "$DATA" --out-csv "$lat_csv" --out-partial-json "$lat_json" ) 2>&1 | tail -20
  bash "$PROJ/steps/eval/evaluate_latency.sh" \
    --partial-json "$lat_json" --manifest-csv "$MAN_DIR/Dev_streaming.csv" \
    --out-json "$ART/enc_sos_int8_devstream_latency.json"
}

step "Preflight source"
ensure_fp32_source
make_manifests

step "Quantize encoder"
build_int8_tree
build_zip
extract_int8_zip

eval_gate smoke "$SMOKE_MAN"
eval_gate devrep${REP_N} "$REP_MAN"
eval_gate devdiag "$MAN_DIR/Dev_diag.csv"
thread_bench
full_latency

if [ "${#OFFICIAL_FAILURES[@]}" -gt 0 ]; then
  echo "OFFICIAL_EVAL_FAILURES ${OFFICIAL_FAILURES[*]}"
  exit 2
fi

echo "ENCODER_SOS_INT8_DONE zip=$INT8_ZIP sha=$INT8_SHA"
