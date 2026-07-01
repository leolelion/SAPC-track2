#!/usr/bin/env bash
# Pod-side diagnostic for the official evaluate.sh failure:
#   "ERROR: preds from sgml-ref1 and sgml-ref2 are not identical!"
#
# Goals (cheap, reuses existing artifacts, DOES NOT stop the pod):
#   1. Run the organizers' official evaluate.sh on the representative Dev-500 hyp
#      for BOTH the FP32 package and the int8 package (the missing control:
#      we never confirmed whether FP32 also fails).
#   2. Preserve the generated SGML / ref / hyp .trn for each.
#   3. Run scripts/debug_sgml_pred_mismatch.py to print the exact utterances
#      where the ref1 vs ref2 reconstructed hypotheses diverge, and whether the
#      divergence is unk-only.
#
# This NEVER edits scorer semantics and NEVER stops the pod.
#
# Upload first (from Mac):
#   scp -P <PORT> -i <KEY> scripts/diagnose_official_eval.sh \
#       scripts/debug_sgml_pred_mismatch.py root@<HOST>:/workspace/
# Then on the pod:
#   cd /workspace && bash diagnose_official_eval.sh
set -uo pipefail

PROJ=${PROJ:-/workspace/sapc-nemotron}
DATA=${DATA:-/workspace/SAPC2}
MAN_DIR="$DATA/manifest"
OUT=${OUT:-/workspace/finetune/nemo_ft}
ART="$OUT/artifacts"
DBG_ROOT="$ART/official_eval_diag"
LD="$PROJ/track2_starting_kit/local_decode.py"
SGML_DEBUG=${SGML_DEBUG:-/workspace/debug_sgml_pred_mismatch.py}
REP_MAN=${REP_MAN:-$OUT/Dev_rep500_seed23.csv}
STREAM_SMOKE=/dev/shm/Dev_streaming_1.csv

FP32_ZIP="$ART/nemotron_encoder_sos_submission.zip"
INT8_ZIP="$ART/nemotron_encoder_sos_int8_submission.zip"
FP32_EXTRACT=/dev/shm/diag_fp32_extract
INT8_EXTRACT=/dev/shm/diag_int8_extract
FP32_CSV=${FP32_CSV:-/dev/shm/fp32_devrep500.csv}
INT8_CSV=${INT8_CSV:-/dev/shm/int8_devrep500.csv}

mkdir -p "$ART" "$DBG_ROOT"
LOG="$OUT/official_eval_diag.log"; : > "$LOG"; exec > >(tee -a "$LOG") 2>&1

if [ -f /workspace/nemoenv/bin/activate ]; then
  # shellcheck disable=SC1091
  source /workspace/nemoenv/bin/activate
fi

step(){ echo; echo "############### $* ###############"; }

ensure_sclite () {
  command -v sclite >/dev/null 2>&1 && { echo "sclite: $(command -v sclite)"; return; }
  [ -x /workspace/SCTK/bin/sclite ] && { export PATH="/workspace/SCTK/bin:$PATH"; echo "sclite: $(command -v sclite)"; return; }
  echo "ERROR: sclite not found; run the full int8 runbook first (it installs SCTK)."
  exit 1
}

ensure_stream_smoke () {
  [ -f "$STREAM_SMOKE" ] && return
  python3 - "$MAN_DIR/Dev_streaming.csv" "$STREAM_SMOKE" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1], newline="")))
with open(sys.argv[2], "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerow(rows[0])
print("stream smoke:", sys.argv[2])
PY
}

extract_zip () {  # $1=zip $2=dir
  local zip="$1" dir="$2"
  [ -f "$dir/model.py" ] && { echo "reuse extract: $dir"; return; }
  test -f "$zip" || { echo "MISSING_ZIP $zip"; return 1; }
  rm -rf "$dir"; mkdir -p "$dir"
  python3 - "$zip" "$dir" <<'PY'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as zf:
    zf.extractall(sys.argv[2])
print("extracted", sys.argv[1], "->", sys.argv[2])
PY
}

decode_if_missing () {  # $1=hyp_csv $2=zip $3=extract_dir
  local hyp_csv="$1" zip="$2" dir="$3"
  if [ -s "$hyp_csv" ]; then
    echo "reuse hyp csv: $hyp_csv ($(wc -l < "$hyp_csv") lines)"; return
  fi
  echo "hyp csv missing ($hyp_csv); decoding from package"
  extract_zip "$zip" "$dir" || return 1
  ensure_stream_smoke
  ( cd "$dir" && SAPC2_THREADS=4 python3 "$LD" --submission-dir "$dir" \
      --manifest-csv "$REP_MAN" --streaming-manifest-csv "$STREAM_SMOKE" \
      --streaming-interval 0 --data-root "$DATA" \
      --out-csv "$hyp_csv" --out-partial-json "${hyp_csv%.csv}.json" ) 2>&1 | tail -8
}

official_and_debug () {  # $1=tag $2=hyp_csv
  local tag="$1" hyp_csv="$2"
  local split="diag_${tag}"
  local ref_dir="$ART/official_refs"
  local dbg="$DBG_ROOT/$tag"
  mkdir -p "$ref_dir" "$dbg"

  step "[$tag] prepare refs (Dev-500)"
  bash "$PROJ/steps/eval/prepare_ref_trn.sh" "$PROJ" \
    --split "$split" --csv "$REP_MAN" \
    --ref1-col norm_text_with_disfluency --ref2-col norm_text_without_disfluency \
    --out-dir "$ref_dir"

  step "[$tag] official evaluate.sh (Dev-500)"
  bash "$PROJ/steps/eval/evaluate.sh" "$PROJ" "$DATA" \
    --split "$split" --hyp-csv "$hyp_csv" --hyp-col raw_hypos \
    --manifest-csv "$REP_MAN" --ref-dir "$ref_dir" \
    --out-json "$dbg/official_metrics.json"
  local rc=$?
  echo "[$tag] OFFICIAL_RC=$rc"

  # Preserve scorer artifacts regardless of pass/fail.
  cp -f "$DATA/eval/sctk/${split}.ref1.sgml" "$dbg/ref1.sgml" 2>/dev/null || true
  cp -f "$DATA/eval/sctk/${split}.ref2.sgml" "$dbg/ref2.sgml" 2>/dev/null || true
  cp -f "$DATA/eval/hyp.${split}.norm.trn" "$dbg/hyp.norm.trn" 2>/dev/null || true
  cp -f "$ref_dir/ref1.${split}.norm.trn" "$dbg/ref1.norm.trn" 2>/dev/null || true
  cp -f "$ref_dir/ref2.${split}.norm.trn" "$dbg/ref2.norm.trn" 2>/dev/null || true
  cp -f "$hyp_csv" "$dbg/hyp.csv" 2>/dev/null || true

  if [ -f "$SGML_DEBUG" ] && [ -f "$dbg/ref1.sgml" ] && [ -f "$dbg/ref2.sgml" ]; then
    step "[$tag] SGML pred-mismatch diagnosis"
    python3 "$SGML_DEBUG" \
      --sgml-ref1 "$dbg/ref1.sgml" --sgml-ref2 "$dbg/ref2.sgml" \
      --hyp-trn "$dbg/hyp.norm.trn" \
      --ref1-trn "$dbg/ref1.norm.trn" --ref2-trn "$dbg/ref2.norm.trn" \
      --manifest-csv "$REP_MAN" --out-json "$dbg/sgml_pred_mismatch.json" \
      --max-rows 40 | tee "$dbg/sgml_pred_mismatch.txt"
    echo "[$tag] SGML_DEBUG_EXIT=${PIPESTATUS[0]}"
  else
    echo "[$tag] SGML_DEBUG_SKIPPED (script or sgml missing)"
  fi

  echo "$tag rc=$rc" >> "$DBG_ROOT/summary.txt"
}

: > "$DBG_ROOT/summary.txt"
ensure_sclite
test -f "$REP_MAN" || { echo "MISSING_REP_MAN $REP_MAN"; exit 1; }

decode_if_missing "$FP32_CSV" "$FP32_ZIP" "$FP32_EXTRACT"
decode_if_missing "$INT8_CSV" "$INT8_ZIP" "$INT8_EXTRACT"

official_and_debug fp32 "$FP32_CSV"
official_and_debug int8 "$INT8_CSV"

step "SUMMARY"
cat "$DBG_ROOT/summary.txt"
echo
echo "If fp32 rc != 0 too -> the assert is hyp/scorer-path, NOT int8-specific."
echo "Artifacts: $DBG_ROOT/{fp32,int8}/ (ref1.sgml, ref2.sgml, sgml_pred_mismatch.{txt,json})"
tar -C "$DBG_ROOT" -czf "$ART/official_eval_diag.tgz" . 2>/dev/null || true
echo "TGZ: $ART/official_eval_diag.tgz"
echo "NOTE: pod intentionally left RUNNING."
