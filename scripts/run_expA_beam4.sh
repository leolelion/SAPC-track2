#!/usr/bin/env bash
# exp A (zipformer rescue) + beam-4 gate. Robust; writes RESULTS.txt. Run on pod via nohup.
set -uo pipefail
OUT=/workspace/finetune/eval/expA_beam4; mkdir -p "$OUT"
RES="$OUT/RESULTS.txt"; : > "$RES"; exec > >(tee -a "$RES") 2>&1
step(){ echo; echo "############### $* ###############"; }
MANDIR=/workspace/SAPC2/manifest
LD=/workspace/sapc-nemotron/track2_starting_kit/local_decode.py
NEMO=/workspace/finetune/nemo_submission
UT=/workspace/sapc-nemotron/utils
A1=/workspace/finetune/onnx/a1
DIAG="$MANDIR/Dev_diag.csv"; RAND="$MANDIR/Dev_rand300.csv"

step "0. sherpa_onnx + manifests"
python3 -c "import sherpa_onnx" 2>/dev/null || pip install -q sherpa-onnx 2>&1 | tail -2
python3 -c "import sherpa_onnx; print('sherpa', sherpa_onnx.__version__)" || echo SHERPA_FAIL
[ -f "$DIAG" ] || python3 /workspace/build_diag_manifest.py "$MANDIR/Dev.csv" "$DIAG"
REL=$(sed -n '2p' "$DIAG" | cut -d, -f4); DR=""
for c in /workspace/SAPC2 /workspace/finetune/data /workspace/finetune /workspace; do [ -e "$c/$REL" ] && { DR="$c"; break; }; done
echo "DATA_ROOT=$DR"

mklinks(){ ln -sf "$A1/encoder.int8.onnx" "$1/encoder.onnx"; ln -sf "$A1/decoder.int8.onnx" "$1/decoder.onnx"; ln -sf "$A1/joiner.int8.onnx" "$1/joiner.onnx"; }
decode(){ ( cd "$1" && python3 "$LD" --submission-dir "$1" --manifest-csv "$2" \
  --streaming-manifest-csv "$MANDIR/Dev_10.csv" --data-root "$DR" --out-csv "$3" --out-partial-json /dev/shm/_p.json ); }

step "1. NEMOTRON re-decode on diag (for head-to-head rescue)"
( cd "$NEMO" && SAPC2_THREADS=8 python3 "$LD" --submission-dir "$NEMO" --manifest-csv "$DIAG" \
  --streaming-manifest-csv "$MANDIR/Dev_10.csv" --data-root "$DR" --out-csv "$OUT/diag_nemo.csv" \
  --out-partial-json /dev/shm/_n.json ) && echo NEMO_OK || echo NEMO_FAIL

step "2. ZIPFORMER greedy on diag (exp A)"
ZG=/workspace/finetune/zf_a1_sub; mkdir -p "$ZG/wheels"; mklinks "$ZG"
[ -f "$ZG/model.py" ] && [ -f "$ZG/tokens.txt" ] || echo "WARN: stage greedy model.py+tokens.txt in $ZG"
decode "$ZG" "$DIAG" "$OUT/diag_zf_greedy.csv" && echo ZFG_OK || echo ZFG_FAIL

step "3. ZIPFORMER beam-4 on diag + rand300 (beam gate)"
ZB=/workspace/finetune/zf_a1_beam4_sub; mkdir -p "$ZB/wheels"; mklinks "$ZB"
cp /workspace/zf_beam4_model.py "$ZB/model.py" 2>/dev/null || echo "WARN: zf_beam4_model.py missing"
cp "$ZG/tokens.txt" "$ZB/tokens.txt" 2>/dev/null
decode "$ZB" "$DIAG" "$OUT/diag_zf_beam4.csv" && echo ZFB_OK || echo ZFB_FAIL
[ -f "$RAND" ] && decode "$ZB" "$RAND" "$OUT/rand300_zf_beam4.csv"

step "4. PER-MODEL CER + empties (official normalizer, two-ref) on diag"
cd "$UT"
for f in diag_nemo diag_zf_greedy diag_zf_beam4; do
  [ -f "$OUT/$f.csv" ] && { echo "--- $f ---"; python3 /workspace/nemotron_repro_analyze.py "$DIAG" "$OUT/$f.csv" 2>&1 | tail -3; }
done

step "5. RESCUE CROSS-TAB: does finetuned ZF fix Nemo's failures? (+ length x severity)"
[ -f "$OUT/diag_zf_greedy.csv" ] && python3 /workspace/diag_crosstab.py "$DIAG" \
  nemo="$OUT/diag_nemo.csv" zf="$OUT/diag_zf_greedy.csv" || echo CROSSTAB_SKIP

step "6. BEAM-4 GATE: greedy vs beam-4 on rand300 (representative)"
[ -f "$RAND" ] && { echo "greedy:"; decode "$ZG" "$RAND" "$OUT/rand300_zf_greedy.csv" >/dev/null 2>&1; \
  python3 /workspace/nemotron_repro_analyze.py "$RAND" "$OUT/rand300_zf_greedy.csv" 2>&1 | tail -2; \
  echo "beam-4:"; python3 /workspace/nemotron_repro_analyze.py "$RAND" "$OUT/rand300_zf_beam4.csv" 2>&1 | tail -2; }

step "BATTERY_DONE"
