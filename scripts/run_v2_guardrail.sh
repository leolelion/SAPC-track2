#!/usr/bin/env bash
# v2 STRUCTURAL guardrail: Gate 0 (wiring) + Gate 1 (guardrail train -> export -> faithful eval -> verdict).
# Conservative arm (research/38): joint_unfreeze, joint_lr x0.1, gain+speed aug, severity cap 2x.
# HARD-CAPPED, gate-protected. STOPS at the pre-registered criterion before any full run.
#
# PREREQUISITE (verify on pod BEFORE Gate 1): locate the .nemo->ONNX cache-aware export that produced
#   $OUT/export_full70_1 (encoder-model.onnx + decoder_joint-model.onnx). v2 retrains the JOINT, so the
#   decoder_joint ONNX MUST be re-exported from the v2 .nemo. Set EXPORT_CMD below to that script.
#   Until confirmed, Gate 1 packaging/eval cannot run (Gate 0 has no such dependency).
set -uo pipefail

OUT=/workspace/finetune/nemo_ft
M="$OUT/manifests_v2"
ART="$OUT/artifacts"
LOG="$OUT/v2_guardrail.log"; mkdir -p "$ART"; : > "$LOG"; exec > >(tee -a "$LOG") 2>&1
source /workspace/nemoenv/bin/activate

DATA=/workspace/SAPC2
TRAINCSV="$DATA/manifest/Train.csv"
FT=/workspace/nemo_finetune_v2.py
PREP=/workspace/prep_nemo_manifest_v2.py
EPOCHS_GUARD=${EPOCHS_GUARD:-4}
# Set this once the pod export script is located (see PREREQUISITE):
EXPORT_CMD=${EXPORT_CMD:-""}   # e.g. "bash /workspace/export_nemo_to_onnx.sh <v2.nemo> $OUT/export_v2"

step(){ echo; echo "############### $* ###############"; }

step "Prep v2 manifests (severity cap 2x, val3k, pd_probe, smoke4k, guardrail subset)"
test -f "$M/guardrail_train.json" || python3 "$PREP" --train-csv "$TRAINCSV" --data-root "$DATA" --out-dir "$M" --target text
ls -la "$M" | cat

step "GATE 0 — wiring smoke (4k, 2 epochs, joint_unfreeze)"
python3 "$FT" --mode smoke --epochs 2 --freeze joint_unfreeze \
  --train-json "$M/smoke4k.json" --val-json "$M/val3k.json" \
  --out-dir "$OUT/v2_wiring" --train-ctx "[70,1]" 2>&1 | tail -40
G0LOG="$OUT/v2_guardrail.log"
echo "--- Gate 0 assertions ---"
grep -q "DIFFERENTIAL adamw" "$G0LOG" && echo "OK differential optimizer configured" || echo "FAIL differential optimizer"
grep -q "param_groups=2" "$G0LOG" && echo "OK 2 param groups at fit" || echo "WARN check param_groups in log"
grep -q "use_cer=True" "$G0LOG" && echo "OK CER selection" || echo "WARN use_cer not set -> FIX before Gate 1"
grep -Eq "aug-check\] dataset augmentor perturbations = 2" "$G0LOG" && echo "OK gain+speed aug APPLIED to dataset" || echo "WARN aug not confirmed applied -> FIX before Gate 1"
grep -q "FINETUNE_V2_DONE" "$G0LOG" && echo "OK training completed" || { echo "FAIL gate0 training"; exit 1; }
test -f "$OUT/v2_wiring/ft_smoke_joint_unfreeze.nemo" && echo "OK .nemo saved" || { echo "FAIL no .nemo"; exit 1; }
echo "GATE0_DONE"

if [ -z "$EXPORT_CMD" ]; then
  echo; echo ">>> STOP: EXPORT_CMD not set. Locate the pod's .nemo->ONNX export (built export_full70_1),"
  echo ">>> set EXPORT_CMD, then re-run for Gate 1. Gate 0 (wiring) is complete and decisive on its own."
  exit 0
fi

# DE-BUNDLED (research/38 corrected): LEAD with Arm A = encoder-only + gain/speed aug (the
# generalization-correct, attributable, lowest-risk lever that targets the empties root cause).
# Arm B (joint_unfreeze) is a CONDITIONAL follow-up only if Arm A leaves empties/DS on the table.
ARM=${ARM:-encoder_only}   # encoder_only (Arm A, default) | joint_unfreeze (Arm B)
step "GATE 1a — guardrail train ($EPOCHS_GUARD epochs, ARM=$ARM + gain/speed aug, severity-weighted subset)"
python3 "$FT" --mode smoke --epochs "$EPOCHS_GUARD" --freeze "$ARM" \
  --train-json "$M/guardrail_train.json" --val-json "$M/val3k.json" \
  --out-dir "$OUT/v2_guardrail" --train-ctx "[70,1]" 2>&1 | tail -30
V2NEMO="$OUT/v2_guardrail/ft_smoke_${ARM}.nemo"
test -f "$V2NEMO" || { echo "FAIL no guardrail .nemo"; exit 1; }

step "GATE 1b — export v2 (.nemo -> ONNX; re-exports retrained JOINT) + package submission dir"
rm -rf "$OUT/export_v2"; mkdir -p "$OUT/export_v2"
eval "$EXPORT_CMD"   # must produce encoder-model.onnx + decoder_joint-model.onnx in $OUT/export_v2
SUB="$OUT/submission_v2"; SRC=/workspace/finetune/nemo_submission
rm -rf "$SUB"; cp -r "$SRC" "$SUB"
rm -f "$SUB"/weights/encoder_model.onnx* "$SUB"/weights/decoder_model.onnx* 2>/dev/null
cp "$OUT/export_v2"/* "$SUB"/weights/
mv "$SUB"/weights/encoder-model.onnx "$SUB"/weights/encoder_model.onnx
mv "$SUB"/weights/decoder_joint-model.onnx "$SUB"/weights/decoder_model.onnx
sed -i 's/^CHUNK_NEW = 56/CHUNK_NEW = 16/' "$SUB"/model.py
sed -i 's/np.array(\[\[0\]\], dtype=np.int32)/np.array([[BLANK_ID]], dtype=np.int32)/g' "$SUB"/model.py
grep -q '_strip_unk' "$SUB"/model.py || echo "WARN unk-strip not present in base model.py"

step "GATE 1c — faithful eval (Dev_diag + Dev-500 + PD probe), FP32 (joint change is fp32)"
LD=/workspace/sapc-nemotron/track2_starting_kit/local_decode.py; BOOT=/workspace/streaming_cer_bootstrap.py
EMP=/workspace/empties_analysis.py; PROJ=/workspace/sapc-nemotron
STREAM_SMOKE=/dev/shm/Dev_streaming_1.csv
python3 - "$DATA/manifest/Dev_streaming.csv" "$STREAM_SMOKE" <<'PY'
import csv,sys
r=list(csv.DictReader(open(sys.argv[1],newline="")))
w=csv.DictWriter(open(sys.argv[2],"w",newline=""),fieldnames=r[0].keys());w.writeheader();w.writerow(r[0])
PY
# representative Dev-500 (seed 23) + PD-from-Dev subset reuse manifests already on pod where possible
eval_set () {  # $1=tag $2=manifest.csv
  HYP="/dev/shm/v2_${1}.csv"
  ( cd "$SUB" && SAPC2_THREADS=8 python3 "$LD" --submission-dir "$SUB" --manifest-csv "$2" \
      --streaming-manifest-csv "$STREAM_SMOKE" --streaming-interval 0 --data-root "$DATA" \
      --out-csv "$HYP" --out-partial-json "/dev/shm/v2_${1}_p.json" ) 2>&1 | tail -3
  python3 "$BOOT" --manifest-csv "$2" --hyp-csv "$HYP" --utils-dir "$PROJ/utils" --out-json "$ART/v2_${1}_boot.json"
  echo "$HYP"
}
DDHYP=$(eval_set devdiag "$DATA/manifest/Dev_diag.csv")
python3 "$EMP" --manifest-csv "$DATA/manifest/Dev_diag.csv" --hyp-csv "$DDHYP" --utils-dir "$PROJ/utils" \
  --out-json "$ART/v2_devdiag_empties.json"
# NOTE: build/reuse Dev-500 + PD-probe CSV manifests on pod; here we assume $OUT/Dev_rep500_seed23.csv exists.
eval_set dev500 "$OUT/Dev_rep500_seed23.csv"
eval_set pd     "$DATA/manifest/Dev_diag.csv"   # placeholder; replace with PD-only manifest in practice

step "GATE 1 VERDICT (pre-registered thresholds)"
python3 /workspace/v2_gate_verdict.py \
  --devdiag-json "$ART/v2_devdiag_boot.json" \
  --devdiag-empties-json "$ART/v2_devdiag_empties.json" \
  --dev500-json "$ART/v2_dev500_boot.json" \
  --pd-json "$ART/v2_pd_boot.json"
echo "V2_GUARDRAIL_DONE"
