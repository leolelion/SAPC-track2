#!/usr/bin/env bash
# Pre-full-run verification: (A) Train/Dev speaker disjointness (leakage?), (B) apples-to-apples
# zero-shot baseline via the SAME NeMo-transcribe-[70,1] path on Dev_diag (isolates the finetuning gain).
set -uo pipefail
OUT=/workspace/finetune/nemo_ft
LOG="$OUT/verify_gate2.log"; : > "$LOG"; exec > >(tee -a "$LOG") 2>&1
source /workspace/nemoenv/bin/activate
BASE="$OUT/nemotron-speech-streaming-en-0.6b.nemo"
DIAG=/workspace/SAPC2/manifest/Dev_diag.csv
step(){ echo; echo "############### $* ###############"; }

step "A. SPEAKER DISJOINTNESS: Train vs Dev (leakage check)"
python3 - <<'PY'
import csv
def spk(p):
    return {r["speaker"] for r in csv.DictReader(open(p))}
tr = spk("/workspace/SAPC2/manifest/Train.csv")
dv = spk("/workspace/SAPC2/manifest/Dev.csv")
ov = tr & dv
print(f"Train speakers={len(tr)}  Dev speakers={len(dv)}  OVERLAP={len(ov)}")
if ov: print("  LEAKAGE — overlapping speakers (up to 20):", list(ov)[:20])
# are the diag-eval speakers (esp. the oversampled worst ones) in Train?
diag = {r["speaker"] for r in csv.DictReader(open("/workspace/SAPC2/manifest/Dev_diag.csv"))}
leaked = diag & tr
print(f"Dev_diag speakers={len(diag)}  of which in Train={len(leaked)}")
if leaked: print("  diag speakers ALSO in Train:", list(leaked)[:20])
print("VERDICT:", "CLEAN (disjoint)" if not ov else "LEAKAGE PRESENT")
PY

step "B. APPLES-TO-APPLES baseline: ZERO-SHOT via NeMo-transcribe-[70,1] on Dev_diag"
echo "(compare to finetuned enc-only on the SAME path = 30.8% CER / 9% empty)"
python3 /workspace/nemo_eval_diag.py "$BASE" "$DIAG" /workspace/SAPC2 "[70,1]" 2>&1 | tail -12

echo "VERIFY_DONE"
