#!/usr/bin/env bash
# Pre-full-run experiments:
#  EXP1a (robust): export finetuned enc-only -> ONNX with cache_support=True (proves the finetuned deploy path)
#  EXP2  (robust): target-text ablation — train smoke enc-only on CASED+punct text, eval vs normalized (30.8%)
#  EXP1b (best-effort): true chunked streaming CER via NeMo streaming-sim at [70,1] (resolves transcribe-vs-stream)
set -uo pipefail
OUT=/workspace/finetune/nemo_ft
LOG="$OUT/prerun_exp.log"; : > "$LOG"; exec > >(tee -a "$LOG") 2>&1
source /workspace/nemoenv/bin/activate
ENC="$OUT/gate2_enc/ft_smoke_encoder_only.nemo"
BASE="$OUT/nemotron-speech-streaming-en-0.6b.nemo"
DIAG=/workspace/SAPC2/manifest/Dev_diag.csv
step(){ echo; echo "############### $* ###############"; }

step "EXP1a. EXPORT finetuned enc-only -> ONNX (cache_support=True)"
python3 - <<PY 2>&1 | tail -18
import os, nemo.collections.asr as na
m = na.models.EncDecRNNTBPEModel.restore_from("$ENC", map_location="cpu")
try:
    m.set_export_config({'cache_support':'True'}); print("cache_support=True set")
except Exception as e:
    print("set_export_config err:", e)
os.makedirs("$OUT/export_enc", exist_ok=True)
try:
    out = m.export("$OUT/export_enc/model.onnx")
    print("export returned:", out)
    print("files:", sorted(os.listdir("$OUT/export_enc")))
except Exception as e:
    print("EXPORT_FAIL:", repr(e))
PY

step "EXP2. TARGET-TEXT ABLATION (CASED+punct vs normalized 30.8%)"
python3 /workspace/prep_nemo_manifest.py --train-csv /workspace/SAPC2/manifest/Train.csv \
  --data-root /workspace/SAPC2 --out-dir "$OUT/manifests_cased" --target text 2>&1 | tail -8
python3 /workspace/nemo_finetune.py --mode smoke --freeze encoder_only \
  --train-json "$OUT/manifests_cased/smoke_train.json" --val-json "$OUT/manifests_cased/dev_internal.json" \
  --out-dir "$OUT/gate2_enc_cased" --train-ctx "[70,1]" 2>&1 | tail -22
echo "--- EVAL cased-trained on Dev_diag (eval normalizer makes it fair vs normalized-trained 30.8%) ---"
python3 /workspace/nemo_eval_diag.py "$OUT/gate2_enc_cased/ft_smoke_encoder_only.nemo" "$DIAG" /workspace/SAPC2 "[70,1]" 2>&1 | tail -12

step "EXP1b. STREAMING-SIM (best-effort) true chunked CER at [70,1]"
cd /workspace
curl -sL -o stream_infer.py https://raw.githubusercontent.com/NVIDIA-NeMo/NeMo/main/examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py && echo "got stream_infer.py" || echo "download failed"
python3 - <<PY
import csv,json,os
rows=list(csv.DictReader(open("$DIAG")))
with open("$OUT/dev_diag_nemo.json","w") as f:
    for r in rows:
        f.write(json.dumps({"audio_filepath":os.path.join("/workspace/SAPC2",r["audio_filepath"]),
                            "duration":float(r["duration"]),"text":r["norm_text_without_disfluency"]})+"\n")
print("manifest rows", len(rows))
PY
echo "--- stream_infer args (confirm interface) ---"
python3 stream_infer.py --help 2>&1 | grep -iE 'att_context|manifest|asr_model|output|chunk|left' | head -25 || true
echo "--- stream-sim: finetuned enc @ [70,1] ---"
python3 stream_infer.py --asr_model "$ENC" --manifest_file "$OUT/dev_diag_nemo.json" --batch_size 16 \
  --att_context_size 70 1 --output_path "$OUT/stream_enc.json" 2>&1 | tail -18 || echo "streamsim enc FAIL (see args above)"
echo "PRERUN_DONE"
