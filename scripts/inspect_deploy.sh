#!/usr/bin/env bash
# Deployment loop step 1: inspect NeMo's exported ONNX I/O (cache tensor names/shapes) vs the existing
# Nemotron submission model.py, so we can build/adapt the 100ms-chunk streaming model.py.
set -uo pipefail
OUT=/workspace/finetune/nemo_ft
LOG="$OUT/inspect_deploy.log"; : > "$LOG"; exec > >(tee -a "$LOG") 2>&1
source /workspace/nemoenv/bin/activate
echo "=== exported ONNX files ==="; ls -la "$OUT/export_enc/"*.onnx 2>/dev/null
python3 - <<'PY'
import onnx
base="/workspace/finetune/nemo_ft/export_enc"
for f in ["encoder-model.onnx","decoder_joint-model.onnx"]:
    p=f"{base}/{f}"
    try:
        m=onnx.load(p, load_external_data=False)
        print(f"\n### {f} ###\nINPUTS:")
        for i in m.graph.input:
            print("  ", i.name, [(d.dim_value or d.dim_param) for d in i.type.tensor_type.shape.dim])
        print("OUTPUTS:")
        for o in m.graph.output:
            print("  ", o.name, [(d.dim_value or d.dim_param) for d in o.type.tensor_type.shape.dim])
    except Exception as e:
        print(f"{f}: load failed -> {e}")
PY
echo
echo "=== EXISTING submission model.py ONNX feed/fetch names (for mapping) ==="
grep -nE '_enc_sess.run|_dec_sess.run|audio_signal|length|cache_last|encoder_outputs|input_states|targets|target_length|"outputs"|encoded_lengths' \
  /workspace/finetune/nemo_submission/model.py 2>/dev/null | head -50
echo "INSPECT_DONE"
