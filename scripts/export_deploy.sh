#!/usr/bin/env bash
# Deployment loop step 2: export the FULL encoder-only winner (and zero-shot base) at the deploy context [70,1]
# with cache_support, and dump NeMo's streaming chunk/cache params (-> the model.py constants for [70,1]).
set -uo pipefail
OUT=/workspace/finetune/nemo_ft
LOG="$OUT/export_deploy.log"; : > "$LOG"; exec > >(tee -a "$LOG") 2>&1
source /workspace/nemoenv/bin/activate
WINNER="$OUT/full_enc/ft_smoke_encoder_only.nemo"
BASE="$OUT/nemotron-speech-streaming-en-0.6b.nemo"

python3 - <<'PY'
import os, nemo.collections.asr as na
for tag, path in [("full70_1", "/workspace/finetune/nemo_ft/full_enc/ft_smoke_encoder_only.nemo"),
                  ("base70_1", "/workspace/finetune/nemo_ft/nemotron-speech-streaming-en-0.6b.nemo")]:
    print(f"\n###### {tag} : {path} ######")
    m = na.models.EncDecRNNTBPEModel.restore_from(path, map_location="cpu")
    m.encoder.set_default_att_context_size([70, 1])
    try:
        m.encoder.setup_streaming_params()
    except Exception as e:
        print("setup_streaming_params:", e)
    sc = getattr(m.encoder, "streaming_cfg", None)
    print("att_context_size now:", m.encoder.att_context_size)
    print("streaming_cfg:", sc)
    if sc is not None:
        for a in ["chunk_size","shift_size","cache_drop_size","last_channel_cache_size",
                  "pre_encode_cache_size","valid_out_len","drop_extra_pre_encoded"]:
            print(f"  cfg.{a} = {getattr(sc, a, '<absent>')}")
    od = f"/workspace/finetune/nemo_ft/export_{tag}"; os.makedirs(od, exist_ok=True)
    m.set_export_config({'cache_support': 'True'})
    out = m.export(os.path.join(od, "model.onnx"))
    print("exported ->", od, ":", sorted([f for f in os.listdir(od) if f.endswith('.onnx')]))
PY
echo "EXPORT_DEPLOY_DONE"
