#!/usr/bin/env python3
"""Inspect Nemotron streaming model: streaming_cfg per att_context_size,
and what NeMo's .export() produces.

Run on the pod with:
    /root/venvs/nemotron_bench/bin/python phase1_inspect.py
"""
import os
import sys
import tempfile

import torch
from omegaconf import open_dict

import nemo.collections.asr as nemo_asr  # noqa: F401
from nemo.collections.asr.models import ASRModel

HF_MODEL = "nvidia/nemotron-speech-streaming-en-0.6b"
ATT_CONTEXTS = [[70, 0], [70, 1], [70, 6], [70, 13]]


def main():
    print(f"Loading {HF_MODEL} (CPU) ...", flush=True)
    model = ASRModel.from_pretrained(HF_MODEL, map_location="cpu")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"loaded ({params_m:.1f}M params)", flush=True)

    print("\n=== Default cfg ===", flush=True)
    print(f"att_context_size (default): {model.encoder.att_context_size}", flush=True)
    try:
        ss = model.encoder.pre_encode.get_sampling_frames()
        print(f"subsampling sampling_frames: {ss}", flush=True)
    except Exception as e:
        print(f"subsampling sampling_frames: n/a ({e})", flush=True)
    print(f"preprocessor.features: {model.cfg.preprocessor.features}", flush=True)
    print(f"encoder layers attr present: {hasattr(model.encoder, 'layers')}", flush=True)
    print(f"d_model: {model.cfg.encoder.get('d_model', 'n/a')}", flush=True)

    print("\n=== streaming_cfg per att_context_size ===", flush=True)
    for ac in ATT_CONTEXTS:
        model.encoder.set_default_att_context_size(ac)
        cfg = model.encoder.streaming_cfg
        cs = list(cfg.chunk_size)
        ss = list(cfg.shift_size)
        pec = list(cfg.pre_encode_cache_size)
        de = cfg.drop_extra_pre_encoded
        # 10ms mel hop -> sec per mel frame
        sec_step0 = cs[0] * 0.01
        sec_cont = cs[1] * 0.01
        print(
            f"  att_context_size={ac}  chunk_size={cs}  shift_size={ss}  "
            f"pre_encode_cache_size={pec}  drop_extra={de}  "
            f"audio_per_step_0={sec_step0*1000:.0f}ms  audio_per_step_cont={sec_cont*1000:.0f}ms",
            flush=True,
        )

    print("\n=== Try NeMo built-in export ===", flush=True)
    model.encoder.set_default_att_context_size([70, 6])
    td = tempfile.mkdtemp(dir="/root", prefix="nemotron_export_")
    print(f"export dir: {td}", flush=True)
    path = os.path.join(td, "nemotron.onnx")
    try:
        ret = model.export(path, check_trace=False)
        print(f"  export() returned: {ret}", flush=True)
        for f in sorted(os.listdir(td)):
            full = os.path.join(td, f)
            if os.path.isfile(full):
                size = os.path.getsize(full) / 1e6
                print(f"  produced: {f}  ({size:.1f} MB)", flush=True)
        import onnx
        for fname in sorted(os.listdir(td)):
            if not fname.endswith(".onnx"):
                continue
            full = os.path.join(td, fname)
            try:
                m = onnx.load(full)
            except Exception as e:
                print(f"  could not load {fname}: {e}", flush=True)
                continue
            print(f"  --- {fname} ---", flush=True)
            print(f"  inputs:", flush=True)
            for inp in m.graph.input:
                shape = [d.dim_value if d.HasField('dim_value') else d.dim_param for d in inp.type.tensor_type.shape.dim]
                print(f"    {inp.name}  shape={shape}", flush=True)
            print(f"  outputs:", flush=True)
            for out in m.graph.output:
                shape = [d.dim_value if d.HasField('dim_value') else d.dim_param for d in out.type.tensor_type.shape.dim]
                print(f"    {out.name}  shape={shape}", flush=True)
    except Exception as e:
        import traceback
        print(f"  export FAILED: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()


if __name__ == "__main__":
    main()
