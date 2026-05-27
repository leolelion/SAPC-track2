#!/usr/bin/env python3
"""Phase 1a — Streaming-aware ONNX export + numerical parity check.

Path: NeMo's built-in streaming export (Option A from plan v2 Q-prompt).

What this script does:
  1. Load nvidia/nemotron-speech-streaming-en-0.6b on GPU.
  2. For the chosen att_context_size, enable cache-aware export:
       model.set_export_config({'cache_support': True})
     This flips encoder.export_cache_support=True, then calls
     setup_streaming_params() so input_example() returns cache tensors.
  3. Call model.export(..., check_trace=False). NeMo splits into two
     graphs for cache-aware RNN-T:
       encoder-<stem>.onnx        with cache_last_channel/time/len I/O
       decoder_joint-<stem>.onnx  (decoder + joint combined)
  4. Print summary: ONNX input/output names + shapes for each graph
     (no per-file dump of external-data weight shards).
  5. Numerical parity check on one Dev_streaming utterance:
       - Run 3 chunks through PyTorch encoder.forward_for_export
         with cache carry-forward.
       - Run the same 3 chunks through ORT (CPU EP).
       - Report max abs diff on encoder output per chunk; threshold 1e-4.

Run:
  /root/venvs/nemotron_bench/bin/python phase1a_export_streaming.py \
      --att-context 70 6 \
      --out-dir /root/nemotron_exports/fp32_70_6

Notes:
  - Encoder FP32 is ~2.4 GB so external data is unavoidable; NeMo
    decides automatically. We just suppress the noisy per-file dump.
  - Default att_context_size=[70,6] mirrors the paper's recommended
    (7,10,7) chunk config and is supported by NeMo's streaming export.
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

import nemo.collections.asr as nemo_asr  # noqa: F401
from nemo.collections.asr.models import ASRModel


HF_MODEL = "nvidia/nemotron-speech-streaming-en-0.6b"
DEV_STREAMING = "/workspace/SAPC2/manifest/Dev_streaming.csv"
SAPC2_AUDIO_ROOT = "/workspace/SAPC2"


def load_first_audio(manifest_csv: str, audio_root: str):
    """Load the first WAV from Dev_streaming.csv as float32 mono 16k."""
    import wave
    with open(manifest_csv) as f:
        reader = csv.DictReader(f)
        row = next(reader)
    rel = row["audio_filepath"]
    path = os.path.join(audio_root, rel)
    with wave.open(path, "rb") as w:
        assert w.getframerate() == 16000
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0, path


def summarize_onnx(path: str):
    import onnx
    m = onnx.load(path, load_external_data=False)
    rows = []
    for inp in m.graph.input:
        shape = [
            d.dim_value if d.HasField("dim_value") else d.dim_param
            for d in inp.type.tensor_type.shape.dim
        ]
        rows.append(("IN ", inp.name, shape))
    for out in m.graph.output:
        shape = [
            d.dim_value if d.HasField("dim_value") else d.dim_param
            for d in out.type.tensor_type.shape.dim
        ]
        rows.append(("OUT", out.name, shape))
    for tag, name, shape in rows:
        print(f"    {tag} {name:35s} {shape}", flush=True)


def export_streaming(model, out_dir: str, stem: str = "nemotron"):
    """Run NeMo's streaming-aware export. Returns (encoder_path, decoder_joint_path)."""
    os.makedirs(out_dir, exist_ok=True)
    onnx_path = os.path.join(out_dir, f"{stem}.onnx")

    # Enable cache-aware export. This flips encoder.export_cache_support=True
    # and calls setup_streaming_params() so input_example includes caches.
    model.set_export_config({"cache_support": True})
    assert model.encoder.export_cache_support is True

    print(f"  exporting to {out_dir}/ ...", flush=True)
    t0 = time.time()
    model.export(onnx_path, check_trace=False)
    dt = time.time() - t0
    print(f"  export done in {dt:.1f}s", flush=True)

    # Locate the produced ONNX files. NeMo names them <stem>-encoder.onnx
    # and <stem>-decoder_joint.onnx (or encoder-<stem>.onnx / decoder_joint-<stem>.onnx
    # depending on version). Find by content.
    encoder_path = None
    decoder_path = None
    onnx_files = sorted(
        f for f in os.listdir(out_dir)
        if f.endswith(".onnx") and not f.startswith(".")
    )
    for f in onnx_files:
        if "encoder" in f and "decoder" not in f:
            encoder_path = os.path.join(out_dir, f)
        elif "decoder" in f or "joint" in f:
            decoder_path = os.path.join(out_dir, f)
    if encoder_path is None or decoder_path is None:
        raise RuntimeError(f"Could not locate encoder/decoder ONNX in {out_dir}: {onnx_files}")

    # Summary: total bytes, file counts.
    total = sum(os.path.getsize(os.path.join(out_dir, f)) for f in os.listdir(out_dir))
    n_files = len(os.listdir(out_dir))
    print(f"  output: {n_files} files, total {total/1e9:.2f} GB", flush=True)
    print(f"  encoder: {os.path.basename(encoder_path)} ({os.path.getsize(encoder_path)/1e6:.1f} MB protobuf)", flush=True)
    print(f"  decoder_joint: {os.path.basename(decoder_path)} ({os.path.getsize(decoder_path)/1e6:.1f} MB protobuf)", flush=True)
    return encoder_path, decoder_path


def parity_check(model, encoder_path: str, audio: np.ndarray, n_chunks: int = 3):
    """Compare PyTorch streaming encoder vs ONNX Runtime encoder over n_chunks."""
    import onnxruntime as ort

    device = next(model.parameters()).device
    enc = model.encoder
    cfg = enc.streaming_cfg
    cs0, cs1 = cfg.chunk_size if isinstance(cfg.chunk_size, list) else (cfg.chunk_size, cfg.chunk_size)
    pec0, pec1 = cfg.pre_encode_cache_size if isinstance(cfg.pre_encode_cache_size, list) else (cfg.pre_encode_cache_size, cfg.pre_encode_cache_size)
    chunk_mel_frames = cs1                  # steady-state chunk_size (continuation step)
    cache_mel_frames = pec1
    enc_input_frames = chunk_mel_frames + cache_mel_frames
    print(f"  streaming_cfg: chunk_size={list(cfg.chunk_size)}  pre_encode_cache_size={list(cfg.pre_encode_cache_size)}  shift_size={list(cfg.shift_size)}", flush=True)
    print(f"  per-chunk mel frames: chunk={chunk_mel_frames}  cache={cache_mel_frames}  total_in={enc_input_frames}", flush=True)

    # Get mel features (use the preprocessor with dither=0, pad_to=0 for determinism).
    enc.eval()
    model.preprocessor.featurizer.dither = 0.0
    model.preprocessor.featurizer.pad_to = 0
    audio_t = torch.from_numpy(audio).unsqueeze(0).to(device)
    audio_len = torch.tensor([len(audio)], dtype=torch.long, device=device)
    with torch.inference_mode():
        mel, mel_len = model.preprocessor(input_signal=audio_t, length=audio_len)
    print(f"  mel shape: {tuple(mel.shape)}  valid_len: {int(mel_len.item())}", flush=True)

    # Init caches from PyTorch helper.
    pt_lc, pt_lt, pt_ll = enc.get_initial_cache_state(batch_size=1, device=device)
    # PyTorch caches use (N, B, ...) layout. forward_for_export internally transposes
    # to (B, N, ...) so we feed (B, N, ...) tensors:
    pt_lc_in = pt_lc.transpose(0, 1).contiguous()
    pt_lt_in = pt_lt.transpose(0, 1).contiguous()
    pt_ll_in = pt_ll.clone()

    # Init ORT caches with matching shapes (B, N, ...).
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(encoder_path, so, providers=["CPUExecutionProvider"])
    # Snapshot ORT input metadata for shape reference.
    print(f"  ORT encoder inputs:", flush=True)
    for inp in sess.get_inputs():
        print(f"    {inp.name:30s} type={inp.type}  shape={inp.shape}", flush=True)

    ort_lc = pt_lc_in.detach().cpu().numpy().astype(np.float32)
    ort_lt = pt_lt_in.detach().cpu().numpy().astype(np.float32)
    ort_ll = pt_ll_in.detach().cpu().numpy().astype(np.int64)

    T_valid = int(mel_len.item())
    diffs = []
    for k in range(n_chunks):
        # New chunk_mel_frames starting at k * chunk_mel_frames.
        start_new = k * chunk_mel_frames
        end_new = start_new + chunk_mel_frames
        if end_new > T_valid:
            print(f"  not enough audio for chunk {k} (need {end_new} mel frames, have {T_valid}); stopping", flush=True)
            break

        # Build mel input: [cache_pre | new_chunk], shape [1, 128, enc_input_frames]
        if k == 0:
            cache_pre = torch.zeros(1, mel.shape[1], cache_mel_frames, device=device, dtype=mel.dtype)
        else:
            cache_pre = mel[:, :, start_new - cache_mel_frames:start_new]
        new_chunk = mel[:, :, start_new:end_new]
        chunk_in = torch.cat([cache_pre, new_chunk], dim=2)
        length_in = torch.tensor([enc_input_frames], dtype=torch.long, device=device)

        # ── PyTorch encoder.forward_for_export ────────────────────────
        with torch.inference_mode():
            pt_out = enc.forward_for_export(
                audio_signal=chunk_in,
                length=length_in,
                cache_last_channel=pt_lc_in,
                cache_last_time=pt_lt_in,
                cache_last_channel_len=pt_ll_in,
            )
        # forward_for_export with caches returns
        # (outputs, encoded_lengths, cache_last_channel_next, cache_last_time_next, cache_last_channel_next_len)
        pt_encoded, pt_enc_len, pt_lc_next, pt_lt_next, pt_ll_next = pt_out
        pt_encoded_np = pt_encoded.detach().cpu().numpy()

        # ── ORT encoder ───────────────────────────────────────────────
        ort_inputs = {
            "audio_signal": chunk_in.detach().cpu().numpy().astype(np.float32),
            "length": length_in.detach().cpu().numpy().astype(np.int64),
            "cache_last_channel": ort_lc,
            "cache_last_time": ort_lt,
            "cache_last_channel_len": ort_ll,
        }
        ort_outs = sess.run(None, ort_inputs)
        # Output order may vary; identify by ORT output names.
        out_names = [o.name for o in sess.get_outputs()]
        named = dict(zip(out_names, ort_outs))
        # Choose the first output ("outputs" or "encoded") by name heuristic.
        enc_out_name = next((n for n in out_names if "output" in n.lower() or "encoded" == n), out_names[0])
        ort_encoded_np = named[enc_out_name]
        # Update ORT caches. Order matters — match the *_len name first, otherwise
        # the broader "cache_last_channel" + "next" pattern swallows it.
        for n in out_names:
            ln = n.lower()
            if "cache_last_channel" in ln and ("_len" in ln or "_length" in ln):
                ort_ll = named[n]
            elif "cache_last_channel" in ln and "next" in ln:
                ort_lc = named[n]
            elif "cache_last_time" in ln and "next" in ln:
                ort_lt = named[n]

        # Update PyTorch caches for next iteration (transpose back from (B,N,...) to itself; forward_for_export returns (B,N,...))
        pt_lc_in = pt_lc_next.contiguous()
        pt_lt_in = pt_lt_next.contiguous()
        pt_ll_in = pt_ll_next.contiguous()

        max_abs = float(np.abs(pt_encoded_np - ort_encoded_np).max())
        diffs.append(max_abs)
        print(
            f"  chunk {k}: PT_out shape={pt_encoded_np.shape}  ORT_out shape={ort_encoded_np.shape}  "
            f"max|PT-ORT|={max_abs:.3e}",
            flush=True,
        )

    if diffs:
        print(f"  PARITY: max-of-max abs diff = {max(diffs):.3e}  (threshold 1e-4)", flush=True)
        return max(diffs)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--att-context", nargs=2, type=int, default=[70, 6], metavar=("LEFT", "RIGHT"))
    ap.add_argument("--out-dir", default="/root/nemotron_exports/fp32_70_6")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--no-parity", action="store_true", help="skip the numerical parity check")
    args = ap.parse_args()

    print(f"=== Phase 1a: streaming export for att_context={args.att_context} ===", flush=True)
    print(f"loading {HF_MODEL} on {args.device} ...", flush=True)
    t0 = time.time()
    model = ASRModel.from_pretrained(HF_MODEL, map_location=args.device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model = model.to(args.device)
    print(f"  loaded in {time.time()-t0:.1f}s ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)", flush=True)

    print(f"setting att_context_size={args.att_context}", flush=True)
    model.encoder.set_default_att_context_size(list(args.att_context))

    encoder_path, decoder_path = export_streaming(model, args.out_dir)

    print(f"\n=== ONNX encoder I/O ===", flush=True)
    summarize_onnx(encoder_path)
    print(f"\n=== ONNX decoder_joint I/O ===", flush=True)
    summarize_onnx(decoder_path)

    if args.no_parity:
        return

    # Move model + preprocessor to CPU for parity (ORT is CPU EP; PT on CPU
    # keeps the comparison apples-to-apples).
    print(f"\n=== Numerical parity (3 chunks, PT-CPU vs ORT-CPU) ===", flush=True)
    model = model.to("cpu")
    audio, audio_path = load_first_audio(DEV_STREAMING, SAPC2_AUDIO_ROOT)
    print(f"  utterance: {os.path.basename(audio_path)}  duration={len(audio)/16000:.2f}s", flush=True)
    max_diff = parity_check(model, encoder_path, audio, n_chunks=3)
    if max_diff is None:
        print("  parity check skipped (not enough audio)", flush=True)
    elif max_diff < 1e-4:
        print(f"  PASS (max diff {max_diff:.3e} < 1e-4)", flush=True)
    else:
        print(f"  WARN: parity above 1e-4 (got {max_diff:.3e}). Investigate before sweeping.", flush=True)


if __name__ == "__main__":
    main()
