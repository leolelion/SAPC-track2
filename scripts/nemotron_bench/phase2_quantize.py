#!/usr/bin/env python3
"""Phase 2 — k-quant / RTN quantization of the encoder ONNX via
onnxruntime.quantization.matmul_nbits_quantizer.

Paper §6.2: quantize encoder only; decoder + joint stay FP32 (they're
small enough that quantization gains are negligible and accuracy risk
is real).

Usage:
    /root/venvs/nemotron_bench/bin/python phase2_quantize.py \\
        --encoder-in /path/to/encoder_fp32.onnx \\
        --encoder-out /path/to/encoder_int8_kquant.onnx \\
        --bits 8 --algo kquant --block-size 128
"""
import argparse
import os
import shutil
import time
from pathlib import Path

import onnx
from onnxruntime.quantization.matmul_nbits_quantizer import (
    MatMulNBitsQuantizer,
    KQuantWeightOnlyQuantConfig,
    RTNWeightOnlyQuantConfig,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder-in", required=True)
    ap.add_argument("--encoder-out", required=True)
    ap.add_argument("--bits", type=int, default=8, choices=[4, 8])
    ap.add_argument("--algo", default="kquant", choices=["kquant", "rtn"])
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--is-symmetric", action="store_true", default=False)
    args = ap.parse_args()

    print(f"=== Quantizing encoder: {args.algo} {args.bits}-bit, block_size={args.block_size} ===", flush=True)
    print(f"  input:  {args.encoder_in}", flush=True)
    print(f"  output: {args.encoder_out}", flush=True)

    in_size = sum(os.path.getsize(args.encoder_in if os.path.isfile(args.encoder_in)
                                  else os.path.dirname(args.encoder_in) + "/" + f)
                  for f in [os.path.basename(args.encoder_in)]) / 1e6
    # If encoder uses external data, the .onnx protobuf is tiny; the weights live
    # in sibling files. For a fair "size" measurement, we'll report the protobuf
    # size only here.
    print(f"  input protobuf size: {os.path.getsize(args.encoder_in)/1e6:.1f} MB", flush=True)

    # Load model (with external data if present).
    print("  loading model ...", flush=True)
    t0 = time.time()
    model = onnx.load(args.encoder_in, load_external_data=True)
    print(f"  loaded in {time.time()-t0:.1f}s; nodes={len(model.graph.node)}", flush=True)

    # Pick config.
    if args.algo == "kquant":
        algo_config = KQuantWeightOnlyQuantConfig()
        print("  algorithm: KQuant", flush=True)
    elif args.algo == "rtn":
        algo_config = RTNWeightOnlyQuantConfig()
        print("  algorithm: RTN", flush=True)
    else:
        raise ValueError(args.algo)

    # Build quantizer.
    quantizer = MatMulNBitsQuantizer(
        model,
        bits=args.bits,
        block_size=args.block_size,
        is_symmetric=args.is_symmetric,
        algo_config=algo_config,
    )

    # Run quantization.
    print("  quantizing ...", flush=True)
    t0 = time.time()
    quantizer.process()
    print(f"  quantize done in {time.time()-t0:.1f}s", flush=True)

    # Save quantized model.
    out_dir = os.path.dirname(args.encoder_out) or "."
    os.makedirs(out_dir, exist_ok=True)
    print("  saving ...", flush=True)
    t0 = time.time()
    # Save with external data so size stays manageable.
    # For 8-bit at ~1.2 GB likely doesn't need external; 4-bit definitely won't.
    onnx.save(
        quantizer.model.model,
        args.encoder_out,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=os.path.basename(args.encoder_out) + ".data",
    )
    print(f"  saved in {time.time()-t0:.1f}s", flush=True)

    # Report sizes.
    total_out_bytes = 0
    out_files = []
    out_dir_path = Path(out_dir)
    base = os.path.basename(args.encoder_out)
    for f in os.listdir(out_dir):
        if f == base or f.startswith(base):
            full = os.path.join(out_dir, f)
            sz = os.path.getsize(full)
            total_out_bytes += sz
            out_files.append((f, sz))
    print(f"  produced ({len(out_files)} files, {total_out_bytes/1e6:.1f} MB total):", flush=True)
    for f, sz in out_files:
        print(f"    {f}  {sz/1e6:.1f} MB", flush=True)

    # Compare to input.
    in_dir = os.path.dirname(args.encoder_in)
    in_base = os.path.basename(args.encoder_in)
    in_total = os.path.getsize(args.encoder_in)
    # Also walk sibling external-data files associated with the input model.
    for f in os.listdir(in_dir):
        if f != in_base and not f.endswith(".onnx") and not f.endswith(".bin") and not f.endswith(".meta"):
            in_total += os.path.getsize(os.path.join(in_dir, f))
    print(f"  approx input bundle: {in_total/1e6:.1f} MB", flush=True)
    print(f"  size reduction: {(1 - total_out_bytes/max(1,in_total))*100:.1f}%", flush=True)


if __name__ == "__main__":
    main()
