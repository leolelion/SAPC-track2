#!/usr/bin/env python3
"""Evaluate a fine-tuned Zipformer ONNX model on SAPC2 Dev set.

Writes a CSV with columns [id, raw_hypos] that can be fed directly into the
official SAPC2 scoring pipeline (evaluate.sh --start_stage 2).
"""

import argparse
import csv
import sys
import time
import wave
from pathlib import Path

import numpy as np


def read_wav_mono_float32(audio_path: Path) -> np.ndarray:
    """Read a 16kHz mono WAV file and return float32 samples in [-1, 1]."""
    with wave.open(str(audio_path), "rb") as wf:
        if wf.getframerate() != 16000:
            raise ValueError(f"Expected 16kHz, got {wf.getframerate()}Hz: {audio_path}")
        if wf.getsampwidth() != 2:
            raise ValueError(f"Expected 16-bit PCM, got {wf.getsampwidth()*8}-bit: {audio_path}")
        raw = wf.readframes(wf.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return samples


def decode_utterance(recognizer, audio: np.ndarray, sample_rate: int = 16000) -> str:
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, audio)
    # Tail padding flushes the final chunk through the streaming encoder
    tail_pad = np.zeros(int(0.3 * sample_rate), dtype=np.float32)
    stream.accept_waveform(sample_rate, tail_pad)
    stream.input_finished()
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)
    return recognizer.get_result(stream).text.strip()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned Zipformer ONNX on SAPC2 Dev"
    )
    parser.add_argument("--onnx-dir", type=Path, required=True,
                        help="Directory with encoder.onnx, decoder.onnx, joiner.onnx, tokens.txt")
    parser.add_argument("--manifest-csv", type=Path, required=True,
                        help="Dev manifest CSV (e.g. /workspace/SAPC2/processed/manifest/Dev.csv)")
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Root that audio_filepath values are relative to")
    parser.add_argument("--out-csv", type=Path, required=True,
                        help="Output hypothesis CSV (id, raw_hypos)")
    parser.add_argument("--num-threads", type=int, default=4,
                        help="ONNX Runtime CPU threads per decoder")
    parser.add_argument("--decoding-method", default="greedy_search",
                        choices=["greedy_search", "modified_beam_search"],
                        help="Decoding method")
    parser.add_argument("--beam-size", type=int, default=4,
                        help="Beam size (only used with modified_beam_search)")
    args = parser.parse_args()

    # ── Validate ONNX files ──
    for fname in ("encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt"):
        p = args.onnx_dir / fname
        if not p.exists():
            print(f"ERROR: Missing {p}", file=sys.stderr)
            sys.exit(1)

    # ── Load recognizer ──
    try:
        import sherpa_onnx
    except ImportError:
        print("ERROR: sherpa_onnx not installed. Run: pip install sherpa-onnx", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model from {args.onnx_dir} ...")
    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=str(args.onnx_dir / "tokens.txt"),
        encoder=str(args.onnx_dir / "encoder.onnx"),
        decoder=str(args.onnx_dir / "decoder.onnx"),
        joiner=str(args.onnx_dir / "joiner.onnx"),
        num_threads=args.num_threads,
        decoding_method=args.decoding_method,
        max_active_paths=args.beam_size,
    )
    print("Model loaded.")

    # ── Read manifest ──
    entries = []
    with open(args.manifest_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = row["id"]
            audio_path = args.data_root / row["audio_filepath"]
            entries.append((uid, audio_path))

    print(f"Evaluating {len(entries)} utterances ...")

    # ── Decode ──
    results = []
    errors = 0
    t0 = time.time()

    for i, (uid, audio_path) in enumerate(entries):
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(entries) - i - 1) / rate
            print(f"  [{i+1}/{len(entries)}] {elapsed:.0f}s elapsed, "
                  f"~{remaining:.0f}s remaining, {rate:.1f} utt/s")

        try:
            audio = read_wav_mono_float32(audio_path)
            text = decode_utterance(recognizer, audio)
        except Exception as e:
            print(f"  WARNING: Failed on {uid} ({audio_path}): {e}", file=sys.stderr)
            text = ""
            errors += 1

        results.append((uid, text))

    elapsed = time.time() - t0
    print(f"\nDecoding complete: {len(results)} utterances in {elapsed:.0f}s "
          f"({len(results)/elapsed:.1f} utt/s), {errors} errors")

    # ── Write output CSV ──
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "raw_hypos"])
        for uid, text in results:
            writer.writerow([uid, text])

    print(f"\nHypotheses saved to: {args.out_csv}")
    print("\nTo score with the official pipeline:")
    print(f"  cd /workspace/SAPC-template")
    print(f"  bash evaluate.sh \\")
    print(f"      --start_stage 2 --stop_stage 2 \\")
    print(f"      --split Dev-all \\")
    print(f"      --hyp-csv {args.out_csv}")


if __name__ == "__main__":
    main()
