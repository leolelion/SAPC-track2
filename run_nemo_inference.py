#!/usr/bin/env python3
"""Generic NeMo ASR inference harness for the SAPC2 benchmark.

Loads any .nemo checkpoint (HF repo id or local path), runs `model.transcribe`
over a manifest CSV, and writes a hypothesis CSV in the format `evaluate.sh`
expects (`id,raw_hypos`).

Usage:
  python run_nemo_inference.py \
      --model nvidia/nemotron-speech-streaming-en-0.6b \
      --manifest /workspace/SAPC2/manifest/Dev.csv \
      --audio-root /workspace/SAPC2 \
      --out hyps/R1_baseline_nemotron.csv \
      --batch-size 16
"""
import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def load_model(model_arg: str):
    from nemo.collections.asr.models import ASRModel

    if model_arg.endswith(".nemo") and os.path.isfile(model_arg):
        print(f"[load] restore_from {model_arg}", flush=True)
        model = ASRModel.restore_from(model_arg)
    else:
        print(f"[load] from_pretrained {model_arg}", flush=True)
        model = ASRModel.from_pretrained(model_arg)
    model.eval()
    return model


def quick_wer(hyps, refs):
    """Token-level WER on a small list, just for sanity preview."""
    import re

    def norm(s):
        return re.sub(r"\s+", " ", re.sub(r"[^a-z' ]", " ", s.lower())).strip()

    total_err, total_ref = 0, 0
    for h, r in zip(hyps, refs):
        h_toks, r_toks = norm(h).split(), norm(r).split()
        # Levenshtein on tokens
        n, m = len(r_toks), len(h_toks)
        if n == 0:
            total_err += m
            continue
        dp = list(range(m + 1))
        for i in range(1, n + 1):
            prev, dp[0] = dp[0], i
            for j in range(1, m + 1):
                cur = dp[j]
                if r_toks[i - 1] == h_toks[j - 1]:
                    dp[j] = prev
                else:
                    dp[j] = 1 + min(prev, dp[j], dp[j - 1])
                prev = cur
        total_err += dp[m]
        total_ref += n
    return (total_err / total_ref) if total_ref else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="HF repo id or local .nemo path")
    ap.add_argument("--manifest", required=True, help="Manifest CSV")
    ap.add_argument("--audio-root", required=True,
                    help="Root prefix for audio_filepath column")
    ap.add_argument("--out", required=True, help="Output hypothesis CSV")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, only process the first N rows (for smoke tests)")
    ap.add_argument("--device", default="cuda",
                    help="cuda or cpu (default cuda; falls back to cpu if unavailable)")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.manifest)
    if args.limit > 0:
        df = df.head(args.limit).copy()
    print(f"[manifest] {args.manifest}: {len(df)} rows", flush=True)
    if not {"id", "audio_filepath"}.issubset(df.columns):
        raise SystemExit(f"manifest missing required columns: {df.columns.tolist()}")

    audio_paths = [os.path.join(args.audio_root, p) for p in df["audio_filepath"].tolist()]
    missing = [p for p in audio_paths if not os.path.isfile(p)]
    if missing:
        print(f"[warn] {len(missing)} audio files missing, e.g. {missing[:3]}", flush=True)

    model = load_model(args.model)

    # Move to device if requested.
    import torch
    if args.device == "cuda" and torch.cuda.is_available():
        model = model.to("cuda")
        device_str = f"cuda ({torch.cuda.get_device_name(0)})"
    else:
        model = model.to("cpu")
        device_str = "cpu"
    print(f"[device] {device_str}", flush=True)

    # Log streaming params if present.
    enc = getattr(model, "encoder", None)
    acs = getattr(enc, "att_context_size", None)
    print(f"[encoder.att_context_size] {acs}", flush=True)

    durations = df["duration"].astype(float) if "duration" in df.columns else None
    if durations is not None:
        print(f"[audio] N={len(df)}, mean dur={durations.mean():.2f}s, "
              f"max={durations.max():.2f}s, total={durations.sum()/3600:.2f}h", flush=True)

    print(f"[transcribe] batch_size={args.batch_size}", flush=True)
    t0 = time.time()
    if torch.cuda.is_available() and args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    hyps_raw = model.transcribe(audio_paths, batch_size=args.batch_size)
    elapsed = time.time() - t0

    # NeMo can return: list[Hypothesis], list[str], or (list, list) tuple.
    if isinstance(hyps_raw, tuple):
        hyps_raw = hyps_raw[0]
    texts = []
    for h in hyps_raw:
        if hasattr(h, "text"):
            texts.append(h.text)
        else:
            texts.append(str(h))

    out_df = pd.DataFrame({"id": df["id"].tolist(), "raw_hypos": texts})
    out_df.to_csv(out_path, index=False)

    n = len(texts)
    audio_h = (durations.sum() / 3600) if durations is not None else float("nan")
    rtf = (elapsed / durations.sum()) if durations is not None else float("nan")
    peak_gb = (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() and args.device == "cuda" else 0.0
    print(f"[done] wrote {out_path} ({n} rows)", flush=True)
    print(f"[stats] wall={elapsed:.1f}s  audio={audio_h:.2f}h  RTF={rtf:.4f}  "
          f"batch_size={args.batch_size}  peak_gpu={peak_gb:.2f}GB", flush=True)

    # Quick WER preview against the first 50 refs (no normalizer — informative only).
    if "norm_text_without_disfluency" in df.columns:
        k = min(50, n)
        wer50 = quick_wer(texts[:k], df["norm_text_without_disfluency"].tolist()[:k])
        print(f"[preview] crude WER (first {k}, no proper normalizer) = {wer50*100:.2f}%", flush=True)


if __name__ == "__main__":
    sys.exit(main())
