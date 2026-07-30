#!/usr/bin/env python3
"""Size the blank-penalty grid from the model instead of guessing it.

`investigations/step01_runbook.md` watch-item #5: the joint emits UNNORMALISED logits, so
we have no prior for what a blank penalty of 1.0 means. A grid guessed at 0-4 could sit
entirely below the responsive region (flat curve, wasted session) or entirely above it
(everything collapses, wasted session).

This runs ONE decode pass, records the blank-vs-runner-up logit margin at every greedy
step, and reports what fraction of blank decisions each candidate beta would flip. One
pass costs ~1/7th of the grid it calibrates.

It touches nothing that ships: the submission's ONNX decoder session is wrapped from the
OUTSIDE (a recording proxy around `Model._dec.run`), so `model.py` runs exactly as
packaged, and `setup.sh` is never executed.

    python3 scripts/probe_blank_margin.py \
        --submission-dir /workspace/parakeet_onnx/extract \
        --manifest-csv $DATA/manifest/Dev_diag.csv --data-root $DATA \
        --max-utts 40 --out-json probe_margin.json

FIRST-ORDER ONLY. Flipping a blank changes the prediction-network state, which changes
every later step, so the reported flip fractions size the grid; they do not predict CER.
Only `evaluate.sh` does that.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import wave
from pathlib import Path
from typing import List

import numpy as np

SAMPLE_RATE = 16000
CHUNK_SIZE = 1600  # 100 ms, the contract


def read_wave(path: str) -> np.ndarray:
    with wave.open(path, "rb") as f:
        if f.getframerate() != SAMPLE_RATE:
            raise ValueError(f"{path}: expected {SAMPLE_RATE} Hz, got {f.getframerate()}")
        if f.getnchannels() != 1:
            raise ValueError(f"{path}: expected mono, got {f.getnchannels()} channels")
        if f.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16-bit, got {f.getsampwidth()*8}-bit")
        raw = f.readframes(f.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def synthetic_audio(seconds: float = 3.0, seed: int = 0) -> np.ndarray:
    """Smoke-test signal for hosts with no SAP data. Produces garbage transcripts on
    purpose -- it exercises the code path and reveals the LOGIT SCALE, nothing more."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    voiced = 0.25 * np.sin(2 * np.pi * 140 * t) * (1 + 0.5 * np.sin(2 * np.pi * 4 * t))
    return (voiced + 0.02 * rng.standard_normal(t.size)).astype(np.float32)


class MarginRecorder:
    """Proxy around the ONNX decoder-joint session. Mirrors model.py's own output
    selection (`out.get("outputs", raw[0])`) so we read the same tensor the argmax does."""

    def __init__(self, sess, out_names: List[str], blank_id: int):
        self._sess = sess
        self._out_names = list(out_names)
        self._blank = blank_id
        self.margins: List[float] = []      # blank - best_other, one per greedy step
        self.blank_won: List[bool] = []

    def run(self, output_names, feeds):
        raw = self._sess.run(output_names, feeds)
        try:
            idx = self._out_names.index("outputs")
        except ValueError:
            idx = 0
        flat = np.asarray(raw[idx]).reshape(-1)
        blank_logit = float(flat[self._blank])
        other = np.delete(flat, self._blank)
        best_other = float(other.max())
        self.margins.append(blank_logit - best_other)
        self.blank_won.append(blank_logit > best_other)
        return raw

    def __getattr__(self, name):  # anything else the wrapper needs passes straight through
        return getattr(self._sess, name)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--submission-dir", required=True, type=Path)
    p.add_argument("--manifest-csv", type=Path, help="omit to run on --synthetic audio")
    p.add_argument("--data-root", type=Path)
    p.add_argument("--max-utts", type=int, default=40)
    p.add_argument("--synthetic", action="store_true", help="smoke-test with generated audio (no SAP data)")
    p.add_argument("--betas", default="0.25,0.5,1.0,1.5,2.0,3.0,4.0,6.0,8.0,12.0")
    p.add_argument("--out-json", type=Path, required=True)
    args = p.parse_args()

    if not args.synthetic and not (args.manifest_csv and args.data_root):
        raise SystemExit("need --manifest-csv + --data-root, or --synthetic")

    sys.path.insert(0, str(args.submission_dir))
    from model import Model  # noqa: E402  (the submission's own wrapper, unmodified)

    model = Model()
    if not hasattr(model, "_dec") or not hasattr(model, "_blank_id"):
        raise SystemExit("this probe targets the parakeet_onnx wrapper (needs _dec/_blank_id)")
    rec = MarginRecorder(model._dec, model._dec_out_names, model._blank_id)
    model._dec = rec

    if args.synthetic:
        clips = [("synthetic", synthetic_audio())]
    else:
        rows = list(csv.DictReader(open(args.manifest_csv, newline="", encoding="utf-8")))
        clips = [
            (r["id"], read_wave(str(args.data_root / r["audio_filepath"])))
            for r in rows[: args.max_utts]
        ]

    hypos = {}
    spans = []  # (uid, start_step, end_step) so margins can be grouped per utterance
    for uid, samples in clips:
        model.reset()
        start_step = len(rec.margins)
        for start in range(0, len(samples), CHUNK_SIZE):
            chunk = samples[start : start + CHUNK_SIZE]
            if len(chunk) < CHUNK_SIZE:
                chunk = np.pad(chunk, (0, CHUNK_SIZE - len(chunk)))
            model.accept_chunk(chunk.astype(np.float32))
        hypos[uid] = model.input_finished()
        spans.append((uid, start_step, len(rec.margins)))

    m = np.asarray(rec.margins, dtype=np.float64)
    won = np.asarray(rec.blank_won, dtype=bool)
    if m.size == 0:
        raise SystemExit("no greedy steps recorded — the wrapper never called the decoder session")

    blank_margins = m[won]  # only blank-winning steps can be flipped by a penalty
    betas = [float(b) for b in args.betas.split(",")]
    qs = [1, 5, 10, 25, 50, 75, 90, 95, 99]

    print("\n=== blank-margin probe ===")
    print(f"utterances            : {len(clips)}")
    print(f"greedy steps          : {m.size}")
    print(f"blank won             : {int(won.sum())} ({100.0*won.mean():.2f}%)")
    print(f"empty hypotheses      : {sum(1 for v in hypos.values() if not v.strip())} / {len(hypos)}")
    print("\nmargin = blank_logit - best_non_blank_logit, over BLANK-WINNING steps:")
    for q in qs:
        print(f"  p{q:<3d}: {np.percentile(blank_margins, q):8.3f}")
    print("\nfirst-order flip fraction by beta (a blank flips when margin < beta):")
    flips = {}
    for b in betas:
        f = float((blank_margins < b).mean())
        flips[b] = f
        print(f"  beta={b:<6.2f} flips {100.0*f:6.2f}% of blank decisions "
              f"({int((blank_margins < b).sum())} steps)")
    # ---- THE DISCRIMINATING MEASUREMENT ----
    # A constant shift can only rescue the empties if their blank decisions sit at LOWER
    # margins than everyone else's. If the two distributions overlap, no single beta can
    # flip the empties without flipping the legitimate blanks in every other utterance,
    # and the lever is dead for the empty tail -- a mechanistically informative negative,
    # bought for minutes instead of a full grid.
    per_utt = []
    for uid, s, e in spans:
        seg = m[s:e]
        seg_won = won[s:e]
        bm = seg[seg_won]
        per_utt.append(
            {
                "id": uid,
                "n_steps": int(e - s),
                "empty": not hypos[uid].strip(),
                "blank_margin_p10": float(np.percentile(bm, 10)) if bm.size else None,
                "blank_margin_p50": float(np.percentile(bm, 50)) if bm.size else None,
                "blank_margin_min": float(bm.min()) if bm.size else None,
            }
        )
    emp = [r for r in per_utt if r["empty"] and r["blank_margin_p50"] is not None]
    non = [r for r in per_utt if not r["empty"] and r["blank_margin_p50"] is not None]
    separation = None
    if emp and non:
        e50 = float(np.median([r["blank_margin_p50"] for r in emp]))
        n50 = float(np.median([r["blank_margin_p50"] for r in non]))
        e10 = float(np.median([r["blank_margin_p10"] for r in emp]))
        n10 = float(np.median([r["blank_margin_p10"] for r in non]))
        separation = {"empty_p50": e50, "nonempty_p50": n50, "delta_p50": e50 - n50,
                      "empty_p10": e10, "nonempty_p10": n10, "delta_p10": e10 - n10}
        print(f"\nempty vs non-empty utterances ({len(emp)} vs {len(non)}), median of per-utterance margins:")
        print(f"  p50 margin: empty {e50:7.3f} · non-empty {n50:7.3f} · delta {e50-n50:+7.3f}")
        print(f"  p10 margin: empty {e10:7.3f} · non-empty {n10:7.3f} · delta {e10-n10:+7.3f}")
        if e50 <= n50:
            print("  -> empties are NOT more confidently blank. A modest beta reaches them first.")
        else:
            print("  -> empties are MORE confidently blank than ordinary blanks. Any beta that")
            print("     flips them also floods every other utterance with insertions. If the gap")
            print("     is large, a constant shift cannot separate the two populations: report")
            print("     this and STOP before spending the grid.")
    else:
        print("\n(no empty/non-empty split available on this input — need real audio for the "
              "discriminating measurement)")

    print("\nGrid advice: pick the sweep so the flip fraction spans roughly 0.5% to 20%.")
    print("A beta whose flip fraction is ~0 cannot change the transcript; one above ~30%")
    print("will bury the output in insertions. FIRST-ORDER ONLY -- flipping a blank changes")
    print("the prediction state and every later step. This sizes the grid; evaluate.sh decides.")

    out = {
        "submission_dir": str(args.submission_dir),
        "synthetic": bool(args.synthetic),
        "n_utts": len(clips),
        "n_steps": int(m.size),
        "n_blank_won": int(won.sum()),
        "blank_win_rate": float(won.mean()),
        "margin_percentiles_blank_steps": {f"p{q}": float(np.percentile(blank_margins, q)) for q in qs},
        "margin_percentiles_all_steps": {f"p{q}": float(np.percentile(m, q)) for q in qs},
        "flip_fraction_by_beta": flips,
        "per_utterance": per_utt,
        "empty_vs_nonempty": separation,
        "hypotheses": hypos,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWritten: {args.out_json}")


if __name__ == "__main__":
    main()
