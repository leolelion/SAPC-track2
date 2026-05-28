#!/usr/bin/env python3
"""Phase 4b — batch (back-to-back, no real-time pacing) decode through the
SAPC2 Model class. Used for Dev_10k accuracy validation only.

NOTE: This is the same Model class as Phase 3/4. Streaming behavior is
preserved (caches roll within each utterance, reset between utterances).
The only difference vs Phase 3 is we drop the AudioSender thread + sleeps
— we feed chunks directly into accept_chunk back-to-back. That mirrors
local_decode.py's run_batch_decode pass (the "accuracy" pass in SAPC2's
two-pass ingestion).

Heartbeat written every --heartbeat-interval utterances (default 100) so
unattended progress is visible from any session.

Outputs:
  <out_prefix>.csv               id,raw_hypos  — for sclite
  <out_prefix>.timing.json       per-utt compute_sec / audio_sec + aggregate
"""
import argparse
import csv
import json
import os
import sys
import time
import wave

import numpy as np

SAMPLE_RATE = 16000
CHUNK_SIZE = 1600


def read_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as f:
        assert f.getframerate() == SAMPLE_RATE
        assert f.getnchannels() == 1
        assert f.getsampwidth() == 2
        raw = f.readframes(f.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--audio-root", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--heartbeat", default=None, help="path for heartbeat file")
    ap.add_argument("--heartbeat-interval", type=int, default=100)
    args = ap.parse_args()

    sys.path.insert(0, args.submission_dir)
    from model import Model

    print(f"[batch] loading Model from {args.submission_dir} ...", flush=True)
    t_load = time.time()
    model = Model()
    model.set_partial_callback(lambda _t: None)
    print(f"[batch] Model load: {time.time()-t_load:.1f}s", flush=True)

    rows = []
    with open(args.manifest, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]
    print(f"[batch] {len(rows)} utterances", flush=True)

    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
    csv_path = args.out_prefix + ".csv"
    timing_path = args.out_prefix + ".timing.json"

    ids = []
    hyps = []
    per_utt = []
    total_compute = 0.0
    total_audio = 0.0
    sweep_start = time.time()

    # Write header now so the CSV exists even if we crash mid-run.
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "raw_hypos"])

    for i, row in enumerate(rows):
        uid = row["id"]
        audio_path = os.path.join(args.audio_root, row["audio_filepath"])
        samples = read_wav(audio_path)
        audio_dur = len(samples) / SAMPLE_RATE

        model.reset()
        t0 = time.perf_counter()
        for start in range(0, len(samples), args.chunk_size):
            model.accept_chunk(samples[start : start + args.chunk_size])
        final_text = model.input_finished() or ""
        compute_sec = time.perf_counter() - t0
        # Note: model.compute_time_sec is the same value, but it's
        # internally measured per-method-call; use our wall measurement
        # here for the whole utterance to keep it simple.

        ids.append(uid)
        hyps.append(final_text)
        per_utt.append(
            {
                "id": uid,
                "compute_sec": compute_sec,
                "audio_sec": audio_dur,
                "rtf": compute_sec / max(1e-6, audio_dur),
            }
        )
        total_compute += compute_sec
        total_audio += audio_dur

        # Append to CSV after every utt so partial runs survive crashes.
        with open(csv_path, "a", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow([uid, final_text])

        if args.heartbeat and ((i + 1) % args.heartbeat_interval == 0 or i + 1 == len(rows)):
            elapsed = time.time() - sweep_start
            running_rtf = total_compute / max(1e-6, total_audio)
            try:
                with open(args.heartbeat, "w") as hb:
                    hb.write(
                        f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} "
                        f"utt={i+1}/{len(rows)} "
                        f"running_rtf={running_rtf:.3f} "
                        f"elapsed={elapsed:.1f}s "
                        f"audio_done={total_audio:.0f}s\n"
                    )
            except Exception:
                pass
            print(
                f"  [{i+1}/{len(rows)}] elapsed={elapsed:.1f}s running_rtf={running_rtf:.3f}",
                flush=True,
            )

    sweep_wall = time.time() - sweep_start

    timing_summary = {
        "n_utt": len(rows),
        "total_compute_sec": total_compute,
        "total_audio_sec": total_audio,
        "aggregate_rtf": total_compute / max(1e-6, total_audio),
        "sweep_wall_sec": sweep_wall,
        "rtf_p50": float(np.percentile([u["rtf"] for u in per_utt], 50)) if per_utt else 0.0,
        "rtf_p90": float(np.percentile([u["rtf"] for u in per_utt], 90)) if per_utt else 0.0,
        "rtf_max": float(np.max([u["rtf"] for u in per_utt])) if per_utt else 0.0,
    }
    with open(timing_path, "w") as f:
        json.dump({"summary": timing_summary, "per_utt": per_utt}, f, indent=2)
    print(f"[batch] timing: {timing_path}", flush=True)
    print(f"[batch] csv:    {csv_path}", flush=True)
    print(json.dumps(timing_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
