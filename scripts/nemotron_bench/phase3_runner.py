#!/usr/bin/env python3
"""Phase 3 — drive the SAPC2 Model class against Dev_streaming using the
*exact* SAPC2 ingestion design (two threads, 100 ms real-time pacing,
partial callback collection), and record:

  - predictions CSV (for sclite)
  - partial_results JSON (for utils/compute_latency.py)
  - per-utterance compute_time_sec (for compute_RTF gating)
  - per-etiology breakdown (CER/WER from sclite output)

Usage:
  /root/venvs/nemotron_bench/bin/python phase3_runner.py \\
      --submission-dir /path/to/nemotron_streaming \\
      --manifest /workspace/SAPC2/manifest/Dev_streaming.csv \\
      --audio-root /workspace/SAPC2 \\
      --out-prefix /workspace/phase3/dev_streaming
"""
import argparse
import csv
import json
import os
import queue
import sys
import threading
import time
import wave
from pathlib import Path

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


def percentile(values, p):
    return float(np.percentile(values, p)) if values else 0.0


def run_one_utt(model, samples, chunk_size, streaming_interval):
    """SAPC2 two-thread streaming pass. Returns (final, events, timing, compute_sec)."""
    events = []
    audio_dur = len(samples) / SAMPLE_RATE
    timing = {}

    def on_partial(text):
        ts = time.time()
        events.append({"event": "partial_callback", "time": ts, "text": text})
        if "first_partial_time" not in timing:
            timing["first_partial_time"] = ts

    model.set_partial_callback(on_partial)
    model.reset()  # also resets m.compute_time_sec

    audio_q: queue.Queue = queue.Queue()
    sender_done = threading.Event()
    final_holder = [""]
    finalized = [False]

    def sender():
        first_send = None
        for start in range(0, len(samples), chunk_size):
            now = time.time()
            if first_send is None:
                first_send = now
            audio_q.put(samples[start:start + chunk_size])
            if streaming_interval > 0:
                time.sleep(streaming_interval)
        audio_q.put(None)
        if first_send is not None:
            timing["audio_send_start_time"] = first_send
            timing["audio_end_oracle_time"] = first_send + audio_dur
        sender_done.set()

    def decoder():
        while True:
            try:
                chunk = audio_q.get(timeout=0.1)
            except queue.Empty:
                if sender_done.is_set():
                    if not finalized[0]:
                        final_holder[0] = model.input_finished() or ""
                        finalized[0] = True
                    break
                continue
            if chunk is None:
                if not finalized[0]:
                    final_holder[0] = model.input_finished() or ""
                    finalized[0] = True
                break
            model.accept_chunk(chunk)

    t_send = threading.Thread(target=sender, name="AudioSender")
    t_dec = threading.Thread(target=decoder, name="Decoder")
    t_send.start()
    t_dec.start()
    t_send.join()
    t_dec.join()

    final_visible = time.time()
    timing["final_visible_time"] = final_visible
    events.append({"event": "final_visible", "time": final_visible, "text": final_holder[0]})

    return final_holder[0], events, timing, float(model.compute_time_sec), audio_dur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--audio-root", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    ap.add_argument("--streaming-interval", type=float, default=0.1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, args.submission_dir)
    from model import Model  # noqa: E402

    print(f"[phase3] loading Model from {args.submission_dir} ...", flush=True)
    model = Model()
    print(f"[phase3] manifest: {args.manifest}", flush=True)

    rows = []
    with open(args.manifest, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[:args.limit]
    print(f"[phase3] {len(rows)} utterances", flush=True)

    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)

    partial_results = {}
    ids = []
    hyps = []
    per_utt_rtf = []
    per_utt_etiology = []

    total_audio_sec = 0.0
    total_compute_sec = 0.0
    sweep_start = time.time()

    for i, row in enumerate(rows):
        uid = row["id"]
        audio_path = os.path.join(args.audio_root, row["audio_filepath"])
        samples = read_wav(audio_path)

        final_text, events, timing, compute_sec, audio_dur = run_one_utt(
            model, samples, args.chunk_size, args.streaming_interval
        )

        partial_results[uid] = {"events": events, "timing": timing}
        ids.append(uid)
        hyps.append(final_text)
        rtf = compute_sec / max(1e-6, audio_dur)
        per_utt_rtf.append({"id": uid, "compute_sec": compute_sec, "audio_sec": audio_dur, "rtf": rtf})
        per_utt_etiology.append({"id": uid, "etiology": row.get("etiology", "")})

        total_audio_sec += audio_dur
        total_compute_sec += compute_sec
        if not args.quiet and ((i + 1) % 10 == 0 or i + 1 == len(rows)):
            running_rtf = total_compute_sec / max(1e-6, total_audio_sec)
            elapsed = time.time() - sweep_start
            print(
                f"  [{i+1}/{len(rows)}] elapsed={elapsed:.1f}s  "
                f"running compute-RTF={running_rtf:.3f}",
                flush=True,
            )

    sweep_wall = time.time() - sweep_start

    # ── Write outputs ───────────────────────────────────────────────
    csv_path = args.out_prefix + ".csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "raw_hypos"])
        for uid, h in zip(ids, hyps):
            w.writerow([uid, h])
    print(f"[phase3] CSV: {csv_path}", flush=True)

    partial_path = args.out_prefix + ".partial_results.json"
    with open(partial_path, "w", encoding="utf-8") as f:
        json.dump(partial_results, f)
    print(f"[phase3] partial: {partial_path}", flush=True)

    # RTF stats per utt + aggregate
    rtfs = [r["rtf"] for r in per_utt_rtf]
    rtf_summary = {
        "n_utt": len(rtfs),
        "compute_rtf_p50": percentile(rtfs, 50),
        "compute_rtf_p90": percentile(rtfs, 90),
        "compute_rtf_p95": percentile(rtfs, 95),
        "compute_rtf_max": float(max(rtfs) if rtfs else 0.0),
        "compute_rtf_aggregate": total_compute_sec / max(1e-6, total_audio_sec),
        "total_compute_sec": total_compute_sec,
        "total_audio_sec": total_audio_sec,
        "sweep_wall_sec": sweep_wall,
    }
    rtf_path = args.out_prefix + ".compute_rtf.json"
    with open(rtf_path, "w", encoding="utf-8") as f:
        json.dump({"summary": rtf_summary, "per_utt": per_utt_rtf}, f, indent=2)
    print(f"[phase3] compute-RTF: {rtf_path}", flush=True)
    print(json.dumps(rtf_summary, indent=2), flush=True)

    # Etiology mapping for downstream breakdown
    et_path = args.out_prefix + ".etiology.json"
    with open(et_path, "w", encoding="utf-8") as f:
        json.dump(per_utt_etiology, f, indent=2)
    print(f"[phase3] etiology: {et_path}", flush=True)


if __name__ == "__main__":
    main()
