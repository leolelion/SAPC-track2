#!/usr/bin/env python3
"""Analyze TTFT/TTLT distribution from partial_results.json."""
import json
import csv
import sys
import numpy as np

partial_json = sys.argv[1]  # e.g. results/threads4_partial_results.json
manifest_csv = sys.argv[2]  # e.g. /workspace/SAPC2/manifest/Dev_streaming.csv

with open(partial_json) as f:
    pr = json.load(f)

durations = {}
with open(manifest_csv) as f:
    for row in csv.DictReader(f):
        durations[row["id"]] = float(row["duration"])

records = []
for uid, record in pr.items():
    timing = record.get("timing", {})
    audio_start = timing.get("audio_send_start_time")
    audio_end = timing.get("audio_end_oracle_time")
    final_vis = timing.get("final_visible_time")
    first_partial = timing.get("first_partial_time")
    if audio_start is None:
        continue
    audio_dur = audio_end - audio_start if audio_end else durations.get(uid, 0)
    ttft = (first_partial - audio_start) * 1000 if first_partial else None
    ttlt = (final_vis - audio_end) * 1000 if (final_vis and audio_end) else None
    n_cb = len([e for e in record.get("events", []) if e["event"] == "partial_callback"])
    records.append(dict(uid=uid, audio_dur=audio_dur, ttft_ms=ttft, ttlt_ms=ttlt, n_callbacks=n_cb))

ttfts = [r["ttft_ms"] for r in records if r["ttft_ms"] is not None]
ttlts = [r["ttlt_ms"] for r in records if r["ttlt_ms"] is not None]

print("=" * 70)
print("LATENCY METRICS")
print("=" * 70)

if ttfts:
    print("\n  TTFT (ms):")
    for lbl, val in [("mean", np.mean(ttfts)), ("P50", np.percentile(ttfts, 50)),
                     ("P75", np.percentile(ttfts, 75)), ("P90", np.percentile(ttfts, 90)),
                     ("P95", np.percentile(ttfts, 95)), ("min", np.min(ttfts)),
                     ("max", np.max(ttfts))]:
        print(f"    {lbl:>5s}: {val:.0f}")

if ttlts:
    print("\n  TTLT (ms):")
    for lbl, val in [("mean", np.mean(ttlts)), ("P50", np.percentile(ttlts, 50)),
                     ("P75", np.percentile(ttlts, 75)), ("P90", np.percentile(ttlts, 90)),
                     ("P95", np.percentile(ttlts, 95))]:
        print(f"    {lbl:>5s}: {val:.0f}")

# Histogram
bins = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000, float("inf")]
labels = ["0-500", "500-1K", "1K-1.5K", "1.5K-2K", "2K-2.5K",
          "2.5K-3K", "3K-3.5K", "3.5K-4K", "4K-5K", "5K+"]
counts = [0] * len(labels)
for v in ttfts:
    for i in range(len(bins) - 1):
        if bins[i] <= v < bins[i + 1]:
            counts[i] += 1
            break

print("\n  TTFT Distribution:")
mx = max(counts) if counts else 1
for label, count in zip(labels, counts):
    bar = "#" * int(count / mx * 30) if mx > 0 else ""
    print(f"  {label:>10s} | {bar:<30s} {count:>3d}")

# Worst 5
ttft_recs = [(r["uid"], r["ttft_ms"], r["audio_dur"]) for r in records if r["ttft_ms"] is not None]
ttft_recs.sort(key=lambda x: -x[1])
print("\n  5 Worst TTFT:")
for uid, t, d in ttft_recs[:5]:
    print(f"    {uid[:52]:<52s} {t:>6.0f}ms  {d:>5.1f}s")

print("\n  5 Best TTFT:")
ttft_recs.sort(key=lambda x: x[1])
for uid, t, d in ttft_recs[:5]:
    print(f"    {uid[:52]:<52s} {t:>6.0f}ms  {d:>5.1f}s")

# Correlation
durs_arr = [r["audio_dur"] for r in records if r["ttft_ms"] is not None]
corr = np.corrcoef(durs_arr, ttfts)[0, 1]
print(f"\n  TTFT vs Duration correlation: {corr:.3f}")

high_ttft = sum(1 for t in ttfts if t > 3500)
no_partial = sum(1 for r in records if r["ttft_ms"] is None)
print(f"  Utterances with TTFT > 3500ms: {high_ttft}/{len(ttfts)}")
print(f"  Utterances with no partials:   {no_partial}/{len(records)}")
