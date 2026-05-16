#!/usr/bin/env python3
"""
Phase 2b/2c: Full Dev_streaming validation through SAPC2 local_decode.py harness.

Runs the two-pass SAPC2 evaluation (batch accuracy + streaming latency),
then computes CER, WER, TTFT, TTLT, RTF and prints TTFT distribution.

Usage:
  # Phase 2b: 4 threads (default)
  python3 validate_full.py --threads 4

  # Phase 2c: threading sensitivity
  python3 validate_full.py --threads 1
  python3 validate_full.py --threads 2
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_ROOT = "/workspace/SAPC2"
MANIFEST_CSV = f"{DATA_ROOT}/manifest/Dev_streaming.csv"
STREAMING_MANIFEST_CSV = f"{DATA_ROOT}/manifest/Dev_streaming.csv"
SUBMISSION_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_RATE = 16000


def edit_distance(ref_tokens, hyp_tokens):
    """Levenshtein edit distance between two token lists."""
    n, m = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[n][m]


def compute_error_rate(refs, hyps, unit="word"):
    """Compute WER or CER across a list of (ref, hyp) pairs."""
    total_tokens = 0
    total_errors = 0
    for ref_text, hyp_text in zip(refs, hyps):
        ref_norm = ref_text.lower().strip()
        hyp_norm = hyp_text.lower().strip()
        if unit == "char":
            ref_tokens = list(ref_norm.replace(" ", ""))
            hyp_tokens = list(hyp_norm.replace(" ", ""))
        else:
            ref_tokens = ref_norm.split()
            hyp_tokens = hyp_norm.split()
        total_tokens += len(ref_tokens)
        total_errors += edit_distance(ref_tokens, hyp_tokens)
    return (total_errors / total_tokens * 100) if total_tokens > 0 else 0.0


def compute_latency_metrics(partial_results):
    """Compute TTFT, TTLT, RTF from partial_results dict."""
    ttfts = []
    ttlts = []
    audio_durs = []
    wall_times = []
    utt_details = []

    for uid, record in partial_results.items():
        timing = record.get("timing", {})
        events = record.get("events", [])

        audio_send_start = timing.get("audio_send_start_time")
        audio_end_oracle = timing.get("audio_end_oracle_time")
        final_visible = timing.get("final_visible_time")
        first_partial = timing.get("first_partial_time")

        if audio_send_start is None or audio_end_oracle is None:
            continue

        audio_dur = audio_end_oracle - audio_send_start
        audio_durs.append(audio_dur)

        # TTFT: time from audio start to first non-empty partial callback
        ttft = None
        if first_partial is not None:
            ttft = first_partial - audio_send_start
            ttfts.append(ttft)

        # TTLT: time from audio end to final visible result
        ttlt = None
        if final_visible is not None:
            ttlt = final_visible - audio_end_oracle
            ttlts.append(ttlt)

        # Wall time for this utterance
        if final_visible is not None:
            wall_times.append(final_visible - audio_send_start)

        utt_details.append({
            "uid": uid,
            "audio_dur": audio_dur,
            "ttft": ttft,
            "ttlt": ttlt,
        })

    total_audio = sum(audio_durs)
    total_wall = sum(wall_times) if wall_times else 0

    metrics = {
        "n_utterances": len(partial_results),
        "total_audio_sec": total_audio,
        "total_wall_sec": total_wall,
        "rtf": total_wall / total_audio if total_audio > 0 else float("nan"),
    }

    if ttfts:
        ttfts_ms = [t * 1000 for t in ttfts]
        metrics["ttft_mean_ms"] = np.mean(ttfts_ms)
        metrics["ttft_p50_ms"] = np.percentile(ttfts_ms, 50)
        metrics["ttft_p75_ms"] = np.percentile(ttfts_ms, 75)
        metrics["ttft_p90_ms"] = np.percentile(ttfts_ms, 90)
        metrics["ttft_p95_ms"] = np.percentile(ttfts_ms, 95)
        metrics["ttft_min_ms"] = np.min(ttfts_ms)
        metrics["ttft_max_ms"] = np.max(ttfts_ms)

    if ttlts:
        ttlts_ms = [t * 1000 for t in ttlts]
        metrics["ttlt_mean_ms"] = np.mean(ttlts_ms)
        metrics["ttlt_p50_ms"] = np.percentile(ttlts_ms, 50)
        metrics["ttlt_p75_ms"] = np.percentile(ttlts_ms, 75)
        metrics["ttlt_p90_ms"] = np.percentile(ttlts_ms, 90)
        metrics["ttlt_p95_ms"] = np.percentile(ttlts_ms, 95)

    return metrics, utt_details


def print_ttft_distribution(utt_details):
    """Print TTFT histogram and worst-5 utterances."""
    ttft_data = [(d["uid"], d["ttft"] * 1000, d["audio_dur"])
                 for d in utt_details if d["ttft"] is not None]
    if not ttft_data:
        print("  No TTFT data available.")
        return

    ttft_values = [t[1] for t in ttft_data]

    # Histogram bins
    bins = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000, float("inf")]
    labels = ["0-500", "500-1000", "1000-1500", "1500-2000", "2000-2500",
              "2500-3000", "3000-3500", "3500-4000", "4000-5000", "5000+"]
    counts = [0] * len(labels)
    for v in ttft_values:
        for i in range(len(bins) - 1):
            if bins[i] <= v < bins[i + 1]:
                counts[i] += 1
                break

    print("\n  TTFT Distribution (ms):")
    print("  " + "-" * 50)
    max_count = max(counts) if counts else 1
    for label, count in zip(labels, counts):
        bar = "#" * int(count / max_count * 30) if max_count > 0 else ""
        print(f"  {label:>10s} | {bar:<30s} {count:>3d}")
    print("  " + "-" * 50)

    # Worst 5
    ttft_data.sort(key=lambda x: -x[1])
    print("\n  5 Worst TTFT utterances:")
    print(f"  {'ID':<55s} {'TTFT_ms':>8s} {'AudioDur':>8s}")
    for uid, ttft_ms, audio_dur in ttft_data[:5]:
        print(f"  {uid:<55s} {ttft_ms:>8.0f} {audio_dur:>8.2f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--manifest", default=MANIFEST_CSV)
    parser.add_argument("--streaming-manifest", default=STREAMING_MANIFEST_CSV)
    parser.add_argument("--submission-dir", default=SUBMISSION_DIR)
    args = parser.parse_args()

    tag = f"threads{args.threads}"
    out_csv = Path(f"/workspace/nemotron_streaming/results/{tag}_predict.csv")
    out_json = Path(f"/workspace/nemotron_streaming/results/{tag}_partial_results.json")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Phase 2b/2c: Full Dev_streaming validation ({args.threads} threads)")
    print("=" * 70)

    # Set thread count via environment variable (read by model.py)
    os.environ["SAPC2_THREADS"] = str(args.threads)

    # Run local_decode.py (the SAPC2 two-pass harness)
    decode_script = os.path.join(args.submission_dir, "local_decode.py")
    cmd = [
        sys.executable, decode_script,
        "--submission-dir", args.submission_dir,
        "--manifest-csv", args.manifest,
        "--streaming-manifest-csv", args.streaming_manifest,
        "--data-root", args.data_root,
        "--out-csv", str(out_csv),
        "--out-partial-json", str(out_json),
        "--chunk-size", "1600",
        "--streaming-interval", "0.1",
    ]
    print(f"\nRunning: {' '.join(cmd)}\n")
    t0 = time.time()
    subprocess.check_call(cmd)
    total_time = time.time() - t0
    print(f"\nHarness completed in {total_time:.1f}s")

    # ---------------------------------------------------------------------------
    # Score accuracy (CER + WER)
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Accuracy Scoring")
    print("=" * 70)

    # Load references from manifest
    refs_by_id = {}
    with open(args.manifest, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            refs_by_id[row["id"]] = row["norm_text_without_disfluency"]

    # Load predictions
    preds_by_id = {}
    with open(out_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            preds_by_id[row["id"]] = row["raw_hypos"]

    # Match
    common_ids = sorted(set(refs_by_id.keys()) & set(preds_by_id.keys()))
    refs = [refs_by_id[uid] for uid in common_ids]
    hyps = [preds_by_id[uid] for uid in common_ids]

    cer = compute_error_rate(refs, hyps, unit="char")
    wer = compute_error_rate(refs, hyps, unit="word")

    print(f"  Utterances scored: {len(common_ids)}")
    print(f"  CER: {cer:.2f}%")
    print(f"  WER: {wer:.2f}%")

    # ---------------------------------------------------------------------------
    # Score latency (TTFT, TTLT, RTF)
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Latency Scoring")
    print("=" * 70)

    with open(out_json, "r", encoding="utf-8") as f:
        partial_results = json.load(f)

    metrics, utt_details = compute_latency_metrics(partial_results)

    print(f"  Utterances: {metrics['n_utterances']}")
    print(f"  Total audio: {metrics['total_audio_sec']:.1f}s")
    print(f"  Total wall:  {metrics['total_wall_sec']:.1f}s")
    print(f"  RTF: {metrics['rtf']:.3f}")
    print()
    if "ttft_p50_ms" in metrics:
        print(f"  TTFT  mean: {metrics['ttft_mean_ms']:.0f} ms")
        print(f"  TTFT   P50: {metrics['ttft_p50_ms']:.0f} ms")
        print(f"  TTFT   P75: {metrics['ttft_p75_ms']:.0f} ms")
        print(f"  TTFT   P90: {metrics['ttft_p90_ms']:.0f} ms")
        print(f"  TTFT   P95: {metrics['ttft_p95_ms']:.0f} ms")
        print(f"  TTFT   min: {metrics['ttft_min_ms']:.0f} ms")
        print(f"  TTFT   max: {metrics['ttft_max_ms']:.0f} ms")
    print()
    if "ttlt_p50_ms" in metrics:
        print(f"  TTLT  mean: {metrics['ttlt_mean_ms']:.0f} ms")
        print(f"  TTLT   P50: {metrics['ttlt_p50_ms']:.0f} ms")
        print(f"  TTLT   P75: {metrics['ttlt_p75_ms']:.0f} ms")
        print(f"  TTLT   P90: {metrics['ttlt_p90_ms']:.0f} ms")
        print(f"  TTLT   P95: {metrics['ttlt_p95_ms']:.0f} ms")

    # TTFT distribution
    print_ttft_distribution(utt_details)

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"SUMMARY ({args.threads} threads)")
    print("=" * 70)
    print(f"  {'Metric':<15s} {'Measured':>10s} {'Expected':>10s} {'Tolerance':>10s} {'Status':>8s}")
    print("  " + "-" * 58)

    checks = [
        ("CER", cer, 20.12, 0.5),
        ("WER", wer, 26.96, 1.0),
    ]
    if "ttft_p50_ms" in metrics:
        checks.append(("TTFT P50 (ms)", metrics["ttft_p50_ms"], 1376, 200))
    if "ttlt_p50_ms" in metrics:
        checks.append(("TTLT P50 (ms)", metrics["ttlt_p50_ms"], 114, 50))
    checks.append(("RTF", metrics["rtf"], 1.031, 0.05))

    all_pass = True
    for name, measured, expected, tol in checks:
        if name in ("CER", "WER", "RTF"):
            ok = abs(measured - expected) <= tol
        else:
            ok = abs(measured - expected) <= tol
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  {name:<15s} {measured:>10.2f} {expected:>10.2f} {'±' + str(tol):>10s} {status:>8s}")

    print()
    if all_pass:
        print("  >>> ALL CHECKS PASSED <<<")
    else:
        print("  >>> SOME CHECKS FAILED — review before proceeding <<<")

    # Save metrics to JSON for easy comparison
    result = {
        "threads": args.threads,
        "cer": cer,
        "wer": wer,
        "rtf": metrics["rtf"],
        "ttft_p50_ms": metrics.get("ttft_p50_ms"),
        "ttlt_p50_ms": metrics.get("ttlt_p50_ms"),
        "ttft_p90_ms": metrics.get("ttft_p90_ms"),
        "ttlt_p90_ms": metrics.get("ttlt_p90_ms"),
        "total_time_sec": total_time,
    }
    metrics_file = out_csv.parent / f"{tag}_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Metrics saved to: {metrics_file}")


if __name__ == "__main__":
    main()
