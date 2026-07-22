#!/usr/bin/env python3
"""
Benchmark driver — orchestrates the official SAPC2 scoring for candidate models.

It does NOT invent decoding or scoring. It calls:
  - offline path : benchmark/models/parakeet_offline.py (NeMo transcribe) -> id,raw_hypos CSV
                   then benchmark/metrics.py (evaluate.sh stages 0-2) -> CER/WER ceiling
  - streaming path: track2_starting_kit/local_decode.py (the REAL harness, both passes)
                   then metrics.py (CER/WER) + latency.py (TTFT/TTLT)

Two subcommands mirror the two questions in docs/benchmark_plan.md:
  offline    -> accuracy ceiling (Stage A). No latency (offline model).
  streaming  -> real streaming CER + TTFT/TTLT (Stage B). Needs a packaged
                submission dir with a 5-method model.py (e.g. the zipformer
                baseline, or a verified nemotron_streaming package).

Nothing here runs on the dev box (no NeMo, no data). This is scaffolding to be
executed on the pod/CPU-runtime once data + env exist.
"""

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmark import metrics, latency  # noqa: E402


def cmd_offline(args):
    """Accuracy ceiling for an offline Parakeet checkpoint (Stage A)."""
    from benchmark.models import parakeet_offline

    out_csv = parakeet_offline.run(
        checkpoint=args.checkpoint,
        manifest_csv=Path(args.manifest_csv),
        data_root=Path(args.data_root),
        out_csv=Path(args.out_csv),
        device=args.device,
        beam_size=args.beam_size,
        limit=args.limit,
    )
    print("\n=== Accuracy (evaluate.sh stages 0-2) ===")
    res = metrics.score_accuracy(str(out_csv), split=args.split)
    print(f"\nCER≈{res['cer']}  WER≈{res['wer']}  (authoritative: see stdout above)")


def cmd_streaming(args):
    """Real streaming CER + TTFT/TTLT for a packaged 5-method submission (Stage B).

    Shells out to the REAL local_decode.py so the ingestion path is identical to
    the organizers'. Never substitute a hand-rolled decode here.
    """
    import subprocess

    local_decode = _REPO_ROOT / "track2_starting_kit" / "local_decode.py"
    out_csv = Path(args.out_csv)
    out_partial = Path(args.out_partial_json)

    cmd = [
        sys.executable,
        str(local_decode),
        "--submission-dir",
        str(args.submission_dir),
        "--manifest-csv",
        str(args.manifest_csv),
        "--streaming-manifest-csv",
        str(args.streaming_manifest_csv),
        "--data-root",
        str(args.data_root),
        "--out-csv",
        str(out_csv),
        "--out-partial-json",
        str(out_partial),
    ]
    print("[run_eval] launching REAL local_decode.py …")
    subprocess.check_call(cmd)

    print("\n=== Accuracy (evaluate.sh stages 0-2) ===")
    acc = metrics.score_accuracy(str(out_csv), split=args.split)
    print("\n=== Latency (TTFT/TTLT) ===")
    lat = latency.summarize_latency(str(out_partial), str(args.streaming_manifest_csv))
    print(
        f"\nCER≈{acc['cer']}  WER≈{acc['wer']}  "
        f"TTFT_p50={latency.p50(lat, 'ttft')}s  TTLT_p50={latency.p50(lat, 'ttlt')}s"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)

    o = sub.add_parser("offline", help="Stage A: offline accuracy ceiling")
    o.add_argument("--checkpoint", required=True)
    o.add_argument("--manifest-csv", required=True)
    o.add_argument("--data-root", required=True)
    o.add_argument("--out-csv", required=True)
    o.add_argument("--split", default="Dev")
    o.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    o.add_argument("--beam-size", type=int, default=1)
    o.add_argument("--limit", type=int, default=0)
    o.set_defaults(func=cmd_offline)

    s = sub.add_parser("streaming", help="Stage B: real streaming CER + latency")
    s.add_argument("--submission-dir", required=True, help="dir with a 5-method model.py")
    s.add_argument("--manifest-csv", required=True)
    s.add_argument("--streaming-manifest-csv", required=True)
    s.add_argument("--data-root", required=True)
    s.add_argument("--out-csv", default="./bench.predict.csv")
    s.add_argument("--out-partial-json", default="./bench.partial.json")
    s.add_argument("--split", default="Dev")
    s.set_defaults(func=cmd_streaming)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
