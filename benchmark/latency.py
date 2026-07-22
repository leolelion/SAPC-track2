#!/usr/bin/env python3
"""
Latency wrapper — TTFT / TTLT for the Parakeet benchmark.

This does NOT reimplement latency. It imports and calls the organizers'
`utils/compute_latency.py` as-is, so our numbers are byte-identical to the
official scorer (house rule: wrap, don't modify; see memory
`validate-against-real-harness`).

  TTFT = first_non_empty_partial - (audio_send_start + mfa_speech_start)
  TTLT = final_visible - audio_end_oracle
Reported as p50 / p90 / p95. Pareto rank uses mean(TTFT, TTLT).
"""

import sys
from pathlib import Path
from typing import Dict, Optional

# Put repo root on the path so `utils` (a namespace package) is importable
# regardless of where this is invoked from.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.compute_latency import compute_latency_from_partial_json  # noqa: E402


def summarize_latency(
    partial_json: str,
    streaming_manifest_csv: str,
    mfa_col: str = "mfa_speech_start",
) -> Dict[str, object]:
    """Compute the official TTFT/TTLT summary from a partial_results JSON.

    Args:
        partial_json: path to `<split>.partial_results.json` produced by the
            streaming pass of `track2_starting_kit/local_decode.py`.
        streaming_manifest_csv: the `*_streaming.csv` manifest carrying the
            `mfa_speech_start` column (TTFT is meaningless without it).
        mfa_col: manifest column name for the forced-aligned speech onset.

    Returns:
        The dict returned by the official scorer, with `ttft_sec`/`ttlt_sec`
        sub-dicts each holding count / median / p50 / p90 / p95.
    """
    return compute_latency_from_partial_json(
        str(partial_json),
        manifest_csv=str(streaming_manifest_csv),
        mfa_col=mfa_col,
    )


def p50(summary: Dict[str, object], which: str) -> Optional[float]:
    """Pull a p50 value out of a summarize_latency() result. which ∈ {ttft, ttlt}."""
    key = "ttft_sec" if which == "ttft" else "ttlt_sec"
    sub = summary.get(key)
    if isinstance(sub, dict):
        return sub.get("p50")
    return None


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--partial-json", required=True)
    ap.add_argument("--streaming-manifest-csv", required=True)
    ap.add_argument("--mfa-col", default="mfa_speech_start")
    args = ap.parse_args()
    print(
        json.dumps(
            summarize_latency(
                args.partial_json, args.streaming_manifest_csv, args.mfa_col
            ),
            indent=2,
        )
    )
