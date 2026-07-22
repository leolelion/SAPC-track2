#!/usr/bin/env python3
"""
Accuracy wrapper — CER / WER for the Parakeet benchmark.

CER/WER require sclite alignment + the two-reference SGML scoring that lives in
`evaluate.sh` -> `steps/eval/*` -> `utils/compute_metrics.py`. We do NOT
reimplement any of that. This module just shells out to the pristine
`evaluate.sh` (stages 0-2) and best-effort-parses the printed CER/WER.

PREREQ (one-time, per CLAUDE.md local-dev-loop): edit the DATA_ROOT / PROJ_ROOT
/ SCTK_DIR placeholders at the top of `evaluate.sh` before calling this. Those
are config, not scoring semantics — do not touch the scoring code itself.

CER is the primary metric. Per-utterance error is clipped at 1.0 by the scorer.
"""

import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EVALUATE_SH = _REPO_ROOT / "evaluate.sh"


def score_accuracy(
    hyp_csv: str,
    split: str = "Dev",
    start_stage: int = 0,
    stop_stage: int = 2,
    hyp_col: str = "raw_hypos",
) -> Dict[str, object]:
    """Run evaluate.sh stages `start_stage..stop_stage` on an id,raw_hypos CSV.

    Returns {"returncode", "stdout", "cer", "wer"} where cer/wer are best-effort
    floats parsed from stdout (None if the parse fails — read stdout directly).
    Raises FileNotFoundError if evaluate.sh is missing.
    """
    if not _EVALUATE_SH.exists():
        raise FileNotFoundError(f"evaluate.sh not found at {_EVALUATE_SH}")

    cmd = [
        "bash",
        str(_EVALUATE_SH),
        "--split",
        split,
        "--hyp-csv",
        str(hyp_csv),
        "--hyp-col",
        hyp_col,
        "--start_stage",
        str(start_stage),
        "--stop_stage",
        str(stop_stage),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    # Echo through so the human sees the authoritative scorer output.
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "cer": _grab(proc.stdout, "cer"),
        "wer": _grab(proc.stdout, "wer"),
    }


def _grab(text: str, metric: str) -> Optional[float]:
    """Best-effort: pull a percentage/float following the metric name in stdout.

    This is a convenience only. The authoritative value is whatever the scorer
    prints/writes; if this returns None, read the echoed stdout.
    """
    m = re.search(rf"{metric}\D*([0-9]+\.[0-9]+)", text, re.IGNORECASE)
    return float(m.group(1)) if m else None


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hyp-csv", required=True)
    ap.add_argument("--split", default="Dev")
    ap.add_argument("--start-stage", type=int, default=0)
    ap.add_argument("--stop-stage", type=int, default=2)
    args = ap.parse_args()
    res = score_accuracy(
        args.hyp_csv, args.split, args.start_stage, args.stop_stage
    )
    print(json.dumps({k: v for k, v in res.items() if k != "stdout"}, indent=2))
