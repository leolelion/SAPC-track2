#!/usr/bin/env python3
"""Unit-test the parakeet_realtime_ft intra-op thread policy WITHOUT NeMo/a pod.

Why this exists: the shipped default decides whether the accuracy pass fits the
15000 s/submission budget on the 24-vCPU eval worker (~20 worker processes, so threads
multiply by 20). exp_nemotron_speed_002 measured 4.2x wall-clock between threads=1 and
threads=4 under that topology. A silent regression here is a timeout, i.e. a zero.

Run: python3 scripts/test_thread_policy.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "track2_starting_kit" / "parakeet_realtime_ft"))
from model import _resolve_threads  # noqa: E402  (import-safe: torch/nemo are lazy)

CASES = [
    # (name,                              quota, nproc, env,                    expect)
    ("codabench worker, default",           24,    24,  {},                        1),
    ("codabench worker, env=2",             24,    24,  {"SAPC2_THREADS": "2"},    2),
    ("codabench worker, env=4",             24,    24,  {"SAPC2_THREADS": "4"},    4),
    ("env cannot exceed quota",             24,    24,  {"SAPC2_THREADS": "99"},  24),
    ("old pod: nproc 128, quota 13.6",     13.6,  128,  {"SAPC2_THREADS": "24"},  13),
    ("quota unlimited falls back to nproc", None,    8,  {"SAPC2_THREADS": "99"},   8),
    ("no quota, default still 1",          None,   24,  {},                        1),
    ("zero/garbage floors to 1",             24,   24,  {"SAPC2_THREADS": "0"},    1),
]

failures = 0
for name, quota, nproc, env, expect in CASES:
    got, want, ceiling = _resolve_threads(quota, nproc, env=env)
    ok = got == expect
    failures += (not ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name:38s} -> threads={got} "
          f"(requested={want} ceiling={ceiling}) expect={expect}")

print(f"\n{len(CASES) - failures}/{len(CASES)} passed")
sys.exit(1 if failures else 0)
