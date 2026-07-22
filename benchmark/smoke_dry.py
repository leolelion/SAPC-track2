#!/usr/bin/env python3
"""
Dry smoke — verify the benchmark scaffolding is import/compile-clean on a box
WITHOUT NeMo, GPU, or data. It does NOT measure accuracy/latency (that needs the
pod). It checks:
  1. every benchmark file byte-compiles,
  2. the pure wrappers (metrics, latency) import and expose their API,
  3. the model modules import WITHOUT triggering NeMo (lazy-import discipline),
  4. the streaming skeleton refuses to fake-stream (raises, never fabricates).

Exit 0 = scaffolding is structurally sound; real validation happens on-pod.
"""

import importlib
import py_compile
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

FILES = [
    "benchmark/__init__.py",
    "benchmark/metrics.py",
    "benchmark/latency.py",
    "benchmark/run_eval.py",
    "benchmark/models/__init__.py",
    "benchmark/models/parakeet_offline.py",
    "benchmark/models/nemotron_streaming.py",
]


def check_compile():
    for rel in FILES:
        py_compile.compile(str(_REPO_ROOT / rel), doraise=True)
    print(f"[1] py_compile OK ({len(FILES)} files)")


def check_wrappers():
    latency = importlib.import_module("benchmark.latency")
    metrics = importlib.import_module("benchmark.metrics")
    assert hasattr(latency, "summarize_latency") and hasattr(latency, "p50")
    assert hasattr(metrics, "score_accuracy")
    # latency wraps the REAL scorer:
    from utils.compute_latency import compute_latency_from_partial_json  # noqa: F401
    print("[2] wrappers import + delegate to official scorers OK")


def check_models_lazy():
    # Importing the modules must NOT require NeMo (imports are lazy in __init__).
    importlib.import_module("benchmark.models.parakeet_offline")
    ns = importlib.import_module("benchmark.models.nemotron_streaming")
    assert hasattr(ns, "Model")
    print("[3] model modules import without NeMo (lazy-import discipline) OK")


def check_no_fake_streaming():
    ns = importlib.import_module("benchmark.models.nemotron_streaming")
    # Call the streaming core directly, bypassing __init__ (which needs NeMo),
    # to prove it raises instead of fabricating a partial.
    inst = ns.Model.__new__(ns.Model)
    try:
        inst._stream_step(b"", is_last=False)  # type: ignore[arg-type]
    except NotImplementedError:
        print("[4] streaming skeleton refuses to fake-stream (raises) OK")
        return
    raise AssertionError("streaming skeleton did NOT raise — fake-streaming risk!")


if __name__ == "__main__":
    check_compile()
    check_wrappers()
    check_models_lazy()
    check_no_fake_streaming()
    print("\nDRY SMOKE PASSED — scaffolding is structurally sound.")
    print("Real accuracy/latency validation happens on the pod (docs/benchmark_plan.md).")
