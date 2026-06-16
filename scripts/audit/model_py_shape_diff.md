# Step 5 — model.py shape comparison (interface, not logic)

baseline `model.py` (16114 B) — icefall streaming zipformer
v7       `model.py` (25132 B) — Nemotron ONNX, nemo-free

## Required `Model` interface (per spec) — exact signature match

| Method | baseline | v7 | Match |
|---|---|---|---|
| class name | `class Model:` | `class Model:` | ✅ exactly `Model`, base `object` |
| `__init__` | `def __init__(self):` | `def __init__(self):` | ✅ |
| `set_partial_callback` | `def set_partial_callback(self, callback) -> None:` | `def set_partial_callback(self, callback) -> None:` | ✅ identical |
| `reset` | `def reset(self) -> None:` | `def reset(self) -> None:` | ✅ identical |
| `accept_chunk` | `def accept_chunk(self, audio_chunk: np.ndarray) -> str:` | `def accept_chunk(self, audio_chunk: np.ndarray) -> str:` | ✅ identical, returns `str` |
| `input_finished` | `def input_finished(self) -> str:` | `def input_finished(self) -> str:` | ✅ identical, returns `str` |

Method definition order of the 5 required methods is the same in both
(`__init__ → set_partial_callback → reset → accept_chunk → input_finished`); v7
interleaves private helpers (`_ensure_loaded`, `_peak_rss_mb`, `_log_mem_once`)
between them, which is invisible to the caller.

Return types match the spec and each other: `accept_chunk` and `input_finished`
both return `str` in both files.

## Module-level structure

| Aspect | baseline | v7 |
|---|---|---|
| Top imports | argparse, os, sys, pathlib, typing, omegaconf, numpy, torch, **sentencepiece**, **kaldifeat** | os, numpy, torch, **onnxruntime**, omegaconf |
| `sys.path` manipulation at import | **YES** — inserts `icefall` + `egs/.../zipformer` (lines 63-64) | none |
| Heavy 2nd-stage imports at module load | **YES** — `from icefall…`, `from decode_stream…`, `from streaming_decode…`, `from train…` (lines 69-73). Requires setup.sh to have installed icefall first. | none — only stdlib + numpy/torch/onnxruntime/omegaconf |
| Module-level constants | `_DIR/_ICEFALL/_ZIPFORMER/_WEIGHTS` paths | `_DIR` + ~15 numeric constants (N_MELS, CHUNK_NEW, VOCAB_SIZE, …) |
| Module-level `print` | none | none |
| `if __name__ == "__main__"` | none | none |
| threading / multiprocessing | none | none |

Notable: **baseline does more at import time** (sys.path surgery + importing four
icefall submodules). v7's module-load is lighter and defers heavy work to
`_ensure_loaded()`. If import-time fragility were the gate, the baseline would be
*more* exposed, not less.

## Side effects

| Side effect | baseline | v7 |
|---|---|---|
| Reads files | weights via icefall checkpoint loader (in `__init__`) | `tokens.txt` (`open`, line 75); ONNX sessions in `_ensure_loaded` |
| Writes files | none | **`/tmp/cpu_diagnostic.json`** (line 448, inside `_log_diagnostic_info`) |
| Reads `/proc/*` | none | `/proc/cpuinfo`, `/proc/uptime`, `/proc/loadavg` (lines 420-444, diagnostic) |
| `os.environ` | none | none at module level |
| Paths | relative to `_DIR` (abspath of file) | relative to `_DIR` (abspath of file) |
| Multiprocessing/threading at module level | none | none |

The `/proc` reads + `/tmp/cpu_diagnostic.json` write live inside
`_log_diagnostic_info()` (a diagnostic helper invoked at runtime, **not** at import
and **not** at module top-level). They run during ingestion, not before it. They
are Linux-only but guarded by the call site; they are not a pre-ingestion gate.

## Verdict (Step 5)
The **caller-visible interface is byte-for-byte equivalent** to the baseline:
same class name, same five method signatures, same return types, same method
order, no module-level print/threading/`__main__`. Differences are entirely in
*implementation* (icefall vs ONNX) and in v7's extra runtime diagnostics — none of
which changes the import-time/launch-time contract Codabench checks. If anything,
v7 imports *less* at module load than the working baseline. No pre-ingestion gate
found in model.py.
