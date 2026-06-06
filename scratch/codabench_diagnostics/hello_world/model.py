#!/usr/bin/env python3
"""Hello-world diagnostic Model for SAPC2 Track 2.

NOT a real ASR submission. Purpose: isolate the Codabench ingestion
contract from any actual model code. If this submission lands a
non-empty score (or even just reaches the scoring stage and reports
WER/CER on "hello world" hypotheses), the contract works and the bug
in the real submission is somewhere in our model / setup. If this
submission fails with the same "Detected splits: [] / No .predict.csv
files found" log, the bug is structural to the zip we ship.

Implements the 5-method streaming interface (see track2 README):
  __init__, set_partial_callback, reset, accept_chunk, input_finished

No torch, no onnxruntime, no NeMo, no numpy imports. Stdlib only.
accept_chunk is a no-op. input_finished returns "hello world".

Heavy logging in __init__ so we can see, from the ingestion log:
  * that Model.__init__ ran at all
  * what cwd / sys.executable / sys.path the ingestion uses
  * what files are visible in the submission dir at runtime
  * whether multiple processes (workers) are constructing Model
"""
import os
import sys
import time


def _log(tag: str, msg: str) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    print(f"[{tag}] {ts} pid={os.getpid()} ppid={os.getppid()} {msg}", flush=True)


class Model:
    def __init__(self):
        _log("HELLO_WORLD_DIAG", "Model.__init__ ENTER")
        try:
            _log("HELLO_WORLD_DIAG", f"cwd={os.getcwd()}")
        except Exception as e:
            _log("HELLO_WORLD_DIAG", f"getcwd_error={e!r}")
        _log("HELLO_WORLD_DIAG", f"sys.executable={sys.executable}")
        _log("HELLO_WORLD_DIAG", f"sys.version={sys.version.split()[0]}")
        _log("HELLO_WORLD_DIAG", f"sys.path[:5]={sys.path[:5]}")
        try:
            here = os.path.dirname(os.path.abspath(__file__))
            _log("HELLO_WORLD_DIAG", f"submission_dir={here}")
            _log("HELLO_WORLD_DIAG", f"listdir(submission_dir)={sorted(os.listdir(here))}")
        except Exception as e:
            _log("HELLO_WORLD_DIAG", f"listdir_error={e!r}")
        try:
            _log("HELLO_WORLD_DIAG", f"listdir(cwd)={sorted(os.listdir('.'))[:50]}")
        except Exception as e:
            _log("HELLO_WORLD_DIAG", f"listdir_cwd_error={e!r}")
        _log("HELLO_WORLD_DIAG", "Model.__init__ EXIT")

    def set_partial_callback(self, callback) -> None:
        _log("HELLO_WORLD_DIAG", "set_partial_callback called")
        self._cb = callback

    def reset(self) -> None:
        # No-op. Called once per audio file; kept silent to avoid log spam.
        pass

    def accept_chunk(self, audio_chunk) -> str:
        # No-op. Ignore chunk content entirely. Return empty partial.
        return ""

    def input_finished(self) -> str:
        # Constant final hypothesis. Lets the scorer compute CER/WER vs
        # ground truth without involving any ML inference.
        return "hello world"
