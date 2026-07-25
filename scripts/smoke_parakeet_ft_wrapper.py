#!/usr/bin/env python3
"""Local contract smoke for track2_starting_kit/parakeet_realtime_ft/model.py.

Runs with torch+numpy present but **NeMo absent**: it bypasses __init__ (which loads
NeMo) via object.__new__, injects a mock cache-aware model, and exercises the parts of
the 5-method contract that are OURS (not NeMo's): buffering cadence, callback firing
(TTFT driver), reset, and <EOU>/special-token stripping. It makes NO accuracy claim —
the NeMo streaming call itself is VERIFY-ON-POD. See model.py VERIFICATION STATUS.

    python3 scripts/smoke_parakeet_ft_wrapper.py     # exits 0 on pass, 1 on fail
"""
import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL_PY = ROOT / "track2_starting_kit" / "parakeet_realtime_ft" / "model.py"

failures = []


def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        failures.append(name)


# ---- 1. import the module with NeMo absent (lazy-import contract) -------------
assert "nemo" not in sys.modules, "test must run without NeMo pre-imported"
spec = importlib.util.spec_from_file_location("parakeet_ft_model", MODEL_PY)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)  # would raise if NeMo were imported at top level
Model = mod.Model
print("1. module import (NeMo absent) — OK")


# ---- 2. mock cache-aware model ----------------------------------------------
class _Enc:
    def get_initial_cache_state(self, batch_size=1):
        return (None, None, None)


class _MockModel:
    """Emits a growing transcription that carries an <EOU> token, so we can prove
    the wrapper strips it. preprocessor/conformer_stream_step just satisfy the calls."""

    def __init__(self):
        self.encoder = _Enc()
        self._n = 0

    def preprocessor(self, input_signal=None, length=None):
        return input_signal, length

    def conformer_stream_step(self, **kw):
        self._n += 1
        # running hyp grows each step; always trailing an <EOU> the wrapper must remove
        transcribed = [" ".join(["the", "cat", "sat"][: self._n]) + " <EOU>"]
        return (None, transcribed, None, None, None, None)  # 6-tuple


def make_model(chunk_samples):
    m = object.__new__(Model)  # bypass __init__ (no NeMo)
    m._torch = torch
    m._device = torch.device("cpu")
    m._partial_callback = None
    m._strip_special = True
    m._special_re = re.compile(r"<[^>]+>")
    m.model = _MockModel()
    m._model_chunk_samples = chunk_samples
    m.reset()
    return m


CH = np.zeros(1600, dtype=np.float32)  # one 100 ms harness chunk

# ---- 3. cadence: 1 model-step per 100 ms chunk (chunk_samples == 1600) --------
print("3. buffering cadence")
m = make_model(1600)
partials = []
m.set_partial_callback(partials.append)
r1 = m.accept_chunk(CH)
check("one 1600-sample chunk -> exactly one step (hyp='the')", r1 == "the")
check("callback fired once with stripped partial", partials == ["the"])
r2 = m.accept_chunk(CH)
check("second chunk -> hyp grows ('the cat')", r2 == "the cat")

# ---- 4. sub-chunk buffering: need 2x1600 to make one 3200-sample step ---------
print("4. multi-chunk buffering (chunk_samples=3200)")
m = make_model(3200)
steps = []
m.set_partial_callback(steps.append)
a = m.accept_chunk(CH)
check("first 1600 buffered, no step yet (empty return)", a == "" and steps == [])
b = m.accept_chunk(CH)
check("second 1600 completes a 3200 step", b == "the" and steps == ["the"])

# ---- 5. <EOU>/special-token stripping ----------------------------------------
print("5. special-token stripping")
m = make_model(1600)
out = m.accept_chunk(CH)
check("no '<EOU>' in accept_chunk output", "<EOU>" not in out and "<" not in out)
check("whitespace normalized (no double spaces / trailing)", out == out.strip() and "  " not in out)

# ---- 6. input_finished flushes tail + returns cleaned final ------------------
print("6. input_finished")
m = make_model(3200)
finals = []
m.set_partial_callback(finals.append)
m.accept_chunk(CH)  # 1600 buffered, no step
fin = m.input_finished()  # flush the 1600 tail -> one step
check("input_finished flushes buffered tail (non-empty final)", fin == "the")
check("input_finished return has no special tokens", "<" not in fin)
check("final also delivered via callback", finals and finals[-1] == "the")

# ---- 7. reset clears state ---------------------------------------------------
print("7. reset semantics")
m = make_model(1600)
m.accept_chunk(CH)
m.accept_chunk(CH)
m.reset()
m.model._n = 0  # fresh mock counter too
after = m.accept_chunk(CH)
check("after reset, hyp restarts from first token", after == "the")

# ---- 8. _extract_text quirk absorption (str / Hypothesis / nested / None) ----
print("8. _extract_text NeMo-return-quirk")
class _Hyp:
    def __init__(self, t): self.text = t
check("plain str", Model._extract_text("hi") == "hi")
check("list of str -> [0]", Model._extract_text(["hi", "no"]) == "hi")
check("Hypothesis .text", Model._extract_text(_Hyp("hi")) == "hi")
check("nested [[Hypothesis]]", Model._extract_text([[_Hyp("hi")]]) == "hi")
check("None -> ''", Model._extract_text(None) == "")
check("empty list -> ''", Model._extract_text([]) == "")

# ---- verdict -----------------------------------------------------------------
print()
if failures:
    print(f"SMOKE FAILED: {len(failures)} check(s): {failures}")
    sys.exit(1)
print("SMOKE PASSED — contract/buffering/callback/strip logic OK (NeMo calls still VERIFY-ON-POD)")
