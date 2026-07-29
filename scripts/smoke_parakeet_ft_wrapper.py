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
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf

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
# Shaped like the real NeMo model so the wrapper's OWN _setup_streaming() can run against
# it. Geometry values are the VERIFIED ones for parakeet_realtime_eou_120m @[70,1]
# (model.py:_setup_streaming docstring): chunk=[9,16] shift=[9,16] pre_cache=[0,9]
# drop_extra_pre_encoded=2, window_stride=0.01, n_fft=512, features=128, normalize='NA'.
HOP, N_FFT, FEAT_IN = 160, 512, 128

PREPROC_CFG = {
    "normalize": "NA",
    "sample_rate": 16000,
    "window_stride": 0.01,
    "n_fft": N_FFT,
    "dither": 1e-5,
    "pad_to": 16,
}


class _MockPreproc:
    """Deterministic stand-in for the mel extractor: [1,N] raw -> [1,128,T] features,
    T = N//HOP + 1.

    Must be TRANSLATION-INVARIANT and CENTERED to mirror the real thing, or it does not
    test what it claims. Frame t is centered on sample t*HOP with n_fft support, zero-padded
    at the edges (NeMo's center=True). Its value depends ONLY on the samples under it —
    never on the frame's index within the extracted segment — because the feature cache
    legitimately re-extracts a tail SEGMENT with left context and splices it. A
    position-dependent mock would fail any correct cache implementation.

    Detection of splice/alignment bugs comes from the audio being seeded random noise: every
    frame's window sum is distinct, so a misaligned splice changes values.
    """

    def to(self, device):
        return self

    def __call__(self, input_signal=None, length=None):
        x = input_signal[0].detach().cpu().numpy()
        n_frames = max(0, x.shape[0] // HOP + 1)
        half = N_FFT // 2
        padded = np.concatenate([np.zeros(half, np.float32), x, np.zeros(half + HOP, np.float32)])
        out = torch.zeros(1, FEAT_IN, n_frames, dtype=torch.float32)
        for t in range(n_frames):
            w = padded[t * HOP: t * HOP + N_FFT]     # centered on sample t*HOP
            out[0, :, t] = float(w.sum())
        return out, torch.tensor([n_frames], dtype=torch.int64)


class _PreEncode:
    def get_sampling_frames(self):
        return [1, 1]


class _Enc:
    _feat_in = FEAT_IN

    def __init__(self):
        self.streaming_cfg = SimpleNamespace(
            chunk_size=[9, 16], shift_size=[9, 16],
            pre_encode_cache_size=[0, 9], drop_extra_pre_encoded=2,
        )
        self.pre_encode = _PreEncode()

    def get_initial_cache_state(self, batch_size=1):
        return (None, None, None)


class _MockModel:
    """Emits a growing transcription carrying an <EOU> the wrapper must strip, and records
    every conformer_stream_step kwarg so the cache-aware feeding contract can be asserted."""

    def __init__(self):
        self.encoder = _Enc()
        self.cfg = OmegaConf.create({"preprocessor": dict(PREPROC_CFG)})
        self._cfg = self.cfg
        self._n = 0
        self.calls = []

    def from_config_dict(self, cfg):
        return _MockPreproc()

    def conformer_stream_step(self, **kw):
        self._n += 1
        self.calls.append(kw)
        # running hyp grows each step; always trailing an <EOU> the wrapper must remove
        transcribed = [" ".join(["the", "cat", "sat", "on", "mat"][: self._n]) + " <EOU>"]
        return (None, transcribed, None, None, None, None)  # 6-tuple


def make_model():
    m = object.__new__(Model)  # bypass __init__ (no NeMo)
    m._torch = torch
    m._device = torch.device("cpu")
    # Drive the REAL setup rather than re-listing attributes here. Hand-listing is exactly
    # what broke this smoke: the input-gain and feature-cache commits each added attributes
    # __init__ sets, the list here rotted, and the 5-method contract went untested.
    m._init_runtime_cfg()
    m.model = _MockModel()
    m._setup_streaming()
    m.reset()
    return m


rng = np.random.default_rng(0)
CH = (rng.standard_normal(1600) * 0.05).astype(np.float32)  # one 100 ms harness chunk

# ---- 3. cadence is FEATURE-FRAME driven, not one-step-per-chunk ---------------
# Geometry: hop=160 so T = N//160 + 1 frames; first step needs chunk_pair[0]=9 frames,
# later steps 16, shift matches. Steps per 100 ms chunk therefore go 1,0,1,1,0 — NOT one
# per chunk. Derived from the geometry, not from the mock.
print("3. buffering cadence (feature-frame geometry)")
m = make_model()
partials = []
m.set_partial_callback(partials.append)
r1 = m.accept_chunk(CH)
check("chunk1 (T=11 >= 9) -> exactly one step, hyp='the'", r1 == "the")
check("callback fired once with stripped partial", partials == ["the"])
r2 = m.accept_chunk(CH)
check("chunk2 (T=21 < 9+16) -> no step, hyp unchanged", r2 == "the" and partials == ["the"])
r3 = m.accept_chunk(CH)
check("chunk3 (T=31 >= 25) -> one step, hyp grows", r3 == "the cat")
r4 = m.accept_chunk(CH)
check("chunk4 (T=41 >= 41) -> one step", r4 == "the cat sat")
r5 = m.accept_chunk(CH)
check("chunk5 (T=51 < 57) -> no step", r5 == "the cat sat")
check("3 model steps over 5 chunks (1,0,1,1,0)", m.model._n == 3)

# ---- 4. cache-aware feeding contract on the conformer_stream_step kwargs ------
print("4. cache-aware feeding contract")
c0, c1 = m.model.calls[0], m.model.calls[1]
check("step0 feeds exactly chunk_pair[0]=9 frames (no pre-cache)", c0["processed_signal"].shape[-1] == 9)
check("step0 drops no pre-encoded frames", c0["drop_extra_pre_encoded"] == 0)
check("step1 feeds pre_cache 9 + chunk 16 = 25 frames", c1["processed_signal"].shape[-1] == 25)
check("step1 drops drop_extra_pre_encoded=2", c1["drop_extra_pre_encoded"] == 2)
check("keep_all_outputs False while streaming", c0["keep_all_outputs"] is False)
check("return_transcription requested", c0["return_transcription"] is True)

# ---- 5. <EOU>/special-token stripping ----------------------------------------
print("5. special-token stripping")
m = make_model()
out = m.accept_chunk(CH)
check("no '<EOU>' in accept_chunk output", "<EOU>" not in out and "<" not in out)
check("whitespace normalized (no double spaces / trailing)", out == out.strip() and "  " not in out)

# ---- 6. input_finished flushes the sub-chunk tail ----------------------------
print("6. input_finished")
m = make_model()
finals = []
m.set_partial_callback(finals.append)
for _ in range(5):
    m.accept_chunk(CH)          # 3 steps, feat_idx=41, T=51 -> 10 frames unconsumed
n_before = m.model._n
fin = m.input_finished()        # tail >= sampling_frames -> one final is_last step
check("input_finished runs the trailing partial chunk", m.model._n == n_before + 1)
check("final hyp includes the flushed step", fin == "the cat sat on")
check("input_finished return has no special tokens", "<" not in fin)
check("final also delivered via callback", finals and finals[-1] == "the cat sat on")
check("last step sets keep_all_outputs=True", m.model.calls[-1]["keep_all_outputs"] is True)

# ---- 7. reset clears state ---------------------------------------------------
print("7. reset semantics")
m = make_model()
m.accept_chunk(CH)
m.accept_chunk(CH)
m.reset()
m.model._n = 0  # fresh mock counter too
after = m.accept_chunk(CH)
check("after reset, hyp restarts from first token", after == "the")
check("reset clears the feature cache", m._feat_cache is not None and m._feat_idx == 9)

# ---- 7b. feature cache (Fix 2) equivalence: on must equal off -----------------
# The on-pod gate proved this byte-identical on real audio; this keeps it locally guarded
# so a splice/indexing regression is caught before it costs pod time.
print("7b. feature-cache on/off equivalence")
import os as _os

def run_stream(cache_on):
    _os.environ["SAPC2_FEAT_CACHE"] = "on" if cache_on else "off"
    mm = make_model()
    outs = [mm.accept_chunk(CH) for _ in range(6)]
    outs.append(mm.input_finished())
    fed = [c["processed_signal"].clone() for c in mm.model.calls]
    _os.environ.pop("SAPC2_FEAT_CACHE", None)
    return outs, fed

o_on, fed_on = run_stream(True)
o_off, fed_off = run_stream(False)
check("cache on/off produce identical hypotheses", o_on == o_off)
check("cache on/off feed the same number of steps", len(fed_on) == len(fed_off))
check("cache on/off feed bit-identical features",
      len(fed_on) == len(fed_off) and all(torch.equal(a, b) for a, b in zip(fed_on, fed_off)))

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
