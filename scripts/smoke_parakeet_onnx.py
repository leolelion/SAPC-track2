#!/usr/bin/env python3
"""Local contract smoke for track2_starting_kit/parakeet_onnx/model.py.

Runs on the Mac with numpy+torch only — no ONNX weights, no NeMo, no pod. It builds a
throwaway submission tree with a synthetic streaming_meta.json and MOCK onnxruntime
sessions, then exercises the parts of the pipeline that are OURS rather than the model's:
the 5-method contract, feature-frame stepping cadence, partial-callback firing (the TTFT
driver), reset isolation, <EOU> stripping, RNN-T greedy blank/max-symbol handling, the
encoder trim-policy matrix, feature-cache bit-equivalence, and the loud-failure paths.

It makes NO accuracy claim. ONNX-vs-NeMo numerics are a separate POD gate
(scripts/parity_parakeet_onnx.py).

    python3 scripts/smoke_parakeet_onnx.py     # exit 0 on pass, 1 on fail
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "track2_starting_kit" / "parakeet_onnx"

failures: list[str] = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' — ' + detail) if detail and not cond else ''}")
    if not cond:
        failures.append(name)


# ---------------------------------------------------------------- geometry
# The VERIFIED values for parakeet_realtime_eou_120m @[70,1] (EXPERIMENT_LOG exp_parakeet_ft):
# chunk=[9,16] shift=[9,16] pre_encode_cache=[0,9] drop_extra_pre_encoded=2, 128 mel,
# hop 160, n_fft 512, normalize=NA. Encoder subsampling is 8x.
FEAT, HOP, NFFT, SUB = 128, 160, 512, 8
ENC_DIM, ENC_LAYERS, LC, LT = 512, 17, 70, 8
PRED_LAYERS, PRED_HID = 1, 640
BLANK = 1024


def build_meta() -> dict:
    return {
        "att_context_size": [70, 1],
        "streaming_cfg": {
            "chunk_size": [9, 16],
            "shift_size": [9, 16],
            "pre_encode_cache_size": [0, 9],
            "drop_extra_pre_encoded": 2,
            "valid_out_len": 2,
            "sampling_frames": [1, 8],
        },
        "feat_in": FEAT,
        "blank_id": BLANK,
        "vocab_size": BLANK + 1,
        "max_symbols_per_step": 3,
        "cache_last_channel_shape": [1, ENC_LAYERS, LC, ENC_DIM],
        "cache_last_time_shape": [1, ENC_LAYERS, ENC_DIM, LT],
        "cache_last_channel_len_shape": [1],
        "pred_state_shapes": [[PRED_LAYERS, 1, PRED_HID], [PRED_LAYERS, 1, PRED_HID]],
        "preprocessor": {
            "sample_rate": 16000,
            "features": FEAT,
            "n_fft": NFFT,
            "win_length": 400,
            "hop_length": HOP,
            "window": "hann",
            "preemph": 0.97,
            "mag_power": 2.0,
            "log_zero_guard_value": 5.960464477539063e-08,
            "log_zero_guard_type": "add",
            "normalize": "NA",
            "center": True,
            "log": True,
        },
    }


# ---------------------------------------------------------------- mock ORT
class _MockSession:
    def __init__(self, inputs, outputs, fn):
        self._in = inputs
        self._out = outputs
        self._fn = fn

    def get_inputs(self):
        return [types.SimpleNamespace(name=n) for n in self._in]

    def get_outputs(self):
        return [types.SimpleNamespace(name=n) for n in self._out]

    def run(self, _out_names, feeds):
        return self._fn(feeds)


def install_mock_ort(state):
    """Encoder: 8x subsampling, output frames carry the mean of their input window so the
    decoder can key on content. Decoder: emits one non-blank per encoder frame whose mean
    is above a threshold, then blank (exercises both the emit and blank paths, plus the
    max-symbols guard when `force_loop` is set)."""

    def enc_fn(feeds):
        sig = feeds["audio_signal"]
        t = sig.shape[-1]
        n = max(0, t // SUB)
        state["enc_calls"].append({"t": t, "length": int(feeds["length"][0]), "n_out": n})
        out = np.zeros((1, ENC_DIM, n), dtype=np.float32)
        for i in range(n):
            out[0, :, i] = float(sig[0, :, i * SUB : (i + 1) * SUB].mean())
        return [
            out,
            np.array([n], dtype=np.int64),
            feeds["cache_last_channel"],
            feeds["cache_last_time"],
            feeds["cache_last_channel_len"] + 1,
        ]

    def dec_fn(feeds):
        enc = feeds["encoder_outputs"]
        val = float(np.asarray(enc).reshape(-1)[0])
        last = int(np.asarray(feeds["targets"]).reshape(-1)[0])
        logits = np.zeros((1, 1, 1, BLANK + 1), dtype=np.float32)
        emit = state["force_loop"] or (val > state["threshold"] and last == BLANK)
        if emit:
            logits[0, 0, 0, state["token"]] = 10.0
        else:
            logits[0, 0, 0, BLANK] = 10.0
        state["dec_calls"] += 1
        return [logits, feeds["input_states_1"] + 1.0, feeds["input_states_2"] + 1.0]

    mod = types.ModuleType("onnxruntime")

    class GraphOptimizationLevel:
        ORT_ENABLE_ALL = 99

    class SessionOptions:
        pass

    def InferenceSession(path, so=None, providers=None):
        if "encoder" in str(path):
            return _MockSession(
                ["audio_signal", "length", "cache_last_channel", "cache_last_time", "cache_last_channel_len"],
                ["outputs", "encoded_lengths", "cache_last_channel_next", "cache_last_time_next", "cache_last_channel_next_len"],
                enc_fn,
            )
        return _MockSession(
            ["encoder_outputs", "targets", "target_length", "input_states_1", "input_states_2"],
            ["outputs", "output_states_1", "output_states_2"],
            dec_fn,
        )

    mod.GraphOptimizationLevel = GraphOptimizationLevel
    mod.SessionOptions = SessionOptions
    mod.InferenceSession = InferenceSession
    sys.modules["onnxruntime"] = mod


def build_tree(tmp: Path, meta: dict) -> Path:
    sub = tmp / "parakeet_onnx"
    (sub / "weights").mkdir(parents=True)
    for name in ("model.py", "localmel.py", "config.yaml"):
        shutil.copy2(SRC / name, sub / name)
    w = sub / "weights"
    (w / "streaming_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (w / "encoder_model.onnx").write_bytes(b"mock")
    (w / "decoder_model.onnx").write_bytes(b"mock")
    tokens = [f"tok{i} {i}" for i in range(BLANK)] + [f"<blk> {BLANK}"]
    tokens[7] = "▁hello 7"
    tokens[11] = "▁world 11"
    tokens[13] = "<EOU> 13"
    (w / "tokens.txt").write_text("\n".join(tokens) + "\n", encoding="utf-8")
    np.save(w / "filterbank.npy", np.random.RandomState(0).rand(FEAT, NFFT // 2 + 1).astype("float32"))
    np.save(w / "window.npy", np.hanning(400).astype("float32"))
    return sub


def load_model_module(sub: Path):
    sys.path.insert(0, str(sub))
    spec = importlib.util.spec_from_file_location("parakeet_onnx_model", sub / "model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- main
def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="parakeet_onnx_smoke_"))
    meta = build_meta()
    sub = build_tree(tmp, meta)

    # 1. model.py must import without onnxruntime present (lazy import contract)
    sys.modules.pop("onnxruntime", None)
    mod = load_model_module(sub)
    print("1. import with onnxruntime absent — OK")
    check("Model class exposes the 5-method contract",
          all(hasattr(mod.Model, m) for m in
              ("set_partial_callback", "reset", "accept_chunk", "input_finished")))

    # 2. thread policy: quota is a ceiling, env wins over config, floor of 1
    print("2. thread policy")
    r = mod._resolve_threads
    check("env overrides config", r(None, 24, env={"SAPC2_THREADS": "3"}, config={"runtime": {"num_threads": 1}})[0] == 3)
    check("quota caps the request", r(2, 24, env={"SAPC2_THREADS": "8"}, config={})[0] == 2)
    check("nproc caps the request", r(None, 2, env={"SAPC2_THREADS": "8"}, config={})[0] == 2)
    check("default is 1 when nothing is set", r(None, 24, env={}, config={})[0] == 1)
    check("never below 1", r(0.2, 24, env={"SAPC2_THREADS": "0"}, config={})[0] == 1)

    # 3. construction with mock ORT sessions
    print("3. construction + streaming state")
    state = {"enc_calls": [], "dec_calls": 0, "threshold": -1e9, "token": 7, "force_loop": False}
    install_mock_ort(state)
    m = mod.Model()
    check("geometry read from meta", (m._chunk_pair, m._shift_pair, m._pre_cache_pair, m._drop_extra)
          == ([9, 16], [9, 16], [0, 9], 2))
    check("cache tensors shaped from meta",
          m._cache_lc.shape == (1, ENC_LAYERS, LC, ENC_DIM) and m._cache_lt.shape == (1, ENC_LAYERS, ENC_DIM, LT))
    check("decoder states shaped from meta", [s.shape for s in m._dec_states] == [(PRED_LAYERS, 1, PRED_HID)] * 2)
    check("blank id from meta", m._blank_id == BLANK)
    check("max_symbols from config", m._max_symbols == 3)
    check("input gain ships OFF", m._gain_on is False)

    # 4. stepping cadence: first step at 9 feature frames, then every 16
    print("4. stepping cadence + callback firing")
    partials: list[str] = []
    m.set_partial_callback(partials.append)
    m.reset()
    state["enc_calls"].clear()
    rng = np.random.RandomState(1)
    audio = (rng.randn(16000) * 0.05).astype(np.float32)  # 1.0 s = 10 chunks
    steps_after = []
    for off in range(0, len(audio), 1600):
        m.accept_chunk(audio[off : off + 1600])
        steps_after.append(m._step)
    # 100 ms of audio -> 10 new frames (hop 160, center padding), so the first step lands
    # on chunk 1 and later steps every ~1.6 chunks.
    check("first step fires on the first 100 ms chunk", steps_after[0] == 1, f"steps={steps_after}")
    check("steps advance monotonically", all(b >= a for a, b in zip(steps_after, steps_after[1:])))
    check("feature pointer advances by shift", m._feat_idx == 9 + 16 * (m._step - 1))
    check("callback fired (TTFT driver)", len(partials) > 0)
    check("encoder input width = pre_cache + chunk on later steps",
          state["enc_calls"][1]["t"] == 9 + 16, str(state["enc_calls"][1]))
    check("encoder input width = chunk only on the first step",
          state["enc_calls"][0]["t"] == 9, str(state["enc_calls"][0]))

    final = m.input_finished()
    check("final text non-empty", bool(final))
    check("final partial fired", partials[-1] == final)
    check("<EOU> never reaches the output", "<EOU>" not in final and "<" not in final)

    # 5. reset isolates files
    print("5. reset isolation")
    m.reset()
    check("reset clears tokens/text/pointer", (m._tokens, m._hyp_text, m._feat_idx, m._step) == ([], "", 0, 0))
    check("reset zeroes encoder cache", float(np.abs(m._cache_lc).max()) == 0.0)
    check("reset restores blank as SOS", int(m._last_token[0, 0]) == BLANK)
    check("empty file returns empty string", m.input_finished() == "")

    # 6. special-token stripping on a hypothesis that contains one
    print("6. output normalisation")
    m.reset()
    m._tokens = [7, 13, 11]  # ▁hello <EOU> ▁world
    m._hyp_text = mod._detokenize(m._tokens, m._vocab)
    check("<EOU> stripped, words kept", m._clean(m._hyp_text) == "hello world", m._clean(m._hyp_text))

    # 7. RNN-T greedy: blank stops the symbol loop; max_symbols caps a non-blank loop
    print("7. greedy decode guards")
    m.reset()
    state["threshold"] = 1e9  # never emit
    before = len(m._tokens)
    enc = np.ones((1, ENC_DIM, 3), dtype=np.float32)
    m._decode_frames(enc, 0, 3)
    check("all-blank frames emit nothing", len(m._tokens) == before)
    state["force_loop"] = True  # always non-blank -> must be capped
    m._decode_frames(enc, 0, 2)
    check("max_symbols caps the non-blank loop", len(m._tokens) == 2 * m._max_symbols, f"{len(m._tokens)}")
    state["force_loop"] = False
    state["threshold"] = -1e9

    # 8. encoder trim-policy matrix (the two knobs parity decides)
    print("8. encoder output window policies")
    m.reset()
    m._step = 1  # a later step, so drop_extra applies under the nemo policy
    m._drop_policy, m._trim_policy = "nemo", "nemo"
    check("nemo/nemo drops drop_extra then trims to valid_out_len", m._encoder_out_window(6, 6, False) == (2, 2))
    m._trim_policy = "none"
    check("nemo/none drops only", m._encoder_out_window(6, 6, False) == (2, 4))
    m._drop_policy = "none"
    check("none/none passes through", m._encoder_out_window(6, 6, False) == (0, 6))
    m._drop_policy, m._trim_policy = "nemo", "nemo"
    check("last step is never trimmed", m._encoder_out_window(6, 6, True) == (2, 4))
    m._step = 0
    check("step 0 never drops", m._encoder_out_window(6, 6, True) == (0, 6))
    check("count never exceeds the tensor", m._encoder_out_window(99, 5, True) == (0, 5))

    # 9. feature cache must be bit-identical to full extraction (real torch STFT)
    print("9. feature cache equivalence")
    m.reset()
    m._feat_cache_on = True
    for off in range(0, len(audio), 1600):
        cached = m._extract_features(audio[: off + 1600])
    m.reset()
    m._feat_cache_on = False
    full = m._extract_features(audio)
    check("cached features == full extraction", cached.shape == full.shape and np.array_equal(cached, full),
          f"{cached.shape} vs {full.shape} maxdiff={np.abs(cached[:, :, :full.shape[-1]] - full[:, :, :cached.shape[-1]]).max() if cached.size and full.size else 'n/a'}")
    # negative control: a geometrically wrong margin must be CAUGHT by the check above
    m.reset()
    m._feat_cache_on, m._feat_margin = True, 0
    for off in range(0, len(audio), 1600):
        bad = m._extract_features(audio[: off + 1600])
    check("negative control: margin=0 is detectably wrong",
          not (bad.shape == full.shape and np.array_equal(bad, full)))
    m._feat_margin = build_meta_margin()

    # 10. loud failures — never a silent fallback
    print("10. loud failure paths")
    from localmel import LocalMel  # noqa: E402  (sub dir is on sys.path)

    bad_meta = build_meta()
    bad_meta["preprocessor"]["normalize"] = "per_feature"
    try:
        LocalMel(str(sub / "weights"), bad_meta)
        check("unimplemented normalisation raises", False)
    except RuntimeError:
        check("unimplemented normalisation raises", True)

    missing = tmp / "no_meta"
    (missing / "weights").mkdir(parents=True)
    for name in ("model.py", "localmel.py", "config.yaml"):
        shutil.copy2(SRC / name, missing / name)
    mod2 = load_model_module(missing)
    try:
        mod2.Model()
        check("missing streaming_meta.json raises", False)
    except RuntimeError:
        check("missing streaming_meta.json raises", True)

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{'ALL CHECKS PASSED' if not failures else 'FAILURES: ' + ', '.join(failures)}")
    return 1 if failures else 0


def build_meta_margin() -> int:
    import math

    return int(math.ceil(NFFT / 2 / HOP)) + 2


if __name__ == "__main__":
    sys.exit(main())
