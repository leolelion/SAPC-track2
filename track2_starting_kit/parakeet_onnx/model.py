#!/usr/bin/env python3
"""
parakeet_realtime (120M) cache-aware streaming, ONNX/onnxruntime — SAPC2 Track 2.
=================================================================================

WHAT THIS IS
------------
The ship vehicle for the banked **parakeet Arm A** (encoder-only dysarthric fine-tune of
`nvidia/parakeet_realtime_eou_120m-v1`, EncDecRNNTBPEModel, cache-aware [70,1] / 80 ms
lookahead). Same model and same streaming geometry as the NeMo-backed sibling
`../parakeet_realtime_ft/`, but with **no NeMo and no torch modules at inference**:

  encoder + prediction-net/joint  ->  ONNX, run by onnxruntime (int8 encoder)
  mel front-end                   ->  localmel.py (bundled filterbank/window .npy)
  RNN-T greedy decode             ->  hand-rolled here (was NeMo's greedy_batch)

Why: `parakeet_realtime_ft/setup.sh` needs PyPI (`nemo_toolkit[asr]`) + HuggingFace at
setup time, and **no NeMo submission has ever scored for us** — every submission that
scored (A1 greedy, beam-4, Nemotron int8) ran ONNX through onnxruntime with bundled
wheels and `pip install --no-index`. See investigations/parakeet_ship_runbook.md.

VERIFICATION STATUS (read before trusting any number produced by this file)
--------------------------------------------------------------------------
VERIFIED LOCALLY (no ORT, no weights): the 5-method contract, feature-frame stepping
cadence, callback firing, reset semantics, <EOU> stripping, feature-cache equivalence,
thread policy — `python3 scripts/smoke_parakeet_onnx.py`.
NOT VERIFIED ANYWHERE YET: that ONNX numerics match NeMo. That is a POD gate:
`scripts/parity_parakeet_onnx.py` (feature -> encoder tensor -> token -> text parity vs
the NeMo wrapper), then the real `local_decode.py` + `evaluate.sh` gate on Dev.
**Do not quote a CER from this file until the parity + real-harness gates have run.**

TWO KNOBS THAT PARITY DECIDES (do not guess them — see runbook "Stage 2")
-------------------------------------------------------------------------
NeMo applies two trims to encoder output that live OUTSIDE `encoder.forward`, in
`cache_aware_stream_step` / `streaming_post_process`:
  * `drop_extra_pre_encoded` — drop the leading encoded frames that correspond to the
    prepended pre-encode cache (0 on step 0, else `drop_extra`);
  * `valid_out_len` — trim intermediate steps to the valid output length.
Whether the *exported graph* already does either is an empirical question about NeMo's
exporter, not something to reason about from the docs. So both are policies here
(`encoder.drop_policy`, `encoder.trim_policy` in config.yaml, env-overridable), the
parity script sweeps the 2x2 grid against NeMo reference tensors, and the winner is
written into config.yaml before packaging. Defaults mirror NeMo's helper.

Interface (called by local_decode.py; Decoder thread only, no locking needed):
  __init__()                        — load ONNX + tokenizer once
  set_partial_callback(fn) -> None  — register fn(text:str)
  reset()             -> None       — per-file state reset
  accept_chunk(np.float32) -> str   — 1600 samples (100 ms), returns running partial
  input_finished()    -> str        — final hypothesis
"""

# =====================================================================
# Section 1: thread policy — must precede any heavy import
# =====================================================================
import json
import math
import os
import re
import sys
from pathlib import Path

_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 1600  # 100 ms, as delivered by local_decode.py


def _load_config():
    """config.yaml without a hard omegaconf dependency (the Codabench runtime has it, but
    this file must stay importable for the local smoke on a box with only numpy)."""
    path = _DIR / "config.yaml"
    try:
        from omegaconf import OmegaConf

        return OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    except Exception:
        import yaml  # PyYAML is in the runtime; last resort

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)


_CONFIG = _load_config()


def _cgroup_cpu_quota():
    """Container CPU quota in cores, or None if unlimited/unreadable. Kept identical to
    parakeet_realtime_ft/model.py so scripts/test_thread_policy.py covers both."""
    try:
        if os.path.exists("/sys/fs/cgroup/cpu.max"):  # cgroup v2
            parts = open("/sys/fs/cgroup/cpu.max").read().split()
            if parts and parts[0] != "max":
                return float(parts[0]) / float(parts[1])
        elif os.path.exists("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"):  # cgroup v1
            quota = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
            period = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
            if quota > 0:
                return quota / period
    except Exception:
        pass
    return None


def _resolve_threads(quota, nproc, env=None, config=None):
    """Intra-op threads. The cgroup quota is a CEILING, never the target.

    The organizers' accuracy pass runs ~20 worker PROCESSES on a 24-vCPU worker
    (confirmed from our own submission's CPU_DIAGNOSTIC), so threads multiply by 20.
    Under that exact topology exp_nemotron_speed_002 measured threads=1 at 4.2x the
    wall-clock throughput of threads=4 with byte-identical transcripts. Wall clock is
    what the 15000 s/submission budget measures, so the default is 1.

    Precedence: SAPC2_THREADS env > config runtime.num_threads > 1, capped by quota/nproc.
    """
    env = os.environ if env is None else env
    config = _CONFIG if config is None else config
    rt = (config or {}).get("runtime", {}) or {}
    want = int(env.get("SAPC2_THREADS", rt.get("num_threads", 1)))
    nproc = max(1, int(nproc or 1))
    ceiling = int(quota) if (quota and quota >= 1) else nproc
    return max(1, min(want, ceiling, nproc)), want, ceiling


_THREADS, _WANT_THREADS, _CEIL_THREADS = _resolve_threads(_cgroup_cpu_quota(), os.cpu_count())
# ORT reads these at session creation; torch (used only by localmel's STFT) at import.
os.environ.setdefault("OMP_NUM_THREADS", str(_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(_THREADS))

import numpy as np  # noqa: E402

# Pin a venv if setup.sh installed one next to this file (Nemotron submission precedent).
import glob as _glob  # noqa: E402

_venv = _glob.glob(str(_DIR / "venv" / "lib" / "python3.*" / "site-packages")) or _glob.glob(
    str(_DIR / "venv" / "*" / "lib" / "python3.*" / "site-packages")
)
if _venv:
    sys.path.insert(0, _venv[0])


# =====================================================================
# Section 2: tokenizer (sentencepiece-style "<piece> <id>" lines)
# =====================================================================
def _load_tokens(path):
    vocab = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            piece, _, idx = line.rpartition(" ")
            try:
                vocab[int(idx)] = piece
            except ValueError:
                continue
    if not vocab:
        raise RuntimeError(f"empty token file: {path}")
    return [vocab.get(i, "") for i in range(max(vocab) + 1)]


def _detokenize(tokens, vocab):
    pieces = [vocab[t] for t in tokens if 0 <= t < len(vocab)]
    return "".join(pieces).replace("▁", " ").strip()


# =====================================================================
# Section 3: Model
# =====================================================================
class Model:
    """Cache-aware streaming RNN-T over ONNX. See VERIFICATION STATUS in the header."""

    # ------------------------------------------------------------------
    def _init_runtime_cfg(self, meta):
        """Every config/meta/env-derived attribute, in ONE place, with no ORT and no torch
        dependency — __init__ calls it and so does the local contract smoke (which has no
        weights). The NeMo sibling learned this the hard way: when its smoke duplicated
        this attribute list by hand, the list rotted and the 5-method contract went three
        commits untested. Do not set runtime attributes anywhere else."""
        _env = os.environ.get
        self._meta = meta
        self._partial_callback = None

        # --- streaming geometry (from the exporting model's own streaming_cfg) ---
        def _pair(v, default):
            if v is None:
                return [int(default), int(default)]
            if isinstance(v, (list, tuple)):
                return [int(v[0]), int(v[-1])]
            return [int(v), int(v)]

        sc = meta["streaming_cfg"]
        self._chunk_pair = _pair(sc.get("chunk_size"), CHUNK_SAMPLES // 160)
        self._shift_pair = _pair(sc.get("shift_size"), self._chunk_pair[0])
        self._pre_cache_pair = _pair(sc.get("pre_encode_cache_size"), 0)
        self._drop_extra = int(sc.get("drop_extra_pre_encoded") or 0)
        self._sampling_pair = _pair(sc.get("sampling_frames"), 1)
        self._valid_out_len = sc.get("valid_out_len")
        self._valid_out_len = None if self._valid_out_len is None else int(self._valid_out_len)
        self._feat_in = int(meta["feat_in"])

        # --- encoder-output trim policies (parity decides; see header) ---
        enc_cfg = (_CONFIG.get("encoder") or {})
        self._drop_policy = _env("SAPC2_ENC_DROP", enc_cfg.get("drop_policy", "nemo")).lower()
        self._trim_policy = _env("SAPC2_ENC_TRIM", enc_cfg.get("trim_policy", "nemo")).lower()
        if self._drop_policy not in ("nemo", "none"):
            raise RuntimeError(f"encoder.drop_policy={self._drop_policy!r} must be nemo|none")
        if self._trim_policy not in ("nemo", "none"):
            raise RuntimeError(f"encoder.trim_policy={self._trim_policy!r} must be nemo|none")

        # --- RNN-T decode ---
        # Precedence: env > config.yaml (only if set) > the checkpoint's own decoding cfg
        # (exported into meta) > 10. config.yaml ships null so the checkpoint wins by
        # default — a hard-coded override here would silently mask a retuned checkpoint.
        dec_cfg = (_CONFIG.get("decoding") or {})
        self._blank_id = int(meta["blank_id"])
        _ms = dec_cfg.get("max_symbols_per_step")
        if _ms is None:
            _ms = meta.get("max_symbols_per_step") or 10
        self._max_symbols = int(_env("SAPC2_MAX_SYMBOLS", _ms))

        # --- output normalisation: strip <EOU> and friends (research/46: 100/123 sweep
        # hyps carried <EOU>). Never emit a special token to the scorer. ---
        out_cfg = (_CONFIG.get("output") or {})
        self._strip_special = bool(out_cfg.get("strip_special_tokens", True))
        self._special_re = re.compile(str(out_cfg.get("special_token_pattern", r"<[^>]+>")))

        # --- causal frozen-scalar input gain: ships OFF (falsified — 0/48 severe empties
        # recovered through the real harness). Kept so the banked A/B can be reproduced
        # on a pod without editing code. ---
        ig = (_CONFIG.get("input_gain") or {})
        self._gain_on = _env("SAPC2_INPUT_GAIN", "on" if ig.get("enabled", False) else "off").lower() != "off"
        _tgt_db = float(_env("SAPC2_TARGET_DBFS", ig.get("target_dbfs", -25.0)))
        _cap_db = float(_env("SAPC2_GAIN_CAP_DB", ig.get("cap_db", 20.0)))
        _flr_db = float(_env("SAPC2_RMS_FLOOR_DBFS", ig.get("floor_dbfs", -45.0)))
        self._gain_target_rms = 10.0 ** (_tgt_db / 20.0)
        self._gain_cap = 10.0 ** (_cap_db / 20.0)
        self._gain_rms_floor = 10.0 ** (_flr_db / 20.0)
        self._gain_dbg = (_tgt_db, _cap_db, _flr_db)

        # --- incremental feature cache (O(N^2) -> O(N); mandatory for the 15000 s budget:
        # the un-cached path made the untimed accuracy pass ~17 min per 425 utts) ---
        pp = meta["preprocessor"]
        self._feat_cache_on = _env("SAPC2_FEAT_CACHE", "on").lower() != "off"
        self._feat_hop = int(pp["hop_length"])
        n_fft = int(pp["n_fft"])
        self._feat_margin = int(math.ceil(n_fft / 2 / self._feat_hop)) + 2
        self._feat_lctx = (int(math.ceil(n_fft / self._feat_hop)) + 1) * self._feat_hop

        # --- cache tensor shapes (encoder state), exported with the graph ---
        self._cache_lc_shape = tuple(meta["cache_last_channel_shape"])
        self._cache_lt_shape = tuple(meta["cache_last_time_shape"])
        self._pred_state_shapes = [tuple(s) for s in meta["pred_state_shapes"]]

    # ------------------------------------------------------------------
    def __init__(self):
        weights = _DIR / "weights"
        meta_path = weights / "streaming_meta.json"
        if not meta_path.exists():
            raise RuntimeError(
                f"missing {meta_path} — the submission tree is incomplete. Build it with "
                "scripts/export_parakeet_onnx.py; never fall back to a guessed geometry."
            )
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self._init_runtime_cfg(meta)

        import onnxruntime as ort

        def _session(path):
            so = ort.SessionOptions()
            so.intra_op_num_threads = _THREADS
            so.inter_op_num_threads = 1
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])

        self._enc = _session(weights / "encoder_model.onnx")
        self._dec = _session(weights / "decoder_model.onnx")
        self._enc_out_names = [o.name for o in self._enc.get_outputs()]
        self._dec_out_names = [o.name for o in self._dec.get_outputs()]
        self._enc_in_names = [i.name for i in self._enc.get_inputs()]
        self._dec_in_names = [i.name for i in self._dec.get_inputs()]
        self._check_io()

        from localmel import LocalMel

        self._preproc = LocalMel(str(weights), meta)
        import torch  # only for the STFT inside localmel

        self._torch = torch
        torch.set_num_threads(_THREADS)
        try:
            torch.set_num_interop_threads(1)
        except Exception:
            pass  # already initialised in this process

        self._vocab = _load_tokens(weights / "tokens.txt")
        if len(self._vocab) <= self._blank_id:
            raise RuntimeError(
                f"tokens.txt has {len(self._vocab)} entries but blank_id={self._blank_id}"
            )

        self.reset()
        print(
            f"[parakeet_onnx] ready (threads={_THREADS} requested={_WANT_THREADS} "
            f"ceiling={_CEIL_THREADS} chunk={self._chunk_pair} shift={self._shift_pair} "
            f"pre_cache={self._pre_cache_pair} drop_extra={self._drop_extra} "
            f"valid_out_len={self._valid_out_len} drop_policy={self._drop_policy} "
            f"trim_policy={self._trim_policy} blank={self._blank_id} "
            f"max_symbols={self._max_symbols} vocab={len(self._vocab)} "
            f"input_gain={'on' if self._gain_on else 'off'})",
            flush=True,
        )

    # ------------------------------------------------------------------
    def _check_io(self):
        """Fail loud on an unexpected graph signature. The decode loop indexes ONNX inputs
        by name; a renamed input would otherwise surface as a wrong-but-plausible
        transcript, which is exactly the failure mode that cost us the Nemotron
        submission (a deterministic wrong output that looked like a real result)."""
        need_enc_in = {"audio_signal", "length", "cache_last_channel", "cache_last_time", "cache_last_channel_len"}
        need_dec_in = {"encoder_outputs", "targets", "target_length"}
        missing = need_enc_in - set(self._enc_in_names)
        if missing:
            raise RuntimeError(f"encoder ONNX missing inputs {sorted(missing)}; has {self._enc_in_names}")
        missing = need_dec_in - set(self._dec_in_names)
        if missing:
            raise RuntimeError(f"decoder ONNX missing inputs {sorted(missing)}; has {self._dec_in_names}")
        # Pair states by their trailing index (input_states_1 <-> output_states_1) rather
        # than by graph order, so a reordered export cannot silently swap LSTM h and c.
        def _by_index(names, prefix):
            return sorted(
                (n for n in names if n.startswith(prefix)),
                key=lambda n: int(re.sub(r"\D", "", n[len(prefix) :]) or 0),
            )

        self._dec_state_in = _by_index(self._dec_in_names, "input_states_")
        self._dec_state_out = _by_index(self._dec_out_names, "output_states_")
        if len(self._dec_state_in) != len(self._pred_state_shapes) or not self._dec_state_in:
            raise RuntimeError(
                f"decoder ONNX state inputs {self._dec_state_in} do not match exported "
                f"pred_state_shapes {self._pred_state_shapes}"
            )
        if len(self._dec_state_out) != len(self._dec_state_in):
            raise RuntimeError(f"decoder state in/out mismatch: {self._dec_state_in} vs {self._dec_state_out}")
        # encoder outputs: [encoded, encoded_len, cache_ch_next, cache_t_next, cache_ch_len_next]
        if len(self._enc_out_names) < 5:
            raise RuntimeError(f"encoder ONNX has {len(self._enc_out_names)} outputs, expected >=5: {self._enc_out_names}")

    # ------------------------------------------------------------------
    # SAPC2 interface
    # ------------------------------------------------------------------
    def set_partial_callback(self, callback) -> None:
        self._partial_callback = callback

    def reset(self) -> None:
        """Fresh encoder cache, decoder state, feature buffer for a new file."""
        self._raw_buf = np.zeros(0, dtype=np.float32)
        self._gain = None
        self._feat_cache = None
        self._feat_idx = 0
        self._step = 0
        self._tokens = []
        self._hyp_text = ""
        self._last_emitted = ""
        self._cache_lc = np.zeros(self._cache_lc_shape, dtype=np.float32)
        self._cache_lt = np.zeros(self._cache_lt_shape, dtype=np.float32)
        self._cache_ll = np.zeros((self._cache_lc_shape[0],), dtype=np.int64)
        self._dec_states = [np.zeros(s, dtype=np.float32) for s in self._pred_state_shapes]
        self._last_token = np.array([[self._blank_id]], dtype=np.int32)
        self._target_length = np.array([1], dtype=np.int32)

    def accept_chunk(self, audio_chunk: np.ndarray) -> str:
        """Accumulate 100 ms of raw audio; run one model step per newly complete
        feature-frame chunk. Partials fire through the callback (this drives TTFT)."""
        self._raw_buf = np.concatenate([self._raw_buf, np.asarray(audio_chunk, dtype=np.float32)])
        feats = self._extract_features(self._raw_buf)
        total = feats.shape[-1]
        while True:
            chunk_size = self._chunk_pair[0] if self._feat_idx == 0 else self._chunk_pair[1]
            if self._feat_idx + chunk_size > total:
                break
            self._stream_step(feats, total, is_last=False)
        return self._clean(self._hyp_text)

    def input_finished(self) -> str:
        """Flush newly complete chunks, then the final short remainder, and finalise."""
        feats = self._extract_features(self._raw_buf)
        total = feats.shape[-1]
        while True:
            chunk_size = self._chunk_pair[0] if self._feat_idx == 0 else self._chunk_pair[1]
            if self._feat_idx + chunk_size > total:
                break
            self._stream_step(feats, total, is_last=False)
        if self._feat_idx < total:
            samp = self._sampling_pair[0] if self._feat_idx == 0 else self._sampling_pair[1]
            if (total - self._feat_idx) >= samp:
                self._stream_step(feats, total, is_last=True)
        final = self._clean(self._hyp_text)
        if self._partial_callback is not None and final:
            self._partial_callback(final)
        return final

    # ------------------------------------------------------------------
    # Features
    # ------------------------------------------------------------------
    def _compute_gain(self, est_window: np.ndarray) -> float:
        """Frozen boost-only gain from the first ~100 ms; 1.0 when that window is
        near-silent (never amplify leading silence by +cap dB)."""
        if not self._gain_on:
            return 1.0
        x = est_window.astype(np.float64)
        rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
        if rms < self._gain_rms_floor:
            return 1.0
        return float(min(max(self._gain_target_rms / rms, 1.0), self._gain_cap))

    def _raw_to_feats(self, raw: np.ndarray) -> np.ndarray:
        torch = self._torch
        wav = torch.from_numpy(np.ascontiguousarray(raw, dtype=np.float32)).unsqueeze(0)
        with torch.inference_mode():
            feats, _ = self._preproc(input_signal=wav, length=torch.tensor([raw.shape[0]]))
        return feats.numpy().astype(np.float32)  # [1, F, T]

    def _extract_features(self, raw: np.ndarray) -> np.ndarray:
        """Un-normalised log-mel for the whole raw buffer -> [1, F, T].

        Two invariants make caching exact rather than approximate: (1) the per-file input
        gain is a frozen scalar, so already-scaled audio never changes retroactively;
        (2) with hop/n_fft fixed, only the last MARGIN frames touch the STFT boundary, so
        we recompute just the tail (with LCTX real left context) and splice it onto the
        stable prefix. Bit-equivalence vs full extraction is asserted in the local smoke
        and re-checked on-pod."""
        if raw.shape[0] < 1:
            return np.zeros((1, self._feat_in, 0), dtype=np.float32)
        if self._gain is None:
            self._gain = self._compute_gain(raw[:CHUNK_SAMPLES])
        scaled = raw if self._gain == 1.0 else np.clip(raw * self._gain, -1.0, 1.0).astype(np.float32)

        if not self._feat_cache_on or self._feat_cache is None:
            feats = self._raw_to_feats(scaled)
            if self._feat_cache_on:
                self._feat_cache = feats
            return feats

        HOP, MARGIN, LCTX = self._feat_hop, self._feat_margin, self._feat_lctx
        Tc = self._feat_cache.shape[-1]
        stable = max(0, Tc - MARGIN)
        start = max(0, stable * HOP - LCTX)  # both are hop multiples -> exact splice
        seg = self._raw_to_feats(scaled[start:])
        drop = max(0, min(stable - (start // HOP), seg.shape[-1]))
        feats = np.concatenate([self._feat_cache[:, :, :stable], seg[:, :, drop:]], axis=-1)
        self._feat_cache = feats
        return feats

    # ------------------------------------------------------------------
    # Streaming step
    # ------------------------------------------------------------------
    def _clean(self, text: str) -> str:
        """Strip <EOU>/special tokens and collapse whitespace on OUTPUT only."""
        if not text:
            return ""
        if self._strip_special:
            text = self._special_re.sub(" ", text)
        return " ".join(text.split())

    def _stream_step(self, feats: np.ndarray, total_frames: int, is_last: bool) -> str:
        """ONE cache-aware step over feature frames [pre_encode_cache | chunk].

        Frame bookkeeping is a straight port of the NeMo sibling's `_stream_step` (itself
        ported from CacheAwareStreamingAudioBuffer), which is the version validated end to
        end through the real harness at Dev_clean2k 13.51% / severe 18.69%. Keep the two
        in step: if this arithmetic changes, change it in both files."""
        first = self._feat_idx == 0
        chunk_size = self._chunk_pair[0] if first else self._chunk_pair[1]
        shift_size = self._shift_pair[0] if first else self._shift_pair[1]
        pre_cache = self._pre_cache_pair[0] if first else self._pre_cache_pair[1]

        chunk = feats[:, :, self._feat_idx : self._feat_idx + chunk_size]

        zeros_pads = None
        if first:
            cache = feats[:, :, 0:0]
        else:
            start = max(0, self._feat_idx - pre_cache)
            cache = feats[:, :, start : self._feat_idx]
            if cache.shape[-1] < pre_cache:
                zeros_pads = np.zeros(
                    (chunk.shape[0], chunk.shape[1], pre_cache - cache.shape[-1]), dtype=np.float32
                )
        added_len = cache.shape[-1]
        fed = np.concatenate((cache, chunk), axis=-1)
        if zeros_pads is not None:
            fed = np.concatenate((zeros_pads, fed), axis=-1)
            added_len += zeros_pads.shape[-1]

        valid = max(0, total_frames - self._feat_idx) + added_len
        length = min(valid, fed.shape[-1])

        enc_out = self._enc.run(
            None,
            {
                "audio_signal": np.ascontiguousarray(fed, dtype=np.float32),
                "length": np.array([length], dtype=np.int64),
                "cache_last_channel": self._cache_lc,
                "cache_last_time": self._cache_lt,
                "cache_last_channel_len": self._cache_ll,
            },
        )
        named = dict(zip(self._enc_out_names, enc_out))
        encoded = named.get("outputs", enc_out[0])                  # [1, D, T_enc]
        n_enc = int(np.asarray(named.get("encoded_lengths", enc_out[1])).reshape(-1)[0])
        for name, val in named.items():
            low = name.lower()
            if "cache_last_channel" in low and ("_len" in low or "_length" in low):
                self._cache_ll = val
            elif "cache_last_channel" in low and name != "cache_last_channel":
                self._cache_lc = val
            elif "cache_last_time" in low and name != "cache_last_time":
                self._cache_lt = val

        start_f, n_frames = self._encoder_out_window(n_enc, encoded.shape[-1], is_last)
        self._decode_frames(encoded, start_f, n_frames)

        self._feat_idx += shift_size
        self._step += 1

        text = self._clean(self._hyp_text)
        if text and text != self._last_emitted:
            self._last_emitted = text
            if self._partial_callback is not None:
                self._partial_callback(text)
        return text

    def _encoder_out_window(self, n_enc: int, t_enc: int, is_last: bool):
        """Which encoder output frames to decode -> (start, count).

        `drop_policy=nemo` reproduces cache_aware_stream_step's `drop_extra_pre_encoded`
        (0 on step 0, else drop_extra). `trim_policy=nemo` reproduces
        streaming_post_process's valid_out_len trim on non-final steps. Either is a no-op
        under `none`. Which pair is correct depends on how much of this NeMo's exporter
        baked into the graph — decided by scripts/parity_parakeet_onnx.py, not by
        argument. Both policies clamp, so a wrong choice degrades text, never crashes."""
        n = max(0, min(n_enc, t_enc))
        start = 0
        if self._drop_policy == "nemo" and self._step > 0 and self._drop_extra > 0:
            start = min(self._drop_extra, n)
            n -= start
        if self._trim_policy == "nemo" and not is_last and self._valid_out_len is not None:
            n = min(n, self._valid_out_len)
        return start, n

    def _decode_frames(self, encoded: np.ndarray, start: int, count: int) -> None:
        """RNN-T greedy over `count` encoder frames. Replaces NeMo's greedy_batch: the
        prediction net advances only on a non-blank emission, and at most
        `max_symbols` symbols are taken per frame (NeMo's max_symbols guard against a
        non-blank loop)."""
        for f in range(start, start + count):
            enc_frame = encoded[:, :, f : f + 1]
            for _ in range(self._max_symbols):
                feeds = {
                    "encoder_outputs": enc_frame,
                    "targets": self._last_token,
                    "target_length": self._target_length,
                }
                for name, state in zip(self._dec_state_in, self._dec_states):
                    feeds[name] = state
                raw = self._dec.run(None, feeds)
                out = dict(zip(self._dec_out_names, raw))
                logits = out.get("outputs", raw[0])  # [B, 1, 1, V]
                token = int(np.argmax(np.asarray(logits).reshape(-1)))
                if token == self._blank_id:
                    break
                self._tokens.append(token)
                self._last_token = np.array([[token]], dtype=np.int32)
                self._dec_states = [out[n] for n in self._dec_state_out]
        if self._tokens:
            self._hyp_text = _detokenize(self._tokens, self._vocab)
