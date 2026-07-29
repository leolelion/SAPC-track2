#!/usr/bin/env python3
"""
parakeet_realtime (120M) cache-aware streaming — SAPC2 Track 2 interface.

======================================================================
VERIFICATION STATUS (read before trusting any number from this file)
----------------------------------------------------------------------
VERIFIED LOCALLY (no NeMo needed, torch/numpy present): the 5-method contract, the
100 ms->model-chunk buffering, callback firing (TTFT driver), reset semantics,
special-token (<EOU>) stripping, and the text-extraction that absorbs the NeMo
list/Hypothesis return quirk. `python3 -m py_compile model.py`, a plain `import`
(NeMo absent), and scripts/smoke_parakeet_ft_wrapper.py (mock-model contract smoke)
all pass because torch/nemo are imported lazily inside __init__.

POD FINDINGS (2026-07-25, NeMo 2.7.3, base=nvidia/parakeet_realtime_eou_120m-v1):
  * FIXED bug#1 — decoding: change_decoding_strategy needs a positional decoding_cfg and
    RNNT streaming crashed on step 2+ ("'dict' object has no attribute 'extend'" in
    rnnt_utils.merge_ on the dict-typed Hypothesis.timestamp). Fix: greedy_batch +
    compute_timestamps=False + preserve_alignments=False. (See __init__.)
  * REWROTE bug#2 — feeding: the old wrapper ran the preprocessor on each isolated 100 ms
    raw chunk -> 0 tokens on real audio for BOTH base and FT. Cache-aware streaming needs
    FEATURE-FRAME feeding: extract features once (unnormalized), step in frames feeding
    [pre_encode_cache | chunk], per-window normalized. Ported from
    CacheAwareStreamingAudioBuffer (streaming_utils.py).
STILL TO CONFIRM ON POD: that this rewrite yields correct text on a real utterance, then
the full run through REAL track2_starting_kit/local_decode.py (both passes) + evaluate.sh
on Dev (house rule: validate-against-real-harness). normalize='NA' handling in _normalize
is a no-op; confirm that matches how the model was trained.
======================================================================

Why this model: research/46 §7 / research/47 — parakeet_realtime is the only zero-shot
model that opened a viable *low-latency* corner (31.8% CER @ 80 ms). This dir deploys the
dysarthric-fine-tuned version. We decode the transducer head at [70,1] (80 ms lookahead).

Required interface (called by local_decode.py):
  __init__()                        — load model once (CPU for Track 2)
  set_partial_callback(fn) -> None  — register fn(text:str); FIRE per emitted step
  reset()             -> None       — fresh encoder cache + decoder state per file
  accept_chunk(np.float32) -> str   — 1600 samples (100 ms); returns running partial
  input_finished()    -> str        — flush tail -> final str

Reference for the streaming calls (port + verify on pod):
  NeMo examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py
"""

import os
import re
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 1600  # 100 ms, as delivered by local_decode.py

_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
_CONFIG = OmegaConf.load(_DIR / "config.yaml")


def _cgroup_cpu_quota():
    """Container CPU quota in cores, or None if unlimited/unreadable. Split out so the
    thread policy can be unit-tested on a box without NeMo (scripts/test_thread_policy.py)."""
    try:
        if os.path.exists("/sys/fs/cgroup/cpu.max"):                  # cgroup v2
            parts = open("/sys/fs/cgroup/cpu.max").read().split()
            if parts and parts[0] != "max":
                return float(parts[0]) / float(parts[1])
        elif os.path.exists("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"):   # cgroup v1
            quota = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
            period = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
            if quota > 0:
                return quota / period
    except Exception:
        pass
    return None


def _resolve_threads(quota, nproc, env=None, config=None):
    """Pick intra-op torch threads. The cgroup quota is a CEILING, not a target.

    Two topologies run this same file and want opposite values:

      accuracy pass  — the organizers' ingestion runs ~20 worker PROCESSES, each loading
        its own Model, on a 24-vCPU worker (confirmed from our submission's
        CPU_DIAGNOSTIC). Threads multiply by 20, so sizing to the quota (24) means 480
        threads on 24 vCPU. Measured under this exact topology (exp_nemotron_speed_002,
        20 workers / 20 CPUs): threads=1 wall RTF 0.029 (34.2x) vs threads=4 wall RTF
        0.123 (8.15x) — 4.2x the wall clock for byte-identical transcripts (40/40 hash
        audit). Wall clock is what the 15000 s/submission budget measures.

      streaming pass — local_decode.py is single-process (one AudioSender + one Decoder
        thread). No contention, so more threads cut per-chunk compute, i.e. TTFT. The
        banked 638 ms TTFT was measured with the old quota-sized cap.

    We cannot distinguish the passes at __init__ (both just construct Model), so this is
    one knob with a chosen default. Default = 1: a blown time budget scores zero, a
    slightly worse TTFT costs a few ms on one Pareto axis.

    >>> VERIFY-ON-POD <<< sweep SAPC2_THREADS=1/2/4 on the STREAMING pass. If per-chunk
    compute at 1 thread exceeds the 100 ms chunk period the stream falls behind real time
    and TTFT/TTLT blow up, forcing a higher default plus a wall-clock recheck.
    """
    env = os.environ if env is None else env
    config = _CONFIG if config is None else config
    rt = config.get("runtime", {}) or {}
    want = int(env.get("SAPC2_THREADS", rt.get("num_threads", 1)))
    nproc = max(1, int(nproc or 1))
    ceiling = int(quota) if (quota and quota >= 1) else nproc
    return max(1, min(want, ceiling, nproc)), want, ceiling


class Model:
    """Cache-aware streaming ASR (parakeet_realtime 120M). See VERIFICATION STATUS above."""

    def _init_runtime_cfg(self):
        """Set every config/env-derived attribute. NeMo-free and torch-free on purpose:
        __init__ calls it, and so does the contract smoke (which bypasses __init__ because
        NeMo is absent locally). Keep ALL such attributes here — when the smoke duplicated
        this list by hand it silently rotted, and the 5-method contract went three commits
        untested while the input-gain and feature-cache attributes were added."""
        self._partial_callback = None

        # Output normalization: strip <EOU>/special tokens the model emits (research/46:
        # 100/123 sweep hyps carried <EOU>). Compiled once; applied on every returned string.
        out_cfg = _CONFIG.get("output", {}) or {}
        self._strip_special = bool(out_cfg.get("strip_special_tokens", True))
        self._special_re = re.compile(str(out_cfg.get("special_token_pattern", r"<[^>]+>")))

        # Causal frozen-scalar input gain (see config.yaml input_gain +
        # exp_parakeet_ft_empties). Env overrides win over config so the pod can A/B and
        # sweep without editing code. Precompute the linear target/cap/floor.
        ig = _CONFIG.get("input_gain", {}) or {}
        _env = os.environ.get
        self._gain_on = _env("SAPC2_INPUT_GAIN", "on" if ig.get("enabled", True) else "off").lower() != "off"
        _tgt_db = float(_env("SAPC2_TARGET_DBFS", ig.get("target_dbfs", -25.0)))
        _cap_db = float(_env("SAPC2_GAIN_CAP_DB", ig.get("cap_db", 20.0)))
        _flr_db = float(_env("SAPC2_RMS_FLOOR_DBFS", ig.get("floor_dbfs", -45.0)))
        self._gain_target_rms = 10.0 ** (_tgt_db / 20.0)   # -25 dBFS -> ~0.0562
        self._gain_cap = 10.0 ** (_cap_db / 20.0)           # +20 dB   -> 10.0x
        self._gain_rms_floor = 10.0 ** (_flr_db / 20.0)     # -45 dBFS -> ~0.0056
        self._gain_dbg = (_tgt_db, _cap_db, _flr_db)

        # Incremental feature cache (Fix 2): the OLD path re-ran the mel STFT over the WHOLE
        # raw buffer every accept_chunk -> O(N^2) over an utterance (invisible in the timed
        # streaming pass, but blew the untimed accuracy pass to ~17-40 min/425u -> a 15000 s
        # budget risk on full Test). With the input gain frozen per file, scaled audio is
        # deterministic, so already-computed frames are STABLE and can be cached; only the
        # growing tail (last few frames touch the STFT boundary) is recomputed, with left
        # context, each call. Verified on-pod byte-identical vs full extraction.
        # Env-toggle for A/B + fallback: SAPC2_FEAT_CACHE=on(default)|off.
        self._feat_cache_on = _env("SAPC2_FEAT_CACHE", "on").lower() != "off"

    def __init__(self):
        # Lazy imports keep this file import-safe on a box without NeMo/torch,
        # so the contract/buffering/strip logic can be inspected and import-checked locally.
        import torch

        # Intra-op thread policy. See _resolve_threads() for the full rationale and the
        # 20-worker measurement it is derived from.
        try:
            _q = _cgroup_cpu_quota()
            _n, _want, _ceil = _resolve_threads(_q, os.cpu_count())
            torch.set_num_threads(_n)
            try:
                torch.set_num_interop_threads(1)
            except Exception:
                pass  # already initialized in this process
            print(
                f"[parakeet_ft] torch threads={torch.get_num_threads()} "
                f"(requested={_want} ceiling={_ceil} cgroup_quota={_q} nproc={os.cpu_count()})"
            )
        except Exception as _e:
            print(f"[parakeet_ft] thread policy skipped: {_e}")

        import nemo.collections.asr as nemo_asr

        self._torch = torch
        self._device = torch.device("cpu")  # Track 2 = CPU-only
        self._init_runtime_cfg()

        nemo_path = _DIR / _CONFIG.weights.nemo_file
        print(f"[parakeet_ft] loading {nemo_path} (CPU) …")
        if nemo_path.exists():
            # Generic ASRModel.restore_from auto-dispatches to the right subclass
            # (RNNT vs TDT vs Hybrid) — do not pin a specific class here.
            self.model = nemo_asr.models.ASRModel.restore_from(
                str(nemo_path), map_location=self._device
            )
        else:
            # Fallback for the benchmark pod if setup.sh's save step was skipped.
            print(f"[parakeet_ft] local .nemo missing; pulling {_CONFIG.weights.model_name}")
            self.model = nemo_asr.models.ASRModel.from_pretrained(
                model_name=_CONFIG.weights.model_name, map_location=self._device
            )
        self.model = self.model.to(self._device).eval()

        # A cache-aware model exposes conformer_stream_step. If it does not, this
        # checkpoint is NOT streaming — fail loud, never buffer-and-rerun (forbidden).
        if not hasattr(self.model, "conformer_stream_step"):
            raise RuntimeError(
                f"{_CONFIG.weights.model_name} has no conformer_stream_step — not a "
                "cache-aware streaming model. Buffer-and-rerun is forbidden fake streaming."
            )

        # Streaming decoding config. VERIFIED-ON-POD (NeMo 2.7.3): cache-aware RNNT
        # streaming crashes on step 2+ because merging partial hypotheses across steps
        # calls Hypothesis.timestamp.extend(), but timestamp is a dict in this NeMo
        # (rnnt_utils.merge_ -> "'dict' object has no attribute 'extend'"). Disabling
        # timestamp + alignment computation avoids that merge path entirely; we do not
        # need model timestamps (latency = callback wall-clock in local_decode.py).
        # Keep greedy_batch (fast, the low-latency path). Empirically: this survives
        # multi-step streaming; loop_labels=False breaks partial_hypotheses support.
        # NOTE: this NeMo's change_decoding_strategy takes a positional decoding_cfg and
        # rejects the older decoder_type= kwarg (which previously no-op'd via except).
        if hasattr(self.model, "change_decoding_strategy"):
            import copy as _copy
            from omegaconf import open_dict as _open_dict
            _dc = _copy.deepcopy(self.model.cfg.decoding)
            with _open_dict(_dc):
                _dc.strategy = "greedy_batch"
                _dc.compute_timestamps = False
                _dc.preserve_alignments = False
                if _dc.get("greedy") is not None:
                    _dc.greedy.preserve_alignments = False
            self.model.change_decoding_strategy(_dc)

        # >>> VERIFY-ON-POD <<<  (block 1 of 2: streaming-param setup)
        att = list(_CONFIG.encoder.att_context_size)
        if hasattr(self.model.encoder, "set_default_att_context_size"):
            self.model.encoder.set_default_att_context_size(att)
        if hasattr(self.model.encoder, "setup_streaming_params"):
            self.model.encoder.setup_streaming_params()

        # Feature-frame streaming geometry + a raw feature extractor, ported faithfully
        # from NeMo's CacheAwareStreamingAudioBuffer. The model steps in FEATURE FRAMES,
        # not raw-audio chunks: features are extracted once (unnormalized), then each step
        # feeds [pre_encode_cache frames | chunk frames]. The previous approach — running
        # the model preprocessor on each isolated 100 ms raw chunk — produced ZERO tokens
        # on real audio for BOTH base and FT (verified on pod 2026-07-25).
        self._setup_streaming()

        self.reset()
        print(
            f"[parakeet_ft] ready (att={att}, chunk={self._chunk_pair} shift={self._shift_pair} "
            f"pre_cache={self._pre_cache_pair} drop_extra={self._drop_extra} "
            f"norm={self._norm_type} strip_special={self._strip_special} "
            f"input_gain={'on' if self._gain_on else 'off'}"
            f"(tgt/cap/floor_db={self._gain_dbg}))"
        )

    # ------------------------------------------------------------------
    def _setup_streaming(self) -> None:
        """Read streaming_cfg (feature-frame geometry) and build an unnormalized feature
        extractor, mirroring CacheAwareStreamingAudioBuffer. chunk/shift/pre_encode_cache
        are [first_step, later_steps] pairs; we pick per step. VERIFIED values for
        parakeet_realtime_eou_120m @[70,1]: chunk=[9,16] shift=[9,16] pre_cache=[0,9]
        drop_extra_pre_encoded=2, window_stride=0.01, features=128."""
        import copy
        from omegaconf import OmegaConf, open_dict

        sc = getattr(self.model.encoder, "streaming_cfg", None)

        def _pair(v, default):
            if v is None:
                return [int(default), int(default)]
            if isinstance(v, (list, tuple)):
                return [int(v[0]), int(v[-1])]
            return [int(v), int(v)]

        self._chunk_pair = _pair(getattr(sc, "chunk_size", None), CHUNK_SAMPLES // 160)
        self._shift_pair = _pair(getattr(sc, "shift_size", None), self._chunk_pair[0])
        self._pre_cache_pair = _pair(getattr(sc, "pre_encode_cache_size", None), 0)
        self._drop_extra = int(getattr(sc, "drop_extra_pre_encoded", 0) or 0)
        # sampling_frames: min frames to produce one output after subsampling; used to
        # drop a too-short final chunk (mirrors the buffer's sampling_frames guard).
        sf = None
        pe = getattr(self.model.encoder, "pre_encode", None)
        if pe is not None and hasattr(pe, "get_sampling_frames"):
            sf = pe.get_sampling_frames()
        self._sampling_pair = _pair(sf, 1)

        # Unnormalized feature extractor (dither/pad off); normalization is applied
        # per-window (online) in _normalize so frames stay stable across re-extraction.
        self._norm_type = self.model.cfg.preprocessor.get("normalize", "per_feature")
        cfg = copy.deepcopy(self.model._cfg)
        OmegaConf.set_struct(cfg.preprocessor, False)
        with open_dict(cfg.preprocessor):
            cfg.preprocessor.dither = 0.0
            cfg.preprocessor.pad_to = 0
            cfg.preprocessor.normalize = "None"
        self._raw_preprocessor = self.model.from_config_dict(cfg.preprocessor).to(
            self._device
        )
        self._feat_in = int(getattr(self.model.encoder, "_feat_in", 128))

        # Feature-cache geometry (Fix 2). hop = window_stride in samples. n_fft bounds the
        # STFT boundary so <= ceil(n_fft/2/hop) tail frames are unstable when audio is appended;
        # MARGIN recomputes them every call. LCTX (a hop multiple >= n_fft) gives the tail
        # recompute real left context so kept frames are bit-identical to full extraction, and
        # keeping it a hop multiple makes the splice alignment `drop = MARGIN` exact.
        pp = self.model.cfg.preprocessor
        sr = int(pp.get("sample_rate", 16000))
        self._feat_hop = max(1, int(round(float(pp.get("window_stride", 0.01)) * sr)))  # 160
        n_fft = int(pp.get("n_fft", 512))
        import math as _math
        self._feat_margin = int(_math.ceil(n_fft / 2 / self._feat_hop)) + 2   # 2 + 2 = 4
        lctx_frames = int(_math.ceil(n_fft / self._feat_hop)) + 1             # ceil(512/160)+1 = 5
        self._feat_lctx = lctx_frames * self._feat_hop                        # 5*160 = 800 samples

    # ------------------------------------------------------------------
    def _compute_gain(self, est_window: np.ndarray) -> float:
        """Frozen boost-only input gain from the first ~100 ms. Returns a scalar in
        [1.0, cap]. A near-silent estimate window (< floor) -> 1.0 so we never amplify
        leading silence/noise by +cap dB. Boost-only: we never attenuate audio that the
        model already handles (only the quiet severe tail needs help)."""
        if not self._gain_on:
            return 1.0
        x = est_window.astype(np.float64)
        rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
        if rms < self._gain_rms_floor:
            return 1.0
        g = self._gain_target_rms / rms
        return float(min(max(g, 1.0), self._gain_cap))

    def _raw_to_feats(self, raw: np.ndarray):
        """Run the unnormalized mel extractor over a 1-D raw segment -> [1, F, t]."""
        torch = self._torch
        wav = torch.tensor(raw, dtype=torch.float32, device=self._device).unsqueeze(0)
        wl = torch.tensor([raw.shape[0]], dtype=torch.int64, device=self._device)
        feats, _ = self._raw_preprocessor(input_signal=wav, length=wl)
        return feats

    def _extract_features(self, raw: np.ndarray):
        """Unnormalized log-mel features for the raw buffer -> [1, F, T].
        dither=0 makes this deterministic, so a given frame index is STABLE as more audio
        arrives (mel is local). Two invariants let us CACHE instead of recomputing the whole
        buffer each call (Fix 2, O(N^2)->O(N)): (1) the per-file input gain is a frozen scalar
        (applied here) so scaled audio never changes retroactively; (2) with hop/n_fft fixed,
        only the last MARGIN frames touch the STFT boundary, so we recompute just the tail
        (with LCTX real left context) and splice it onto the cached stable prefix. Bit-exact
        vs full extraction is gated numerically on-pod before this path is trusted."""
        torch = self._torch
        if raw.shape[0] < 1:
            return torch.zeros(1, self._feat_in, 0, device=self._device)
        if self._gain is None:
            self._gain = self._compute_gain(raw[:CHUNK_SAMPLES])
        scaled = raw if self._gain == 1.0 else np.clip(raw * self._gain, -1.0, 1.0).astype(np.float32)

        if not self._feat_cache_on or self._feat_cache is None:
            feats = self._raw_to_feats(scaled)
            if self._feat_cache_on:
                self._feat_cache = feats
            return feats

        # Incremental: trust cached frames [0:stable], recompute [stable:] with left context.
        HOP, MARGIN, LCTX = self._feat_hop, self._feat_margin, self._feat_lctx
        Tc = self._feat_cache.shape[-1]
        stable = max(0, Tc - MARGIN)
        start = max(0, stable * HOP - LCTX)          # LCTX and stable*HOP are hop multiples
        seg_feats = self._raw_to_feats(scaled[start:])
        drop = stable - (start // HOP)               # seg frames preceding global `stable`
        drop = max(0, min(drop, seg_feats.shape[-1]))
        feats = torch.cat([self._feat_cache[:, :, :stable], seg_feats[:, :, drop:]], dim=-1)
        self._feat_cache = feats
        return feats

    # ------------------------------------------------------------------
    def _normalize(self, fed):
        """Per-window feature normalization (online streaming). No-op when the model
        uses no global normalize (e.g. normalize='NA'/None) — then the raw features are
        already what the encoder expects."""
        nt = self._norm_type
        if nt in (None, "None", "NA", "na", ""):
            return fed
        try:
            from nemo.collections.asr.parts.preprocessing.features import normalize_batch
            torch = self._torch
            seq_len = torch.tensor([fed.shape[-1]] * fed.shape[0], device=self._device)
            fed, _, _ = normalize_batch(x=fed, seq_len=seq_len, normalize_type=nt)
        except Exception:
            pass  # if normalize_batch rejects the type, feed unnormalized
        return fed

    # ------------------------------------------------------------------
    def set_partial_callback(self, callback) -> None:
        self._partial_callback = callback

    def reset(self) -> None:
        """Fresh encoder cache + decoder state + feature buffer for a new file."""
        self._raw_buf = np.zeros(0, dtype=np.float32)
        self._gain = None    # per-file frozen input gain; set once from the first ~100 ms
        self._feat_cache = None  # incremental feature cache (Fix 2); rebuilt per file
        self._feat_idx = 0   # feature-frame pointer (== buffer_idx in the NeMo buffer)
        self._step = 0       # step counter (drives first-chunk sizing + drop_extra)
        self._hyp_text = ""
        self._prev_hypotheses = None
        self._prev_pred_out = None
        (self._cache_ch, self._cache_t, self._cache_ch_len) = (
            self.model.encoder.get_initial_cache_state(batch_size=1)
        )

    # ------------------------------------------------------------------
    def accept_chunk(self, audio_chunk: np.ndarray) -> str:
        """Accumulate 100 ms of raw audio; emit one model step per full feature-frame
        chunk now available. Partials fire via the callback (drives TTFT). The trailing
        remainder (< one chunk) waits for more audio or for input_finished."""
        self._raw_buf = np.concatenate(
            [self._raw_buf, np.asarray(audio_chunk, dtype=np.float32)]
        )
        feats = self._extract_features(self._raw_buf)  # [1, F, T], unnormalized, stable
        total = feats.shape[-1]
        while True:
            chunk_size = self._chunk_pair[0] if self._feat_idx == 0 else self._chunk_pair[1]
            if self._feat_idx + chunk_size > total:
                break
            text = self._stream_step(feats, total, is_last=False)
            if text and self._partial_callback is not None:
                self._partial_callback(text)  # already special-stripped
        return self._clean(self._hyp_text)

    def input_finished(self) -> str:
        """Flush any newly-complete chunks, then the final partial remainder (if it is
        large enough to yield an output after subsampling), and return the final hyp."""
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
    def _clean(self, text: str) -> str:
        """Strip <EOU>/special tokens + normalize whitespace on OUTPUT only.
        Never emit a literal special token to the scorer (cf. the `unk` assertion
        that fails a whole submission — brief §3.10)."""
        if not text:
            return ""
        if self._strip_special:
            text = self._special_re.sub(" ", text)
        return " ".join(text.split())

    # ------------------------------------------------------------------
    def _stream_step(self, feats, total_frames: int, is_last: bool) -> str:
        """ONE cache-aware step over feature frames [pre_encode_cache | chunk], mirroring
        CacheAwareStreamingAudioBuffer.__iter__. Advances self._feat_idx by shift_size.

        conformer_stream_step returns a 6-tuple:
        (pred_out, transcribed, cache_ch, cache_t, cache_ch_len, best_hyp). We feed
        best_hyp back as previous_hypotheses and pred_out as previous_pred_out (the exact
        pattern from NeMo's streaming example). `_extract_text` absorbs the Hypothesis/
        list return quirk; decoder SOS is handled inside NeMo (no hand-rolled greedy)."""
        torch = self._torch
        first = self._feat_idx == 0
        chunk_size = self._chunk_pair[0] if first else self._chunk_pair[1]
        shift_size = self._shift_pair[0] if first else self._shift_pair[1]
        pre_cache = self._pre_cache_pair[0] if first else self._pre_cache_pair[1]

        chunk = feats[:, :, self._feat_idx : self._feat_idx + chunk_size]

        # pre-encode cache: prior frames prepended so the pre-encoder has left context
        # (none on the first step; zero-padded if fewer than pre_cache prior frames exist)
        zeros_pads = None
        if first:
            cache = feats[:, :, 0:0]  # empty (pre_cache_pair[0] == 0)
        else:
            start = max(0, self._feat_idx - pre_cache)
            cache = feats[:, :, start : self._feat_idx]
            if cache.shape[-1] < pre_cache:
                zeros_pads = torch.zeros(
                    (chunk.shape[0], chunk.shape[1], pre_cache - cache.shape[-1]),
                    dtype=chunk.dtype, device=self._device,
                )
        added_len = cache.shape[-1]
        fed = torch.cat((cache, chunk), dim=-1)

        fed = self._normalize(fed)  # per-window online normalization (no-op if norm off)

        if zeros_pads is not None:
            fed = torch.cat((zeros_pads, fed), dim=-1)
            added_len += zeros_pads.shape[-1]

        # valid (non-padding) length = real frames remaining + prepended cache frames
        valid = max(0, total_frames - self._feat_idx) + added_len
        length = min(valid, fed.shape[-1])
        length_t = torch.tensor([length], dtype=torch.int64, device=self._device)

        # first step uses no caching -> drop nothing; later steps drop the pre-encoded
        # cache frames (calc_drop_extra_pre_encoded: 0 at step 0, else drop_extra).
        drop = 0 if self._step == 0 else self._drop_extra

        out = self.model.conformer_stream_step(
            processed_signal=fed,
            processed_signal_length=length_t,
            cache_last_channel=self._cache_ch,
            cache_last_time=self._cache_t,
            cache_last_channel_len=self._cache_ch_len,
            keep_all_outputs=is_last,
            previous_hypotheses=self._prev_hypotheses,
            previous_pred_out=self._prev_pred_out,
            drop_extra_pre_encoded=drop,
            return_transcription=True,
        )
        (
            pred_out,
            transcribed,
            self._cache_ch,
            self._cache_t,
            self._cache_ch_len,
            prev_hyps,
        ) = out[:6]
        self._prev_pred_out = pred_out
        self._prev_hypotheses = prev_hyps

        self._feat_idx += shift_size
        self._step += 1

        text = self._extract_text(transcribed)
        if text:
            self._hyp_text = text
        return self._clean(self._hyp_text)

    # ------------------------------------------------------------------
    @staticmethod
    def _extract_text(transcribed) -> str:
        """Normalize NeMo's transcription return to a plain str.

        Handles: plain str; a Hypothesis object (.text); a list of either
        (batch=1 -> take [0]); None. Absorbs the NeMo list-return quirk
        (docs/results/parakeet_comparison.md). Special-token stripping happens
        later in _clean, not here (keep raw hyp intact for internal state)."""
        if transcribed is None:
            return ""
        item = transcribed
        if isinstance(item, (list, tuple)):
            if not item:
                return ""
            item = item[0]
            if isinstance(item, (list, tuple)):  # nested [[hyp]]
                item = item[0] if item else ""
        if isinstance(item, str):
            return item
        text = getattr(item, "text", None)
        return text if isinstance(text, str) else ""
