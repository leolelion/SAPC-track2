#!/usr/bin/env python3
"""
Nemotron Streaming ASR (ONNX, int8) — SAPC2 Track 2 Submission
==============================================================

Wraps the danielbodart prebuilt int8 ONNX export of
nvidia/nemotron-speech-streaming-en-0.6b in the SAPC2 5-method streaming
interface. Cache-aware streaming FastConformer-RNNT at att_context_size
=[70, 6] (560 ms model chunks).

Layout:
  weights/
    encoder_model.onnx              int8 (danielbodart prebuilt)
    encoder_model.onnx.data         external weights
    decoder_model.onnx              fp32 (decoder + joint combined)
    decoder_model.onnx.data         external weights
    tokens.txt                      sentencepiece "<piece> <id>" lines
  config.yaml                       runtime knobs

Threading: SAPC2 spec guarantees all 5 methods are called from the
Decoder thread only. The partial_callback runs from accept_chunk only.
No locking required.

RTF instrumentation: self.compute_time_sec accumulates wall time inside
accept_chunk + input_finished, excluding the SAPC2 audio-arrival sleeps.
This is the SAPC2 RTF denominator. Reset per utterance in reset().
"""

# =====================================================================
# Section 1: SETUP_VERIFY probe + base-Python imports (NO VENV, NO NEMO)
#
# Nemo-free variant. Built on top of the v5 / nemotron_streaming_novenv
# kit with one further change: drop NeMo entirely and replace its mel
# preprocessor with a hand-rolled torch + numpy implementation (see
# MinimalMelPreprocessor below). Rationale: Codabench ingestion
# crashed with a numpy.dtype binary ABI mismatch while importing
# nemo → lightning → torch._dynamo. v5 removed our explicit
# `numpy<2` pin, but transitive deps (numba via librosa via nemo)
# still forced numpy down to 1.26.4 while the lightning wheel was
# compiled against numpy 2.x. Without NeMo, neither lightning nor
# numba nor librosa is pulled in, so the ABI mismatch cannot occur.
#
# The MinimalMelPreprocessor was validated byte-equivalent to
# nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor
# (with the exact Nemotron checkpoint args) at max abs diff 1.9e-6
# on three Dev_streaming utts — well under the 1e-4 acceptance
# threshold from scripts/audit/reference_submission_diff.md.
#
# SETUP_VERIFY no longer imports nemo — only torch + onnxruntime.
# If the eval VM ingestion log shows [SETUP_VERIFY] imports OK, the
# numpy ABI failure mode that bit v1/v2/v3 cannot occur.
# =====================================================================
import os
import sys

print(
    f"[SETUP_VERIFY] starting; pid={os.getpid()}; "
    f"python={sys.version.split()[0]}; cwd={os.getcwd()}",
    flush=True,
)
try:
    import torch as _verify_torch
    import onnxruntime as _verify_ort
    print(
        f"[SETUP_VERIFY] imports OK: "
        f"torch={_verify_torch.__version__} ort={_verify_ort.__version__}",
        flush=True,
    )
    del _verify_torch, _verify_ort
except Exception:
    import traceback
    print("[SETUP_VERIFY] IMPORT FAILED — traceback follows:", flush=True)
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    raise

_THREADS = int(os.environ.get("SAPC2_THREADS", "4"))
os.environ.setdefault("OMP_NUM_THREADS", str(_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(_THREADS))

_DIR = os.path.dirname(os.path.abspath(__file__))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(_THREADS)
torch.set_num_interop_threads(1)

import onnxruntime as ort  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

# =====================================================================
# Section 2: Constants for att_context_size=[70, 6] / chunk_mel=56
# These match danielbodart's config.json. If you ever switch
# att_context_size, regenerate the ONNX and update these.
# =====================================================================
SAMPLE_RATE = 16000
N_MELS = 128
HOP_SAMPLES = 160       # 10 ms at 16 kHz
WINDOW_SAMPLES = 400    # 25 ms at 16 kHz

CHUNK_NEW = 56          # mel frames per encoder step
CACHE_FRAMES = 9        # pre-encode cache mel frames
ENC_INPUT_FRAMES = CHUNK_NEW + CACHE_FRAMES  # 65

ENC_LAYERS = 24
ENC_DIM = 1024
LAST_CHANNEL_CACHE = 70
LAST_TIME_FRAMES = 8

DEC_LAYERS = 2
DEC_HIDDEN = 640
VOCAB_SIZE = 1025       # 1024 + blank
BLANK_ID = 1024
MAX_SYMBOLS_PER_FRAME = 10


# =====================================================================
# Section 3: Tokenizer (sentencepiece-style "<piece> <id>" file)
# =====================================================================
def _load_tokens(path: str):
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
    return [vocab[i] for i in range(max(vocab) + 1)]


def _detokenize(tokens, vocab):
    pieces = [vocab[t] for t in tokens if 0 <= t < len(vocab)]
    text = "".join(pieces).replace("▁", " ").strip()
    return text


# =====================================================================
# Section 4: Mel preprocessor — pure torch + numpy, NeMo-equivalent
#
# Drop-in replacement for nemo.collections.asr.modules.
# AudioToMelSpectrogramPreprocessor as configured for Nemotron. We
# replicate FilterbankFeatures (the default backend of NeMo's
# preprocessor, in eval mode) with these hardcoded args:
#
#   sample_rate=16000, window_size=0.025, window_stride=0.010,
#   window="hann", features=128, n_fft=512, dither=0.0, pad_to=0,
#   normalize="NA", frame_splicing=1, log=True, pad_value=0.0
#
# Plus the NeMo FilterbankFeatures internal defaults that the
# AudioToMelSpectrogramPreprocessor wrapper does not let us override:
#
#   preemph=0.97, mag_power=2.0, log_zero_guard_value=2**-24,
#   mel_norm="slaney"  (i.e. librosa.filters.mel(htk=False, norm="slaney"))
#
# Steps in forward (eval mode):
#   1. preemph: x[t] -= 0.97 * x[t-1], then mask beyond seq_len to 0.
#   2. torch.stft(center=True, pad_mode="constant", window=hann(periodic=False))
#   3. power = (sqrt(re^2 + im^2)) ** 2     (mag_power=2.0)
#   4. mel = librosa-slaney filterbank @ power
#   5. log(mel + 2**-24)
#   6. mask frames beyond seq_len_out with pad_value=0
#
# Byte-equivalence vs NeMo (validated in
# /tmp/preproc_dev/test_equiv.py on 3 Dev_streaming utts):
#   max abs diff = 1.9e-6 (well under the 1e-4 acceptance threshold).
# The remaining error is fp32 rounding noise. The filterbank is
# computed in fp64 numpy at module load time to match librosa's
# precision; torchaudio's default fp32 filterbank diverged in the
# high mel bins (max abs diff ~1.4-1.7e-4, just over threshold).
# =====================================================================
def _slaney_hz_to_mel(hz):
    """Slaney mel scale (htk=False). Mirrors librosa.hz_to_mel."""
    hz = np.asarray(hz, dtype=np.float64)
    f_sp = 200.0 / 3
    mels = hz / f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    return np.where(
        hz >= min_log_hz, min_log_mel + np.log(hz / min_log_hz) / logstep, mels
    )


def _slaney_mel_to_hz(mels):
    """Inverse of _slaney_hz_to_mel."""
    mels = np.asarray(mels, dtype=np.float64)
    f_sp = 200.0 / 3
    hz = f_sp * mels
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    return np.where(
        mels >= min_log_mel, min_log_hz * np.exp(logstep * (mels - min_log_mel)), hz
    )


def _librosa_mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    """Slaney-normalized triangular mel filterbank in fp64, matching
    librosa.filters.mel(htk=False, norm='slaney'). Returns (n_mels, n_freqs)."""
    n_freqs = n_fft // 2 + 1
    fft_freqs = np.linspace(0.0, sr / 2.0, n_freqs, dtype=np.float64)
    mel_pts = np.linspace(
        _slaney_hz_to_mel(fmin), _slaney_hz_to_mel(fmax), n_mels + 2, dtype=np.float64
    )
    hz_pts = _slaney_mel_to_hz(mel_pts)
    fb = np.zeros((n_mels, n_freqs), dtype=np.float64)
    for m in range(n_mels):
        f_left, f_center, f_right = hz_pts[m], hz_pts[m + 1], hz_pts[m + 2]
        rising = (fft_freqs - f_left) / (f_center - f_left)
        falling = (f_right - fft_freqs) / (f_right - f_center)
        fb[m] = np.maximum(0.0, np.minimum(rising, falling))
    enorm = 2.0 / (hz_pts[2 : n_mels + 2] - hz_pts[:n_mels])
    fb *= enorm[:, np.newaxis]
    return fb


class MinimalMelPreprocessor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._n_fft = 512
        self._win_length = 400
        self._hop_length = 160
        self._preemph = 0.97
        self._log_zero_guard_value = 2.0 ** -24
        self._pad_value = 0.0

        window = torch.hann_window(self._win_length, periodic=False)
        self.register_buffer("_window", window)

        fb_np = _librosa_mel_filterbank(
            sr=16000, n_fft=self._n_fft, n_mels=N_MELS, fmin=0.0, fmax=8000.0
        )
        self.register_buffer("_fb", torch.from_numpy(fb_np.astype(np.float32)))

    def _get_seq_len(self, seq_len):
        # center=True + n_fft even: pad_amount = n_fft, output_len = floor(L / hop).
        return torch.floor_divide(seq_len, self._hop_length).to(torch.long)

    @torch.no_grad()
    def forward(self, input_signal, length):
        x = input_signal
        seq_len_time = length
        seq_len_out_raw = self._get_seq_len(length)
        seq_len_out = torch.where(
            length == 0, torch.zeros_like(seq_len_out_raw), seq_len_out_raw
        )

        # Preemphasis + mask beyond seq_len.
        timemask = torch.arange(x.shape[1], device=x.device).unsqueeze(0) < seq_len_time.unsqueeze(1)
        x = torch.cat((x[:, :1], x[:, 1:] - self._preemph * x[:, :-1]), dim=1)
        x = x.masked_fill(~timemask, 0.0)

        # STFT.
        stft_out = torch.stft(
            x,
            n_fft=self._n_fft,
            hop_length=self._hop_length,
            win_length=self._win_length,
            center=True,
            window=self._window.to(dtype=x.dtype, device=x.device),
            return_complex=True,
            pad_mode="constant",
        )

        # Power spectrum (NeMo: sqrt(re^2+im^2) then ^2).
        mag = torch.sqrt(stft_out.real.pow(2) + stft_out.imag.pow(2))
        power = mag.pow(2.0)

        # Mel filterbank.
        mel = torch.matmul(self._fb.to(power.dtype), power)

        # Log with zero-guard.
        mel = torch.log(mel + self._log_zero_guard_value)

        # Mask beyond seq_len_out.
        max_len = mel.size(-1)
        mask = torch.arange(max_len, device=mel.device).repeat(mel.size(0), 1) >= seq_len_out.unsqueeze(1)
        mel = mel.masked_fill(mask.unsqueeze(1), self._pad_value)

        return mel, seq_len_out


def _load_preprocessor():
    """Build the deterministic Nemotron mel preprocessor (pure torch, no NeMo)."""
    pp = MinimalMelPreprocessor()
    pp.eval()
    return pp


# =====================================================================
# Section 5: ORT session builder
# =====================================================================
def _make_session(path: str, num_threads: int) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = num_threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


# =====================================================================
# Section 6: Model — SAPC2 5-method streaming interface
# =====================================================================
class Model:
    """Streaming ASR model wrapping the int8 Nemotron ONNX export.

    Lifecycle (per SAPC2 ingestion):
      m = Model()                       # load once
      m.set_partial_callback(fn)        # once per evaluation pass
      for each audio file:
          m.reset()
          for chunk in audio_chunks_100ms:
              m.accept_chunk(chunk)
          final_text = m.input_finished()
    """

    # ----- Construction -----
    def __init__(self):
        self._log_diagnostic_info()

        cfg_path = os.path.join(_DIR, "config.yaml")
        if os.path.exists(cfg_path):
            cfg = OmegaConf.load(cfg_path)
        else:
            cfg = OmegaConf.create({})

        weights_dir = os.path.join(_DIR, OmegaConf.select(cfg, "weights.dir", default="weights"))
        encoder_path = os.path.join(weights_dir, OmegaConf.select(cfg, "weights.encoder", default="encoder_model.onnx"))
        decoder_path = os.path.join(weights_dir, OmegaConf.select(cfg, "weights.decoder", default="decoder_model.onnx"))
        tokens_path = os.path.join(weights_dir, OmegaConf.select(cfg, "weights.tokens", default="tokens.txt"))
        num_threads = int(OmegaConf.select(cfg, "model.num_threads", default=_THREADS))

        print(f"[nemotron_streaming] loading encoder: {encoder_path}", flush=True)
        self._enc_sess = _make_session(encoder_path, num_threads)
        print(f"[nemotron_streaming] loading decoder: {decoder_path}", flush=True)
        self._dec_sess = _make_session(decoder_path, num_threads)
        self._enc_out_names = [o.name for o in self._enc_sess.get_outputs()]
        self._dec_out_names = [o.name for o in self._dec_sess.get_outputs()]

        print(f"[nemotron_streaming] loading tokens: {tokens_path}", flush=True)
        self._vocab = _load_tokens(tokens_path)

        print(f"[nemotron_streaming] building preprocessor (torch+numpy, CPU) ...", flush=True)
        self._preprocessor = _load_preprocessor()

        self._partial_callback = lambda _t: None
        self._reset_state()

        # ── Memory-usage instrumentation (telemetry; eval-VM diagnostic) ──
        # Per-utt counter and atexit hook so the FINAL input_finished is
        # always covered, even if it doesn't land on a periodic milestone.
        self._utt_counter = 0
        import atexit
        atexit.register(self._log_mem, "atexit_final")

        print(
            f"[nemotron_streaming] ready (threads={num_threads}, "
            f"chunk={CHUNK_NEW} mel, cache={CACHE_FRAMES} mel, blank={BLANK_ID})",
            flush=True,
        )
        self._log_mem("init_done")

    # ----- SAPC2 interface -----
    def set_partial_callback(self, callback) -> None:
        """Register callback for partial results: callback(text: str)."""
        self._partial_callback = callback

    def reset(self) -> None:
        """Reset state for a new audio file."""
        self._reset_state()

    # ----- Memory telemetry -----
    @staticmethod
    def _peak_rss_mb() -> float:
        """Peak RSS in MB. On Linux ru_maxrss is in KB; on macOS in bytes.
        Codabench runs Linux so KB is the expected interpretation."""
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        except Exception:
            return -1.0

    def _log_mem(self, event: str) -> None:
        print(
            f"[MEM_DIAGNOSTIC] event={event} "
            f"utt={getattr(self, '_utt_counter', 0)} "
            f"peak_rss_mb={self._peak_rss_mb():.1f}",
            flush=True,
        )

    def accept_chunk(self, audio_chunk: np.ndarray) -> str:
        """Feed one 100ms (1600 sample) float32 mono 16kHz chunk."""
        import time
        t0 = time.perf_counter()
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32, copy=False)
        self._raw_chunks.append(audio_chunk)
        self._total_samples += len(audio_chunk)
        self._run_steps(is_final=False)
        self.compute_time_sec += (time.perf_counter() - t0)
        return self._last_emitted

    def input_finished(self) -> str:
        """Signal end of audio; flush remaining chunks and drain encoder."""
        import time
        t0 = time.perf_counter()
        self._run_steps(is_final=True)
        self._run_drain()
        self.compute_time_sec += (time.perf_counter() - t0)
        self._utt_counter += 1
        # Periodic milestone every 100 utts, plus the first utt for an early
        # signal. atexit covers the very last utt regardless of where it lands.
        if self._utt_counter == 1 or self._utt_counter % 100 == 0:
            self._log_mem(f"utt_done_{self._utt_counter}")
        return self._last_emitted

    # ----- CPU/host diagnostic (logged once at model load) -----
    @staticmethod
    def _log_diagnostic_info() -> None:
        """Dump host CPU/memory + a small matmul benchmark.

        Goal: definitively learn the eval VM's CPU spec from a real
        submission. ~2 second overhead at model load. Output goes to
        both stdout (ingestion captures it) and /tmp/cpu_diagnostic.json
        if the FS is writable.
        """
        import json
        import platform
        import subprocess
        import time
        info: dict = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count_os": os.cpu_count(),
        }
        try:
            info["cpu_count_sched"] = len(os.sched_getaffinity(0))
        except Exception as e:
            info["cpu_count_sched_error"] = str(e)
        try:
            info["lscpu"] = subprocess.check_output(["lscpu"], text=True, timeout=5)
        except Exception as e:
            info["lscpu_error"] = str(e)
        try:
            with open("/proc/cpuinfo") as f:
                info["cpuinfo_head"] = "".join(f.readlines()[:50])
        except Exception as e:
            info["cpuinfo_error"] = str(e)
        try:
            info["meminfo"] = subprocess.check_output(["free", "-h"], text=True, timeout=5)
        except Exception as e:
            info["meminfo_error"] = str(e)
        try:
            import numpy as _np
            x = _np.random.randn(1024, 1024).astype(_np.float32)
            t0 = time.perf_counter()
            for _ in range(20):
                _ = x @ x
            info["matmul_1024_20iter_ms"] = (time.perf_counter() - t0) * 1000
        except Exception as e:
            info["matmul_error"] = str(e)
        info["uptime"] = None
        try:
            with open("/proc/uptime") as f:
                info["uptime"] = f.read().strip()
        except Exception:
            pass
        try:
            info["loadavg"] = open("/proc/loadavg").read().strip()
        except Exception:
            pass
        try:
            with open("/tmp/cpu_diagnostic.json", "w") as f:
                json.dump(info, f, indent=2, default=str)
        except Exception as e:
            info["diag_write_error"] = str(e)
        print("[CPU_DIAGNOSTIC]", json.dumps(info, default=str), flush=True)

    # ----- Internal state -----
    def _reset_state(self):
        self._raw_chunks = []
        self._total_samples = 0
        self._buffer_idx = 0            # current position in mel frames
        self._step_num = 0
        self._emitted_tokens = []
        self._last_emitted = ""
        self._cached_mel = None
        self._cached_feat_len = 0
        self._cached_n_samples = 0
        self._drain_done = False
        # RTF instrumentation: pure compute time across the utterance,
        # excluding any sleeps the harness inserts for audio arrival.
        self.compute_time_sec = 0.0
        # Encoder cache (rolling)
        self._cache_lc = np.zeros((1, ENC_LAYERS, LAST_CHANNEL_CACHE, ENC_DIM), dtype=np.float32)
        self._cache_lt = np.zeros((1, ENC_LAYERS, ENC_DIM, LAST_TIME_FRAMES), dtype=np.float32)
        self._cache_ll = np.zeros((1,), dtype=np.int64)
        # Decoder LSTM state (rolling)
        self._dec_h = np.zeros((DEC_LAYERS, 1, DEC_HIDDEN), dtype=np.float32)
        self._dec_c = np.zeros((DEC_LAYERS, 1, DEC_HIDDEN), dtype=np.float32)
        self._last_token = np.array([[0]], dtype=np.int32)
        self._target_length = np.array([1], dtype=np.int32)

    @staticmethod
    def _min_samples_for_frames(n_frames: int) -> int:
        if n_frames <= 0:
            return 0
        return (n_frames - 1) * HOP_SAMPLES + WINDOW_SAMPLES

    def _ensure_features(self) -> None:
        """(Re)compute mel features when more audio has arrived since last call."""
        if not self._raw_chunks or self._cached_n_samples == self._total_samples:
            return
        all_audio = np.concatenate(self._raw_chunks)
        audio_t = torch.from_numpy(all_audio).unsqueeze(0)
        length_t = torch.tensor([len(all_audio)], dtype=torch.long)
        with torch.inference_mode():
            mel, mel_len = self._preprocessor(input_signal=audio_t, length=length_t)
        self._cached_mel = mel[0].numpy().astype(np.float32)
        self._cached_feat_len = int(mel_len.item())
        self._cached_n_samples = self._total_samples

    def _run_steps(self, is_final: bool) -> None:
        """Advance the encoder/decoder while enough mel frames are available."""
        if not self._raw_chunks:
            return
        if not is_final and self._total_samples < self._min_samples_for_frames(
            self._buffer_idx + CHUNK_NEW
        ):
            return
        self._ensure_features()
        T = self._cached_feat_len
        if T <= 0:
            return

        while True:
            chunk_end = self._buffer_idx + CHUNK_NEW
            if chunk_end <= T:
                mel_chunk = self._cached_mel[:, self._buffer_idx:chunk_end]
                valid_in = CHUNK_NEW
            elif is_final and self._buffer_idx < T:
                # Final partial chunk: take what's left, pad to CHUNK_NEW.
                remaining = T - self._buffer_idx
                if remaining <= 0:
                    break
                mel_real = self._cached_mel[:, self._buffer_idx:T]
                pad = np.zeros((N_MELS, CHUNK_NEW - remaining), dtype=np.float32)
                mel_chunk = np.concatenate([mel_real, pad], axis=1)
                valid_in = remaining
            else:
                break

            # Pre-encode cache: zeros at step 0, last 9 real mel at step k>0.
            if self._step_num == 0:
                cache_pre = np.zeros((N_MELS, CACHE_FRAMES), dtype=np.float32)
            else:
                start_c = max(0, self._buffer_idx - CACHE_FRAMES)
                cache_pre = self._cached_mel[:, start_c:self._buffer_idx]
                if cache_pre.shape[1] < CACHE_FRAMES:
                    pad = np.zeros((N_MELS, CACHE_FRAMES - cache_pre.shape[1]), dtype=np.float32)
                    cache_pre = np.concatenate([pad, cache_pre], axis=1)

            chunk_input = np.concatenate([cache_pre, mel_chunk], axis=1)[None, :, :].astype(np.float32)
            chunk_length = np.array([CACHE_FRAMES + valid_in], dtype=np.int64)

            self._encode_and_decode(chunk_input, chunk_length, is_drain=False)

            self._buffer_idx += CHUNK_NEW
            self._step_num += 1
            if is_final and chunk_end >= T:
                break

    def _run_drain(self) -> None:
        """One extra all-zero chunk to flush the encoder's right-context."""
        if self._drain_done or self._step_num == 0:
            return
        T = self._cached_feat_len
        start_c = max(0, T - CACHE_FRAMES)
        cache_real = self._cached_mel[:, start_c:T]
        if cache_real.shape[1] < CACHE_FRAMES:
            pad = np.zeros((N_MELS, CACHE_FRAMES - cache_real.shape[1]), dtype=np.float32)
            cache_real = np.concatenate([pad, cache_real], axis=1)
        new_chunk = np.zeros((N_MELS, CHUNK_NEW), dtype=np.float32)
        chunk_input = np.concatenate([cache_real, new_chunk], axis=1)[None, :, :].astype(np.float32)
        chunk_length = np.array([CACHE_FRAMES + CHUNK_NEW], dtype=np.int64)
        self._encode_and_decode(chunk_input, chunk_length, is_drain=True)
        self._drain_done = True

    def _encode_and_decode(self, chunk_input: np.ndarray, chunk_length: np.ndarray, *, is_drain: bool) -> None:
        """Run one encoder step + RNN-T greedy decode + emit partial."""
        enc_outs = self._enc_sess.run(
            None,
            {
                "audio_signal": chunk_input,
                "length": chunk_length,
                "cache_last_channel": self._cache_lc,
                "cache_last_time": self._cache_lt,
                "cache_last_channel_len": self._cache_ll,
            },
        )
        named = dict(zip(self._enc_out_names, enc_outs))
        encoder_out = named["outputs"]
        n_enc = int(named["encoded_lengths"][0])
        for n, v in named.items():
            ln = n.lower()
            if "cache_last_channel" in ln and ("_len" in ln or "_length" in ln):
                self._cache_ll = v
            elif "cache_last_channel" in ln and "next" in ln:
                self._cache_lc = v
            elif "cache_last_time" in ln and "next" in ln:
                self._cache_lt = v

        for f_idx in range(n_enc):
            enc_frame = encoder_out[:, :, f_idx:f_idx + 1]
            for _sym in range(MAX_SYMBOLS_PER_FRAME):
                dec_outs = self._dec_sess.run(
                    None,
                    {
                        "encoder_outputs": enc_frame,
                        "targets": self._last_token,
                        "target_length": self._target_length,
                        "input_states_1": self._dec_h,
                        "input_states_2": self._dec_c,
                    },
                )
                dnamed = dict(zip(self._dec_out_names, dec_outs))
                logits = dnamed["outputs"]  # [B, 1, 1, vocab]
                token = int(np.argmax(logits[0, 0, 0]))
                if token == BLANK_ID:
                    break
                self._emitted_tokens.append(token)
                self._last_token = np.array([[token]], dtype=np.int32)
                self._dec_h = dnamed["output_states_1"]
                self._dec_c = dnamed["output_states_2"]

        text = _detokenize(self._emitted_tokens, self._vocab)
        text = " ".join(text.split())
        if not is_drain and text != self._last_emitted:
            self._partial_callback(text)
        self._last_emitted = text
