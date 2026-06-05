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
# Section 1: Thread + venv setup (must come before any heavy imports)
# =====================================================================
import os
import sys
import glob as _glob

_THREADS = int(os.environ.get("SAPC2_THREADS", "4"))
os.environ.setdefault("OMP_NUM_THREADS", str(_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(_THREADS))

import numpy as np  # noqa: E402

# Pin venv if setup.sh installed one alongside this file.
_DIR = os.path.dirname(os.path.abspath(__file__))
_venv_candidates = _glob.glob(
    os.path.join(_DIR, "venv", "lib", "python3.*", "site-packages")
) or _glob.glob(
    os.path.join(_DIR, "venv", "*", "lib", "python3.*", "site-packages")
)
if _venv_candidates:
    sys.path.insert(0, _venv_candidates[0])

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
# Section 4: NeMo mel preprocessor — load once at __init__
#
# CRITICAL memory note: instantiate the preprocessor module DIRECTLY
# from the Nemotron config, NOT by loading the full 618M-param ASR
# model and reading `model.preprocessor`. The full model load peaks
# at ~4 GB RSS and OOM-kills on tight CPU eval VMs. Direct
# instantiation peaks at ~1.1 GB total.
#
# The constructor args mirror the Nemotron checkpoint's
# `model_config.yaml` preprocessor block exactly (verified by reading
# the .nemo tarball directly):
#
#   _target_: nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor
#   sample_rate: 16000
#   window_size: 0.025
#   window_stride: 0.010
#   window: hann
#   features: 128
#   n_fft: 512
#   dither: 1.0e-05   ← we override to 0.0 for determinism
#   pad_to: 0
#   normalize: NA
#   frame_splicing: 1
#   log: true
#   pad_value: 0.0
# =====================================================================
def _load_preprocessor():
    """Build the deterministic Nemotron mel preprocessor (memory-light)."""
    from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor
    pp = AudioToMelSpectrogramPreprocessor(
        sample_rate=16000,
        window_size=0.025,
        window_stride=0.010,
        window="hann",
        features=128,
        n_fft=512,
        dither=0.0,        # override the checkpoint's 1e-5 for determinism
        pad_to=0,
        normalize="NA",    # NeMo's sentinel for "no normalization"
        frame_splicing=1,
        log=True,
        pad_value=0.0,
    )
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

        print(f"[nemotron_streaming] building preprocessor (NeMo, CPU) ...", flush=True)
        self._preprocessor = _load_preprocessor()

        self._partial_callback = lambda _t: None
        self._reset_state()

        print(
            f"[nemotron_streaming] ready (threads={num_threads}, "
            f"chunk={CHUNK_NEW} mel, cache={CACHE_FRAMES} mel, blank={BLANK_ID})",
            flush=True,
        )

    # ----- SAPC2 interface -----
    def set_partial_callback(self, callback) -> None:
        """Register callback for partial results: callback(text: str)."""
        self._partial_callback = callback

    def reset(self) -> None:
        """Reset state for a new audio file."""
        self._reset_state()

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
