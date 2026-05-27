#!/usr/bin/env python3
"""Phase 1b — ONNX FP32 streaming sweep on Dev_streaming.

Mirrors SAPC2 two-thread ingestion:
  - Sender thread feeds 100ms float32 chunks at real-time pace (16kHz mono).
  - Decoder thread calls accept_chunk / input_finished synchronously
    in a single Python thread; partial_callback fires from inside
    accept_chunk only (matches SAPC2 spec).

Per utterance we record:
  audio_send_start_time, mfa_speech_start, partial events (text+ts),
  final_visible_time, audio_end_oracle_time, decoder compute time.

Latency definitions (from Track2 README):
  TTFT = first_non_empty_partial_time - (audio_send_start + mfa_speech_start)
  TTLT = final_visible_time - audio_end_oracle_time

RTF (Track2 gate): sum(decoder_compute_time) / sum(audio_duration).

Outputs:
  <out_prefix>.csv               id,raw_hypos
  <out_prefix>.partial.json      per-utt events (callback timestamps)
  <out_prefix>.summary.json      aggregated metrics (CER/WER, P50/P90/P95)
"""
import argparse
import csv
import json
import os
import queue
import sys
import threading
import time
import wave
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
import torch

import nemo.collections.asr as nemo_asr  # noqa: F401
from nemo.collections.asr.models import ASRModel

SAMPLE_RATE = 16000
HOP_SAMPLES = 160
WINDOW_SAMPLES = 400
N_MELS = 128
BLANK_ID = 1024


# =====================================================================
# Streaming Nemotron Model (ONNX, 5-method interface)
# =====================================================================

class NemotronStreamingONNX:
    """ONNX streaming wrapper. Accepts 100ms (1600 sample) float32 chunks.

    Internally:
      - Mel preprocessor is the NeMo `AudioToMelSpectrogramPreprocessor`
        run via PyTorch (deterministic with dither=0, pad_to=0).
      - When enough mel frames accumulated for the chunk_size at current
        att_context, run encoder ONNX with carried-forward caches.
      - For each new encoder frame, RNN-T greedy decode via decoder_joint.
    """

    def __init__(self, encoder_path: str, decoder_path: str,
                 ref_model: ASRModel, att_context: List[int],
                 num_threads: int = 4):
        self.encoder_path = encoder_path
        self.decoder_path = decoder_path
        self.num_threads = num_threads

        # ── Preprocessor (PyTorch CPU, deterministic) ─────────────────
        self._preprocessor = ref_model.preprocessor
        self._preprocessor.featurizer.dither = 0.0
        self._preprocessor.featurizer.pad_to = 0
        self._preprocessor.eval()

        # ── Tokenizer ─────────────────────────────────────────────────
        self._tokenizer = ref_model.tokenizer

        # ── streaming_cfg for this config ─────────────────────────────
        ref_model.encoder.set_default_att_context_size(att_context)
        cfg = ref_model.encoder.streaming_cfg
        self._chunk_size = list(cfg.chunk_size)          # [step0, steady]
        self._shift_size = list(cfg.shift_size)
        self._pre_encode_cache = list(cfg.pre_encode_cache_size)
        # encoder arch shapes (for cache init)
        self._enc_layers = ref_model.encoder._cfg.n_layers
        self._enc_dim = ref_model.encoder._cfg.d_model
        self._last_channel_cache = cfg.last_channel_cache_size
        # last_time cache size: pull from get_initial_cache_state
        lc, lt, ll = ref_model.encoder.get_initial_cache_state(batch_size=1, device="cpu")
        # lc layout (N, B, T, H) -> we feed (B, N, T, H); last_time (N, B, H, T) -> (B, N, H, T)
        self._init_lc = lc.transpose(0, 1).contiguous().numpy().astype(np.float32)
        self._init_lt = lt.transpose(0, 1).contiguous().numpy().astype(np.float32)
        self._init_ll = ll.contiguous().numpy().astype(np.int64)

        # ── Decoder hidden / vocab ────────────────────────────────────
        self._dec_layers = 2
        self._dec_hidden = 640
        self._vocab_size = 1025  # 1024 + blank

        # ── ORT sessions ──────────────────────────────────────────────
        self._enc_sess = self._make_session(encoder_path)
        self._dec_sess = self._make_session(decoder_path)
        self._enc_out_names = [o.name for o in self._enc_sess.get_outputs()]
        self._dec_out_names = [o.name for o in self._dec_sess.get_outputs()]

        # ── Per-utt state, initialised in reset() ─────────────────────
        self._partial_callback: Optional[Callable[[str], None]] = lambda _t: None
        self._reset_state()

        # Cumulative compute time across utts (excludes preprocessor)
        self.cumulative_decoder_seconds = 0.0

    # ----------------------------------------------------------------- public
    def set_partial_callback(self, fn: Callable[[str], None]) -> None:
        self._partial_callback = fn

    def reset(self) -> None:
        self._reset_state()

    def accept_chunk(self, chunk: np.ndarray) -> str:
        """Feed one 100ms (1600 samples) float32 chunk. Return current hypothesis."""
        t0 = time.perf_counter()
        self._raw_chunks.append(chunk.astype(np.float32, copy=False))
        self._total_samples += len(chunk)
        self._run_steps(is_final=False)
        self.cumulative_decoder_seconds += (time.perf_counter() - t0)
        return self._last_emitted

    def input_finished(self) -> str:
        t0 = time.perf_counter()
        self._run_steps(is_final=True)
        self.cumulative_decoder_seconds += (time.perf_counter() - t0)
        return self._last_emitted

    # ----------------------------------------------------------------- internal
    def _make_session(self, path: str) -> ort.InferenceSession:
        so = ort.SessionOptions()
        so.intra_op_num_threads = self.num_threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])

    def _reset_state(self):
        self._raw_chunks: List[np.ndarray] = []
        self._total_samples = 0
        self._buffer_idx = 0           # current position in mel frames
        self._step_num = 0
        self._last_emitted = ""
        self._drain_done = False
        # Encoder cache (rolling)
        self._cache_lc = self._init_lc.copy()
        self._cache_lt = self._init_lt.copy()
        self._cache_ll = self._init_ll.copy()
        # Decoder state (rolling)
        self._dec_h = np.zeros((self._dec_layers, 1, self._dec_hidden), dtype=np.float32)
        self._dec_c = np.zeros((self._dec_layers, 1, self._dec_hidden), dtype=np.float32)
        self._last_token = np.array([[0]], dtype=np.int32)
        self._target_length = np.array([1], dtype=np.int32)
        self._emitted_tokens: List[int] = []
        # Mel feature cache (avoid full recompute every chunk)
        self._cached_mel: Optional[np.ndarray] = None
        self._cached_feat_len = 0
        self._cached_n_samples = 0

    def _cfg_val(self, lst, step):
        return lst[0] if step == 0 else lst[1]

    def _min_samples_for_frames(self, n_frames: int) -> int:
        if n_frames <= 0:
            return 0
        return (n_frames - 1) * HOP_SAMPLES + WINDOW_SAMPLES

    def _ensure_features(self):
        """Recompute mel from accumulated audio when more samples have arrived."""
        if self._cached_n_samples == self._total_samples or not self._raw_chunks:
            return
        all_audio = np.concatenate(self._raw_chunks)
        audio_t = torch.from_numpy(all_audio).unsqueeze(0)
        length_t = torch.tensor([len(all_audio)], dtype=torch.long)
        with torch.inference_mode():
            mel, mel_len = self._preprocessor(input_signal=audio_t, length=length_t)
        self._cached_mel = mel[0].numpy().astype(np.float32)   # [128, T]
        self._cached_feat_len = int(mel_len.item())
        self._cached_n_samples = self._total_samples

    def _run_steps(self, is_final: bool):
        """Streaming loop matching PyTorch conformer_stream_step behavior.

        The ONNX graph bakes in drop_extra_pre_encoded=DROP_EXTRA (=2 at [70,6]).
        PyTorch passes cache=None at step 0 which skips drop_extra entirely;
        at step k>0 it passes real cache and drop_extra fires on the cache_pre.

        We can't pass cache=None to ONNX, but we can match PT by compensating:
          step 0:  audio = [DROP_EXTRA zero frames] + [chunk_size[0] real mel],
                   length = DROP_EXTRA + chunk_size[0].
                   drop_extra eats the zeros, encoder sees chunk_size[0] real frames.
          step k:  audio = [pre_encode_cache[1] real mel] + [chunk_size[1] new mel],
                   length = pre_encode_cache[1] + chunk_size[1].
                   drop_extra eats first DROP_EXTRA of the cache (matches PT).

        Drain: after audio runs out, feed one extra chunk of zeros so the
        encoder's right-context attention can resolve the last frames.
        """
        DROP_EXTRA = 2  # baked into ONNX for [70,*] (subsampling=8, sampling_frames=[1,8])

        if not self._raw_chunks:
            return
        cs1 = self._chunk_size[1]
        pcs1 = self._pre_encode_cache[1]

        # Early exit: need enough audio for the next step's content.
        if self._step_num == 0:
            need_mel = self._chunk_size[0]
        else:
            need_mel = self._buffer_idx + cs1 - self._buffer_idx  # = cs1 (just for clarity)
            need_mel = self._buffer_idx + cs1
        # Compute required samples to cover need_mel mel frames.
        if self._step_num == 0:
            needed_samples = self._min_samples_for_frames(need_mel)
        else:
            needed_samples = self._min_samples_for_frames(need_mel)
        if not is_final and self._total_samples < needed_samples:
            return
        self._ensure_features()
        T = self._cached_feat_len
        if T <= 0:
            return

        while True:
            # Step-specific chunk geometry (matches PT conformer_stream_step).
            if self._step_num == 0:
                cs = self._chunk_size[0]
                ss = self._shift_size[0]
                pcs = self._pre_encode_cache[0]   # 0 for [70,*]
                pad_extra = DROP_EXTRA
                use_real_cache = False
            else:
                cs = cs1
                ss = self._shift_size[1]
                pcs = pcs1
                pad_extra = 0
                use_real_cache = True

            chunk_end = self._buffer_idx + cs
            mel_chunk: Optional[np.ndarray]
            if chunk_end <= T:
                mel_chunk = self._cached_mel[:, self._buffer_idx:chunk_end]
                valid_in = cs
            elif is_final and self._buffer_idx < T:
                # Partial final chunk: take what's left, then we'll also append a drain step.
                remaining = T - self._buffer_idx
                if remaining <= 0:
                    break
                mel_real = self._cached_mel[:, self._buffer_idx:T]
                pad = np.zeros((N_MELS, cs - remaining), dtype=np.float32)
                mel_chunk = np.concatenate([mel_real, pad], axis=1)
                valid_in = remaining
            else:
                break

            # Build cache_pre:  pad_extra zeros + (optional) real cache mel
            real_cache: np.ndarray
            if use_real_cache and pcs > 0:
                start_c = max(0, self._buffer_idx - pcs)
                real_cache = self._cached_mel[:, start_c:self._buffer_idx]
                if real_cache.shape[1] < pcs:
                    pad = np.zeros((N_MELS, pcs - real_cache.shape[1]), dtype=np.float32)
                    real_cache = np.concatenate([pad, real_cache], axis=1)
            else:
                real_cache = np.empty((N_MELS, 0), dtype=np.float32)

            if pad_extra > 0:
                pad_zeros = np.zeros((N_MELS, pad_extra), dtype=np.float32)
                cache_pre = np.concatenate([pad_zeros, real_cache], axis=1)
            else:
                cache_pre = real_cache

            chunk_input = np.concatenate([cache_pre, mel_chunk], axis=1)[None, :, :].astype(np.float32)
            # length: pad_extra + real_cache + valid_in real mel frames.
            # The pad_extra frames are dropped by the encoder (drop_extra_pre_encoded).
            chunk_len = np.array([pad_extra + real_cache.shape[1] + valid_in], dtype=np.int64)

            # ── Encoder ───────────────────────────────────────────────
            enc_outs = self._enc_sess.run(
                None,
                {
                    "audio_signal": chunk_input,
                    "length": chunk_len,
                    "cache_last_channel": self._cache_lc,
                    "cache_last_time": self._cache_lt,
                    "cache_last_channel_len": self._cache_ll,
                },
            )
            named = dict(zip(self._enc_out_names, enc_outs))
            encoder_out = named["outputs"]            # [1, 1024, n_enc]
            n_enc = int(named["encoded_lengths"][0])
            # update caches (match by name; _len before *_next to avoid prefix overlap)
            for n, v in named.items():
                ln = n.lower()
                if "cache_last_channel" in ln and ("_len" in ln or "_length" in ln):
                    self._cache_ll = v
                elif "cache_last_channel" in ln and "next" in ln:
                    self._cache_lc = v
                elif "cache_last_time" in ln and "next" in ln:
                    self._cache_lt = v

            # ── RNN-T greedy decode over new encoder frames ───────────
            for f_idx in range(n_enc):
                enc_frame = encoder_out[:, :, f_idx:f_idx + 1]  # [1, 1024, 1]
                for _sym in range(10):
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
                    logits = dnamed["outputs"]               # [1,1,1,1025]
                    token = int(np.argmax(logits[0, 0, 0]))
                    if token == BLANK_ID:
                        break
                    self._emitted_tokens.append(token)
                    self._last_token = np.array([[token]], dtype=np.int32)
                    self._dec_h = dnamed["output_states_1"]
                    self._dec_c = dnamed["output_states_2"]

            # ── Emit partial if changed ───────────────────────────────
            text = self._tokenizer.ids_to_text(self._emitted_tokens)
            text = " ".join(text.split())
            if not is_final and text != self._last_emitted:
                self._partial_callback(text)
            self._last_emitted = text

            self._buffer_idx += ss
            self._step_num += 1
            if is_final and chunk_end >= T:
                break

        # Drain step: one extra all-zero chunk to flush the encoder's
        # right-context attention so the last 6 real frames get final outputs.
        # Only runs once at end-of-utterance.
        if is_final and not self._drain_done and self._step_num > 0:
            self._drain_done = True
            real_cache_start = max(0, T - pcs1)
            real_cache = self._cached_mel[:, real_cache_start:T]
            if real_cache.shape[1] < pcs1:
                pad = np.zeros((N_MELS, pcs1 - real_cache.shape[1]), dtype=np.float32)
                real_cache = np.concatenate([pad, real_cache], axis=1)
            mel_chunk = np.zeros((N_MELS, cs1), dtype=np.float32)
            chunk_input = np.concatenate([real_cache, mel_chunk], axis=1)[None, :, :].astype(np.float32)
            chunk_len = np.array([real_cache.shape[1] + cs1], dtype=np.int64)
            enc_outs = self._enc_sess.run(
                None,
                {
                    "audio_signal": chunk_input,
                    "length": chunk_len,
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
                for _sym in range(10):
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
                    logits = dnamed["outputs"]
                    token = int(np.argmax(logits[0, 0, 0]))
                    if token == BLANK_ID:
                        break
                    self._emitted_tokens.append(token)
                    self._last_token = np.array([[token]], dtype=np.int32)
                    self._dec_h = dnamed["output_states_1"]
                    self._dec_c = dnamed["output_states_2"]
            text = self._tokenizer.ids_to_text(self._emitted_tokens)
            text = " ".join(text.split())
            self._last_emitted = text


# =====================================================================
# Driver: two-"thread" real-time pacing
# =====================================================================

def run_one_utterance(model: NemotronStreamingONNX, audio: np.ndarray,
                      mfa_speech_start: Optional[float]) -> dict:
    """Drive `model` through `audio` at real-time pace via a sender thread.

    Returns event log + key timestamps for TTFT/TTLT computation.
    """
    events: List[dict] = []
    audio_dur = len(audio) / SAMPLE_RATE
    sender_q: "queue.Queue[Tuple[str, Optional[np.ndarray]]]" = queue.Queue()

    def callback(text: str) -> None:
        # Called from Decoder ("main") thread inside accept_chunk.
        events.append({"event": "partial", "time": time.time(), "text": text})

    model.set_partial_callback(callback)
    model.reset()

    n_chunks = (len(audio) + 1599) // 1600
    chunks = [audio[i * 1600:(i + 1) * 1600] for i in range(n_chunks)]

    audio_send_start = time.time()
    audio_end_oracle = audio_send_start + audio_dur

    def sender():
        for i, ch in enumerate(chunks):
            target = audio_send_start + (i + 1) * 0.1
            now = time.time()
            if target > now:
                time.sleep(target - now)
            sender_q.put(("chunk", ch))
        sender_q.put(("end", None))

    sender_thread = threading.Thread(target=sender, daemon=True)
    sender_thread.start()

    final_text = ""
    while True:
        kind, data = sender_q.get()
        if kind == "end":
            final_text = model.input_finished()
            break
        model.accept_chunk(data)

    final_visible_time = time.time()
    sender_thread.join()

    events.append({"event": "final_visible", "time": final_visible_time, "text": final_text})

    # First non-empty partial timestamp (else final_visible per Track2 README).
    first_non_empty = None
    for ev in events:
        if ev["event"] == "partial" and ev["text"].strip():
            first_non_empty = ev["time"]
            break
    if first_non_empty is None:
        first_non_empty = final_visible_time

    return {
        "events": events,
        "timing": {
            "audio_send_start_time": audio_send_start,
            "audio_end_oracle_time": audio_end_oracle,
            "first_partial_time": first_non_empty,
            "final_visible_time": final_visible_time,
            "audio_dur_sec": audio_dur,
            "mfa_speech_start": mfa_speech_start,
        },
        "final_text": final_text,
    }


# =====================================================================
# Manifest loader + metric helpers
# =====================================================================

def read_manifest(manifest_csv: str) -> List[dict]:
    rows = []
    with open(manifest_csv) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def read_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        if w.getframerate() != SAMPLE_RATE:
            raise ValueError(f"{path}: expected {SAMPLE_RATE}Hz")
        if w.getnchannels() != 1:
            raise ValueError(f"{path}: expected mono")
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16-bit")
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def normalize_text(s: str) -> str:
    """Light normalization for jiwer; sclite is authoritative later."""
    s = s.lower().strip()
    # Strip punctuation we won't use for ASR scoring.
    for ch in ",.?!:;\"'":
        s = s.replace(ch, "")
    return " ".join(s.split())


def compute_cer_wer_min2(refs1: List[str], refs2: List[str], hyps: List[str]) -> Tuple[float, float]:
    """Per-utterance min-of-two-refs WER & CER aggregated via jiwer."""
    import jiwer
    assert len(refs1) == len(refs2) == len(hyps)
    # jiwer per-utterance, take min ref edit count for each metric.
    wer_total_subs, wer_total_dels, wer_total_ins, wer_total_ref = 0, 0, 0, 0
    cer_total_subs, cer_total_dels, cer_total_ins, cer_total_ref = 0, 0, 0, 0
    for r1, r2, h in zip(refs1, refs2, hyps):
        r1n, r2n, hn = normalize_text(r1), normalize_text(r2), normalize_text(h)
        # WER: pick the ref with lower edits for this utt.
        out1 = jiwer.process_words(r1n, hn)
        out2 = jiwer.process_words(r2n, hn)
        if (out1.substitutions + out1.deletions + out1.insertions) <= (out2.substitutions + out2.deletions + out2.insertions):
            wer_total_subs += out1.substitutions; wer_total_dels += out1.deletions; wer_total_ins += out1.insertions; wer_total_ref += out1.hits + out1.substitutions + out1.deletions
        else:
            wer_total_subs += out2.substitutions; wer_total_dels += out2.deletions; wer_total_ins += out2.insertions; wer_total_ref += out2.hits + out2.substitutions + out2.deletions
        # CER: same min trick, character edits.
        cout1 = jiwer.process_characters(r1n, hn)
        cout2 = jiwer.process_characters(r2n, hn)
        if (cout1.substitutions + cout1.deletions + cout1.insertions) <= (cout2.substitutions + cout2.deletions + cout2.insertions):
            cer_total_subs += cout1.substitutions; cer_total_dels += cout1.deletions; cer_total_ins += cout1.insertions; cer_total_ref += cout1.hits + cout1.substitutions + cout1.deletions
        else:
            cer_total_subs += cout2.substitutions; cer_total_dels += cout2.deletions; cer_total_ins += cout2.insertions; cer_total_ref += cout2.hits + cout2.substitutions + cout2.deletions
    wer = (wer_total_subs + wer_total_dels + wer_total_ins) / max(1, wer_total_ref)
    cer = (cer_total_subs + cer_total_dels + cer_total_ins) / max(1, cer_total_ref)
    return cer, wer


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, p))


# =====================================================================
# Main
# =====================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--decoder", required=True)
    ap.add_argument("--att-context", nargs=2, type=int, required=True, metavar=("LEFT", "RIGHT"))
    ap.add_argument("--manifest", default="/workspace/SAPC2/manifest/Dev_streaming.csv")
    ap.add_argument("--audio-root", default="/workspace/SAPC2")
    ap.add_argument("--out-prefix", required=True, help="output prefix (no extension)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="cap utterances (debug)")
    ap.add_argument("--heartbeat", default="/workspace/tmux_logs/phase1b.heartbeat")
    ap.add_argument("--config-label", default=None, help="for heartbeat lines")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    # Load PT model just for preprocessor + tokenizer + streaming_cfg lookup.
    print(f"[load] PT model for preprocessor/tokenizer ...", flush=True)
    ref_model = ASRModel.from_pretrained("nvidia/nemotron-speech-streaming-en-0.6b", map_location="cpu")
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    print(f"[load] ONNX sessions ...", flush=True)
    model = NemotronStreamingONNX(args.encoder, args.decoder, ref_model, args.att_context, args.threads)

    print(f"[manifest] {args.manifest}", flush=True)
    rows = read_manifest(args.manifest)
    if args.limit:
        rows = rows[:args.limit]
    print(f"  utterances: {len(rows)}", flush=True)

    # Per-utt outputs + accumulators
    hyps: List[str] = []
    refs1: List[str] = []
    refs2: List[str] = []
    ids: List[str] = []
    per_utt: List[dict] = []
    ttfts_ms: List[float] = []
    ttlts_ms: List[float] = []
    total_audio_dur = 0.0

    sweep_start = time.time()
    config_label = args.config_label or f"{args.att_context[0]}_{args.att_context[1]}"
    heartbeat_path = args.heartbeat
    os.makedirs(os.path.dirname(heartbeat_path), exist_ok=True)

    for i, row in enumerate(rows):
        utt_id = row["id"]
        audio_path = os.path.join(args.audio_root, row["audio_filepath"])
        audio = read_wav(audio_path)
        mfa_start = float(row.get("mfa_speech_start", 0.0) or 0.0)

        result = run_one_utterance(model, audio, mfa_start)
        timing = result["timing"]
        hyp = result["final_text"]
        hyps.append(hyp)
        refs1.append(row.get("norm_text_with_disfluency", row.get("text", "")))
        refs2.append(row.get("norm_text_without_disfluency", row.get("text", "")))
        ids.append(utt_id)

        ttft_sec = timing["first_partial_time"] - (timing["audio_send_start_time"] + timing["mfa_speech_start"])
        ttlt_sec = timing["final_visible_time"] - timing["audio_end_oracle_time"]
        ttfts_ms.append(ttft_sec * 1000.0)
        ttlts_ms.append(ttlt_sec * 1000.0)
        total_audio_dur += timing["audio_dur_sec"]

        per_utt.append({
            "id": utt_id,
            "hyp": hyp,
            "timing": timing,
            "events": result["events"],
            "ttft_sec": ttft_sec,
            "ttlt_sec": ttlt_sec,
        })

        # Heartbeat after each utt.
        with open(heartbeat_path, "w") as hb:
            elapsed = time.time() - sweep_start
            hb.write(
                f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} config={config_label} "
                f"utt={i+1}/{len(rows)} cum_dec_s={model.cumulative_decoder_seconds:.2f} "
                f"cum_audio_s={total_audio_dur:.2f} "
                f"running_rtf={model.cumulative_decoder_seconds/max(0.001,total_audio_dur):.3f} "
                f"elapsed_s={elapsed:.1f}\n"
            )
        if (i + 1) % 10 == 0 or i + 1 == len(rows):
            print(f"  [{i+1}/{len(rows)}] running RTF={model.cumulative_decoder_seconds/max(0.001,total_audio_dur):.3f} "
                  f"TTFT_running_p50={percentile(ttfts_ms,50):.0f}ms", flush=True)

    sweep_wall = time.time() - sweep_start

    # ── Metrics ───────────────────────────────────────────────────────
    cer, wer = compute_cer_wer_min2(refs1, refs2, hyps)
    rtf = model.cumulative_decoder_seconds / max(1e-6, total_audio_dur)
    summary = {
        "att_context_size": list(args.att_context),
        "config_label": config_label,
        "format": "onnx_fp32",
        "threads": args.threads,
        "n_utt": len(rows),
        "wer": wer,
        "cer": cer,
        "ttft_p50_ms": percentile(ttfts_ms, 50),
        "ttft_p90_ms": percentile(ttfts_ms, 90),
        "ttft_p95_ms": percentile(ttfts_ms, 95),
        "ttlt_p50_ms": percentile(ttlts_ms, 50),
        "ttlt_p90_ms": percentile(ttlts_ms, 90),
        "ttlt_p95_ms": percentile(ttlts_ms, 95),
        "avg_lat_ms": 0.5 * (percentile(ttfts_ms, 50) + percentile(ttlts_ms, 50)),
        "rtf": rtf,
        "rtf_gate_pass": bool(rtf < 1.0),
        "total_audio_sec": total_audio_dur,
        "cumulative_decoder_sec": model.cumulative_decoder_seconds,
        "sweep_wall_sec": sweep_wall,
        "encoder_onnx": args.encoder,
        "decoder_onnx": args.decoder,
        "manifest": args.manifest,
    }

    out_prefix = args.out_prefix
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)

    # CSV (id, raw_hypos)
    with open(out_prefix + ".csv", "w") as f:
        w = csv.writer(f)
        w.writerow(["id", "raw_hypos"])
        for uid, h in zip(ids, hyps):
            w.writerow([uid, h])

    # Partial events JSON (per SAPC2 latency tool format)
    partial = {}
    for u in per_utt:
        partial[u["id"]] = {
            "events": u["events"],
            "audio_send_start_time": u["timing"]["audio_send_start_time"],
            "audio_end_oracle_time": u["timing"]["audio_end_oracle_time"],
            "final_visible_time": u["timing"]["final_visible_time"],
        }
    with open(out_prefix + ".partial.json", "w") as f:
        json.dump(partial, f)

    # Per-utt latencies sidecar
    with open(out_prefix + ".per_utt.json", "w") as f:
        json.dump(per_utt, f)

    # Summary
    with open(out_prefix + ".summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
