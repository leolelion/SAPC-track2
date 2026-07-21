"""SAPC2 Track 2 — A1 finetuned streaming Zipformer (self-contained, offline).

Decoding: modified_beam_search, max_active_paths=4 (beam-4). Validated on the 2k Dev
ruler (research/09): 19.01% CER vs 21.61% greedy (-2.60), at identical latency
(TTFT p50 1208ms vs 1234ms). Same ONNX weights as the greedy A1 submission.
"""
import os
from pathlib import Path
import numpy as np
import sherpa_onnx

_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
SR, FEAT_DIM, NUM_THREADS = 16000, 80, 4

class Model:
    def __init__(self):
        self._cb = None
        self._rec = sherpa_onnx.OnlineRecognizer.from_transducer(
            encoder=str(_DIR / "encoder.onnx"),
            decoder=str(_DIR / "decoder.onnx"),
            joiner=str(_DIR / "joiner.onnx"),
            tokens=str(_DIR / "tokens.txt"),
            num_threads=NUM_THREADS, provider="cpu",
            sample_rate=SR, feature_dim=FEAT_DIM,
            decoding_method="modified_beam_search", max_active_paths=4,
        )
        self._stream = None

    def set_partial_callback(self, callback):
        self._cb = callback

    def reset(self):
        self._stream = self._rec.create_stream()

    def _drain(self):
        while self._rec.is_ready(self._stream):
            self._rec.decode_stream(self._stream)

    def accept_chunk(self, audio_chunk):
        self._stream.accept_waveform(sample_rate=SR, waveform=audio_chunk)
        self._drain()
        text = self._rec.get_result(self._stream).strip()
        if text and self._cb is not None:
            self._cb(text)
        return text

    def input_finished(self):
        tail = np.zeros(int(SR * 0.3), dtype=np.float32)
        self._stream.accept_waveform(sample_rate=SR, waveform=tail)
        self._stream.input_finished()
        self._drain()
        final = self._rec.get_result(self._stream).strip()
        if self._cb is not None:
            self._cb(final)
        return final
