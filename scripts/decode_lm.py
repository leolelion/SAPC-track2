#!/usr/bin/env python3
"""Decode SAPC2 utts with a sherpa-onnx streaming transducer — greedy OR modified_beam_search,
optionally with external LM shallow fusion. For the decoding/LM experiments in research/08.

Run on the pod (sherpa-onnx env). Reuses the validated decode loop from finetune/eval/decode.py
(100ms chunks, 0.3s tail). Output CSV (id,hypo) is scored by sapc-nemotron utils/evaluate.py.

Usage:
  python decode_lm.py MODEL_DIR MANIFEST OUT.csv [--method greedy_search|modified_beam_search]
      [--max-active-paths N] [--lm LM.onnx] [--lm-scale F] [--limit N]

NOTE (verify on pod): sherpa-onnx ONLINE transducer LM support — `from_transducer(..., lm=, lm_scale=)`
expects an ONNX RNN-LM for online modified_beam_search. If only RNN-LM (not KenLM n-gram) is supported
online, train a small LSTM LM on SAPC2 text via icefall rnn_lm and export ONNX (see build_lm.sh).
Confirm the exact kwarg names with: python -c "import sherpa_onnx, inspect; print(inspect.signature(sherpa_onnx.OnlineRecognizer.from_transducer))"
"""
import sys, os, csv, wave, argparse
import numpy as np
import sherpa_onnx


def read_wave(path):
    with wave.open(path, "rb") as w:
        sr, n, sw, ch = w.getframerate(), w.getnframes(), w.getsampwidth(), w.getnchannels()
        raw = w.readframes(n)
    if sw == 2:
        a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        a = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        a = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128.0
    if ch > 1:
        a = a.reshape(-1, ch)[:, 0]
    return np.ascontiguousarray(a), sr


def build_recognizer(model_dir, method, max_active_paths, lm, lm_scale, num_threads=4):
    kw = dict(
        tokens=f"{model_dir}/tokens.txt",
        encoder=f"{model_dir}/encoder.onnx",
        decoder=f"{model_dir}/decoder.onnx",
        joiner=f"{model_dir}/joiner.onnx",
        num_threads=num_threads, sample_rate=16000, feature_dim=80,
        decoding_method=method, provider="cpu",
    )
    if method == "modified_beam_search":
        kw["max_active_paths"] = max_active_paths
        if lm:  # shallow fusion; verify kwarg names on pod
            kw["lm"] = lm
            kw["lm_scale"] = lm_scale
    return sherpa_onnx.OnlineRecognizer.from_transducer(**kw)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model_dir"); p.add_argument("manifest"); p.add_argument("out")
    p.add_argument("--data-root", default="/workspace/SAPC2")
    p.add_argument("--method", default="modified_beam_search")
    p.add_argument("--max-active-paths", type=int, default=4)
    p.add_argument("--lm", default=""); p.add_argument("--lm-scale", type=float, default=0.0)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()

    rec = build_recognizer(a.model_dir, a.method, a.max_active_paths, a.lm, a.lm_scale)
    rows = list(csv.DictReader(open(a.manifest)))
    if a.limit:
        rows = rows[:a.limit]
    out = []
    for r in rows:
        audio, sr = read_wave(os.path.join(a.data_root, r["audio_filepath"]))
        s = rec.create_stream()
        for i in range(0, len(audio), 1600):
            s.accept_waveform(sr, audio[i:i + 1600])
            while rec.is_ready(s):
                rec.decode_stream(s)
        s.accept_waveform(sr, np.zeros(int(0.3 * sr), dtype=np.float32))
        s.input_finished()
        while rec.is_ready(s):
            rec.decode_stream(s)
        out.append((r["id"], str(rec.get_result(s)).strip().lower()))
    w = csv.writer(open(a.out, "w")); w.writerow(["id", "hypo"]); w.writerows(out)
    print(f"decoded {len(out)} method={a.method} paths={a.max_active_paths} lm={a.lm or 'none'} scale={a.lm_scale}")


if __name__ == "__main__":
    main()
