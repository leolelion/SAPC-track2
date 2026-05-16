#!/usr/bin/env python3
"""
Phase 2a: 3-utterance smoke test for Nemotron streaming submission.

Compares our Model class (chunk-by-chunk streaming) against NeMo's offline
model.transcribe() on the same 3 utterances. Also prints encoder tensor shapes
and cache evolution for the first 3 model steps to verify cache-state bookkeeping.

Usage:
  python3 smoke_test_2a.py \
      --manifest /workspace/SAPC2/manifest/Dev_streaming.csv \
      --data-root /workspace/SAPC2

Expects model.py in the same directory as this script.
"""
import argparse
import csv
import json
import os
import sys
import time
import wave
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Monkey-patch our Model class to instrument _run_steps for shape debugging
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
CHUNK_SIZE = 1600  # 100ms


def read_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as f:
        if f.getframerate() != SAMPLE_RATE:
            raise ValueError(f"{path}: expected {SAMPLE_RATE}Hz, got {f.getframerate()}")
        if f.getnchannels() != 1:
            raise ValueError(f"{path}: mono expected")
        if f.getsampwidth() != 2:
            raise ValueError(f"{path}: 16-bit expected")
        raw = f.readframes(f.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def pick_utterances(manifest_path, data_root, target_lengths=(3.0, 8.0, 20.0)):
    """Pick utterances closest to target durations."""
    rows = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            audio_path = os.path.join(data_root, row["audio_filepath"])
            if os.path.exists(audio_path):
                samples = read_wav(audio_path)
                dur = len(samples) / SAMPLE_RATE
                rows.append((row["id"], audio_path, dur, samples))

    picks = []
    for target in target_lengths:
        best = min(rows, key=lambda r: abs(r[2] - target))
        picks.append(best)
        rows.remove(best)  # don't pick the same one twice

    return picks


def run_streaming(model_cls, audio_path, samples, verbose_steps=3):
    """Run our Model class in streaming mode, optionally printing shapes."""
    model = None
    # Reuse the global model instance (expensive to reload)
    global _shared_model
    if "_shared_model" not in globals() or _shared_model is None:
        _shared_model = model_cls()
    model = _shared_model

    # Instrument: capture shape info for first N steps
    step_info = []
    original_run_steps = model._run_steps.__func__

    def instrumented_run_steps(self, is_final=False):
        """Wrapper that captures tensor shapes during model steps."""
        if not self._raw_chunks:
            return

        # Call original logic but capture pre/post state for each step
        # We do this by temporarily patching conformer_stream_step
        orig_stream_step = self._model.conformer_stream_step

        def capturing_stream_step(**kwargs):
            pre_cache_ch_shape = kwargs["cache_last_channel"].shape
            pre_cache_t_shape = kwargs["cache_last_time"].shape
            input_shape = kwargs["processed_signal"].shape
            drop_extra = kwargs["drop_extra_pre_encoded"]
            keep_all = kwargs["keep_all_outputs"]

            result = orig_stream_step(**kwargs)

            post_cache_ch_shape = result[2].shape
            post_cache_t_shape = result[3].shape

            if len(step_info) < verbose_steps:
                step_info.append({
                    "step": self._step_num,
                    "input_shape": list(input_shape),
                    "pre_cache_channel": list(pre_cache_ch_shape),
                    "pre_cache_time": list(pre_cache_t_shape),
                    "post_cache_channel": list(post_cache_ch_shape),
                    "post_cache_time": list(post_cache_t_shape),
                    "drop_extra": drop_extra,
                    "keep_all": keep_all,
                })
            return result

        self._model.conformer_stream_step = capturing_stream_step

        # Actually run the steps
        original_run_steps(self, is_final=is_final)

        # Restore
        self._model.conformer_stream_step = orig_stream_step

    # Patch temporarily
    import types
    model._run_steps = types.MethodType(instrumented_run_steps, model)

    # Run streaming
    partials = []
    model.reset()
    model.set_partial_callback(lambda text: partials.append(text))

    t0 = time.time()
    for start in range(0, len(samples), CHUNK_SIZE):
        chunk = samples[start:start + CHUNK_SIZE]
        model.accept_chunk(chunk)
    final = model.input_finished()
    elapsed = time.time() - t0

    # Restore original _run_steps
    model._run_steps = types.MethodType(original_run_steps, model)

    return final, partials, step_info, elapsed


def run_offline_transcribe(samples, audio_path):
    """Run NeMo's offline model.transcribe() as ground truth."""
    from nemo.collections.asr.models import ASRModel
    from omegaconf import open_dict

    global _offline_model
    if "_offline_model" not in globals() or _offline_model is None:
        print("  Loading offline reference model ...")
        m = ASRModel.from_pretrained(
            "nvidia/nemotron-speech-streaming-en-0.6b", map_location="cpu"
        )
        m.to("cpu")
        m.eval()
        m.encoder.set_default_att_context_size([70, 1])
        decoding_cfg = m.cfg.decoding
        with open_dict(decoding_cfg):
            decoding_cfg.strategy = "greedy"
            decoding_cfg.preserve_alignments = False
            if hasattr(m, "joint"):
                decoding_cfg.greedy.max_symbols = 10
                decoding_cfg.fused_batch_size = -1
        m.change_decoding_strategy(decoding_cfg)
        _offline_model = m

    model = _offline_model

    # Use model.transcribe with the audio file path
    with torch.inference_mode():
        result = model.transcribe([audio_path])

    # Extract text from result
    from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
    if result and isinstance(result[0], list):
        result = result[0]
    if result:
        item = result[0]
        if isinstance(item, Hypothesis):
            return item.text or ""
        return str(item) if item else ""
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Dev_streaming.csv path")
    parser.add_argument("--data-root", required=True, help="SAPC2 data root")
    parser.add_argument("--submission-dir", default=None,
                        help="Directory containing model.py (default: same dir as this script)")
    args = parser.parse_args()

    submission_dir = args.submission_dir or os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, submission_dir)

    print("=" * 70)
    print("Phase 2a: 3-utterance smoke test")
    print("=" * 70)

    # Pick 3 utterances: ~3s, ~8s, ~20s
    print("\nPicking utterances ...")
    picks = pick_utterances(args.manifest, args.data_root, target_lengths=(3.0, 8.0, 20.0))
    for uid, path, dur, _ in picks:
        print(f"  {uid}: {dur:.2f}s  {os.path.basename(path)}")

    # Load our Model class
    print("\n--- Loading streaming Model class ---")
    from model import Model as SubmissionModel

    for i, (uid, audio_path, dur, samples) in enumerate(picks):
        print(f"\n{'=' * 70}")
        print(f"Utterance {i+1}/3: {uid} ({dur:.2f}s, {len(samples)} samples)")
        print("=" * 70)

        # Streaming
        print("\n[STREAMING] Running accept_chunk loop ...")
        streaming_text, partials, step_info, elapsed = run_streaming(
            SubmissionModel, audio_path, samples, verbose_steps=3
        )
        n_chunks = (len(samples) + CHUNK_SIZE - 1) // CHUNK_SIZE
        print(f"  Chunks sent: {n_chunks}, Model steps: {len(step_info)}+, Time: {elapsed:.2f}s")
        print(f"  Final text: \"{streaming_text}\"")
        if partials:
            print(f"  Partials emitted: {len(partials)}")
            print(f"  First partial: \"{partials[0]}\"")

        # Print shape info for first 3 steps
        if step_info:
            print(f"\n  --- Encoder shape trace (first {len(step_info)} steps) ---")
            for si in step_info:
                print(f"  Step {si['step']}:")
                print(f"    input tensor:       {si['input_shape']}")
                print(f"    cache_channel pre:  {si['pre_cache_channel']}")
                print(f"    cache_channel post: {si['post_cache_channel']}")
                print(f"    cache_time pre:     {si['pre_cache_time']}")
                print(f"    cache_time post:    {si['post_cache_time']}")
                print(f"    drop_extra={si['drop_extra']}, keep_all={si['keep_all']}")

        # Offline reference
        print("\n[OFFLINE] Running model.transcribe() ...")
        offline_text = run_offline_transcribe(samples, audio_path)
        print(f"  Final text: \"{offline_text}\"")

        # Compare
        print(f"\n  --- COMPARISON ---")
        match = streaming_text.strip() == offline_text.strip()
        print(f"  Streaming: \"{streaming_text}\"")
        print(f"  Offline:   \"{offline_text}\"")
        if match:
            print(f"  Result:    EXACT MATCH")
        else:
            # Check if close (minor whitespace/token diffs)
            s_tokens = streaming_text.strip().lower().split()
            o_tokens = offline_text.strip().lower().split()
            if s_tokens == o_tokens:
                print(f"  Result:    MATCH (case/whitespace diff only)")
            else:
                # Compute simple token-level edit distance
                common = sum(1 for a, b in zip(s_tokens, o_tokens) if a == b)
                total = max(len(s_tokens), len(o_tokens))
                pct = common / total * 100 if total > 0 else 0
                print(f"  Result:    DIVERGENCE ({common}/{total} tokens match = {pct:.0f}%)")
                if pct < 80:
                    print(f"  *** WARNING: Substantial divergence — likely streaming bug ***")

    print("\n" + "=" * 70)
    print("Phase 2a complete. Review the comparisons above.")
    print("=" * 70)


if __name__ == "__main__":
    main()
