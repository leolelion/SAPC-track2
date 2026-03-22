#!/bin/bash
# Emformer RNNT: pre-download model weights.
#
# torchaudio is pre-installed in the competition Docker image
# (xiuwenz2/sapc2-runtime, PyTorch 2.5.0+cu124 + torchaudio).
# This script ONLY downloads the model weights — no venv, no pip install.
#
# If torchaudio is NOT available in your local environment, install it:
#   pip install torchaudio  (matching your torch version)
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Emformer RNNT: installing sentencepiece ==="
pip install --quiet sentencepiece

echo "=== Emformer RNNT: checking torchaudio ==="
python3 -c "
import torchaudio
print(f'torchaudio {torchaudio.__version__}: OK')

from torchaudio.pipelines import EMFORMER_RNNT_BASE_LIBRISPEECH
bundle = EMFORMER_RNNT_BASE_LIBRISPEECH
print(f'  sample_rate:          {bundle.sample_rate} Hz')
print(f'  segment_length:       {bundle.segment_length} frames ({bundle.segment_length * bundle.hop_length / bundle.sample_rate * 1000:.0f}ms)')
print(f'  right_context_length: {bundle.right_context_length} frames ({bundle.right_context_length * bundle.hop_length / bundle.sample_rate * 1000:.0f}ms)')
print(f'  hop_length:           {bundle.hop_length} samples ({bundle.hop_length / bundle.sample_rate * 1000:.0f}ms)')
"

echo "=== Emformer RNNT: pre-downloading model weights ==="
python3 -c "
import torch
from torchaudio.pipelines import EMFORMER_RNNT_BASE_LIBRISPEECH

bundle = EMFORMER_RNNT_BASE_LIBRISPEECH
print('Downloading Emformer RNNT weights ...')
# In torchaudio 2.6+, get_model() was removed.
# The model lives inside the decoder as decoder.model.
decoder = bundle.get_decoder()
token_processor = bundle.get_token_processor()
params = sum(p.numel() for p in decoder.model.parameters()) / 1e6
print(f'Model loaded: {params:.1f}M params')
print('Decoder and token processor loaded.')
print('Weights cached in torchaudio hub cache.')
"

echo "=== Emformer RNNT: verifying streaming interface ==="
python3 -c "
import torch
import numpy as np
from torchaudio.pipelines import EMFORMER_RNNT_BASE_LIBRISPEECH

bundle = EMFORMER_RNNT_BASE_LIBRISPEECH
seg  = bundle.segment_length        # 16 frames
ctx  = bundle.right_context_length  # 4 frames
hop  = bundle.hop_length            # 160 samples/frame

# With center-padded STFT: T frames need (T-1)*hop samples.
window_samples = (seg + ctx - 1) * hop   # 3040 samples -> 20 frames
stride_samples = (seg - 1) * hop          # 2400 samples (advance per step)
print(f'window_samples: {window_samples} ({window_samples / bundle.sample_rate * 1000:.0f}ms)')
print(f'stride_samples: {stride_samples} ({stride_samples / bundle.sample_rate * 1000:.0f}ms per step)')

feature_extractor = bundle.get_streaming_feature_extractor()
decoder = bundle.get_decoder()
token_processor = bundle.get_token_processor()

# Feed 1D tensor (not batched) — required in torchaudio 2.6+
segment = torch.zeros(window_samples)      # 1D, not (1, N)
features, lengths = feature_extractor(segment)
print(f'features shape: {features.shape}  lengths: {lengths.item()}')
assert features.shape[0] == seg + ctx, f'Expected {seg+ctx} frames, got {features.shape[0]}'

# decoder.infer takes (features, lengths, beam_width, state, hypothesis) — no model arg
state = None
hypothesis = None
with torch.inference_mode():
    hypos, state = decoder.infer(features, lengths, 1, state=state, hypothesis=hypothesis)
# hypothesis is List[Tuple[tokens, score, hypo_state, lm_score]]
tokens = hypos[0][0]
text = token_processor(tokens)
print(f'First step output: {text!r}  (empty expected on silence)')
print('Streaming interface verified.')
"

echo ""
echo "=== Emformer RNNT: setup complete ==="
echo ""
echo "Next steps:"
echo "  cd $(dirname "$0")"
echo "  python3 test_streaming.py --mock   # interface test (no model needed)"
echo "  python3 test_streaming.py          # real model test (requires download)"
echo "  python3 test_streaming.py --audio /workspace/SAPC2/processed/Dev/<utt>.wav"
