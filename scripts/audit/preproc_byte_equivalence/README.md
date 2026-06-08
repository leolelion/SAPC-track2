# Preprocessor byte-equivalence audit

**Companion to:** [`track2_starting_kit/nemotron_streaming_nemo_free/`](../../../track2_starting_kit/nemotron_streaming_nemo_free/).

## Purpose

Validate that the hand-rolled `MinimalMelPreprocessor` (pure torch +
numpy) shipped inline in `nemotron_streaming_nemo_free/model.py`
produces byte-equivalent output to NeMo's
`AudioToMelSpectrogramPreprocessor` with the exact Nemotron checkpoint
args. This is a hard precondition for shipping the nemo-free
submission — without it, we can't claim Dev_10k CER 21.59 carries over
unchanged.

## Files

- `minimal_preprocessor.py` — standalone version of the preprocessor
  (identical to the inline class in `model.py`).
- `test_equiv.py` — loads a WAV, runs both preprocessors, prints
  shape + max/mean/rel diff + per-frame and per-mel-bin localization.
  Threshold: max abs diff < 1e-4.

## Result (3 Dev_streaming utterances, ran 2026-06-08 inside `xiuwenz2/sapc2-runtime:latest`)

| File | shape match | **max abs diff** | mean abs diff | max rel diff | verdict |
|---|---|---|---|---|---|
| `..._1052_4303.wav` (3.49 s, 350 frames) | ✓ | **1.9e-6** | 1.8e-8 | 2.3e-7 | PASS |
| `..._1053_4303.wav` (4.73 s, 474 frames) | ✓ | **1.9e-6** | 1.8e-8 | 3.7e-7 | PASS |
| `..._1054_4303.wav` (3.57 s, 358 frames) | ✓ | **1.9e-6** | 1.3e-8 | 1.2e-7 | PASS |

The 1.9e-6 residual is fp32 rounding noise. End-to-end consequence on
the 5-utt decode: **predict.csv byte-identical to the prior
nemo-based baseline** (SHA `89c828cb…`).

## How to re-run

```bash
docker run --rm --platform linux/amd64 -v $(pwd):/work -w /work \
  -v <repo>/dev100_bundle:/data:ro \
  xiuwenz2/sapc2-runtime:latest bash -c '
    SYS_TORCH=$(python3 -c "import torch; print(torch.__version__)")
    echo "torch==${SYS_TORCH}" > /tmp/c.txt
    python3 -m pip install --no-cache-dir --prefer-binary -q \
      -c /tmp/c.txt "omegaconf>=2.3" "huggingface_hub>=0.24" \
      sentencepiece "nemo_toolkit[asr]>=2.5.0,<2.6"
    python3 /work/test_equiv.py /data/audio/0a71bb2c-..._1052_4303.wav
  '
```

Setup install runs once; the equivalence check itself is ~1 second.

## Iteration history

1. **First attempt: torchaudio `melscale_fbanks(mel_scale='slaney', norm='slaney')`.**
   Max abs diff ~1.4-1.7e-4 across 3 files — just over threshold.
   Error concentrated in higher mel bins. Root cause: torchaudio
   computes the filterbank entirely in fp32; mel-to-hz inversion at
   high mel values loses precision at fp32.
2. **Second attempt: hand-rolled fp64 slaney filterbank** (kept in
   `minimal_preprocessor.py`). Pure numpy fp64 throughout, cast to
   fp32 only at the very end. Max abs diff dropped to 1.9e-6 — well
   under threshold. PASS.
