# Inference Debugging: Empty Transcriptions from Fine-tuned Zipformer

## Context
- Model: Streaming Zipformer2 (66M params, causal, chunk-size 16, left-context 128)
- Fine-tuned 1 epoch on SAPC2 data (336k utts, 743h dysarthric speech)
- Starting from LibriSpeech epoch-30.pt checkpoint
- Training loss: 7.37 → 0.035 (healthy convergence)
- Framework: icefall/k2, exported to ONNX via `export-onnx-streaming.py`

---

## Problem 1: sherpa-onnx produces empty strings

**Symptom:**
```python
recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(...)
stream = recognizer.create_stream()
stream.accept_waveform(16000, audio)
stream.input_finished()
while recognizer.is_ready(stream):
    recognizer.decode_stream(stream)
result = recognizer.get_result(stream)  # returns "" always
```

**What we tried:**
- Feeding audio as one block vs. 0.1s chunks
- Extending tail padding to 1.5s
- Checked `is_ready` returns True and decode loop runs (26 iterations)
- Checked audio is valid (16kHz, correct amplitude)
- sherpa-onnx version 1.12.35

**Status:** Not resolved — moved to lower-level ONNX testing to isolate

---

## Problem 2: Direct ONNX inference (onnxruntime) also produces empty/wrong output

**ONNX encoder metadata:**
```json
{
  "T": "45",
  "decode_chunk_len": "32",
  "left_context_len": "128,64,32,16,32,64"
}
```

**First attempt (wrong):** Prepended zero-padding to each chunk instead of using sliding window
- Root cause: Misread the inference algorithm. The encoder takes T=45 frames total, advances by decode_chunk_len=32 each step. The first 13 frames of each chunk are the actual PREVIOUS audio frames (not zeros).
- Result: Only 2 tokens emitted (token 34 `▁` and token 2 `<unk>`)

**Second attempt (sliding window):** Correct windowing but still `⁇` output
- Used `features[i*32:(i*32)+45]` as input each step
- Caches updated via `name[4:]` to strip `new_` prefix
- Result: Same `⁇` output

**Logit analysis (direct onnxruntime):**
```
t=0: argmax=34 (▁), logit=6.11, blank=5.86
t=1: argmax=34 (▁), logit=6.11, blank=5.86
...same for all frames
```
- Token 34 is `▁` (word-boundary/space character)
- Only margin of 0.25 between top token and blank — suspicious

---

## Problem 3: PyTorch model also produces empty output

**Symptom:** Even calling the PyTorch model directly gives empty transcriptions:
```python
enc_out, enc_lens = model.encoder(feat, feat_lens)
hyp = greedy_search(model, enc_out, enc_lens)
# hyp = [] (empty)
```

**Logit analysis (PyTorch, direct frame-by-frame):**
```
t=0: argmax=34, logit=6.46, blank=5.52
t=1: argmax=2 (<unk>), logit=14.56, blank=-2.22  ← after emitting ▁
t=2: argmax=0 (blank), blank=7.14
t=3: argmax=0 (blank)
... blank for all remaining frames
```
- Model emits `▁` (token 34) at frame 0
- Then emits `<unk>` (token 2) at frame 1 (after the decoder context changes)
- Then predicts blank for everything else
- Total tokens emitted: 2

**Critical discovery:** `enc_out.shape = (1, 865, 512)` — 865 encoder output frames for an 8.67s utterance. This is WRONG. The encoder should subsample by ~4x → ~215 frames.

**Root cause identified:** The `AsrModel` has two separate components:
- `model.encoder_embed` — Conv2dSubsampling (T → (T-7)//2, then //2 again = ~T/4)
- `model.encoder` — Zipformer2 (takes already-subsampled input)

Calling `model.encoder(feat, feat_lens)` **bypasses the subsampling**. The correct call is:
```python
x, x_lens = model.encoder_embed(feat, feat_lens)  # subsample
enc_out, enc_lens = model.encoder(x, x_lens)       # encode
```

**Status:** Identified but not yet tested — stopped to create this document

---

## What to redo from scratch

### 1. Fix the PyTorch inference test
Call `encoder_embed` first:
```python
x, x_lens = model.encoder_embed(feat, feat_lens)
enc_out, enc_lens = model.encoder(x, x_lens)
hyp = greedy_search(model, enc_out, enc_lens)
```
This is the correct path. If this works, the model weights are fine.

### 2. Fix the ONNX sliding window inference
The onnxruntime test was using the wrong inference approach. The reference icefall script (`onnx_pretrained-streaming.py`) uses kaldifeat's `OnlineFbank` which handles framing automatically. Without kaldifeat, manually implement the sliding window:
- Feed `T=45` fbank frames per encoder call
- Advance by `offset=decode_chunk_len=32` frames per step
- This means the first `T-offset=13` frames of each chunk overlap with the previous

### 3. Investigate sherpa-onnx empty output
Once ONNX direct inference is confirmed working, compare with sherpa-onnx. Likely issues:
- Feature normalization mismatch (`normalize_samples` param in sherpa-onnx)
- sherpa-onnx 1.12.35 may have a regression for this model type

### 4. Consider re-running fine-tuning with different settings
The current fine-tuning used:
- `--max-duration 300` (reasonable)  
- `--num-epochs 1` (may need 2-3 epochs for dysarthric adaptation)
- `--base-lr 0.0045` (1/10th of original, standard for fine-tuning)

The very fast loss convergence (0.035 in one epoch) is unusual and may indicate the model is fitting the LibriSpeech-style transcriptions in SAPC2 rather than actually adapting to dysarthric acoustics. Consider:
- Running 2-3 epochs
- Lower LR: `--base-lr 0.001`
- Check `best-valid-loss.pt` vs `epoch-1.pt`

---

## File locations on pod

```
/workspace/finetune/exp/standard/epoch-1.pt          # fine-tuned checkpoint
/workspace/finetune/exp/standard/epoch-0.pt          # original LibriSpeech epoch-30
/workspace/finetune/exp/standard/best-valid-loss.pt  # best validation checkpoint
/workspace/finetune/onnx/standard/encoder.onnx       # exported ONNX
/workspace/finetune/onnx/standard/decoder.onnx
/workspace/finetune/onnx/standard/joiner.onnx
/workspace/finetune/onnx/standard/tokens.txt
/workspace/finetune/onnx/standard/bpe.model
/workspace/finetune/data/sapc2_train_cuts.jsonl.gz   # lhotse manifests
/workspace/finetune/data/sapc2_dev_cuts.jsonl.gz
/workspace/SAPC2/manifest/Train.csv                  # 336k utterances
/workspace/SAPC2/manifest/Dev.csv
```
