# Fresh NeMo → ONNX export vs danielbodart prebuilt: 1.5 CER point gap

## Observed delta (2026-05-27, Dev_streaming, 123 utts, sclite scoring)

| Variant | CER% | WER% | Source |
|---|---|---|---|
| ONNX FP32 [70,6] | **24.45** | 29.90 | danielbodart prebuilt |
| ONNX FP32 [70,6] | **~26**   | ~33   | our fresh NeMo export (jiwer-scored 26.30 → sclite delta ~−1.5 → ≈24.7–26 sclite, full sclite re-eval pending on our export) |

Approximate sclite re-score of our fresh export would be ~25–26% CER, still ~1.5 CER points worse than danielbodart's prebuilt FP32.

## Hypotheses to check during Phase 2 re-export

1. **Mel preprocessing config drift.** danielbodart's `nemo_export_onnx.py` dumps `preprocessor.config` and `filterbank.bin` and uses them at inference time. Our pipeline reads mel via PyTorch `model.preprocessor`. If their mel filter banks (slaney norm, n_mels=128, etc.) differ in any parameter, the encoder sees different features.
   - **Action**: diff the `preprocessor.config` they ship against our `model.cfg.preprocessor` dict. Look at `dither`, `pad_to`, `normalize`, `preemph`, `mel_norm`, `n_fft`, `win_length`, `hop_length`, `window`.

2. **ONNX op fusion differences.** Paper §6.1 says multi-head attention fusion is a key speedup. NeMo's `model.export(check_trace=False)` may run different ONNX passes than danielbodart's pipeline. Run `onnx.shape_inference` and dump node-type histograms on both — look for fusion ops (e.g. `Attention`, `MultiHeadAttention`, `LayerNormalization`) present in one but not the other.
   - **Action**: `onnx.helper`-based node-type count diff between `encoder_model.onnx` (theirs) and `encoder-nemotron.onnx` (ours).

3. **opset version.** Default opset for `model.export()` in NeMo 2.5.3 may differ from what danielbodart used (probably 1.x). Higher opset can change fused-op availability.
   - **Action**: read `model.opset_import` from both ONNX files.

4. **drop_extra_pre_encoded baked value.** Both ONNX should bake `drop_extra=2` for [70,6], but verify by inspecting any `Slice` ops on the audio_signal input near the encoder entry.

5. **Decoder shape semantics.** danielbodart's decoder_joint uses fixed `target_length=1` and `targets=[[token]]`. Confirm our exported decoder_joint accepts the same shapes; check whether our shape is more "dynamic" (which can disable ORT graph optimizations and shift accuracy slightly).

6. **GPU vs CPU export device.** danielbodart's script does `model.cuda()` before export. Ours does too (via `--device cuda` in phase1a_export_streaming.py). But verify by comparing fp32 weights byte-for-byte on a few tensors to rule out fp16 mixed-precision contamination.

## Status

**Parked.** Not blocking Phase 2 — Phase 2 will re-export from .nemo with k-quant tooling (Olive) anyway, so we naturally get a chance to investigate during that work. If we close the gap, document the fix here. If not, ship int8/int4 k-quant on top of *our* fresh export and note the residual delta.

## See also

- danielbodart's export pipeline: `https://github.com/danielbodart/nemotron-speech-600m-onnx/blob/main/nemo_export_onnx.py`
- Our export script: `scripts/nemotron_bench/phase1a_export_streaming.py`
- Phase 1b ledger rows: `track2_starting_kit/experiments/results.jsonl` (search "danielbodart")
