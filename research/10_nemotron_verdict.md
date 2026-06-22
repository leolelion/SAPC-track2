# 10 — Nemotron-600M verdict for Track-2 (CONCLUDED)

Decision (2026-06-22): **stop investing in Nemotron-600M for Track-2.** Packaging is solved; the wall is SPEED.
See memory [[nemotron-higher-ceiling]], [[submission-offline-packaging]].

## What we learned (rigorous, end-to-end)
1. **Packaging/gate PASSED.** Test-uploaded the 821 MB self-contained zip → **uploaded + ran** (10521/10521
   predictions, no crash). Size is NOT the blocker; the numpy-ABI fix (drop NeMo, local-mel, bundled ORT
   wheel) works on the real Codabench worker. The 7 prior upload failures were the NeMo numpy-ABI crash — fixed.
2. **Test1 result: CER 51.52% / WER 58.99%**, TTFT p50 2270 ms / p90 6205 ms (worker).
3. **Diagnosis — reproduced the exact submission on Dev 2k + official scorer → CER 24.96%** (matches the
   25.07% NeMo-decode). So the submission path is CORRECT (not a bug, not punctuation — the official
   EnglishTextNormalizer strips punctuation/casing). **Dev≈25% on our fast pod vs Test1=51% on the slow worker
   ⇒ the Test1 collapse is speed-induced truncation/timeout.** The stack is RTF≫1 (couldn't finish 2000 utts
   in 33 min on a 192-core box).

## Root cause of the slowness
The inference stack is a **hand-rolled Python cache-aware loop + per-frame×per-symbol RNN-T greedy** (many
tiny `decoder.run()` calls) on **ONNX Runtime CPU**, int8 ONNX, torch-based local mel. The bottleneck is
plausibly the **Python decode loop overhead**, not the model's FLOPs. Potentially fixable via a **native
runtime** (sherpa-onnx Nemotron-3.5 streaming / C++ greedy / TDT frame-skip), but **as-built it's unusable**
on the CPU-real-time constraint.

## Why we stop
To make Nemotron competitive needs **two** big, uncertain efforts: (a) a native-runtime speed rebuild AND
(b) finetuning (zero-shot 24.96% Dev is already worse than A1's 21.61%). Meanwhile the **zipformer is #1 and
improving for free** (beam-4 −2.6 CER ready; RNN-LM fusion next). Lower-risk, higher-EV.

## Banked / reusable
- Offline packaging recipe ([[submission-offline-packaging]]) — works for any ONNX model.
- The local-mel reimplementation of NeMo's preprocessor (exact: preemph 0.97, n_fft 512, win 400, hop 160,
  128 mel, log; bundled fb+window) — validated identical to NeMo.
- Conclusion: the ~150 MB / CPU-real-time budget favors small fast models (zipformer), not 600M FastConformers.
