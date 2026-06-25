# 16 — Cache-Aware FastConformer-RNNT: verified facts + characterization & smoke experiment

Model confirmed (user): **Cache-Aware FastConformer-RNNT** (RNN-T, not TDT — matches the exported decoder:
argmax + break-on-blank, no duration head). This doc verifies the load-bearing questions by search, then
specifies the cheap experiment that replaces Parakeet-analogy with Nemotron-direct evidence. Gates
[[14_nemotron_finetune_plan]].

## Verified answers (sources at bottom)
1. **The "prompt" mechanism = LANGUAGE-ID conditioning.** Nemotron-Speech streaming fuses a one-hot
   `target_lang` vector with the FastConformer acoustic embedding before the RNNT decoder (multilingual
   routing). It is NOT a free-text prompt. For English dysarthric finetuning: **pin `target_lang=en` and keep it
   fixed**; the `nemotron-speech-streaming-en-0.6b` export has no language input → already English-pinned.
   Risk reduced from "unknown conditioning" to "keep one constant."
2. **Cache-aware finetuning is a training-config requirement.** `att_context_size={left,right}` in **80ms
   frames**; multi-lookahead models train over a *list* with `att_context_probs` (random per step; first item =
   inference default). NeMo guidance: **evaluate/deploy at the same low-latency streaming setting you'll ship**
   (e.g. `[56,0]` = 80ms chunk, 0ms lookahead). => finetune in cache-aware mode at the deployment context.
   **LEVER:** a smaller right-context lowers TTFT (latency-scored Track 2) at some CER cost — sweep it.
3. **PEFT/adapters mitigate catastrophic forgetting.** NeMo supports adapter/LoRA finetuning for transducers
   (only adapter modules update → avoids forgetting, faster). Literature: "Updating Only Encoders Prevents
   Catastrophic Forgetting"; "Damage Control During Domain Adaptation for Transducers." => the freeze/LoRA arm
   is real and important; **do not default to fully-unfrozen** (that was a Parakeet-1.1B import).
4. **Export path exists.** NeMo exports cache-aware RNNT to **3 ONNX (encoder/decoder/joiner)**, per-component
   int8 → drops straight into our validated offline zip ([[submission-offline-packaging]]).

## Gate-0 COMPLETE (2026-06-25) — MEASURED from the loaded `.nemo` (isolated venv)
Env validated: an isolated `/workspace` venv with `nemo_toolkit[asr]` loads the model
(`EncDecRNNTBPEModel ... successfully restored`, `cuda True`). torchvision is simply *absent* (not broken) —
NeMo ASR doesn't need it. **Measured architecture (use these for the smoke config, not Parakeet guesses):**
- Class **EncDecRNNTBPEModel**; **RNN-T** (`RNNTDecoder`, `pred_rnn_layers=2`; `RNNTJoint`, `joint_hidden=640`).
  TDT ruled out. ✓
- Encoder **ConformerEncoder**, `n_layers=24`, `d_model=1024`, `subsampling=dw_striding` factor **8**, conv_channels 256.
- **att_context_size = `[[70,13],[70,6],[70,1],[70,0]]`** (multi-lookahead; chunk sizes per HF card =
  1120/560/160/80 ms; default/first = `[70,13]`; export/submission uses `[70,6]`=560ms chunk; `[70,0]`=80ms =
  lowest latency). `att_context_style = chunked_limited`. (Corrected: earlier "480ms" was right-context-only;
  the model card's chunk size for [70,6] is 560ms.)
- **`att_context_probs = None`** → trained with uniform sampling across the lookahead list. For finetuning we can
  reweight toward our deployment context (or keep uniform to retain the latency lever).
- **Tokenizer = SentencePiece BPE, `vocab_size=1024`** (+blank=1024 → matches the export's BLANK_ID). Reuse as-is.
- **Loss = `warprnnt_numba` with `fastemit_lambda=0.005`** already enabled (FastEmit → earlier emission / lower
  TTFT). Keep it; it's latency-favorable for Track 2.
- No language/prompt head present (English-only). ✓
**Venv location:** this run built on `/opt` (container disk, 20GB free) = EPHEMERAL. Script now fixed to build on
`/workspace/nemoenv` (persistent) + idempotent reuse → one more build next session, then reuse forever.
**Still to decide (the one real open item):** target-text case/punct (model emits punct+caps; SAP refs are
normalized) — ablate before full training.

## (superseded) earlier partial results — from the HF model card + a failed load attempt
- **Confirmed:** 24 FastConformer layers, **RNNT** decoder (not TDT), 600M, **English-only** (no language head
  to manage — simplest case), native punct+caps.
- **att_context is MULTI-LOOKAHEAD:** `[70,0]`, `[70,1]`, `[70,6]`, `[70,13]` (left=70×80ms=5.6s; right =
  {0,80,480,1040} ms). Submission runs `[70,6]`. **LATENCY LEVER:** finetune/deploy at `[70,0]`/`[70,1]` to cut
  TTFT (Track-2 is latency-scored), trading some CER. Sweep this.
- **d_model / vocab / att_context_probs / tokenizer:** still need the in-`.nemo` `model_config.yaml`
  (`scripts/nemo_characterize.py` now reads it dependency-free via tar — the `.nemo` is downloaded + persisted
  at `/workspace/finetune/nemo_ft/`; run on the next pod window, seconds). NOTE: NeMo's finetune loads d_model/
  vocab/tokenizer from the checkpoint automatically, so these are confirm-only, not blocking.

## ⚠️ REAL BLOCKER surfaced by the failed Gate-0 (cheap catch, high value)
`pip install nemo_toolkit[asr]` into the pod's `pytorch:2.4.0` image **upgraded torch 2.4.1 → 2.12.1**, breaking
the pre-built torchvision/torchaudio (`RuntimeError: operator torchvision::nms does not exist`). **A working NeMo
training env is the prerequisite for ALL of Gate 1/2 (any finetuning), and naive pip install does NOT give one.**
Options, to decide before the next pod run:
- (A) **Launch the pod from an NVIDIA NeMo container** (`nvcr.io/nvidia/nemo:*`) — our `/workspace` MooseFS volume
  persists across images, so we keep all data + get a consistent NeMo/torch/vision/audio stack. Cleanest.
- (B) Pin a NeMo version compatible with the image's torch 2.4.1 (risk: may be too old to load a Jan-2026 model).
- (C) After NeMo upgrades torch, reinstall matching torchvision+torchaudio for the new torch (finicky version match).
Recommendation: (A).

## Remaining unknowns — resolved ONLY by loading the actual checkpoint (Gate 0)
- Exact `att_context_size` list of the released `en-0.6b` (export used `[70,6]` ≈ 5.6s left / 480ms right) and
  whether it is multi-lookahead (has `att_context_probs`).
- Whether the en checkpoint instantiates the language/prompt head at all (English-only may omit it).
- Tokenizer (BPE vocab/size) — reuse as-is for domain finetune.
- Confirm decoder = RNNT (TDT ruled out by the export, but verify in cfg).
- The `limit_train_batches`≈1000 trap (NeMo issue #15782).

## THE EXPERIMENT (cheap → converts analogy to measurement)
**Gate 0 — Characterize** (`scripts/nemo_characterize.py`, minutes, CPU ok): download
`nemotron-speech-streaming-en-0.6b.nemo` (2.47 GB), load in NeMo, dump: decoder type, `att_context_size`(+probs)
+ style, encoder layers/d_model, tokenizer vocab size, presence of any language/prompt head, and the default
streaming config. Output pins every hyperparameter that the Parakeet evidence could not.

**Gate 1 — Overfit a single batch** (GPU, ~minutes): ~20–50 hard ALS/CP utts, **augmentation OFF**, train CER→~0.
If it can't, the data/target/prompt/loss wiring is broken — STOP. Distinct from and prior to the smoke.

**Gate 2 — Direct Nemotron smoke** (GPU, ~1 hr): tiny-subset finetune (a few k utts, few epochs) of the actual
`.nemo`, cache-aware mode at the deployment `att_context`. Measure on speaker-disjoint internal-dev via the
faithful pipeline: (a) do **Nemotron's own empties/failures move**? (b) does **mild-speech (Parkinson's) CER
regress**? Run 2–3 arms: full-unfrozen vs encoder-only vs LoRA. This is the FIRST Nemotron-specific evidence and
the true go/no-go for the full plan.

**Decision:** Gate-2 shows failures move AND mild speech preserved (best arm) → green-light the full finetune
(§14). Else → stop; the Parakeet analogy did not hold for our 0.6B streaming model, and the zipformer path
(beam-4 shipped + LM) is the line.

## Env note
Finetuning needs `nemo_toolkit[asr]` + GPU on the pod (NOT the offline submission env — that's where NeMo's
numpy-ABI break bit us; for *training* it's fine and expected). Pod has network for the download/install.

Sources: prompt=language conditioning — https://huggingface.co/blog/nvidia/fine-tuning-nemotron-35-asr ,
https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b ; att_context/cache-aware finetune —
https://huggingface.co/nvidia/stt_en_fastconformer_hybrid_large_streaming_multi ,
https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html ; PEFT/forgetting —
https://arxiv.org/pdf/2210.03255 , https://arxiv.org/pdf/2207.00216 ; ONNX export —
https://github.com/NVIDIA-NeMo/NeMo (cache_aware_streaming configs) .
