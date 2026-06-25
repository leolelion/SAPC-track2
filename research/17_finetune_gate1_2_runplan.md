# 17 — Gate-1 / Gate-2 finetune run plan (configs ready for review)

Built from the MEASURED config ([[16_nemotron_characterization_and_smoke]]): EncDecRNNTBPEModel, 24L Conformer
d1024, RNN-T (2-layer pred RNN, joint 640), BPE vocab 1024, att_context `[[70,13],[70,6],[70,1],[70,0]]`,
FastEmit 0.005, English-only. Scripts: `prep_nemo_manifest.py`, `nemo_finetune.py`. **Nothing here runs training
without sign-off.**

## Independent review applied (v2, 2026-06-25)
A cold NeMo-engineer review found real bugs; `nemo_finetune.py` is fixed:
- **Explicit optimizer** (adamw+cosine) replacing the inherited scheduler + **`setup_optimization()`** call —
  fixes the "optimizer never rebuilt" bug AND the Noam-LR-is-a-scale trap (raw lr now means raw lr).
- **Single flat training context** via `encoder.set_default_att_context_size()` (the assignment to
  `cfg.encoder.att_context_size` did NOT re-take on the built encoder). Multi-lookahead + `att_context_probs`
  deferred to the full run. Smoke trains at `[70,1]` (low-latency lever) by default.
- **Punctuation-stripped overfit CER** (model emits punct+caps; targets are normalized) — else Gate 1 fails for
  the wrong reason. **Dataloader-length guard** for the smoke run (the ~1000-cap trap). Overfit utts now **≤8s**.
- Still TODO for the full run (not gates): **val-WER logging + checkpoint averaging + a LoRA arm**; and a
  decision to possibly drive NeMo's official `examples/asr/speech_to_text_finetune.py` via Hydra instead of the
  custom script (it handles optim/data/trainer wiring natively — the safer long-term path).

## Step A0 — PRE-FLIGHT (1 min, settles the 2 unverified items) [v2]
First lines of the overfit run already print `base optim cfg` (→ confirm scheduler name / whether it was Noam)
and `att_context now = …` (→ confirm `set_default_att_context_size` actually re-took). Read these before trusting
the run. If att_context didn't re-take or optim looks wrong → switch to the official Hydra script.

## Step A — persistent venv + data prep (cheap, no training)
    source /workspace/nemoenv/bin/activate         # idempotent build if absent (one-time, on /workspace)
    python3 /workspace/prep_nemo_manifest.py \
      --train-csv /workspace/SAPC2/manifest/Train.csv --data-root /workspace/SAPC2 \
      --out-dir /workspace/finetune/nemo_ft/manifests --target norm_text_without_disfluency
Outputs train.json / dev_internal.json (SPEAKER-DISJOINT) / overfit.json (30 hard ALS+CP utts) /
smoke_train.json (~4k). Prints hours + #speakers — **this also answers the still-open "how much data do we
have" question.**

## Step B — GATE 1: overfit a single batch (the wiring proof)
    python3 /workspace/nemo_finetune.py --mode overfit --freeze full \
      --train-json .../overfit.json --val-json .../dev_internal.json \
      --out-dir /workspace/finetune/nemo_ft/gate1
- Aug OFF, single batch hammered for 400 steps, LR 1e-3.
- **PASS = mean CER on the overfit set → ~0%.** If it can't memorize 30 utts, the pipeline (data/target/loss/
  att_context/tokenizer) is broken → STOP and fix. ~minutes of GPU. This is where NeMo-API specifics get shaken
  out (the `[VERIFY]` spots in `nemo_finetune.py`: att_context list-of-lists accepted; `transcribe()` return type).

## Step C — GATE 2: direct Nemotron smoke (the real go/no-go), 2 arms
    # arm 1: full unfrozen (Parakeet prior)
    python3 /workspace/nemo_finetune.py --mode smoke --freeze full \
      --train-json .../smoke_train.json --val-json .../dev_internal.json \
      --att-context "[[70,6],[70,1],[70,0]]" --out-dir .../gate2_full
    # arm 2: encoder-only (anti-forgetting; lit: "updating only encoders prevents forgetting")
    python3 /workspace/nemo_finetune.py --mode smoke --freeze encoder_only ... --out-dir .../gate2_enc
- ~4k utts, 3000 steps, LR 2e-4, cache-aware multi-lookahead training INCLUDING [70,0] (keeps the low-latency
  deploy option strong — the Track-2 latency lever).
- Then EXPORT each arm → ONNX → faithful-harness eval on Dev (reuse `nemotron_repro_*`), broken down by etiology:
  **(a) do Nemotron's OWN empties/ALS-CP failures drop? (b) does Parkinson's (mild) CER stay ≤ ~11% (no
  catastrophic forgetting)?**

## Decision criteria (gates the FULL run, [[14_nemotron_finetune_plan]])
- Gate 1 fails → pipeline bug; fix before anything else.
- Gate 2: best arm shows empties dropping AND mild speech preserved on speaker-disjoint dev → GREEN-light the
  full finetune (more data, more epochs, avg-5, att_context sweep, target-text ablation).
- Gate 2: no movement, or mild-speech regresses badly in both arms → the Parakeet analogy didn't transfer to our
  0.6B streaming model → stop; zipformer path (beam-4 shipped + LM) is the line.

## Open design choice to resolve during Gate 2 (not before)
- **Target text:** start with `norm_text_without_disfluency` (matches the scorer). If the model fights unlearning
  its native punct+caps, ablate `--target text` (raw cased) and rely on the scorer's normalizer. One sweep.
- **att_context for deploy:** [70,6] (current) vs [70,1]/[70,0] (lower TTFT). Measure CER×latency after Gate 2.

## Cost / safety
- Gate 1 ≈ minutes GPU; Gate 2 ≈ ~1–2 h GPU per arm. Persist all to `/workspace/finetune/nemo_ft/` (MooseFS).
- Use the GPU-wait autorun pattern (port-race fixed). Stop the pod after each gate; report before escalating.
