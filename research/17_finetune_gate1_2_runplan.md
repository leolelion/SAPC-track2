# 17 — Gate-1 / Gate-2 finetune run plan (configs ready for review)

Built from the MEASURED config ([[16_nemotron_characterization_and_smoke]]): EncDecRNNTBPEModel, 24L Conformer
d1024, RNN-T (2-layer pred RNN, joint 640), BPE vocab 1024, att_context `[[70,13],[70,6],[70,1],[70,0]]`,
FastEmit 0.005, English-only. Scripts: `prep_nemo_manifest.py`, `nemo_finetune.py`. **Nothing here runs training
without sign-off.**

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
