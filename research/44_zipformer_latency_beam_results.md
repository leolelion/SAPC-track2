# 44 — Zipformer beam/latency pod results (2026-07-21)

> Faithful-harness results from a gated RunPod session. Every number below is from the
> organizers' real pipeline: `track2_starting_kit/local_decode.py` (both passes) →
> `evaluate.sh --split Dev_streaming` → SCTK sclite (min-two-refs, clip@1). No proxy scorer.
> Artifacts copied back to `experiments/pod_devstream_20260721/`. Pod stopped after copy.

## Premise correction (important — the plan targeted the wrong model)
`OPUS_STEP_BY_STEP_PLAN.md` assumed the kit `track2_starting_kit/streaming_zipformer/`
(icefall-python, `weights/exp/epoch-30.pt`) was the shipped A1. **It is not.** On the pod:
- `epoch-30.pt` (Mar 18, 1.06 GB) is the **raw LibriSpeech pretrained** HF download.
- The **finetuned A1** is `/workspace/finetune/exp/a1_sp_backup/epoch-16.pt` — a 16-epoch
  finetune (`run_a1.sh`: `--num-epochs 16 --do-finetune 1 --finetune-ckpt .../standard/epoch-0.pt
  --chunk-size 16 --left-context-frames 128`), exported to the **shipped sherpa-onnx int8 ONNX**
  at `/workspace/finetune/onnx/a1/`. The live #1 submission (`zf_a1_beam4_sub`) is this
  sherpa-onnx model, `max_active_paths=4`.

So the sweep was run on the **correct shipped runtime** (sherpa-onnx int8 A1), not the kit clone.
The config-only kit variants staged locally (research/43) are moot for measurement — they clone
the un-finetuned model. (They remain valid as a *pattern*; the real knobs live in the sherpa export.)

## Results — 3-point surface on Dev_streaming (123 utts)

| config | CER | WER | TTFT p50 | TTLT p50 | mean(TTFT,TTLT) p50 |
|---|---:|---:|---:|---:|---:|
| beam-4 (chunk16, **shipped anchor**) | 12.14% | 16.96% | 1155 ms | 67 ms | 611 ms |
| **beam-8** (chunk16) | **11.62%** | 16.13% | 1157 ms | 68 ms | 612 ms |
| **cs8** (chunk8, beam-4; re-export of epoch-16) | 13.85% | 18.31% | **926 ms** | 81 ms | **504 ms** |

## Findings
1. **beam-8 Pareto-dominates beam-4.** −0.52 CER / −0.83 WER at **identical latency**
   (+2 ms TTFT, +1 ms TTLT = noise). The int8 model's RTF is tiny, so doubling beam width
   barely moves wall-clock. This settles the long-open G4 ("beam-8 latency never measured")
   and matches research/09's ~−0.68 CER prediction. It is a **strict upgrade** — it cannot
   lose on the Pareto axis and improves CER. **New ship-upgrade candidate.**
2. **cs8 opens the low-latency frontier corner.** Halving the decode chunk trades **+1.71 CER**
   for **−107 ms mean / −229 ms TTFT (−20%)**. A distinct **non-dominated** point (lower latency,
   higher CER) that a CER-chasing competitor cannot dominate. Optional 2nd frontier submission
   (frontier prize splits across all non-dominated teams). Export was clean (int8, ~1 min) and
   smoke output was coherent — the chunk-8 regime (untrained; model trained at chunk-16) decodes fine.

## Caveats (do not over-read)
- **Dev_streaming is an easy subset** (beam-4 CER 12.14% here vs 21.28% on Test1). Absolute
  numbers are NOT Test-comparable; only the **deltas between configs on the same ruler** are load-bearing.
- The beam-8 delta (−0.52) and cs8 trade (+1.71) may differ on the **severe tail / Test**.
  Beam helping is directionally consistent with prior evidence (research/09: beam helps hardest
  etiologies most), which de-risks beam-8; cs8's CER cost on the severe tail is unmeasured.
- The **speaker-disjoint heldout gate** (research/43 tool) was NOT used here — Dev_streaming is
  the organizers' streaming manifest, a fixed subset. Run the heldout gate + Dev_diag before any submission.
- Not measured: cs8 at beam-8 (the low-latency + wider-beam combo), other chunk/left-context points.

## Recommendation
1. **Ship beam-8** as the new #1: it's a runtime one-liner (`max_active_paths=8`) on the exact
   shipped model, strictly dominates the live beam-4, near-zero risk. Confirm on Dev_diag +
   speaker-disjoint heldout (and ideally a Test submission) before flipping the live entry.
2. **Consider cs8 (chunk-8) as a 2nd, low-latency frontier submission** — but first check its
   CER on the severe tail (Dev_diag); if the tail doesn't blow up, it's cheap frontier insurance.
   Likely best as **cs8 + beam-8** (combine the two independent wins) — measure that combo next.
3. Next cheap pod pass (all on sherpa-onnx, epoch-16): cs8+beam8 combo, and a Dev_diag +
   heldout run of beam-8 to confirm the delta holds off the easy subset.

## Pod session bookkeeping
- Pod `1ppb7l0i5xuna8` (H100), `/workspace/SAPC-template` + `/workspace/SAPC2` data + `/workspace/SCTK`.
- sherpa-onnx 1.13.4; scorer deps (`tqdm`, `torchmetrics`) had to be pip-installed into system python3
  (the pod's prior runs used a proxy scorer, so the real evaluate.sh had never run there).
- cs8 ONNX re-export via `export-onnx-streaming.py --epoch 16 --avg 1 --use-averaged-model 0
  --causal 1 --chunk-size 8 --left-context-frames 128 --enable-int8-quantization 1`; new ONNX in
  `/workspace/finetune/exp/a1_sp_backup/*chunk-8*`, sub at `/workspace/finetune/zf_a1_cs8_sub`.
- Artifacts (predict.csv, partial.json, accuracy/latency logs, model snapshots) →
  `experiments/pod_devstream_20260721/exp_a1_{beam4,beam8,cs8}_devstream/`.
- **Pod stopped** (`runpodctl stop pod`) after copy-back; `/workspace` volume persists for reuse.
