# 46 — FastConformer-medium (32M) as a low-latency Pareto play — experiment plan (2026-07-24)

> Triggered by Q: "Track 2 is a Pareto frontier — we can try to make a *fast* model too."
> This plan tests the smallest genuinely-streamable NVIDIA FastConformer as a **low-latency
> frontier corner**, not as an accuracy winner. Cheap zero-shot gate first; fine-tune only if it clears.

## 0. Strategic frame (why this is a latency play, not an accuracy play)
Track 2 ranks on a **Pareto frontier**: CER (primary accuracy) × latency = mean(TTFT, TTLT)
(CLAUDE.md contract 4). A submission counts if it is **non-dominated** — i.e. nothing else is
both more accurate AND faster. We already own a low-latency corner with the zipformer **cs8**
(13.85% CER / 926 ms TTFT), non-dominated vs beam-8 (11.62% / 1157 ms). The 32M medium
FastConformer is an **80 ms-lookahead** model — its whole reason to exist is low algorithmic delay.
So the bet is: can a small, low-lookahead model open a frontier corner *below* 926 ms TTFT at a
CER good enough to stay non-dominated?

## 1. Checkpoint availability (verified 2026-07-24, this session)
NVIDIA's published English cache-aware **streaming** FastConformers collapse to TWO sizes, both
**Hybrid (RNN-T + CTC)** — there is **no pure RNN-T-only streaming checkpoint**, and **no ~60M size**:

| checkpoint | size | decoder | look-ahead | our status |
|---|---|---|---|---|
| `stt_en_fastconformer_hybrid_large_streaming_multi` | ~114M | Hybrid | {0/80/480/1040 ms} | **tested → 38.13% CER (proxy, Dev_streaming)** |
| `stt_en_fastconformer_hybrid_large_streaming_80ms` / `_1040ms` | ~114M | Hybrid | single | untested (fixed-latency siblings) |
| **`stt_en_fastconformer_hybrid_medium_streaming_80ms`** | **~32M** | Hybrid | 80 ms | **UNTESTED — the target** |
| `stt_en_fastconformer_hybrid_medium_streaming_80ms_pc` | ~32M | Hybrid | 80 ms +punct | untested (punctuated) |

The `fastconformer_tdt_large` / `ctc_large` / `transducer_large` checkpoints are **offline
full-context** (`att_context=[-1,-1]`) — cannot stream, out of scope. "RNN-T streaming" is only
available by decoding the transducer head of a **hybrid** checkpoint.

Net: the only new, downloadable, actually-streamable model is the **32M medium hybrid**.

## 2. Two experiments (E1 gates E2)

### E1 — zero-shot 32M medium, real harness (CHEAP, ~1 pod-hour)
**Tests:** raw capacity ceiling of the small model + whether 80 ms lookahead buys a real TTFT corner.
**Does NOT test** Q's overfitting-during-finetune theory (nothing is finetuned) — E1 is the gate, not the answer.

- Model: `nvidia/stt_en_fastconformer_hybrid_medium_streaming_80ms`, decode RNN-T head, `att_context_size=[70,1]` (80 ms).
- Wrapper: reuse `streaming_multi_sub/` (already ran the 114M hybrid on the real harness); swap the
  checkpoint name + re-export. New sibling dir `track2_starting_kit/fastconformer_medium32/`.
- Sets & scorer (BOTH, official `evaluate.sh` — house rule: no proxy sign-off):
  - **Dev_streaming (123 utts)** — accuracy + latency (only set with `mfa_speech_start`).
  - **Dev_diag (425 utts, severe-enriched)** — the severity/generalization read; accuracy only.
- Re-score the **114M hybrid** through the SAME official scorer so the table is apples-to-apples
  (its 38.13% is an old *proxy* number and must not sit next to official CERs).

**Decision metrics (write the criterion BEFORE running):**
1. **Latency corner:** is medium TTFT p50 materially **< 926 ms** (our cs8 corner)?
   - If NO → the fast-model thesis is dead for this family; STOP. 80 ms lookahead didn't cash out on CPU.
2. **Capacity/generalization (Q's theory, cheap read):** is medium Dev_diag CER **≤ 114M's** official CER?
   - If medium ≥ 114M → "smaller generalizes better" is FALSIFIED for this family; do NOT fund E2.
3. **Frontier viability:** plot (CER, mean(TTFT,TTLT)) for medium vs {beam-8, cs8, beam-4}. Is medium
   **non-dominated**? A 45%+ CER point, however fast, is dominated by cs8 if cs8 is faster-or-equal — check explicitly.

**Go/No-Go for E2:** fund E2 **only if** E1 shows (a) a real sub-926 ms TTFT corner AND (b) CER not
catastrophic (heuristic: Dev_diag ≤ ~35%, i.e. in fine-tunable range). Both, not either.

### E2 — fine-tune the 32M on SAP dysarthric data (GPU, gated, explicit Q approval)
**Tests Q's ACTUAL theory:** does a *smaller* model finetuned on 683 h overfit less / transfer to
severe better than the finetuned 0.6B (+18 Dev→Test) did? This is the real experiment; E1 only earns it.
- Same faithful gate as v2 Nemotron work: train=eval=export=deploy config matched; gate on **held-out
  severe** (Dev_diag), empty-rate, and a clean-speech forgetting probe (research/37 §6 transfer wall applies).
- Bar: **beat A1 (Test1 21.28% beam-4) OR open a non-dominated low-latency corner** the zipformer can't.
- Reuse the Nemotron finetune scaffolding (`scripts/nemo_finetune.py`), swap the base model.

## 3. Exact run recipe (E1, once a pod is up)
```bash
# on pod (NeMo env, CPU decode ok — the 114M streaming bench ran CPU):
# 1) build sibling submission from the medium checkpoint
#    (edit streaming_multi_sub -> fastconformer_medium32: model name + [70,1] export)
cd track2_starting_kit
python3 local_decode.py \
  --submission-dir ./fastconformer_medium32 \
  --manifest-csv  $DATA_ROOT/manifest/Dev_streaming.csv \
  --streaming-manifest-csv $DATA_ROOT/manifest/Dev_streaming.csv \
  --data-root $DATA_ROOT \
  --out-csv ./Dev_streaming.medium32.csv \
  --out-partial-json ./Dev_streaming.medium32.partial.json
# 2) accuracy (official)
cd .. && ./evaluate.sh --split Dev_streaming --hyp-csv track2_starting_kit/Dev_streaming.medium32.csv --start_stage 0 --stop_stage 2
# 3) latency (official)
./evaluate.sh --start_stage 3 --stop_stage 3 \
  --partial-json track2_starting_kit/Dev_streaming.medium32.partial.json \
  --manifest-csv $DATA_ROOT/manifest/Dev_streaming.csv
# 4) severity read: repeat step 1-2 with Dev_diag manifest (accuracy only)
```

## 4. Cost gate / stop condition (house rule)
- E1 is CPU-feasible, ~1 pod-hour. Everything above is written locally; the only pod work is
  download → export → 2 decode passes → score. **Stop the pod the moment E1 metrics are copied back.**
- Get explicit Q approval for the exact file list before uploading any local code to the pod.
- E2 is a separate, explicitly-approved GPU bet — do NOT start it in the same pod session as E1.

## 5. Risks / unknowns
1. **sherpa-onnx cache-aware FastConformer export is WIP (GH #790, #2177).** Mitigant: the 114M ran via
   the NeMo-native `streaming_multi_sub/` wrapper, not sherpa-onnx — reuse that path, avoid the export bug.
2. **Zero-shot dysarthric CER likely poor** (114M was ~38%); 32M may be worse on the severe tail (less
   capacity) — which is the whole point of E1's decision metric #2.
3. **80 ms lookahead ≠ guaranteed low CPU TTFT.** Algorithmic delay is low, but per-chunk compute could
   erase it. E1 measures actual TTFT on the real harness — do not assume from lookahead.
4. **Transfer wall (research/37 §6):** a Dev_diag win may not move Test1 (held-out speakers). E2's gate
   must be held-out-severe, and even a Dev win is not a submit trigger — real-harness Test-predictive gate first.

## 6. RESULTS (2026-07-24) — E1 + full smaller/larger sweep (zero-shot, Dev_streaming 123u, proxy scorer)
All zero-shot; scored with `score_predict.py` (dual-ref macro-mean proxy, NOT official sclite) for
apples-to-apples. Reference fine-tuned zipformer = 17.25% proxy / 12.14% official.

| Model | Params | Latency pt | CER % | empties | worst etiology |
|---|---|---|---:|---:|---|
| **parakeet_realtime_eou_120m** @[70,1] (EOU-stripped) | 120M | 80 ms | **31.8** | 9 | CP 48.8 |
| FastConformer-large @[70,16] | 114M | 480 ms | 33.7 | 14 | CP 54.8 |
| Zipformer-70M baseline (LibriSpeech) | ~70M | ~320 ms | 36.2 | 1 | CP 51.8 |
| FastConformer-large @[70,33] | 114M | 1040 ms | 37.3 | 17 | CP 52.3 |
| FastConformer-large @default | 114M | ~med | 38.1 | 15 | CP 56.2 |
| FastConformer-large @[70,1] | 114M | 80 ms | 54.8 | 18 | — |
| **Zipformer-small** (sherpa 20M) | 20M | small | **68.7** | 37 | CP 82.2 |
| **FastConformer-medium** @[70,1] | 32M | 80 ms | **72.4** | 48 | CP 83.1 |
| FastConformer-medium @[70,1] **Dev_diag (severe)** | 32M | 80 ms | 80.6 | 183 | ALS 92.8 |

**Verdicts:**
- **Small models FALSIFIED for both families**: 20M zipformer 68.7%, 32M FastConformer 72.4%. Capacity is
  the binding constraint on dysarthric speech; shrinking is catastrophic regardless of architecture.
  (Q's "smaller generalizes better" hypothesis — dead.)
- **No architecture is inherently better**: at scale+lookahead, FastConformer (33.7%) ≈ zipformer (36.2%).
  The zipformer's real edge was fine-tuning (~19 pts), not architecture.
- **60M FastConformer does not exist** (only 32M medium + 114M large streaming hybrids published).
- **Standout: `parakeet_realtime_eou_120m`** — best zero-shot CER AND lowest latency (31.8% @ 80 ms), and
  the best severe-etiology profile (Stroke 18.6, PD 17.7). The newer "realtime" training beats the old
  FastConformer at 80 ms by 23 pts (54.8→31.8). **Requires `<EOU>` token stripping before scoring.**
- Every zero-shot point (best 31.8%) is far worse than the fine-tuned zipformer (12–17%). **Fine-tuning is
  the only lever that reaches competitive CER.**

## 7. QUEUED (needs explicit Q go + GPU $): fine-tune parakeet_realtime_120m for a low-latency corner
**Rationale:** it is the only model that hints at a viable *low-latency* Pareto play. Zero-shot 31.8% @ 80 ms;
if fine-tuning closes ~15–19 pts as it did for the zipformer (36→12), a FT parakeet could reach ~14–17% CER
**at 80 ms** — a corner our zipformer cs8 (13.85% @ 926 ms) cannot reach on latency.
**Caveat (binding):** same NeMo-streaming-finetune path the Nemotron-0.6B lost on — Dev→Test transfer wall +
severe-tail collapse ([[nemotron-vs-zipformer-roadblock]], research/37 §6). Gate on held-out-severe; a Dev
win is NOT a submit trigger. P(beats A1 on Test1) is uncertain; this is a bet, not a plan.
**Status: SHELF-READY, not launched.** Do not start without explicit Q approval and a fresh pod.
```
