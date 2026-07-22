# Benchmark & Pod Plan — Parakeet-RNNT / TDT vs the real Track 2 baseline

_Corrected-premise plan for the "are Parakeet-RNNT/TDT better for SAPC2 Track 2?" investigation._
_Date: 2026-07-13. Companion to `docs/repo_analysis.md`. **Nothing has been run; no pod started.**_
_Requires Q approval on (a) data location + upload list, (b) pod/budget, before any spend._

---

## 0. Premise corrections baked into this plan
- **Baseline is NOT Parakeet-CTC.** The real Track 2 baseline = icefall **streaming Zipformer-Transducer**
  (`track2_starting_kit/streaming_zipformer`). The comparison table's "Existing CTC" row is replaced by
  **Zipformer-Transducer** (and, optionally, the previously-submitted **Nemotron int8** as "current best NVIDIA").
- **Parakeet-TDT/RNNT are offline, full-context** (verified — see repo_analysis §Premise-corrections 6).
  They do **not** support real cache-aware streaming. Therefore this plan splits the question in two:
  - **Q-accuracy:** what CER *ceiling* do these models reach on dysarthric Dev (offline `transcribe`)?
    Cheap, honest, no fake-streaming. Answers "is the acoustic model better than zipformer at all?"
  - **Q-streaming:** can any NVIDIA model be run as *real* streaming and beat zipformer on CER+TTFT+TTLT?
    The only legitimate candidate is the **cache-aware RNN-T** sibling
    (`nemotron-speech-streaming-en-0.6b` / `fastconformer_hybrid_large_streaming_multi`), i.e. the
    path the prior Nemotron attempt already lost on. TDT/RNNT offline checkpoints are **excluded** from
    the streaming submission path (buffered re-run = forbidden + poor CPU latency).

**Decision rule (from task success criteria):** rank on **CER + TTFT + TTLT**, not CER alone; Pareto
rank uses mean(TTFT, TTLT). A model only advances to fine-tuning if it is Pareto-competitive with
zipformer on the *streaming* path, OR its offline CER ceiling beats zipformer by a margin large enough
to justify solving the streaming-export gap.

---

## 1. Verified checkpoint names (HF, 2026-07-13)
| Role | Checkpoint | Streaming? | Use in this plan |
|---|---|---|---|
| Offline TDT | `nvidia/parakeet-tdt-0.6b-v2` | ❌ offline | accuracy ceiling only |
| Offline TDT (newer) | `nvidia/parakeet-tdt-0.6b-v3` | ❌ offline | optional ceiling |
| Offline RNNT | `nvidia/parakeet-rnnt-0.6b` | ❌ offline | accuracy ceiling only |
| Offline RNNT/TDT 1.1B | `nvidia/parakeet-rnnt-1.1b`, `nvidia/parakeet-tdt-1.1b` | ❌ offline | ceiling (heavy; likely CPU-infeasible for streaming) |
| **Cache-aware streaming** | `nvidia/nemotron-speech-streaming-en-0.6b` | ✅ real streaming | **the streaming candidate** |
| Cache-aware streaming (older) | `stt_en_fastconformer_hybrid_large_streaming_multi` | ✅ real streaming | fallback streaming candidate |

> Confirm each `from_pretrained` name resolves on the pod before use (task Phase 4 requirement).
> 1.1B streaming on CPU real-time is presumed infeasible until measured — de-prioritize.

---

## 2. Benchmark framework (Phase 3) — build locally, dry, before any pod
Directory (new siblings; never touch `streaming_zipformer/`):
```
benchmark/
  models/
    parakeet_offline.py     # wraps NeMo transcribe() -> CER ceiling ONLY (not a Track2 Model)
    nemotron_streaming.py   # 5-method streaming Model wrapping NeMo cache-aware RNNT
  run_eval.py               # thin driver that SHELLS OUT to the real local_decode.py + evaluate.sh
  metrics.py                # imports utils/compute_metrics.py (no reimplementation)
  latency.py                # imports utils/compute_latency.py (no reimplementation)
```
Hard rules:
- `metrics.py` / `latency.py` **import and call** `utils/compute_metrics.py` & `utils/compute_latency.py`
  as-is. **Do not reinvent CER/WER or TTFT/TTLT** (repo house rule + memory `validate-against-real-harness`).
- The streaming `Model` must implement the exact 5-method contract and fire the partial callback per
  emitted token (TTFT depends on the callback, not the return value — see repo_analysis §3).
- The offline wrapper is for the accuracy-ceiling table only; it is explicitly **not** submitted and
  **not** run through the streaming pass (would be fake streaming).
- Local dry checks (no data, no GPU): `python -m py_compile`, import-lint, and a synthetic-noise
  `accept_chunk` smoke that just verifies state advances (no correctness claim).

## 3. Streaming implementation (Phase 5) — real cache-aware, not buffered
For `nemotron_streaming.py`, reuse NeMo's cache-aware streaming primitives (encoder cache + decoder
state carried across chunks), mapping the harness's 100 ms cadence onto the model's fixed chunk size
(pick the closest supported operating point: 80/160/320 ms). This is the same integration the prior
Nemotron submission built; **start from that model.py** (memory: "harness is adapt-not-rebuild") rather
than rewriting, and carry forward the fixes already discovered (SOS init, warmup-frame drop,
drop-extra-pre-encoded). Explicitly measure that state size stays bounded per chunk (i.e. actually
incremental, not O(n) re-encode).

---

## 4. Environment / pod options
Two hosts satisfy the tracks; pick per phase:
- **CPU (faithful to Track 2 scoring):** the official Docker `xiuwenz2/sapc2-runtime:latest`
  (torch 2.5.0+cu124, CPU mode) on a linux x86_64 CPU box **or** RunPod CPU pod. This is where
  streaming CER + TTFT/TTLT must be measured (latency is meaningless off the target CPU).
- **GPU (fast accuracy ceiling only):** a cheap GPU pod to run offline `transcribe()` ceilings for
  the Parakeet TDT/RNNT models quickly. GPU latency numbers are **not** reportable for Track 2.

Prereqs Q must supply / approve:
1. **Data location** for SAP `Dev` + `Dev_streaming` (manifests + WAVs). Not on this box.
2. **Upload approval** for the exact file list if we push local code/artifacts to the pod
   (sandbox policy; per CLAUDE.md cost gate).
3. **Pod type + budget cap** and who starts it (`runpodctl pod start ...` is a paid one-way action).

---

## 5. Staged run plan (smoke → guardrail → full; stop at each decision metric)

**Stage A — offline accuracy ceiling (GPU pod, cheap, ~1 GPU-hr).** Answers Q-accuracy.
```
# on pod, inside NeMo env
python benchmark/models/parakeet_offline.py \
    --checkpoint nvidia/parakeet-tdt-0.6b-v2 \
    --manifest $DATA_ROOT/manifest/Dev.csv --data-root $DATA_ROOT \
    --out-csv Dev.parakeet_tdt.csv
./evaluate.sh --split Dev --hyp-csv Dev.parakeet_tdt.csv --start_stage 0 --stop_stage 2
# repeat for parakeet-rnnt-0.6b (and optionally 1.1b, tdt-v3)
```
- **SMOKE first:** 20 utterances, confirm the checkpoint loads + CER computes end-to-end.
- **STOP CONDITION A:** if best Parakeet offline CER on Dev is **not** meaningfully below the
  zipformer streaming CER, the acoustic model is not better → do **not** pursue the streaming port;
  report and stop. (Offline CER is an upper bound; real streaming will be worse.)

**Stage B — real streaming on the CPU runtime (only if Stage A passes).** Answers Q-streaming.
```
# inside xiuwenz2/sapc2-runtime:latest (CPU)
cd track2_starting_kit
python3 local_decode.py \
  --submission-dir ../benchmark/streaming_nemotron_pkg \
  --manifest-csv $DATA_ROOT/manifest/Dev.csv \
  --streaming-manifest-csv $DATA_ROOT/manifest/Dev_streaming.csv \
  --data-root $DATA_ROOT \
  --out-csv Dev.nemotron.csv --out-partial-json Dev.nemotron.partial.json
../evaluate.sh --split Dev --hyp-csv Dev.nemotron.csv --start_stage 0 --stop_stage 2
../evaluate.sh --start_stage 3 --stop_stage 3 \
  --partial-json Dev.nemotron.partial.json \
  --manifest-csv $DATA_ROOT/manifest/Dev_streaming.csv
```
- **SMOKE first:** 20 utterances through the REAL harness (both passes). Verify partials are non-empty
  and TTFT/TTLT compute.
- **GUARDRAIL:** speaker-disjoint ~200-utt subset before full Dev.
- **STOP CONDITION B:** compare (CER, TTFT_p50, TTLT_p50) vs zipformer on the **same** Dev split via
  the **same** harness. Advance to fine-tuning only if Pareto-competitive or clearly better.

**Baseline row:** run the zipformer through the identical harness on the identical split so the table
is apples-to-apples (do not reuse historical numbers from a different split/harness).

---

## 6. Cost gate & stop discipline (from CLAUDE.md)
- All code/scaffolding (Phase 3/5) is written and import-checked **locally first**; the pod only runs
  measurements.
- Minimum experiment size first: smoke (20) → guardrail (200) → full Dev; **stop the moment the
  decision metric for that stage is known.**
- Copy artifacts back (`Dev.*.csv`, `*.partial.json`, metrics JSON, CPU/RAM/RTF logs), then **stop the
  pod immediately.**
- **No Codabench upload** from this investigation — it is a benchmark, not a submission. Any eventual
  submission goes through the faithful-validation gate separately.

---

## 7. Deliverables the pod run will fill in (Phase 9 table)
| Model | Params | Streaming? | CER | WER | TTFT P50 | TTLT P50 | RTF | RAM | CPU |
|------|--------|-----------|-----|-----|----------|----------|-----|-----|-----|
| Zipformer-Transducer (baseline) | ~70M | ✅ real | | | | | | | |
| Nemotron int8 (prior submit, optional) | 0.6B | ✅ real | | | | | | | |
| Parakeet-RNNT-0.6b | 0.6B | ❌ offline (ceiling) | | | n/a | n/a | | | |
| Parakeet-TDT-0.6b-v2 | 0.6B | ❌ offline (ceiling) | | | n/a | n/a | | | |
| Nemotron-streaming-0.6b (cache-aware RNNT) | 0.6B | ✅ real | | | | | | | |

Then answer: best accuracy? best latency? who's on the CER×latency Pareto frontier? worth fine-tuning?

---

## 8. My recommendation before we spend
Run **Stage A only first** (cheap GPU accuracy ceiling for parakeet-rnnt-0.6b + tdt-0.6b-v2 vs the
zipformer's known Dev CER). It answers the cheapest, highest-information question — *is the NVIDIA
acoustic model even more accurate on dysarthric speech?* — for ~1 GPU-hr and no streaming-integration
work. If the ceiling doesn't beat zipformer, we've closed the whole line of investigation for a few
dollars. Only if Stage A is promising do we pay for the harder Stage B streaming port, which we already
know is where the prior NVIDIA attempt died.

**Blockers before I can proceed:** (1) SAP Dev / Dev_streaming data location, (2) pod + budget approval,
(3) upload approval for the code file list. Give me those and I'll start with the Stage A smoke.
