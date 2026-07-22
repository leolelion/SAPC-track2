# Repo Analysis — SAPC2 Track 2 (Streaming ASR)

_Phase 1 audit for the "Parakeet-RNNT vs Parakeet-TDT vs current baseline" investigation._
_Author: coding agent. Date: 2026-07-13. Status: read-only audit; nothing modified._

> **Read this first — the task premise does not match the repo.** The task says the
> current Track 2 baseline is **Parakeet-CTC**. There is **no Parakeet-CTC anywhere in
> this repo**. The actual Track 2 baseline is an **icefall streaming Zipformer-Transducer**.
> The only Parakeet present is a **Track 1** (offline, non-streaming) baseline that is
> already **Parakeet-TDT-0.6B-v2**. See "Premise corrections" at the bottom before acting
> on Phases 2–10.

---

## 1. Current architecture

### Two tracks, two interfaces
| | Track 1 (offline ASR) | Track 2 (streaming ASR) — **our focus** |
|---|---|---|
| Dir | `track1_starting_kit/{parakeet,whisper,canary_qwen}/` | `track2_starting_kit/streaming_zipformer/` |
| Interface | `Model.predict(wav_path) -> str` (whole file) | 5-method streaming `Model` (below) |
| Compute | GPU (parakeet `raise RuntimeError` if no CUDA) | **CPU-only**, 15000 s/submission |
| Scored on | CER/WER | CER/WER **+ TTFT/TTLT latency** |

### Track 2 submission structure
```
track2_starting_kit/
├── local_decode.py            # organizers' ingestion harness (2 passes)
├── streaming_zipformer/       # THE Track 2 baseline submission
│   ├── model.py               # Model class (5-method streaming API)
│   ├── config.yaml            # tunable hyperparams (chunk_size, beam, etc.)
│   └── setup.sh               # clones icefall, downloads weights (linux x86_64/cp311)
└── streaming_zipformer.zip    # packaged submission
```

### Scoring pipeline (shared by both tracks)
```
evaluate.sh
 ├─ stage 0: install SCTK (sclite)
 ├─ stage 1: steps/eval/prepare_ref_trn.sh  -> build ref TRN/SGML from manifest
 ├─ stage 2: steps/eval/evaluate.sh -> sclite align -> utils/compute_metrics.py (CER/WER)
 └─ stage 3: utils/compute_latency.py (TTFT/TTLT from partial JSON)
```

---

## 2. Current baseline model (Track 2)

**icefall Zipformer-Transducer, streaming/causal**, decoded with `modified_beam_search`.

- File: `track2_starting_kit/streaming_zipformer/model.py`
- Encoder: Zipformer, `causal=True`, `chunk_size=16` frames (~320 ms), `left_context_frames=128`.
- Decoder: RNN-T / Transducer predictor, `context_size=2`.
- Tokenizer: SentencePiece BPE-500 (`data/lang_bpe_500/bpe.model`), `<blk>` blank + `<unk>`.
- Weights: `weights/exp/epoch-30.pt` (downloaded by `setup.sh`).
- Feature front-end: kaldifeat `Fbank`, 80 mel bins, 10 ms shift / 25 ms window, dither off.
- Config-driven: hyperparams in `config.yaml`; `features.mode` = `incremental` (O(n), sherpa-onnx
  style) or `full` (O(n²) recompute).

**This is genuine cache-aware streaming** — icefall's `decode_one_chunk` / `get_init_states`
carry encoder+decoder state across chunks; it does NOT re-run the full model per chunk.

### Prior Track 2 experiments (from docs/memory — context for this investigation)
- The team already tried an **NVIDIA streaming FastConformer/Nemotron** model:
  `nvidia/nemotron-speech-streaming-en-0.6b` (a.k.a. nemotron-3.5-asr-streaming-0.6b) and
  `nvidia/stt_en_fastconformer_hybrid_large_streaming_multi`.
- Result (memory `nemotron-vs-zipformer-roadblock`): **it lost to the zipformer** — root cause
  was a **streaming-export gap** (the exact "fake streaming vs real cache-aware" trap Phase 5
  warns about), not model capacity. All rescue arms failed; recorded default = **ship zipformer
  beam-4**. There was also a `_t1` int8 submission (memory `int8-submission-status`): Test1 CER
  27.97% vs an earlier 23.44%, severity-driven.
- **Implication for THIS task:** the "NVIDIA cache-aware streaming Parakeet/Conformer beats
  zipformer" hypothesis has already been tested once and failed on the export/streaming boundary.
  A re-attempt must budget explicitly for that gap or it repeats the loss.

---

## 3. Model interface (the contract new models MUST implement)

Track 2 streaming `Model` (called only on the Decoder thread → no locking needed):

```python
class Model:
    def __init__(self):                 # load weights/tokenizer ONCE
    def set_partial_callback(self, fn): # register fn(text:str); called once per partial
    def reset(self):                    # per-file state reset
    def accept_chunk(self, chunk):      # np.float32, 16kHz mono, 1600 samples (100 ms) -> partial str
    def input_finished(self):           # -> final str
```

### How 100 ms chunks are delivered (`local_decode.py`)
- `SAMPLE_RATE = 16000`, `CHUNK_SIZE = 1600` samples = **100 ms** per `accept_chunk`.
- Audio is read from 16 kHz / mono / 16-bit PCM WAV, normalized `int16 -> float32 / 32768.0`.
- **Two ingestion passes over the manifest:**
  - **Pass 1 — batch/accuracy** (`run_batch_decode`): no pacing, `set_partial_callback` is a
    no-op; collects `input_finished()` per file → writes `--out-csv` (`id,raw_hypos`) → CER/WER.
  - **Pass 2 — streaming/latency** (`run_streaming_decode`): a sender thread paces chunks at
    `--streaming-interval` (default **0.1 s = real-time**) into a queue; a decoder thread pulls
    them, times each partial callback, and records `audio_send_start_time`,
    `audio_end_oracle_time`, `first_partial_time`, `final_visible_time` → `--out-partial-json`.
- Note: `accept_chunk` return value is used by the batch pass; the **callback** (`set_partial_callback`)
  is what drives latency timing in the streaming pass. A model must emit partials via the callback
  to get a good TTFT — the return value alone won't lower latency.

### Sample rate / normalization / transcript formatting
- Sample rate: **16 kHz** enforced (`read_wave` raises on mismatch). Mono, 16-bit enforced.
- Normalization: `/32768.0` to float32 in [-1, 1). No further gain/VAD in the harness.
- Transcript: raw hypothesis string per id; ref normalization/`unk` handling happens in scoring,
  not in the model. Do **not** emit literal `unk` expecting it free (see `parse_sgml_csdi`).

---

## 4. CER/WER calculation (`utils/compute_metrics.py`)

- sclite aligns hyp vs 2 refs → SGML → `compute_from_sgml`.
- **min-over-two-refs**: ref1 = `norm_text_with_disfluency`, ref2 = `norm_text_without_disfluency`.
- `CharErrorRateMinTwoRefs` / `WordErrorRateMinTwoRefs`, **`clip_at_one=True`** (per-utt error
  capped at 100%). **CER is the primary metric.**
- `unk` reconciliation: S/D/I ops involving the `unk` token are handled in `parse_sgml_csdi`
  (`process_unk=True`).

## 5. TTFT/TTLT calculation (`utils/compute_latency.py`)

- `TTFT = first_non_empty_partial_time − (audio_send_start_time + mfa_speech_start)`
  - `mfa_speech_start` comes from the `*_streaming.csv` manifest column (forced-aligned true
    speech onset). Empty partials don't count — first **non-empty** partial sets TTFT.
- `TTLT = final_visible_time − audio_end_oracle_time` (counted only if ≥ 0).
- Reported as **P50** (median), with P90/P95 in detail. **Pareto rank uses mean(TTFT, TTLT).**
- Lever: emit a non-empty partial earlier → lower TTFT; finalize fast after audio end → lower TTLT.

## 6. Manifest schema (`utils/manifest.py`)
Columns: `id, speaker, etiology, audio_filepath, duration, text`.
- `id` = wav stem; `speaker` = prefix before first `_` → build **speaker-disjoint** dev splits.
- `etiology` (e.g. "Cerebral Palsy") → per-disease analysis.
- Streaming manifest (`*_streaming.csv`) additionally carries `mfa_speech_start` for TTFT.

## 7. Dependency requirements
- Submission runtime: Docker `xiuwenz2/sapc2-runtime:latest`, **PyTorch 2.5.0+cu124**, Track 2
  **CPU-only**, 15000 s/submission budget.
- `setup.sh` runs before model load; `requirements.txt` auto-installed after.
- Baseline `setup.sh` pins **linux x86_64 / cp311** wheels for `k2` + `kaldifeat`
  (`manylinux2014_x86_64`) → **will not install on macOS/arm natively.**

---

## 8. How to add a new model
1. Create a **new sibling dir** under `track2_starting_kit/` (e.g. `streaming_parakeet_rnnt/`),
   never edit the pristine `streaming_zipformer/`.
2. Provide `model.py` (5-method streaming `Model`), `config.yaml`, `setup.sh`.
3. Package offline per memory `submission-offline-packaging` (self-contained zip; weights baked in).
4. Validate with the **real** `local_decode.py` (both passes) → `evaluate.sh` on Dev **before**
   any Codabench upload (memory `validate-against-real-harness`). Hand-rolled decode = exploration
   only, never sign-off.
5. Log a row in `experiments/summary.csv` + `experiments/exp_XXX/` (config + git hash + metrics).

---

## 9. Potential integration issues (for RNNT/TDT via NeMo)

1. **`predict(wav_path)` ≠ streaming interface.** The Track 1 Parakeet uses whole-file
   `transcribe()`. Track 2 needs incremental `accept_chunk(100ms)`. NeMo's default
   `transcribe()` is offline — wrapping it as `accept_chunk` = **fake streaming** (buffer +
   re-run), which the task explicitly forbids and which is exactly how the prior Nemotron
   attempt lost. Real path = NeMo **cache-aware streaming** (`FrameBatchASR` /
   `CacheAwareStreamingAudioBuffer` + encoder cache), which only some checkpoints support.
2. **CPU-only + latency budget.** Parakeet-RNNT/TDT 0.6B–1.1B are large for a CPU real-time
   streaming target. RTF and TTFT/TTLT on a 24-core EPYC worker (memory `eval-worker-cpu-confirmed`)
   are the real question — CER alone is misleading. TDT's token-and-duration decoding skips frames
   (good for RTF) but its emission cadence vs TTFT needs measurement, not assumption.
3. **GPU-only baseline code.** `track1/parakeet/model.py` hard-`raise`s without CUDA and reads a
   venv at `/opt/parakeet_venv` → cannot run as-is on this CPU/darwin box.
4. **Env mismatch.** Runtime is torch **2.5.0+cu124**; this dev box is torch **2.6.0**, Python
   **3.13**, **no NeMo**, darwin/arm. NeMo pins matter (many NeMo builds want torch 2.x + specific
   numpy < 2). numpy here is 2.2.4.
5. **`accept_chunk` return vs callback.** For good TTFT a NeMo wrapper must fire the partial
   callback the moment a token is emitted, not only at `input_finished`.
6. **Streaming chunk-size mismatch.** NeMo cache-aware models are exported at fixed att-context /
   chunk sizes (e.g. [70,1], [70,16]); mapping the harness's 100 ms cadence onto NeMo's frame
   chunking is the fiddly part the prior attempt got wrong (SOS init, warmup frames, drop-extra-pre-encoded).

---

## Premise corrections (must resolve before Phases 2–10)

1. **No "Parakeet-CTC baseline" exists.** Track 2 baseline = icefall Zipformer-Transducer (RNN-T).
   The comparison table's "Existing CTC" row has no referent in this repo. The honest baseline to
   compare against is **streaming_zipformer** (or the earlier submitted Nemotron int8, if we mean
   "current best NVIDIA attempt").
2. **The only Parakeet here is TDT already** — Track 1's offline `parakeet-tdt-0.6b-v2`, GPU-only,
   non-streaming. It is not a Track 2 baseline and cannot be one without a streaming rewrite.
3. **No data on this box.** No manifests, no WAVs; `DATA_ROOT` in `evaluate.sh` is `/path/to/data`.
   Phase 2 ("find the Dev_streaming subset") has **nothing to find locally** — the dev set is not
   present. Phases 6–9 (measure CER/WER/TTFT/TTLT) are **not runnable here**.
4. **No NeMo / wrong platform.** NeMo not installed; torch 2.6 vs runtime 2.5; darwin/arm; k2/kaldifeat
   wheels are linux-x86_64-only. Phases 4–8 require a linux CPU (Docker runtime) or GPU pod + the SAP
   data. Per `CLAUDE.md` cost gate, a paid pod must not be started to write/scaffold code.
5. **Prior art:** NVIDIA cache-aware streaming already lost to the zipformer once on the
   export/streaming boundary. This does not kill the idea, but the investigation must treat the
   streaming-export gap as the primary risk, not accuracy.

6. **VERIFIED (web, 2026-07-13): Parakeet-TDT and Parakeet-RNNT are OFFLINE, full-context
   models — not cache-aware streaming.** The Parakeet family (CTC/RNNT/TDT/hybrid, 0.6B/1.1B)
   is trained full-context; the HF cards for `parakeet-tdt-0.6b-v2`/`v3` have explicit "Streaming?"
   threads confirming offline-only. NVIDIA states there was a **bug with TDT for chunked streaming
   inference**, so they built a **separate** cache-aware architecture — `nemotron-speech-streaming-en-0.6b`
   / `nemotron-3.5-asr-streaming-0.6b` / `stt_en_fastconformer_hybrid_large_streaming_multi`
   (FastConformer-**CacheAware-RNNT**) — for real time. Consequence for Track 2:
   - Pretrained **Parakeet-TDT/RNNT** on the real 100 ms `accept_chunk` path ⇒ only via **buffered
     re-run (fake streaming)** = forbidden by the task, bad TTFT/RTF on CPU. They are usable only as
     an **offline accuracy ceiling** (transcribe on Dev), not as a deployable streaming submission.
   - The genuine "streaming NVIDIA" candidate is the **cache-aware RNN-T sibling** (nemotron-streaming
     / fastconformer-streaming-multi), which is the exact path the prior Nemotron attempt used and lost.
   - Sources: huggingface.co/nvidia/parakeet-tdt-0.6b-v2 (discussion 3), .../parakeet-tdt-0.6b-v3
     (discussion 11), huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b,
     huggingface.co/blog/nvidia/nemotron-speech-asr-scaling-voice-agents, NeMo ASR models docs.
