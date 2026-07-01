# HANDOFF → Codex: Make Nemotron fast enough to win Track-2

**Date:** 2026-06-22 · **From:** Claude (Opus) · **To:** Codex
**Mission:** Nemotron-0.6B has the best *accuracy* ceiling we've found for SAPC2 Track-2,
but it is **too slow** in our current inference stack and gets truncated on the eval
worker. **Your job: squeeze the speed out of it** (native runtime / loop rewrite / feature
fix), prove it runs real-time on the Codabench CPU worker, *then* finetune. Speed is the gate.

> Read `CLAUDE.md` (repo + `/Users/o/Downloads/CLAUDE.md`) for the operating protocol:
> prediction-before-action, stop-on-failure, verify-don't-assume, don't edit the scorer,
> `git add .` forbidden, only use the GPU pod when strictly necessary (work local + git).

---

## 1. TL;DR thesis (why this is worth your time)

- **Accuracy:** Nemotron zero-shot = **24.96% CER** on our 2k Dev ruler (official scorer).
  Finetuned, it projects to **~13–16% CER** (same adaptation gain that took our zipformer
  36%→21.6%). That would be a commanding lead.
- **The wall is SPEED, not accuracy, not size, not a packaging bug.** Proven below.
- If you make the stack real-time on the worker, Nemotron becomes the strongest Track-2 model.

---

## 2. The evidence (how we know it's speed)

| Run | CER | WER | Notes |
|---|---|---|---|
| Nemotron zero-shot, Dev 2k, **our NeMo decode** + sclite | 25.07% | — | fast pod |
| Nemotron, Dev 2k, **EXACT submission (local-mel)** + official scorer | **24.96%** | 31.27% | fast pod |
| Nemotron, **Test1** (same submission, same scorer) | **51.52%** | 58.99% | eval worker; TTFT p50 2270ms / p90 6205ms, TTLT 290ms; 10521/10521 matched |
| A1 zipformer (our live #1), Test1 | 23.44% | — | for reference |

**The decisive logic:** the *exact submission* scores **24.96% on Dev** (our fast pod) but
**51.52% on Test1** (the eval worker). Same code, same weights, same scorer. The official
`EnglishTextNormalizer` strips punctuation/casing, so formatting is ruled out. Dev≈25% ⇒ the
submission path is **correct**. The only variable left is the **machine**: on the slower,
resource-constrained worker the decode can't keep up → utterances truncate/time out → CER
doubles. The decode is RTF≫1 (it could not finish 2000 utts in ~33 min even on a 192-core
pod running 24-way parallel). On a single real-time-paced worker stream, that's fatal.

Full writeup: `research/10_nemotron_verdict.md`. Strategic memory:
`/Users/o/.claude/projects/-Users-o-Downloads-SAPC-template/memory/nemotron-higher-ceiling.md`.

---

## 3. Hard constraints (Track-2 rules — don't violate)

- **CPU-only.** No GPU at eval. Streaming: audio arrives in **100 ms (1600-sample)** float32
  16 kHz mono chunks; you must emit partials as you go.
- **Budget:** 15000 s per submission, ~50 subs/person. **Size is NOT a limit** — the 821 MB
  zip uploaded and ran fine. (We previously *guessed* a 150 MB cap; that was wrong — verified.)
- **Offline worker: NO network.** Everything must be bundled (weights + wheels), installed
  with `pip --no-index`. No HuggingFace, no NeMo download, no pip-from-internet.
- **Scoring is fixed — never edit it.** Official: `EnglishTextNormalizer` (Whisper) →
  sclite alignment → **min-over-two-refs** CER/WER, per-utt error clipped at 1.0. **CER is
  primary.** Latency: TTFT (first non-empty partial − speech-start) and TTLT (final − audio-end),
  reported P50; Pareto rank uses mean(TTFT,TTLT). Latency is secondary to CER.
- **Submission interface** (`model.py` class `Model`): `__init__`, `set_partial_callback(fn)`,
  `reset()`, `accept_chunk(np.ndarray)->str`, `input_finished()->str`. All called on one thread.

---

## 4. The current stack — and TWO inefficiencies I already spotted

Model: **danielbodart int8 ONNX export of `nvidia/nemotron-speech-streaming-en-0.6b`** —
cache-aware streaming **FastConformer-RNNT**, `att_context_size=[70,6]` (560 ms model chunks),
24 encoder layers dim 1024, 2-layer decoder LSTM hidden 640, vocab 1025 (blank=1024),
chunk_mel=56, pre-encode cache=9. Mel: local torch reimpl of NeMo preprocessor (preemph 0.97,
n_fft 512, win 400, hop 160, 128 mel, log). The submission `model.py` is a **hand-rolled
Python streaming loop over ONNX Runtime CPU** (encoder session + combined decoder+joint session).

The current `model.py` is extracted at **`/tmp/nemo_hand/model.py`** (+ `localmel.py`,
`config.yaml`, `setup.sh`). Read it. Two things jump out as prime speed targets:

**(A) O(N²) feature recompute — likely a big silent cost.**
`_ensure_features()` does `np.concatenate(self._raw_chunks)` and runs the **full mel
preprocessor over the ENTIRE utterance every time a new 100 ms chunk arrives**. For a T-second
utterance that's ~10·T mel passes over growing audio → quadratic. Should be **incremental**:
keep a rolling sample buffer, compute mel only for new frames. Cheap to fix, possibly large win.

**(B) Per-symbol Python→ORT round-trips in the greedy loop.**
`_encode_and_decode()` runs:
```
for f_idx in range(n_enc):                 # each encoder frame
    for _sym in range(MAX_SYMBOLS_PER_FRAME=10):
        self._dec_sess.run(... tiny tensors ...)   # one ORT call per symbol
```
The decoder ONNX is **decoder-LSTM + joint combined**, so every symbol re-runs the LSTM, and
each call is a separate Python→ORT dispatch on tiny tensors — dispatch overhead dominates.
This is the classic reason a hand-rolled Python RNN-T loop is slow vs a native C++ loop.

---

## 5. The speed plan (ranked — but PROFILE before you commit)

**Step 0 — PROFILE first. Don't guess.** Run `py-spy`/`cProfile` on one utterance's decode on
the pod and get the RTF breakdown: **mel recompute vs encoder.run vs the decode loop**. The fix
order depends entirely on which dominates. (This repo's protocol: hypothesis → measure → conclude.)

Then, by expected leverage:

1. **Native runtime — `sherpa-onnx OnlineRecognizer`** (biggest structural win *if it supports
   this export*). C++ greedy loop + incremental features + no Python per-symbol overhead.
   **Research question:** does sherpa-onnx support NeMo **cache-aware streaming FastConformer
   transducer** with this att_context export? (sherpa-onnx has NeMo transducer support and
   prebuilt NeMo streaming models — verify our specific export's I/O names match, or re-export
   in sherpa-onnx's expected format.) Must be **offline-bundleable** (the python wheel + its
   libs), and must not drag in a numpy that breaks the ABI (see §7).
2. **Fix the O(N²) mel** (§4A) → incremental features. Works even staying in pure Python; likely
   a quick, safe win. Validate transcripts are bit-identical to the current path first.
3. **TDT / frame-skipping.** Is the model a **Token-and-Duration Transducer**? If the export has
   a duration head, exploit it to skip frames → fewer encoder/decoder steps. Check danielbodart's
   config / decoder outputs.
4. **Tune ORT threads to the actual worker.** The submission hardcodes 4 threads. `model.py`
   already logs `[CPU_DIAGNOSTIC]` + writes `/tmp/cpu_diagnostic.json` (CPU count + matmul bench)
   at load. **Ask Q for the full Test1 ingestion log** and read `[CPU_DIAGNOSTIC]` to learn the
   worker's real core count/speed, then set `intra_op_num_threads`/`OMP_NUM_THREADS` accordingly.
5. **Quantize the decoder+joint** (currently fp32) → int8 dynamic. Smaller per-call tensors.
6. **Batch the joint across frames** where blank-free; restructure greedy to cut ORT calls.
7. **Chunk / look-ahead tradeoff.** Nemotron supports look-aheads {0,80,480,1040 ms}; smaller
   lowers TTFT but the real-time problem is *compute per chunk*, so weigh carefully.

**Sequencing rule:** prove real-time on the worker spec FIRST (gate). Only finetune once the
stack is real-time — zero-shot 24.96% already beats unadapted, and finetuning won't matter if
the model truncates on the worker.

---

## 6. Where everything lives

**GPU pod (RunPod) — STOPPED; restart only when you must run/profile.**
- `runpodctl pod start <id>` then get the new SSH port from `runpodctl get pod`. Pod id
  `3dwiczo41jeg1y`, host `38.80.152.249` (SSH port changes per restart). Key:
  `ssh -i /Users/o/.runpod/ssh/RunPod-Key-Go -p <port> -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@38.80.152.249`
- Offline submission: `/workspace/finetune/nemo_submission/` (`model.py`, `localmel.py`,
  `setup.sh`, `weights/`: `encoder_model.onnx`+`.data` ~800MB, `decoder_model.onnx`+`.data`,
  `tokens.txt`, `filterbank.npy`, `window.npy`) + `submission_*.zip`.
- Diagnostic result persisted: `/workspace/finetune/eval/e1_results/nemotron_dev_diagnostic.txt`.
- Scorer + decode harness: `/workspace/sapc-nemotron/utils/` (`evaluate.py`, `compute_metrics.py`,
  `compute_latency.py`, `normalize_hyp.py` — uses `from normalizer.text_normalizer_hf import
  EnglishTextNormalizer`, `local_decode.py`, `metrics/cer.py` needs torchmetrics at `/opt/pylibs`).
- **MooseFS is slow** for random/concurrent reads → stage model + audio on **`/dev/shm`** before
  decoding, or model-load storms across parallel shards will dominate wall time.

**Local (this Mac):**
- `/Users/o/Downloads/nemo_submission.zip` (820 MB, the validated upload) ·
  `/Users/o/Downloads/submission_a1_int8.zip` (68 MB, the live #1).
- Extracted text artifacts: `/tmp/nemo_hand/{model.py,localmel.py,config.yaml,setup.sh,tokens.txt}`.
- `sherpa-onnx` installs on macOS — you can prototype the native runtime locally without the pod.

**Git:** fork `https://github.com/leolelion/SAPC-track2.git` (remote `fork`). Research branch
`claude/asr-research-docs`. `origin` is the read-only upstream (`xiuwenz2/SAPC-template`) — don't
push there. Research notes: `research/01..10`. Decoding experiment results: `research/09`.

---

## 7. Landmines (history — do not repeat)

- **The 7 prior upload failures = numpy C-ABI break.** Installing `nemo_toolkit[asr]` in
  `setup.sh`, then `import nemo.collections.asr`, broke numpy's C-ABI
  (`ValueError: numpy.dtype size changed, Expected 96 got 88`) → deterministic ingestion crash.
  **Fix that's already in place: NO NeMo in the submission.** Local-mel replaces the NeMo
  preprocessor; bundle **only** the `onnxruntime` wheel (NOT a numpy wheel — use the runtime's
  numpy so the ABI never moves). **Whatever native runtime you add, keep this invariant**: don't
  install anything that recompiles/replaces numpy or torch in the worker venv.
- **Offline only.** `setup.sh` must `pip install --no-index --find-links wheels`. No network.
- **Don't edit** `compute_metrics.py`, `compute_latency.py`, `evaluate.*`, `local_decode.py`
  semantics — they mirror the organizers' scoring. Wrap, don't modify.
- **`head` on this Mac is a broken perl tool** — don't pipe to `head`; use `grep`/`sed -n`.
- **SSH to the pod drops (255) intermittently** — make pod-side work atomic scripts + `nohup` +
  sentinel files, don't hold long interactive sessions.

---

## 8. Open questions for you to resolve

1. **RTF breakdown?** (mel vs encoder vs decode loop) — answer by profiling. Determines the fix.
2. **Does sherpa-onnx support this exact cache-aware NeMo streaming transducer export?** If not,
   can we re-export to its format, or must we optimize the Python loop in place?
3. **Is the model TDT (duration head) or vanilla RNN-T?** Decides whether frame-skipping is on
   the table. Check danielbodart's config / the decoder ONNX outputs.
4. **What is the worker's CPU spec?** Read `[CPU_DIAGNOSTIC]` from the Test1 ingestion log (ask Q).
   You're optimizing for *that* machine, not the 192-core pod.
5. After speed is solved: **finetune Nemotron** (NeMo cache-aware RNNT) on SAPC2 — but only then.

---

## 9. Your first 3 concrete actions

1. Read `/tmp/nemo_hand/model.py` end-to-end; confirm §4A/§4B by inspection.
2. Restart the pod, stage model+a few utts on `/dev/shm`, and **profile one utterance's decode**
   (`py-spy record` / `cProfile`) → get the RTF breakdown. Report it before changing code.
3. Implement the cheapest high-leverage fix the profile points to (likely incremental features,
   §4A), re-decode the 2k Dev ruler through the official scorer, confirm CER unchanged (~25%) and
   measure the RTF/wall-clock delta. One change at a time; verify each.

**Definition of done for the speed gate:** the submission decodes the streaming set at RTF < 1
on the *worker* CPU spec (or comfortably within the 15000 s budget with margin), and a re-upload
no longer collapses on Test1 (Test1 CER ≈ Dev CER, not 2×). Then — and only then — finetune.

**Slow is smooth. Smooth is fast.**
