# nemotron_streaming_v7_minimal — SAPC2 Track 2 submission

**Copy-the-baseline play.** Built by stripping our v6 nemo-free kit
down to the minimum that mirrors the upstream
`track2_starting_kit/streaming_zipformer/` baseline (the one
publicly confirmed working on Codabench at CER 34.59 on Test1).

## What's stripped vs v6

- `[SETUP_VERIFY]` block at top of model.py — gone.
- `_THREADS` / `OMP_NUM_THREADS` / `MKL_NUM_THREADS` env juggling —
  gone. Falls back to ORT / PyTorch defaults.
- `torch.set_num_threads` / `set_num_interop_threads` calls — gone.
- `setup.sh` smoke test — gone entirely (not wrapped, removed).
- Final `pip list` + import check block at end of `setup.sh` — gone.
- `atexit` hook in Model — gone.
- Periodic MEM_DIAGNOSTIC every 100 utts — gone.

## What's kept

- 5-method `Model` class — `__init__`, `set_partial_callback`,
  `reset`, `accept_chunk`, `input_finished`.
- MinimalMelPreprocessor (pure torch + numpy mel implementation,
  byte-equivalent to NeMo at max abs diff 1.9e-6 — see
  `scripts/audit/preproc_byte_equivalence/`).
- Lazy ORT session loading — `__init__` is cheap, sessions open
  on first `accept_chunk` / `input_finished`.
- Greedy RNN-T decode, encoder/decoder state cache, drain on
  `input_finished`.
- One-shot `[CPU_DIAGNOSTIC]` print at construction.
- One-shot `[MEM_DIAGNOSTIC] event=init_done` print at construction.

## What's added (one line)

At the end of `input_finished()`, after the result is computed but
before return:

```
[END_OF_RUN] utts_processed=<N> final_state=ok
```

Converts the question *"did the model finish each utterance?"* from
silent to single-line-visible IF anything ends up in a captured log.

## Files

| File | Notes |
|---|---|
| `model.py` | Simplified Model class. No SETUP_VERIFY, no thread env, no atexit. Inline MinimalMelPreprocessor. |
| `setup.sh` | Three stages matching `streaming_zipformer/setup.sh` shape: detect env, install packages, download weights. `set -e` (not `set -euo pipefail`). No smoke test. |
| `config.yaml` | Same as v6. |
| `weights/` (created by setup.sh) | encoder_model.onnx (int8), decoder_model.onnx (fp32), tokens.txt — ~880 MB total. |

## Reference numbers (carry forward — model + decode unchanged)

| Set | Scorer | CER% | WER% | Source |
|---|---|---|---|---|
| Dev_10k (10,521 utts) | sclite, dual-ref MIN | **21.59** | **27.46** | `f160e39` |

## Outcome interpretation

- Scores any CER → the surface area we'd been adding (SETUP_VERIFY,
  thread env, smoke test, atexit) was the issue.
- Same `Detected splits: []` with NO captured output → failure is
  not in our submission code; need organizer-side ingestion log.
- Same scoring log but with `[END_OF_RUN]` lines visible somewhere
  → we ran through utterances but predict.csv didn't land where
  scoring expects. Codabench contract / output-path issue.
