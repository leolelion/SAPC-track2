# 00 - Problem Statement And Verified Pipeline

Executive summary:
1. The exact Nemotron submission path is locally inspectable and loads on CPU, but local Mac timings are only provisional.
2. The implementation is a two-ONNX-session cache-aware FastConformer RNNT with Python-managed streaming state.
3. The handoff's two suspected inefficiencies are real: full-utterance feature recompute and per-symbol decoder ORT calls.
4. Static ONNX inspection shows no duration/TDT head in the bundled decoder export.
5. The actual optimization target is still underspecified until we have the Codabench worker CPU diagnostic and SAP data profile.

## Scope

Mission: make the Nemotron-0.6B Track-2 submission fast enough that the official worker no longer truncates/timeouts, while preserving the validated Dev accuracy path.

The current strategic fact pattern from prior notes is:
- Exact submission on 2k Dev via official scorer: about 24.96% CER / 31.27% WER.
- Same submission on Test1 worker: 51.52% CER / 58.99% WER with TTFT p50 2270 ms / p90 6205 ms.
- Handoff inference: the Test1 collapse is speed-induced, not packaging or text-normalization failure.

## Verified Local Paths

- Repo: `/Users/o/Downloads/SAPC-template`
- Handoff: `/Users/o/Downloads/SAPC-template/HANDOFF_NEMOTRON_CODEX.md`
- Exact upload zip: `/Users/o/Downloads/nemo_submission.zip`
- Extracted local working copy for profiling: `/tmp/nemo_submission_codex_profile`
- Handoff text-only extraction: `/tmp/nemo_hand`
- Submission entrypoint: `/tmp/nemo_submission_codex_profile/model.py`
- Frontend: `/tmp/nemo_submission_codex_profile/localmel.py`
- ONNX files:
  - `/tmp/nemo_submission_codex_profile/weights/encoder_model.onnx`
  - `/tmp/nemo_submission_codex_profile/weights/encoder_model.onnx.data`
  - `/tmp/nemo_submission_codex_profile/weights/decoder_model.onnx`
  - `/tmp/nemo_submission_codex_profile/weights/decoder_model.onnx.data`

## Runtime Contract

The SAPC2 interface is preserved by `model.py`:
- `Model.__init__()`
- `set_partial_callback(fn)`
- `reset()`
- `accept_chunk(np.ndarray) -> str`
- `input_finished() -> str`

Audio arrives as 100 ms, 1600-sample, float32, 16 kHz mono chunks. The code assumes a single Decoder thread, so there is no internal locking.

## Competition-Specific Warmup Interpretation

Based on `Track2.md` and the local organizer-style `track2_starting_kit/local_decode.py`, warmup is technically possible if it is done inside `Model.__init__()`:
- `__init__()` is called once at startup to load weights/tokenizer/etc. A dummy encoder/decoder pass for kernel/session warmup fits this startup-preparation role.
- In the local streaming harness, `audio_send_start_time` is recorded by the audio sender thread after model construction, so `__init__()` warmup does not directly enter TTFT/TTLT timestamps.
- The official Track 2 page still imposes a 15000 s time limit per submission, so warmup consumes total budget even if it does not count as streaming latency.
- The batch accuracy pass is described as multiprocess. If the official ingestion creates multiple model processes, `__init__()` warmup may run once per worker, multiplying startup cost.

Warmup should not be placed in `reset()`, `accept_chunk()`, or `input_finished()` for latency purposes. `reset()` runs per file and, in the streaming pass, can race with the audio sender, so doing heavy work there would increase backlog and likely harm TTFT/TTLT.

Safe warmup requirements:
- Use only synthetic/zero input and bundled model assets.
- Call `reset()` after warmup so no dummy audio, tokens, caches, callback text, or timing state leaks into a scored utterance.
- Keep it bounded and benchmarked separately against total ingestion wall time.
- Treat it as a tail-latency polish, not the primary fix for RTF > 1.

## Corrected Stack Facts

Verified true:
- Encoder is cache-aware streaming FastConformer-style ONNX with rolling channel/time caches.
- Decoder ONNX combines prediction-network LSTM and joiner.
- The local mel frontend uses bundled `filterbank.npy` and `window.npy` with pre-emphasis 0.97, `n_fft=512`, window 400, hop 160, 128 mel bins, log.
- Thread count is controlled by `SAPC2_THREADS` or default `4`, then applied to ORT and PyTorch.

Corrections / caveats:
- `config.yaml` is not read by `model.py`; changing it will not affect thread count or file names unless code changes too.
- The zip includes a Linux NumPy wheel in `wheels/`, despite the handoff warning to avoid NumPy replacement. Current `setup.sh` only asks pip to install `onnxruntime` if missing, so whether NumPy is touched depends on the worker environment and pip dependency resolution.
- Local `sherpa_onnx` is installed, but the current submission does not use it.
- Static decoder ONNX inspection found no duration/TDT output. The bundled decoder appears to be vanilla RNNT logits plus LSTM state, so TDT frame-skipping is not available from this exact export.

## Pipeline Reconstruction

For each utterance:
1. `reset()` clears raw audio chunks, mel cache, encoder caches, decoder LSTM state, emitted tokens, and timing counters.
2. Each `accept_chunk()` appends the 100 ms audio chunk to `_raw_chunks`, increments sample count, and calls `_run_steps(is_final=False)`.
3. `_run_steps()` waits until enough samples exist for the next 56-frame model chunk. Once ready, `_ensure_features()` concatenates all raw chunks and reruns the full mel frontend over the entire utterance so far.
4. `_run_steps()` slices the current 56 new mel frames, prepends 9 cached mel frames, and calls `_encode_and_decode()`.
5. `_encode_and_decode()` runs the encoder ONNX once for that chunk, updates encoder caches, then greedily decodes frame-by-frame. For each encoder frame it calls decoder ONNX repeatedly until blank or `MAX_SYMBOLS_PER_FRAME=10`.
6. Non-drain partials are detokenized and emitted through the callback when changed.
7. `input_finished()` processes any final partial chunk and runs one all-zero drain chunk for right-context flush.

## Phase-0 Required Fields

Repo path / entrypoint:
- `/tmp/nemo_submission_codex_profile/model.py` for the exact upload.
- Official challenge harness remains under `/Users/o/Downloads/SAPC-template/track2_starting_kit/local_decode.py`, but repo rules say not to edit its semantics.

Hardware:
- Local provisional machine: Apple M3 MacBook Air, 8 cores (4 performance + 4 efficiency), 8 GB RAM, macOS 15.3.2 arm64, Python 3.13.0, PyTorch 2.6.0, ONNX Runtime 1.24.4.
- Target worker: unknown. Need the Test1 ingestion log line `[CPU_DIAGNOSTIC]` or `/tmp/cpu_diagnostic.json` from Codabench.

Target:
- Working assumption: one streaming decoder instance must keep up with real-time 100 ms chunk pacing on the Codabench CPU worker.
- Hard requirement: full submission must finish within the 15000 s challenge budget.
- Practical speed gate: per-stream RTF comfortably below 1.0 on the worker, with enough headroom that Test1 CER is close to Dev CER rather than doubled.
- TTFT goal: improve materially from observed worker TTFT p50 2270 ms / p90 6205 ms. A concrete p50/p90 target should be chosen after worker CPU and scoring weight are known.

Accuracy guardrail:
- Speed-only changes should preserve the exact current decoding result. Working guardrail: no measurable regression on the 2k Dev ruler; any absolute CER increase over 0.1 points needs explicit justification and repeat scoring.
- Structural runtime changes must be gated by official scorer output, not transcript eyeballing.

## Current Unknowns

- Actual Codabench worker CPU model, core count, SIMD support, memory, and contention.
- Real SAP utterance duration distribution seen by the Nemotron submission.
- Whether Test1 truncation came from per-stream RTF > 1, global 15000 s timeout, harness concurrency oversubscription, or a combination.
- Whether sherpa-onnx can ingest this exact cache-aware NeMo export without re-export.
- Whether a Linux x86 ORT profile ranks blockers the same way as the local ARM profile.
