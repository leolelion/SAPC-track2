# nemotron_streaming — SAPC2 Track 2 submission

Streaming ASR wrapping the int8-static ONNX export of
[`nvidia/nemotron-speech-streaming-en-0.6b`](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b)
(via [`danielbodart/nemotron-speech-600m-onnx`](https://huggingface.co/danielbodart/nemotron-speech-600m-onnx),
pinned revision in `setup.sh`). att_context_size=[70, 6], 560 ms model chunks,
RNN-T greedy decode.

## Files

| File | Purpose |
|---|---|
| `model.py` | 5-method `Model` class (set_partial_callback / reset / accept_chunk / input_finished). ONNX-based; uses NeMo only for the deterministic mel preprocessor. Logs a CPU diagnostic line on `__init__` for telemetry. |
| `config.yaml` | num_threads + weights paths |
| `setup.sh` | Builds a local venv (--system-site-packages so torch is inherited), pins NeMo[asr]==2.5.x, ORT, numpy<2 to the system torch version, downloads weights, runs a silence smoke test |
| `weights/` (created by setup.sh) | encoder_model.onnx (int8), decoder_model.onnx (fp32), tokens.txt — ~880 MB total |

## Reference numbers (Dev_streaming, 123 utts, sclite scoring)

| Metric | Value |
|---|---|
| CER% | 22.31 |
| WER% | 28.08 |
| TTFT P50 (ms) | ~1590 |
| TTLT P50 (ms) | ~240 (host-jitter-inflated; eval VM will differ) |
| compute-RTF P50 | 0.22 (Xeon Platinum 8568Y+, 4 threads — eval VM TBD) |
| compute-RTF max | 0.34 |

## Notes for the eval VM

- `model.py.__init__` prints `[CPU_DIAGNOSTIC] {...}` to stdout AND writes
  `/tmp/cpu_diagnostic.json`. Contents: lscpu, /proc/cpuinfo head, meminfo,
  loadavg, and a 1024² × 20 iter matmul timing — purpose is to calibrate
  RTF/TTFT/TTLT against a known reference. The same diagnostic on this
  development host returned matmul_1024_20iter_ms ≈ 604 ms.
- Encoder is int8 (876 MB), decoder is fp32 (36 MB). Quantizing the
  decoder didn't help in our ablation (CER drift > 0.5 with no size
  benefit at this scale).
- No pass-1/pass-2 differentiation — could add later if SAPC2 ingestion
  source confirms it's permitted.
