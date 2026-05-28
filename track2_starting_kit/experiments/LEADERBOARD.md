# Track 2 Leaderboard

_Auto-generated from results.jsonl — 16 runs. Do not edit by hand; run `python experiments.py board`._

Gate = SAPC2 Track 2 hard constraint RTF < 1.0. Streaming latency (TTFT/TTLT) only meaningful for true streaming runs.

TTLT cells with `†` suffix were measured on a contended shared host (see entry's `latency_environment` field) and are reliable for **relative** comparison between variants only, **not** for absolute submission gating. TTFT on the same host is absolute-reliable (large enough baseline that jitter is not material). Absolute TTLT requires a dedicated-CPU host or the Codabench eval VM.

| Dataset | Model | WER% | CER% | TTFT P50 ms | TTLT P50 ms | RTF | Gate | Date | Commit | Verified |
|---|---|---|---|---|---|---|---|---|---|---|
| Dev | qwen3_asr | 23.34 | 14.23 | — | — | — | **FAIL** | 2026-03-01 | — | **NO** |
| Dev | whisper-large-v2 | 26.08 | 20.36 | — | — | — | — | 2026-03-01 | — | **NO** |
| Dev | parakeet-tdt-0.6b-v2 | 51.64 | 44.64 | — | — | — | — | 2026-03-01 | — | **NO** |
| Dev100 | sherpa_zipformer/standard-finetuned-1ep | 11.8 | 7.75 | — | — | — | — | 2026-04-07 | — | **NO** |
| Dev100 | sherpa_zipformer/kroko | 12.54 | 5.87 | — | — | — | — | 2026-04-07 | — | **NO** |
| Dev100 | sherpa_zipformer/standard | 16.84 | 7.95 | — | — | — | — | 2026-04-07 | — | **NO** |
| Dev_10k | nemotron_streaming/modelclass_int8static | 27.46 | 21.59 | — | — | 0.116 | PASS | 2026-05-28 | 27bf6d1 | yes |
| Dev_10k | sherpa_zipformer/kroko | 33.57 | 23.92 | — | — | — | — | 2026-05-28 | 27bf6d1 | yes |
| Dev_streaming | nemotron_streaming/modelclass_int8static | 28.08 | 22.31 | 1592 | 270† | 0.243 | PASS | 2026-05-27 | 27bf6d1 | yes |
| Dev_streaming | nemotron_streaming/danielbodart_int8_static | 28.44 | 22.72 | 1635.0 | 181.0† | — | — | 2026-05-27 | 27bf6d1 | yes |
| Dev_streaming | nemotron_streaming/danielbodart_int8_dynamic | 28.44 | 22.72 | 1628.0 | 168.0† | — | — | 2026-05-27 | 27bf6d1 | yes |
| Dev_streaming | nemotron_streaming/danielbodart_fp32 | 29.9 | 24.45 | 1698.0 | 278.0† | — | — | 2026-05-27 | 27bf6d1 | yes |
| Dev_streaming | nemotron_streaming/int8_kquant_onnx | 31.04 | 24.61 | 1687 | 263† | — | — | 2026-05-27 | 27bf6d1 | yes |
| Dev_streaming | nemotron_streaming | — | 17.81 | 1895.0 | — | 1.117 | **FAIL** | 2026-05-16 | b40d653 | yes |
| Dev_streaming | nemotron_streaming/phaedrus-children-v17 | — | 25.48 | — | — | — | — | 2026-05-16 | b40d653 | yes |
| Test1 | sherpa_zipformer/standard | 52.77 | 34.59 | 1025 | 423 | — | — | 2026-03-01 | — | **NO** |
