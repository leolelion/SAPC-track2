# Track 2 Leaderboard

_Auto-generated from results.jsonl — 9 runs. Do not edit by hand; run `python experiments.py board`._

Gate = SAPC2 Track 2 hard constraint RTF < 1.0. Streaming latency (TTFT/TTLT) only meaningful for true streaming runs.

| Dataset | Model | WER% | CER% | TTFT P50 ms | TTLT P50 ms | RTF | Gate | Date | Commit | Verified |
|---|---|---|---|---|---|---|---|---|---|---|
| Dev | qwen3_asr | 23.34 | 14.23 | — | — | — | **FAIL** | 2026-03-01 | — | **NO** |
| Dev | whisper-large-v2 | 26.08 | 20.36 | — | — | — | — | 2026-03-01 | — | **NO** |
| Dev | parakeet-tdt-0.6b-v2 | 51.64 | 44.64 | — | — | — | — | 2026-03-01 | — | **NO** |
| Dev100 | sherpa_zipformer/standard-finetuned-1ep | 11.8 | 7.75 | — | — | — | — | 2026-04-07 | — | **NO** |
| Dev100 | sherpa_zipformer/kroko | 12.54 | 5.87 | — | — | — | — | 2026-04-07 | — | **NO** |
| Dev100 | sherpa_zipformer/standard | 16.84 | 7.95 | — | — | — | — | 2026-04-07 | — | **NO** |
| Dev_streaming | nemotron_streaming | — | 17.81 | 1895.0 | — | 1.117 | **FAIL** | 2026-05-16 | b40d653 | yes |
| Dev_streaming | nemotron_streaming/phaedrus-children-v17 | — | 25.48 | — | — | — | — | 2026-05-16 | b40d653 | yes |
| Test1 | sherpa_zipformer/standard | 52.77 | 34.59 | 1025 | 423 | — | — | 2026-03-01 | — | **NO** |
