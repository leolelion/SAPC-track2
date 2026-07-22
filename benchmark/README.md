# benchmark/ — Parakeet-RNNT/TDT vs the Track 2 baseline

Scaffolding for the investigation in `docs/benchmark_plan.md` (plan) and
`docs/repo_analysis.md` (repo audit). **Built and dry-checked locally on
2026-07-14. No accuracy/latency has been measured — no NeMo, no data, no GPU on
the dev box.** Real numbers come from the pod (Q is lining up data + host).

## What's here
```
benchmark/
├── run_eval.py               # driver: `offline` (Stage A) | `streaming` (Stage B)
├── metrics.py                # CER/WER via evaluate.sh (wraps, never reimplements)
├── latency.py                # TTFT/TTLT via utils/compute_latency.py (wraps)
├── smoke_dry.py              # dry, NeMo-free structural check (run this now)
└── models/
    ├── parakeet_offline.py   # offline NeMo transcribe -> CER ceiling ONLY
    └── nemotron_streaming.py # 5-method streaming Model SKELETON (UNVERIFIED)
```

## Design rules honored
- **Scoring is delegated, not reinvented.** `latency.py` imports the organizers'
  `utils/compute_latency.py`; `metrics.py` shells out to `evaluate.sh`. Numbers
  are byte-identical to the official scorer (house rule; memory
  `validate-against-real-harness`).
- **No fake streaming.** Parakeet-TDT/RNNT are offline-only (verified), so
  `parakeet_offline.py` produces an accuracy CSV **only** and never touches the
  streaming pass. `nemotron_streaming.py` targets the real cache-aware
  FastConformer-RNNT and *raises* rather than buffer-and-rerun.
- **NeMo imports are lazy** so everything import-checks on this box.
- **Baseline = zipformer**, not "Parakeet-CTC" (which does not exist here).

## Dry check (works now, no deps beyond the repo)
```bash
python3 benchmark/smoke_dry.py      # expect: all 4 checks pass, exit 0
```

## On the pod (once $DATA_ROOT + NeMo exist)
Prereq: set `DATA_ROOT` / `PROJ_ROOT` / `SCTK_DIR` in `evaluate.sh` first.

**Stage A — offline accuracy ceiling** (smoke 20 utts first):
```bash
python3 benchmark/run_eval.py offline \
  --checkpoint nvidia/parakeet-tdt-0.6b-v2 \
  --manifest-csv $DATA_ROOT/manifest/Dev.csv --data-root $DATA_ROOT \
  --out-csv Dev.parakeet_tdt.csv --split Dev --device cuda --limit 20
# then drop --limit for full Dev; repeat for nvidia/parakeet-rnnt-0.6b
```
STOP if best offline CER isn't clearly below the zipformer's Dev CER.

**Stage B — real streaming** (only if Stage A promising; CPU runtime for latency):
```bash
python3 benchmark/run_eval.py streaming \
  --submission-dir <packaged 5-method dir> \
  --manifest-csv $DATA_ROOT/manifest/Dev.csv \
  --streaming-manifest-csv $DATA_ROOT/manifest/Dev_streaming.csv \
  --data-root $DATA_ROOT \
  --out-csv Dev.stream.csv --out-partial-json Dev.stream.partial.json
```
`nemotron_streaming.py._stream_step` must be wired to the real NeMo cache-aware
call and validated here before any conclusion (it currently raises on purpose).

## Known open items (for the pod, not the dev box)
1. Wire + verify `nemotron_streaming._stream_step` against NeMo's
   `speech_to_text_cache_aware_streaming_infer.py`; set `_model_chunk_samples`
   from the model's streaming params.
2. Confirm each checkpoint name resolves via `from_pretrained`.
3. `metrics.py` CER/WER parse is best-effort; the authoritative value is the
   echoed `evaluate.sh` stdout.
