# Track 2 experiment tracking

Experiments run on ephemeral RunPod pods. Raw eval CSVs are gitignored and die
with the pod. This directory keeps the **summary metrics** in git so results
are never lost.

## Files

| File | Role |
|---|---|
| `results.jsonl` | Committed source of truth. One JSON object per eval run, append-only. |
| `LEADERBOARD.md` | Auto-generated view, sorted by WER. Never edit by hand. |
| `experiments.py` | Append entries and regenerate the leaderboard. Stdlib only. |

`results.jsonl` is **not** gitignored (`.gitignore` excludes `*.csv` and
`results/`, not `.jsonl` or `experiments/`). Raw CSVs stay out of git — archive
them and record where in the `raw_output` field.

## Entry schema

```jsonc
{
  "id": "2026-05-16-parakeet_ctc",   // unique; auto = <date>-<model>
  "date": "2026-05-16",
  "model": "parakeet_ctc",            // kit name or model variant
  "kit": "parakeet_ctc",              // dir under track2_starting_kit/
  "model_source": "scott-morgan-foundation/sapc2-track1-parakeet-ctc",
  "git_commit": "02aab0f",            // repo commit the run used
  "dataset": "Dev100",                // Dev100 | Dev | Test1
  "n_utt": 100,
  "wer": 12.5, "cer": 5.9,            // percent; from evaluate.sh stage 2
  "ttft_p50_ms": 480,                 // streaming latency; null if batch-only
  "ttlt_p50_ms": 210,
  "rtf": 0.83,                        // real-time factor on CPU
  "rtf_gate": true,                   // auto-set: rtf < 1.0
  "hardware": "RunPod CPU 16c / xiuwenz2/sapc2-runtime",
  "raw_output": "s3://sapc-results/2026-05-16-parakeet_ctc/",
  "verified": true,                   // false if a metric was estimated
  "status": "complete",               // complete | failed | partial
  "notes": "first true-streaming run"
}
```

Unknown numeric metrics → `null`. Never invent values; leave `verified: false`
with a note if anything is estimated.

## Run protocol (Claude follows this for every RunPod eval)

1. **Before the run** record the repo commit (`git rev-parse --short HEAD`),
   the kit, and the config used.
2. **Run** the eval on the pod (`local_decode.py` + `evaluate.sh`, or the kit's
   `eval_dev100.py`).
3. **Collect metrics**: WER/CER from `evaluate.sh` stage 2; TTFT/TTLT P50 from
   the partial-results JSON; RTF from wall-clock vs. audio duration.
4. **Archive the raw CSV** off-pod (S3 / persistent volume) before the pod is
   destroyed. Note the location.
5. **Append the entry**:
   ```bash
   python track2_starting_kit/experiments/experiments.py add \
       --model parakeet_ctc --kit parakeet_ctc --dataset Dev100 --n-utt 100 \
       --wer 12.5 --cer 5.9 --rtf 0.83 --ttft-p50-ms 480 --ttlt-p50-ms 210 \
       --git-commit "$(git rev-parse --short HEAD)" \
       --hardware "RunPod CPU 16c" --raw-output "s3://..." --verified \
       --notes "first streaming run"
   ```
   `add` auto-regenerates `LEADERBOARD.md`.
6. **Commit** `results.jsonl` and `LEADERBOARD.md` in the same commit:
   `git commit -m "experiment: <model> on <dataset> — WER X% / RTF Y"`.
7. A run that **fails** still gets an entry with `status: failed` and a note —
   negative results stop us repeating dead ends.

## Rules

- One entry per (model, config, dataset) eval run. Re-runs get a new entry.
- A run is not "done" until its entry is committed.
- `rtf_gate: false` means the model violates the Track 2 RTF < 1.0 constraint —
  unusable for submission regardless of accuracy.
- Historical `hist-*` entries were backfilled from notes (`verified: false`).
