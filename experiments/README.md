# experiments/

File-based experiment tracking for SAPC2 Track 2.

## Layout
```
experiments/
├── summary.csv          # one row per experiment (machine-readable leaderboard)
├── README.md            # this file
└── exp_XXX/             # per-experiment artifacts
    ├── config.yaml      # exact config snapshot used
    ├── git.txt          # git commit hash + dirty/clean status
    ├── metrics.json     # CER/WER + latency summary (from compute_metrics/compute_latency)
    ├── decode.log       # decode stdout (optional)
    └── notes.md         # hypothesis, observations, conclusion, next step
```

## Protocol
1. Allocate the next `exp_XXX`. Snapshot config + `git rev-parse HEAD` into the dir.
2. Run decode (`local_decode.py`) → run `evaluate.sh` stages 2 and 3 → save metrics.json.
3. Add a row to `summary.csv` and an entry to `../EXPERIMENT_LOG.md`.
4. Compare against the current best; record keep/reject and the next question in `notes.md`.

`summary.csv` columns:
`exp_id,date,git_commit,model,change_vs_prev,cer,wer,cer_stream,ttft_p50_ms,ttlt_p50_ms,latency_mean_ms,rtf,status,notes`

Pareto figure of merit = `latency_mean_ms = mean(ttft_p50_ms, ttlt_p50_ms)`, plotted against `cer`.
