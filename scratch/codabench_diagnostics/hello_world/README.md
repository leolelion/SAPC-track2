# Hello-world diagnostic — SAPC2 Track 2

**This is not a real submission.** It is a minimal-contract probe to
isolate the Codabench ingestion failure mode from anything in our
actual model code.

## What this submission contains

- `model.py` — 5-method `Model` class. `accept_chunk` is a no-op.
  `input_finished` returns the constant string `"hello world"`. Heavy
  logging in `__init__` so the ingestion log reveals cwd, sys.executable,
  sys.path, submission-dir contents, and PID/PPID.
- `setup.sh` — installs nothing, downloads nothing. Dumps host facts:
  Python version, free -h, /proc/meminfo, /tmp writability,
  outbound network probe to huggingface.co. Exits 0.
- `config.yaml` — placeholder.
- `README.md` — this file.

No torch, no onnxruntime, no NeMo, no numpy.

## What outcome means what

| Codabench outcome | What we learn |
|---|---|
| Score reported (any value), `[HELLO_WORLD_DIAG]` lines visible in stdout | Contract works. The bug in our real submission is somewhere in our model or setup. |
| Same `Detected splits: [] / No .predict.csv files found` as v1–v3 | The bug is structural to the zip shape — not our model code. |
| Score reported, but no `[HELLO_WORLD_DIAG]` lines visible | The stdout we get from Codabench is the scoring log, not ingestion. (We already suspect this.) |
| `[SETUP_DIAGNOSTIC]` visible but no `[HELLO_WORLD_DIAG]` | Setup ran; Model never instantiated. Points at ingestion-side import / path issue. |

## How to use

Build the zip:

```bash
cd scratch/codabench_diagnostics/hello_world
zip -r ../hello_world_diagnostic.zip model.py setup.sh config.yaml README.md
```

Upload via the Codabench submission form **only if** organizer email
yields no actionable response in ~24h. Method name suggestion:

```
hello-world-contract-probe
```
