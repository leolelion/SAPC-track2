# SAPC-template — Claude instructions

SAPC2 Challenge. Track 2 = streaming ASR for dysarthric speech, CPU-only eval,
hard constraint **RTF < 1.0**. Model kits live in `track2_starting_kit/`.

## Experiment tracking — MANDATORY

Every evaluation run (RunPod or local) MUST be recorded in the ledger before the
work is considered done. The protocol and schema are in
[track2_starting_kit/experiments/README.md](track2_starting_kit/experiments/README.md).

Rules:
- After any eval run, append an entry with
  `python track2_starting_kit/experiments/experiments.py add ...`.
- Archive the raw eval CSV off-pod **before** the RunPod pod is destroyed, and
  record its location in the entry's `raw_output` field.
- Commit `results.jsonl` and `LEADERBOARD.md` together. A run is not finished
  until its entry is committed.
- Failed runs get an entry too (`status: failed`) so dead ends aren't repeated.
- Never invent metrics. Unknown values are `null`; estimated values get
  `verified: false` and an explanatory note.

Check `track2_starting_kit/experiments/LEADERBOARD.md` for current standings
before proposing new experiments.
