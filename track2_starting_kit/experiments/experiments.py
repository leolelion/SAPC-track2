#!/usr/bin/env python3
"""Track 2 experiment ledger: append eval results and render the leaderboard.

results.jsonl is the committed source of truth (one JSON object per eval run).
Raw eval CSVs stay gitignored -- archive them off-pod and record the path in
the `raw_output` field so a finished run is never lost when a pod is torn down.

Usage:
  python experiments.py add --file entry.json
  python experiments.py add --model parakeet_ctc --kit parakeet_ctc \\
      --dataset Dev100 --n-utt 100 --wer 12.5 --cer 5.9 --rtf 0.83 \\
      --ttft-p50-ms 480 --ttlt-p50-ms 210 --git-commit 02aab0f \\
      --hardware "RunPod CPU 16c / xiuwenz2/sapc2-runtime" \\
      --raw-output "s3://sapc-results/2026-05-16-parakeet-ctc/" --verified \\
      --notes "first streaming run"
  python experiments.py board          # regenerate LEADERBOARD.md
  python experiments.py show <id>      # print one entry
  python experiments.py validate       # check ledger integrity
"""
import argparse
import datetime
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LEDGER = os.path.join(HERE, "results.jsonl")
BOARD = os.path.join(HERE, "LEADERBOARD.md")
RTF_GATE = 1.0  # SAPC2 Track 2 hard constraint: RTF must be < 1.0

# Flag -> ledger field. Floats parsed as float, ints as int.
ARG_FIELDS = {
    "id": str, "date": str, "model": str, "kit": str, "model_source": str,
    "git_commit": str, "dataset": str, "n_utt": int, "wer": float, "cer": float,
    "ttft_p50_ms": float, "ttlt_p50_ms": float, "rtf": float, "hardware": str,
    "raw_output": str, "status": str, "notes": str,
}


def load():
    if not os.path.exists(LEDGER):
        return []
    rows = []
    with open(LEDGER) as f:
        for i, ln in enumerate(f, 1):
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError as e:
                sys.exit(f"results.jsonl line {i} is not valid JSON: {e}")
    return rows


def cmd_add(args):
    entry = json.load(open(args.file)) if args.file else {}
    for field in ARG_FIELDS:
        val = getattr(args, field, None)
        if val is not None:
            entry[field] = val
    if args.verified:
        entry["verified"] = True
    entry.setdefault("verified", False)
    entry.setdefault("status", "complete")
    entry.setdefault("date", datetime.date.today().isoformat())
    if not entry.get("id"):
        entry["id"] = f"{entry['date']}-{entry.get('model', 'run').replace('/', '-')}"
    rtf = entry.get("rtf")
    entry["rtf_gate"] = None if rtf is None else (rtf < RTF_GATE)
    with open(LEDGER, "a") as f:
        f.write(json.dumps(entry) + "\n")
    gate = {True: "PASS", False: "FAIL", None: "n/a"}[entry["rtf_gate"]]
    print(f"appended: {entry['id']}  (RTF gate: {gate})")
    cmd_board(args)


def _fmt(v):
    return "—" if v is None else v


def cmd_board(args):
    rows = load()
    rows.sort(key=lambda r: (str(r.get("dataset", "")),
                             r.get("wer") if r.get("wer") is not None else 1e9))
    lines = [
        "# Track 2 Leaderboard",
        "",
        f"_Auto-generated from results.jsonl — {len(rows)} runs. "
        "Do not edit by hand; run `python experiments.py board`._",
        "",
        "Gate = SAPC2 Track 2 hard constraint RTF < 1.0. "
        "Streaming latency (TTFT/TTLT) only meaningful for true streaming runs.",
        "",
        "| Dataset | Model | WER% | CER% | TTFT P50 ms | TTLT P50 ms | RTF | Gate | Date | Commit | Verified |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        gate = {True: "PASS", False: "**FAIL**", None: "—"}[r.get("rtf_gate")]
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            _fmt(r.get("dataset")), _fmt(r.get("model")), _fmt(r.get("wer")),
            _fmt(r.get("cer")), _fmt(r.get("ttft_p50_ms")), _fmt(r.get("ttlt_p50_ms")),
            _fmt(r.get("rtf")), gate, _fmt(r.get("date")), _fmt(r.get("git_commit")),
            "yes" if r.get("verified") else "**NO**"))
    with open(BOARD, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {BOARD} ({len(rows)} runs)")


def cmd_show(args):
    for r in load():
        if r.get("id") == args.id:
            print(json.dumps(r, indent=2))
            return
    sys.exit(f"no entry with id {args.id!r}")


def cmd_validate(args):
    rows = load()
    seen, problems = set(), []
    for r in rows:
        rid = r.get("id")
        if not rid:
            problems.append("entry missing 'id'")
        elif rid in seen:
            problems.append(f"duplicate id: {rid}")
        seen.add(rid)
        for req in ("model", "dataset", "date"):
            if not r.get(req):
                problems.append(f"{rid}: missing '{req}'")
    if problems:
        print("\n".join(problems))
        sys.exit(1)
    print(f"OK — {len(rows)} entries, no issues")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="append a result")
    a.add_argument("--file", help="JSON file with a prepared entry")
    a.add_argument("--verified", action="store_true",
                   help="set if metrics were produced by a trusted run, not estimated")
    for field, typ in ARG_FIELDS.items():
        a.add_argument("--" + field.replace("_", "-"), type=typ, dest=field)
    a.set_defaults(func=cmd_add)

    b = sub.add_parser("board", help="regenerate LEADERBOARD.md")
    b.set_defaults(func=cmd_board)

    s = sub.add_parser("show", help="print one entry")
    s.add_argument("id")
    s.set_defaults(func=cmd_show)

    v = sub.add_parser("validate", help="check ledger integrity")
    v.set_defaults(func=cmd_validate)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
