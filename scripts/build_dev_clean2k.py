#!/usr/bin/env python3
"""Build Dev_clean2k.csv — the model-comparison slice that no arm was selected against.

Why this exists (2026-07-29). Every parakeet arm selects checkpoints on `val.json`, which turns out
to span only **7 speakers** across its 2000 utterances, and `Val2k` is literally those same 2000
utterances — so Val2k is fully contaminated for parakeet and narrow for anyone. It also gives wildly
optimistic-or-pessimistic readings: the SAME zipformer under the SAME official scorer scores 29.78%
on Val2k and 18.19% here. An 11.6-point swing from the speaker draw alone.

This slice is drawn from Dev excluding both `val.json` ids and `Dev_diag` ids, round-robin across
speakers so no speaker dominates. Result: 2000 utts / 122 speakers / <=17 per speaker / all five
etiologies.

Reproduces exactly with the pinned seed. Writes into the manifest dir so `evaluate.sh --split
Dev_clean2k` resolves references the normal way (stage 1 rebuilds them from this CSV).

Usage:
  build_dev_clean2k.py --manifest-dir /workspace/SAPC2/manifest --val-json /workspace/nemo_ft/val.json
"""
import argparse, collections, csv, json, os, random

ap = argparse.ArgumentParser()
ap.add_argument("--manifest-dir", default="/workspace/SAPC2/manifest")
ap.add_argument("--val-json", default="/workspace/nemo_ft/val.json")
ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--seed", type=int, default=20260729)
a = ap.parse_args()

D = a.manifest_dir
val = {os.path.basename(json.loads(l)["audio_filepath"])[:-4] for l in open(a.val_json) if l.strip()}
dd = {r["id"] for r in csv.DictReader(open(f"{D}/Dev_diag.csv", newline=""))}
rows = [r for r in csv.DictReader(open(f"{D}/Dev.csv", newline=""))
        if r["id"] not in val and r["id"] not in dd]
print("eligible Dev rows:", len(rows), "speakers:", len({r["speaker"] for r in rows}))

by = collections.defaultdict(list)
for r in rows:
    by[r["speaker"]].append(r)
rng = random.Random(a.seed)
for v in by.values():
    rng.shuffle(v)

# Round-robin across speakers: taking a flat random sample would let the heaviest speakers dominate,
# which is exactly the defect that made val.json (7 speakers) useless for model selection.
pick, spk, i = [], sorted(by), 0
while len(pick) < a.n:
    progressed = False
    for s in spk:
        if i < len(by[s]):
            pick.append(by[s][i])
            progressed = True
            if len(pick) >= a.n:
                break
    if not progressed:
        break
    i += 1

hdr = list(rows[0].keys())
out = f"{D}/Dev_clean2k.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=hdr)
    w.writeheader()
    w.writerows(pick)

c = collections.Counter(r["speaker"] for r in pick)
print(f"WROTE {out} n={len(pick)} speakers={len(c)} max_per_speaker={max(c.values())}")
print("etiologies:", dict(collections.Counter(r["etiology"] for r in pick)))
print("DEV_CLEAN2K_DONE")
