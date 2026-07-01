#!/usr/bin/env python3
# Phase-0 free audit: how many TRAIN utts does max_duration=40s drop, by etiology?
# Sizes the forced-align long-audio lever. If ALS/DS are over-dropped (slow severe speech ->
# longer utts), long-audio segmentation rises above oversampling as the severe-tail lever.
#   python3 phase0_dropped_by_etiology.py --train-csv /workspace/SAPC2/manifest/Train.csv --max-duration 40
import argparse, csv, collections, json
ap = argparse.ArgumentParser()
ap.add_argument("--train-csv", required=True)
ap.add_argument("--max-duration", type=float, default=40.0)
ap.add_argument("--out-json", default="")
a = ap.parse_args()

rows = list(csv.DictReader(open(a.train_csv)))
tot = collections.Counter(); dropped = collections.Counter()
dur_tot = collections.defaultdict(float); dur_drop = collections.defaultdict(float)
for r in rows:
    et = r.get("etiology", "?")
    try: d = float(r["duration"])
    except Exception: d = 0.0
    tot[et] += 1; dur_tot[et] += d
    if d > a.max_duration:
        dropped[et] += 1; dur_drop[et] += d

print(f"max_duration={a.max_duration}s  total_utts={sum(tot.values())}  "
      f"dropped_utts={sum(dropped.values())} ({100*sum(dropped.values())/max(1,sum(tot.values())):.2f}%)")
print(f"{'etiology':<22}{'utts':>8}{'dropped':>9}{'drop%':>8}{'hrs_dropped':>13}")
table = []
for et in sorted(tot, key=lambda k: -dropped[k]):
    pct = 100 * dropped[et] / tot[et] if tot[et] else 0.0
    print(f"{et:<22}{tot[et]:>8}{dropped[et]:>9}{pct:>7.2f}%{dur_drop[et]/3600:>12.2f}h")
    table.append({"etiology": et, "utts": tot[et], "dropped": dropped[et],
                  "drop_pct": round(pct, 2), "hrs_dropped": round(dur_drop[et] / 3600, 3)})
if a.out_json:
    json.dump({"max_duration": a.max_duration, "table": table}, open(a.out_json, "w"), indent=2)
    print("WROTE", a.out_json)
print("PHASE0_DROP_AUDIT_DONE")
