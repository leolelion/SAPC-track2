#!/usr/bin/env python3
"""D2 prep — merge real SAP train with a capped kNN-VC subset into ONE manifest.

`nemo_finetune_v2.py` takes a single `--train-json`, so the "<=25% of training steps" cap from
PLANNED.md has to be enforced here, at manifest-build time, and *asserted* rather than assumed.

Cap semantics. Steps are batches, and batches are drawn uniformly at random from the manifest
(train_ds shuffle=True), so the synthetic share of STEPS equals its share of UTTERANCES, not of
hours. We therefore check the utterance fraction against --max-frac and additionally report the
duration fraction, because a duration blow-out would change effective exposure per step.

This is D2 only (mixed, single stage). D4 (train synth, then re-anneal on real) is a different
run: two invocations of the trainer, real last -- do not fake it by reordering lines here, the
loader shuffles.

Inputs must already be leakage-filtered: build_knnvc_train_manifest.py drops Dev-provenance and
`unknown` files. This script re-checks nothing about provenance -- it cannot, the paths carry no
split label. Run the builder, do not hand-roll the synth json.

Usage:
  build_d2_mixed_manifest.py --real /workspace/nemo_ft/train.json \
      --synth /workspace/nemo_ft/knnvc_train.json \
      --out /workspace/nemo_ft/train_d2_knnvc.json --max-frac 0.25
"""
import argparse, json, random, sys

ap = argparse.ArgumentParser()
ap.add_argument("--real", required=True)
ap.add_argument("--synth", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--max-frac", type=float, default=0.25,
                help="hard cap on synthetic share of utterances (= share of steps)")
ap.add_argument("--seed", type=int, default=13)
a = ap.parse_args()


def load(path):
    rows = []
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                sys.exit(f"FATAL {path}:{ln} not JSON: {e}")
            for k in ("audio_filepath", "duration", "text"):
                if k not in r:
                    sys.exit(f"FATAL {path}:{ln} missing key {k!r}")
            rows.append(r)
    return rows


real = load(a.real)
synth = load(a.synth)

# Trim synth to the cap. n_synth / (n_real + n_synth) <= f  <=>  n_synth <= f/(1-f) * n_real
rng = random.Random(a.seed)
cap_n = int(a.max_frac / (1.0 - a.max_frac) * len(real))
if len(synth) > cap_n:
    rng.shuffle(synth)
    print(f"[cap] synth {len(synth)} -> {cap_n} utts to hold <= {a.max_frac:.0%} of steps")
    synth = synth[:cap_n]

merged = real + synth
rng.shuffle(merged)

n_frac = len(synth) / len(merged)
h_real = sum(r["duration"] for r in real) / 3600.0
h_syn = sum(r["duration"] for r in synth) / 3600.0
d_frac = h_syn / (h_real + h_syn)

# Hard gate: the cap is pre-registered and not negotiable, so fail loudly rather than warn.
if n_frac > a.max_frac + 1e-9:
    sys.exit(f"FATAL synth utterance fraction {n_frac:.4f} exceeds cap {a.max_frac}")

with open(a.out, "w") as f:
    for r in merged:
        f.write(json.dumps(r) + "\n")

print(f"[real ] {len(real):7d} utts {h_real:8.2f} h  {a.real}")
print(f"[synth] {len(synth):7d} utts {h_syn:8.2f} h  {a.synth}")
print(f"[mixed] {len(merged):7d} utts {h_real + h_syn:8.2f} h  -> {a.out}")
print(f"[frac ] utterances/steps = {n_frac:.4f} (cap {a.max_frac})  duration = {d_frac:.4f}")
print("D2_MIXED_MANIFEST_DONE")
