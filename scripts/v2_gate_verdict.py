#!/usr/bin/env python3
# Pre-registered Gate-1 (guardrail) verdict for v2 vs v1 (research/38). Reads the faithful-eval bootstrap
# JSONs and prints PASS/FAIL against thresholds fixed BEFORE looking. PROCEED to full run iff all pass.
import argparse, json, sys

ap = argparse.ArgumentParser()
ap.add_argument("--devdiag-json", required=True)   # streaming_cer_bootstrap output on Dev_diag
ap.add_argument("--devdiag-empties-json", required=True)
ap.add_argument("--dev500-json", required=True)
ap.add_argument("--pd-json", required=True)
a = ap.parse_args()

def load(p):
    try: return json.load(open(p))
    except Exception as e: print(f"[FAIL] cannot read {p}: {e}"); sys.exit(2)

dd = load(a.devdiag_json); de = load(a.devdiag_empties_json)
d5 = load(a.dev500_json); pd = load(a.pd_json)

dd_cer = dd["mean_cer_pct"]
empty_frac = de["empty_frac_pct"]
dd_ds = dd.get("by_etiology", {}).get("Down Syndrome", {}).get("mean_cer_pct")
pd_cer = pd["mean_cer_pct"]
d5_cer = d5["mean_cer_pct"]

# v1 baselines (research/37): Dev_diag 23.58% / empties 7.3% / DS 33.39% / PD 4.49% / Dev-500 ~9.91-10.6%
checks = [
    ("Dev_diag empty-rate < 5.0%",        empty_frac, empty_frac < 5.0),
    ("Dev_diag CER < 22.0% (v1 23.58)",   dd_cer,     dd_cer < 22.0),
    ("PD CER <= 5.0% (forgetting; v1 4.49)", pd_cer,  pd_cer <= 5.0),
    ("Dev-500 CER <= 10.5% (v1 ~9.91)",   d5_cer,     d5_cer <= 10.5),
    ("DS CER not worse than 33.39%",      dd_ds,      (dd_ds is not None and dd_ds <= 33.39)),
]
print("=== v2 GUARDRAIL GATE-1 VERDICT (pre-registered, research/38) ===")
allpass = True
for name, val, ok in checks:
    allpass = allpass and ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:38s} -> {val}")
print(f"\nVERDICT: {'PROCEED to full run' if allpass else 'STOP (do not pay for full run)'}")
sys.exit(0 if allpass else 1)
