#!/usr/bin/env python3
"""Phase 0 — rate-stratified + speaker-stratified read of a banked decomposition,
with utterance-level bootstrap confidence intervals.

WHY THIS EXISTS
---------------
`scripts/error_decomposition.py` established the error budget on Dev_diag severe
(deletions 13.04 of 18.73 CER pts, whole words) and that CER tracks speaking RATE
(spearman -0.254, slowest quartile 40.14% vs fastest 10.31%) and NOT word count
(+0.027). Every lever we now consider -- lookahead, front-end time compression,
rate-targeted training -- is aimed at that rate axis specifically.

An arm that lowers the AVERAGE CER without lowering the SLOW-QUARTILE CER did not
attack the target; it got lucky somewhere else. This script makes the stratified read
a one-liner on any banked run, so the decision metric is never the average alone.

It also answers the second Phase-0 question: Dev_diag is n=425 and one speaker
(55c1784a) carries 14.1% of all errors. If our gates are decided by 1-CER-point
differences, we need (a) the bootstrap CI on a difference that size, and (b) the
leave-one-speaker-out sensitivity, or we will ship on noise or on one talker.

INPUT
-----
The `*.summary.json` written by `error_decomposition.py --out-summary`. It carries
per-utterance `ref_chars`, `errors`, `duration`, `speaker`, `etiology`, `char_ops`
with the reference/hypothesis TEXT stripped (licensed SAP data), which is exactly the
numeric substrate this analysis needs -- so this runs locally, with no pod and no data.

SELF-GATE
---------
Refuses to print anything until it reconstructs the file's own official aggregate --
CER = sum(per-utterance clipped errors) / sum(ref_chars) -- to 1e-12 from the
per-utterance records. If the stored records do not re-derive the number the official
scorer produced, the records are not what was scored and no analysis on them is valid.

Wraps, never modifies, the scoring code (house rule).

USAGE
    python3 scripts/analyze_rate_and_speakers.py experiments/exp_s0_s1_probe/decomp_Dev_diag.fp32_shipped.summary.json
    python3 scripts/analyze_rate_and_speakers.py BASE.summary.json --compare ARM.summary.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from typing import Dict, List, Optional, Sequence, Tuple

GATE_TOL = 1e-12


# ------------------------------------------------------------------ loading
def load_run(path: str) -> Tuple[List[dict], dict]:
    """Load a summary.json; return (per_utterance, totals). Fails loudly."""
    with open(path) as fh:
        doc = json.load(fh)
    for key in ("per_utterance", "totals"):
        if key not in doc:
            sys.exit(f"FATAL: {path} has no '{key}' -- not an error_decomposition summary")
    utts = doc["per_utterance"]
    for u in utts:
        # duration is stored as the manifest string; rate is undefined without it
        u["_dur"] = float(u["duration"]) if u.get("duration") not in (None, "") else float("nan")
        u["_rate"] = (u["ref_chars"] / u["_dur"]) if u["_dur"] > 0 else float("nan")
    return utts, doc["totals"]


def gate_reconstruct(utts: Sequence[dict], totals: dict, label: str) -> None:
    """Reproduce the official micro CER from the per-utterance records, or die."""
    ref_chars = sum(u["ref_chars"] for u in utts)
    errors = sum(u["errors"] for u in utts)
    if ref_chars <= 0:
        sys.exit(f"GATE FAIL [{label}]: zero reference characters")
    cer = errors / ref_chars
    delta = abs(cer - totals["cer"])
    if delta > GATE_TOL:
        sys.exit(
            f"GATE FAIL [{label}]: reconstructed CER {cer:.12f} != stored {totals['cer']:.12f} "
            f"(delta {delta:.3e}). The per-utterance records are not what was scored."
        )
    if len(utts) != totals["n_utts"]:
        sys.exit(f"GATE FAIL [{label}]: {len(utts)} records vs totals.n_utts {totals['n_utts']}")
    print(f"[gate] {label}: reconstructed CER {cer:.6%} from {len(utts)} records "
          f"(delta {delta:.1e}) -- OK")


# ------------------------------------------------------------------ metrics
def micro(utts: Sequence[dict]) -> Tuple[float, float, float]:
    """(cer, errors, ref_chars) -- char-weighted micro, the metric's own currency."""
    rc = sum(u["ref_chars"] for u in utts)
    er = sum(u["errors"] for u in utts)
    return (er / rc if rc else float("nan")), er, rc


def cer_points(utts: Sequence[dict], total_ref_chars: float) -> float:
    """CER points this slice contributes to the whole-set CER (additive across slices)."""
    return 100.0 * sum(u["errors"] for u in utts) / total_ref_chars


def op_counts(utts: Sequence[dict]) -> Dict[str, int]:
    out = {"S": 0, "D": 0, "I": 0}
    for u in utts:
        for k in out:
            out[k] += u.get("char_ops", {}).get(k, 0)
    return out


def quartile_edges(values: Sequence[float]) -> Tuple[float, float, float]:
    v = sorted(x for x in values if not math.isnan(x))
    if len(v) < 4:
        sys.exit("FATAL: fewer than 4 usable durations -- cannot form rate quartiles")

    def q(p: float) -> float:
        idx = p * (len(v) - 1)
        lo, hi = int(math.floor(idx)), int(math.ceil(idx))
        return v[lo] + (v[hi] - v[lo]) * (idx - lo)

    return q(0.25), q(0.50), q(0.75)


def assign_quartiles(utts: Sequence[dict], edges: Tuple[float, float, float]) -> Dict[str, List[dict]]:
    """Q1 = SLOWEST (lowest chars/s). Membership comes from ONE run so an A/B compares
    the same utterances in the same bucket."""
    q1, q2, q3 = edges
    buckets: Dict[str, List[dict]] = {"Q1_slowest": [], "Q2": [], "Q3": [], "Q4_fastest": [], "no_duration": []}
    for u in utts:
        r = u["_rate"]
        if math.isnan(r):
            buckets["no_duration"].append(u)
        elif r <= q1:
            buckets["Q1_slowest"].append(u)
        elif r <= q2:
            buckets["Q2"].append(u)
        elif r <= q3:
            buckets["Q3"].append(u)
        else:
            buckets["Q4_fastest"].append(u)
    return buckets


# ------------------------------------------------------------------ bootstrap
def bootstrap_cer(utts: Sequence[dict], n_boot: int, seed: int) -> Tuple[float, float]:
    """95% CI on micro CER by resampling UTTERANCES with replacement.

    Utterance-level (not character-level) because errors are correlated within an
    utterance -- a whole-word deletion run is one event, not 5 independent char events.
    """
    rng = random.Random(seed)
    n = len(utts)
    er = [u["errors"] for u in utts]
    rc = [u["ref_chars"] for u in utts]
    draws = []
    for _ in range(n_boot):
        se = sr = 0.0
        for _ in range(n):
            i = rng.randrange(n)
            se += er[i]
            sr += rc[i]
        draws.append(se / sr if sr else float("nan"))
    draws.sort()
    return draws[int(0.025 * n_boot)], draws[int(0.975 * n_boot) - 1]


def bootstrap_delta(base: Sequence[dict], arm: Sequence[dict], n_boot: int, seed: int) -> Tuple[float, float, float]:
    """PAIRED bootstrap on (arm CER - base CER): resample utterance INDICES once and
    apply the same draw to both runs. Unpaired CIs on two runs over the identical
    utterance set overstate the uncertainty of their difference."""
    rng = random.Random(seed)
    n = len(base)
    draws = []
    for _ in range(n_boot):
        be = br = ae = ar = 0.0
        for _ in range(n):
            i = rng.randrange(n)
            be += base[i]["errors"]
            br += base[i]["ref_chars"]
            ae += arm[i]["errors"]
            ar += arm[i]["ref_chars"]
        draws.append((ae / ar) - (be / br))
    draws.sort()
    point = (sum(u["errors"] for u in arm) / sum(u["ref_chars"] for u in arm)
             - sum(u["errors"] for u in base) / sum(u["ref_chars"] for u in base))
    return point, draws[int(0.025 * n_boot)], draws[int(0.975 * n_boot) - 1]


# ------------------------------------------------------------------ reports
def report_rate(utts: Sequence[dict], edges, total_rc: float, n_boot: int, seed: int) -> None:
    buckets = assign_quartiles(utts, edges)
    print(f"\n=== CER by SPEAKING RATE quartile (ref chars / s; Q1 = slowest) ===")
    print(f"    edges: q25={edges[0]:.2f}  q50={edges[1]:.2f}  q75={edges[2]:.2f} chars/s")
    print(f"{'bucket':<12} {'n':>4} {'rate':>12} {'CER%':>7} {'95% CI':>16} {'pts':>6} "
          f"{'ref%':>6} {'err%':>6} {'delD':>6} {'empty':>6}")
    total_err = sum(u["errors"] for u in utts)
    for name, grp in buckets.items():
        if not grp:
            continue
        cer, err, rc = micro(grp)
        lo, hi = bootstrap_cer(grp, n_boot, seed)
        rates = [u["_rate"] for u in grp if not math.isnan(u["_rate"])]
        rng_s = f"{min(rates):.1f}-{max(rates):.1f}" if rates else "n/a"
        ops = op_counts(grp)
        empt = sum(1 for u in grp if u.get("hyp_empty"))
        # ref% vs err%: a bucket can be very HARD (high CER) and still hold little error
        # MASS. Difficulty and mass are different statistics -- confusing them is exactly
        # what produced the retracted "51% of error mass is 13+ word utterances" claim.
        print(f"{name:<12} {len(grp):>4} {rng_s:>12} {100*cer:>7.2f} "
              f"[{100*lo:>5.2f},{100*hi:>5.2f}] {cer_points(grp, total_rc):>6.2f} "
              f"{100*rc/total_rc:>6.1f} {100*err/total_err:>6.1f} {ops['D']:>6} {empt:>6}")


def report_speakers(utts: Sequence[dict], total_rc: float, top: int) -> None:
    by_spk: Dict[str, List[dict]] = {}
    for u in utts:
        by_spk.setdefault(u.get("speaker", "?"), []).append(u)
    total_er = sum(u["errors"] for u in utts)
    rows = []
    for spk, grp in by_spk.items():
        er = sum(u["errors"] for u in grp)
        rc = sum(u["ref_chars"] for u in grp)
        loo_rc, loo_er = total_rc - rc, total_er - er
        rows.append({
            "spk": spk, "n": len(grp),
            "share_err": er / total_er if total_er else 0.0,
            "cer": er / rc if rc else float("nan"),
            "loo_cer": loo_er / loo_rc if loo_rc else float("nan"),
            "empties": sum(1 for u in grp if u.get("hyp_empty")),
            "etiology": grp[0].get("etiology", "?"),
            "rate": sum(u["_rate"] for u in grp if not math.isnan(u["_rate"])) / max(1, sum(1 for u in grp if not math.isnan(u["_rate"]))),
        })
    rows.sort(key=lambda r: -r["share_err"])
    whole = total_er / total_rc
    print(f"\n=== SPEAKER concentration + leave-one-speaker-out (whole-set CER {100*whole:.2f}%) ===")
    print(f"{'speaker':<14} {'etiol':<10} {'n':>4} {'err%':>6} {'CER%':>7} {'LOO CER%':>9} {'shift':>7} {'rate':>6} {'empty':>6}")
    for r in rows[:top]:
        print(f"{r['spk'][:12]:<14} {str(r['etiology'])[:9]:<10} {r['n']:>4} "
              f"{100*r['share_err']:>6.1f} {100*r['cer']:>7.2f} {100*r['loo_cer']:>9.2f} "
              f"{100*(r['loo_cer']-whole):>+7.2f} {r['rate']:>6.2f} {r['empties']:>6}")
    print(f"    ({len(rows)} speakers total; 'shift' = what whole-set CER becomes if this speaker is dropped)")


def report_compare(base_u: List[dict], arm_u: List[dict], edges, n_boot: int, seed: int,
                   base_name: str, arm_name: str) -> None:
    bmap = {u["utt"]: u for u in base_u}
    amap = {u["utt"]: u for u in arm_u}
    common = [k for k in bmap if k in amap]
    if len(common) != len(bmap) or len(common) != len(amap):
        sys.exit(f"FATAL: utterance sets differ (base {len(bmap)}, arm {len(amap)}, common {len(common)})")
    b = [bmap[k] for k in common]
    a = [amap[k] for k in common]
    point, lo, hi = bootstrap_delta(b, a, n_boot, seed)
    print(f"\n=== COMPARE  {arm_name}  vs  {base_name}  (paired bootstrap, B={n_boot}) ===")
    print(f"    delta CER = {100*point:+.3f} pts   95% CI [{100*lo:+.3f}, {100*hi:+.3f}]"
          f"   {'SIGNIFICANT' if (lo > 0) == (hi > 0) else 'NOT SIGNIFICANT (CI spans 0)'}")
    bo, ao = op_counts(b), op_counts(a)
    print(f"    char ops  D {bo['D']:>6} -> {ao['D']:>6} ({ao['D']-bo['D']:+d})   "
          f"S {bo['S']:>5} -> {ao['S']:>5} ({ao['S']-bo['S']:+d})   "
          f"I {bo['I']:>5} -> {ao['I']:>5} ({ao['I']-bo['I']:+d})")
    print(f"    empties   {sum(1 for u in b if u.get('hyp_empty')):>3} -> {sum(1 for u in a if u.get('hyp_empty')):>3}")
    # the gate that matters: did it move the SLOW quartile, or just the average?
    bq = assign_quartiles(b, edges)
    aq = assign_quartiles(a, edges)
    print(f"\n    per-rate-quartile delta (the on-target test):")
    print(f"    {'bucket':<12} {'base CER%':>10} {'arm CER%':>9} {'delta':>8} {'95% CI':>18}")
    for name in ("Q1_slowest", "Q2", "Q3", "Q4_fastest"):
        if not bq[name]:
            continue
        keys = [u["utt"] for u in bq[name]]
        bb = [bmap[k] for k in keys]
        aa = [amap[k] for k in keys]
        p, l, h = bootstrap_delta(bb, aa, n_boot, seed)
        print(f"    {name:<12} {100*micro(bb)[0]:>10.2f} {100*micro(aa)[0]:>9.2f} "
              f"{100*p:>+8.2f} [{100*l:>+7.2f},{100*h:>+7.2f}]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("summary", help="baseline *.summary.json from error_decomposition.py")
    ap.add_argument("--compare", help="second *.summary.json to A/B against the baseline")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260730)
    ap.add_argument("--top-speakers", type=int, default=12)
    args = ap.parse_args()

    base_u, base_t = load_run(args.summary)
    gate_reconstruct(base_u, base_t, args.summary.split("/")[-1])

    _, _, total_rc = micro(base_u)
    lo, hi = bootstrap_cer(base_u, args.n_boot, args.seed)
    print(f"[whole set] CER {100*base_t['cer']:.4f}%  95% CI [{100*lo:.2f}, {100*hi:.2f}]  "
          f"n={len(base_u)}  ref_chars={total_rc:.0f}")
    print(f"            char ops S {base_t['char_ops']['S']} / D {base_t['char_ops']['D']} / I {base_t['char_ops']['I']}")

    edges = quartile_edges([u["_rate"] for u in base_u])
    report_rate(base_u, edges, total_rc, args.n_boot, args.seed)
    report_speakers(base_u, total_rc, args.top_speakers)

    if args.compare:
        arm_u, arm_t = load_run(args.compare)
        gate_reconstruct(arm_u, arm_t, args.compare.split("/")[-1])
        report_compare(base_u, arm_u, edges, args.n_boot, args.seed,
                       args.summary.split("/")[-1], args.compare.split("/")[-1])


if __name__ == "__main__":
    main()
