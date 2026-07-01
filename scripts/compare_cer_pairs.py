#!/usr/bin/env python3
"""Paired CER deltas for two Track 2 local_decode hypothesis CSVs."""

import argparse
import csv
import json
import random
import re
import sys
from collections import defaultdict


def load_normalizer(utils_dir):
    if utils_dir:
        sys.path.insert(0, utils_dir)
    try:
        from normalizer.text_normalizer_hf import EnglishTextNormalizer

        return EnglishTextNormalizer().norm
    except Exception as exc:
        print(f"[WARN] official normalizer unavailable ({exc}); using fallback", file=sys.stderr)

    def fallback(text):
        text = (text or "").lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    return fallback


def cer(hyp, ref):
    hyp, ref = list(hyp), list(ref)
    if not ref:
        return 0.0 if not hyp else 1.0
    dp = list(range(len(ref) + 1))
    for i in range(1, len(hyp) + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, len(ref) + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (hyp[i - 1] != ref[j - 1]))
            prev = cur
    return min(1.0, dp[len(ref)] / len(ref))


def refs_for(row):
    refs = []
    for key in ("norm_text_with_disfluency", "norm_text_without_disfluency", "text", "raw_trans"):
        val = (row.get(key) or "").strip()
        if val and val not in refs:
            refs.append(val)
    return refs


def pct(values, q):
    if not values:
        return None
    values = sorted(values)
    idx = int(round((len(values) - 1) * q))
    return values[max(0, min(idx, len(values) - 1))]


def bootstrap_speaker_delta(rows, iters, seed):
    by_speaker = defaultdict(list)
    for row in rows:
        by_speaker[row["speaker"]].append(row["delta"])
    speakers = sorted(by_speaker)
    if not speakers:
        return {"iters": 0, "p2.5": None, "p50": None, "p97.5": None}
    rng = random.Random(seed)
    means = []
    for _ in range(iters):
        total = 0.0
        count = 0
        for _ in speakers:
            vals = by_speaker[rng.choice(speakers)]
            total += sum(vals)
            count += len(vals)
        means.append(total / max(1, count))
    return {"iters": iters, "p2.5": pct(means, 0.025), "p50": pct(means, 0.5), "p97.5": pct(means, 0.975)}


def load_hyps(path):
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["id"]] = row.get("raw_hypos", "")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-csv", required=True)
    ap.add_argument("--base-hyp-csv", required=True)
    ap.add_argument("--candidate-hyp-csv", required=True)
    ap.add_argument("--utils-dir", default="/workspace/sapc-nemotron/utils")
    ap.add_argument("--bootstrap-iters", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--out-json")
    args = ap.parse_args()

    norm = load_normalizer(args.utils_dir)
    base = load_hyps(args.base_hyp_csv)
    cand = load_hyps(args.candidate_hyp_csv)

    rows = []
    missing = {"base": 0, "candidate": 0, "refs": 0}
    with open(args.manifest_csv, newline="", encoding="utf-8") as f:
        for ref_row in csv.DictReader(f):
            uid = ref_row["id"]
            if uid not in base:
                missing["base"] += 1
                continue
            if uid not in cand:
                missing["candidate"] += 1
                continue
            refs = [norm(ref) for ref in refs_for(ref_row)]
            refs = [ref for ref in refs if ref]
            if not refs:
                missing["refs"] += 1
                continue
            base_raw = base[uid]
            cand_raw = cand[uid]
            base_cer = min(cer(norm(base_raw), ref) for ref in refs)
            cand_cer = min(cer(norm(cand_raw), ref) for ref in refs)
            speaker = ref_row.get("speaker") or uid.split("_", 1)[0]
            etiology = ref_row.get("etiology") or "UNKNOWN"
            rows.append(
                {
                    "id": uid,
                    "speaker": speaker,
                    "etiology": etiology,
                    "base_cer": base_cer,
                    "candidate_cer": cand_cer,
                    "delta": cand_cer - base_cer,
                    "base_empty": int(not base_raw.strip()),
                    "candidate_empty": int(not cand_raw.strip()),
                    "base_wh_prefix": int(base_raw.strip().startswith("Wh ")),
                    "candidate_wh_prefix": int(cand_raw.strip().startswith("Wh ")),
                }
            )

    n = max(1, len(rows))
    by_etiology = defaultdict(list)
    for row in rows:
        by_etiology[row["etiology"]].append(row)
    base_mean = sum(row["base_cer"] for row in rows) / n
    cand_mean = sum(row["candidate_cer"] for row in rows) / n
    delta = cand_mean - base_mean
    ci = bootstrap_speaker_delta(rows, args.bootstrap_iters, args.seed)
    result = {
        "manifest_csv": args.manifest_csv,
        "base_hyp_csv": args.base_hyp_csv,
        "candidate_hyp_csv": args.candidate_hyp_csv,
        "n": len(rows),
        "speakers": len({row["speaker"] for row in rows}),
        "missing": missing,
        "base_mean_cer": base_mean,
        "base_mean_cer_pct": 100.0 * base_mean,
        "candidate_mean_cer": cand_mean,
        "candidate_mean_cer_pct": 100.0 * cand_mean,
        "delta_cer": delta,
        "delta_cer_pct": 100.0 * delta,
        "base_empty": sum(row["base_empty"] for row in rows),
        "candidate_empty": sum(row["candidate_empty"] for row in rows),
        "changed_empty": sum(row["base_empty"] != row["candidate_empty"] for row in rows),
        "base_wh_prefix": sum(row["base_wh_prefix"] for row in rows),
        "candidate_wh_prefix": sum(row["candidate_wh_prefix"] for row in rows),
        "speaker_block_delta_bootstrap": ci,
        "by_etiology": {},
    }
    for eti, vals in sorted(by_etiology.items()):
        m = max(1, len(vals))
        result["by_etiology"][eti] = {
            "n": len(vals),
            "base_mean_cer_pct": 100.0 * sum(row["base_cer"] for row in vals) / m,
            "candidate_mean_cer_pct": 100.0 * sum(row["candidate_cer"] for row in vals) / m,
            "delta_cer_pct": 100.0 * sum(row["delta"] for row in vals) / m,
            "base_empty": sum(row["base_empty"] for row in vals),
            "candidate_empty": sum(row["candidate_empty"] for row in vals),
        }

    print(
        "BASE={:.2f}% CAND={:.2f}% DELTA={:+.2f}% n={} speakers={} "
        "empty {}->{} changed={} wh {}->{} deltaCI95=[{:+.2f},{:+.2f}]%".format(
            result["base_mean_cer_pct"],
            result["candidate_mean_cer_pct"],
            result["delta_cer_pct"],
            result["n"],
            result["speakers"],
            result["base_empty"],
            result["candidate_empty"],
            result["changed_empty"],
            result["base_wh_prefix"],
            result["candidate_wh_prefix"],
            100.0 * (ci["p2.5"] or 0.0),
            100.0 * (ci["p97.5"] or 0.0),
        )
    )
    for eti, vals in result["by_etiology"].items():
        print(
            f"  {eti:20s} n={vals['n']:4d} "
            f"base={vals['base_mean_cer_pct']:6.2f}% "
            f"cand={vals['candidate_mean_cer_pct']:6.2f}% "
            f"delta={vals['delta_cer_pct']:+6.2f}% "
            f"empty={vals['base_empty']}->{vals['candidate_empty']}"
        )

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
