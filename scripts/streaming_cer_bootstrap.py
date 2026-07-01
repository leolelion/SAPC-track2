#!/usr/bin/env python3
"""CER summary + speaker-block bootstrap for Track 2 local_decode outputs."""

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


def bootstrap_speakers(values_by_speaker, iters, seed):
    speakers = sorted(values_by_speaker)
    if not speakers:
        return {"iters": 0, "p2.5": None, "p50": None, "p97.5": None}
    rng = random.Random(seed)
    means = []
    for _ in range(iters):
        total = 0.0
        count = 0
        for _ in speakers:
            vals = values_by_speaker[rng.choice(speakers)]
            total += sum(vals)
            count += len(vals)
        means.append(total / max(1, count))
    return {"iters": iters, "p2.5": pct(means, 0.025), "p50": pct(means, 0.5), "p97.5": pct(means, 0.975)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-csv", required=True)
    ap.add_argument("--hyp-csv", required=True)
    ap.add_argument("--utils-dir", default="/workspace/sapc-nemotron/utils")
    ap.add_argument("--bootstrap-iters", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--out-json")
    args = ap.parse_args()

    norm = load_normalizer(args.utils_dir)
    manifest = {row["id"]: row for row in csv.DictReader(open(args.manifest_csv, newline=""))}
    hyps = list(csv.DictReader(open(args.hyp_csv, newline="")))

    rows = []
    by_speaker = defaultdict(list)
    by_etiology = defaultdict(list)
    missing_ref = 0
    for hyp_row in hyps:
        uid = hyp_row["id"]
        ref_row = manifest.get(uid)
        if not ref_row:
            continue
        refs = [norm(ref) for ref in refs_for(ref_row)]
        refs = [ref for ref in refs if ref]
        if not refs:
            missing_ref += 1
            continue
        raw = hyp_row.get("raw_hypos", "")
        hyp_norm = norm(raw)
        utt_cer = min(cer(hyp_norm, ref) for ref in refs)
        speaker = ref_row.get("speaker") or uid.split("_", 1)[0]
        etiology = ref_row.get("etiology") or "UNKNOWN"
        rec = {
            "id": uid,
            "speaker": speaker,
            "etiology": etiology,
            "cer": utt_cer,
            "empty": int(not raw.strip()),
            "wh_prefix": int(raw.strip().startswith("Wh ")),
        }
        rows.append(rec)
        by_speaker[speaker].append(utt_cer)
        by_etiology[etiology].append(utt_cer)

    mean_cer = sum(r["cer"] for r in rows) / max(1, len(rows))
    result = {
        "manifest_csv": args.manifest_csv,
        "hyp_csv": args.hyp_csv,
        "n": len(rows),
        "speakers": len(by_speaker),
        "missing_ref": missing_ref,
        "mean_cer": mean_cer,
        "mean_cer_pct": 100.0 * mean_cer,
        "empty": sum(r["empty"] for r in rows),
        "wh_prefix": sum(r["wh_prefix"] for r in rows),
        "speaker_block_bootstrap": bootstrap_speakers(by_speaker, args.bootstrap_iters, args.seed),
        "by_etiology": {
            key: {"n": len(vals), "mean_cer_pct": 100.0 * sum(vals) / max(1, len(vals))}
            for key, vals in sorted(by_etiology.items())
        },
    }

    ci = result["speaker_block_bootstrap"]
    print(
        "CER={:.2f}% n={} speakers={} empty={} wh={} speakerCI95=[{:.2f},{:.2f}]%".format(
            result["mean_cer_pct"],
            result["n"],
            result["speakers"],
            result["empty"],
            result["wh_prefix"],
            100.0 * (ci["p2.5"] or 0.0),
            100.0 * (ci["p97.5"] or 0.0),
        )
    )
    for eti, vals in result["by_etiology"].items():
        print(f"  {eti:20s} n={vals['n']:4d} CER={vals['mean_cer_pct']:6.2f}%")

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
