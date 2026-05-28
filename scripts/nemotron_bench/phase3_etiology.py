#!/usr/bin/env python3
"""Per-etiology CER/WER breakdown of phase3 predictions on Dev_streaming."""
import argparse
import csv
import json
import jiwer


def norm(s: str) -> str:
    s = s.lower().strip()
    for ch in ",.?!:;\"'":
        s = s.replace(ch, "")
    return " ".join(s.split())


def score(ids, hyps, refs1, refs2):
    ws = wd = wi = wr = 0
    cs = cd = ci = cr = 0
    n_empty = 0
    n_skip = 0
    for uid in ids:
        h = norm(hyps[uid])
        r1 = norm(refs1[uid])
        r2 = norm(refs2[uid])
        if not h:
            n_empty += 1
        if not r1 and not r2:
            n_skip += 1
            continue
        # Treat empty ref as "all hyp tokens are insertions". jiwer can't
        # handle empty references; replace with the other ref if empty.
        if not r1:
            r1 = r2
        if not r2:
            r2 = r1
        o1 = jiwer.process_words(r1, h)
        o2 = jiwer.process_words(r2, h)
        if (o1.substitutions + o1.deletions + o1.insertions) <= (
            o2.substitutions + o2.deletions + o2.insertions
        ):
            ws += o1.substitutions; wd += o1.deletions; wi += o1.insertions
            wr += o1.hits + o1.substitutions + o1.deletions
        else:
            ws += o2.substitutions; wd += o2.deletions; wi += o2.insertions
            wr += o2.hits + o2.substitutions + o2.deletions
        c1 = jiwer.process_characters(r1, h)
        c2 = jiwer.process_characters(r2, h)
        if (c1.substitutions + c1.deletions + c1.insertions) <= (
            c2.substitutions + c2.deletions + c2.insertions
        ):
            cs += c1.substitutions; cd += c1.deletions; ci += c1.insertions
            cr += c1.hits + c1.substitutions + c1.deletions
        else:
            cs += c2.substitutions; cd += c2.deletions; ci += c2.insertions
            cr += c2.hits + c2.substitutions + c2.deletions
    wer = (ws + wd + wi) / max(1, wr)
    cer = (cs + cd + ci) / max(1, cr)
    return len(ids), n_empty, cer, wer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyp-csv", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    refs1, refs2, etio = {}, {}, {}
    with open(args.manifest) as f:
        for r in csv.DictReader(f):
            refs1[r["id"]] = r.get("norm_text_with_disfluency", r.get("text", ""))
            refs2[r["id"]] = r.get("norm_text_without_disfluency", r.get("text", ""))
            etio[r["id"]] = r.get("etiology", "unknown")
    hyps = {}
    with open(args.hyp_csv) as f:
        for r in csv.DictReader(f):
            hyps[r["id"]] = r["raw_hypos"]

    groups = {}
    for uid in hyps:
        e = etio.get(uid, "unknown")
        groups.setdefault(e, []).append(uid)

    rows = []
    print(f"{'Etiology':<20s} {'N':>4s} {'Empty':>6s} {'CER%':>7s} {'WER%':>7s}")
    print("-" * 50)
    for et in sorted(groups):
        n, ne, cer, wer = score(groups[et], hyps, refs1, refs2)
        rows.append({"etiology": et, "n": n, "n_empty": ne, "cer": cer, "wer": wer})
        print(f"{et:<20s} {n:>4d} {ne:>6d} {cer*100:>7.2f} {wer*100:>7.2f}")
    all_ids = list(hyps.keys())
    n, ne, cer, wer = score(all_ids, hyps, refs1, refs2)
    print("-" * 50)
    print(f"{'TOTAL':<20s} {n:>4d} {ne:>6d} {cer*100:>7.2f} {wer*100:>7.2f}")
    summary = {"per_etiology": rows, "total": {"n": n, "n_empty": ne, "cer": cer, "wer": wer}}

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
