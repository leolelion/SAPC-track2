#!/usr/bin/env python3
"""Sanity check: compute Dev_10k CER/WER against ref1 only, ref2 only,
and min(ref1, ref2). Uses jiwer (NOT sclite) so absolute numbers will
differ from the sclite-scored 21.59 / 27.46. The *relative* gap between
single-ref and dual-ref tells us whether dual-ref was material.

Normalization: minimal (lowercase + punctuation strip) to keep
comparison clean.
"""
import argparse
import csv

import jiwer


def norm(s: str) -> str:
    s = (s or "").lower().strip()
    for ch in ",.?!:;\"'":
        s = s.replace(ch, "")
    return " ".join(s.split())


def score(preds, refs, clip_per_utt=True):
    err_c = ref_c = err_w = ref_w = 0
    for uid in preds:
        h = norm(preds[uid])
        r = norm(refs.get(uid, ""))
        if not r:
            continue
        if not h:
            err_c += len(r); ref_c += len(r)
            err_w += len(r.split()); ref_w += len(r.split())
            continue
        oc = jiwer.process_characters(r, h)
        ec = oc.substitutions + oc.deletions + oc.insertions
        if clip_per_utt:
            ec = min(ec, len(r))
        err_c += ec; ref_c += len(r)
        ow = jiwer.process_words(r, h)
        ew = ow.substitutions + ow.deletions + ow.insertions
        if clip_per_utt:
            ew = min(ew, len(r.split()))
        err_w += ew; ref_w += len(r.split())
    return (err_c / max(1, ref_c), err_w / max(1, ref_w))


def score_min2(preds, refs1, refs2, clip_per_utt=True):
    err_c = ref_c = err_w = ref_w = 0
    for uid in preds:
        h = norm(preds[uid])
        r1 = norm(refs1.get(uid, ""))
        r2 = norm(refs2.get(uid, ""))
        candidates = [r for r in (r1, r2) if r]
        if not candidates:
            continue
        # per-ref CER/WER, then pick the ref with lower edit count per metric
        best_c = best_w = None
        for r in candidates:
            if not h:
                ec = len(r); ew = len(r.split())
            else:
                oc = jiwer.process_characters(r, h)
                ec = oc.substitutions + oc.deletions + oc.insertions
                ow = jiwer.process_words(r, h)
                ew = ow.substitutions + ow.deletions + ow.insertions
            if clip_per_utt:
                ec = min(ec, len(r))
                ew = min(ew, len(r.split()))
            cer_v = ec / max(1, len(r))
            wer_v = ew / max(1, len(r.split()))
            if best_c is None or cer_v < best_c[0]:
                best_c = (cer_v, ec, len(r))
            if best_w is None or wer_v < best_w[0]:
                best_w = (wer_v, ew, len(r.split()))
        err_c += best_c[1]; ref_c += best_c[2]
        err_w += best_w[1]; ref_w += best_w[2]
    return (err_c / max(1, ref_c), err_w / max(1, ref_w))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds-csv", required=True)
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args()

    preds = {}
    with open(args.preds_csv) as f:
        for r in csv.DictReader(f):
            preds[r["id"]] = r.get("raw_hypos", "")
    refs1 = {}
    refs2 = {}
    with open(args.manifest) as f:
        for r in csv.DictReader(f):
            refs1[r["id"]] = r.get("norm_text_with_disfluency", r.get("text", ""))
            refs2[r["id"]] = r.get("norm_text_without_disfluency", r.get("text", ""))

    cer_r1, wer_r1 = score(preds, refs1)
    cer_r2, wer_r2 = score(preds, refs2)
    cer_min, wer_min = score_min2(preds, refs1, refs2)

    print(f"{'Mode':<28} {'CER%':>7} {'WER%':>7}")
    print("-" * 46)
    print(f"{'Single ref (with disfl)':<28} {cer_r1*100:>7.2f} {wer_r1*100:>7.2f}")
    print(f"{'Single ref (without disfl)':<28} {cer_r2*100:>7.2f} {wer_r2*100:>7.2f}")
    print(f"{'Min(both refs)':<28} {cer_min*100:>7.2f} {wer_min*100:>7.2f}")
    print()
    print(f"Δ ref1-only vs min: CER {cer_r1*100 - cer_min*100:+.2f}, WER {wer_r1*100 - wer_min*100:+.2f}")
    print(f"Δ ref2-only vs min: CER {cer_r2*100 - cer_min*100:+.2f}, WER {wer_r2*100 - wer_min*100:+.2f}")


if __name__ == "__main__":
    main()
