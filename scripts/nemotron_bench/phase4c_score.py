#!/usr/bin/env python3
"""Phase 4c — score forced-decode vs empty baseline on the empty subset.

For the 1463 utts that produced empty under normal greedy:
  - CER_empty   = score "" against each ref (current/shipped behavior)
  - CER_forced  = score forced output against ref

Uses min-of-two-refs (SAPC2 official: lower of the two refs per utt).
SAPC2 clips per-utt CER/WER to 100% — we replicate that here.

Output: table to stdout.
"""
import argparse
import csv
from typing import Tuple

import jiwer


def norm(s: str) -> str:
    s = (s or "").lower().strip()
    for ch in ",.?!:;\"'":
        s = s.replace(ch, "")
    return " ".join(s.split())


def char_errors(ref: str, hyp: str) -> Tuple[int, int]:
    """Return (errors, ref_chars). Clipped at ref_chars (CER<=100%)."""
    if not ref:
        return (0, 0)
    if not hyp:
        return (len(ref), len(ref))
    out = jiwer.process_characters(ref, hyp)
    errs = out.substitutions + out.deletions + out.insertions
    return (min(errs, len(ref)), len(ref))


def word_errors(ref: str, hyp: str) -> Tuple[int, int]:
    if not ref:
        return (0, 0)
    if not hyp:
        rw = ref.split()
        return (len(rw), len(rw))
    out = jiwer.process_words(ref, hyp)
    errs = out.substitutions + out.deletions + out.insertions
    rw = ref.split()
    return (min(errs, len(rw)), len(rw))


def score_set(uids, hyps_dict, refs1, refs2):
    """Min-of-two-refs CER/WER with SAPC2 per-utt 100% clip."""
    c_err = c_ref = w_err = w_ref = 0
    per_utt = []
    for uid in uids:
        h = norm(hyps_dict.get(uid, ""))
        r1 = norm(refs1.get(uid, ""))
        r2 = norm(refs2.get(uid, ""))
        candidates = [r for r in (r1, r2) if r]
        if not candidates:
            continue
        c_choices = [char_errors(r, h) for r in candidates]
        w_choices = [word_errors(r, h) for r in candidates]
        # Pick the ref with lower edit count for each metric
        cer_pick = min(c_choices, key=lambda x: x[0] / max(1, x[1]))
        wer_pick = min(w_choices, key=lambda x: x[0] / max(1, x[1]))
        c_err += cer_pick[0]; c_ref += cer_pick[1]
        w_err += wer_pick[0]; w_ref += wer_pick[1]
        per_utt.append({
            "id": uid,
            "ref": candidates[0],
            "hyp": h,
            "cer": cer_pick[0] / max(1, cer_pick[1]),
            "wer": wer_pick[0] / max(1, wer_pick[1]),
        })
    cer = c_err / max(1, c_ref)
    wer = w_err / max(1, w_ref)
    return cer, wer, per_utt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--empties-csv", required=True, help="empties_dev10k.csv (the 1463 utt IDs + refs)")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--forced-csv", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    # Load refs from manifest, restricted to the empty subset
    targets = []
    with open(args.empties_csv) as f:
        for r in csv.DictReader(f):
            targets.append(r["id"])
    target_set = set(targets)

    refs1, refs2 = {}, {}
    with open(args.manifest) as f:
        for r in csv.DictReader(f):
            uid = r["id"]
            if uid not in target_set:
                continue
            refs1[uid] = r.get("norm_text_with_disfluency", r.get("text", ""))
            refs2[uid] = r.get("norm_text_without_disfluency", r.get("text", ""))

    forced = {}
    with open(args.forced_csv) as f:
        for r in csv.DictReader(f):
            forced[r["id"]] = r["raw_hypos"]

    # Score with "empty" output (shipped behavior on these utts)
    empty_hyps = {uid: "" for uid in targets}

    cer_emp, wer_emp, _ = score_set(targets, empty_hyps, refs1, refs2)
    cer_for, wer_for, per_utt = score_set(targets, forced, refs1, refs2)

    # Per-utt CER distribution for forced
    cers = sorted([p["cer"] for p in per_utt])
    p50 = cers[len(cers)//2] if cers else 0
    p25 = cers[len(cers)//4] if cers else 0
    p75 = cers[3*len(cers)//4] if cers else 0
    n_lt100 = sum(1 for c in cers if c < 1.0 - 1e-9)
    n_lt75  = sum(1 for c in cers if c < 0.75)
    n_lt50  = sum(1 for c in cers if c < 0.5)
    n_lt25  = sum(1 for c in cers if c < 0.25)

    # Aggregate impact projection on the FULL Dev_10k (10521 utts at sclite CER 21.59%)
    # If the empty subset CER drops from cer_emp to cer_for under forced,
    # ref-char-weighted impact on the aggregate is:
    #   delta_cer = (subset_ref_chars / total_ref_chars) * (cer_for - cer_emp)
    # We approximate subset_ref_chars from per_utt; total via sclite_total.
    subset_ref_chars = sum(len(p["ref"]) for p in per_utt)
    # All-Dev_10k ref-char count: use manifest
    total_ref_chars = 0
    with open(args.manifest) as f:
        for r in csv.DictReader(f):
            ref = norm(r.get("norm_text_without_disfluency", r.get("text", "")))
            total_ref_chars += len(ref)
    delta = (subset_ref_chars / max(1, total_ref_chars)) * (cer_for - cer_emp)

    md = []
    md.append("# Phase 4c — forced-decode diagnostic results\n")
    md.append(f"Empty subset size: {len(targets)} utts (13.91% of Dev_10k).\n")
    md.append("## CER / WER on the empty subset\n")
    md.append("Scoring: jiwer min-of-two-refs, per-utt CER/WER clipped at 100% (SAPC2 rule).")
    md.append("Note: jiwer ≈ sclite within ±1-2 pp; relative comparison here is what matters.\n")
    md.append("| Mode | CER% | WER% |")
    md.append("|---|---:|---:|")
    md.append(f"| Empty (shipped behavior) | {cer_emp*100:.2f} | {wer_emp*100:.2f} |")
    md.append(f"| Forced (non-blank greedy) | {cer_for*100:.2f} | {wer_for*100:.2f} |")
    md.append(f"| Δ (forced − empty) | {(cer_for-cer_emp)*100:+.2f} | {(wer_for-wer_emp)*100:+.2f} |\n")

    md.append("## Per-utt CER distribution under forced mode (1463 utts)\n")
    md.append(f"- p25: {p25*100:.1f}%   p50: {p50*100:.1f}%   p75: {p75*100:.1f}%")
    md.append(f"- Utts with CER < 100%: {n_lt100} ({n_lt100/len(cers)*100:.1f}%)")
    md.append(f"- Utts with CER <  75%: {n_lt75} ({n_lt75/len(cers)*100:.1f}%)")
    md.append(f"- Utts with CER <  50%: {n_lt50} ({n_lt50/len(cers)*100:.1f}%)")
    md.append(f"- Utts with CER <  25%: {n_lt25} ({n_lt25/len(cers)*100:.1f}%)\n")

    md.append("## Projected impact on full-Dev_10k aggregate CER\n")
    md.append(f"- Subset ref-chars: {subset_ref_chars:,}")
    md.append(f"- Full Dev_10k ref-chars: {total_ref_chars:,}")
    md.append(f"- Subset share of total chars: {subset_ref_chars/max(1,total_ref_chars)*100:.2f}%")
    md.append(f"- Δ on full Dev_10k CER if we shipped forced mode: {delta*100:+.3f} pp\n")

    md.append("## Eyeball: 10 forced outputs vs references\n")
    md.append("| id (prefix) | ref | forced hyp (first 80 chars) | per-utt CER |")
    md.append("|---|---|---|---:|")
    for p in per_utt[:10]:
        hyp_show = (p["hyp"] or "")[:80]
        ref_show = (p["ref"] or "")[:80]
        md.append(f"| `{p['id'][:12]}…` | {ref_show!r} | {hyp_show!r} | {p['cer']*100:.0f}% |")
    md.append("")

    out = args.out_md
    with open(out, "w") as f:
        f.write("\n".join(md))
    print(f"wrote {out}\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
