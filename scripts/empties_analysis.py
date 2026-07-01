#!/usr/bin/env python3
"""Empties decomposition for a Track-2 local_decode hyp csv, using the SAME scoring as
streaming_cer_bootstrap.py (utterance-averaged CER, min-over-refs, clip@1, official normalizer).

Each empty hyp contributes exactly 1.0 to the utt-averaged CER, so:
  total_cer = (sum nonempty utt-CERs + n_empty * 1.0) / N
We report how much of the CER is "model emitted nothing" vs "model emitted wrong words",
overall and per etiology, plus a list of empty utterance ids (so a fallback can be tested
on exactly those). This is a diagnostic, not a submission gate.
"""
import argparse, csv, json, re, sys
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
        prev = dp[0]; dp[0] = i
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-csv", required=True)
    ap.add_argument("--hyp-csv", required=True)
    ap.add_argument("--utils-dir", default="/workspace/sapc-nemotron/utils")
    ap.add_argument("--out-json")
    ap.add_argument("--empty-ids-out")
    a = ap.parse_args()

    norm = load_normalizer(a.utils_dir)
    manifest = {row["id"]: row for row in csv.DictReader(open(a.manifest_csv, newline=""))}
    hyps = list(csv.DictReader(open(a.hyp_csv, newline="")))

    N = 0
    sum_cer = 0.0
    n_empty = 0
    sum_cer_nonempty = 0.0
    by_eti = defaultdict(lambda: {"n": 0, "empty": 0, "sum_cer": 0.0, "sum_cer_nonempty": 0.0})
    empty_ids = []
    for h in hyps:
        ref_row = manifest.get(h["id"])
        if not ref_row:
            continue
        refs = [norm(r) for r in refs_for(ref_row)]
        refs = [r for r in refs if r]
        if not refs:
            continue
        raw = h.get("raw_hypos", "")
        is_empty = int(not raw.strip())
        c = min(cer(norm(raw), r) for r in refs)
        eti = ref_row.get("etiology") or "UNKNOWN"
        N += 1
        sum_cer += c
        be = by_eti[eti]
        be["n"] += 1
        be["sum_cer"] += c
        if is_empty:
            n_empty += 1
            be["empty"] += 1
            empty_ids.append(h["id"])
        else:
            sum_cer_nonempty += c
            be["sum_cer_nonempty"] += c

    n_ne = N - n_empty
    total = 100.0 * sum_cer / max(1, N)
    empty_pts = 100.0 * n_empty / max(1, N)               # exact CER points from empties
    nonempty_cer = 100.0 * sum_cer_nonempty / max(1, n_ne)  # CER among utts that DID emit
    # what total CER becomes if empties were instead at non-empty-average quality:
    if_empties_avg = 100.0 * (sum_cer_nonempty + n_empty * (sum_cer_nonempty / max(1, n_ne))) / max(1, N)
    if_empties_zero = 100.0 * sum_cer_nonempty / max(1, N)  # if empties were perfect (lower bound)

    out = {
        "n": N, "n_empty": n_empty, "empty_frac_pct": empty_pts,
        "total_cer_pct": round(total, 3),
        "empty_contribution_pts": round(empty_pts, 3),
        "nonempty_cer_pct": round(nonempty_cer, 3),
        "total_if_empties_at_nonempty_avg_pct": round(if_empties_avg, 3),
        "total_if_empties_perfect_pct": round(if_empties_zero, 3),
        "by_etiology": {
            k: {"n": v["n"], "empty": v["empty"],
                "empty_frac_pct": round(100.0 * v["empty"] / max(1, v["n"]), 2),
                "cer_pct": round(100.0 * v["sum_cer"] / max(1, v["n"]), 2),
                "nonempty_cer_pct": round(100.0 * v["sum_cer_nonempty"] / max(1, v["n"] - v["empty"]), 2)}
            for k, v in sorted(by_eti.items())
        },
    }
    print(json.dumps(out, indent=2))
    print(f"\nSUMMARY: total={total:.2f}%  empties={n_empty}/{N} ({empty_pts:.2f} pts)  "
          f"nonempty={nonempty_cer:.2f}%  | fixing empties->avg gives {if_empties_avg:.2f}%, "
          f"->perfect gives {if_empties_zero:.2f}%")
    if a.out_json:
        json.dump(out, open(a.out_json, "w"), indent=2, sort_keys=True)
    if a.empty_ids_out:
        with open(a.empty_ids_out, "w") as f:
            f.write("\n".join(empty_ids) + ("\n" if empty_ids else ""))
    print("EMPTIES_ANALYSIS_DONE")


if __name__ == "__main__":
    main()
