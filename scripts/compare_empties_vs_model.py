#!/usr/bin/env python3
"""Cross-model check: how does ANOTHER model do on the utts THIS model emptied on?

Given the parakeet streaming-empty ids and a second model's hyp csv (e.g. the fine-tuned
zipformer's Dev_diag decode), report on exactly those utts:
  * how many the other model transcribes (non-empty)
  * the other model's CER on them (min-over-refs, clip@1, official normalizer)
so we can tell whether the empties are PARAKEET-SPECIFIC (other model handles them =>
model choice / normalize='NA' is the cause) or DATA-BOUND (other model also fails =>
genuinely hard, no model helps).

All double-quotes on purpose (safe inside a single-quoted ssh arg).

    python3 scripts/compare_empties_vs_model.py \
      --manifest /workspace/SAPC2/manifest/Dev_diag.csv \
      --empty-ids /workspace/parakeet_ft/error_analysis/empty_ids.txt \
      --other-hyp-csv /workspace/finetune/eval/expA_beam4/diag_zf_beam4.csv \
      --other-name zipformer_beam4 --utils-dir /workspace/SAPC-template/utils
"""
import argparse
import csv
import json
import re
import sys


def load_normalizer(utils_dir):
    if utils_dir:
        sys.path.insert(0, utils_dir)
    try:
        from normalizer.text_normalizer_hf import EnglishTextNormalizer
        return EnglishTextNormalizer().norm, True
    except Exception as exc:
        print("[WARN] official normalizer unavailable (%s); fallback" % exc, file=sys.stderr)
    def fb(t):
        t = (t or "").lower()
        t = re.sub(r"[^a-z0-9\s]", " ", t)
        return re.sub(r"\s+", " ", t).strip()
    return fb, False


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
    out = []
    for k in ("norm_text_with_disfluency", "norm_text_without_disfluency", "text"):
        v = (row.get(k) or "").strip()
        if v and v not in out:
            out.append(v)
    return out


def load_hyp(path):
    d = {}
    for r in csv.DictReader(open(path, newline="")):
        idk = [k for k in r if k.lower() in ("id", "utt", "utt_id")][0]
        hk = [k for k in r if "hyp" in k.lower() or k.lower() in ("text", "raw_hypos", "pred")][0]
        d[r[idk]] = (r[hk] or "").strip()
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--empty-ids", required=True)
    ap.add_argument("--other-hyp-csv", required=True)
    ap.add_argument("--other-name", default="other_model")
    ap.add_argument("--utils-dir", default="/workspace/SAPC-template/utils")
    ap.add_argument("--out-json")
    a = ap.parse_args()

    norm, official = load_normalizer(a.utils_dir)
    manifest = {r["id"]: r for r in csv.DictReader(open(a.manifest, newline=""))}
    other = load_hyp(a.other_hyp_csv)
    ids = [l.strip() for l in open(a.empty_ids) if l.strip()]

    rows = []
    nonempty = 0
    sum_cer = 0.0
    scored = 0
    good = 0  # CER < 0.5
    for i in ids:
        row = manifest.get(i)
        if not row or i not in other:
            continue
        refs = [norm(r) for r in refs_for(row)]
        refs = [r for r in refs if r]
        if not refs:
            continue
        h = other[i]
        ne = bool(h)
        nonempty += ne
        c = min(cer(norm(h), r) for r in refs)
        sum_cer += c; scored += 1
        good += (c < 0.5)
        rows.append({"id": i, "nonempty": ne, "cer": round(c, 3),
                     "other_hyp": h, "ref": refs[0]})

    mean_cer = 100.0 * sum_cer / max(1, scored)
    out = {
        "other_name": a.other_name, "normalizer_official": official,
        "n_parakeet_empty": scored,
        "other_nonempty": nonempty,
        "other_nonempty_pct": round(100.0 * nonempty / max(1, scored), 1),
        "other_mean_cer_pct_on_these": round(mean_cer, 2),
        "other_good_lt50cer": good,
        "other_good_pct": round(100.0 * good / max(1, scored), 1),
        "per_utt": sorted(rows, key=lambda r: r["cer"]),
    }
    print(json.dumps({k: v for k, v in out.items() if k != "per_utt"}, indent=2))
    print("\n==== %s ON THE %d PARAKEET-EMPTY UTTS ====" % (a.other_name, scored))
    print("non-empty: %d/%d (%.1f%%) | mean CER on them: %.2f%% | good(<50%%CER): %d (%.1f%%)"
          % (nonempty, scored, out["other_nonempty_pct"], mean_cer, good, out["other_good_pct"]))
    print("(parakeet scores 100%% CER on all %d by definition — every one is a full loss for parakeet)" % scored)
    print("--- best %s recoveries on parakeet-empty utts ---" % a.other_name)
    for r in out["per_utt"][:20]:
        print("  cer=%.2f  hyp=%r  ref=%r" % (r["cer"], r["other_hyp"][:55], r["ref"][:55]))
    if a.out_json:
        json.dump(out, open(a.out_json, "w"), indent=2)
        print("[wrote] %s" % a.out_json)
    print("COMPARE_EMPTIES_DONE")


if __name__ == "__main__":
    main()
