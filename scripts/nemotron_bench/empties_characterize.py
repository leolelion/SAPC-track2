#!/usr/bin/env python3
"""Step 2 — characterize the Dev_10k empty set against a non-empty control.

Six distributions per the triage plan:
  1. audio duration histogram (empty vs non-empty), bucketed
  2. speech-active duration (mfa_speech_end - mfa_speech_start), summary stats
  3. reference word count, summary stats
  4. per-etiology empty rate (we already have it, recompute for sanity)
  5. per-speaker top-10 by empty count
  6. encoder-step count for empties: floor(audio_duration / 0.56)

Output is a markdown report ready to paste into empties_analysis.md.
"""
import argparse
import csv
import json
import statistics as stats
from collections import Counter, defaultdict


def parse_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def bucket_label(seconds: float) -> str:
    if seconds < 0.5:
        return "<0.5s"
    if seconds < 1.0:
        return "0.5-1s"
    if seconds < 1.5:
        return "1-1.5s"
    if seconds < 2.0:
        return "1.5-2s"
    if seconds < 3.0:
        return "2-3s"
    if seconds < 5.0:
        return "3-5s"
    return ">5s"


BUCKETS = ["<0.5s", "0.5-1s", "1-1.5s", "1.5-2s", "2-3s", "3-5s", ">5s"]


def summarize(values):
    if not values:
        return {}
    vs = sorted(values)
    return {
        "n": len(vs),
        "min": vs[0],
        "p10": vs[int(0.10 * len(vs))],
        "p25": vs[int(0.25 * len(vs))],
        "p50": vs[len(vs) // 2],
        "mean": sum(vs) / len(vs),
        "p75": vs[int(0.75 * len(vs))],
        "p90": vs[int(0.90 * len(vs))],
        "p99": vs[int(0.99 * len(vs))],
        "max": vs[-1],
    }


def fmt_summary(s, units=""):
    if not s:
        return "(empty)"
    return (f"n={s['n']} min={s['min']:.2f}{units} p10={s['p10']:.2f} p25={s['p25']:.2f} "
            f"p50={s['p50']:.2f} mean={s['mean']:.2f} p75={s['p75']:.2f} "
            f"p90={s['p90']:.2f} p99={s['p99']:.2f} max={s['max']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions-csv", required=True)
    ap.add_argument("--manifest-csv", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    # Load hyps
    hyps = {}
    with open(args.predictions_csv) as f:
        for r in csv.DictReader(f):
            hyps[r["id"]] = r.get("raw_hypos", "")

    rows = []
    with open(args.manifest_csv) as f:
        for r in csv.DictReader(f):
            uid = r["id"]
            if uid not in hyps:
                continue
            empty = not (hyps[uid] or "").strip()
            ref = (r.get("norm_text_without_disfluency") or r.get("text") or "").strip()
            mfa_s = parse_float(r.get("mfa_speech_start"))
            mfa_e = parse_float(r.get("mfa_speech_end"))
            speech_active = (mfa_e - mfa_s) if (mfa_s is not None and mfa_e is not None) else None
            rows.append({
                "id": uid,
                "speaker": r.get("speaker", ""),
                "etiology": r.get("etiology", ""),
                "duration": parse_float(r.get("duration")),
                "speech_active": speech_active,
                "ref_word_count": len(ref.split()) if ref else 0,
                "empty": empty,
            })

    empties = [r for r in rows if r["empty"]]
    non_empties = [r for r in rows if not r["empty"]]

    md = []
    md.append("# Step 2 — Empty-prediction characterization (Dev_10k Nemotron int8-static)\n")
    md.append(f"Total: {len(rows)}  |  Empty: {len(empties)} ({len(empties)/max(1,len(rows))*100:.2f}%)  |  Non-empty: {len(non_empties)}\n")

    # ── 1. Audio-duration histogram (empty rate per bucket) ──────────
    bucket_total = Counter()
    bucket_empty = Counter()
    for r in rows:
        if r["duration"] is None:
            continue
        b = bucket_label(r["duration"])
        bucket_total[b] += 1
        if r["empty"]:
            bucket_empty[b] += 1
    md.append("## 1. Audio-duration buckets — empty rate per bucket\n")
    md.append("| Bucket | N total | Empty | Empty rate |")
    md.append("|---|---:|---:|---:|")
    for b in BUCKETS:
        n = bucket_total.get(b, 0)
        e = bucket_empty.get(b, 0)
        rate = e / n if n else 0.0
        md.append(f"| {b} | {n} | {e} | {rate*100:.2f}% |")
    md.append("")

    # ── 2. Speech-active duration (mfa_end - mfa_start) summary ──────
    sa_empty = [r["speech_active"] for r in empties if r["speech_active"] is not None]
    sa_non = [r["speech_active"] for r in non_empties if r["speech_active"] is not None]
    md.append("## 2. Speech-active duration (mfa_speech_end − mfa_speech_start), seconds\n")
    md.append(f"- Empty: {fmt_summary(summarize(sa_empty), 's')}")
    md.append(f"- Non-empty: {fmt_summary(summarize(sa_non), 's')}\n")

    # ── 3. Reference word count ──────────────────────────────────────
    wc_empty = [r["ref_word_count"] for r in empties]
    wc_non = [r["ref_word_count"] for r in non_empties]
    md.append("## 3. Reference word count\n")
    md.append(f"- Empty: {fmt_summary(summarize(wc_empty))}")
    md.append(f"- Non-empty: {fmt_summary(summarize(wc_non))}")
    md.append("")
    md.append("Distribution of empty-set reference word counts (low end):")
    md.append("| Words | Empty count | Empty share |")
    md.append("|---:|---:|---:|")
    wc_counter = Counter(wc_empty)
    for w in sorted(wc_counter.keys())[:15]:
        c = wc_counter[w]
        md.append(f"| {w} | {c} | {c/max(1,len(empties))*100:.2f}% |")
    md.append("")

    # ── 4. Per-etiology empty rate ───────────────────────────────────
    et_total = Counter(); et_empty = Counter()
    for r in rows:
        et_total[r["etiology"]] += 1
        if r["empty"]:
            et_empty[r["etiology"]] += 1
    md.append("## 4. Per-etiology empty rate\n")
    md.append("| Etiology | N | Empty | Empty rate |")
    md.append("|---|---:|---:|---:|")
    for e in sorted(et_total.keys()):
        n = et_total[e]
        ne = et_empty[e]
        md.append(f"| {e} | {n} | {ne} | {ne/max(1,n)*100:.2f}% |")
    md.append("")

    # ── 5. Per-speaker top-10 by empty count ─────────────────────────
    sp_total = Counter(); sp_empty = Counter()
    sp_etio = {}
    for r in rows:
        sp_total[r["speaker"]] += 1
        if r["empty"]:
            sp_empty[r["speaker"]] += 1
        sp_etio[r["speaker"]] = r["etiology"]
    top10 = sorted(sp_empty.items(), key=lambda kv: kv[1], reverse=True)[:10]
    md.append("## 5. Top-10 speakers by empty count\n")
    md.append("| Speaker (prefix) | Etiology | N utts | Empty | Empty rate |")
    md.append("|---|---|---:|---:|---:|")
    for sp, ec in top10:
        n = sp_total[sp]
        md.append(f"| `{sp[:12]}…` | {sp_etio[sp]} | {n} | {ec} | {ec/max(1,n)*100:.2f}% |")
    md.append("")

    # ── 6. Encoder-step count per empty utt ──────────────────────────
    # At [70,6], one encoder step ≈ 560 ms of audio buffered.
    # n_steps_lower_bound = floor(audio_duration / 0.56)
    # n_steps_partial-chunk-or-zero = audio_duration < 0.56  (no full chunk)
    md.append("## 6. Encoder-step count for empties (≈ floor(audio_duration / 0.56))\n")
    step_buckets = Counter()
    no_step = 0
    no_mel = 0
    for r in empties:
        d = r["duration"]
        if d is None:
            continue
        # T = 0 path: total_samples < 400 ≈ 25 ms → no mel frames at all
        if d < 0.025:
            no_mel += 1
            step_buckets["T=0 (<25 ms)"] += 1
        elif d < 0.56:
            no_step += 1
            step_buckets["<1 chunk (25-560 ms)"] += 1
        else:
            n_steps = int(d / 0.56)
            if n_steps < 5:
                step_buckets[f"{n_steps} chunk(s)"] += 1
            elif n_steps < 10:
                step_buckets["5-9 chunks"] += 1
            else:
                step_buckets["10+ chunks"] += 1
    order = ["T=0 (<25 ms)", "<1 chunk (25-560 ms)"] + \
            [f"{i} chunk(s)" for i in [1, 2, 3, 4]] + \
            ["5-9 chunks", "10+ chunks"]
    md.append("| Steps available | Empty count | Empty share |")
    md.append("|---|---:|---:|")
    for k in order:
        c = step_buckets.get(k, 0)
        if c:
            md.append(f"| {k} | {c} | {c/max(1,len(empties))*100:.2f}% |")
    md.append("")
    md.append(f"Counts (raw): T=0 (<25 ms): **{no_mel}**, "
              f"under 1 full chunk (25–560 ms): **{no_step}**, "
              f"≥1 full chunk: **{len(empties)-no_mel-no_step}**.\n")

    # Cross: per-etiology breakdown of "T < 1 chunk"
    sub_chunk_etio = Counter()
    for r in empties:
        if r["duration"] is not None and r["duration"] < 0.56:
            sub_chunk_etio[r["etiology"]] += 1
    if sum(sub_chunk_etio.values()) > 0:
        md.append("Of the under-1-chunk empties, by etiology:")
        md.append("| Etiology | Count |")
        md.append("|---|---:|")
        for e in sorted(sub_chunk_etio.keys()):
            md.append(f"| {e} | {sub_chunk_etio[e]} |")
        md.append("")

    with open(args.out_md, "w") as f:
        f.write("\n".join(md))
    print(f"wrote {args.out_md}")
    print("\n".join(md))


if __name__ == "__main__":
    main()
