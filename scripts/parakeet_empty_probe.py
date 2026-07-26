#!/usr/bin/env python3
"""Empty-utterance ROOT-CAUSE probe for a Track-2 hyp csv (parakeet Arm A Dev_diag).

This is the CHEAP, NeMo-free half of the error analysis. It does NOT decide anything on
its own — it produces the signals that discriminate the competing theories for WHY
~11% of severe utts come back empty, so we don't buy Arm B (joint-unfreeze, ~2.5h GPU)
on faith. Pair it with parakeet_offline_recovery.py (the T3/T5 discriminator that needs
the model). Decision rule lives in run_parakeet_error_analysis.sh.

Theories and the signal THIS script gives for each:
  T1  frozen-joint blank bias (Arm B is the fix): non-empty errors should be
      DELETION-heavy. -> del:sub:ins ratio on non-empty utts (char-level).
  T2  encoder-energy / low input gain (free RMS-norm fix, NO gpu): empties should be
      systematically QUIETER / shorter. -> RMS(dBFS), peak, duration: empty vs non-empty.
  T4  genuinely unintelligible acoustics (no cheap lever helps): empties spread across
      many speakers with high error everywhere. -> per-speaker / per-etiology empty rate.

Reuses the SAME conventions as empties_analysis.py / streaming_cer_bootstrap.py: official
normalizer when available, min-over-refs, refs = with/without-disfluency. Audio read is
soundfile-only; if a file can't be read we COUNT it (never silently skip -> no `or {}`).

    python3 scripts/parakeet_empty_probe.py \
      --manifest-csv /workspace/SAPC2/manifest/Dev_diag.csv \
      --hyp-csv      /workspace/parakeet_ft/Dev_diag.armA.predict.csv \
      --utils-dir    /workspace/SAPC-template/utils \
      --data-root    /workspace/SAPC2 \
      --out-json     /workspace/parakeet_ft/empty_probe.json
"""
import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict


# ----------------------------------------------------------------------------- normalizer
def load_normalizer(utils_dir):
    """Official HF normalizer if importable, else a documented lowercase/strip fallback.
    Fallback is flagged loudly so a run on the fallback is never mistaken for the real one."""
    if utils_dir:
        sys.path.insert(0, utils_dir)
    try:
        from normalizer.text_normalizer_hf import EnglishTextNormalizer
        return EnglishTextNormalizer().norm, True
    except Exception as exc:
        print(f"[WARN] official normalizer unavailable ({exc}); using fallback", file=sys.stderr)

    def fallback(text):
        text = (text or "").lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    return fallback, False


def refs_for(row):
    refs = []
    for key in ("norm_text_with_disfluency", "norm_text_without_disfluency", "text", "raw_trans"):
        val = (row.get(key) or "").strip()
        if val and val not in refs:
            refs.append(val)
    return refs


# ----------------------------------------------------------------------------- alignment (T1)
def levenshtein_ops(hyp, ref):
    """Char-level edit distance with op backtrace. Returns (sub, ins, del, ref_len).
    ins  = chars in hyp not in ref (model said too much)
    del  = chars in ref not in hyp (model dropped them -> RNN-T blank/deletion signal)
    Standard DP; ties resolved sub>del>ins (arbitrary but fixed)."""
    h, r = list(hyp), list(ref)
    n, m = len(h), len(r)
    if m == 0:
        return 0, n, 0, 0
    # dp[i][j] = edits to turn h[:i] into r[:j]; keep back-pointers
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    bp = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
        bp[i][0] = "I"  # deleting hyp char == insertion relative to ref
    for j in range(1, m + 1):
        dp[0][j] = j
        bp[0][j] = "D"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if h[i - 1] == r[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                bp[i][j] = "M"
                continue
            sub = dp[i - 1][j - 1] + 1
            dele = dp[i][j - 1] + 1   # ref char unmatched -> deletion
            ins = dp[i - 1][j] + 1    # hyp char unmatched -> insertion
            best = min(sub, dele, ins)
            dp[i][j] = best
            bp[i][j] = "S" if best == sub else ("D" if best == dele else "I")
    i, j = n, m
    s = d = ins = 0
    while i > 0 or j > 0:
        op = bp[i][j]
        if op == "M":
            i, j = i - 1, j - 1
        elif op == "S":
            s += 1; i, j = i - 1, j - 1
        elif op == "D":
            d += 1; j -= 1
        else:  # "I"
            ins += 1; i -= 1
    return s, ins, d, m


# ----------------------------------------------------------------------------- audio (T2)
def audio_stats(path):
    """(duration_s, rms_dbfs, peak_dbfs) or None if unreadable. soundfile only."""
    try:
        import numpy as np
        import soundfile as sf
    except Exception as exc:
        raise SystemExit(f"[FATAL] need numpy+soundfile for the T2 energy probe: {exc}")
    try:
        wav, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception:
        return None
    if getattr(wav, "ndim", 1) > 1:
        wav = wav.mean(axis=1)
    if wav.size == 0:
        return (0.0, -120.0, -120.0)
    rms = float((wav.astype("float64") ** 2).mean() ** 0.5)
    peak = float(abs(wav).max())
    to_db = lambda x: 20.0 * (__import__("math").log10(x)) if x > 1e-9 else -120.0
    return (wav.size / float(sr), to_db(rms), to_db(peak))


def summarize(vals):
    """median + IQR of a list of floats (empty -> Nones)."""
    xs = sorted(v for v in vals if v is not None)
    if not xs:
        return {"n": 0, "median": None, "p25": None, "p75": None}
    q = lambda p: xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))]
    return {"n": len(xs), "median": round(q(0.5), 3), "p25": round(q(0.25), 3), "p75": round(q(0.75), 3)}


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-csv", required=True)
    ap.add_argument("--hyp-csv", required=True)
    ap.add_argument("--utils-dir", default="/workspace/SAPC-template/utils")
    ap.add_argument("--data-root", default="", help="prepended to relative audio_filepath")
    ap.add_argument("--no-audio", action="store_true", help="skip the T2 audio-energy probe")
    ap.add_argument("--out-json")
    ap.add_argument("--empty-ids-out")
    a = ap.parse_args()

    norm, official = load_normalizer(a.utils_dir)
    manifest = {row["id"]: row for row in csv.DictReader(open(a.manifest_csv, newline=""))}
    hyps = list(csv.DictReader(open(a.hyp_csv, newline="")))

    empty_ids = []
    # T1 accumulators (non-empty only)
    tot_sub = tot_ins = tot_del = tot_reflen = 0
    # T2 accumulators
    ea = {"empty": defaultdict(list), "nonempty": defaultdict(list)}
    audio_missing = 0
    audio_read = 0
    # T4 accumulators
    by_spk = defaultdict(lambda: {"n": 0, "empty": 0})
    by_eti = defaultdict(lambda: {"n": 0, "empty": 0})
    n = 0

    for h in hyps:
        row = manifest.get(h["id"])
        if not row:
            continue
        refs = [norm(r) for r in refs_for(row)]
        refs = [r for r in refs if r]
        if not refs:
            continue
        n += 1
        raw = (h.get("raw_hypos") or "").strip()
        is_empty = not raw
        spk = row.get("speaker") or (h["id"].split("_")[0] if "_" in h["id"] else "UNK")
        eti = row.get("etiology") or "UNKNOWN"
        by_spk[spk]["n"] += 1
        by_eti[eti]["n"] += 1
        bucket = "empty" if is_empty else "nonempty"

        if is_empty:
            empty_ids.append(h["id"])
            by_spk[spk]["empty"] += 1
            by_eti[eti]["empty"] += 1
        else:
            # T1: char-align against the ref giving fewest total edits
            # (min-over-refs, consistent with the official scorer).
            hn = norm(raw)
            best_ops = None
            for r in refs:
                s, ins, d, rl = levenshtein_ops(hn, r)
                if best_ops is None or (s + ins + d) < sum(best_ops[:3]):
                    best_ops = (s, ins, d, rl)
            s, ins, d, rl = best_ops
            tot_sub += s; tot_ins += ins; tot_del += d; tot_reflen += rl

        # T2: audio energy for BOTH buckets (so we can compare)
        if not a.no_audio:
            ap_ = row.get("audio_filepath") or ""
            if ap_ and not os.path.isabs(ap_) and a.data_root:
                ap_ = os.path.join(a.data_root, ap_)
            st = audio_stats(ap_) if ap_ else None
            if st is None:
                audio_missing += 1
            else:
                audio_read += 1
                dur, rms, peak = st
                ea[bucket]["dur"].append(dur)
                ea[bucket]["rms"].append(rms)
                ea[bucket]["peak"].append(peak)
        # duration from manifest as a fallback signal even without audio
        try:
            ea[bucket]["dur_manifest"].append(float(row.get("duration") or 0.0))
        except ValueError:
            pass

    n_empty = len(empty_ids)
    denom = max(1, tot_reflen)
    out = {
        "normalizer_official": official,
        "n": n,
        "n_empty": n_empty,
        "empty_frac_pct": round(100.0 * n_empty / max(1, n), 2),
        "T1_nonempty_error_mix": {
            "sub": tot_sub, "ins": tot_ins, "del": tot_del, "ref_chars": tot_reflen,
            "sub_rate_pct": round(100.0 * tot_sub / denom, 2),
            "del_rate_pct": round(100.0 * tot_del / denom, 2),
            "ins_rate_pct": round(100.0 * tot_ins / denom, 2),
            "del_to_sub_ratio": round(tot_del / max(1, tot_sub), 3),
        },
        "T2_energy": {
            "audio_read": audio_read, "audio_missing": audio_missing,
            "empty": {k: summarize(ea["empty"][k]) for k in ("dur", "rms", "peak", "dur_manifest")},
            "nonempty": {k: summarize(ea["nonempty"][k]) for k in ("dur", "rms", "peak", "dur_manifest")},
        },
        "T4_by_speaker": {
            k: {"n": v["n"], "empty": v["empty"],
                "empty_pct": round(100.0 * v["empty"] / max(1, v["n"]), 1)}
            for k, v in sorted(by_spk.items(), key=lambda kv: -kv[1]["empty"])
        },
        "T4_by_etiology": {
            k: {"n": v["n"], "empty": v["empty"],
                "empty_pct": round(100.0 * v["empty"] / max(1, v["n"]), 1)}
            for k, v in sorted(by_eti.items())
        },
    }

    print(json.dumps(out, indent=2))
    t1 = out["T1_nonempty_error_mix"]
    t2e, t2n = out["T2_energy"]["empty"], out["T2_energy"]["nonempty"]
    print("\n==== READ-OUT ====")
    print(f"empties: {n_empty}/{n} ({out['empty_frac_pct']}%)")
    print(f"T1 non-empty error mix: del/sub ratio = {t1['del_to_sub_ratio']} "
          f"(del {t1['del_rate_pct']}% vs sub {t1['sub_rate_pct']}%) "
          f"-> >1.3 favors blank-propensity (T1/Arm B), <0.8 favors acoustic (encoder).")
    if not a.no_audio:
        print(f"T2 energy (RMS dBFS median): empty={t2e['rms']['median']} vs nonempty={t2n['rms']['median']} "
              f"| dur median: empty={t2e['dur']['median']}s vs nonempty={t2n['dur']['median']}s "
              f"(read {out['T2_energy']['audio_read']}, missing {out['T2_energy']['audio_missing']}) "
              f"-> empties much quieter => T2 (free RMS-norm), not GPU.")
    print("T4: see per-speaker table — concentrated in few speakers => tractable; "
          "spread wide + high error => T4 (data-bound, cheap levers won't save it).")

    if a.out_json:
        json.dump(out, open(a.out_json, "w"), indent=2, sort_keys=True)
    if a.empty_ids_out:
        with open(a.empty_ids_out, "w") as f:
            f.write("\n".join(empty_ids) + ("\n" if empty_ids else ""))
        print(f"[wrote] empty ids -> {a.empty_ids_out}")
    print("PARAKEET_EMPTY_PROBE_DONE")


if __name__ == "__main__":
    main()
