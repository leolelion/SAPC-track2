#!/usr/bin/env python3
"""D0 — synthetic dysarthric corpus forensics (CPU-only, no NeMo, no GPU).

Answers, before any GPU is paid for, whether the two synthetic corpora
(F5-TTS v3 LoRA dys, kNN-VC inverse dys) actually contain parakeet's failure
region and whether they carry information SAP Train does not already have.

The failure region, measured on Dev_diag severe (experiments/exp_parakeet_ft_empties):
  - 48/425 utterances decode EMPTY, contributing +11.29 CER points
  - 22/48 of them are <=3 words (wake-words / short commands)
  - their onset (first 100 ms) median is -58.5 dBFS on a loud body (full RMS -29.6 dBFS)

Three pre-registered gates (see experiments/PLANNED.md; verdicts printed at the end):
  G-COVER  >=5% of utts <=3 words AND onset-dBFS p25 <= -45   -> corpus contains the failure region
  G-PROV   exact-text match vs SAP Train < 80%                -> corpus adds new lexical information
  G-EOS    |median trailing silence F5 - kNN-VC| < 300 ms     -> mixing is not a trailing-silence clash

Run `--mode discover` FIRST. The on-disk transcript layout is unverified; discover
reports it and exits without measuring anything. Only then run `--mode audit`.

  python3 scripts/d0_synth_forensics.py --mode discover \
      --f5-root    /workspace/data/processed/SAPC2_v3_synth \
      --knnvc-root /workspace/data/processed/SAPC2_v3_knnvc

  python3 scripts/d0_synth_forensics.py --mode audit \
      --f5-root    /workspace/data/processed/SAPC2_v3_synth \
      --knnvc-root /workspace/data/processed/SAPC2_v3_knnvc \
      --sap-train-csv $DATA_ROOT/manifest/Train.csv \
      --empty-refs experiments/exp_parakeet_ft_empties/empty_slice_refs.json \
      --out experiments/exp_d0_synth_forensics/d0_forensics.json
"""
import argparse
import collections
import json
import os
import random
import re
import sys

try:
    import numpy as np
except ImportError:
    sys.exit("numpy is required (pip install numpy)")

AUDIO_EXT = (".wav", ".flac", ".ogg", ".mp3")
TEXT_EXT = (".json", ".jsonl", ".csv", ".tsv", ".txt", ".manifest")

# Frame-level silence detection. Threshold is relative to the utterance's own loudest
# frame so it survives the large level differences between corpora, with an absolute
# floor so an all-silence file cannot define its own "speech".
FRAME_MS = 20.0
REL_THRESH_DB = 35.0
ABS_FLOOR_DBFS = -60.0
ONSET_MS = 100.0  # matches the empties probe definition


# ---------------------------------------------------------------- audio backend

def _read_audio(path):
    """Return (mono float32 in [-1,1], sample_rate). Prefers soundfile, falls back to stdlib wave."""
    try:
        import soundfile as sf
        x, sr = sf.read(path, dtype="float32", always_2d=True)
        return x.mean(axis=1), sr
    except ImportError:
        pass
    import wave
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        nch = w.getnchannels()
        width = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError(f"{path}: sampwidth={width}, install soundfile for non-16-bit audio")
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if nch > 1:
        x = x.reshape(-1, nch).mean(axis=1)
    return x, sr


def _dbfs(x):
    if x.size == 0:
        return float("-inf")
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    return 20.0 * np.log10(rms) if rms > 0 else float("-inf")


def _frame_dbfs(x, sr):
    n = max(1, int(sr * FRAME_MS / 1000.0))
    nf = x.size // n
    if nf == 0:
        return np.array([_dbfs(x)])
    f = x[: nf * n].reshape(nf, n).astype(np.float64)
    rms = np.sqrt(np.mean(f * f, axis=1))
    with np.errstate(divide="ignore"):
        return 20.0 * np.log10(np.maximum(rms, 1e-12))


def probe_audio(path):
    """Per-file acoustics. Returns None if the file cannot be read (counted, not fatal)."""
    x, sr = _read_audio(path)
    if x.size == 0:
        return None
    dur = x.size / float(sr)
    onset_n = max(1, int(sr * ONSET_MS / 1000.0))
    fd = _frame_dbfs(x, sr)
    peak_db = float(fd.max())
    thresh = max(peak_db - REL_THRESH_DB, ABS_FLOOR_DBFS)
    speech = np.flatnonzero(fd > thresh)
    frame_s = FRAME_MS / 1000.0
    if speech.size:
        lead_s = float(speech[0]) * frame_s
        trail_s = float(fd.size - 1 - speech[-1]) * frame_s
    else:
        lead_s, trail_s = dur, dur  # no frame clears threshold: treat the file as silent
    return {
        "sr": sr,
        "dur_s": dur,
        "rms_dbfs": _dbfs(x),
        "onset_dbfs": _dbfs(x[:onset_n]),
        "peak_frame_dbfs": peak_db,
        "lead_sil_s": lead_s,
        "trail_sil_s": trail_s,
        "all_silent": bool(speech.size == 0),
    }


# ------------------------------------------------------------------- text side

_PUNCT = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")


def norm_text(s):
    return _WS.sub(" ", _PUNCT.sub(" ", (s or "").lower())).strip()


def _iter_text_records(path):
    """Yield (key, text) from a manifest-ish file. Key is a wav stem when we can find one."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".json", ".jsonl", ".manifest"):
        with open(path, "r", errors="replace") as f:
            first = f.read(1)
            f.seek(0)
            if first == "[":  # a single JSON array
                try:
                    rows = json.load(f)
                except Exception:
                    return
                for r in rows if isinstance(rows, list) else []:
                    if isinstance(r, dict):
                        yield _stem(r.get("audio_filepath") or r.get("audio") or ""), r.get("text", "")
                return
            for line in f:  # jsonl
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if isinstance(r, dict):
                    yield _stem(r.get("audio_filepath") or r.get("audio") or ""), r.get("text", "")
    elif ext in (".csv", ".tsv"):
        import csv
        delim = "\t" if ext == ".tsv" else ","
        with open(path, "r", newline="", errors="replace") as f:
            for r in csv.DictReader(f, delimiter=delim):
                text = r.get("text") or r.get("transcript") or r.get("sentence") or ""
                key = r.get("id") or _stem(r.get("audio_filepath") or r.get("path") or "")
                yield key, text
    elif ext == ".txt":
        with open(path, "r", errors="replace") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                parts = line.split(None, 1)
                if len(parts) == 2 and not parts[0].isdigit():
                    yield _stem(parts[0]), parts[1]
                else:
                    yield "", line


def _stem(p):
    return os.path.splitext(os.path.basename(p or ""))[0]


def load_sap_train_text(csv_path):
    import csv
    texts = set()
    with open(csv_path, "r", newline="", errors="replace") as f:
        for r in csv.DictReader(f):
            t = norm_text(r.get("text", ""))
            if t:
                texts.add(t)
    return texts


# ------------------------------------------------------------------- discovery

def scan_corpus(root):
    """Walk once. Return audio paths bucketed by their directory, plus candidate text files."""
    buckets = collections.defaultdict(list)
    text_files = []
    for dirpath, _dirnames, filenames in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        for fn in filenames:
            low = fn.lower()
            if low.endswith(AUDIO_EXT):
                buckets[rel].append(os.path.join(dirpath, fn))
            elif low.endswith(TEXT_EXT):
                text_files.append(os.path.join(dirpath, fn))
    return buckets, text_files


def discover(name, root):
    if not os.path.isdir(root):
        print(f"[{name}] MISSING: {root}")
        return
    buckets, text_files = scan_corpus(root)
    total = sum(len(v) for v in buckets.values())
    print(f"\n=== {name} : {root} ===")
    print(f"audio files: {total}   buckets(dirs): {len(buckets)}")
    for b in sorted(buckets)[:40]:
        print(f"  {len(buckets[b]):>8}  {b}")
    if len(buckets) > 40:
        print(f"  ... and {len(buckets) - 40} more buckets")
    print(f"candidate transcript files: {len(text_files)}")
    for t in sorted(text_files)[:25]:
        try:
            sz = os.path.getsize(t)
        except OSError:
            sz = -1
        print(f"  {sz:>12}  {os.path.relpath(t, root)}")
    if len(text_files) > 25:
        print(f"  ... and {len(text_files) - 25} more")
    if not text_files:
        print("  !! NO transcript files found under this root. G-PROV cannot run.")
        print("     Locate the transcripts and pass --f5-text / --knnvc-text explicitly.")
    for b in sorted(buckets)[:3]:
        if buckets[b]:
            print(f"  example [{b}]: {os.path.basename(buckets[b][0])}")


# ----------------------------------------------------------------------- audit

def pct(vals, q):
    return float(np.percentile(np.asarray(vals, dtype=np.float64), q)) if len(vals) else None


def dist(vals):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    if not v:
        return {"n": 0}
    return {
        "n": len(v),
        "min": float(np.min(v)), "p25": pct(v, 25), "median": pct(v, 50),
        "p75": pct(v, 75), "p90": pct(v, 90), "max": float(np.max(v)),
        "mean": float(np.mean(v)),
    }


def audit_corpus(name, root, text_paths, sap_texts, empty_vocab, n_per_bucket, seed):
    if not os.path.isdir(root):
        return {"root": root, "error": "missing"}
    buckets, found_text = scan_corpus(root)
    total = sum(len(v) for v in buckets.values())

    # --- text ---
    sources = text_paths or found_text
    texts = {}
    for p in sources:
        try:
            for k, t in _iter_text_records(p):
                if t:
                    texts.setdefault(k, t)
        except Exception as e:
            print(f"[{name}] transcript read failed {p}: {e}")
    keyed = {k: v for k, v in texts.items() if k}

    normed = [norm_text(t) for t in texts.values()]
    normed = [t for t in normed if t]
    wc = [len(t.split()) for t in normed]
    short = sum(1 for c in wc if c <= 3)
    exact_hits = sum(1 for t in normed if t in sap_texts) if sap_texts else 0
    vocab = collections.Counter(w for t in normed for w in t.split())
    empty_cov = {w: vocab.get(w, 0) for w in empty_vocab}

    text_stats = {
        "sources": [os.path.basename(p) for p in sources],
        "n_transcripts": len(normed),
        "n_keyed_to_wav_stem": len(keyed),
        "short_le3_words": short,
        "short_le3_words_pct": (100.0 * short / len(normed)) if normed else None,
        "word_count": dist(wc),
        "exact_match_sap_train": exact_hits,
        "exact_match_sap_train_pct": (100.0 * exact_hits / len(normed)) if normed else None,
        "vocab_size": len(vocab),
        "empty_slice_vocab_covered": sum(1 for w in empty_vocab if vocab.get(w, 0) > 0),
        "empty_slice_vocab_total": len(empty_vocab),
        "empty_slice_vocab_counts": empty_cov,
    }

    # --- audio (sampled per bucket) ---
    rng = random.Random(seed)
    per_bucket = {}
    agg = collections.defaultdict(list)
    unreadable = 0
    for b in sorted(buckets):
        files = buckets[b]
        pick = files if len(files) <= n_per_bucket else rng.sample(files, n_per_bucket)
        rows = []
        for p in pick:
            try:
                r = probe_audio(p)
            except Exception:
                unreadable += 1
                continue
            if r is None:
                unreadable += 1
                continue
            rows.append(r)
            for k, v in r.items():
                if isinstance(v, (int, float)):
                    agg[k].append(v)
        if rows:
            per_bucket[b] = {
                "n_files": len(files),
                "n_sampled": len(rows),
                "dur_s": dist([r["dur_s"] for r in rows]),
                "rms_dbfs": dist([r["rms_dbfs"] for r in rows]),
                "onset_dbfs": dist([r["onset_dbfs"] for r in rows]),
                "trail_sil_s": dist([r["trail_sil_s"] for r in rows]),
                "lead_sil_s": dist([r["lead_sil_s"] for r in rows]),
                "all_silent": sum(1 for r in rows if r["all_silent"]),
            }

    sampled = sum(v["n_sampled"] for v in per_bucket.values())
    onset = agg["onset_dbfs"]
    quiet_onset = sum(1 for v in onset if v <= -45.0)
    return {
        "root": root,
        "n_audio_files": total,
        "n_buckets": len(buckets),
        "n_sampled": sampled,
        "n_unreadable": unreadable,
        "sample_rates": dict(collections.Counter(int(v) for v in agg["sr"])),
        "dur_s": dist(agg["dur_s"]),
        "est_total_hours_from_sample": (
            (float(np.mean(agg["dur_s"])) * total / 3600.0) if agg["dur_s"] else None
        ),
        "rms_dbfs": dist(agg["rms_dbfs"]),
        "onset_dbfs": dist(onset),
        "onset_le_neg45_pct": (100.0 * quiet_onset / len(onset)) if onset else None,
        "trail_sil_s": dist(agg["trail_sil_s"]),
        "lead_sil_s": dist(agg["lead_sil_s"]),
        "per_bucket": per_bucket,
        "text": text_stats,
    }


# ------------------------------------------------------------------- verdicts

def verdicts(f5, knn):
    out = {}
    for name, c in (("f5", f5), ("knnvc", knn)):
        t = c.get("text", {})
        short_pct = t.get("short_le3_words_pct")
        onset_p25 = (c.get("onset_dbfs") or {}).get("p25")
        cover = (short_pct is not None and short_pct >= 5.0
                 and onset_p25 is not None and onset_p25 <= -45.0)
        prov_pct = t.get("exact_match_sap_train_pct")
        out[f"G-COVER[{name}]"] = {
            "pass": bool(cover),
            "short_le3_words_pct": short_pct,
            "onset_dbfs_p25": onset_p25,
            "rule": "PASS if short<=3w >= 5% AND onset p25 <= -45 dBFS",
            "if_fail": "corpus lacks the failure region; D2/D3 demote to generic robustness",
        }
        out[f"G-PROV[{name}]"] = {
            "pass": (None if prov_pct is None else bool(prov_pct < 80.0)),
            "exact_match_sap_train_pct": prov_pct,
            "rule": "PASS if exact-text match vs SAP Train < 80%",
            "if_fail": "synth is a re-weighting of SAP, not new data; cap mix fraction hard",
        }
    a = (f5.get("trail_sil_s") or {}).get("median")
    b = (knn.get("trail_sil_s") or {}).get("median")
    gap_ms = (abs(a - b) * 1000.0) if (a is not None and b is not None) else None
    out["G-EOS"] = {
        "pass": (None if gap_ms is None else bool(gap_ms < 300.0)),
        "f5_trail_sil_median_ms": (a * 1000.0 if a is not None else None),
        "knnvc_trail_sil_median_ms": (b * 1000.0 if b is not None else None),
        "gap_ms": gap_ms,
        "rule": "PASS if |median trailing silence F5 - kNN-VC| < 300 ms",
        "if_fail": "trailing-silence clash is the mechanical candidate for the observed EOS collapse",
    }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["discover", "audit"], required=True)
    ap.add_argument("--f5-root", default="/workspace/data/processed/SAPC2_v3_synth")
    ap.add_argument("--knnvc-root", default="/workspace/data/processed/SAPC2_v3_knnvc")
    ap.add_argument("--f5-text", nargs="*", default=None,
                    help="explicit transcript file(s) for F5 (else auto-discovered under the root)")
    ap.add_argument("--knnvc-text", nargs="*", default=None)
    ap.add_argument("--sap-train-csv", default=None,
                    help="SAP Train.csv; required for G-PROV")
    ap.add_argument("--empty-refs",
                    default="experiments/exp_parakeet_ft_empties/empty_slice_refs.json")
    ap.add_argument("--n-per-bucket", type=int, default=400)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default="experiments/exp_d0_synth_forensics/d0_forensics.json")
    a = ap.parse_args()

    if a.mode == "discover":
        discover("F5-TTS", a.f5_root)
        discover("kNN-VC", a.knnvc_root)
        print("\nDiscovery only — nothing measured. Re-run with --mode audit once the "
              "transcript layout above is understood.")
        return

    sap_texts = set()
    if a.sap_train_csv:
        sap_texts = load_sap_train_text(a.sap_train_csv)
        print(f"[sap] {len(sap_texts)} unique normalized Train transcripts")
    else:
        print("[sap] --sap-train-csv not given -> G-PROV will report None")

    empty_vocab = []
    if os.path.exists(a.empty_refs):
        empty_vocab = json.load(open(a.empty_refs)).get("vocab", [])
        print(f"[empty-slice] {len(empty_vocab)} vocabulary items from the 48-utterance empty slice")
    else:
        print(f"[empty-slice] {a.empty_refs} not found -> coverage counts skipped")

    print("[audit] F5-TTS ...")
    f5 = audit_corpus("F5-TTS", a.f5_root, a.f5_text, sap_texts, empty_vocab,
                      a.n_per_bucket, a.seed)
    print("[audit] kNN-VC ...")
    knn = audit_corpus("kNN-VC", a.knnvc_root, a.knnvc_text, sap_texts, empty_vocab,
                       a.n_per_bucket, a.seed)

    v = verdicts(f5, knn)
    result = {
        "reference_empty_slice": {
            "n": 48, "of": 425, "cer_contribution_pts": 11.29,
            "short_le3_words": 22, "onset_dbfs_median": -58.5, "full_rms_dbfs_median": -29.6,
        },
        "params": {"n_per_bucket": a.n_per_bucket, "seed": a.seed},
        "f5": f5, "knnvc": knn, "verdicts": v,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(result, f, indent=1)

    print("\n================ D0 VERDICTS ================")
    for k in sorted(v):
        g = v[k]
        state = {True: "PASS", False: "FAIL", None: "N/A "}[g["pass"]]
        print(f"{state}  {k}")
        for kk, vv in g.items():
            if kk not in ("pass", "rule", "if_fail"):
                print(f"        {kk} = {vv}")
        print(f"        rule: {g['rule']}")
        if g["pass"] is False:
            print(f"        consequence: {g['if_fail']}")
    print(f"\nwrote {a.out}")
    print("D0_FORENSICS_DONE")


if __name__ == "__main__":
    main()
