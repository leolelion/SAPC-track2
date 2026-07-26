#!/usr/bin/env python3
"""Per-speaker audio characterization — is the empty-dominated speaker's audio
systematically BAD (low-energy / clipped / distorted) or just hard speech?

Follow-up to parakeet_empty_probe.py's T4 finding (one speaker = ~half the empties).
Contrasts four groups on raw-audio statistics:
  (1) TARGET speaker, empty utts     (2) TARGET speaker, non-empty utts
  (3) OTHER speakers, empty utts      (4) OTHER speakers, non-empty utts
so we can tell:
  * target much quieter than others          -> gain story (free RMS-norm test next)
  * target's empties quieter than its own non-empties -> within-speaker gain story
  * target high clip_frac / peak pinned at 0 dBFS     -> clipping/distortion (bad capture)
  * target empties ~= its non-empties on energy       -> NOT energy; speaking-style/data-bound

Stats per utt: duration, rms_dbfs, peak_dbfs, clip_frac (|x|>=0.999), dc_offset.
soundfile only; a file that can't be read is COUNTED, never silently dropped.

    python3 scripts/speaker_audio_probe.py \
      --manifest /workspace/SAPC2/manifest/Dev_diag.csv \
      --empty-ids /workspace/parakeet_ft/error_analysis/empty_ids.txt \
      --data-root /workspace/SAPC2 --speaker 55c1784a-ece4-414a-25b4-08dc18e8f490 \
      --out-json /workspace/parakeet_ft/error_analysis/speaker_probe.json
"""
import argparse
import csv
import json
import math
import os
from collections import defaultdict


def stats(path):
    import numpy as np
    import soundfile as sf
    try:
        wav, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception:
        return None
    if getattr(wav, "ndim", 1) > 1:
        wav = wav.mean(axis=1)
    if wav.size == 0:
        return {"dur": 0.0, "rms_dbfs": -120.0, "peak_dbfs": -120.0, "clip_frac": 0.0, "dc": 0.0}
    x = wav.astype("float64")
    rms = float((x ** 2).mean() ** 0.5)
    peak = float(np.abs(x).max())
    db = lambda v: 20.0 * math.log10(v) if v > 1e-9 else -120.0
    return {
        "dur": wav.size / float(sr),
        "rms_dbfs": db(rms),
        "peak_dbfs": db(peak),
        "clip_frac": float((np.abs(x) >= 0.999).mean()),
        "dc": float(x.mean()),
    }


def summ(rows, key):
    xs = sorted(r[key] for r in rows if r is not None)
    if not xs:
        return {"n": 0, "median": None, "p25": None, "p75": None, "min": None, "max": None}
    q = lambda p: xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))]
    return {"n": len(xs), "median": round(q(.5), 4), "p25": round(q(.25), 4),
            "p75": round(q(.75), 4), "min": round(xs[0], 4), "max": round(xs[-1], 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--empty-ids", required=True)
    ap.add_argument("--data-root", default="")
    ap.add_argument("--speaker", required=True, help="target speaker id (or its unique prefix)")
    ap.add_argument("--out-json")
    a = ap.parse_args()

    manifest = {r["id"]: r for r in csv.DictReader(open(a.manifest, newline=""))}
    empties = {ln.strip() for ln in open(a.empty_ids) if ln.strip()}

    groups = defaultdict(list)          # group -> list[stats]
    per_utt_target = []                 # detailed rows for the target speaker
    missing = 0
    for uid, row in manifest.items():
        spk = row.get("speaker") or (uid.split("_")[0] if "_" in uid else "")
        is_target = spk == a.speaker or spk.startswith(a.speaker) or uid.startswith(a.speaker)
        is_empty = uid in empties
        p = row.get("audio_filepath") or ""
        if p and not os.path.isabs(p) and a.data_root:
            p = os.path.join(a.data_root, p)
        st = stats(p) if p else None
        if st is None:
            missing += 1
            continue
        g = ("target" if is_target else "other") + ("_empty" if is_empty else "_nonempty")
        groups[g].append(st)
        if is_target:
            per_utt_target.append({"id": uid, "empty": is_empty, **{k: round(v, 4) for k, v in st.items()}})

    keys = ["rms_dbfs", "peak_dbfs", "clip_frac", "dur", "dc"]
    out = {"speaker": a.speaker, "audio_missing": missing,
           "groups": {g: {k: summ(rows, k) for k in keys} | {"n_utts": len(rows)}
                      for g, rows in sorted(groups.items())},
           "target_per_utt": sorted(per_utt_target, key=lambda r: (not r["empty"], r["rms_dbfs"]))}
    print(json.dumps(out, indent=2))

    def med(g, k):
        return out["groups"].get(g, {}).get(k, {}).get("median")
    print("\n==== READ-OUT ====")
    print(f"RMS dBFS (median): target_empty={med('target_empty','rms_dbfs')} "
          f"target_nonempty={med('target_nonempty','rms_dbfs')} "
          f"other_empty={med('other_empty','rms_dbfs')} "
          f"other_nonempty={med('other_nonempty','rms_dbfs')}")
    print(f"peak dBFS (median): target_empty={med('target_empty','peak_dbfs')} "
          f"other_nonempty={med('other_nonempty','peak_dbfs')}   "
          f"clip_frac: target_empty={med('target_empty','clip_frac')} "
          f"other_nonempty={med('other_nonempty','clip_frac')}")
    print("Interpret: target much lower RMS than others => gain story (test RMS-norm, free).")
    print("           target empty ~= target nonempty on RMS => NOT energy; speaking-style/data-bound.")
    print("           target clip_frac high / peak ~0 dBFS => clipped/distorted capture.")
    if a.out_json:
        json.dump(out, open(a.out_json, "w"), indent=2)
        print(f"[wrote] {a.out_json}")
    print("SPEAKER_AUDIO_PROBE_DONE")


if __name__ == "__main__":
    main()
