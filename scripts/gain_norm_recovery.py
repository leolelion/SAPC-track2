#!/usr/bin/env python3
"""FREE-FIX TEST: does RMS-normalizing the streaming-empty utts recover them OFFLINE?

Motivation: speaker_audio_probe found the empty-dominated speaker sits ~10 dB below the
corpus, and this model's preprocessor is normalize='NA' (no per-feature norm) -> absolute
level reaches the encoder -> out-of-distribution-quiet audio -> blank. This tests theory C's
FREE fix (a fixed input gain, legal in Pass 1 AND streaming) before any GPU on Arm B.

For each empty id: read wav, scale so RMS -> --target-dbfs, but cap the gain so the peak
stays <= --peak-ceiling (no new clipping); write a temp wav; transcribe OFFLINE with the SAME
model/decoding as parakeet_offline_recovery (imported, not duplicated). Report recovery,
per-speaker, vs the un-normalized 20.8% baseline.

    python3 scripts/gain_norm_recovery.py \
      --nemo /workspace/parakeet_ft/armA_full/ft_smoke_encoder_only.nemo \
      --manifest /workspace/SAPC2/manifest/Dev_diag.csv \
      --empty-ids /workspace/parakeet_ft/error_analysis/empty_ids.txt \
      --data-root /workspace/SAPC2 --att 70 1 --target-dbfs -25 \
      --out-json /workspace/parakeet_ft/error_analysis/gain_norm_recovery.json
"""
import argparse
import csv
import json
import math
import os
import re
import tempfile

from parakeet_offline_recovery import build_offline_model  # same dir on pod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nemo", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--empty-ids", required=True)
    ap.add_argument("--data-root", default="")
    ap.add_argument("--att", nargs=2, type=int, default=[70, 1])
    ap.add_argument("--target-dbfs", type=float, default=-25.0, help="corpus-median RMS target")
    ap.add_argument("--peak-ceiling", type=float, default=0.99, help="cap gain to keep peak below this")
    ap.add_argument("--speaker", default="", help="optional: restrict to one speaker prefix")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--strip-pattern", default=r"<[^>]+>")
    ap.add_argument("--out-json")
    a = ap.parse_args()

    import numpy as np
    import soundfile as sf

    manifest = {r["id"]: r for r in csv.DictReader(open(a.manifest, newline=""))}
    ids = [ln.strip() for ln in open(a.empty_ids) if ln.strip()]
    if a.speaker:
        ids = [i for i in ids if i.startswith(a.speaker) or
               (manifest.get(i, {}).get("speaker", "").startswith(a.speaker))]
    if not ids:
        raise SystemExit("[FATAL] no empty ids (after optional --speaker filter)")

    tgt_rms = 10 ** (a.target_dbfs / 20.0)
    tmpdir = tempfile.mkdtemp(prefix="gainnorm_")
    kept, tmp_paths, applied = [], [], []
    for i in ids:
        row = manifest.get(i)
        if not row:
            continue
        p = row.get("audio_filepath") or ""
        if p and not os.path.isabs(p) and a.data_root:
            p = os.path.join(a.data_root, p)
        if not p or not os.path.exists(p):
            print(f"[gain][WARN] audio missing for {i}: {p}")
            continue
        wav, sr = sf.read(p, dtype="float32", always_2d=False)
        if getattr(wav, "ndim", 1) > 1:
            wav = wav.mean(axis=1)
        x = wav.astype("float64")
        rms = (x ** 2).mean() ** 0.5
        peak = float(np.abs(x).max()) or 1e-9
        gain = tgt_rms / max(rms, 1e-9)
        # peak-safe: never push the peak past the ceiling (avoids introducing clipping)
        gain = min(gain, a.peak_ceiling / peak)
        y = np.clip(x * gain, -1.0, 1.0).astype("float32")
        db = lambda v: 20.0 * math.log10(v) if v > 1e-9 else -120.0
        applied.append({"id": i, "gain_db": round(db(gain), 2),
                        "rms_before": round(db(rms), 2),
                        "rms_after": round(db((y.astype("float64") ** 2).mean() ** 0.5), 2)})
        op = os.path.join(tmpdir, i.replace("/", "_") + ".wav")
        sf.write(op, y, sr)
        kept.append(i); tmp_paths.append(op)

    att = None if tuple(a.att) == (-1, -1) else a.att
    model = build_offline_model(a.nemo, att, a.device)
    print(f"[gain] offline transcribe of {len(tmp_paths)} RMS-normed empties (target {a.target_dbfs} dBFS) …")
    hyps = model.transcribe(tmp_paths, batch_size=8)
    if isinstance(hyps, tuple):
        hyps = hyps[0]

    def _text(x):
        if isinstance(x, str):
            return x
        t = getattr(x, "text", None)
        return t if isinstance(t, str) else ""

    strip_re = re.compile(a.strip_pattern)
    by_spk = {}
    rows = []
    recovered = 0
    gmap = {r["id"]: r for r in applied}
    for i, h in zip(kept, hyps):
        raw = _text(h)
        stripped = " ".join(strip_re.sub(" ", raw).split())
        ok = bool(stripped)
        recovered += ok
        spk = manifest[i].get("speaker") or i.split("_")[0]
        d = by_spk.setdefault(spk, {"n": 0, "recovered": 0})
        d["n"] += 1; d["recovered"] += ok
        rows.append({"id": i, "recovered": ok, "gain_db": gmap[i]["gain_db"],
                     "rms_after": gmap[i]["rms_after"], "raw": raw})

    tot = len(kept)
    out = {
        "target_dbfs": a.target_dbfs, "att_context": att, "n": tot,
        "recovered": recovered, "recovered_pct": round(100.0 * recovered / max(1, tot), 1),
        "baseline_recovered_pct_unnormalized": 20.8,  # from offline_recovery.json (48 utts)
        "by_speaker": {k: {**v, "recovered_pct": round(100.0 * v["recovered"] / max(1, v["n"]), 1)}
                       for k, v in sorted(by_spk.items(), key=lambda kv: -kv[1]["n"])},
        "per_utt": sorted(rows, key=lambda r: (not r["recovered"], r["gain_db"])),
    }
    print("\n==== GAIN-NORM RECOVERY READ-OUT ====")
    print(f"RMS-normed to {a.target_dbfs} dBFS: recovered {recovered}/{tot} "
          f"({out['recovered_pct']}%)  vs un-normalized baseline ~20.8%")
    print("VERDICT: big jump => FREE gain fix recovers empties (no Arm B needed for them).")
    print("         no jump  => level was not the trigger => data-bound/speaking-style (Arm B unlikely to help either).")
    for r in out["per_utt"][:25]:
        tag = "OK " if r["recovered"] else "..."
        print(f"  [{tag}] gain{r['gain_db']:+6.1f}dB -> {r['rms_after']:6.1f}dBFS  {r['id']}  raw={r['raw']!r}")
    if a.out_json:
        json.dump(out, open(a.out_json, "w"), indent=2)
        print(f"[wrote] {a.out_json}")
    print("GAIN_NORM_RECOVERY_DONE")


if __name__ == "__main__":
    main()
