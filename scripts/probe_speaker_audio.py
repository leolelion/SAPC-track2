#!/usr/bin/env python3
"""H-F — acoustic forensics on the speaker that owns 14.1% of our error mass.

WHAT PROVOKED THIS
------------------
`scripts/analyze_rate_and_speakers.py` on the banked Dev_diag run:

    speaker        etiol   n   err%   CER%   LOO CER%   shift   rate  empty
    55c1784a-ece   ALS    28   14.1  86.65      16.60   -2.14   4.04     21

75% of this one talker's utterances decode to NOTHING, its non-empty CER is still
38.86%, and dropping it moves whole-set CER by 2.14 points -- the same order as fixing
the entire slow-speech program. Other ALS speakers score 14.28% with 5 empties in 82.

The local numbers already EXCLUDE the obvious explanations. Its empty and non-empty
utterances are indistinguishable from each other and from the rest of the corpus on
duration (6.38 / 5.91 / 7.00 s), speaking rate (4.21 / 4.01 / 4.92 chars/s) and word
count (6 / 5 / 6). This is not graded severity and it is not the rate axis. Whatever it
is lives in the waveform, which is why this script needs the pod.

WHAT IT MEASURES
----------------
Per utterance, from the raw wav, dependency-free (stdlib `wave` + numpy):
  level     : full RMS dBFS, peak dBFS, clipping fraction, DC offset
  structure : noise floor (p10 of 20 ms frame RMS), speech level (p90), crude SNR,
              speech-activity ratio, leading silence before speech onset
  channel   : spectral centroid, HF/LF energy ratio -- a different mic, room or codec
              shows up here even when levels look normal

Then three contrasts, each reported with a rank-based separation statistic (AUC =
P(random target > random control); 0.5 = no separation, >=0.8 or <=0.2 = strong):
  1. target speaker  vs  all other speakers
  2. target speaker  vs  same-etiology speakers only   (controls for the disorder)
  3. empty-hypothesis utterances vs non-empty, corpus-wide (does the same descriptor
     that separates this speaker also separate the empties generally?)

Contrast 3 is what makes this more than one speaker's problem: a descriptor that
separates BOTH is a front-end fix; one that separates only the speaker is that
speaker's recording session.

NOTE ON A PRIOR RESULT: the frozen-scalar `input_gain` lever was already falsified
against the empties (0/48 recovered, official CER worse). That tested a level fix. It
did not test whether the audio is out-of-family on SNR or channel, which is what this
measures. Do not read a level finding here as reviving `input_gain`.

Emits NUMBERS ONLY -- no reference or hypothesis text (licensed SAP data).

USAGE (pod)
    python3 scripts/probe_speaker_audio.py \
        --manifest-csv /workspace/SAPC2/manifest/Dev_diag.csv \
        --data-root /workspace/SAPC2 \
        --decomp-summary /workspace/artifacts/step02/decomp_Dev_diag.banked.summary.json \
        --target-speaker 55c1784a \
        --out-json /workspace/artifacts/step03/speaker_forensics.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import wave
from typing import Dict, List, Optional, Sequence

import numpy as np

EPS = 1e-12
FRAME_MS = 20.0

# descriptors reported for every contrast, in the order a human should read them
DESCRIPTORS = [
    "rms_dbfs", "peak_dbfs", "clip_frac", "dc_offset",
    "noise_floor_dbfs", "speech_dbfs", "snr_db", "speech_activity",
    "lead_silence_s", "centroid_hz", "hf_lf_ratio_db",
]


def read_wav(path: str) -> tuple[np.ndarray, int]:
    """Read a wav to float32 mono in [-1,1]. Fails loudly on anything unexpected."""
    with wave.open(path, "rb") as wf:
        n_ch, width, sr, n_frames = wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.getnframes()
        raw = wf.readframes(n_frames)
    if width == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif width == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"{path}: unsupported sample width {width}")
    if n_ch > 1:
        x = x.reshape(-1, n_ch).mean(axis=1)
    return x, sr


def db(x: float) -> float:
    return 20.0 * math.log10(max(float(x), EPS))


def descriptors(x: np.ndarray, sr: int) -> Dict[str, float]:
    if x.size == 0:
        return {k: float("nan") for k in DESCRIPTORS}
    frame = max(1, int(sr * FRAME_MS / 1000.0))
    n_fr = x.size // frame
    if n_fr < 3:
        frames = x[None, :]
    else:
        frames = x[: n_fr * frame].reshape(n_fr, frame)
    fr_rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1) + EPS)

    noise = float(np.percentile(fr_rms, 10))
    speech = float(np.percentile(fr_rms, 90))
    # onset = first frame that clears the noise floor by 10 dB; NaN if it never does
    thresh = noise * (10 ** (10.0 / 20.0))
    above = np.flatnonzero(fr_rms >= thresh)
    lead = float(above[0] * frame / sr) if above.size else float("nan")

    # channel signature from the magnitude spectrum of the whole utterance
    n_fft = 1 << max(8, int(math.ceil(math.log2(min(x.size, 65536)))))
    seg = x[:n_fft] if x.size >= n_fft else np.pad(x, (0, n_fft - x.size))
    mag = np.abs(np.fft.rfft(seg * np.hanning(n_fft)))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    p = mag ** 2
    centroid = float((freqs * p).sum() / (p.sum() + EPS))
    lo = p[(freqs > 100) & (freqs <= 1000)].sum()
    hi = p[(freqs > 4000)].sum()

    return {
        "rms_dbfs": db(np.sqrt((x.astype(np.float64) ** 2).mean())),
        "peak_dbfs": db(np.abs(x).max()),
        "clip_frac": float((np.abs(x) >= 0.999).mean()),
        "dc_offset": float(x.mean()),
        "noise_floor_dbfs": db(noise),
        "speech_dbfs": db(speech),
        "snr_db": db(speech) - db(noise),
        "speech_activity": float((fr_rms >= thresh).mean()),
        "lead_silence_s": lead,
        "centroid_hz": centroid,
        "hf_lf_ratio_db": 10.0 * math.log10((hi + EPS) / (lo + EPS)),
    }


def auc(a: Sequence[float], b: Sequence[float]) -> float:
    """P(random a > random b), ties at 0.5 (Mann-Whitney U / n1n2). No scipy."""
    a = [v for v in a if not math.isnan(v)]
    b = [v for v in b if not math.isnan(v)]
    if not a or not b:
        return float("nan")
    merged = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    ranks: Dict[int, float] = {}
    i = 0
    rank_sum_a = 0.0
    while i < len(merged):
        j = i
        while j + 1 < len(merged) and merged[j + 1][0] == merged[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            if merged[k][1] == 0:
                rank_sum_a += avg_rank
        i = j + 1
    n1, n2 = len(a), len(b)
    u = rank_sum_a - n1 * (n1 + 1) / 2.0
    return u / (n1 * n2)


def med(vals: Sequence[float]) -> float:
    v = sorted(x for x in vals if not math.isnan(x))
    return float("nan") if not v else (v[len(v) // 2] if len(v) % 2 else 0.5 * (v[len(v) // 2 - 1] + v[len(v) // 2]))


def contrast(name: str, group: List[Dict], control: List[Dict], out: Dict) -> None:
    print(f"\n=== {name}   (target n={len(group)}, control n={len(control)}) ===")
    print(f"{'descriptor':<20} {'target med':>11} {'control med':>12} {'delta':>10} {'AUC':>6}  flag")
    rows = {}
    for d in DESCRIPTORS:
        g = [u["ac"][d] for u in group]
        c = [u["ac"][d] for u in control]
        mg, mc, a = med(g), med(c), auc(g, c)
        flag = "STRONG" if (not math.isnan(a) and (a >= 0.80 or a <= 0.20)) else ""
        print(f"{d:<20} {mg:>11.3f} {mc:>12.3f} {mg-mc:>+10.3f} {a:>6.2f}  {flag}")
        rows[d] = {"target_median": mg, "control_median": mc, "delta": mg - mc, "auc": a}
    out[name] = {"n_target": len(group), "n_control": len(control), "descriptors": rows}
    strong = [d for d, r in rows.items() if not math.isnan(r["auc"]) and (r["auc"] >= 0.80 or r["auc"] <= 0.20)]
    print(f"  separating descriptors (AUC>=0.80 or <=0.20): {strong if strong else 'NONE'}")
    if not strong:
        print("  -> this contrast finds NO acoustic signature. Do not invent one from the medians.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest-csv", required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--decomp-summary", required=True,
                    help="summary.json from error_decomposition.py -- supplies hyp_empty + speaker")
    ap.add_argument("--target-speaker", required=True, help="speaker id or unique prefix")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    with open(args.decomp_summary) as fh:
        summary = json.load(fh)
    by_id = {u["id"]: u for u in summary["per_utterance"]}

    utts: List[Dict] = []
    with open(args.manifest_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            fp = row["audio_filepath"]
            if args.data_root and not os.path.isabs(fp):
                fp = os.path.join(args.data_root, fp)
            rec = by_id.get(row["id"])
            if rec is None:
                continue
            if not os.path.exists(fp):
                sys.exit(f"FATAL: wav missing: {fp}")
            x, sr = read_wav(fp)
            utts.append({
                "id": row["id"], "speaker": row.get("speaker", rec.get("speaker", "?")),
                "etiology": row.get("etiology", rec.get("etiology", "?")),
                "sr": sr, "n_samples": int(x.size),
                "hyp_empty": bool(rec.get("hyp_empty")),
                "cer": rec.get("cer"), "ac": descriptors(x, sr),
            })
    if not utts:
        sys.exit("FATAL: no utterances joined between manifest and decomposition summary")
    print(f"[load] {len(utts)} utterances; sample rates {sorted({u['sr'] for u in utts})}")

    target = [u for u in utts if u["speaker"].startswith(args.target_speaker)]
    if not target:
        sys.exit(f"FATAL: no utterances for speaker prefix {args.target_speaker}")
    others = [u for u in utts if not u["speaker"].startswith(args.target_speaker)]
    etio = target[0]["etiology"]
    same_etio = [u for u in others if u["etiology"] == etio]

    out: Dict = {"manifest": args.manifest_csv, "target_speaker": args.target_speaker,
                 "target_etiology": etio, "n_utts": len(utts)}
    contrast("C1 target vs all other speakers", target, others, out)
    if same_etio:
        contrast(f"C2 target vs same-etiology ({etio}) speakers", target, same_etio, out)
    else:
        print(f"\n=== C2 skipped: no other {etio} speakers in this split ===")
    contrast("C3 empty vs non-empty (corpus-wide)",
             [u for u in utts if u["hyp_empty"]], [u for u in utts if not u["hyp_empty"]], out)
    contrast("C4 target's empties vs target's non-empties",
             [u for u in target if u["hyp_empty"]], [u for u in target if not u["hyp_empty"]], out)

    print("\n=== READING GUIDE ===")
    print("  C1+C2 strong, C3 flat  -> that speaker's RECORDING SESSION. Fix is data-side or")
    print("                            per-speaker; it will not generalise to Test.")
    print("  C1+C2 strong, C3 strong-> a FRONT-END gap that also drives the corpus empties.")
    print("                            Highest value: one fix, two problems.")
    print("  C1+C2 flat             -> the audio is in-family; the failure is in the MODEL for")
    print("                            this talker. No front-end fix exists. Needs training data.")
    print("  C4 strong              -> within this speaker, the empties are acoustically distinct")
    print("                            -> a per-utterance trigger exists, not a per-speaker one.")

    out["per_utterance"] = [{k: v for k, v in u.items()} for u in utts]
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nwrote {args.out_json}")
    print("SPEAKER_FORENSICS_DONE")


if __name__ == "__main__":
    main()
