#!/usr/bin/env python3
"""OFFLINE-RECOVERY probe (T3/T5) — the decisive discriminator before Arm B.

Question: the 48 severe utts the STREAMING wrapper returns empty — does the SAME Arm A
checkpoint recover them when decoded OFFLINE (whole utterance in one forward, no chunked
online emission)?

  offline RECOVERS most empties  => the fault is the streaming/online-emission path
     (chunked cache-aware decode starving/early-blanking), i.e. the H7 train->streaming
     gap — NOT the joint's blank bias. **Arm B (joint-unfreeze) would NOT fix it**; the
     lever is lookahead/left-context or the emission policy. This is the single most
     important thing to know, because it can veto a ~2.5h GPU run.
  offline STILL empty             => the model genuinely emits blank given full context
     => consistent with T1 (frozen-joint blank bias) -> Arm B is on-target.

Also prints the RAW (pre-<EOU>-strip) offline hyp, which surfaces T5: if the raw output
is only "<EOU>"/special tokens, the model fired end-of-utterance immediately (EOU misfire),
a different bug than a true blank.

Faithfulness: model load + decoding_cfg (greedy_batch, timestamps/alignments OFF) and the
att_context [70,1] are COPIED from track2_starting_kit/parakeet_realtime_ft/model.py so the
acoustic/decoder path matches deploy. The ONLY difference vs the wrapper is offline vs
chunked streaming — that is exactly the variable we want to isolate. NeMo required -> POD.

    python3 scripts/parakeet_offline_recovery.py \
      --nemo       /workspace/parakeet_ft/armA_full/ft_smoke_encoder_only.nemo \
      --manifest   /workspace/SAPC2/manifest/Dev_diag.csv \
      --empty-ids  /workspace/parakeet_ft/empty_ids.txt \
      --data-root  /workspace/SAPC2 \
      --att 70 1 \
      --out-json   /workspace/parakeet_ft/offline_recovery.json
"""
import argparse
import copy
import json
import os
import re
import sys


def build_offline_model(nemo_path, att, device_str):
    """Restore the .nemo and set the SAME greedy_batch decoding config the wrapper uses.
    Kept byte-for-byte in spirit with model.py __init__ (see comments there)."""
    import torch  # noqa
    import nemo.collections.asr as nemo_asr
    from omegaconf import open_dict

    device = __import__("torch").device(device_str)
    print(f"[recovery] restoring {nemo_path} on {device} …")
    model = nemo_asr.models.ASRModel.restore_from(nemo_path, map_location=device)
    model = model.to(device).eval()

    if not hasattr(model, "conformer_stream_step"):
        # not fatal for offline transcribe, but note it: means it isn't the cache-aware model
        print("[recovery][WARN] model has no conformer_stream_step (not the streaming ckpt?)")

    # decoding cfg — mirror model.py (greedy_batch, no timestamps/alignments)
    if hasattr(model, "change_decoding_strategy"):
        dc = copy.deepcopy(model.cfg.decoding)
        with open_dict(dc):
            dc.strategy = "greedy_batch"
            dc.compute_timestamps = False
            dc.preserve_alignments = False
            if dc.get("greedy") is not None:
                dc.greedy.preserve_alignments = False
        model.change_decoding_strategy(dc)

    # att_context [70,1] to match deploy; encoder still decodes the WHOLE utterance offline.
    if att and hasattr(model.encoder, "set_default_att_context_size"):
        model.encoder.set_default_att_context_size(list(att))
        print(f"[recovery] att_context set to {list(att)}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nemo", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--empty-ids", required=True, help="one utt id per line (from empties_analysis.py)")
    ap.add_argument("--data-root", default="")
    ap.add_argument("--att", nargs=2, type=int, default=[70, 1], help="att_context_size; use -1 -1 to skip")
    ap.add_argument("--limit", type=int, default=0, help="cap #utts (0 = all empties)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--strip-pattern", default=r"<[^>]+>")
    ap.add_argument("--out-json")
    a = ap.parse_args()

    import csv
    manifest = {row["id"]: row for row in csv.DictReader(open(a.manifest, newline=""))}
    ids = [ln.strip() for ln in open(a.empty_ids) if ln.strip()]
    if a.limit:
        ids = ids[: a.limit]
    if not ids:
        raise SystemExit("[FATAL] no empty ids given — nothing to test")

    paths, kept = [], []
    for i in ids:
        row = manifest.get(i)
        if not row:
            print(f"[recovery][WARN] id {i} not in manifest; skipping")
            continue
        p = row.get("audio_filepath") or ""
        if p and not os.path.isabs(p) and a.data_root:
            p = os.path.join(a.data_root, p)
        if not p or not os.path.exists(p):
            print(f"[recovery][WARN] audio missing for {i}: {p}")
            continue
        paths.append(p)
        kept.append(i)
    if not paths:
        raise SystemExit("[FATAL] no readable audio for any empty id — check --data-root")

    att = None if tuple(a.att) == (-1, -1) else a.att
    model = build_offline_model(a.nemo, att, a.device)

    print(f"[recovery] offline transcribe of {len(paths)} formerly-empty utts …")
    hyps = model.transcribe(paths, batch_size=a.batch)
    # NeMo may return list[str] or list[Hypothesis]; normalize (same quirk as model.py)
    def _text(x):
        if isinstance(x, str):
            return x
        t = getattr(x, "text", None)
        return t if isinstance(t, str) else ""
    if isinstance(hyps, tuple):  # some NeMo versions return (best, all)
        hyps = hyps[0]
    raw_hyps = [_text(h) for h in hyps]

    strip_re = re.compile(a.strip_pattern)
    recovered = eou_only = still_empty = 0
    rows = []
    for i, p, raw in zip(kept, paths, raw_hyps):
        stripped = " ".join(strip_re.sub(" ", raw).split())
        had_special = bool(strip_re.search(raw or ""))
        if stripped:
            recovered += 1
            status = "RECOVERED"
        elif had_special:
            eou_only += 1
            status = "EOU_ONLY"       # T5: emitted only special tokens
        else:
            still_empty += 1
            status = "STILL_EMPTY"    # consistent with true blank (T1 / Arm B)
        rows.append({"id": i, "status": status, "raw": raw, "stripped": stripped})

    tot = len(kept)
    out = {
        "nemo": a.nemo, "att_context": att, "n": tot,
        "recovered": recovered, "eou_only": eou_only, "still_empty": still_empty,
        "recovered_pct": round(100.0 * recovered / max(1, tot), 1),
        "still_empty_pct": round(100.0 * still_empty / max(1, tot), 1),
        "per_utt": rows,
    }
    print("\n==== OFFLINE-RECOVERY READ-OUT ====")
    print(f"of {tot} streaming-empty utts, OFFLINE: recovered={recovered} "
          f"({out['recovered_pct']}%)  eou_only={eou_only}  still_empty={still_empty}")
    print("VERDICT GUIDE:")
    print("  recovered >~60%  => STREAMING-path fault (H7 gap). Arm B WON'T fix it;")
    print("                      lever = lookahead/left-context or emission policy. VETO Arm B on empties.")
    print("  still_empty >~60% => true blank given full context => Arm B (joint-unfreeze) on-target.")
    print("  eou_only high     => T5 EOU-misfire, a separate bug (fix EOU handling, not the joint).")
    for r in rows[:20]:
        print(f"  [{r['status']:11s}] {r['id']}  raw={r['raw']!r}")

    if a.out_json:
        json.dump(out, open(a.out_json, "w"), indent=2)
        print(f"[wrote] {a.out_json}")
    print("PARAKEET_OFFLINE_RECOVERY_DONE")


if __name__ == "__main__":
    main()
