#!/usr/bin/env python3
"""H-A stage 1 — is this checkpoint usable at a LARGER lookahead than [70,1]?

THE HYPOTHESIS
--------------
The shipped model runs `att_context_size=[70,1]` = 80 ms of right context = ONE encoder
frame of future. `scripts/analyze_rate_and_speakers.py` on the banked Dev_diag run shows
CER 39.70 / 31.03 / 18.19 / 10.31 across speaking-rate quartiles, and the deletions are
whole words at every rate. More right context is the only lever we have that can move
Q3+Q4, which hold 74% of the reference characters -- front-end time compression and
rate-conditioned decode both target Q1/Q2, capped at ~3.7 CER points by mass.

Precedent in the same family (`exp_sweep_fc114_*`): FastConformer-114M zero-shot went
54.77 -> 33.75 CER moving [70,1] -> [70,16].

WHAT THIS DECIDES
-----------------
`set_default_att_context_size()` will accept any context on a cache-aware encoder; that
does NOT mean the weights support it. Two separate questions, and this script separates
them because conflating them is how we would burn an export + decode session:

  Q1 STRUCTURAL: was the encoder TRAINED multi-context? Signals: `att_context_size` is a
     list-of-lists in the model config, `att_context_probs` is present, and
     `att_context_style == 'chunked_limited'`. Our finetune pinned a SINGLE context
     (`nemo_finetune_v2.py:  --train-ctx "[70,1]"`), so even a multi-context BASE may
     have had its other contexts degraded by our training.

  Q2 FUNCTIONAL: does it still transcribe at the new context? A tiny offline screen.

THE SCREEN CAN ONLY KILL, NEVER APPROVE.
It uses a punct-stripped single-reference CER, which is a PROXY. Our proxy scorers have
been wrong by 11 CER points (see the retraction in `experiments/PLANNED.md`). A good
proxy number here buys nothing except permission to spend the export + real-harness
gate; a bad one saves that spend. Never quote this number as a result.

USAGE (pod, nemo venv)
    python3 scripts/probe_lookahead_support.py \
        --nemo /workspace/parakeet_ft/armA_full/ft_smoke_encoder_only.nemo \
        --contexts 70,1 70,3 70,6 70,13 \
        --screen-manifest /workspace/SAPC2/manifest/Dev_diag.csv \
        --screen-n 30 --data-root /workspace/SAPC2 \
        --out-json /workspace/artifacts/step03/lookahead_probe.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional


def parse_ctx(s: str) -> List[int]:
    parts = [int(x) for x in s.replace("[", "").replace("]", "").split(",")]
    if len(parts) != 2:
        sys.exit(f"FATAL: context '{s}' is not left,right")
    return parts


def norm(s: str) -> str:
    return re.sub(r"<[^>]+>", " ", s or "").lower()


def strip_punct(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", norm(s))).strip()


def cer(hyp: str, ref: str) -> float:
    h, r = list(hyp), list(ref)
    if not r:
        return 0.0 if not h else 1.0
    dp = list(range(len(r) + 1))
    for i in range(1, len(h) + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, len(r) + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (h[i - 1] != r[j - 1]))
            prev = cur
    return min(1.0, dp[len(r)] / len(r))


def structural_report(model, out: Dict) -> None:
    """Q1: what does the checkpoint say about itself?"""
    enc = model.encoder
    cfg_enc = model.cfg.get("encoder", {})
    fields = {}
    for key in ("att_context_size", "att_context_probs", "att_context_style",
                "conv_context_size", "self_attention_model", "n_layers", "d_model",
                "subsampling", "subsampling_factor"):
        val = cfg_enc.get(key, None) if hasattr(cfg_enc, "get") else None
        if val is None:
            val = getattr(enc, key, None)
        try:
            json.dumps(val)
        except TypeError:
            val = str(val)
        fields[key] = val
    out["structural"] = fields

    acs = fields.get("att_context_size")
    multi = isinstance(acs, (list, tuple)) and len(acs) > 0 and isinstance(acs[0], (list, tuple))
    probs = fields.get("att_context_probs")
    out["structural"]["trained_multi_context"] = bool(multi)
    print("=== Q1 STRUCTURAL ===")
    for k, v in fields.items():
        print(f"  {k:<24} {v}")
    if multi:
        print(f"  VERDICT: base config declares {len(acs)} contexts -> multi-lookahead trained.")
        print("           BUT our finetune pinned a single context; only the screen can show")
        print("           whether the other contexts survived OUR training.")
    else:
        print("  VERDICT: config declares ONE context. Larger lookahead is OUT OF DISTRIBUTION.")
        print("           H-A becomes a GPU arm (retrain pinned at the new context), not a re-export.")
    if probs is not None:
        print(f"  att_context_probs present: {probs}")


def read_manifest(path: str, n: int, data_root: Optional[str]) -> List[Dict[str, str]]:
    import csv
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            fp = r["audio_filepath"]
            if data_root and not os.path.isabs(fp):
                fp = os.path.join(data_root, fp)
            rows.append({"id": r.get("id", ""), "audio_filepath": fp, "text": r.get("text", "")})
            if len(rows) >= n:
                break
    return rows


def functional_screen(model, contexts: List[List[int]], rows: List[Dict[str, str]],
                      batch_size: int, out: Dict) -> None:
    """Q2: does it still work at each context? OFFLINE transcribe, proxy CER, KILL-ONLY."""
    import torch
    refs = [strip_punct(r["text"]) for r in rows]
    paths = [r["audio_filepath"] for r in rows]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        sys.exit(f"FATAL: {len(missing)} screen wavs missing, first: {missing[0]}")

    print(f"\n=== Q2 FUNCTIONAL SCREEN (offline transcribe, n={len(rows)}, PROXY -- kill-only) ===")
    print(f"{'context':<12} {'proxy CER%':>11} {'empty':>6} {'vs [70,1]':>10}")
    model.eval()
    base_cer = None
    out["screen"] = {}
    for ctx in contexts:
        model.encoder.set_default_att_context_size(list(ctx))
        with torch.no_grad():
            hyps = model.transcribe(paths, batch_size=batch_size)
        texts = [(h.text if hasattr(h, "text") else h) for h in hyps]
        texts = [strip_punct(t) for t in texts]
        vals = [cer(t, r) for t, r in zip(texts, refs)]
        mean_cer = sum(vals) / len(vals)
        empty = sum(1 for t in texts if not t)
        if base_cer is None:
            base_cer = mean_cer
        delta = 100 * (mean_cer - base_cer)
        print(f"{str(ctx):<12} {100*mean_cer:>11.2f} {empty:>6} {delta:>+10.2f}")
        out["screen"][str(ctx)] = {"proxy_cer": mean_cer, "empty": empty,
                                   "delta_vs_first": mean_cer - base_cer}

    # ---- pre-registered kill rule (investigations/step03_runbook.md, G-HA0) ----
    best = min(out["screen"].items(), key=lambda kv: kv[1]["proxy_cer"])
    print(f"\n  best screened context: {best[0]} (proxy {100*best[1]['proxy_cer']:.2f}%)")
    survivors = [c for c, v in out["screen"].items() if v["delta_vs_first"] <= 0.03]
    out["survivors_G_HA0"] = survivors
    if len(survivors) <= 1:
        print("  G-HA0 KILL: no larger context is within +3 proxy pts of [70,1].")
        print("  -> our single-context finetune did not leave the other contexts usable.")
        print("  -> STOP the export/decode stages. H-A is a GPU arm. Do not spend the session.")
    else:
        print(f"  G-HA0 PASS: {survivors} survive the screen.")
        print("  -> permitted to spend export + ONNX parity + the REAL harness gate.")
        print("  -> the screen has NOT shown an improvement; only the official scorer can.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nemo", required=True)
    ap.add_argument("--contexts", nargs="+", default=["70,1", "70,3", "70,6", "70,13"],
                    help="FIRST one is the reference; deltas are measured against it")
    ap.add_argument("--screen-manifest", default=None, help="skip the screen if omitted")
    ap.add_argument("--screen-n", type=int, default=30)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    contexts = [parse_ctx(c) for c in args.contexts]
    out: Dict = {"nemo": args.nemo, "contexts": contexts}

    import nemo.collections.asr as nemo_asr
    print(f"=== restore {args.nemo} ===")
    model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(args.nemo, map_location="cpu")
    structural_report(model, out)

    if args.screen_manifest:
        rows = read_manifest(args.screen_manifest, args.screen_n, args.data_root)
        functional_screen(model, contexts, rows, args.batch_size, out)

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nwrote {args.out_json}")
    print("LOOKAHEAD_PROBE_DONE")


if __name__ == "__main__":
    main()
