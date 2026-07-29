#!/usr/bin/env python3
"""Probe: what does the exported parakeet encoder graph actually do with pre-encode frames?

Answers two questions that decide the ONNX wrapper's stepping, and that cannot be settled
by reading NeMo's source (the exporter may bake `drop_extra_pre_encoded` / `valid_out_len`
into the graph):

  Q1. Does the graph drop the pre-encoded cache region itself? Replay each NeMo step and
      compare the graph's encoded frame count against NeMo's POST-drop count.
  Q2. Can step 0 be fed the way NeMo feeds it (9 frames, no pre-cache)? If the graph drops
      unconditionally, that underflows to a zero-length sequence and Conv errors — in which
      case the Nemotron-style step 0 (zero-pad the pre-cache to full width) is required.

Prints one line per step; the last block is the verdict. POD-ONLY.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

CHUNK = 1600


def load_module(path: Path, name: str):
    sys.path.insert(0, str(path.parent.resolve()))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-dir", type=Path, required=True)
    ap.add_argument("--nemo-dir", type=Path, required=True)
    ap.add_argument("--wav", type=Path, required=True)
    args = ap.parse_args()

    import inspect

    import soundfile as sf

    nemo_mod = load_module(args.nemo_dir / "model.py", "p_nemo")
    onnx_mod = load_module(args.onnx_dir / "model.py", "p_onnx")
    nm = nemo_mod.Model()
    om = onnx_mod.Model()

    audio, sr = sf.read(str(args.wav), dtype="float32")
    assert sr == 16000

    steps = []
    enc = nm.model.encoder
    original = enc.cache_aware_stream_step
    sig = inspect.signature(original)

    def rec(*a, **kw):
        b = sig.bind(*a, **kw)
        b.apply_defaults()
        ar = b.arguments
        out = original(*a, **kw)
        steps.append(
            {
                "fed": ar["processed_signal"].detach().cpu().numpy().astype(np.float32),
                "len": int(ar["processed_signal_length"].detach().cpu().numpy().reshape(-1)[0]),
                "lc": ar["cache_last_channel"].detach().cpu().numpy().astype(np.float32),
                "lt": ar["cache_last_time"].detach().cpu().numpy().astype(np.float32),
                "ll": ar["cache_last_channel_len"].detach().cpu().numpy(),
                "drop": int(ar.get("drop_extra_pre_encoded") or 0),
                "ref": out[0].detach().cpu().numpy().astype(np.float32),
                "ref_n": int(out[1].detach().cpu().numpy().reshape(-1)[0]),
            }
        )
        return out

    enc.cache_aware_stream_step = rec
    nm.reset()
    for off in range(0, len(audio), CHUNK):
        nm.accept_chunk(audio[off : off + CHUNK])
    nm.input_finished()
    enc.cache_aware_stream_step = original

    perm = om._meta.get("cache_perm", {})

    def g(key, arr):
        p = perm.get(key)
        return np.ascontiguousarray(np.transpose(arr, p)) if p and list(p) != sorted(p) else arr

    def run(st, fed, length):
        out = om._enc.run(
            None,
            {
                "audio_signal": np.ascontiguousarray(fed, dtype=np.float32),
                "length": np.array([length], dtype=np.int64),
                "cache_last_channel": g("cache_last_channel", st["lc"]),
                "cache_last_time": g("cache_last_time", st["lt"]),
                "cache_last_channel_len": g("cache_last_channel_len", st["ll"]),
            },
        )
        named = dict(zip(om._enc_out_names, out))
        return named.get("outputs", out[0]), int(np.asarray(named.get("encoded_lengths", out[1])).reshape(-1)[0])

    print(f"steps={len(steps)} feat_frames_fed={[s['fed'].shape[-1] for s in steps][:6]}...")
    verdict = {"matches_post_drop": 0, "matches_pre_drop": 0, "errors": 0, "n": 0}
    for i, st in enumerate(steps):
        try:
            enc_out, n = run(st, st["fed"], st["len"])
            status = "ok"
        except Exception as e:
            status = f"ERR {str(e).splitlines()[-1][:60]}"
            enc_out, n = None, -1
        line = (f"step {i:>3} fed={st['fed'].shape[-1]:>3} nemo_drop={st['drop']} "
                f"nemo_n={st['ref_n']:>2} onnx_n={n:>3} {status}")
        if enc_out is not None:
            verdict["n"] += 1
            if n == st["ref_n"]:
                verdict["matches_post_drop"] += 1
                m = min(n, st["ref"].shape[-1])
                line += f" | maxdiff(aligned)={np.abs(enc_out[:, :, :m] - st['ref'][:, :, :m]).max():.4g}"
            elif n == st["ref_n"] + st["drop"]:
                verdict["matches_pre_drop"] += 1
                m = st["ref_n"]
                d = st["drop"]
                line += f" | maxdiff(after wrapper-drop {d})={np.abs(enc_out[:, :, d:d+m] - st['ref'][:, :, :m]).max():.4g}"
        else:
            verdict["errors"] += 1
        print(line)

        # step 0 alternative: Nemotron-style zero-padded pre-encode cache
        if i == 0:
            pre = om._pre_cache_pair[1]
            fed2 = np.concatenate(
                [np.zeros((1, st["fed"].shape[1], pre), dtype=np.float32), st["fed"]], axis=-1
            )
            try:
                e2, n2 = run(st, fed2, st["len"] + pre)
                print(f"step   0 ALT zero-pad pre={pre} fed={fed2.shape[-1]} -> onnx_n={n2} (nemo_n={st['ref_n']})")
                m = min(n2, st["ref"].shape[-1])
                if m:
                    print(f"         maxdiff vs nemo first {m}: "
                          f"{np.abs(e2[:, :, :m] - st['ref'][:, :, :m]).max():.4g}")
            except Exception as e:
                print(f"step   0 ALT zero-pad FAILED: {str(e).splitlines()[-1][:80]}")

    print("\nVERDICT:", verdict)
    if verdict["matches_post_drop"] > verdict["matches_pre_drop"]:
        print("=> graph applies drop_extra itself  -> wrapper drop_policy=none")
    elif verdict["matches_pre_drop"] > 0:
        print("=> graph returns pre-drop frames    -> wrapper drop_policy=nemo")
    else:
        print("=> inconclusive; inspect the per-step lines")


if __name__ == "__main__":
    main()
