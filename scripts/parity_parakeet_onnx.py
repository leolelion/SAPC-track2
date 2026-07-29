#!/usr/bin/env python3
"""ONNX-vs-NeMo fidelity gate for the parakeet Arm A export. POD-ONLY (needs both).

This is the gate that stands between "we exported a graph" and "we may quote a CER from
it". It compares the two implementations at three increasingly integrated levels, so a
mismatch is localised instead of showing up as an unexplained CER delta:

  feat  localmel.py vs NeMo's mel front-end on real audio                  (tensor)
  enc   ORT encoder vs NeMo encoder.cache_aware_stream_step, step by step  (tensor)
        -> also SWEEPS the (drop_policy, trim_policy) 2x2 and reports which pair
           reproduces NeMo's frame count/values. This is how config.yaml's
           encoder.{drop,trim}_policy get set — by measurement, not by argument.
  text  full ONNX wrapper vs full NeMo wrapper, 100 ms chunking, N utts    (transcript)

Each stage compares against the NeMo sibling `track2_starting_kit/parakeet_realtime_ft`,
which is the implementation whose numbers are banked (Dev_clean2k 13.51%, severe 18.69%).

Usage (pod):
    python3 scripts/parity_parakeet_onnx.py \
        --onnx-dir /workspace/parakeet_onnx_fp32 \
        --nemo-dir track2_starting_kit/parakeet_realtime_ft \
        --wavs /workspace/data/parity/*.wav \
        --stages feat enc text --out-json /workspace/artifacts/parity.json

Exit code 0 iff every requested stage passes its tolerance. A failure here means DO NOT
package — never "submit to test a hypothesis".
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

CHUNK = 1600  # 100 ms at 16 kHz, as local_decode.py delivers


def load_module(path: Path, name: str):
    """Import a submission's model.py with its own dir on sys.path (local_decode.py does
    the same, and model.py imports localmel as a top-level module)."""
    sys.path.insert(0, str(path.parent.resolve()))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_wav(path: Path) -> np.ndarray:
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32")
    if sr != 16000:
        raise RuntimeError(f"{path}: sample rate {sr} != 16000")
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype(np.float32)
    return np.ascontiguousarray(audio, dtype=np.float32)


def stream(model, audio: np.ndarray) -> str:
    """Drive a wrapper exactly as local_decode.py does: reset, 100 ms chunks, finish."""
    model.reset()
    for off in range(0, len(audio), CHUNK):
        model.accept_chunk(audio[off : off + CHUNK])
    return model.input_finished()


# ---------------------------------------------------------------- feat
def stage_feat(nemo_model, onnx_model, wavs, tol) -> dict:
    """localmel vs the NeMo preprocessor, un-normalised, on real audio."""
    import torch

    worst = 0.0
    per_file = []
    for w in wavs:
        audio = read_wav(w)
        # the MASKED path the wrapper actually feeds the encoder (NeMo zeroes every frame
        # at/after seq_len), not the raw extractor
        onnx_model.reset()
        ours = onnx_model._extract_features(audio)[0]
        wav_t = torch.from_numpy(audio).unsqueeze(0)
        with torch.inference_mode():
            ref, _ = nemo_model._raw_preprocessor(
                input_signal=wav_t, length=torch.tensor([len(audio)])
            )
        ref = ref[0].numpy()
        n = min(ours.shape[-1], ref.shape[-1])
        diff = float(np.abs(ours[:, :n] - ref[:, :n]).max()) if n else float("inf")
        worst = max(worst, diff)
        per_file.append(
            {"wav": w.name, "frames_ours": int(ours.shape[-1]), "frames_nemo": int(ref.shape[-1]), "max_abs_diff": diff}
        )
    return {"max_abs_diff": worst, "tol": tol, "pass": worst <= tol, "per_file": per_file}


# ---------------------------------------------------------------- enc
def stage_enc(nemo_model, onnx_model, wavs, tol) -> dict:
    """Record every encoder call the NeMo wrapper makes, replay it through ORT.

    Per-step ISOLATED comparison: ORT is fed the caches NeMo had at that step, so a
    mismatch is attributed to the step that produced it instead of to cache drift.
    """
    import inspect

    import torch

    encoder = nemo_model.model.encoder
    original = encoder.cache_aware_stream_step
    sig = inspect.signature(original)
    steps: list[dict] = []

    def recorder(*a, **kw):
        bound = sig.bind(*a, **kw)
        bound.apply_defaults()
        args = bound.arguments
        out = original(*a, **kw)
        encoded, encoded_len = out[0], out[1]
        steps.append(
            {
                "fed": args["processed_signal"].detach().cpu().numpy().astype(np.float32),
                "length": int(args["processed_signal_length"].detach().cpu().numpy().reshape(-1)[0]),
                "cache_lc": args["cache_last_channel"].detach().cpu().numpy().astype(np.float32),
                "cache_lt": args["cache_last_time"].detach().cpu().numpy().astype(np.float32),
                "cache_ll": args["cache_last_channel_len"].detach().cpu().numpy(),
                "drop_extra": int(args.get("drop_extra_pre_encoded") or 0),
                "keep_all": bool(args.get("keep_all_outputs", True)),
                "ref_encoded": encoded.detach().cpu().numpy().astype(np.float32),
                "ref_len": int(encoded_len.detach().cpu().numpy().reshape(-1)[0]),
            }
        )
        return out

    encoder.cache_aware_stream_step = recorder
    try:
        for w in wavs:
            stream(nemo_model, read_wav(w))
    finally:
        encoder.cache_aware_stream_step = original

    if not steps:
        return {"pass": False, "error": "NeMo wrapper made no encoder calls"}

    # Replay through ORT once per step; the policy sweep is applied to the SAME raw
    # encoder output, so the 2x2 costs nothing extra.
    grid = {}
    for drop_policy in ("nemo", "none"):
        for trim_policy in ("nemo", "none"):
            grid[f"drop={drop_policy},trim={trim_policy}"] = {
                "frame_mismatch": 0,
                "max_abs_diff": 0.0,           # steps >= 1
                "first_step_max_abs_diff": 0.0,  # step 0 only (known zero-pad difference)
            }

    # NeMo's runtime cache layout is LAYER-first while the exported graph is BATCH-first,
    # so recorded tensors must be permuted before replay (the exporter records the
    # permutation it used; identity when the two already agree).
    perm = onnx_model._meta.get("cache_perm", {})

    def to_graph(key, arr):
        p = perm.get(key)
        return np.ascontiguousarray(np.transpose(arr, p)) if p and list(p) != sorted(p) else arr

    raw_diff = 0.0
    errors = []
    for i, st in enumerate(steps):
        fed, length = st["fed"], st["length"]
        # Step 0: NeMo feeds a bare chunk with drop_extra=0, but the exported graph drops
        # unconditionally and underflows on it (measured: scripts/probe_parakeet_enc.py).
        # The wrapper pads the step-0 pre-encode cache with zeros, so replay it that way —
        # otherwise this stage tests an input the shipped wrapper never produces.
        if st["drop_extra"] == 0 and onnx_model._first_step_pad:
            pre = onnx_model._pre_cache_pair[1]
            fed = np.concatenate(
                [np.zeros((fed.shape[0], fed.shape[1], pre), dtype=np.float32), fed], axis=-1
            )
            length += pre
        try:
            ort_out = onnx_model._enc.run(
                None,
                {
                    "audio_signal": np.ascontiguousarray(fed, dtype=np.float32),
                    "length": np.array([length], dtype=np.int64),
                    "cache_last_channel": to_graph("cache_last_channel", st["cache_lc"]),
                    "cache_last_time": to_graph("cache_last_time", st["cache_lt"]),
                    "cache_last_channel_len": to_graph("cache_last_channel_len", st["cache_ll"]),
                },
            )
        except Exception as exc:  # record and continue: one bad step must not hide the rest
            errors.append({"step": i, "fed_frames": int(fed.shape[-1]), "error": str(exc).splitlines()[-1][:200]})
            continue
        named = dict(zip(onnx_model._enc_out_names, ort_out))
        encoded = named.get("outputs", ort_out[0])
        n_enc = int(np.asarray(named.get("encoded_lengths", ort_out[1])).reshape(-1)[0])
        ref = st["ref_encoded"]
        ref_n = st["ref_len"]

        is_last = i == len(steps) - 1
        first = st["drop_extra"] == 0
        for drop_policy in ("nemo", "none"):
            for trim_policy in ("nemo", "none"):
                onnx_model._drop_policy = drop_policy
                onnx_model._trim_policy = trim_policy
                onnx_model._step = 0 if first else 1  # emulate step>0 iff NeMo dropped
                start, count = onnx_model._encoder_out_window(n_enc, encoded.shape[-1], is_last)
                key = f"drop={drop_policy},trim={trim_policy}"
                if count != ref_n:
                    grid[key]["frame_mismatch"] += 1
                    continue
                got = encoded[:, :, start : start + count]
                cmp_n = min(got.shape[-1], ref.shape[-1])
                d = float(np.abs(got[:, :, :cmp_n] - ref[:, :, :cmp_n]).max()) if cmp_n else float("inf")
                # Step 0 is EXPECTED to differ: NeMo sees no left context there, the graph
                # sees the zero pad we must feed it. Track it separately so it neither hides
                # a real regression nor fails the tensor tolerance for a known cause.
                bucket = "first_step_max_abs_diff" if first else "max_abs_diff"
                grid[key][bucket] = max(grid[key].get(bucket, 0.0), d)
        cmp_n = min(encoded.shape[-1], ref.shape[-1])
        if cmp_n:
            raw_diff = max(raw_diff, float(np.abs(encoded[:, :, :cmp_n] - ref[:, :, :cmp_n]).max()))

    best = min(
        grid.items(),
        key=lambda kv: (kv[1]["frame_mismatch"], kv[1]["max_abs_diff"]),
    )
    return {
        "steps": len(steps),
        "steps_run": len(steps) - len(errors),
        "step_errors": errors,
        "policy_grid": grid,
        "best_policy": best[0],
        "best": best[1],
        "untrimmed_max_abs_diff": raw_diff,
        "tol": tol,
        "pass": (
            not errors
            and best[1]["frame_mismatch"] == 0
            and best[1]["max_abs_diff"] <= tol
        ),
    }


# ---------------------------------------------------------------- text
def stage_text(nemo_model, onnx_model, wavs, min_exact) -> dict:
    rows = []
    exact = 0
    for w in wavs:
        audio = read_wav(w)
        ref = stream(nemo_model, audio)
        got = stream(onnx_model, audio)
        ok = ref.strip() == got.strip()
        exact += int(ok)
        rows.append({"wav": w.name, "nemo": ref, "onnx": got, "exact": ok})
    rate = exact / len(wavs) if wavs else 0.0
    return {"n": len(wavs), "exact": exact, "exact_rate": rate, "min_exact_rate": min_exact,
            "pass": rate >= min_exact, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-dir", type=Path, required=True)
    ap.add_argument("--nemo-dir", type=Path, required=True)
    ap.add_argument("--wavs", type=Path, nargs="+", required=True)
    ap.add_argument("--stages", nargs="+", default=["feat", "enc", "text"])
    ap.add_argument("--feat-tol", type=float, default=1e-3)
    ap.add_argument("--enc-tol", type=float, default=1e-2)
    ap.add_argument("--min-exact-rate", type=float, default=0.9)
    ap.add_argument("--out-json", type=Path)
    args = ap.parse_args()

    wavs = [w for w in args.wavs if w.is_file()]
    if not wavs:
        raise SystemExit("no wav files given")

    nemo_mod = load_module(args.nemo_dir / "model.py", "parakeet_nemo_model")
    onnx_mod = load_module(args.onnx_dir / "model.py", "parakeet_onnx_model")
    print(f"[parity] loading NeMo wrapper from {args.nemo_dir}")
    nemo_model = nemo_mod.Model()
    print(f"[parity] loading ONNX wrapper from {args.onnx_dir}")
    onnx_model = onnx_mod.Model()

    results = {"wavs": [w.name for w in wavs], "stages": {}}
    if "feat" in args.stages:
        results["stages"]["feat"] = stage_feat(nemo_model, onnx_model, wavs, args.feat_tol)
    if "enc" in args.stages:
        results["stages"]["enc"] = stage_enc(nemo_model, onnx_model, wavs, args.enc_tol)
        # restore the shipped policy before the text stage so it tests what we package
        onnx_model._drop_policy, onnx_model._trim_policy = (
            results["stages"]["enc"]["best_policy"].split(",")[0].split("=")[1],
            results["stages"]["enc"]["best_policy"].split(",")[1].split("=")[1],
        )
        print(f"[parity] using best policy for text stage: {results['stages']['enc']['best_policy']}")
    if "text" in args.stages:
        results["stages"]["text"] = stage_text(nemo_model, onnx_model, wavs, args.min_exact_rate)

    failed = [k for k, v in results["stages"].items() if not v.get("pass")]
    results["failed_stages"] = failed
    blob = json.dumps(results, indent=2)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(blob, encoding="utf-8")
    print(blob[:4000])
    if failed:
        raise SystemExit(f"PARITY FAILED: {failed}")
    print("PARITY OK")


if __name__ == "__main__":
    main()
