#!/usr/bin/env python3
"""Export the parakeet Arm A checkpoint to a self-contained ONNX submission tree.

POD-ONLY (needs NeMo + torch). Produces everything track2_starting_kit/parakeet_onnx/
needs to run with no NeMo, no HuggingFace and no network:

    <out>/weights/encoder_model.onnx[.data]   cache-aware streaming encoder
    <out>/weights/decoder_model.onnx[.data]   prediction net + joint
    <out>/weights/tokens.txt                  "<piece> <id>" lines
    <out>/weights/filterbank.npy              NeMo's own mel filterbank
    <out>/weights/window.npy                  NeMo's own analysis window
    <out>/weights/streaming_meta.json         geometry + cache shapes + preproc params
    <out>/{model.py,localmel.py,config.yaml,setup.sh,requirements.txt}

Why the meta file: the Nemotron submission hard-coded chunk/cache/vocab/mel constants in
model.py, so the wrapper could disagree with the graph and nothing would notice. Here
every such number is read out of the checkpoint at export time and shipped beside the
graph.

Usage (pod):
    python3 scripts/export_parakeet_onnx.py \
        --nemo /workspace/parakeet_ft/armA_full/ft_smoke_encoder_only.nemo \
        --src-dir  track2_starting_kit/parakeet_onnx \
        --out-dir  /workspace/parakeet_onnx_fp32 \
        --att-context 70 1

Then: scripts/parity_parakeet_onnx.py (fidelity gate) -> scripts/quantize_nemotron_encoder.py
(int8, generic over any submission tree with weights/{encoder,decoder}_model.onnx) ->
scripts/package_parakeet_onnx.py (zip + structural asserts).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

SUPPORTED_NORMALIZE = (None, "None", "NA", "na", "")


def _resolve_log_guard(value):
    """NeMo allows a float, 'tiny' or 'eps'. Resolve to the float the graph will use."""
    import torch

    if isinstance(value, str):
        if value == "tiny":
            return float(torch.finfo(torch.float32).tiny)
        if value == "eps":
            return float(torch.finfo(torch.float32).eps)
        raise RuntimeError(f"unsupported log_zero_guard_value={value!r}")
    return float(value)


def preprocessor_meta(model) -> dict:
    """Everything localmel.py needs to reproduce the mel front-end, taken from the
    checkpoint's own preprocessor cfg. Anything localmel does not implement fails HERE,
    at export, rather than silently producing wrong features at inference."""
    pp = model.cfg.preprocessor
    sr = int(pp.get("sample_rate", 16000))
    normalize = pp.get("normalize", "per_feature")
    if normalize not in SUPPORTED_NORMALIZE:
        raise RuntimeError(
            f"preprocessor normalize={normalize!r}: localmel.py only reproduces "
            "un-normalised features. Implement per-window normalisation before exporting."
        )
    if pp.get("log_zero_guard_type", "add") != "add":
        raise RuntimeError(f"log_zero_guard_type={pp.get('log_zero_guard_type')!r} unsupported")
    if pp.get("window", "hann") != "hann":
        raise RuntimeError(f"window={pp.get('window')!r} unsupported")
    if bool(pp.get("exact_pad", False)):
        raise RuntimeError("exact_pad=True unsupported (localmel uses center=True padding)")
    if not bool(pp.get("log", True)):
        raise RuntimeError("log=False unsupported")

    meta = {
        "sample_rate": sr,
        "features": int(pp.get("features", 128)),
        "n_fft": int(pp.get("n_fft", 512)),
        "win_length": int(round(float(pp.get("window_size", 0.025)) * sr)),
        "hop_length": int(round(float(pp.get("window_stride", 0.01)) * sr)),
        "window": pp.get("window", "hann"),
        "preemph": pp.get("preemph", 0.97),
        "mag_power": float(pp.get("mag_power", 2.0)),
        "log_zero_guard_value": _resolve_log_guard(pp.get("log_zero_guard_value", 2 ** -24)),
        "log_zero_guard_type": pp.get("log_zero_guard_type", "add"),
        "normalize": normalize,
        "center": True,
        "log": True,
        # recorded for the record only — the export forces them off in the reference path
        "dither_in_ckpt": float(pp.get("dither", 0.0)),
        "pad_to_in_ckpt": int(pp.get("pad_to", 0)),
    }

    # Padding behaviour, read from the FEATURIZER OBJECT rather than the cfg: NeMo pads the
    # STFT with zeros ("constant", not reflect) and masks every frame at/after
    # get_seq_len(n) to pad_value. Both are invisible in cfg and both change real output —
    # reflect padding moves the first frames by ~3.5, and the unmasked tail frame differs
    # by |log(log_zero_guard)| = 16.6 on every utterance.
    feat = model.preprocessor.featurizer
    meta["pad_mode"] = "constant"
    meta["pad_value"] = float(getattr(feat, "pad_value", 0.0))
    meta["stft_pad_amount"] = getattr(feat, "stft_pad_amount", None)
    if getattr(feat, "frame_splicing", 1) != 1:
        raise RuntimeError(f"frame_splicing={feat.frame_splicing} unsupported")
    return meta


def dump_filterbank(model, weights: Path, meta: dict) -> None:
    """Write NeMo's own mel filterbank + analysis window so localmel is not a
    re-derivation of them (a re-derived filterbank is a silent accuracy bug)."""
    import numpy as np

    feat = model.preprocessor.featurizer
    fb = getattr(feat, "filter_banks", None)
    win = getattr(feat, "window", None)
    if fb is None or win is None:
        raise RuntimeError(
            f"preprocessor.featurizer lacks filter_banks/window "
            f"(has: {sorted(k for k in dir(feat) if not k.startswith('_'))[:40]})"
        )
    fb = fb.detach().cpu().numpy().astype("float32").squeeze()
    win = win.detach().cpu().numpy().astype("float32")
    exp_fb = (meta["features"], meta["n_fft"] // 2 + 1)
    if fb.shape != exp_fb:
        raise RuntimeError(f"filterbank shape {fb.shape} != expected {exp_fb}")
    if win.shape != (meta["win_length"],):
        raise RuntimeError(f"window shape {win.shape} != expected {(meta['win_length'],)}")
    np.save(weights / "filterbank.npy", fb)
    np.save(weights / "window.npy", win)


def dump_tokens(model, weights: Path) -> int:
    """tokens.txt in the "<piece> <id>" format model.py's _load_tokens expects."""
    tok = model.tokenizer
    sp = getattr(tok, "tokenizer", None)
    lines = []
    if sp is not None and hasattr(sp, "get_piece_size"):
        for i in range(sp.get_piece_size()):
            lines.append(f"{sp.id_to_piece(i)} {i}")
    elif hasattr(tok, "vocab"):
        for i, piece in enumerate(tok.vocab):
            lines.append(f"{piece} {i}")
    else:
        raise RuntimeError(f"cannot enumerate vocabulary from tokenizer {type(tok)}")
    blank_id = len(lines)  # RNN-T blank is appended after the BPE vocabulary
    lines.append(f"<blk> {blank_id}")
    (weights / "tokens.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return blank_id


def align_to_graph(name: str, runtime_shape: list[int], graph_dims: list) -> list[int]:
    """Return the runtime cache shape expressed in the EXPORTED GRAPH's dim order.

    NeMo's runtime and NeMo's exporter do not agree on cache layout: at runtime
    `get_initial_cache_state(batch_size=1)` returns LAYER-first tensors ([17,1,70,512]),
    while the exported graph declares BATCH-first ([dyn,17,dyn,512]). Feeding the runtime
    order to ORT fails with "invalid dimensions"; worse, a wrong-but-accepted order would
    decode silently wrong, because zero-filled caches are content-symmetric.

    Rule: keep the identity order if it satisfies every STATIC graph dim; otherwise take
    the unique permutation that does, with batch (size 1) at dim 0. Ambiguity raises —
    never pick one and hope.
    """
    import itertools

    static = {i: int(d) for i, d in enumerate(graph_dims) if isinstance(d, int)}

    def ok(shape):
        return all(shape[i] == v for i, v in static.items())

    if len(runtime_shape) != len(graph_dims):
        raise RuntimeError(f"{name}: rank {len(runtime_shape)} != graph rank {len(graph_dims)}")
    if ok(runtime_shape):
        return list(runtime_shape), list(range(len(runtime_shape)))
    # search over index permutations (not value permutations — dims can repeat)
    cands = {
        perm
        for perm in itertools.permutations(range(len(runtime_shape)))
        if ok([runtime_shape[i] for i in perm]) and runtime_shape[perm[0]] == 1
    }
    shapes = {tuple(runtime_shape[i] for i in p) for p in cands}
    if len(shapes) != 1:
        raise RuntimeError(
            f"{name}: cannot align runtime shape {runtime_shape} to graph dims {graph_dims} "
            f"({len(shapes)} candidate orders: {sorted(shapes)})"
        )
    perm = sorted(cands)[0]
    aligned = [runtime_shape[i] for i in perm]
    print(f"[export] {name}: runtime {runtime_shape} -> graph order {aligned} (perm {list(perm)})")
    return aligned, list(perm)


def streaming_meta(model) -> dict:
    sc = getattr(model.encoder, "streaming_cfg", None)
    if sc is None:
        raise RuntimeError("encoder has no streaming_cfg — not a cache-aware model")

    def _get(name):
        val = getattr(sc, name, None)
        if val is None:
            return None
        if isinstance(val, (list, tuple)):
            return [int(v) for v in val]
        return int(val)

    sampling = None
    pre_encode = getattr(model.encoder, "pre_encode", None)
    if pre_encode is not None and hasattr(pre_encode, "get_sampling_frames"):
        sampling = pre_encode.get_sampling_frames()
        if isinstance(sampling, (list, tuple)):
            sampling = [int(v) for v in sampling]
        else:
            sampling = int(sampling)

    return {
        "chunk_size": _get("chunk_size"),
        "shift_size": _get("shift_size"),
        "pre_encode_cache_size": _get("pre_encode_cache_size"),
        "cache_drop_size": _get("cache_drop_size"),
        "last_channel_cache_size": _get("last_channel_cache_size"),
        "valid_out_len": _get("valid_out_len"),
        "drop_extra_pre_encoded": _get("drop_extra_pre_encoded"),
        "sampling_frames": sampling,
    }


def export_onnx(model, work: Path) -> tuple[Path, Path]:
    """NeMo splits a multi-module model into one ONNX per module. cache_support=True is
    what makes the encoder graph take/return the streaming caches (same call as
    scripts/export_deploy.sh, which produced the Nemotron graphs that scored)."""
    work.mkdir(parents=True, exist_ok=True)
    model.set_export_config({"cache_support": "True"})
    model.export(str(work / "model.onnx"))
    files = sorted(p for p in work.iterdir() if p.suffix == ".onnx")
    enc = [p for p in files if "encoder" in p.name]
    dec = [p for p in files if "decoder" in p.name or "joint" in p.name]
    if len(enc) != 1 or len(dec) != 1:
        raise RuntimeError(f"unexpected export output: {[p.name for p in files]}")
    return enc[0], dec[0]


def rehome(src: Path, dst: Path) -> None:
    """Copy an ONNX graph to its shipped name, re-homing external weights to
    <dst>.data. Renaming the file alone would break the external-data pointers."""
    import onnx

    model = onnx.load(str(src), load_external_data=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    data_name = dst.name + ".data"
    for stale in (dst, dst.parent / data_name):
        if stale.exists():
            stale.unlink()
    onnx.save(
        model,
        str(dst),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_name,
        size_threshold=1024,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nemo", type=Path, required=True, help="Arm A .nemo checkpoint")
    ap.add_argument("--src-dir", type=Path, required=True, help="track2_starting_kit/parakeet_onnx")
    ap.add_argument("--out-dir", type=Path, required=True, help="submission tree to build")
    ap.add_argument("--work-dir", type=Path, default=None, help="scratch for raw NeMo export")
    ap.add_argument("--att-context", type=int, nargs=2, default=[70, 1])
    args = ap.parse_args()

    import torch
    import nemo.collections.asr as nemo_asr

    out = args.out_dir.resolve()
    weights = out / "weights"
    work = (args.work_dir or (out.parent / (out.name + "_raw"))).resolve()
    weights.mkdir(parents=True, exist_ok=True)

    print(f"[export] restoring {args.nemo}")
    model = nemo_asr.models.ASRModel.restore_from(str(args.nemo), map_location="cpu").eval()

    # Deploy context MUST equal the training/eval context. research/19 own-goal: a
    # [70,1]-trained model deployed at [70,6] silently loses accuracy.
    model.encoder.set_default_att_context_size(list(args.att_context))
    model.encoder.setup_streaming_params()
    if not hasattr(model, "conformer_stream_step"):
        raise RuntimeError("checkpoint is not cache-aware streaming — wrong model")

    pp_meta = preprocessor_meta(model)
    dump_filterbank(model, weights, pp_meta)
    blank_id = dump_tokens(model, weights)

    # Cross-check the blank id against the model's own heads. A blank id that is off by
    # one decodes to fluent-looking garbage, so this must fail at export, not at eval.
    model_blank = getattr(model.decoder, "blank_idx", None)
    joint_classes = getattr(model.joint, "num_classes_with_blank", None)
    if model_blank is not None and int(model_blank) != blank_id:
        raise RuntimeError(f"decoder.blank_idx={model_blank} != tokens.txt blank {blank_id}")
    if joint_classes is not None and int(joint_classes) != blank_id + 1:
        raise RuntimeError(f"joint.num_classes_with_blank={joint_classes} != {blank_id + 1}")

    cache_lc, cache_lt, cache_ll = model.encoder.get_initial_cache_state(batch_size=1)
    pred_states = model.decoder.initialize_state(torch.zeros(1, 1, 1))

    dec_cfg = model.cfg.decoding
    greedy_cfg = dec_cfg.get("greedy", {}) or {}
    meta = {
        "source_checkpoint": str(args.nemo),
        "model_class": type(model).__name__,
        "att_context_size": list(args.att_context),
        "streaming_cfg": streaming_meta(model),
        "feat_in": int(getattr(model.encoder, "_feat_in", pp_meta["features"])),
        "blank_id": blank_id,
        "vocab_size": blank_id + 1,
        "max_symbols_per_step": int(greedy_cfg.get("max_symbols", 10) or 10),
        "runtime_cache_shapes": {  # NeMo's own order, kept for the record / parity replay
            "cache_last_channel": list(cache_lc.shape),
            "cache_last_time": list(cache_lt.shape),
            "cache_last_channel_len": list(cache_ll.shape),
        },
        "pred_state_shapes": [list(s.shape) for s in pred_states],
        "preprocessor": pp_meta,
    }
    print("[export] meta:", json.dumps(meta, indent=2)[:2000])

    enc_src, dec_src = export_onnx(model, work)
    print(f"[export] raw graphs: {enc_src.name}, {dec_src.name}")
    rehome(enc_src, weights / "encoder_model.onnx")
    rehome(dec_src, weights / "decoder_model.onnx")

    for name in ("model.py", "localmel.py", "config.yaml", "setup.sh", "requirements.txt"):
        shutil.copy2(args.src_dir / name, out / name)
    (out / "setup.sh").chmod(0o755)

    # Record the ONNX I/O signature next to the geometry so a graph/wrapper mismatch is
    # visible in the artifact itself, not only at load time.
    import onnxruntime as ort

    for tag, path in (("encoder_io", weights / "encoder_model.onnx"), ("decoder_io", weights / "decoder_model.onnx")):
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        meta[tag] = {
            "inputs": [{"name": i.name, "shape": list(i.shape), "type": i.type} for i in sess.get_inputs()],
            "outputs": [{"name": o.name, "shape": list(o.shape), "type": o.type} for o in sess.get_outputs()],
        }

    # Cache shapes are recorded in the GRAPH's dim order, not NeMo's runtime order — the
    # two disagree (see align_to_graph). model.py allocates from these, so they must match
    # what the session will accept.
    enc_dims = {i["name"]: i["shape"] for i in meta["encoder_io"]["inputs"]}
    dec_dims = {i["name"]: i["shape"] for i in meta["decoder_io"]["inputs"]}
    rt = meta["runtime_cache_shapes"]
    perms = {}
    for key in ("cache_last_channel", "cache_last_time", "cache_last_channel_len"):
        meta[f"{key}_shape"], perms[key] = align_to_graph(key, rt[key], enc_dims[key])
    state_names = sorted(n for n in dec_dims if n.startswith("input_states_"))
    aligned_states = []
    for name, shape in zip(state_names, meta["pred_state_shapes"]):
        shp, perm = align_to_graph(name, shape, dec_dims[name])
        aligned_states.append(shp)
        perms[name] = perm
    meta["pred_state_shapes"] = aligned_states
    # Permutations parity needs to replay NeMo-recorded tensors through ORT.
    meta["cache_perm"] = perms

    (weights / "streaming_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[export] wrote {out}")
    for p in sorted(out.rglob("*")):
        if p.is_file():
            print(f"  {p.stat().st_size/1e6:9.1f} MB  {p.relative_to(out)}")


if __name__ == "__main__":
    main()
