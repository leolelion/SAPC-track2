#!/usr/bin/env python3
"""Zip the parakeet ONNX submission tree and assert it is offline-valid before upload.

The checks encode the failure modes this project has actually hit:
  * network-dependent setup.sh (the recurring cause of submission death) -> setup.sh must
    install with --no-index and the wheels must be present;
  * a nested root directory (Codabench expects model.py at the TOP level of the zip);
  * missing external ONNX weight files (.onnx.data) -> a graph that loads locally because
    the data file sits outside the zip;
  * a wrapper/graph mismatch -> streaming_meta.json must be present and consistent with
    the ONNX I/O signature;
  * unverified trim policies -> config.yaml must have been set by parity (pass
    --parity-json to enforce it).

Usage (pod):
    python3 scripts/package_parakeet_onnx.py --submission-dir /workspace/parakeet_onnx_int8 \
        --zip /workspace/artifacts/parakeet_armA_int8.zip --parity-json /workspace/artifacts/parity.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

REQUIRED = [
    "model.py",
    "localmel.py",
    "config.yaml",
    "setup.sh",
    "requirements.txt",
    "weights/encoder_model.onnx",
    "weights/decoder_model.onnx",
    "weights/tokens.txt",
    "weights/filterbank.npy",
    "weights/window.npy",
    "weights/streaming_meta.json",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_tree(src: Path, parity: Path | None) -> dict:
    problems = []
    for rel in REQUIRED:
        if not (src / rel).is_file():
            problems.append(f"missing {rel}")

    setup = (src / "setup.sh").read_text(encoding="utf-8") if (src / "setup.sh").is_file() else ""
    if "--no-index" not in setup:
        problems.append("setup.sh does not install with --no-index (offline install unproven)")
    for host in ("huggingface.co", "pypi.org", "ngc.nvidia.com", "from_pretrained"):
        if host in setup:
            problems.append(f"setup.sh references {host} — network dependency")

    wheels = sorted((src / "wheels").glob("*.whl")) if (src / "wheels").is_dir() else []
    if not any("onnxruntime" in w.name for w in wheels):
        problems.append("no onnxruntime wheel bundled under wheels/")
    for w in wheels:
        if "cp311" in w.name and "manylinux" not in w.name:
            problems.append(f"wheel not manylinux: {w.name}")

    req = (src / "requirements.txt").read_text(encoding="utf-8") if (src / "requirements.txt").is_file() else ""
    live = [l.strip() for l in req.splitlines() if l.strip() and not l.strip().startswith("#")]
    if live:
        problems.append(f"requirements.txt would install from the network: {live}")

    meta = {}
    meta_path = src / "weights" / "streaming_meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        enc_inputs = {i["name"] for i in meta.get("encoder_io", {}).get("inputs", [])}
        need = {"audio_signal", "length", "cache_last_channel", "cache_last_time", "cache_last_channel_len"}
        if enc_inputs and not need <= enc_inputs:
            problems.append(f"encoder ONNX inputs {sorted(enc_inputs)} missing {sorted(need - enc_inputs)}")

    # external ONNX payloads must be inside the tree
    for onnx_rel in ("weights/encoder_model.onnx", "weights/decoder_model.onnx"):
        data = src / (onnx_rel + ".data")
        onnx_file = src / onnx_rel
        if onnx_file.is_file() and onnx_file.stat().st_size < 50_000_000 and not data.is_file():
            # small graph file with no sibling .data is only OK if weights are internal
            try:
                import onnx

                m = onnx.load(str(onnx_file), load_external_data=False)
                ext = [
                    e.value
                    for init in m.graph.initializer
                    if init.data_location == onnx.TensorProto.EXTERNAL
                    for e in init.external_data
                    if e.key == "location"
                ]
                for rel in set(ext):
                    if not (onnx_file.parent / rel).is_file():
                        problems.append(f"{onnx_rel} references missing external data {rel}")
            except ImportError:
                problems.append(f"onnx not importable — cannot verify external data for {onnx_rel}")

    if parity is not None:
        p = json.loads(parity.read_text(encoding="utf-8"))
        # feat/enc are HARD gates: they prove the graph and the front-end reproduce NeMo's
        # tensors. `text` is ADVISORY. Greedy RNN-T is chaotic under any numeric difference
        # — a 5e-5 encoder delta flips one argmax and the prediction-net state diverges for
        # the rest of the utterance — so exact-transcript agreement understates fidelity.
        # Measured 2026-07-29 on the official scorer, Dev_diag severe (425 utts): ONNX fp32
        # 18.73% CER vs the banked NeMo Arm A 18.69%, with text agreement only 63%. The
        # authority is the official harness (house rule validate-against-real-harness), so
        # a failing text stage is reported, not blocking.
        hard = [s for s in p.get("failed_stages", []) if s in ("feat", "enc")]
        if hard:
            problems.append(f"parity HARD stages failed {hard} — do not package")
        advisory = [s for s in p.get("failed_stages", []) if s not in ("feat", "enc")]
        if advisory:
            rate = p.get("stages", {}).get("text", {}).get("exact_rate")
            print(f"[package] ADVISORY: parity stage(s) {advisory} below threshold "
                  f"(text exact_rate={rate}); gate on the official harness CER instead.")
        enc = p.get("stages", {}).get("enc", {})
        grid, tol = enc.get("policy_grid", {}), enc.get("tol", 1e-2)
        if grid:
            # ACCEPTABLE = every policy pair that reproduced NeMo's frames within tolerance,
            # not just the argmin. Several pairs can be exactly equivalent: on this model
            # valid_out_len never binds (n_enc == valid_out_len == 2), so trim=nemo and
            # trim=none are the same computation and `best_policy` picks between them
            # arbitrarily. Demanding the argmin string would fail a correct config.
            ok = {
                k for k, v in grid.items()
                if v.get("frame_mismatch", 1) == 0 and v.get("max_abs_diff", 1e9) <= tol
            }
            cfg = (src / "config.yaml").read_text(encoding="utf-8")
            have = (
                f"drop={'nemo' if 'drop_policy: nemo' in cfg else 'none'},"
                f"trim={'nemo' if 'trim_policy: nemo' in cfg else 'none'}"
            )
            if not ok:
                problems.append("parity found NO encoder policy reproducing NeMo — do not package")
            elif have not in ok:
                problems.append(
                    f"config.yaml policies ({have}) are not among the parity-verified pairs {sorted(ok)}"
                )
    return {"problems": problems, "meta": meta, "wheels": [w.name for w in wheels]}


def build_zip(src: Path, out: Path) -> None:
    tmp = out.with_suffix(out.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(src.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            arc = path.relative_to(src).as_posix()
            zf.write(path, arc)
            if arc == "setup.sh":
                zf.getinfo(arc).external_attr = (0o755 & 0xFFFF) << 16
    tmp.replace(out)


def verify_zip(out: Path) -> dict:
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
    problems = [f"missing {r} in zip" for r in REQUIRED if r not in names]
    roots = {n.split("/")[0] for n in names if "/" in n}
    if "model.py" not in names:
        problems.append("model.py is not at the zip ROOT (nested submission dir)")
    return {"entries": len(names), "problems": problems, "top_level_dirs": sorted(roots)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission-dir", type=Path, required=True)
    ap.add_argument("--zip", type=Path, required=True)
    ap.add_argument("--parity-json", type=Path)
    ap.add_argument("--allow-problems", action="store_true", help="report but do not fail (inspection only)")
    args = ap.parse_args()

    src = args.submission_dir.resolve()
    tree = check_tree(src, args.parity_json)
    print(json.dumps({"tree_problems": tree["problems"], "wheels": tree["wheels"]}, indent=2))
    if tree["problems"] and not args.allow_problems:
        raise SystemExit("tree checks failed — not packaging")

    build_zip(src, args.zip)
    zres = verify_zip(args.zip)
    digest = sha256(args.zip)
    args.zip.with_suffix(args.zip.suffix + ".sha256").write_text(
        f"{digest}  {args.zip.name}\n", encoding="utf-8"
    )
    print(json.dumps({"zip": str(args.zip), "bytes": args.zip.stat().st_size, "sha256": digest, **zres}, indent=2))
    if zres["problems"] and not args.allow_problems:
        raise SystemExit("zip checks failed")
    print("PACKAGE OK")


if __name__ == "__main__":
    main()
