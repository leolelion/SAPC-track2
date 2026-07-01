#!/usr/bin/env python3
"""Build the Nemotron runtime-fix submission zip from an extracted package.

This avoids staging a second 900 MB tree. It writes the patched model.py as
`model.py`, then streams the remaining required submission files from the
source extraction.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path


REQUIRED_FILES = ("localmel.py", "config.yaml", "setup.sh")
REQUIRED_DIRS = ("weights", "wheels")


def _iter_files(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def _add_file(zf: zipfile.ZipFile, source: Path, arcname: str) -> None:
    zf.write(source, arcname)
    info = zf.getinfo(arcname)
    if source.name == "setup.sh":
        info.external_attr = (0o755 & 0xFFFF) << 16


def _dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in _iter_files(path))


def _check_inputs(source_dir: Path, patched_model: Path) -> None:
    missing: list[str] = []
    if not source_dir.is_dir():
        missing.append(str(source_dir))
    if not patched_model.is_file():
        missing.append(str(patched_model))
    for name in REQUIRED_FILES:
        if not (source_dir / name).is_file():
            missing.append(str(source_dir / name))
    for name in REQUIRED_DIRS:
        if not (source_dir / name).is_dir():
            missing.append(str(source_dir / name))
    if missing:
        raise FileNotFoundError("Missing required input(s): " + ", ".join(missing))


def _payload_size(source_dir: Path) -> int:
    total = _dir_size(source_dir / "weights") + _dir_size(source_dir / "wheels")
    total += sum((source_dir / name).stat().st_size for name in REQUIRED_FILES)
    return total


def _check_free_space(output_zip: Path, source_dir: Path, min_free_gb: float) -> tuple[int, int, int]:
    output_parent = output_zip.resolve().parent
    output_parent.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(output_parent)
    # Conservative: original uncompressed payload size is an upper bound plus margin.
    payload = _payload_size(source_dir)
    required = payload
    required += int(min_free_gb * (1024**3))
    if usage.free < required:
        raise RuntimeError(
            f"Not enough free space in {output_parent}: "
            f"free={usage.free / (1024**3):.2f} GiB, "
            f"required_with_margin={required / (1024**3):.2f} GiB. "
            "Choose a larger output volume or lower --min-free-gb deliberately."
        )
    return usage.free, payload, required


def build_zip(
    source_dir: Path,
    patched_model: Path,
    output_zip: Path,
    *,
    min_free_gb: float,
    dry_run: bool,
) -> None:
    source_dir = source_dir.resolve()
    patched_model = patched_model.resolve()
    output_zip = output_zip.resolve()
    _check_inputs(source_dir, patched_model)
    free, payload, required = _check_free_space(output_zip, source_dir, min_free_gb)
    print(
        "package inputs OK: "
        f"free={free / (1024**3):.2f} GiB, "
        f"payload={payload / (1024**3):.2f} GiB, "
        f"required_with_margin={required / (1024**3):.2f} GiB"
    )
    if dry_run:
        return
    tmp_zip = output_zip.with_suffix(output_zip.suffix + ".tmp")
    if tmp_zip.exists():
        tmp_zip.unlink()

    with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        _add_file(zf, patched_model, "model.py")
        for name in REQUIRED_FILES:
            _add_file(zf, source_dir / name, name)
        for dirname in REQUIRED_DIRS:
            for path in _iter_files(source_dir / dirname):
                arcname = path.relative_to(source_dir).as_posix()
                _add_file(zf, path, arcname)

    tmp_zip.replace(output_zip)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True, help="Extracted original Nemotron package")
    parser.add_argument(
        "--patched-model",
        type=Path,
        default=Path("experiments/exp_nemotron_speed_002/model_worker1_runtimefix.py"),
        help="Patched runtime-fix model.py to store as model.py in the zip",
    )
    parser.add_argument("--output-zip", type=Path, required=True)
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=0.75,
        help="Free-space margin to leave after accounting for the uncompressed payload",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs/free space without writing zip")
    args = parser.parse_args()
    build_zip(
        args.source_dir,
        args.patched_model,
        args.output_zip,
        min_free_gb=args.min_free_gb,
        dry_run=args.dry_run,
    )
    if not args.dry_run:
        size_mb = args.output_zip.stat().st_size / (1024**2)
        print(f"wrote {args.output_zip} ({size_mb:.1f} MiB)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
