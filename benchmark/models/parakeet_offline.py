#!/usr/bin/env python3
"""
Parakeet OFFLINE accuracy-ceiling runner (NOT a Track 2 streaming Model).

Purpose (Stage A of docs/benchmark_plan.md): measure the CER *ceiling* the
offline Parakeet checkpoints reach on dysarthric Dev via NeMo `transcribe()`.
This answers "is the NVIDIA acoustic model even more accurate than the
zipformer?" cheaply, WITHOUT any (forbidden) buffered fake-streaming.

Why this is offline-only: Parakeet-CTC/RNNT/TDT are full-context FastConformer
models — they do not support real cache-aware streaming (verified 2026-07-13;
see docs/repo_analysis.md and memory `parakeet-offline-not-streaming`). So this
runner deliberately produces an accuracy CSV only; it has no accept_chunk /
partial-callback surface and must never be pointed at the streaming pass.

Output: an `id,raw_hypos` CSV compatible with `evaluate.sh` stage 2.

Runs on GPU or CPU wherever NeMo is installed. NeMo is imported lazily so this
file import-checks on a box without NeMo (the dev box).
"""

import argparse
import csv
from pathlib import Path
from typing import List, Tuple


# Verified checkpoint names (HF, 2026-07-13). Confirm each resolves on the pod.
KNOWN_CHECKPOINTS = (
    "nvidia/parakeet-rnnt-0.6b",
    "nvidia/parakeet-tdt-0.6b-v2",
    "nvidia/parakeet-tdt-0.6b-v3",
    "nvidia/parakeet-rnnt-1.1b",
    "nvidia/parakeet-tdt-1.1b",
)


class ParakeetOfflineRunner:
    """Loads a pretrained NeMo ASR model once and transcribes whole files."""

    def __init__(self, checkpoint: str, device: str = "cuda", beam_size: int = 1):
        # Lazy import: keep this module import-safe where NeMo is absent.
        import torch  # noqa: F401
        import nemo.collections.asr as nemo_asr

        print(f"[parakeet_offline] loading {checkpoint} on {device} …")
        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=checkpoint)
        self.model = self.model.to(device)
        self.model.eval()

        # Optional beam search (mirrors track1/parakeet baseline). beam_size=1
        # keeps it greedy (fast) for a first ceiling read.
        if beam_size and beam_size > 1 and hasattr(self.model, "change_decoding_strategy"):
            cfg = self.model.cfg
            cfg.decoding.strategy = "beam"
            cfg.decoding.beam.beam_size = beam_size
            cfg.decoding.beam.return_best_hypothesis = True
            self.model.change_decoding_strategy(cfg.decoding)
        print("[parakeet_offline] model ready")

    def transcribe_paths(self, wav_paths: List[str]) -> List[str]:
        outputs = self.model.transcribe(wav_paths, verbose=False)
        texts: List[str] = []
        for o in outputs or []:
            if isinstance(o, str):
                texts.append(o.strip())
            else:
                texts.append((getattr(o, "text", "") or "").strip())
        return texts


def _load_manifest(manifest_csv: Path, data_root: Path) -> List[Tuple[str, str]]:
    """Return [(id, absolute_wav_path)] — same id/path convention as local_decode."""
    entries: List[Tuple[str, str]] = []
    with open(manifest_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            entries.append((row["id"], str(data_root / row["audio_filepath"])))
    return entries


def run(
    checkpoint: str,
    manifest_csv: Path,
    data_root: Path,
    out_csv: Path,
    device: str = "cuda",
    beam_size: int = 1,
    limit: int = 0,
) -> Path:
    entries = _load_manifest(manifest_csv, data_root)
    if limit and limit > 0:
        entries = entries[:limit]  # smoke slice
    print(f"[parakeet_offline] transcribing {len(entries)} utts from {manifest_csv}")

    runner = ParakeetOfflineRunner(checkpoint, device=device, beam_size=beam_size)
    ids = [uid for uid, _ in entries]
    hypos = runner.transcribe_paths([p for _, p in entries])

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "raw_hypos"])
        for uid, hyp in zip(ids, hypos):
            w.writerow([uid, hyp])
    print(f"[parakeet_offline] wrote {out_csv}")
    return out_csv


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help=f"one of {KNOWN_CHECKPOINTS} or a .nemo path")
    ap.add_argument("--manifest-csv", required=True, type=Path)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--out-csv", required=True, type=Path)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--beam-size", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="smoke: only first N utts (0=all)")
    args = ap.parse_args()
    run(
        args.checkpoint,
        args.manifest_csv,
        args.data_root,
        args.out_csv,
        device=args.device,
        beam_size=args.beam_size,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
