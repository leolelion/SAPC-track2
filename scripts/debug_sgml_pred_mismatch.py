#!/usr/bin/env python3
"""Diagnose official two-ref SGML hypothesis reconstruction mismatches.

This is a *diagnostic* helper only. It intentionally does not change scorer
semantics. It mirrors the current ``utils.compute_metrics.parse_sgml_csdi``
logic closely enough to report which utterance(s) make the reconstructed
hypothesis differ between the ref1 and ref2 sclite SGML files.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any


def strip_q(x: str) -> str:
    x = x.strip()
    if len(x) >= 2 and x[0] == '"' and x[-1] == '"':
        return x[1:-1]
    return x


def remove_unk_tokens(text: str, unk_token: str = "unk") -> str:
    return " ".join(tok for tok in text.split() if tok != unk_token)


def parse_sgml_records(path: Path, *, process_unk: bool = True, unk_token: str = "unk") -> list[dict[str, Any]]:
    """Parse PATH SGML records and reconstruct pred/target like compute_metrics."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    records: list[dict[str, Any]] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line.startswith("<PATH"):
            i += 1
            continue

        match = re.search(r'id="([^"]+)"', line)
        if not match:
            i += 1
            continue
        utt_id = match.group(1)
        if utt_id.startswith("(") and utt_id.endswith(")"):
            utt_id = utt_id[1:-1]

        j = i + 1
        while j < len(lines) and (not lines[j].strip() or lines[j].lstrip().startswith("<")):
            j += 1
        if j >= len(lines):
            break
        align_line = lines[j].strip()

        ref_tokens: list[str] = []
        hyp_tokens: list[str] = []
        for seg in align_line.split(":"):
            seg = seg.strip()
            if not seg:
                continue
            op = seg[0]
            rest = seg[2:]
            parts = [p.strip() for p in rest.split(",")]

            if op in ("C", "S"):
                if len(parts) >= 2:
                    ref_tok = strip_q(parts[0])
                    hyp_tok = strip_q(parts[1])
                else:
                    ref_tok = ""
                    hyp_tok = ""
                if process_unk and op == "S" and ref_tok == unk_token and hyp_tok:
                    ref_tok = hyp_tok
                if ref_tok:
                    ref_tokens.append(ref_tok)
                if hyp_tok:
                    hyp_tokens.append(hyp_tok)
            elif op == "D":
                ref_tok = strip_q(parts[0]) if parts else ""
                if process_unk and ref_tok == unk_token:
                    ref_tok = ""
                if ref_tok:
                    ref_tokens.append(ref_tok)
            elif op == "I":
                hyp_tok = ""
                if len(parts) == 1:
                    hyp_tok = strip_q(parts[0])
                elif len(parts) >= 2:
                    hyp_tok = strip_q(parts[1])
                if process_unk and hyp_tok == unk_token:
                    hyp_tok = ""
                if hyp_tok:
                    hyp_tokens.append(hyp_tok)

        records.append(
            {
                "utt_id": utt_id,
                "pred": " ".join(hyp_tokens),
                "target": " ".join(ref_tokens),
                "align_line": align_line,
            }
        )
        i = j + 1
    return records


def load_trn(path: Path | None) -> dict[str, str]:
    if not path:
        return {}
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.match(r"^(.*) \(([^)]+)\)\s*$", line)
        if match:
            out[match.group(2)] = match.group(1)
    return out


def load_manifest(path: Path | None) -> list[dict[str, str]]:
    if not path or not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def row_index_from_utt(utt_id: str) -> int | None:
    match = re.fullmatch(r"utt(\d+)", utt_id)
    if not match:
        return None
    return int(match.group(1)) - 1


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sgml-ref1", type=Path, required=True)
    parser.add_argument("--sgml-ref2", type=Path, required=True)
    parser.add_argument("--hyp-trn", type=Path)
    parser.add_argument("--ref1-trn", type=Path)
    parser.add_argument("--ref2-trn", type=Path)
    parser.add_argument("--manifest-csv", type=Path)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--max-rows", type=int, default=25)
    parser.add_argument("--max-text-chars", type=int, default=240)
    args = parser.parse_args()

    rec1 = parse_sgml_records(args.sgml_ref1)
    rec2 = parse_sgml_records(args.sgml_ref2)
    hyp_trn = load_trn(args.hyp_trn)
    ref1_trn = load_trn(args.ref1_trn)
    ref2_trn = load_trn(args.ref2_trn)
    manifest = load_manifest(args.manifest_csv)

    mismatches: list[dict[str, Any]] = []
    for idx, (a, b) in enumerate(zip(rec1, rec2)):
        if a["pred"] == b["pred"]:
            continue
        utt_id = a["utt_id"]
        row_idx = row_index_from_utt(utt_id)
        manifest_row = manifest[row_idx] if row_idx is not None and 0 <= row_idx < len(manifest) else {}
        pred1_no_unk = remove_unk_tokens(a["pred"])
        pred2_no_unk = remove_unk_tokens(b["pred"])
        mismatches.append(
            {
                "ordinal": idx + 1,
                "utt_id_ref1": a["utt_id"],
                "utt_id_ref2": b["utt_id"],
                "manifest_id": manifest_row.get("id"),
                "speaker": manifest_row.get("speaker"),
                "etiology": manifest_row.get("etiology"),
                "pred_ref1": a["pred"],
                "pred_ref2": b["pred"],
                "pred_ref1_no_unk": pred1_no_unk,
                "pred_ref2_no_unk": pred2_no_unk,
                "only_unk_token_delta": pred1_no_unk == pred2_no_unk,
                "official_hyp_trn": hyp_trn.get(utt_id),
                "ref1_target": a["target"],
                "ref2_target": b["target"],
                "ref1_trn": ref1_trn.get(utt_id),
                "ref2_trn": ref2_trn.get(utt_id),
                "align_ref1": a["align_line"],
                "align_ref2": b["align_line"],
            }
        )

    result: dict[str, Any] = {
        "sgml_ref1": str(args.sgml_ref1),
        "sgml_ref2": str(args.sgml_ref2),
        "len_ref1": len(rec1),
        "len_ref2": len(rec2),
        "compared": min(len(rec1), len(rec2)),
        "mismatch_count": len(mismatches),
        "only_unk_token_delta_count": sum(1 for row in mismatches if row["only_unk_token_delta"]),
        "first_mismatches": mismatches[: args.max_rows],
    }

    print(
        "SGML_PRED_MISMATCH len_ref1={} len_ref2={} mismatches={} only_unk_token_delta={}".format(
            result["len_ref1"],
            result["len_ref2"],
            result["mismatch_count"],
            result["only_unk_token_delta_count"],
        )
    )
    for row in mismatches[: args.max_rows]:
        print(
            "\n[{ordinal}] {utt} manifest_id={mid} speaker={speaker} etiology={eti} only_unk_delta={unk}".format(
                ordinal=row["ordinal"],
                utt=row["utt_id_ref1"],
                mid=row.get("manifest_id"),
                speaker=row.get("speaker"),
                eti=row.get("etiology"),
                unk=row["only_unk_token_delta"],
            )
        )
        for key in ("official_hyp_trn", "pred_ref1", "pred_ref2", "ref1_trn", "ref2_trn"):
            val = row.get(key)
            if val is not None:
                print(f"  {key}: {truncate(val, args.max_text_chars)}")
        print(f"  align_ref1: {truncate(row['align_ref1'], args.max_text_chars)}")
        print(f"  align_ref2: {truncate(row['align_ref2'], args.max_text_chars)}")

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote {args.out_json}")

    if len(rec1) != len(rec2):
        sys.exit(2)
    if mismatches:
        sys.exit(1)


if __name__ == "__main__":
    main()
