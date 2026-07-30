#!/usr/bin/env python3
"""Step 0 — official-scorer error decomposition.

Every error analysis we have run so far came from hand-rolled proxy scripts. One of
them was 11 CER points wrong (single-ref, no min-over-two-refs, no `unk`
reconciliation) and it steered months of work. This script analyses the SAME artifact
the official scorer consumes -- the sclite SGML alignments written by
`utils/evaluate.py` -- and refuses to report anything unless its own aggregate CER/WER
reproduces the official metric classes exactly.

What it answers, in the metric's own currency (CER is char-weighted micro:
sum(edit_distance) / sum(ref_chars), per-utterance clipped at 1.0):

  * how many CER POINTS each slice contributes (etiology / speaker / utterance length /
    empty-vs-non-empty) -- i.e. share of the total error mass, not share of utterances;
  * the char-level substitution / deletion / insertion split, which decomposes CER
    additively;
  * the same split at word level, taken straight from sclite's C/S/D/I ops;
  * the worst individual contributors, so a human can go listen to them.

Inputs are the two SGML files stage 2 leaves behind:
    $DATA_ROOT/eval/sctk/<split>.ref1.sgml
    $DATA_ROOT/eval/sctk/<split>.ref2.sgml
NOTE: stage 2 OVERWRITES these per split. Copy them aside (or run this) immediately
after each scoring run, before scoring anything else.

Utterance identity: `utils/evaluate.py` writes sequential ids `utt000001..uttNNNNNN`
in MANIFEST ROW ORDER, so row i of --manifest-csv is utt{i+1:06d}. The join is checked
on count and fails loudly on mismatch.

Wraps, never modifies, the scoring code (house rule).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UTILS_DIR = os.path.join(REPO_ROOT, "utils")

Op = Tuple[str, str, str]  # (op, ref_token, hyp_token)


# ---------------------------------------------------------------- SGML parsing
def parse_sgml_ops(
    path: str, process_unk: bool = True, unk_token: str = "unk"
) -> List[Tuple[str, List[Op]]]:
    """Parse sclite PATH-style SGML into [(utt_id, [(op, ref_tok, hyp_tok), ...]), ...].

    Mirrors `utils/compute_metrics.py:parse_sgml_csdi` op-for-op, including the three
    `unk` reconciliation rules, but KEEPS the utterance id and the per-token ops instead
    of collapsing to two strings. `verify_against_official()` proves the mirror holds by
    reconstructing the official parser's exact output from these ops.
    """
    out: List[Tuple[str, List[Op]]] = []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("<PATH"):
            m = re.search(r'id="([^"]+)"', line)
            if not m:
                i += 1
                continue
            utt_id = m.group(1)
            if utt_id.startswith("(") and utt_id.endswith(")"):
                utt_id = utt_id[1:-1]

            j = i + 1
            while j < len(lines) and (
                not lines[j].strip() or lines[j].lstrip().startswith("<")
            ):
                j += 1
            if j >= len(lines):
                break
            align_line = lines[j].strip()

            ops: List[Op] = []
            for seg in align_line.split(":"):
                seg = seg.strip()
                if not seg:
                    continue
                op = seg[0]
                rest = seg[2:]
                parts = [p.strip() for p in rest.split(",")]

                def strip_q(x: str) -> str:
                    x = x.strip()
                    if len(x) >= 2 and x[0] == '"' and x[-1] == '"':
                        return x[1:-1]
                    return x

                if op in ("C", "S"):
                    if len(parts) >= 2:
                        ref_tok, hyp_tok = strip_q(parts[0]), strip_q(parts[1])
                    else:
                        ref_tok = hyp_tok = ""
                    if process_unk and op == "S" and ref_tok == unk_token and hyp_tok:
                        ref_tok = hyp_tok  # S on unk: adopt the hyp token as the ref
                    ops.append((op, ref_tok, hyp_tok))
                elif op == "D":
                    ref_tok = strip_q(parts[0]) if parts else ""
                    if process_unk and ref_tok == unk_token:
                        ref_tok = ""  # D on unk: the unk is free
                    ops.append(("D", ref_tok, ""))
                elif op == "I":
                    if len(parts) == 1:
                        hyp_tok = strip_q(parts[0])
                    elif len(parts) >= 2:
                        hyp_tok = strip_q(parts[1])
                    else:
                        hyp_tok = ""
                    if process_unk and hyp_tok == unk_token:
                        hyp_tok = ""  # I of unk: the unk is free
                    ops.append(("I", "", hyp_tok))
                # other ops ignored, exactly as the official parser does
            out.append((utt_id, ops))
            i = j
        i += 1

    if not out:
        raise RuntimeError(f"No PATH alignment found in SGML: {path}")
    return out


def ops_to_strings(ops: Sequence[Op]) -> Tuple[str, str]:
    """Collapse ops back to (hyp, ref) the way the official parser builds them."""
    ref = [r for (_o, r, _h) in ops if r]
    hyp = [h for (_o, _r, h) in ops if h]
    return " ".join(hyp), " ".join(ref)


def sclite_csdi(ops: Sequence[Op]) -> Dict[str, int]:
    """sclite's own word-level C/S/D/I counts, AFTER unk reconciliation.

    Kept as a secondary view only. sclite's alignment is NOT minimal under the cost model
    torchmetrics uses, so these counts do not reproduce the official WER -- verified on a
    synthetic case where sclite scores S+D+I=3 for a pair whose Levenshtein distance is 2.
    The headline word split comes from `seq_align` on the reconstructed strings, which is
    exactly what the official metric measures.

    A `D` whose ref token was dropped by the unk rule (and an `I` whose hyp token was
    dropped) is no longer an error, so it is counted as `unk_freed`, not as an error.
    """
    c = Counter()
    for op, r, h in ops:
        if op == "D" and not r:
            c["unk_freed"] += 1
        elif op == "I" and not h:
            c["unk_freed"] += 1
        elif op == "S" and r == h:
            c["C"] += 1  # unk rule turned this S into an identity
        else:
            c[op] += 1
    return {k: int(c.get(k, 0)) for k in ("C", "S", "D", "I", "unk_freed")}


# ------------------------------------------------------------ edit alignment
def seq_align(a: Sequence, b: Sequence) -> Tuple[int, Dict[str, int]]:
    """Levenshtein distance with backtrace. a = hypothesis, b = reference.

    Called with `list(hyp)` / `list(ref)` for the char level and `hyp.split()` /
    `ref.split()` for the word level -- the exact sequences torchmetrics' `_edit_distance`
    sees inside the official CER and WER -- so the returned distance is the official
    per-utterance edit distance before clipping.

    Returns (distance, {"S":.., "D":.., "I":.., "C":..}) where D = a ref element with no
    hyp counterpart (the model dropped it) and I = a hyp element with no ref counterpart.
    """
    a, b = list(a), list(b)
    n, m = len(a), len(b)
    # d[i][j] = distance between a[:i] and b[:j]
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = i
    for j in range(1, m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        ai = a[i - 1]
        di, dprev = d[i], d[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            di[j] = min(dprev[j] + 1, di[j - 1] + 1, dprev[j - 1] + cost)

    counts = Counter()
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if a[i - 1] == b[j - 1] else 1
            if d[i][j] == d[i - 1][j - 1] + cost:
                counts["C" if cost == 0 else "S"] += 1
                i, j = i - 1, j - 1
                continue
        if i > 0 and d[i][j] == d[i - 1][j] + 1:
            counts["I"] += 1  # hyp char with no ref counterpart
            i -= 1
            continue
        if j > 0 and d[i][j] == d[i][j - 1] + 1:
            counts["D"] += 1  # ref char the model dropped
            j -= 1
            continue
        raise AssertionError("backtrace fell off the DP table")

    dist = d[n][m]
    ops = {k: int(counts.get(k, 0)) for k in ("C", "S", "D", "I")}
    assert ops["S"] + ops["D"] + ops["I"] == dist, "ops do not sum to the distance"
    return dist, ops


# ------------------------------------------------------------------ selection
def choose_ref(ed1: int, len1: int, ed2: int, len2: int) -> str:
    """Replicate `_cer_update_min_two_refs`'s per-utterance choice.

    Errors/lengths are accumulated from whichever reference gives the lower per-utterance
    CER; on an exact tie both are averaged at 0.5 weight. len == 0 gives CER = inf.
    """
    cer1 = float(ed1) / len1 if len1 > 0 else float("inf")
    cer2 = float(ed2) / len2 if len2 > 0 else float("inf")
    if cer1 < cer2:
        return "ref1"
    if cer2 < cer1:
        return "ref2"
    return "tie"


# ----------------------------------------------------------------- manifest
def load_manifest(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def bucket_words(n: int) -> str:
    if n <= 1:
        return "1"
    if n <= 3:
        return "2-3"
    if n <= 6:
        return "4-6"
    if n <= 12:
        return "7-12"
    return "13+"


# ----------------------------------------------------------------- reporting
def _pct(x: float) -> str:
    return f"{x:6.2f}"


def print_slice_table(title: str, rows: List[dict], total_err: float) -> None:
    print(f"\n--- {title} ---")
    print(
        f"{'slice':<22}{'n':>6}{'ref_ch':>9}{'err':>9}{'CER%':>8}"
        f"{'CERpts':>8}{'err%':>7}{'empty':>7}{'D%':>7}{'S%':>7}{'I%':>7}"
    )
    for r in rows:
        print(
            f"{r['slice'][:22]:<22}{r['n']:>6}{r['ref_chars']:>9}{r['errors']:>9.0f}"
            f"{_pct(r['cer_pct']):>8}{_pct(r['cer_points']):>8}"
            f"{_pct(r['error_share_pct']):>7}{r['n_empty']:>7}"
            f"{_pct(r['char_del_pct']):>7}{_pct(r['char_sub_pct']):>7}"
            f"{_pct(r['char_ins_pct']):>7}"
        )
    print(
        "  CERpts = this slice's contribution to the overall CER, in absolute points "
        f"(they sum to the total CER). err% = share of the {total_err:.0f} total errors."
    )


def summarize(records: List[dict], key: str, total_err: float, total_ref: float) -> List[dict]:
    groups: Dict[str, List[dict]] = defaultdict(list)
    for r in records:
        groups[str(r.get(key, "?"))].append(r)
    rows = []
    for name, rs in groups.items():
        err = sum(r["errors"] for r in rs)
        ref = sum(r["ref_chars"] for r in rs)
        cs, cd, ci = (sum(r["char_ops"][k] for r in rs) for k in ("S", "D", "I"))
        rows.append(
            {
                "slice": name,
                "n": len(rs),
                "ref_chars": int(ref),
                "errors": err,
                "cer_pct": 100.0 * err / ref if ref else 0.0,
                "cer_points": 100.0 * err / total_ref if total_ref else 0.0,
                "error_share_pct": 100.0 * err / total_err if total_err else 0.0,
                "n_empty": sum(1 for r in rs if r["hyp_empty"]),
                "char_del_pct": 100.0 * cd / ref if ref else 0.0,
                "char_sub_pct": 100.0 * cs / ref if ref else 0.0,
                "char_ins_pct": 100.0 * ci / ref if ref else 0.0,
            }
        )
    rows.sort(key=lambda r: -r["cer_points"])
    return rows


# --------------------------------------------------------------------- main
def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sgml-ref1", required=True, help="<split>.ref1.sgml from stage 2")
    p.add_argument("--sgml-ref2", required=True, help="<split>.ref2.sgml from stage 2")
    p.add_argument("--manifest-csv", help="split manifest; row order defines uttNNNNNN. "
                                          "Without it, etiology/speaker/duration slices are skipped.")
    p.add_argument("--hyp-csv", help="optional raw hypothesis CSV, only to print raw text for worst offenders")
    p.add_argument("--hyp-col", default="raw_hypos")
    p.add_argument("--out-json", required=True)
    p.add_argument("--top-n", type=int, default=30, help="worst individual contributors to print")
    p.add_argument("--skip-official-check", action="store_true",
                   help="DO NOT USE for anything that informs a decision. Skips the gate that "
                        "proves this script reproduces the official CER/WER.")
    return p


def verify_against_official(
    sgml1: str, sgml2: str, ops1: List[Tuple[str, List[Op]]], ops2: List[Tuple[str, List[Op]]],
    my_cer: float, my_wer: float,
) -> None:
    """Two gates. Either failing means STOP -- the analysis below would be fiction."""
    sys.path.insert(0, UTILS_DIR)
    from compute_metrics import parse_sgml_csdi  # noqa: E402
    from metrics.cer import CharErrorRateMinTwoRefs  # noqa: E402
    from metrics.wer import WordErrorRateMinTwoRefs  # noqa: E402

    # Gate A: our op-level parse reconstructs the official parser's strings exactly.
    for tag, sgml, ops in (("ref1", sgml1, ops1), ("ref2", sgml2, ops2)):
        off_preds, off_targets = parse_sgml_csdi(sgml, process_unk=True, unk_token="unk")
        mine = [ops_to_strings(o) for (_uid, o) in ops]
        my_preds = [p for (p, _t) in mine]
        my_targets = [t for (_p, t) in mine]
        if my_preds != off_preds or my_targets != off_targets:
            n_bad = sum(1 for a, b in zip(my_preds, off_preds) if a != b) + sum(
                1 for a, b in zip(my_targets, off_targets) if a != b
            )
            raise SystemExit(
                f"GATE A FAILED on {tag}: our SGML parse differs from "
                f"utils/compute_metrics.py:parse_sgml_csdi ({n_bad} mismatched strings, "
                f"lens {len(my_preds)}/{len(off_preds)}). STOP -- do not read the report."
            )

    # Gate B: our per-utterance sums reproduce the official aggregate metrics.
    p1, t1 = parse_sgml_csdi(sgml1, process_unk=True, unk_token="unk")
    p2, t2 = parse_sgml_csdi(sgml2, process_unk=True, unk_token="unk")
    off_cer = float(CharErrorRateMinTwoRefs(clip_at_one=True)(p1, p2, t1, t2))
    off_wer = float(WordErrorRateMinTwoRefs(clip_at_one=True)(p1, p2, t1, t2))
    for name, mine_v, off_v in (("CER", my_cer, off_cer), ("WER", my_wer, off_wer)):
        if abs(mine_v - off_v) > 1e-6:
            raise SystemExit(
                f"GATE B FAILED: our {name} {mine_v:.8f} != official {off_v:.8f}. "
                "STOP -- the per-utterance attribution does not match the metric."
            )
    print(f"[gate A] SGML parse matches utils/compute_metrics.py exactly")
    print(f"[gate B] CER {my_cer*100:.4f}% and WER {my_wer*100:.4f}% reproduce the official metrics")


def main() -> None:
    args = get_parser().parse_args()

    ops1 = parse_sgml_ops(args.sgml_ref1)
    ops2 = parse_sgml_ops(args.sgml_ref2)
    if len(ops1) != len(ops2):
        raise SystemExit(f"SGML utterance counts differ: {len(ops1)} vs {len(ops2)}")

    manifest = load_manifest(args.manifest_csv) if args.manifest_csv else None
    if manifest is not None and len(manifest) != len(ops1):
        raise SystemExit(
            f"manifest has {len(manifest)} rows but SGML has {len(ops1)} utterances -- "
            "the uttNNNNNN join would be wrong. Pass the manifest that was scored."
        )

    raw_hyp: Dict[str, str] = {}
    if args.hyp_csv:
        with open(args.hyp_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                raw_hyp[row["id"]] = row.get(args.hyp_col, "")

    records: List[dict] = []
    tot_err = 0.0
    tot_ref = 0.0
    tot_werr = 0.0
    tot_wref = 0.0
    n_clipped = 0

    for idx, ((uid1, o1), (uid2, o2)) in enumerate(zip(ops1, ops2)):
        if uid1 != uid2:
            raise SystemExit(f"SGML ids diverge at row {idx}: {uid1} vs {uid2}")

        hyp1, ref1 = ops_to_strings(o1)
        hyp2, ref2 = ops_to_strings(o2)

        # CER and WER each run their own min-over-two-refs selection, on their own
        # tokenization. They can land on different references for the same utterance.
        ced1_raw, cops1 = seq_align(list(hyp1), list(ref1))
        ced2_raw, cops2 = seq_align(list(hyp2), list(ref2))
        clen1, clen2 = len(ref1), len(ref2)
        ced1 = min(ced1_raw, clen1) if clen1 > 0 else ced1_raw
        ced2 = min(ced2_raw, clen2) if clen2 > 0 else ced2_raw
        if ced1 != ced1_raw or ced2 != ced2_raw:
            n_clipped += 1

        wed1_raw, wops1 = seq_align(hyp1.split(), ref1.split())
        wed2_raw, wops2 = seq_align(hyp2.split(), ref2.split())
        wlen1, wlen2 = len(ref1.split()), len(ref2.split())
        wed1 = min(wed1_raw, wlen1) if wlen1 > 0 else wed1_raw
        wed2 = min(wed2_raw, wlen2) if wlen2 > 0 else wed2_raw

        pick = choose_ref(ced1, clen1, ced2, clen2)
        if pick == "ref1":
            err, ref_chars, cops, ops, ref_txt, hyp_txt = ced1, clen1, cops1, o1, ref1, hyp1
        elif pick == "ref2":
            err, ref_chars, cops, ops, ref_txt, hyp_txt = ced2, clen2, cops2, o2, ref2, hyp2
        else:
            err, ref_chars = 0.5 * (ced1 + ced2), 0.5 * (clen1 + clen2)
            # A tie needs one representative alignment for the op split; take ref1's
            # rather than averaging two different alignments into a fiction.
            cops, ops, ref_txt, hyp_txt = cops1, o1, ref1, hyp1

        wpick = choose_ref(wed1, wlen1, wed2, wlen2)
        if wpick == "ref1":
            w_err, w_ref, wops = wed1, wlen1, wops1
        elif wpick == "ref2":
            w_err, w_ref, wops = wed2, wlen2, wops2
        else:
            w_err, w_ref, wops = 0.5 * (wed1 + wed2), 0.5 * (wlen1 + wlen2), wops1

        tot_err += err
        tot_ref += ref_chars
        tot_werr += w_err
        tot_wref += w_ref

        rec = {
            "utt": uid1,
            "row": idx,
            "chosen_ref": pick,
            "chosen_ref_wer": wpick,
            "errors": err,
            "ref_chars": ref_chars,
            "cer": (err / ref_chars) if ref_chars else 0.0,
            "word_errors": w_err,
            "hyp_empty": hyp_txt.strip() == "",
            "ref_words": w_ref,
            "word_bucket": bucket_words(int(w_ref)),
            "char_ops": cops,
            "word_ops": wops,
            "sclite_word_ops": sclite_csdi(ops),
            "ref_text": ref_txt,
            "hyp_text": hyp_txt,
        }
        if manifest is not None:
            m = manifest[idx]
            rec["id"] = m.get("id", "")
            rec["speaker"] = m.get("speaker", "")
            rec["etiology"] = m.get("etiology", "")
            rec["duration"] = m.get("duration", "")
            if raw_hyp:
                rec["raw_hyp"] = raw_hyp.get(rec["id"], "")
        records.append(rec)

    cer = tot_err / tot_ref if tot_ref else 0.0
    wer = tot_werr / tot_wref if tot_wref else 0.0

    if args.skip_official_check:
        print("!! official-scorer check SKIPPED -- these numbers carry no ship authority !!")
    else:
        verify_against_official(args.sgml_ref1, args.sgml_ref2, ops1, ops2, cer, wer)

    # ---- report ----
    n = len(records)
    empties = [r for r in records if r["hyp_empty"]]
    emp_err = sum(r["errors"] for r in empties)
    cS, cD, cI = (sum(r["char_ops"][k] for r in records) for k in ("S", "D", "I"))
    wS, wD, wI = (sum(r["word_ops"][k] for r in records) for k in ("S", "D", "I"))

    print("\n=== Step 0: official-scorer error decomposition ===")
    print(f"utterances            : {n}")
    print(f"total ref chars       : {tot_ref:.0f}")
    print(f"total errors (clipped): {tot_err:.0f}")
    print(f"CER / WER             : {cer*100:.4f}% / {wer*100:.4f}%")
    print(f"utts hitting the 1.0 clip: {n_clipped}")
    print("\n--- char-level split (these three sum to the CER) ---")
    print(f"deletions    : {cD:>8}  = {100.0*cD/tot_ref:6.2f} CER pts")
    print(f"substitutions: {cS:>8}  = {100.0*cS/tot_ref:6.2f} CER pts")
    print(f"insertions   : {cI:>8}  = {100.0*cI/tot_ref:6.2f} CER pts")
    if cS and cI:
        print(f"del:sub = {cD/cS:.2f}:1   del:ins = {cD/cI:.2f}:1")
    print("\n--- word-level split (same rule, word tokens) ---")
    if wS:
        print(f"D {wD}  S {wS}  I {wI}   del:sub = {wD/wS:.2f}:1")
    else:
        print(f"D {wD}  S {wS}  I {wI}")
    print("\n--- empty hypotheses ---")
    print(f"count      : {len(empties)} / {n} ({100.0*len(empties)/n:.2f}% of utterances)")
    print(f"error mass : {emp_err:.0f} / {tot_err:.0f} ({100.0*emp_err/tot_err:.2f}% of all errors)")
    print(f"CER cost   : {100.0*emp_err/tot_ref:.2f} points of the {cer*100:.2f}% total")
    print("  ^ this is the number the 2026-07-26 proxy analysis got wrong: empties are")
    print("    short, and CER is char-weighted, so utterance count overstates their cost.")

    slices = {"word_bucket": "by reference length (words)"}
    if manifest is not None:
        slices["etiology"] = "by etiology"
        slices["speaker"] = "by speaker"

    agg = {}
    for key, title in slices.items():
        rows = summarize(records, key, tot_err, tot_ref)
        agg[key] = rows
        print_slice_table(title, rows if key != "speaker" else rows[:15], tot_err)

    worst = sorted(records, key=lambda r: -r["errors"])[: args.top_n]
    print(f"\n--- top {len(worst)} individual contributors ---")
    for r in worst:
        tag = "EMPTY" if r["hyp_empty"] else "     "
        who = f"{r.get('etiology','?')[:14]:<14} {r.get('speaker','?')[:10]:<10}" if manifest else ""
        print(
            f"{r['utt']} {tag} {who} err={r['errors']:>5.0f} ref_ch={r['ref_chars']:>4.0f} "
            f"pts={100.0*r['errors']/tot_ref:5.2f}  ref={r['ref_text'][:60]!r}"
        )

    out = {
        "source": {
            "sgml_ref1": os.path.abspath(args.sgml_ref1),
            "sgml_ref2": os.path.abspath(args.sgml_ref2),
            "manifest_csv": os.path.abspath(args.manifest_csv) if args.manifest_csv else None,
            "official_check": not args.skip_official_check,
        },
        "totals": {
            "n_utts": n,
            "ref_chars": tot_ref,
            "errors": tot_err,
            "cer": cer,
            "wer": wer,
            "n_clipped": n_clipped,
            "char_ops": {"S": cS, "D": cD, "I": cI},
            "word_ops": {"S": wS, "D": wD, "I": wI},
        },
        "empties": {
            "n": len(empties),
            "error_mass": emp_err,
            "cer_points": 100.0 * emp_err / tot_ref if tot_ref else 0.0,
            "utts": [r["utt"] for r in empties],
            "ids": [r.get("id") for r in empties] if manifest else None,
        },
        "slices": agg,
        "per_utterance": records,
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWritten: {args.out_json}")


if __name__ == "__main__":
    main()
