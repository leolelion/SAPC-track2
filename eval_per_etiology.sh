#!/usr/bin/env bash
# Per-etiology evaluation of a hypothesis CSV against the SAPC2 Dev manifest.
# Splits Dev.csv into per-etiology sub-manifests, runs evaluate.sh on each,
# then prints a CER/WER breakdown table.
#
# Usage: ./eval_per_etiology.sh HYP_CSV [SPLIT]
#   HYP_CSV: path to hypothesis CSV (id,raw_hypos)
#   SPLIT:   manifest split name (default: Dev)
set -euo pipefail

HYP_CSV="${1:?Usage: $0 HYP_CSV [SPLIT]}"
SPLIT="${2:-Dev}"

DATA_ROOT="${DATA_ROOT:-/workspace/SAPC2}"
PROJ_ROOT="${PROJ_ROOT:-/workspace/SAPC-template}"
SCTK_DIR="${SCTK_DIR:-/workspace/SCTK}"
export PATH="${SCTK_DIR}/bin:$PATH"

MANIFEST="${DATA_ROOT}/manifest/${SPLIT}.csv"
WORK_DIR="${PROJ_ROOT}/metrics/per_etiology"
mkdir -p "${WORK_DIR}"

echo "[split] manifest=${MANIFEST}"
python3 - "$MANIFEST" "$HYP_CSV" "$WORK_DIR" "$SPLIT" << 'PY'
import sys, os, csv
import pandas as pd

manifest, hyp_csv, work_dir, split = sys.argv[1:5]
df = pd.read_csv(manifest)
hyp_df = pd.read_csv(hyp_csv)
hyp_ids = set(hyp_df["id"].astype(str))

groups = []
for eti, sub in df.groupby("etiology"):
    safe = eti.replace(" ", "_").replace("'", "")
    name = f"{split}__{safe}"
    sub_csv = os.path.join(work_dir, f"{name}.csv")
    sub.to_csv(sub_csv, index=False)
    n_in_hyp = sum(1 for i in sub["id"].astype(str) if i in hyp_ids)
    print(f"  {eti:24s}  {len(sub):6d} utts  ({n_in_hyp} in hyp)  -> {sub_csv}")
    groups.append((eti, name, sub_csv))

with open(os.path.join(work_dir, "_groups.txt"), "w") as f:
    for eti, name, sub_csv in groups:
        f.write(f"{eti}\t{name}\t{sub_csv}\n")
PY

GROUPS_FILE="${WORK_DIR}/_groups.txt"
mkdir -p "${WORK_DIR}/refs"

while IFS=$'\t' read -r ETI NAME SUB_CSV; do
  # 1) Generate per-group ref TRN files into the per-etiology refs dir.
  bash "${PROJ_ROOT}/steps/eval/prepare_ref_trn.sh" "${PROJ_ROOT}" \
    --split "${NAME}" --csv "${SUB_CSV}" \
    --ref1-col "norm_text_with_disfluency" \
    --ref2-col "norm_text_without_disfluency" \
    --out-dir "${WORK_DIR}/refs" >/dev/null

  # 2) Run the eval (uses sub-manifest + per-group refs).
  bash "${PROJ_ROOT}/steps/eval/evaluate.sh" "${PROJ_ROOT}" "${DATA_ROOT}" \
    --split "${NAME}" --hyp-csv "${HYP_CSV}" --hyp-col "raw_hypos" \
    --ref-dir "${WORK_DIR}/refs" --manifest-csv "${SUB_CSV}" \
    --out-json "${WORK_DIR}/metrics_${NAME}.json" 2>&1 | tail -3
done < "${GROUPS_FILE}"

echo
echo "===== Per-etiology results: ${HYP_CSV} ====="
python3 - "${WORK_DIR}" "${GROUPS_FILE}" << 'PY'
import sys, os, json
work_dir, groups_file = sys.argv[1:3]
rows = []
with open(groups_file) as f:
    for line in f:
        eti, name, _ = line.rstrip("\n").split("\t")
        m = json.load(open(os.path.join(work_dir, f"metrics_{name}.json")))
        rows.append((eti, m["n_utts"], m["cer"], m["wer"]))
rows.sort(key=lambda r: -r[2])  # worst CER first
print(f"{'Etiology':24s} {'N':>7s}  {'CER%':>7s} {'WER%':>7s}")
print("-" * 50)
for eti, n, cer, wer in rows:
    print(f"{eti:24s} {n:7d}  {cer*100:7.2f} {wer*100:7.2f}")
PY
