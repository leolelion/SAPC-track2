"""Local tests for the deletion-localization additions to error_decomposition.py.

No SGML, no torchmetrics, no SAP data. Two things must hold before this goes near a pod:

  1. Extracting the DP table so `seq_align_trace` can share it did not change
     `seq_align` -- that function is what Gate B uses to reproduce the official CER/WER.
  2. The trace describes the SAME alignment as the counts, so a deletion position is a
     position in the alignment the CER was computed from, not a second opinion.
"""
import random
import sys
from collections import Counter

sys.path.insert(0, "/Users/o/Downloads/SAPC-template/scripts")
from error_decomposition import (  # noqa: E402
    deletion_profile,
    report_deletion_localization,
    seq_align,
    seq_align_trace,
    spearman,
)

# ---------------------------------------------------------------- test 1
# Property test: over random pairs, the trace's op counts must equal seq_align's, and
# seq_align's ops must still sum to the Levenshtein distance (its own internal assert).
random.seed(20260730)
ALPHA = "abcdef"
for trial in range(400):
    a = [random.choice(ALPHA) for _ in range(random.randint(0, 12))]
    b = [random.choice(ALPHA) for _ in range(random.randint(0, 12))]
    dist, ops = seq_align(a, b)
    tr = Counter(op for op, _ in seq_align_trace(a, b))
    assert tr["D"] == ops["D"], (a, b, tr, ops)
    assert tr["S"] == ops["S"], (a, b, tr, ops)
    assert tr["C"] == ops["C"], (a, b, tr, ops)
    # every non-insertion op consumes exactly one reference element, in order
    idx = [j for _, j in seq_align_trace(a, b)]
    assert idx == list(range(len(b))), (a, b, idx)
print("test1 PASS: 400 random pairs — trace and counts describe one alignment")

# ---------------------------------------------------------------- test 2
# Hand-checked positions and runs.
p = deletion_profile("one two", "one two three four five")
assert p["n_ref_words"] == 5
assert p["del_runs"] == [3], p                      # three, four, five: one run of 3
assert [round(x, 2) for x in p["del_positions"]] == [0.50, 0.70, 0.90], p
print(f"test2a PASS: tail deletion -> runs {p['del_runs']} positions "
      f"{[round(x,2) for x in p['del_positions']]}")

p = deletion_profile("one three five", "one two three four five")
assert p["del_runs"] == [1, 1], p                   # two and four: two singletons
assert [round(x, 2) for x in p["del_positions"]] == [0.30, 0.70], p
print(f"test2b PASS: scattered deletion -> runs {p['del_runs']}")

p = deletion_profile("", "one two three")
assert p["del_runs"] == [3] and len(p["del_positions"]) == 3, p
print("test2c PASS: empty hypothesis deletes the whole reference as one run")

# ---------------------------------------------------------------- test 3
# The report must be able to SEE a planted signal. Build two synthetic populations:
# one deleting only late in the utterance, one deleting uniformly, and check the
# histogram separates them. This is what the pod run is being asked to decide.
def rec(hyp, ref, dur, empty=False):
    p = deletion_profile(hyp, ref)
    ref_w = ref.split()
    d = len(ref_w) - len(hyp.split())
    return {
        "hyp_empty": empty,
        "n_ref_words_align": p["n_ref_words"],
        "del_positions": p["del_positions"],
        "del_runs": p["del_runs"],
        "word_bucket": "13+" if len(ref_w) >= 13 else "7-12",
        "ref_chars": len(ref),
        "errors": float(sum(len(w) + 1 for w in ref_w[len(hyp.split()):])),
        "cer": 0.0,
        "ref_words": len(ref_w),
        "word_ops": {"S": 0, "D": max(0, d), "I": 0},
        "duration_f": dur,
    }


REF = " ".join(f"w{i:02d}" for i in range(16))
late = [rec(" ".join(REF.split()[: 16 - k]), REF, 6.0 + 0.1 * k) for k in range(1, 9)] * 3
uniform = []
for k in range(1, 9):
    words = REF.split()
    keep = [w for i, w in enumerate(words) if i % (16 // k or 16) != 0]
    uniform.append(rec(" ".join(keep), REF, 6.0))
uniform *= 3

print("\n--- planted signal A: deletions only at the END ---")
resA = report_deletion_localization(late, sum(r["ref_chars"] for r in late))
print("\n--- planted signal B: deletions spread UNIFORMLY ---")
resB = report_deletion_localization(uniform, sum(r["ref_chars"] for r in uniform))

hA, hB = resA["position_hist_all"], resB["position_hist_all"]
assert hA[4] > hA[0], f"late-deletion population did not show a rising histogram: {hA}"
assert hA[4] / max(1, sum(hA)) > 0.4, hA
assert max(hB) / max(1, sum(hB)) < 0.45, f"uniform population looks peaked: {hB}"
print(f"\ntest3 PASS: histogram separates the two — late {hA} vs uniform {hB}")

# ---------------------------------------------------------------- test 4
assert abs(spearman([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) - 1.0) < 1e-9
assert abs(spearman([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) + 1.0) < 1e-9
assert spearman([1.0], [1.0]) is None
assert spearman([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None  # zero variance, not 0.0
print("test4 PASS: spearman endpoints, zero-variance and short-input guards")

print("\nALL 4 PASS")
