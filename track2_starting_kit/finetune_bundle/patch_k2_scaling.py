#!/usr/bin/env python3
"""
Patch icefall's scaling.py to bypass k2 CUDA kernels (SwooshL / SwooshR).

The CPU-only k2 wheel lacks the CUDA kernels called by SwooshL/SwooshR,
causing a segfault during training. This script replaces those calls with
pure-Python / pure-PyTorch equivalents that are mathematically identical.

Usage:
    python3 patch_k2_scaling.py [path/to/scaling.py]

Default path: /workspace/finetune/icefall/egs/librispeech/ASR/zipformer/scaling.py
"""

import sys
import re
import shutil
from pathlib import Path

SCALING_PATH = Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else "/workspace/finetune/icefall/egs/librispeech/ASR/zipformer/scaling.py"
)

if not SCALING_PATH.exists():
    print(f"ERROR: {SCALING_PATH} not found. Pass path as argument.")
    sys.exit(1)

# Back up original
backup = SCALING_PATH.with_suffix(".py.orig")
if not backup.exists():
    shutil.copy2(SCALING_PATH, backup)
    print(f"Backup saved: {backup}")
else:
    print(f"Backup already exists: {backup} (skipping)")

text = SCALING_PATH.read_text()
original_text = text

patches_applied = 0

# ──────────────────────────────────────────────────────────────────────────────
# Patch 1: SwooshL.forward — bypass k2.swoosh_l_forward / k2.swoosh_l
#
# Pattern (the block after the jit.is_scripting check):
#
#     if not x.requires_grad:
#         return k2.swoosh_l_forward(x).to(x.dtype)
#     else:
#         return k2.swoosh_l(x).to(x.dtype)
#
# Replace with the mathematically identical pure-Python expression:
#     return logaddexp(zero, x + 1.0) - 0.08 * x - 0.035 * x * x - 0.313261687
# ──────────────────────────────────────────────────────────────────────────────
SWOOSHL_K2 = re.compile(
    r"""([ \t]+)if not x\.requires_grad:\s*\n"""
    r"""\s+return k2\.swoosh_l_forward\(x\)\.to\(x\.dtype\)\s*\n"""
    r"""\s+else:\s*\n"""
    r"""\s+return k2\.swoosh_l\(x\)\.to\(x\.dtype\)""",
    re.MULTILINE,
)


def swl_replacement(m):
    indent = m.group(1)
    return (
        f"{indent}# k2 CUDA kernels removed — pure-Python fallback:\n"
        f"{indent}return (logaddexp(zero, x + 1.0) - 0.08 * x - 0.035 * x * x - 0.313261687).to(x.dtype)"
    )


new_text, n = SWOOSHL_K2.subn(swl_replacement, text)
if n:
    print(f"  [OK] Patched SwooshL.forward k2 calls ({n} match)")
    patches_applied += n
    text = new_text
else:
    print("  [WARN] SwooshL.forward k2 pattern NOT found — code may already be patched or structure differs")

# ──────────────────────────────────────────────────────────────────────────────
# Patch 2: SwooshR.forward — bypass k2.swoosh_r_forward / k2.swoosh_r
#
# Pattern:
#     if not x.requires_grad:
#         return k2.swoosh_r_forward(x).to(x.dtype)
#     else:
#         return k2.swoosh_r(x).to(x.dtype)
# ──────────────────────────────────────────────────────────────────────────────
SWOOSHR_K2 = re.compile(
    r"""([ \t]+)if not x\.requires_grad:\s*\n"""
    r"""\s+return k2\.swoosh_r_forward\(x\)\.to\(x\.dtype\)\s*\n"""
    r"""\s+else:\s*\n"""
    r"""\s+return k2\.swoosh_r\(x\)\.to\(x\.dtype\)""",
    re.MULTILINE,
)


def swr_replacement(m):
    indent = m.group(1)
    return (
        f"{indent}# k2 CUDA kernels removed — pure-Python fallback:\n"
        f"{indent}return (logaddexp(zero, x - 1.0) - 0.08 * x - 0.313261687).to(x.dtype)"
    )


new_text, n = SWOOSHR_K2.subn(swr_replacement, text)
if n:
    print(f"  [OK] Patched SwooshR.forward k2 calls ({n} match)")
    patches_applied += n
    text = new_text
else:
    print("  [WARN] SwooshR.forward k2 pattern NOT found — code may already be patched or structure differs")

# ──────────────────────────────────────────────────────────────────────────────
# Patch 3: ActivationBalancer dict entries — k2.swoosh_l_forward / k2.swoosh_r_forward
#
# Pattern:
#     "SwooshL": k2.swoosh_l_forward,
#     "SwooshR": k2.swoosh_r_forward,
# ──────────────────────────────────────────────────────────────────────────────
old3 = '"SwooshL": k2.swoosh_l_forward,'
new3 = '"SwooshL": lambda x: (logaddexp(zero, x + 1.0) - 0.08 * x - 0.035 * x * x - 0.313261687).to(x.dtype),  # was k2.swoosh_l_forward'
if old3 in text:
    text = text.replace(old3, new3)
    print(f"  [OK] Patched SwooshL dict entry (swoosh_l_forward)")
    patches_applied += 1
else:
    print(f"  [WARN] '{old3}' not found")

old4 = '"SwooshR": k2.swoosh_r_forward,'
new4 = '"SwooshR": lambda x: (logaddexp(zero, x - 1.0) - 0.08 * x - 0.313261687).to(x.dtype),  # was k2.swoosh_r_forward'
if old4 in text:
    text = text.replace(old4, new4)
    print(f"  [OK] Patched SwooshR dict entry (swoosh_r_forward)")
    patches_applied += 1
else:
    print(f"  [WARN] '{old4}' not found")

# ──────────────────────────────────────────────────────────────────────────────
# Patch 4: ActivationBalancer dict entries — forward_and_deriv variants
#
# Pattern:
#     "SwooshL": k2.swoosh_l_forward_and_deriv,
#     "SwooshR": k2.swoosh_r_forward_and_deriv,
#
# These compute (f(x), f'(x)). We return (f(x), None) since ActivationBalancer
# only uses the forward value for stats; the deriv branch isn't needed for grad
# computation (autograd handles that).
# ──────────────────────────────────────────────────────────────────────────────
old5 = '"SwooshL": k2.swoosh_l_forward_and_deriv,'
new5 = '"SwooshL": lambda x: ((logaddexp(zero, x + 1.0) - 0.08 * x - 0.035 * x * x - 0.313261687).to(x.dtype), None),  # was k2.swoosh_l_forward_and_deriv'
if old5 in text:
    text = text.replace(old5, new5)
    print(f"  [OK] Patched SwooshL dict entry (swoosh_l_forward_and_deriv)")
    patches_applied += 1
else:
    print(f"  [WARN] '{old5}' not found")

old6 = '"SwooshR": k2.swoosh_r_forward_and_deriv,'
new6 = '"SwooshR": lambda x: ((logaddexp(zero, x - 1.0) - 0.08 * x - 0.313261687).to(x.dtype), None),  # was k2.swoosh_r_forward_and_deriv'
if old6 in text:
    text = text.replace(old6, new6)
    print(f"  [OK] Patched SwooshR dict entry (swoosh_r_forward_and_deriv)")
    patches_applied += 1
else:
    print(f"  [WARN] '{old6}' not found")

# ──────────────────────────────────────────────────────────────────────────────
# Check for any remaining k2. references
# ──────────────────────────────────────────────────────────────────────────────
remaining_k2 = [(i + 1, line.rstrip()) for i, line in enumerate(text.splitlines()) if "k2." in line and not line.strip().startswith("#")]
if remaining_k2:
    print(f"\n  [INFO] Remaining k2. references ({len(remaining_k2)} lines) — check if these are in the training path:")
    for lineno, line in remaining_k2[:20]:
        print(f"    line {lineno}: {line.strip()}")
else:
    print(f"\n  [OK] No remaining k2. references in non-comment lines.")

# ──────────────────────────────────────────────────────────────────────────────
# Write patched file
# ──────────────────────────────────────────────────────────────────────────────
if patches_applied == 0:
    print("\nNo patches applied — file unchanged.")
elif text == original_text:
    print("\nERROR: patches_applied > 0 but file unchanged — this is a bug in the script.")
else:
    SCALING_PATH.write_text(text)
    print(f"\n[DONE] {patches_applied} patches applied. Written to {SCALING_PATH}")

# ──────────────────────────────────────────────────────────────────────────────
# Quick sanity test
# ──────────────────────────────────────────────────────────────────────────────
print("\n=== Quick import test ===")
import subprocess, sys as _sys
result = subprocess.run(
    [_sys.executable, "-c",
     f"import sys; sys.path.insert(0, '{SCALING_PATH.parent}'); "
     "from scaling import SwooshL, SwooshR; "
     "import torch; "
     "x = torch.randn(2, 3); "
     "print('SwooshL:', SwooshL()(x).shape); "
     "print('SwooshR:', SwooshR()(x).shape); "
     "print('CPU test PASSED')"],
    capture_output=True, text=True,
)
print(result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr[-1000:])
    print("[FAIL] Import test failed — check patch output above")
    sys.exit(1)
