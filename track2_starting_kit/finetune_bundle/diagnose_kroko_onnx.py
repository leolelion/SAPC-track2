#!/usr/bin/env python3
"""
Diagnose whether Kroko ONNX weights can be reverse-mapped into an
Icefall Zipformer2 PyTorch model for fine-tuning.

PHASE 1 (runs anywhere — only needs `onnx`):
  Inspect Kroko ONNX files, extract all weight tensor names and shapes,
  infer architectural hyperparameters (d_model, num_heads, num_layers, etc.)

PHASE 2 (runs on pod after `setup_pod.sh` — needs Icefall + k2):
  Load Icefall's Zipformer2 model, compare parameter shapes, score
  mapping feasibility, and produce a concrete go/no-go recommendation.

Usage:
  # Phase 1 — local or pod
  python3 diagnose_kroko_onnx.py --onnx-dir /path/to/kroko/weights

  # Phase 2 — pod only
  python3 diagnose_kroko_onnx.py \
      --onnx-dir /path/to/kroko/weights \
      --icefall-dir /workspace/finetune/icefall \
      --phase2
"""

import argparse
import sys
import json
from pathlib import Path
from collections import defaultdict


# ══════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════

def shape_signature(shapes: list[tuple]) -> str:
    """Canonical fingerprint of a set of tensor shapes (sorted)."""
    return json.dumps(sorted(str(s) for s in shapes))


def count_params(shapes: list[tuple]) -> int:
    import math
    return sum(math.prod(s) if s else 1 for s in shapes)


# ══════════════════════════════════════════════════════════════
# Phase 1 — ONNX inspection
# ══════════════════════════════════════════════════════════════

def inspect_onnx(onnx_path: Path) -> dict:
    """Return {name: shape} for all weight initializers in an ONNX file."""
    try:
        import onnx as onnx_lib
    except ImportError:
        print("ERROR: `onnx` not installed. Run: pip install onnx", file=sys.stderr)
        sys.exit(1)

    model = onnx_lib.load(str(onnx_path))
    weights = {}
    for init in model.graph.initializer:
        weights[init.name] = tuple(init.dims)
    return weights


def infer_zipformer2_config(encoder_weights: dict) -> dict:
    """
    Heuristically infer Zipformer2 hyperparameters from ONNX initializer shapes.

    Zipformer2 weight naming in ONNX export tends to follow the pattern:
      /encoder/zipformer_encoder/...  or numeric node names

    We look for patterns in the shapes to infer:
      - d_model (embedding dim): largest square weight shapes
      - feed-forward dims
      - number of encoder blocks
      - attention head counts
    """
    config = {}

    # Group weights by shape
    by_shape = defaultdict(list)
    for name, shape in encoder_weights.items():
        by_shape[shape].append(name)

    # Find unique 2D square shapes — likely self-attention projections
    square_2d = sorted(
        [s for s in by_shape if len(s) == 2 and s[0] == s[1]],
        reverse=True,
    )
    if square_2d:
        config["candidate_d_model"] = list(square_2d[:5])

    # Find unique 2D shapes — likely linear layer weights
    rect_2d = sorted(
        [s for s in by_shape if len(s) == 2 and s[0] != s[1]],
        key=lambda s: s[0] * s[1],
        reverse=True,
    )
    config["candidate_linear_shapes"] = list(rect_2d[:10])

    # Count total parameters
    all_shapes = list(encoder_weights.values())
    config["total_params"] = count_params(all_shapes)
    config["num_weight_tensors"] = len(encoder_weights)

    # Estimate number of encoder "blocks" by looking for repeated shape groups
    shape_counts = {s: len(names) for s, names in by_shape.items()}
    most_repeated = sorted(shape_counts.items(), key=lambda x: x[1], reverse=True)
    config["most_repeated_shapes"] = [
        {"shape": list(s), "count": c} for s, c in most_repeated[:10]
    ]

    return config


def phase1(onnx_dir: Path):
    print("=" * 60)
    print("  PHASE 1: ONNX Weight Inspection")
    print("=" * 60)

    results = {}
    for component in ("encoder", "decoder", "joiner"):
        path = onnx_dir / f"{component}.onnx"
        if not path.exists():
            print(f"\n  WARNING: {path} not found — skipping")
            continue

        print(f"\n--- {component.upper()} ({path.stat().st_size / 1e6:.1f} MB) ---")
        weights = inspect_onnx(path)
        results[component] = weights

        all_shapes = list(weights.values())
        total_params = count_params(all_shapes)
        print(f"  Weight tensors:  {len(weights)}")
        print(f"  Total params:    {total_params:,}  ({total_params/1e6:.1f}M)")

        # Print all weights sorted by size
        print(f"\n  All weight tensors (sorted by param count):")
        sorted_weights = sorted(weights.items(), key=lambda x: count_params([x[1]]), reverse=True)
        for name, shape in sorted_weights[:50]:
            params = count_params([shape])
            print(f"    {params:>10,}  {str(shape):<30}  {name}")
        if len(sorted_weights) > 50:
            print(f"    ... ({len(sorted_weights) - 50} more)")

        if component == "encoder":
            config = infer_zipformer2_config(weights)
            print(f"\n  Inferred config hints:")
            print(f"    Total params:           {config['total_params']:,}")
            print(f"    Candidate d_model vals: {config['candidate_d_model']}")
            print(f"    Top linear shapes:      {config['candidate_linear_shapes'][:5]}")
            print(f"    Most repeated shapes:")
            for entry in config["most_repeated_shapes"][:5]:
                print(f"      shape={entry['shape']}, count={entry['count']}")

    # ── Quantization analysis ──
    print("\n" + "-" * 40)
    print("  QUANTIZATION ANALYSIS")
    print("-" * 40)
    for component, weights in results.items():
        q_names = [n for n in weights if n.endswith("_quantized")]
        scale_names = [n for n in weights if n.endswith("_scale") or n.endswith("_zero_point")]
        non_q = [n for n in weights if not n.endswith(("_quantized", "_scale", "_zero_point"))]
        print(f"  {component}: {len(q_names)} quantized tensors, "
              f"{len(scale_names)} scale/zp scalars, {len(non_q)} unquantized")

    if any(n.endswith("_quantized") for w in results.values() for n in w):
        print("\n  ⚠️  INT8 QUANTIZATION DETECTED")
        print("  Weights must be dequantized before loading into PyTorch:")
        print("    float_w = (int8_w.astype(float32) - zero_point) * scale")
        print("  This is feasible but adds ~30 min of extraction work.")

    # ── Vocabulary size ──
    print("\n" + "-" * 40)
    print("  VOCABULARY / TOKENIZER")
    print("-" * 40)
    dec = results.get("decoder", {})
    emb_shapes = [s for n, s in dec.items() if "embedding" in n.lower() and "weight" in n.lower()
                  and not n.endswith(("_scale", "_zero_point"))]
    if emb_shapes:
        vocab, d = emb_shapes[0][0], emb_shapes[0][1]
        print(f"  Decoder embedding: {emb_shapes[0]}  →  vocab_size={vocab}, d_model={d}")
        if vocab == 500:
            print("  Vocabulary matches LibriSpeech 500-BPE — encoder+decoder transfer possible")
        else:
            print(f"  Vocabulary ({vocab}) differs from LibriSpeech 500-BPE")
            print("  → Encoder-only transfer is the viable path")
            print("    (pair Kroko encoder with Icefall's 500-token predictor/joiner)")

    # Save results for Phase 2 comparison
    out_path = onnx_dir / "kroko_onnx_weights.json"
    serializable = {
        comp: {name: list(shape) for name, shape in w.items()}
        for comp, w in results.items()
    }
    out_path.write_text(json.dumps(serializable, indent=2))
    print(f"\n  Weight shapes saved to: {out_path}")
    print("  (Phase 2 will use this for comparison against Icefall Zipformer2)")

    return results


# ══════════════════════════════════════════════════════════════
# Phase 2 — Compare against Icefall Zipformer2
# ══════════════════════════════════════════════════════════════

def load_icefall_zipformer2(icefall_dir: Path) -> dict | None:
    """
    Try to instantiate Icefall's Zipformer2 transducer and return its state dict.

    Icefall's Zipformer2 lives in:
      egs/librispeech/ASR/zipformer/  (unified zipformer/zipformer2 code)
      or egs/librispeech/ASR/zipformer2/

    The model config that matches Kroko's ~70M encoder is typically
    the "large" streaming variant:
      --encoder-dim 192,256,256,512,512,512,256
      --encoder-unmasked-dim 192,192,192,192,192,192,192
      --num-encoder-layers 2,2,2,2,2,2,2
      ... (from Icefall's run.sh or RESULTS.md)

    We also try the standard "large" non-streaming config.
    """
    sys.path.insert(0, str(icefall_dir))

    # Try to find the zipformer model file
    for candidate in [
        icefall_dir / "egs/librispeech/ASR/zipformer/zipformer.py",
        icefall_dir / "egs/librispeech/ASR/zipformer2/zipformer.py",
        icefall_dir / "egs/librispeech/ASR/zipformer/model.py",
    ]:
        if candidate.exists():
            print(f"  Found Icefall model file: {candidate}")
            sys.path.insert(0, str(candidate.parent))
            break
    else:
        print("  ERROR: Could not find Icefall Zipformer model file.")
        print("  Searched:")
        print("    egs/librispeech/ASR/zipformer/zipformer.py")
        print("    egs/librispeech/ASR/zipformer2/zipformer.py")
        return None

    # Attempt to instantiate several known Zipformer2 configurations
    # These correspond to the "large streaming" recipe used in published results
    # NOTE: Phase 1 analysis of Kroko encoder.onnx revealed:
    #   - FF shapes: (512, 1920) — suggests feedforward_dim=1920 at d_model=512 level
    #     (standard Icefall large uses 2048; Banafo uses a "modified pipeline")
    #   - d_model fingerprint: 192, 256, 384, 512 — matches Zipformer2 large
    #   - Vocab: 650 tokens (vs LibriSpeech 500) — encoder-only transfer needed
    # We try both the standard 2048 config and the Kroko-inferred 1920 config.
    configs_to_try = [
        {
            "name": "streaming-large standard (feedforward_dim=2048 at d512)",
            "encoder_dim": "192,256,256,512,512,512,256",
            "encoder_unmasked_dim": "192,192,192,192,192,192,192",
            "num_encoder_layers": "2,2,2,2,2,2,2",
            "feedforward_dim": "768,1024,1024,2048,2048,2048,1024",
            "num_heads": "4,4,4,8,8,8,4",
        },
        {
            "name": "streaming-large Kroko-inferred (feedforward_dim=1920 at d512)",
            # 1920 = 512*3.75; 1536=512*3; 1280=384*3.33; 1152=512*2.25 or 384*3
            # Best guesses for all 7 encoder stages from ONNX shape inspection
            "encoder_dim": "192,256,256,512,512,512,256",
            "encoder_unmasked_dim": "192,192,192,192,192,192,192",
            "num_encoder_layers": "2,2,2,2,2,2,2",
            "feedforward_dim": "768,1024,1024,1920,1920,1920,1024",
            "num_heads": "4,4,4,8,8,8,4",
        },
        {
            "name": "streaming-large 6-layer variant",
            "encoder_dim": "256,256,512,512,512,256",
            "encoder_unmasked_dim": "192,192,192,192,192,192",
            "num_encoder_layers": "2,4,4,4,4,2",
            "feedforward_dim": "1024,1024,2048,2048,2048,1024",
            "num_heads": "4,4,8,8,8,4",
        },
    ]

    try:
        import torch
        # Try importing the zipformer module
        import importlib.util
        spec = importlib.util.spec_from_file_location("zipformer", candidate)
        zipformer_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(zipformer_mod)

        for cfg in configs_to_try:
            print(f"\n  Trying config: {cfg['name']} ...")
            try:
                # This will vary by Icefall version — adjust as needed
                model = zipformer_mod.Zipformer2(
                    output_downsampling_factor=2,
                    downsampling_factor=(1, 2, 4, 8, 4, 2, 1),
                    num_encoder_layers=tuple(int(x) for x in cfg["num_encoder_layers"].split(",")),
                    encoder_dim=tuple(int(x) for x in cfg["encoder_dim"].split(",")),
                    encoder_unmasked_dim=tuple(int(x) for x in cfg["encoder_unmasked_dim"].split(",")),
                    query_head_dim=32,
                    pos_head_dim=4,
                    value_head_dim=12,
                    num_heads=tuple(int(x) for x in cfg["num_heads"].split(",")),
                    feedforward_dim=tuple(int(x) for x in cfg["feedforward_dim"].split(",")),
                    cnn_module_kernel=(31, 31, 15, 15, 15, 31, 31),
                )
                state_dict = model.state_dict()
                total = sum(p.numel() for p in model.parameters())
                print(f"    Instantiated! Params: {total:,} ({total/1e6:.1f}M)")
                print(f"    State dict keys: {len(state_dict)}")
                return state_dict
            except Exception as e:
                print(f"    Failed: {e}")

    except Exception as e:
        print(f"  ERROR importing Icefall Zipformer2: {e}")
        print("  Make sure k2 is installed and Icefall is in sys.path")

    return None


def compare_shapes(onnx_weights: dict, pytorch_state: dict) -> dict:
    """
    Compare ONNX initializer shapes against PyTorch state dict shapes.
    Returns a feasibility report.
    """
    onnx_shapes = {tuple(s) for s in onnx_weights.values()}
    pytorch_shapes = {tuple(p.shape) for p in pytorch_state.values()}

    # Shape-level overlap
    shared = onnx_shapes & pytorch_shapes
    onnx_only = onnx_shapes - pytorch_shapes
    pytorch_only = pytorch_shapes - onnx_shapes

    # Count how many PyTorch params have a shape match in ONNX
    matched_params = 0
    total_pytorch_params = 0
    for name, param in pytorch_state.items():
        shape = tuple(param.shape)
        total_pytorch_params += param.numel()
        if shape in onnx_shapes:
            matched_params += param.numel()

    coverage = matched_params / total_pytorch_params if total_pytorch_params > 0 else 0

    return {
        "onnx_unique_shapes": len(onnx_shapes),
        "pytorch_unique_shapes": len(pytorch_shapes),
        "shared_shapes": len(shared),
        "onnx_only_shapes": len(onnx_only),
        "pytorch_only_shapes": len(pytorch_only),
        "shape_overlap_pct": len(shared) / max(len(pytorch_shapes), 1) * 100,
        "param_coverage_pct": coverage * 100,
        "shared_shape_list": sorted(str(s) for s in shared),
        "pytorch_only_shape_list": sorted(str(s) for s in list(pytorch_only)[:20]),
        "onnx_only_shape_list": sorted(str(s) for s in list(onnx_only)[:20]),
    }


def phase2(onnx_dir: Path, icefall_dir: Path):
    print("\n" + "=" * 60)
    print("  PHASE 2: Icefall Zipformer2 Comparison")
    print("=" * 60)

    # Load ONNX shapes from Phase 1 output
    shapes_file = onnx_dir / "kroko_onnx_weights.json"
    if not shapes_file.exists():
        print("  ERROR: Run Phase 1 first (missing kroko_onnx_weights.json)")
        sys.exit(1)

    onnx_all = json.loads(shapes_file.read_text())
    encoder_weights = onnx_all.get("encoder", {})

    print(f"\n  Loaded {len(encoder_weights)} encoder weight tensors from ONNX")

    # Load Icefall Zipformer2
    print("\n  Loading Icefall Zipformer2 model...")
    pytorch_state = load_icefall_zipformer2(icefall_dir)

    if pytorch_state is None:
        print("\n  RESULT: INCONCLUSIVE — could not load Icefall Zipformer2")
        print("  Manual step: inspect the Icefall model class constructor and retry")
        return

    # Compare
    report = compare_shapes(encoder_weights, pytorch_state)

    print("\n" + "-" * 60)
    print("  SHAPE COMPARISON REPORT")
    print("-" * 60)
    print(f"  ONNX unique shapes:     {report['onnx_unique_shapes']}")
    print(f"  PyTorch unique shapes:  {report['pytorch_unique_shapes']}")
    print(f"  Shared shapes:          {report['shared_shapes']}")
    print(f"  Shape overlap:          {report['shape_overlap_pct']:.1f}%")
    print(f"  Param coverage:         {report['param_coverage_pct']:.1f}%")
    print()
    print(f"  Shapes in PyTorch but NOT in ONNX ({len(report['pytorch_only_shape_list'])}):")
    for s in report["pytorch_only_shape_list"][:10]:
        print(f"    {s}")
    print()
    print(f"  Shapes in ONNX but NOT in PyTorch ({len(report['onnx_only_shape_list'])}):")
    for s in report["onnx_only_shape_list"][:10]:
        print(f"    {s}")

    # ── GO / NO-GO recommendation ──
    print("\n" + "=" * 60)
    print("  GO / NO-GO RECOMMENDATION")
    print("=" * 60)

    coverage = report["param_coverage_pct"]
    overlap = report["shape_overlap_pct"]

    if coverage >= 90 and overlap >= 80:
        verdict = "GO"
        reason = (
            f"High shape coverage ({coverage:.0f}%) and overlap ({overlap:.0f}%). "
            "Weight mapping is likely feasible with manual name alignment. "
            "Estimated additional work: 1-2 hours of parameter renaming + verification."
        )
    elif coverage >= 70:
        verdict = "MAYBE"
        reason = (
            f"Moderate coverage ({coverage:.0f}%). Some architectural differences exist "
            "but core weights may map correctly. "
            "Worth inspecting the mismatched shapes before deciding. "
            "Risk: Banafo's 'modifications' may include structural changes."
        )
    else:
        verdict = "NO-GO"
        reason = (
            f"Low shape coverage ({coverage:.0f}%). "
            "Architectural differences are too large for reliable weight mapping. "
            "Fine-tune Standard Zipformer only; submit Kroko zero-shot."
        )

    print(f"\n  VERDICT: {verdict}")
    print(f"  REASON:  {reason}")
    print()

    if verdict == "GO":
        print("  NEXT STEPS if proceeding:")
        print("  1. Extract weight tensors from ONNX:")
        print("       import onnx, numpy as np")
        print("       m = onnx.load('encoder.onnx')")
        print("       weights = {i.name: np.array(i.float_data).reshape(i.dims)")
        print("                  for i in m.graph.initializer}")
        print("  2. Build a name mapping: onnx_name → pytorch_name")
        print("     (sort both by shape, align manually, verify with shape assertion)")
        print("  3. Load into PyTorch model as state dict")
        print("  4. Run one forward pass on a test utterance to confirm correctness")
        print("  5. Proceed with run_finetune.sh using this checkpoint instead of")
        print("     the LibriSpeech Standard checkpoint")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Diagnose Kroko ONNX → Icefall Zipformer2 weight mapping feasibility"
    )
    parser.add_argument("--onnx-dir", type=Path, required=True,
                        help="Directory containing encoder.onnx, decoder.onnx, joiner.onnx")
    parser.add_argument("--icefall-dir", type=Path, default=None,
                        help="Path to cloned Icefall repo (required for --phase2)")
    parser.add_argument("--phase2", action="store_true",
                        help="Run Phase 2 (requires Icefall + k2 installed)")
    args = parser.parse_args()

    onnx_results = phase1(args.onnx_dir)

    if args.phase2:
        if args.icefall_dir is None:
            print("\nERROR: --icefall-dir required for --phase2", file=sys.stderr)
            sys.exit(1)
        if not args.icefall_dir.exists():
            print(f"\nERROR: Icefall dir not found: {args.icefall_dir}", file=sys.stderr)
            sys.exit(1)
        phase2(args.onnx_dir, args.icefall_dir)
    else:
        print("\n" + "=" * 60)
        print("  Phase 1 complete.")
        print("  Review the weight shapes above, then run Phase 2 on the pod:")
        print(f"  python3 diagnose_kroko_onnx.py \\")
        print(f"      --onnx-dir {args.onnx_dir} \\")
        print(f"      --icefall-dir /workspace/finetune/icefall \\")
        print(f"      --phase2")
        print("=" * 60)


if __name__ == "__main__":
    main()
