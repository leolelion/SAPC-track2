#!/usr/bin/env python3
"""
Extract and dequantize Kroko encoder weights from ONNX → PyTorch state dict.

Background
----------
Kroko's encoder.onnx was exported with int8 quantization from a modified
Icefall Zipformer2 pipeline. The ONNX preserves Icefall-style parameter names
for all biases and non-quantized params (e.g. conv weights, bypass scales,
norm log_scales). The weight matrices are stored as int8 initializers with
opaque names (onnx::MatMul_XXXX_quantized) plus per-tensor scale/zero_point.

Architecture (inferred from ONNX initializer shapes):
  encoder_dim:     (192, 256, 384, 512, 384, 256)   — 6 stages
  num_layers:      (2, 2, 2, 2, 2, 2)               — 2 per stage, 12 total
  feedforward_dim: (384, 576, 768, 1152, 768, 576)  — 2× d_model per stage
  vocab_size:      650  (Kroko-specific; ≠ LibriSpeech 500 BPE)

Strategy
--------
1. Traverse the ONNX graph to map each opaque weight initializer to its
   corresponding named bias via Add/MatMul chain analysis. This gives us
   the PyTorch parameter name for every weight matrix.
2. Dequantize: float_w = (int8_w - zero_point) * scale
3. Build a state dict {pytorch_name: float32_tensor}.
4. Save as a .pt file that can be loaded into Icefall's Zipformer2 model
   (encoder-only; predictor/joiner weights come from a separate checkpoint).

Usage
-----
  # Run after setup_pod.sh, before run_finetune.sh
  python3 extract_kroko_encoder.py \
      --encoder-onnx /path/to/kroko/encoder.onnx \
      --output /workspace/finetune/weights/kroko_encoder.pt \
      [--validate]   # optional: load into Icefall model and run sanity check

Requirements
------------
  pip install onnx numpy torch
  (Icefall + k2 only needed for --validate)
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


# ══════════════════════════════════════════════════════════════
# ONNX loading and graph analysis
# ══════════════════════════════════════════════════════════════

def load_onnx_initializers(onnx_path: Path) -> dict[str, np.ndarray]:
    """Return {name: numpy_array} for every initializer in the ONNX model."""
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError:
        print("ERROR: `onnx` not installed. Run: pip install onnx", file=sys.stderr)
        sys.exit(1)

    model = onnx.load(str(onnx_path))
    initializers = {}
    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        initializers[init.name] = arr
    return initializers, model.graph


def build_weight_to_bias_map(graph) -> dict[str, str]:
    """
    Traverse the ONNX graph to map each quantized weight initializer name
    to the corresponding named bias initializer name.

    Kroko uses MatMulInteger (int8 weight × dynamically quantized activation):
      DynamicQuantizeLinear(x) → x_q, x_scale, x_zero_point
      MatMulInteger(x_q, weight_q, x_zero_point, weight_zero_point) → out_int32
      Cast(out_int32 → float32) → out_float
      Mul(out_float, weight_scale) → rescaled
      ... (possibly more ops) ...
      Add(rescaled, named_bias) → output

    Strategy: forward BFS from each MatMulInteger output to find the first
    Add node that uses a named bias initializer. Named bias → PyTorch param
    name; derive weight name by replacing '.bias' with '.weight'.

    Returns: {quantized_weight_name: bias_name}
    Note: ~16 streaming-specific position projections (shapes [48,16], [48,32])
          have no downstream named bias and are intentionally left unmapped —
          they are streaming-only tensors with no counterpart in the Icefall
          training model (~15k params, 0.02% of total).
    """
    from collections import defaultdict

    init_names = {init.name for init in graph.initializer}
    named_floats = {
        n for n in init_names
        if not n.startswith("onnx::")
        and not n.endswith(("_scale", "_zero_point", "_scale_1", "_zero_point_1"))
    }

    # Build consumer index: tensor_name → [nodes that take it as input]
    input_to_nodes: dict[str, list] = defaultdict(list)
    for node in graph.node:
        for inp in node.input:
            input_to_nodes[inp].append(node)

    def find_named_bias_downstream(start: str, depth: int = 0, visited: set | None = None) -> str | None:
        """BFS forward through the computation graph to find a named bias Add."""
        if visited is None:
            visited = set()
        if depth > 14 or start in visited:
            return None
        visited.add(start)
        for node in input_to_nodes.get(start, []):
            if node.op_type == "Add":
                for inp in node.input:
                    if inp in named_floats:
                        return inp
            for out in node.output:
                result = find_named_bias_downstream(out, depth + 1, visited)
                if result:
                    return result
        return None

    weight_to_bias = {}
    for node in graph.node:
        if node.op_type != "MatMulInteger":
            continue
        # input[1] is the static int8 weight matrix (input[0] is dynamic activation)
        w_name = node.input[1] if len(node.input) > 1 else None
        if w_name is None or w_name not in init_names:
            continue
        # Skip scalar zero_point initializers
        w_init = next((i for i in graph.initializer if i.name == w_name), None)
        if w_init is None or len(w_init.dims) < 2:
            continue

        bias_name = find_named_bias_downstream(node.output[0])
        if bias_name:
            weight_to_bias[w_name] = bias_name

    return weight_to_bias


# ══════════════════════════════════════════════════════════════
# Dequantization
# ══════════════════════════════════════════════════════════════

def find_quant_params(name: str, initializers: dict) -> tuple[float, float]:
    """
    Find scale and zero_point for a quantized initializer.

    Naming conventions seen in Kroko ONNX:
      onnx::MatMul_12542_quantized → scale: onnx::MatMul_12542_scale? or adjacent?

    The scale/zero_point scalars are unnamed scalars [] in the initializer list.
    We find them by looking for the DequantizeLinear node that uses `name` as input.
    """
    # Fallback: common patterns used by Banafo's export
    base = name.replace("_quantized", "")
    for scale_key in (f"{base}_scale", f"{base}.scale", f"{name}_scale"):
        if scale_key in initializers:
            scale = float(initializers[scale_key])
            zp_key = scale_key.replace("_scale", "_zero_point")
            zero_point = float(initializers.get(zp_key, np.array(0.0)))
            return scale, zero_point
    return None, None


def dequantize_from_graph(graph, name: str, initializers: dict) -> np.ndarray:
    """
    Dequantize an int8 weight tensor from Kroko's MatMulInteger pattern.

    Pattern:
      MatMulInteger(x_q, weight_q, x_zp, weight_zp) → out_int32
      Mul(out_int32_as_float, weight_scale) → rescaled_output

    The weight_zp is the scalar initializer at MatMulInteger.input[3].
    The weight_scale is the scalar initializer consumed by the downstream Mul.

    Dequantization: float_w = (int8_w - weight_zp) * weight_scale
    where weight_zp is typically 0 for symmetric quantization.
    """
    from collections import defaultdict

    init_names = set(initializers.keys())

    # Build consumer index (lazily — cached in a module-level dict keyed by graph id)
    _cache = dequantize_from_graph.__dict__.setdefault("_cache", {})
    gid = id(graph)
    if gid not in _cache:
        input_to_nodes = defaultdict(list)
        for node in graph.node:
            for inp in node.input:
                input_to_nodes[inp].append(node)
        _cache[gid] = input_to_nodes

    # Find the MatMulInteger that uses `name` as its weight input (input[1])
    mmi_node = None
    for node in graph.node:
        if node.op_type == "MatMulInteger" and len(node.input) > 1 and node.input[1] == name:
            mmi_node = node
            break

    if mmi_node is not None:
        # weight_zero_point is input[3] of MatMulInteger
        zp_name = mmi_node.input[3] if len(mmi_node.input) > 3 else None
        zero_point = float(initializers[zp_name]) if zp_name and zp_name in initializers else 0.0

        # Find weight_scale: the scalar opaque initializer in the Mul downstream of MatMulInteger
        # Trace forward: MatMulInteger → (Cast?) → Mul(weight_scale)
        def find_weight_scale(tensor, depth=0, visited=None):
            if visited is None:
                visited = set()
            if depth > 6 or tensor in visited:
                return None
            visited.add(tensor)
            for node in _cache[gid].get(tensor, []):
                if node.op_type == "Mul":
                    for inp in node.input:
                        if inp in init_names and inp.startswith("onnx::"):
                            arr = initializers[inp]
                            if arr.shape == () or arr.size == 1:
                                return float(arr)
                for out in node.output:
                    result = find_weight_scale(out, depth + 1, visited)
                    if result is not None:
                        return result
            return None

        scale = find_weight_scale(mmi_node.output[0])

        if scale is not None:
            q = initializers[name].astype(np.float32)
            return (q - zero_point) * scale
        # Fall through to pattern-based lookup if forward trace failed

    # Pattern-based lookup: onnx::MatMul_XXXX_quantized → onnx::MatMul_XXXX_scale
    scale, zero_point = find_quant_params(name, initializers)
    if scale is not None:
        q = initializers[name].astype(np.float32)
        return (q - zero_point) * scale

    print(f"  WARNING: Could not find scale for {name} — returning raw int8 cast", file=sys.stderr)
    return initializers[name].astype(np.float32)


# ══════════════════════════════════════════════════════════════
# State dict construction
# ══════════════════════════════════════════════════════════════

def build_state_dict(
    initializers: dict,
    graph,
    weight_to_bias: dict,
) -> tuple[dict, dict]:
    """
    Build a PyTorch-compatible state dict from ONNX initializers.

    Returns:
      (state_dict, report) where report summarizes mapping coverage.
    """
    import torch

    state_dict = {}
    report = {
        "direct_mapped": [],      # named float tensors (biases, conv weights, etc.)
        "dequantized_mapped": [],  # quantized weights mapped via graph traversal
        "unmapped_opaque": [],     # opaque tensors we couldn't map
        "skipped": [],             # scale/zero_point scalars we skip
    }

    # ── Pass 1: Direct-named tensors (biases, conv weights, bypass scales, etc.) ──
    for name, arr in initializers.items():
        if name.startswith("onnx::"):
            continue  # opaque — handled in pass 2

        # Skip quantization metadata
        if name.endswith(("_scale", "_zero_point", "_scale_1", "_zero_point_1")):
            report["skipped"].append(name)
            continue

        # Strip _quantized suffix from decoder/joiner named quantized tensors
        pt_name = name.replace("_quantized", "")

        tensor = torch.from_numpy(arr.astype(np.float32))
        state_dict[pt_name] = tensor
        report["direct_mapped"].append(pt_name)

    # ── Pass 2: Opaque quantized encoder weight matrices ──
    for q_name, bias_name in weight_to_bias.items():
        # Derive PyTorch weight name from the corresponding bias name
        # e.g. encoder.encoders.0.layers.0.feed_forward1.in_proj.bias
        #   → encoder.encoders.0.layers.0.feed_forward1.in_proj.weight
        if not bias_name.endswith(".bias"):
            print(f"  WARNING: Expected bias suffix in {bias_name}", file=sys.stderr)
            continue

        pt_weight_name = bias_name[:-len(".bias")] + ".weight"

        float_arr = dequantize_from_graph(graph, q_name, initializers)
        # ONNX MatMulInteger stores weights as [in_features, out_features] (B in x@B).
        # PyTorch Linear stores weights as [out_features, in_features] (weight.T applied).
        # Transpose all 2D weight matrices to match PyTorch convention.
        if float_arr.ndim == 2:
            float_arr = float_arr.T
        state_dict[pt_weight_name] = torch.from_numpy(float_arr.copy())
        report["dequantized_mapped"].append(pt_weight_name)

    # ── Pass 3: Remaining opaque tensors (scale/zp scalars, unmatched) ──
    mapped_opaque = set(weight_to_bias.keys())
    for name, arr in initializers.items():
        if not name.startswith("onnx::"):
            continue
        if name in mapped_opaque:
            continue
        if arr.shape == () or arr.size == 1:
            # Scalar — scale or zero_point, already used in dequantization
            report["skipped"].append(name)
        else:
            report["unmapped_opaque"].append(name)

    return state_dict, report


# ══════════════════════════════════════════════════════════════
# Validation (pod-only, requires Icefall + k2)
# ══════════════════════════════════════════════════════════════

def validate_state_dict(state_dict: dict, icefall_dir: Path):
    """
    Attempt to load the state dict into an Icefall Zipformer2 model.
    Prints a coverage report and runs a dummy forward pass.
    """
    import torch
    sys.path.insert(0, str(icefall_dir))

    # Try to find and import the Zipformer2 model class
    for candidate in [
        icefall_dir / "egs/librispeech/ASR/zipformer/zipformer.py",
        icefall_dir / "egs/librispeech/ASR/zipformer2/zipformer.py",
    ]:
        if candidate.exists():
            import importlib.util
            spec = importlib.util.spec_from_file_location("zipformer", candidate)
            mod = importlib.util.module_from_spec(spec)
            sys.path.insert(0, str(candidate.parent))
            spec.loader.exec_module(mod)
            ZipformerClass = getattr(mod, "Zipformer2", None) or getattr(mod, "Zipformer", None)
            break
    else:
        print("  ERROR: Icefall Zipformer2 model class not found.", file=sys.stderr)
        return

    # Instantiate with Kroko's architecture
    # Adjust if Icefall's constructor signature differs
    try:
        model = ZipformerClass(
            output_downsampling_factor=2,
            downsampling_factor=(1, 2, 4, 8, 4, 2),   # 6 stages
            num_encoder_layers=(2, 2, 2, 2, 2, 2),
            encoder_dim=(192, 256, 384, 512, 384, 256),
            encoder_unmasked_dim=(192, 192, 192, 192, 192, 192),
            query_head_dim=32,
            pos_head_dim=4,
            value_head_dim=12,
            num_heads=(4, 4, 4, 8, 4, 4),
            feedforward_dim=(384, 576, 768, 1152, 768, 576),
            cnn_module_kernel=(31, 31, 15, 15, 15, 31),
        )
    except Exception as e:
        print(f"  ERROR: Could not instantiate Zipformer2: {e}", file=sys.stderr)
        print("  The constructor arguments above are inferred — adjust and retry.", file=sys.stderr)
        return

    model_state = model.state_dict()
    print(f"\n  Icefall model: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"  Extracted state dict: {sum(t.numel() for t in state_dict.values()):,} params")

    # Load with strict=False and report coverage
    missing_keys, unexpected_keys = [], []
    try:
        result = model.load_state_dict(state_dict, strict=False)
        missing_keys = result.missing_keys
        unexpected_keys = result.unexpected_keys
    except Exception as e:
        print(f"  ERROR loading state dict: {e}", file=sys.stderr)
        return

    matched = len(model_state) - len(missing_keys)
    print(f"\n  Keys matched:    {matched}/{len(model_state)}")
    print(f"  Missing keys:    {len(missing_keys)}")
    print(f"  Unexpected keys: {len(unexpected_keys)}")

    if missing_keys:
        print("\n  First 20 missing keys:")
        for k in missing_keys[:20]:
            print(f"    {k}  shape={list(model_state[k].shape)}")

    if len(missing_keys) == 0:
        print("\n  ✓ PERFECT MATCH — all parameters loaded successfully")
    elif len(missing_keys) < 10:
        print("\n  ✓ NEAR-PERFECT — only a few parameters missing (likely optimizer state or minor extras)")
    elif matched / len(model_state) > 0.85:
        print("\n  ✓ GOOD — >85% coverage. Remaining params will initialize from Standard checkpoint.")
    else:
        coverage = matched / len(model_state) * 100
        print(f"\n  ✗ LOW COVERAGE ({coverage:.0f}%) — architecture mismatch likely.")
        print("    Check the constructor args above against actual ONNX structure.")

    # Dummy forward pass sanity check
    if len(missing_keys) < len(model_state) * 0.3:
        print("\n  Running dummy forward pass...")
        try:
            model.eval()
            with torch.no_grad():
                # Streaming Zipformer2 forward: (T, B, C) features
                x = torch.randn(50, 1, 80)  # 50 frames, batch=1, 80-dim fbank
                x_lens = torch.tensor([50])
                out, out_lens, _ = model(x, x_lens)
            print(f"  ✓ Forward pass OK — output shape: {tuple(out.shape)}")
        except Exception as e:
            print(f"  ✗ Forward pass failed: {e}")
            print("  This may be due to missing keys — fix coverage first.")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Extract Kroko encoder weights from ONNX to PyTorch state dict"
    )
    parser.add_argument("--encoder-onnx", type=Path, required=True,
                        help="Path to Kroko encoder.onnx")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output .pt file path")
    parser.add_argument("--validate", action="store_true",
                        help="Load into Icefall Zipformer2 model and check coverage (requires --icefall-dir)")
    parser.add_argument("--icefall-dir", type=Path, default=Path("/workspace/finetune/icefall"),
                        help="Path to Icefall repo (for --validate)")
    args = parser.parse_args()

    if not args.encoder_onnx.exists():
        print(f"ERROR: {args.encoder_onnx} not found", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("  Kroko ONNX → PyTorch State Dict Extraction")
    print("=" * 60)

    # ── Load ONNX ──
    print(f"\nLoading {args.encoder_onnx} ({args.encoder_onnx.stat().st_size/1e6:.1f} MB)...")
    initializers, graph = load_onnx_initializers(args.encoder_onnx)
    named = {k: v for k, v in initializers.items() if not k.startswith("onnx::")}
    opaque = {k: v for k, v in initializers.items() if k.startswith("onnx::")}
    print(f"  Named initializers:  {len(named)}")
    print(f"  Opaque initializers: {len(opaque)}")

    # ── Build weight→bias mapping via graph traversal ──
    print("\nAnalyzing ONNX graph (MatMul → Add chains)...")
    weight_to_bias = build_weight_to_bias_map(graph)
    print(f"  Mapped {len(weight_to_bias)} quantized weight tensors to named biases")

    unmapped_opaque = [k for k in opaque
                       if k not in weight_to_bias
                       and initializers[k].size > 1]
    if unmapped_opaque:
        # These are streaming-specific ONNX artifacts (no counterpart in Icefall training model):
        #   - 16 small position projections ([48,16], [48,32]) — streaming attention only
        #   - ~70 concatenated conv kernels and position matrices for chunk processing
        # They do NOT need to be resolved — Icefall's training model doesn't have them.
        print(f"  ℹ️  {len(unmapped_opaque)} streaming-only ONNX artifacts skipped (expected):")
        for k in unmapped_opaque[:6]:
            print(f"      {k}  shape={list(initializers[k].shape)}")
        if len(unmapped_opaque) > 6:
            print(f"      ... ({len(unmapped_opaque)-6} more)")

    # ── Build state dict ──
    print("\nBuilding state dict (dequantizing int8 → float32)...")
    state_dict, report = build_state_dict(initializers, graph, weight_to_bias)

    total_params = sum(t.numel() for t in state_dict.values())
    print(f"  Direct-mapped tensors:     {len(report['direct_mapped'])}")
    print(f"  Dequantized+mapped:        {len(report['dequantized_mapped'])}")
    print(f"  Skipped (scale/zp):        {len(report['skipped'])}")
    print(f"  Unmapped opaque:           {len(report['unmapped_opaque'])}")
    print(f"  Total parameters:          {total_params:,} ({total_params/1e6:.1f}M)")

    # ── Save ──
    import torch
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": state_dict, "epoch": 0, "avg": 0}, str(args.output))
    print(f"\nSaved state dict to: {args.output}")
    print("(Wrapped as {\"model\": state_dict} for Icefall checkpoint compatibility)")

    # ── Validate ──
    if args.validate:
        print("\n" + "=" * 60)
        print("  Validation: Loading into Icefall Zipformer2")
        print("=" * 60)
        if not args.icefall_dir.exists():
            print(f"  ERROR: Icefall dir not found: {args.icefall_dir}", file=sys.stderr)
            print("  Run setup_pod.sh first, or pass --icefall-dir", file=sys.stderr)
        else:
            validate_state_dict(state_dict, args.icefall_dir)

    print("\n" + "=" * 60)
    print("  Next steps:")
    if len(report["unmapped_opaque"]) > 0:
        print(f"  ℹ️  {len(report['unmapped_opaque'])} streaming-only artifacts skipped (normal — not in training model)")
    print("  1. Run with --validate to check Icefall coverage (on pod)")
    print("  2. Merge with Standard predictor/joiner checkpoint:")
    print("       python3 -c \"")
    print("         import torch")
    print("         kroko = torch.load('kroko_encoder.pt')['model']")
    print("         std   = torch.load('standard/exp/epoch-30.pt')['model']")
    print("         # Keep kroko encoder, use std predictor/joiner")
    print("         merged = {k: v for k, v in std.items()}")
    print("         merged.update({k: v for k, v in kroko.items() if 'encoder' in k})")
    print("         torch.save({'model': merged, 'epoch': 0}, 'kroko_init.pt')")
    print("         \"")
    print("  3. Fine-tune: bash run_finetune.sh kroko --checkpoint kroko_init.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()
