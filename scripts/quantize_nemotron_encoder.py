#!/usr/bin/env python3
"""Build a dynamic-int8 encoder variant of a Nemotron SAPC2 submission tree."""

from __future__ import annotations

import argparse
import collections
import hashlib
import inspect
import json
import shutil
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel_external_files(onnx_module: Any, model_path: Path) -> list[str]:
    model = onnx_module.load(str(model_path), load_external_data=False)
    out = set()
    for init in model.graph.initializer:
        if init.data_location != onnx_module.TensorProto.EXTERNAL:
            continue
        for entry in init.external_data:
            if entry.key == "location":
                out.add(entry.value)
    return sorted(out)


def dtype_name(onnx_module: Any, elem_type: int) -> str:
    try:
        return onnx_module.TensorProto.DataType.Name(elem_type)
    except Exception:
        return str(elem_type)


def dim_value(dim: Any) -> Any:
    if dim.dim_param:
        return dim.dim_param
    if dim.dim_value:
        return dim.dim_value
    return None


def value_info(onnx_module: Any, vi: Any) -> dict[str, Any]:
    tensor_type = vi.type.tensor_type
    shape = [dim_value(dim) for dim in tensor_type.shape.dim]
    return {"name": vi.name, "dtype": dtype_name(onnx_module, tensor_type.elem_type), "shape": shape}


def inspect_model(onnx_module: Any, ort_module: Any, model_path: Path, *, load_session: bool) -> dict[str, Any]:
    model = onnx_module.load(str(model_path), load_external_data=False)
    op_counts = collections.Counter(node.op_type for node in model.graph.node)
    init_dtypes = collections.Counter(dtype_name(onnx_module, init.data_type) for init in model.graph.initializer)
    external = rel_external_files(onnx_module, model_path)
    parent = model_path.parent
    external_sizes = {name: (parent / name).stat().st_size for name in external if (parent / name).exists()}
    info: dict[str, Any] = {
        "path": str(model_path),
        "sha256": sha256(model_path) if model_path.exists() else None,
        "model_bytes": model_path.stat().st_size if model_path.exists() else None,
        "external_files": external,
        "external_bytes": sum(external_sizes.values()),
        "missing_external_files": [name for name in external if not (parent / name).exists()],
        "ir_version": model.ir_version,
        "opset_imports": {op.domain or "": op.version for op in model.opset_import},
        "node_count": len(model.graph.node),
        "op_counts": dict(sorted(op_counts.items())),
        "initializer_count": len(model.graph.initializer),
        "initializer_dtype_counts": dict(sorted(init_dtypes.items())),
        "graph_inputs": [value_info(onnx_module, vi) for vi in model.graph.input],
        "graph_outputs": [value_info(onnx_module, vi) for vi in model.graph.output],
        "quantized_op_counts": {
            key: op_counts.get(key, 0)
            for key in (
                "DynamicQuantizeLinear",
                "DynamicQuantizeMatMul",
                "MatMulInteger",
                "MatMulIntegerToFloat",
                "QLinearMatMul",
                "QuantizeLinear",
                "DequantizeLinear",
            )
        },
    }
    if load_session:
        so = ort_module.SessionOptions()
        so.graph_optimization_level = ort_module.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess = ort_module.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
        info["session_providers"] = sess.get_providers()
        info["session_inputs"] = [
            {"name": x.name, "type": x.type, "shape": list(x.shape)} for x in sess.get_inputs()
        ]
        info["session_outputs"] = [
            {"name": x.name, "type": x.type, "shape": list(x.shape)} for x in sess.get_outputs()
        ]
    return info


def call_quantize_dynamic(quantize_dynamic: Any, quant_type: Any, **kwargs: Any) -> None:
    sig = inspect.signature(quantize_dynamic)
    filtered = {key: val for key, val in kwargs.items() if key in sig.parameters}
    if "weight_type" in sig.parameters and "weight_type" not in filtered:
        filtered["weight_type"] = quant_type.QInt8
    quantize_dynamic(**filtered)


def remove_encoder_payload(onnx_module: Any, source_encoder: Path, output_encoder: Path, output_decoder: Path) -> list[str]:
    old_encoder_external = set(rel_external_files(onnx_module, source_encoder))
    decoder_external = set(rel_external_files(onnx_module, output_decoder))
    removed = []
    if output_encoder.exists():
        output_encoder.unlink()
        removed.append(str(output_encoder))
    for rel in sorted(old_encoder_external - decoder_external):
        path = output_encoder.parent / rel
        try:
            resolved = path.resolve()
            resolved.relative_to(output_encoder.parent.resolve())
        except Exception:
            continue
        if path.exists() and path.is_file():
            path.unlink()
            removed.append(str(path))
    return removed


def build(args: argparse.Namespace) -> dict[str, Any]:
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic

    source = args.source_submission.resolve()
    output = args.output_submission.resolve()
    work_dir = args.work_dir.resolve()
    source_weights = source / "weights"
    output_weights = output / "weights"
    source_encoder = source_weights / "encoder_model.onnx"
    source_decoder = source_weights / "decoder_model.onnx"
    if not source_encoder.is_file() or not source_decoder.is_file():
        raise FileNotFoundError(f"Missing source ONNX files under {source_weights}")

    if output.exists():
        shutil.rmtree(output)
    shutil.copytree(source, output)
    output_encoder = output_weights / "encoder_model.onnx"
    output_decoder = output_weights / "decoder_model.onnx"
    work_dir.mkdir(parents=True, exist_ok=True)

    before_encoder = inspect_model(onnx, ort, source_encoder, load_session=args.load_source_session)
    before_decoder = inspect_model(onnx, ort, source_decoder, load_session=args.load_source_session)
    removed = remove_encoder_payload(onnx, source_encoder, output_encoder, output_decoder)

    call_quantize_dynamic(
        quantize_dynamic,
        QuantType,
        model_input=str(source_encoder),
        model_output=str(output_encoder),
        op_types_to_quantize=args.op_types,
        per_channel=args.per_channel,
        reduce_range=args.reduce_range,
        use_external_data_format=True,
    )

    after_encoder = inspect_model(onnx, ort, output_encoder, load_session=True)
    after_decoder = inspect_model(onnx, ort, output_decoder, load_session=True)
    input_names_match = [x["name"] for x in before_encoder["graph_inputs"]] == [
        x["name"] for x in after_encoder["graph_inputs"]
    ]
    output_names_match = [x["name"] for x in before_encoder["graph_outputs"]] == [
        x["name"] for x in after_encoder["graph_outputs"]
    ]
    quantized_counts = after_encoder["quantized_op_counts"]
    quantized_total = sum(quantized_counts.values())
    result = {
        "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__,
        "available_providers": ort.get_available_providers(),
        "source_submission": str(source),
        "output_submission": str(output),
        "op_types_requested": args.op_types,
        "per_channel": args.per_channel,
        "reduce_range": args.reduce_range,
        "removed_encoder_payload": removed,
        "before_encoder": before_encoder,
        "after_encoder": after_encoder,
        "before_decoder": before_decoder,
        "after_decoder": after_decoder,
        "checks": {
            "encoder_input_names_match": input_names_match,
            "encoder_output_names_match": output_names_match,
            "decoder_sha256_match": before_decoder["sha256"] == after_decoder["sha256"],
            "after_encoder_has_quantized_ops": quantized_total > 0,
            "after_encoder_session_loads": bool(after_encoder.get("session_providers")),
            "after_decoder_session_loads": bool(after_decoder.get("session_providers")),
        },
    }
    failed = [key for key, ok in result["checks"].items() if not ok]
    result["failed_checks"] = failed
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if failed:
        raise RuntimeError("Quantization preflight failed checks: " + ", ".join(failed))
    print(json.dumps({"checks": result["checks"], "after_encoder_quantized_ops": quantized_counts}, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-submission", type=Path, required=True)
    parser.add_argument("--output-submission", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--op-types", nargs="+", default=["MatMul", "Gemm"])
    parser.add_argument("--per-channel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reduce-range", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--load-source-session", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
