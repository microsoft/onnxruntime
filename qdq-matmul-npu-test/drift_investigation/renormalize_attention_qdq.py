#!/usr/bin/env python3
"""Diagnose attention lowering by adding reciprocal scaling around mask-Add QDQ."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
from onnx import helper, numpy_helper

from drift_investigation.qdq_ablation import attention_core_groups
from run_acc import load_inputs, measure_float_output


def renormalize(model: ir.Model, selected_softmax_names: set[str] | None = None) -> tuple[onnx.ModelProto, list[str]]:
    groups = attention_core_groups(model.graph)
    if selected_softmax_names is not None:
        found = {softmax.name for softmax, *_ in groups}
        missing = selected_softmax_names - found
        if missing:
            raise ValueError(f"unknown QDQ-wrapped attention Softmax nodes: {sorted(missing)}")
        groups = [group for group in groups if group[0].name in selected_softmax_names]
    if not groups:
        raise ValueError("no QDQ-wrapped attention groups selected")

    proto = ir.serde.serialize_model(model)
    nodes = {node.name: node for node in proto.graph.node}
    initializers = {value.name: value for value in proto.graph.initializer}
    existing_names = {
        name for node in proto.graph.node for name in (*node.input, *node.output) if name
    } | set(initializers)
    for name in ("diagnostic_qdq_double", "diagnostic_qdq_half"):
        if name in existing_names:
            raise ValueError(f"diagnostic value {name!r} already exists")
    proto.graph.initializer.extend((
        numpy_helper.from_array(np.asarray(2.0, dtype=np.float32), "diagnostic_qdq_double"),
        numpy_helper.from_array(np.asarray(0.5, dtype=np.float32), "diagnostic_qdq_half"),
    ))

    before: dict[str, onnx.NodeProto] = {}
    after: dict[str, onnx.NodeProto] = {}
    selected = []
    old_scales = set()
    for index, (softmax, _, add_pair, _) in enumerate(groups):
        qname, dname = add_pair.quantize.name, add_pair.dequantize.name
        if qname is None or dname is None or qname not in nodes or dname not in nodes:
            raise ValueError(f"{softmax.name}: named Q/DQ nodes were not found")
        q, d = nodes[qname], nodes[dname]
        if (
            q.op_type != "QuantizeLinear" or d.op_type != "DequantizeLinear" or
            len(q.input) != 3 or len(d.input) != 3 or q.output[0] != d.input[0] or
            q.input[1:] != d.input[1:]
        ):
            raise ValueError(f"{softmax.name}: expected a direct QDQ with shared scale and zero point")
        old_scale = q.input[1]
        scale_proto = initializers.get(old_scale)
        if scale_proto is None:
            raise ValueError(f"{softmax.name}: constant QDQ scale {old_scale!r} was not found")
        scale = numpy_helper.to_array(scale_proto)
        if scale.shape != () or scale.dtype != np.float32 or not np.isfinite(scale).all() or scale.item() <= 0:
            raise ValueError(f"{softmax.name}: expected a finite positive scalar float32 QDQ scale")

        scale_name = f"diagnostic_attention_scale_{index}"
        input_name = f"diagnostic_attention_before_qdq_{index}"
        output_name = f"diagnostic_attention_after_dq_{index}"
        for name in (scale_name, input_name, output_name):
            if name in existing_names:
                raise ValueError(f"diagnostic value {name!r} already exists")
        proto.graph.initializer.append(numpy_helper.from_array(scale * np.float32(2), scale_name))
        old_scales.add(old_scale)
        before[qname] = helper.make_node(
            "Mul", [q.input[0], "diagnostic_qdq_double"], [input_name],
            name=f"diagnostic_scale_two_before_add_qdq_{index}",
        )
        after[dname] = helper.make_node(
            "Mul", [output_name, "diagnostic_qdq_half"], [d.output[0]],
            name=f"diagnostic_scale_half_after_add_dq_{index}",
        )
        q.input[0] = input_name
        q.input[1] = scale_name
        d.output[0] = output_name
        d.input[1] = scale_name
        selected.append(softmax.name)

    ordered = []
    for node in proto.graph.node:
        if node.name in before:
            ordered.append(before[node.name])
        ordered.append(node)
        if node.name in after:
            ordered.append(after[node.name])
    proto.graph.ClearField("node")
    proto.graph.node.extend(ordered)
    used = {name for node in proto.graph.node for name in node.input if name}
    graph_inputs = {value.name for value in proto.graph.input}
    kept = [
        value for value in proto.graph.initializer
        if value.name not in old_scales or value.name in used or value.name in graph_inputs
    ]
    proto.graph.ClearField("initializer")
    proto.graph.initializer.extend(kept)
    proto.metadata_props.add(key="diagnostic_attention_qdq_renormalized", value=str(len(selected)))
    return proto, selected


def verify_cpu(source: Path, variant: Path, archives: list[Path], valid_mask_name: str | None) -> None:
    original = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"])
    changed = ort.InferenceSession(str(variant), providers=["CPUExecutionProvider"])
    original_names = [output.name for output in original.get_outputs()]
    if original_names != [output.name for output in changed.get_outputs()]:
        raise ValueError("rewritten model has different graph outputs")
    for archive in archives:
        if not archive.is_file():
            raise ValueError(f"verification inputs do not exist: {archive}")
        with np.load(archive, allow_pickle=False) as data:
            mask_name = valid_mask_name if valid_mask_name in data.files else None
        inputs, _ = load_inputs(original, archive, mask_name)
        for name, expected, actual in zip(original_names, original.run(None, inputs), changed.run(None, inputs)):
            if expected.shape != actual.shape or expected.dtype != actual.dtype:
                raise ValueError(f"{archive}: rewritten output {name!r} has different shape or type")
            if not np.array_equal(expected, actual):
                metrics = measure_float_output(expected, actual, 0.0, 0.0)
                raise RuntimeError(
                    f"{archive}: rewritten CPU output {name!r} changed "
                    f"(MAE {metrics['mean_abs_error']}, max error {metrics['max_abs_error']}); "
                    f"model is retained at {variant} for diagnosis"
                )
        print(f"CPU outputs match bitwise on {archive}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Unmodified QDQ encoder or attention prefix.")
    parser.add_argument("--output", type=Path, required=True, help="New model beside the source external data.")
    parser.add_argument("--softmax-name", action="append", help="Restrict to exact Softmax node names (repeatable).")
    parser.add_argument(
        "--verify-inputs", type=Path, action="append", required=True,
        help="NPZ input on which original and rewritten CPU outputs must match bitwise (repeatable).",
    )
    parser.add_argument("--valid-mask", help="Optional extra boolean array in some input archives.")
    args = parser.parse_args()
    source, output = args.model.resolve(), args.output.resolve()
    manifest = output.with_suffix(".renormalization.json")
    if (
        not source.is_file() or output.exists() or manifest.exists() or
        source == output or output.parent != source.parent
    ):
        raise ValueError("source must exist; model and manifest outputs must be new files beside its external data")
    if args.softmax_name and len(set(args.softmax_name)) != len(args.softmax_name):
        raise ValueError("--softmax-name must not repeat a node")
    if any(not path.is_file() for path in args.verify_inputs):
        raise ValueError("all --verify-inputs archives must exist")
    transformed, selected = renormalize(ir.load(source), set(args.softmax_name) if args.softmax_name else None)
    onnx.save(transformed, output)
    verify_cpu(source, output, args.verify_inputs, args.valid_mask)
    manifest.write_text(
        json.dumps(
            {
                "source": str(source),
                "model": str(output),
                "selected_softmax_nodes": selected,
                "diagnostic_only": True,
                "transform": "Mul(2) -> Add-output QDQ(scale*2) -> Mul(0.5)",
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"Saved QDQ-preserving diagnostic for {len(selected)} attention groups: {output}")


if __name__ == "__main__":
    main()
