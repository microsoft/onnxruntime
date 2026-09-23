#!/usr/bin/env python3
"""Extract an attention-score island and replay real upstream tensors on any EP."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort

from drift_investigation.qdq_ablation import attention_core_groups
from run_acc import load_inputs
from split_gemma_vision_pooler import find_value, infer_qdq_value_metadata, save_component


@dataclass(frozen=True)
class IslandBoundary:
    inputs: dict[str, ir.Value]
    output: ir.Value
    nodes: set[ir.Node]


def _required_input(value: ir.Value | None, softmax_name: str) -> ir.Value:
    if value is None:
        raise ValueError(f"{softmax_name}: attention island has an absent input")
    return value


def select_island(model: ir.Model, softmax_name: str) -> IslandBoundary:
    matches = [group for group in attention_core_groups(model.graph) if group[0].name == softmax_name]
    if len(matches) != 1:
        raise ValueError(f"expected one attention pattern with Softmax name {softmax_name!r}, found {len(matches)}")
    softmax, score_pair, add_pair, probability_pair = matches[0]
    score_source = _required_input(score_pair.quantize.inputs[0], softmax_name)
    add_source = _required_input(add_pair.quantize.inputs[0], softmax_name)
    qk_matmul = score_source.producer()
    mask_add = add_source.producer()
    consumers = [
        use.node for use in probability_pair.dequantize.outputs[0].uses()
        if use.node.op_type == "MatMul" and use.idx == 0
    ]
    if qk_matmul is None or mask_add is None or len(consumers) != 1:
        raise ValueError(f"{softmax_name}: attention MatMul or probability consumer is ambiguous")
    value_matmul = consumers[0]
    mask_inputs = [value for value in mask_add.inputs if value is not score_pair.dequantize.outputs[0]]
    if len(mask_inputs) != 1:
        raise ValueError(f"{softmax_name}: expected one separate attention mask input")

    inputs: dict[str, ir.Value] = {
        "query": _required_input(qk_matmul.inputs[0], softmax_name),
        "key": _required_input(qk_matmul.inputs[1], softmax_name),
        "value": _required_input(value_matmul.inputs[1], softmax_name),
        "attention_mask": _required_input(mask_inputs[0], softmax_name),
    }
    for value in inputs.values():
        infer_qdq_value_metadata(value)

    output = value_matmul.outputs[0]
    if output.type is None or output.shape is None:
        for use in output.uses():
            if use.node.op_type == "Identity" and use.node.outputs[0] in model.graph.outputs:
                output.type = output.type or use.node.outputs[0].type
                output.shape = output.shape or use.node.outputs[0].shape
                break
    if output.type is None or output.shape is None:
        raise ValueError(f"{softmax_name}: attention island output needs type and shape metadata")

    nodes = {
        qk_matmul, mask_add, softmax, value_matmul,
        *(node for pair in (score_pair, add_pair, probability_pair)
          for node in (pair.quantize, pair.dequantize)),
    }
    return IslandBoundary(inputs, output, nodes)


def make_tapped_model(source: ir.Model, boundary: IslandBoundary) -> ir.Model:
    model = source.clone(deep_copy=False)
    graph = model.graph
    for value in (*boundary.inputs.values(), boundary.output):
        cloned = find_value(graph, value.name)
        if cloned in graph.outputs:
            raise ValueError(f"{value.name!r} is already a graph output")
        if cloned.shape is None:
            cloned.shape = value.shape
        if cloned.type is None:
            cloned.type = value.type
        graph.outputs.append(cloned)
    return model


def make_island_model(source: ir.Model, boundary: IslandBoundary) -> ir.Model:
    selected_indices = {index for index, node in enumerate(source.graph) if node in boundary.nodes}
    model = source.clone(deep_copy=False)
    graph = model.graph
    nodes = list(graph)
    graph.inputs.clear()
    graph.outputs.clear()
    graph.remove([node for index, node in enumerate(nodes) if index not in selected_indices], safe=False)

    for input_name, source_value in boundary.inputs.items():
        old_value = find_value(graph, source_value.name)
        new_value = ir.Value(name=input_name, shape=source_value.shape, type=source_value.type)
        for node in graph:
            for index, value in enumerate(node.inputs):
                if value is old_value:
                    node.replace_input_with(index, new_value)
        graph.inputs.append(new_value)

    output = find_value(graph, boundary.output.name)
    output.name = "attention_context"
    graph.outputs.append(output)
    used_initializers = {
        value.name for node in graph for value in node.inputs
        if value is not None and value.is_initializer()
    }
    for name in list(graph.initializers):
        if name not in used_initializers:
            graph.initializers.pop(name)
    graph.sort()
    return model


def capture_inputs(
    tapped_path: Path,
    island_path: Path,
    inputs_path: Path,
    baseline_arrays: Path,
    island_inputs_path: Path,
    boundary: IslandBoundary,
    valid_mask: str | None,
) -> None:
    tapped = ort.InferenceSession(str(tapped_path), providers=["CPUExecutionProvider"])
    inputs, _ = load_inputs(tapped, inputs_path, valid_mask)
    names = [value.name for value in tapped.get_outputs()]
    outputs = dict(zip(names, tapped.run(None, inputs)))
    with np.load(baseline_arrays, allow_pickle=False) as baseline:
        original_outputs = [name for name in names if name not in {
            *(value.name for value in boundary.inputs.values()), boundary.output.name,
        }]
        for index, name in enumerate(original_outputs):
            if not np.array_equal(outputs[name], baseline[f"cpu_{index}"]):
                raise RuntimeError(f"adding diagnostic outputs changed original CPU output {name!r}")

    island_inputs = {
        short_name: outputs[value.name] for short_name, value in boundary.inputs.items()
    }
    replay = ort.InferenceSession(str(island_path), providers=["CPUExecutionProvider"])
    context = replay.run(None, island_inputs)[0]
    np.testing.assert_allclose(context, outputs[boundary.output.name], rtol=1e-6, atol=1e-6)
    np.savez_compressed(island_inputs_path, **island_inputs)
    print(
        f"Verified original CPU output and attention-island replay; saved {island_inputs_path}"
        f" with inputs {[(name, array.shape) for name, array in island_inputs.items()]}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Original (not QDQ-ablated) ONNX model.")
    parser.add_argument("--softmax", required=True, help="Exact Softmax node name identifying an attention island.")
    parser.add_argument("--tap-output", type=Path, required=True, help="New full model exposing the island inputs.")
    parser.add_argument("--island-output", type=Path, required=True, help="New standalone attention island.")
    parser.add_argument("--inputs", type=Path, required=True, help="Saved full-model input .npz.")
    parser.add_argument("--valid-mask", help="Extra boolean array in --inputs, not a model input.")
    parser.add_argument("--baseline-arrays", type=Path, required=True, help="Original run_acc --output-arrays .npz.")
    parser.add_argument("--island-inputs", type=Path, required=True, help="New .npz for the standalone island.")
    args = parser.parse_args()
    source_path = args.model.resolve()
    paths = [source_path, args.tap_output.resolve(), args.island_output.resolve(), args.island_inputs.resolve()]
    if not source_path.is_file() or not args.inputs.is_file() or not args.baseline_arrays.is_file():
        raise ValueError("the source model, input archive, and baseline output arrays must exist")
    if len(set(paths)) != len(paths) or any(path.exists() for path in paths[1:]):
        raise ValueError("outputs must be new and distinct from each other and from the source model")
    if args.tap_output.resolve().parent != source_path.parent:
        raise ValueError("tapped model must be beside the source so its external weights remain accessible")

    source = ir.load(source_path)
    source.graph.sort()
    boundary = select_island(source, args.softmax)
    island = make_island_model(source, boundary)
    save_component(island, args.island_output.resolve(), overwrite=False)
    onnx.save(ir.serde.serialize_model(make_tapped_model(source, boundary)), args.tap_output)
    capture_inputs(
        args.tap_output.resolve(), args.island_output.resolve(), args.inputs.resolve(),
        args.baseline_arrays.resolve(), args.island_inputs.resolve(), boundary, args.valid_mask,
    )


if __name__ == "__main__":
    main()
