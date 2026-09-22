#!/usr/bin/env python3
"""Split a Mobius Gemma 4 vision model before its dynamic spatial pooler."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import onnx
import onnx_ir as ir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Mobius Gemma 4 vision ONNX model.")
    parser.add_argument(
        "--encoder-output",
        type=Path,
        help="Static encoder output model; defaults beside the input model.",
    )
    parser.add_argument(
        "--pooler-output",
        type=Path,
        help="Dynamic pooler/projector output model; defaults beside the input model.",
    )
    parser.add_argument(
        "--pooler-marker",
        default="pooler",
        help="Substring identifying pooler nodes; default: pooler.",
    )
    parser.add_argument(
        "--primary-input",
        default="pixel_values",
        help="Input whose dependency identifies the encoder-to-pooler boundary.",
    )
    parser.add_argument(
        "--boundary-name",
        default="vision_features",
        help="Readable name assigned to the cross-component tensor.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output models and external-data files.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.model.is_file():
        raise ValueError(f"input model does not exist: {args.model}")
    if not args.pooler_marker:
        raise ValueError("--pooler-marker cannot be empty")
    if not args.primary_input:
        raise ValueError("--primary-input cannot be empty")
    if not args.boundary_name:
        raise ValueError("--boundary-name cannot be empty")


def graph_input_dependencies(value: ir.Value, seen: set[ir.Value] | None = None) -> set[str]:
    if seen is None:
        seen = set()
    if value in seen:
        return set()
    seen.add(value)

    if value.is_graph_input():
        return {value.name}

    producer = value.producer()
    if producer is None:
        return set()

    dependencies: set[str] = set()
    for node_input in producer.inputs:
        if node_input is not None:
            dependencies.update(graph_input_dependencies(node_input, seen))
    return dependencies


def ancestor_nodes(outputs: Iterable[ir.Value], stop_values: set[ir.Value]) -> set[ir.Node]:
    selected: set[ir.Node] = set()
    pending = list(outputs)
    while pending:
        value = pending.pop()
        if value in stop_values or value.is_graph_input() or value.is_initializer():
            continue

        producer = value.producer()
        if producer is None or producer in selected:
            continue

        selected.add(producer)
        pending.extend(node_input for node_input in producer.inputs if node_input is not None)
    return selected


def find_value(graph: ir.Graph, name: str) -> ir.Value:
    for value in graph.inputs:
        if value.name == name:
            return value
    for value in graph.outputs:
        if value.name == name:
            return value
    initializer = graph.initializers.get(name)
    if initializer is not None:
        return initializer
    for node in graph:
        for value in (*node.inputs, *node.outputs):
            if value is not None and value.name == name:
                return value
    raise ValueError(f"value {name!r} was not found in graph {graph.name!r}")


def infer_qdq_value_metadata(value: ir.Value) -> None:
    if value.shape is not None and value.type is not None:
        return

    current = value
    while current.producer() is not None and current.producer().op_type in {
        "QuantizeLinear",
        "DequantizeLinear",
    }:
        producer = current.producer()
        data_input = producer.inputs[0]
        if data_input is None:
            break
        if value.shape is None and data_input.shape is not None:
            value.shape = data_input.shape
        if value.type is None:
            if producer.op_type == "DequantizeLinear":
                scale = producer.inputs[1]
                if scale is not None and scale.type is not None:
                    value.type = scale.type
            elif len(producer.inputs) > 2:
                zero_point = producer.inputs[2]
                if zero_point is not None and zero_point.type is not None:
                    value.type = zero_point.type
        current = data_input

    if value.shape is None or value.type is None:
        raise ValueError(
            f"could not infer complete type/shape metadata for boundary value {value.name!r}"
        )


def required_graph_inputs(nodes: set[ir.Node], graph_inputs: Iterable[ir.Value]) -> list[str]:
    used = {
        node_input.name
        for node in nodes
        for node_input in node.inputs
        if node_input is not None and node_input.is_graph_input()
    }
    return [value.name for value in graph_inputs if value.name in used]


def select_components(
    model: ir.Model,
    pooler_marker: str,
    primary_input: str,
) -> tuple[set[int], set[int], str, list[str], list[str]]:
    graph = model.graph
    nodes = list(graph)
    pooler_nodes = {
        node for node in nodes if node.name is not None and pooler_marker in node.name
    }
    if not pooler_nodes:
        raise ValueError(f"no nodes contain pooler marker {pooler_marker!r}")

    pooler_inputs = {
        value
        for node in pooler_nodes
        for value in node.inputs
        if value is not None
        and not value.is_initializer()
        and value.producer() not in pooler_nodes
    }
    boundary_values = [
        value
        for value in pooler_inputs
        if not value.is_graph_input()
        and primary_input in graph_input_dependencies(value)
    ]
    if len(boundary_values) != 1:
        names = sorted(value.name for value in boundary_values)
        raise ValueError(
            f"expected one boundary value depending on {primary_input!r}, found {names}"
        )

    boundary = boundary_values[0]
    infer_qdq_value_metadata(boundary)
    encoder_nodes = ancestor_nodes([boundary], set())
    pooler_component_nodes = ancestor_nodes(graph.outputs, {boundary})
    node_indices = {node: index for index, node in enumerate(nodes)}

    encoder_inputs = required_graph_inputs(encoder_nodes, graph.inputs)
    pooler_inputs = required_graph_inputs(pooler_component_nodes, graph.inputs)
    pooler_inputs.append(boundary.name)

    return (
        {node_indices[node] for node in encoder_nodes},
        {node_indices[node] for node in pooler_component_nodes},
        boundary.name,
        encoder_inputs,
        pooler_inputs,
    )


def prune_component(
    source_model: ir.Model,
    selected_indices: set[int],
    input_names: list[str],
    output_names: list[str],
    boundary_original_name: str,
    boundary_name: str,
    component_name: str,
) -> ir.Model:
    model = source_model.clone(deep_copy=False)
    graph = model.graph
    nodes = list(graph)

    graph.inputs.clear()
    graph.outputs.clear()
    graph.remove(
        [node for index, node in enumerate(nodes) if index not in selected_indices],
        safe=False,
    )

    boundary = find_value(graph, boundary_original_name)
    infer_qdq_value_metadata(boundary)

    boundary_input: ir.Value | None = None
    if boundary_original_name in input_names:
        boundary_input = ir.Value(
            name=boundary_name,
            shape=boundary.shape,
            type=boundary.type,
        )
        for node in graph:
            for index, node_input in enumerate(node.inputs):
                if node_input is boundary:
                    node.replace_input_with(index, boundary_input)
    else:
        boundary.name = boundary_name

    inputs = [
        boundary_input if name == boundary_original_name else find_value(graph, name)
        for name in input_names
    ]
    outputs = [
        boundary if name == boundary_original_name else find_value(graph, name)
        for name in output_names
    ]

    graph.inputs.extend(inputs)
    graph.outputs.extend(outputs)

    used_initializers = {
        value.name
        for node in graph
        for value in node.inputs
        if value is not None and value.is_initializer()
    }
    for initializer_name in list(graph.initializers):
        if initializer_name not in used_initializers:
            graph.initializers.pop(initializer_name)

    graph.sort()
    model.metadata_props["gemma_vision_split_component"] = component_name
    model.metadata_props["gemma_vision_split_boundary"] = boundary_name
    return model


def output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    input_model = args.model.resolve()
    encoder_output = (
        args.encoder_output
        if args.encoder_output is not None
        else input_model.with_name(f"{input_model.stem}_encoder.onnx")
    ).resolve()
    pooler_output = (
        args.pooler_output
        if args.pooler_output is not None
        else input_model.with_name(f"{input_model.stem}_pooler_projector.onnx")
    ).resolve()
    if encoder_output == pooler_output:
        raise ValueError("encoder and pooler outputs must be different files")
    return encoder_output, pooler_output


def save_component(model: ir.Model, path: Path, overwrite: bool) -> None:
    external_data_name = f"{path.name}.data"
    external_data_path = path.parent / external_data_name
    if not overwrite:
        existing = [candidate for candidate in (path, external_data_path) if candidate.exists()]
        if existing:
            raise ValueError(f"output already exists: {existing[0]}")

    path.parent.mkdir(parents=True, exist_ok=True)
    ir.save(model, path, external_data=external_data_name)
    if model.ir_version <= onnx.IR_VERSION:
        onnx.checker.check_model(path)
    else:
        print(
            f"Skipped ONNX checker for {path.name}: model IR version "
            f"{model.ir_version} exceeds checker support {onnx.IR_VERSION}."
        )


def describe_component(label: str, model: ir.Model, path: Path) -> None:
    graph = model.graph
    print(f"{label}: {path}")
    print(f"  Nodes:        {graph.num_nodes()}")
    print(f"  Inputs:       {[value.name for value in graph.inputs]}")
    print(f"  Outputs:      {[value.name for value in graph.outputs]}")
    print(f"  Initializers: {len(graph.initializers)}")


def split_components(
    source_model: ir.Model,
    pooler_marker: str = "pooler",
    primary_input: str = "pixel_values",
    boundary_name: str = "vision_features",
) -> tuple[ir.Model, ir.Model]:
    (
        encoder_indices,
        pooler_indices,
        boundary_original_name,
        encoder_inputs,
        pooler_inputs,
    ) = select_components(source_model, pooler_marker, primary_input)

    encoder_model = prune_component(
        source_model,
        encoder_indices,
        encoder_inputs,
        [boundary_original_name],
        boundary_original_name,
        boundary_name,
        "encoder",
    )
    pooler_model = prune_component(
        source_model,
        pooler_indices,
        pooler_inputs,
        [value.name for value in source_model.graph.outputs],
        boundary_original_name,
        boundary_name,
        "pooler_projector",
    )
    return encoder_model, pooler_model


def main() -> None:
    args = parse_args()
    validate_args(args)
    encoder_output, pooler_output = output_paths(args)

    encoder_model, pooler_model = split_components(
        ir.load(args.model.resolve()),
        args.pooler_marker,
        args.primary_input,
        args.boundary_name,
    )
    save_component(encoder_model, encoder_output, args.overwrite)
    save_component(pooler_model, pooler_output, args.overwrite)
    describe_component("Encoder model", encoder_model, encoder_output)
    describe_component("Pooler/projector model", pooler_model, pooler_output)


if __name__ == "__main__":
    main()
