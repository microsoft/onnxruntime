"""Cut a self-contained ONNX attention prefix at one named upstream activation."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx_ir as ir

from split_gemma_vision_pooler import (
    ancestor_nodes,
    find_value,
    infer_qdq_value_metadata,
    prune_component,
    required_graph_inputs,
    save_component,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Source ONNX model.")
    parser.add_argument("--boundary", required=True, help="Existing float tensor to promote to a graph input.")
    parser.add_argument("--input-name", default="hidden_states", help="Name assigned to the promoted input.")
    parser.add_argument("--graph-name", default="attention_prefix", help="Name assigned to the extracted graph.")
    parser.add_argument("--output", type=Path, required=True, help="New ONNX model path.")
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if not source.is_file() or output.exists() or output == source:
        raise ValueError("source must exist and output must be a new model")

    model = ir.load(source)
    graph = model.graph
    if len(graph.outputs) != 1:
        raise ValueError("source must have exactly one graph output")
    boundary = find_value(graph, args.boundary)
    if boundary.is_graph_input():
        raise ValueError("boundary is already a graph input")
    infer_qdq_value_metadata(boundary)
    selected = ancestor_nodes(graph.outputs, {boundary})
    node_indices = {index for index, node in enumerate(graph) if node in selected}
    if not node_indices:
        raise ValueError("no nodes remain after cutting at the boundary")
    input_names = required_graph_inputs(selected, graph.inputs)
    if args.input_name in input_names:
        raise ValueError("new input name conflicts with an existing graph input")
    input_names.append(boundary.name)
    cut = prune_component(
        model,
        node_indices,
        input_names,
        [graph.outputs[0].name],
        boundary.name,
        args.input_name,
        args.graph_name,
    )
    save_component(cut, output, False)
    print(f"Saved {output}: {cut.graph.num_nodes()} nodes, inputs={[value.name for value in cut.graph.inputs]}")


if __name__ == "__main__":
    main()
