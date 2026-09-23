"""Cut an attention ONNX graph at multiple named float tensors."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx_ir as ir

from split_gemma_vision_pooler import (
    ancestor_nodes,
    find_value,
    infer_qdq_value_metadata,
    required_graph_inputs,
    save_component,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Source ONNX model.")
    parser.add_argument("--cut", action="append", required=True, help="SOURCE_VALUE=NEW_INPUT_NAME (repeatable).")
    parser.add_argument("--output", type=Path, required=True, help="New ONNX model path.")
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if not source.is_file() or output.exists() or output == source:
        raise ValueError("source must exist and output must be a new model")
    cuts: dict[str, str] = {}
    for entry in args.cut:
        original, separator, new_name = entry.partition("=")
        if not separator or not original or not new_name or original in cuts:
            raise ValueError(f"invalid or repeated cut {entry!r}")
        cuts[original] = new_name
    if len(set(cuts.values())) != len(cuts):
        raise ValueError("cut input names must be unique")

    model = ir.load(source)
    graph = model.graph
    if len(graph.outputs) != 1:
        raise ValueError("source must have exactly one output")
    output_name = graph.outputs[0].name
    boundaries = {name: find_value(graph, name) for name in cuts}
    for value in boundaries.values():
        if value.is_graph_input():
            raise ValueError(f"cut boundary is already a graph input: {value.name}")
        infer_qdq_value_metadata(value)
    selected = ancestor_nodes(graph.outputs, set(boundaries.values()))
    selected_indices = {index for index, node in enumerate(graph) if node in selected}
    input_names = required_graph_inputs(selected, graph.inputs)
    if not selected_indices or set(input_names) & set(cuts.values()):
        raise ValueError("no graph remains or a cut input conflicts with an existing input")

    result = model.clone(deep_copy=False)
    graph = result.graph
    nodes = list(graph)
    graph.inputs.clear()
    graph.outputs.clear()
    graph.remove([node for index, node in enumerate(nodes) if index not in selected_indices], safe=False)
    for name in input_names:
        graph.inputs.append(find_value(graph, name))
    for original, new_name in cuts.items():
        old = find_value(graph, original)
        source_value = boundaries[original]
        replacement = ir.Value(name=new_name, type=source_value.type, shape=source_value.shape)
        for node in graph:
            for index, node_input in enumerate(node.inputs):
                if node_input is old:
                    node.replace_input_with(index, replacement)
        graph.inputs.append(replacement)
    graph.outputs.append(find_value(graph, output_name))
    used = {
        value.name
        for node in graph
        for value in node.inputs
        if value is not None and value.is_initializer()
    }
    for name in list(graph.initializers):
        if name not in used:
            graph.initializers.pop(name)
    graph.sort()
    save_component(result, output, False)
    print(f"Saved {output}: {graph.num_nodes()} nodes, inputs={[value.name for value in graph.inputs]}")


if __name__ == "__main__":
    main()
