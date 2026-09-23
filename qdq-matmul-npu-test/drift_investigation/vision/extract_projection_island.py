"""Extract a projection MatMul with activation QDQ and weight dequantization."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx_ir as ir

from split_gemma_vision_pooler import find_value, infer_qdq_value_metadata, save_component


def _only_consumer(value: ir.Value, op_type: str) -> ir.Node:
    uses = [use.node for use in value.uses() if use.idx == 0 and use.node.op_type == op_type]
    if len(uses) != 1:
        raise ValueError(f"{value.name}: expected one {op_type} consumer, found {len(uses)}")
    return uses[0]


def _producer(value: ir.Value | None, op_type: str) -> ir.Node:
    if value is None or value.producer() is None or value.producer().op_type != op_type:
        raise ValueError(f"expected {op_type} producer for {value.name if value else 'absent input'}")
    return value.producer()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Source ONNX model.")
    parser.add_argument("--layer", type=int, help="Gemma layer number used to derive the MatMul name fragment.")
    parser.add_argument("--projection", choices=("q", "k", "v", "o"), default="o")
    parser.add_argument(
        "--matmul-name-contains",
        help="Select the unique MatMul whose name contains this text instead of deriving a Gemma name from --layer.",
    )
    parser.add_argument("--input-name", default="activation")
    parser.add_argument("--output-name", default="projection_output")
    parser.add_argument("--output", type=Path, required=True, help="New ONNX model path.")
    args = parser.parse_args()
    if (args.layer is None) == (args.matmul_name_contains is None):
        parser.error("specify exactly one of --layer or --matmul-name-contains")

    source = args.source.resolve()
    output = args.output.resolve()
    if not source.is_file() or output.exists() or source == output:
        raise ValueError("source must exist and output must be a new model")
    name_fragment = (
        args.matmul_name_contains
        if args.matmul_name_contains is not None
        else f"/layers.{args.layer}/self_attn/{args.projection}_proj/MatMul_node_"
    )
    model = ir.load(source)
    graph = model.graph
    matches = [
        node
        for node in graph
        if node.op_type == "MatMul" and node.name is not None and name_fragment in node.name
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one matching projection MatMul for {name_fragment!r}, got {len(matches)}")
    matmul = matches[0]
    activation_dq = _producer(matmul.inputs[0], "DequantizeLinear")
    activation_q = _producer(activation_dq.inputs[0], "QuantizeLinear")
    weight_dq = _producer(matmul.inputs[1], "DequantizeLinear")
    output_q = _only_consumer(matmul.outputs[0], "QuantizeLinear")
    output_dq = _only_consumer(output_q.outputs[0], "DequantizeLinear")
    original_activation = activation_q.inputs[0]
    if original_activation is None or original_activation.is_initializer():
        raise ValueError("projection activation must be a nonconstant float tensor")
    infer_qdq_value_metadata(original_activation)
    original_output = output_dq.outputs[0]
    infer_qdq_value_metadata(original_output)
    selected = {activation_q, activation_dq, weight_dq, matmul, output_q, output_dq}
    selected_indices = {index for index, node in enumerate(graph) if node in selected}

    reduced = model.clone(deep_copy=False)
    graph = reduced.graph
    nodes = list(graph)
    graph.inputs.clear()
    graph.outputs.clear()
    graph.remove([node for index, node in enumerate(nodes) if index not in selected_indices], safe=False)
    boundary = find_value(graph, original_activation.name)
    new_input = ir.Value(name=args.input_name, type=original_activation.type, shape=original_activation.shape)
    for node in graph:
        for index, value in enumerate(node.inputs):
            if value is boundary:
                node.replace_input_with(index, new_input)
    graph.inputs.append(new_input)
    result = find_value(graph, original_output.name)
    result.name = args.output_name
    graph.outputs.append(result)
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
    save_component(reduced, output, False)
    print(
        f"Saved {output}: {graph.num_nodes()} nodes; original activation={original_activation.name}; "
        f"original output={original_output.name}"
    )


if __name__ == "__main__":
    main()
