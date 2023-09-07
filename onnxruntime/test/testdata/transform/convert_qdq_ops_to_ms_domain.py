"""
Loads a model and updates the domain of QuantizeLinear and DequantizeLinear nodes to 'com.microsoft'.
This is used to create models for testing QDQ transformations with the contrib QDQ ops.

Usage: python3 convert_qdq_ops_to_ms_domain.py <onnx model>

Models created with this script:
- qdq_with_multi_consumer_dq_nodes.fixed.qdq_contrib.onnx
- fusion/constant_folding_dequantizelinear.qdq_contrib.onnx
- fusion/constant_folding_qdq_node_unit.qdq_contrib.onnx
- fusion/constant_folding_qdq_node_unit.graph_output.qdq_contrib.onnx
"""
import os
import sys

import onnx

QDQ_OPS = ("QuantizeLinear", "DequantizeLinear")


def print_usage(prog_name: str):
    """
    Prints the program's command-line arguments and usage.
    """

    print(f"Usage: {prog_name} <onnx model>")


def update_qdq_node_domains(graph):
    """
    Updates the domain of all QuantizeLinear and DequantizeLinear nodes
    in a graph to 'com.microsoft'.
    """

    for node in graph.node:
        # Handle subgraphs:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                update_qdq_node_domains(attr.g)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for subgraph in attr.graphs:
                    update_qdq_node_domains(subgraph)

        # Update Q/DQ domains
        if node.op_type in QDQ_OPS:
            node.domain = "com.microsoft"


def main():
    prog_name, *argv = sys.argv

    if len(argv) != 1:
        print_usage(prog_name)
        sys.exit(1)

    model = onnx.load(argv[0])

    has_ms_domain = False
    for opset in model.opset_import:
        if opset.domain == "com.microsoft":
            has_ms_domain = True
            break

    if not has_ms_domain:
        model.opset_import.extend([onnx.helper.make_opsetid("com.microsoft", 1)])

    update_qdq_node_domains(model.graph)
    onnx.checker.check_model(model, True)
    base_model_name = os.path.splitext(argv[0])[0]
    onnx.save_model(model, base_model_name + ".qdq_contrib.onnx")


if __name__ == "__main__":
    main()
