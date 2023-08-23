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


def print_usage(prog_name: str):
    print(f"Usage: {prog_name} <onnx model>")


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

    for node in model.graph.node:
        if node.op_type in ("QuantizeLinear", "DequantizeLinear"):
            node.domain = "com.microsoft"

    onnx.checker.check_model(model, True)
    base_model_name = os.path.splitext(argv[0])[0]
    onnx.save_model(model, base_model_name + ".qdq_contrib.onnx")


if __name__ == "__main__":
    main()
