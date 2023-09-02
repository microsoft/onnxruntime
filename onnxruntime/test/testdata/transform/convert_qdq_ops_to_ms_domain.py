# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""
Loads a model and updates the domain of QuantizeLinear and DequantizeLinear nodes to 'com.microsoft'.
Optionally updates zero-points to 16bit data types.

This is used to create models for testing QDQ transformations with the contrib QDQ ops.

Usage:
python3 convert_qdq_ops_to_ms_domain.py --input_model <input onnx model> --output_model <output model> --use_16bit_qdq

Models created with this script:
- qdq_with_multi_consumer_dq_nodes.fixed.qdq_contrib.onnx
- fusion/constant_folding_dequantizelinear.qdq_contrib.onnx
- fusion/constant_folding_qdq_node_unit.qdq_contrib.onnx
- fusion/constant_folding_qdq_node_unit.graph_output.qdq_contrib.onnx
"""
import argparse
import os
import struct
import sys
from typing import Dict

import onnx

QDQ_OPS = ("QuantizeLinear", "DequantizeLinear")


def convert_zero_point_to_16bit(name_to_initializer: Dict[str, onnx.TensorProto], node: onnx.NodeProto):
    zp_input = node.input[2] if len(node.input) > 2 else None

    if zp_input in name_to_initializer:
        initializer = name_to_initializer[zp_input]

        # Do not support external data
        if initializer.HasField("data_location") and initializer.data_location == onnx.TensorProto.EXTERNAL:
            raise Exception("Do not support zero-point initializers with external data")

        byte_order = ">" if sys.byteorder == "big" else "<"

        # Convert uint8 to uint16, int8 to int16
        if initializer.data_type == onnx.TensorProto.UINT8:
            if initializer.HasField("raw_data"):
                num_zps = len(initializer.raw_data)  # Number of zero-point byte values

                # Extract uint8 zero-points as int32s
                zp_int32s = struct.unpack(f"{byte_order}{num_zps}B", initializer.raw_data)

                # Repack int32 zero-points as uint16s
                initializer.raw_data = struct.pack(f"{byte_order}{num_zps}H", *zp_int32s)

            initializer.data_type = onnx.TensorProto.UINT16
        elif initializer.data_type == onnx.TensorProto.INT8:
            if initializer.HasField("raw_data"):
                num_zps = len(initializer.raw_data)  # Number of zero-point values

                # Extract int8 zero-points as int32s
                zp_int32s = struct.unpack(f"{byte_order}{num_zps}b", initializer.raw_data)

                # Repack int32 zero-points as int16s
                initializer.raw_data = struct.pack(f"{byte_order}{num_zps}h", *zp_int32s)

            initializer.data_type = onnx.TensorProto.INT16
    else:
        raise Exception("Only support Q/DQ ops with explicit zero-point inputs")


def update_qdq_node_domains(graph: onnx.GraphProto, use_16bit_qdq: bool):
    name_to_initializer = {initializer.name: initializer for initializer in graph.initializer}

    for node in graph.node:
        # Handle subgraphs:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                update_qdq_node_domains(attr.g, use_16bit_qdq)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for subgraph in attr.graphs:
                    update_qdq_node_domains(subgraph, use_16bit_qdq)

        # Update Q/DQ domains
        if node.op_type in QDQ_OPS:
            node.domain = "com.microsoft"

            if use_16bit_qdq:
                convert_zero_point_to_16bit(name_to_initializer, node)


def main():
    parser = argparse.ArgumentParser(description="Convert Q/DQ ops to com.microsoft domain (or 16-bit)")
    parser.add_argument("--input_model", type=str, required=True, help="Input onnx model path")
    parser.add_argument("--output_model", type=str, required=False, help="Output onnx model path")
    parser.add_argument("--use_16bit_qdq", required=False, action="store_true", help="Convert to 16-bit QDQ")

    args = parser.parse_args()

    model = onnx.load(args.input_model)

    has_ms_domain = False
    for opset in model.opset_import:
        if opset.domain == "com.microsoft":
            has_ms_domain = True
            break

    if not has_ms_domain:
        model.opset_import.extend([onnx.helper.make_opsetid("com.microsoft", 1)])

    update_qdq_node_domains(model.graph, args.use_16bit_qdq)
    onnx.checker.check_model(model, True)

    output_model_path = args.output_model
    if not output_model_path:
        base_model_name = os.path.splitext(args.input_model)[0]
        suffix = ".qdq16_contrib" if args.use_16bit_qdq else ".qdq_contrib"
        output_model_path = base_model_name + suffix + ".onnx"

    onnx.save_model(model, output_model_path)
    print(f"[INFO] Saved model: {output_model_path}")


if __name__ == "__main__":
    main()
