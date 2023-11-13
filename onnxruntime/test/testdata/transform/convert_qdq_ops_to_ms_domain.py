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
- qdq_with_multi_consumer_dq_nodes.fixed.qdq16_contrib.onnx
- fusion/constant_folding_dequantizelinear.qdq_contrib.onnx
- fusion/constant_folding_dequantizelinear.qdq16_contrib.onnx
- fusion/constant_folding_qdq_node_unit.qdq_contrib.onnx
- fusion/constant_folding_qdq_node_unit.qdq16_contrib.onnx
- fusion/constant_folding_qdq_node_unit.graph_output.qdq_contrib.onnx
- fusion/constant_folding_qdq_node_unit.graph_output.qdq16_contrib.onnx
"""
from __future__ import annotations

import argparse
import os
import struct
import sys

import onnx
from onnx import shape_inference

QDQ_OPS = ("QuantizeLinear", "DequantizeLinear")
QDQ_CONVERT_TYPES = {onnx.TensorProto.UINT8: onnx.TensorProto.UINT16, onnx.TensorProto.INT8: onnx.TensorProto.INT16}
TYPE_TO_STRUCT_LABEL = {
    onnx.TensorProto.UINT8: "B",
    onnx.TensorProto.INT8: "b",
    onnx.TensorProto.UINT16: "H",
    onnx.TensorProto.INT16: "h",
}


def convert_initializer_to_16bits(initializer: onnx.TensorProto, target_type: onnx.TensorProto.DataType):
    byte_order = ">" if sys.byteorder == "big" else "<"
    byte_label = TYPE_TO_STRUCT_LABEL[initializer.data_type]
    short_label = TYPE_TO_STRUCT_LABEL[target_type]

    # Do not support external data
    if initializer.HasField("data_location") and initializer.data_location == onnx.TensorProto.EXTERNAL:
        raise Exception("Do not support initializers with external data")

    # Need to convert raw_data bytes to 16-bit values.
    # NOTE: For tensors that use .int32_data instead of .raw_data, we don't need any special handling
    # other than updating the data type. This is because the upper 24 bits are already cleared to zero.
    if initializer.HasField("raw_data"):
        num_byte_vals = len(initializer.raw_data)

        # Extract 8-bit values as int32s
        int32_vals = struct.unpack(f"{byte_order}{num_byte_vals}{byte_label}", initializer.raw_data)

        # Repack int32 values as 16-bit values
        initializer.raw_data = struct.pack(f"{byte_order}{num_byte_vals}{short_label}", *int32_vals)

    initializer.data_type = target_type


def convert_qdq_op_to_16bit(
    name_to_initializer: dict[str, onnx.TensorProto],
    name_to_values: dict[str, onnx.ValueInfoProto],
    name_to_inputs: dict[str, onnx.ValueInfoProto],
    name_to_outputs: dict[str, onnx.ValueInfoProto],
    node: onnx.NodeProto,
):
    zp_input = node.input[2] if len(node.input) > 2 else None

    if zp_input in name_to_initializer:
        zp_initializer = name_to_initializer[zp_input]

        zp_target_type = QDQ_CONVERT_TYPES.get(zp_initializer.data_type)
        if zp_target_type:
            convert_initializer_to_16bits(zp_initializer, zp_target_type)

        if node.op_type == "DequantizeLinear":
            input0 = node.input[0]

            if input0 in name_to_initializer:
                input_initializer = name_to_initializer[input0]
                input_target_type = QDQ_CONVERT_TYPES.get(input_initializer.data_type)
                if input_target_type:
                    convert_initializer_to_16bits(input_initializer, input_target_type)
            elif input0 in name_to_values:
                input_val = name_to_values[input0]
                input_target_type = QDQ_CONVERT_TYPES.get(input_val.type.tensor_type.elem_type)
                if input_target_type:
                    input_val.type.tensor_type.elem_type = input_target_type
            elif input0 in name_to_inputs:
                input_val = name_to_inputs[input0]
                input_target_type = QDQ_CONVERT_TYPES.get(input_val.type.tensor_type.elem_type)
                if input_target_type:
                    input_val.type.tensor_type.elem_type = input_target_type
        else:
            # QuantizeLinear
            output0 = node.output[0]

            if output0 in name_to_values:
                output_val = name_to_values[output0]
                output_target_type = QDQ_CONVERT_TYPES.get(output_val.type.tensor_type.elem_type)
                if output_target_type:
                    output_val.type.tensor_type.elem_type = output_target_type
            elif output0 in name_to_outputs:
                output_val = name_to_outputs[output0]
                output_target_type = QDQ_CONVERT_TYPES.get(output_val.type.tensor_type.elem_type)
                if output_target_type:
                    output_val.type.tensor_type.elem_type = output_target_type
    else:
        raise Exception("Only support Q/DQ ops with explicit zero-point inputs")


def update_qdq_node_domains(graph: onnx.GraphProto, use_16bit_qdq: bool):
    name_to_initializer = {initializer.name: initializer for initializer in graph.initializer}
    name_to_values = {value.name: value for value in graph.value_info}
    name_to_inputs = {g_input.name: g_input for g_input in graph.input}
    name_to_outputs = {g_output.name: g_output for g_output in graph.output}

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
                convert_qdq_op_to_16bit(name_to_initializer, name_to_values, name_to_inputs, name_to_outputs, node)


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
    model = shape_inference.infer_shapes(model)
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
