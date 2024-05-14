#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import onnx
from onnx import TensorProto, helper


def generate_model(model_name):
    nodes = [
        helper.make_node(
            "DisentangledAttention_TRT",
            ["input1", "input2", "input3"],
            ["output"],
            "DisentangledAttention_TRT",
            domain="trt.plugins",
            factor=0.123,
            span=128,
        ),
    ]

    graph = helper.make_graph(
        nodes,
        "trt_plugin_custom_op",
        [  # input
            helper.make_tensor_value_info("input1", TensorProto.FLOAT, [12, 256, 256]),
            helper.make_tensor_value_info("input2", TensorProto.FLOAT, [12, 256, 256]),
            helper.make_tensor_value_info("input3", TensorProto.FLOAT, [12, 256, 256]),
        ],
        [  # output
            helper.make_tensor_value_info("output", TensorProto.FLOAT, [12, 256, 256]),
        ],
    )

    model = helper.make_model(graph)
    onnx.save(model, model_name)


if __name__ == "__main__":
    generate_model("trt_plugin_custom_op_test.onnx")
