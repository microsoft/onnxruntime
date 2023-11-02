#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import onnx
from onnx import TensorProto, helper


def generate_model(model_name):
    nodes = [
        helper.make_node(
            "Reshape",
            ["data", "shape"],
            ["reshaped"],
            "Reshape",
        ),
    ]

    graph = helper.make_graph(
        nodes,
        "trt_engine_wrapper",
        [  # input
            helper.make_tensor_value_info("data", TensorProto.FLOAT, ["N",2]),
            helper.make_tensor_value_info("shape", TensorProto.INT64, [2,]),
        ],
        [  # output
            helper.make_tensor_value_info("reshaped", TensorProto.FLOAT, [4, 1]),
        ],
    )

    model = helper.make_model(graph)
    onnx.save(model, model_name)


if __name__ == "__main__":
    generate_model("trt_reshape.onnx")
