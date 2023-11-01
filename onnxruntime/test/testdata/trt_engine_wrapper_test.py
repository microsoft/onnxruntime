#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import onnx
from onnx import TensorProto, helper


def generate_model(model_name):
    nodes = [
        helper.make_node(
            "EPContext",
            ["data"],
            ["resnetv17_dense0_fwd"],
            "EPContext",
            domain="com.microsoft",
            embed_mode=0,
            ep_cache_context="trt_engine_wrapper.engine"
        ),
    ]

    graph = helper.make_graph(
        nodes,
        "trt_engine_wrapper",
        [  # input
            helper.make_tensor_value_info("data", TensorProto.FLOAT, ["N", 3, 224, 224]),
        ],
        [  # output
            helper.make_tensor_value_info("resnetv17_dense0_fwd", TensorProto.FLOAT, ["N", 1000]),
        ],
    )

    model = helper.make_model(graph)
    onnx.save(model, model_name)


if __name__ == "__main__":
    generate_model("trt_engine_wrapper.onnx")
