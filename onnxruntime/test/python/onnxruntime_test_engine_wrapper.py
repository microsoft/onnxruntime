# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import unittest

import numpy as np
import onnx
from helper import get_name
from onnx import TensorProto, helper

import onnxruntime as onnxrt


class TestInferenceSessionWithCtxNode(unittest.TestCase):
    trt_engine_cache_path_ = "./trt_engine_cache"
    ctx_node_model_name_ = "ctx_node.onnx"

    def test_ctx_node(self):
        if "TensorrtExecutionProvider" in onnxrt.get_available_providers():
            providers = [
                (
                    "TensorrtExecutionProvider",
                    {"trt_engine_cache_enable": True, "trt_engine_cache_path": self.trt_engine_cache_path_},
                )
            ]
            self.run_model(providers)

    def create_ctx_node(self, ctx_embed_mode=0, cache_path=""):
        if ctx_embed_mode:
            # Get engine buffer from engine cache
            with open(cache_path, "rb") as file:
                engine_buffer = file.read()
            ep_cache_context_content = engine_buffer
        else:
            ep_cache_context_content = cache_path

        nodes = [
            helper.make_node(
                "EPContext",
                ["X"],
                ["Y"],
                "EPContext",
                domain="com.microsoft",
                embed_mode=ctx_embed_mode,
                ep_cache_context=ep_cache_context_content,
            ),
        ]

        graph = helper.make_graph(
            nodes,
            "trt_engine_wrapper",
            [  # input
                helper.make_tensor_value_info("X", TensorProto.FLOAT, ["N", 2]),
            ],
            [  # output
                helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["N", 1]),
            ],
        )
        model = helper.make_model(graph)
        onnx.save(model, self.ctx_node_model_name_)

    def run_model(self, providers):
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)

        session = onnxrt.InferenceSession(get_name("matmul_2.onnx"), providers=providers)

        # One regular run to create engine cache
        session.run(
            ["Y"],
            {"X": x},
        )

        cache_name = ""
        for f in os.listdir(self.trt_engine_cache_path_):
            if f.endswith(".engine"):
                cache_name = f
        print(cache_name)

        # Second run to test ctx node with engine cache path
        self.create_ctx_node(cache_path=os.path.join(self.trt_engine_cache_path_, cache_name))
        providers = [("TensorrtExecutionProvider", {})]
        session = onnxrt.InferenceSession(get_name(self.ctx_node_model_name_), providers=providers)
        session.run(
            ["Y"],
            {"X": x},
        )

        # Third run to test ctx node with engine binary content
        self.create_ctx_node(ctx_embed_mode=1, cache_path=os.path.join(self.trt_engine_cache_path_, cache_name))
        session = onnxrt.InferenceSession(get_name(self.ctx_node_model_name_), providers=providers)
        session.run(
            ["Y"],
            {"X": x},
        )


if __name__ == "__main__":
    unittest.main()
