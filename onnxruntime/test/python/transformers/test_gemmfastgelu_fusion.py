# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import os
import unittest
from typing import List

import numpy as np
import onnx
from onnx import TensorProto, helper
from parity_utilities import find_transformers_source

if find_transformers_source():
    from fusion_options import FusionOptions
    from onnx_model import OnnxModel
    from optimizer import optimize_model
else:
    from onnxruntime.transformers.fusion_options import FusionOptions
    from onnxruntime.transformers.onnx_model import OnnxModel
    from onnxruntime.transformers.optimizer import optimize_model


onnxdomain = onnx.OperatorSetIdProto()
onnxdomain.version = 12
# The empty string ("") or absence of this field implies the operator set that is defined as part of the ONNX specification.
onnxdomain.domain = ""
msdomain = onnx.OperatorSetIdProto()
msdomain.version = 1
msdomain.domain = "com.microsoft"
opsets = [onnxdomain, msdomain]


def float_tensor(name: str, shape: List[int], random=False):
    low = 0.0
    high = 1.0
    total_elements = 1
    for x in shape:
        total_elements *= x
    weights = [np.random.uniform(low, high) for _ in range(total_elements)] if random else [1.0] * total_elements
    return helper.make_tensor(name, TensorProto.FLOAT, shape, weights)


def create_mat_mul_fast_gelu_without_bias(batch_size, m, n, k):
    # MatMul + FastGelu
    nodes = [
        helper.make_node("MatMul", ["input", "matmul_weight"], ["fastgelu_input"], "matmul"),
    ]
    fastgelu_node = helper.make_node("FastGelu", ["fastgelu_input"], ["output"], "fastgelu")
    fastgelu_node.domain = "com.microsoft"
    nodes.append(fastgelu_node)

    initializers = [float_tensor("matmul_weight", [k, n])]  # initializers

    graph = helper.make_graph(
        [node for node in nodes if node],
        "GemmFastGeluNoBiasModel",  # name
        [  # inputs
            helper.make_tensor_value_info(
                "input",
                TensorProto.FLOAT,
                [batch_size, m, k],
            )
        ],
        [  # outputs
            helper.make_tensor_value_info(
                "output",
                TensorProto.FLOAT,
                [batch_size, m, n],
            ),
        ],
        initializers,
    )

    return helper.make_model(graph)


def create_mat_mul_fast_gelu_with_bias(batch_size, m, n, k):
    # MatMul + FastGelu
    nodes = [
        helper.make_node("MatMul", ["input", "matmul_weight"], ["fastgelu_input"], "matmul"),
    ]
    fastgelu_node = helper.make_node("FastGelu", ["fastgelu_input", "fastgelu_bias"], ["output"], "fastgelu")
    fastgelu_node.domain = "com.microsoft"
    nodes.append(fastgelu_node)

    initializers = [float_tensor("matmul_weight", [k, n]), float_tensor("fastgelu_bias", [n])]  # initializers

    graph = helper.make_graph(
        [node for node in nodes if node],
        "GemmFastGeluWithBiasModel",  # name
        [  # inputs
            helper.make_tensor_value_info(
                "input",
                TensorProto.FLOAT,
                [batch_size, m, k],
            )
        ],
        [  # outputs
            helper.make_tensor_value_info(
                "output",
                TensorProto.FLOAT,
                [batch_size, m, n],
            ),
        ],
        initializers,
    )

    return helper.make_model(graph, opset_imports=opsets)


class TestFusion(unittest.TestCase):
    def verify_fusion(self, optimized_model, expected_model_filename):
        optimized_model.topological_sort()

        expected_model_path = os.path.join(os.path.dirname(__file__), "test_data", "models", expected_model_filename)
        expected_model = OnnxModel(onnx.load(expected_model_path))
        expected_model.topological_sort()

        self.assertEqual(str(optimized_model.model.graph), str(expected_model.model.graph))

    def test_gemmfastgelu_fusion_withoutbias(self):
        model = create_mat_mul_fast_gelu_without_bias(32, 128, 64, 1024)
        dir = "."
        model_path = os.path.join(dir, "gemmfastgelu_nobias.onnx")
        onnx.save(model, model_path)

        fusion_opt = FusionOptions("bert")
        fusion_opt.enable_gemm_fast_gelu = True
        optimized_model = optimize_model(input=model_path, optimization_options=fusion_opt)
        os.remove(model_path)

        self.verify_fusion(optimized_model, "gemmfastgelu_nobias_opt.onnx")

    def test_gemmfastgelu_fusion_withbias(self):
        model = create_mat_mul_fast_gelu_with_bias(32, 128, 64, 1024)
        dir = "."
        model_path = os.path.join(dir, "gemmfastgelu_withbias.onnx")
        onnx.save(model, model_path)

        fusion_opt = FusionOptions("bert")
        fusion_opt.enable_gemm_fast_gelu = True
        optimized_model = optimize_model(input=model_path, optimization_options=fusion_opt)

        os.remove(model_path)

        self.verify_fusion(optimized_model, "gemmfastgelu_withbias_opt.onnx")


if __name__ == "__main__":
    unittest.main()
