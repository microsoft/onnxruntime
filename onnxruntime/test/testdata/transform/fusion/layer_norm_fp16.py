# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def GenerateModel(model_name):  # noqa: N802
    # Same decomposed LayerNorm subgraph as layer_norm.onnx, but with float16
    # tensors. Exporters targeting an opset below 17 emit this shape, and it is
    # the case where the missing WebGPU entry in the Level-2 LayerNormFusion
    # allowlist leaves the subgraph unfused.
    nodes = [
        helper.make_node("ReduceMean", ["X"], ["ReduceMean_output"], axes=[-1]),
        helper.make_node("Sub", ["X", "ReduceMean_output"], ["Sub_o"]),
        helper.make_node("Pow", ["Sub_o", "Pow_B"], ["Pow_o"]),
        helper.make_node("ReduceMean", ["Pow_o"], ["rm2_o"], axes=[-1]),
        helper.make_node("Add", ["rm2_o", "add_B"], ["add_o"]),
        helper.make_node("Sqrt", ["add_o"], ["sqrt_o"]),
        helper.make_node("Div", ["Sub_o", "sqrt_o"], ["div_o"]),
        helper.make_node("Mul", ["Scale", "div_o"], ["mul_o"]),
        helper.make_node("Add", ["mul_o", "B"], ["Y"]),
    ]

    initializers = [
        numpy_helper.from_array(np.array([2], dtype=np.float16), "Pow_B"),
        numpy_helper.from_array(np.array([1e-5], dtype=np.float16), "add_B"),
    ]

    graph = helper.make_graph(
        nodes,
        "LayerNormFp16",
        [
            helper.make_tensor_value_info("X", TensorProto.FLOAT16, ["M", "N"]),
            helper.make_tensor_value_info("Scale", TensorProto.FLOAT16, ["N"]),
            helper.make_tensor_value_info("B", TensorProto.FLOAT16, ["N"]),
        ],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT16, ["M", "N"])],
        initializers,
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 10)])
    onnx.save(model, model_name)


if __name__ == "__main__":
    GenerateModel("layer_norm_fp16.onnx")
