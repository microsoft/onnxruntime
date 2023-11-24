# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import onnx
from onnx import TensorProto, helper


def GenerateModel(model_name, valid):  # noqa: N802
    nodes = [
        helper.make_node("Transpose", ["input_0"], ["transposed_input_0"], perm=[2, 1, 3, 0]),
        helper.make_node("Add", ["transposed_input_0", "input_1"], ["output"]),
    ]

    if valid:
        inputs = [
            helper.make_tensor_value_info("input_0", TensorProto.FLOAT, [1, 1, 3, 3]),
            helper.make_tensor_value_info("input_1", TensorProto.FLOAT, [3, 1, 3, 1]),
        ]
        outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT, [3, 1, 3, 1])]
    else:
        inputs = [
            helper.make_tensor_value_info("input_0", TensorProto.FLOAT, [1, 2, 3, 3]),
            helper.make_tensor_value_info("input_1", TensorProto.FLOAT, [3, 2, 3, 1]),
        ]
        outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT, [3, 2, 3, 1])]

    graph = helper.make_graph(
        nodes,
        "TransposeAndAdd",  # name
        inputs,
        outputs,
        [],
    )

    model = helper.make_model(graph)
    onnx.save(model, model_name)


GenerateModel("transpose_to_reshape_valid.onnx", True)
GenerateModel("transpose_to_reshape_invalid.onnx", False)
