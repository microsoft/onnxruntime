# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import onnx
from onnx import helper
graph = helper.make_graph(
    [helper.make_node("Foo", ["input1"], ["output1"], "", "", "com.examples")],
    "external_custom_op_example_model",
    [helper.make_tensor_value_info("input1", helper.TensorProto.FLOAT, [2,2])],
    [helper.make_tensor_value_info("output1", helper.TensorProto.FLOAT, [2,2])],
    []
)
model = helper.make_model(graph)
opset = model.opset_import.add()
opset.version = 1
opset.domain = "com.examples"
onnx.checker.check_model(model)
onnx.save(model, "model.onnx")
