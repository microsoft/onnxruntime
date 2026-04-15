"""Generate a simple Mul ONNX model for testing.

Produces mul.onnx in the same directory as this script.
The model computes z = x * y (element-wise) for float32 tensors of shape [2, 3].
"""

import os

import onnx
from onnx import TensorProto, helper

X = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
Y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])
Z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [2, 3])

mul_node = helper.make_node("Mul", inputs=["x", "y"], outputs=["z"])

graph = helper.make_graph([mul_node], "mul_graph", [X, Y], [Z])
model = helper.make_model(graph, producer_name="onnxruntime-webgpu-ep-test")
model.opset_import[0].version = 13

onnx.checker.check_model(model)

output_path = os.path.join(os.path.dirname(__file__), "mul.onnx")
onnx.save(model, output_path)
print(f"Saved {output_path}")
