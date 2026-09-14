import os
import tempfile

import onnx
from onnx import TensorProto, helper

import onnxruntime as ort


def make_nested_loop(depth):
    if depth == 0:
        body = helper.make_graph(
            [
                helper.make_node("Identity", ["cond_in"], ["cond_out"]),
                helper.make_node("Identity", ["x_in"], ["x_out"]),
            ],
            "base_body",
            [
                helper.make_tensor_value_info("iter", TensorProto.INT64, []),
                helper.make_tensor_value_info("cond_in", TensorProto.BOOL, []),
                helper.make_tensor_value_info("x_in", TensorProto.FLOAT, [1]),
            ],
            [
                helper.make_tensor_value_info("cond_out", TensorProto.BOOL, []),
                helper.make_tensor_value_info("x_out", TensorProto.FLOAT, [1]),
            ],
        )
        return body

    inner_body = make_nested_loop(depth - 1)
    loop_node = helper.make_node(
        "Loop", inputs=["iter", "cond_in", "x_in"], outputs=["x_mid"], body=inner_body, name=f"loop_{depth}"
    )
    body = helper.make_graph(
        [
            loop_node,
            helper.make_node("Identity", ["cond_in"], ["cond_out"]),
            helper.make_node("Identity", ["x_mid"], ["x_out"]),
        ],
        f"body_{depth}",
        [
            helper.make_tensor_value_info("iter", TensorProto.INT64, []),
            helper.make_tensor_value_info("cond_in", TensorProto.BOOL, []),
            helper.make_tensor_value_info("x_in", TensorProto.FLOAT, [1]),
        ],
        [
            helper.make_tensor_value_info("cond_out", TensorProto.BOOL, []),
            helper.make_tensor_value_info("x_out", TensorProto.FLOAT, [1]),
        ],
    )

    return body


model = helper.make_model(make_nested_loop(30))
onnx.checker.check_model(model)

output_path = os.path.join(tempfile.gettempdir(), "30_nested_loops.onnx")
if os.path.exists(output_path):
    os.remove(output_path)

onnx.save(model, output_path)
print(f"Model saved to {output_path}")

try:
    sess = ort.InferenceSession(output_path)
    print("Loaded model successfully")
except Exception as exc:
    print(f"Model was created successfully, but loading failed: {exc}")
