import onnx
from onnx import helper, TensorProto
import numpy as np


def make_model():
    graph_def = helper.make_graph(
        nodes=[
            helper.make_node("Cast", ["exponent"], ["exponent_fp16"],
                             name="cast_exponent", to=TensorProto.FLOAT16),
            helper.make_node("Pow", ["x", "exponent_fp16"], ["pow_out"], name="pow"),
            helper.make_node("ReduceMean", ["pow_out"], ["mean_out"],
                             name="reduce_mean", axes=[-1], keepdims=1),
            helper.make_node("Add", ["mean_out", "bias"], ["add_out"], name="add"),
            helper.make_node("Sqrt", ["add_out"], ["sqrt_out"], name="sqrt"),
            helper.make_node("Div", ["x", "sqrt_out"], ["div_out"], name="div"),
            helper.make_node("Mul", ["div_out", "scale"], ["y"], name="mul"),
        ],
        name="simplified_layer_norm_fp16_pow_edge",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, 4, 8]),
            helper.make_tensor_value_info("bias", TensorProto.FLOAT16, [1, 4, 1]),
            helper.make_tensor_value_info("scale", TensorProto.FLOAT16, [1, 4, 1]),
        ],
        outputs=[
            helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, 4, 8]),
        ],
        initializer=[
            helper.make_tensor("exponent", TensorProto.FLOAT, [1],
                               np.array([2.0], dtype=np.float32)),
        ],
    )

    opsets = [helper.make_opsetid("", 12)]
    model = helper.make_model(graph_def, producer_name="onnx-test-gen",
                              opset_imports=opsets)
    onnx.checker.check_model(model)
    return model


if __name__ == "__main__":
    model = make_model()
    path = "simplified_layer_norm_fp16_pow_edge.onnx"
    onnx.save(model, path)
    print(f"Saved {path}")
