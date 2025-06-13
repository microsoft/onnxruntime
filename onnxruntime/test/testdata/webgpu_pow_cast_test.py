import onnx
from onnx import TensorProto, helper

# tests fix for precision error in florence2 model by using WebGPU built-in sqrt(x) function instead of pow(x, y) when the exponent is 0.5.
# The sqrt(x) built-in is both faster and more stable than using pow(x, 0.5).
# Example:
# Cast: input = 576 (int), output = 576 (f32)
# Pow: input = 576(f32), 0.5(f32), output = 23.99999(f32)
# Cast: input = 23.9999(f32), output = 23(int)


graph_proto = helper.make_graph(
    [
        helper.make_node(
            "Pow",
            inputs=["x", "y"],
            outputs=["pow_output"],
            name="/Pow",
        ),
        helper.make_node(
            "Cast",
            inputs=["pow_output"],
            outputs=["out"],
            name="/Cast",
            to=TensorProto.INT64,
        ),
    ],
    "Main_graph",
    [
        helper.make_tensor_value_info("x", TensorProto.FLOAT, [1]),
        helper.make_tensor_value_info("y", TensorProto.FLOAT, [1]),
    ],
    [
        helper.make_tensor_value_info("out", TensorProto.INT64, [1]),
    ],
)

model = helper.make_model(graph_proto)
onnx.checker.check_model(model, True)
onnx.save(model, "webgpu_pow_cast_test.onnx")
