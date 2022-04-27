import onnx
from onnx import OperatorSetIdProto, TensorProto, helper

onnxdomain = OperatorSetIdProto()
onnxdomain.version = 12
# The empty string ("") or absence of this field implies the operator set that is defined as part of the ONNX specification.
onnxdomain.domain = ""
msdomain = OperatorSetIdProto()
msdomain.version = 1
msdomain.domain = "com.microsoft"
opsets = [onnxdomain, msdomain]


def save(model_path, nodes, inputs, outputs, initializers):
    graph = helper.make_graph(nodes, "TransposeGemmTest", inputs, outputs, initializers)

    model = helper.make_model(graph, opset_imports=opsets, producer_name="onnxruntime-test")

    onnx.save(model, model_path)


# (A')'B' = AB'
def gemm_transpose_2outputs_from_transpose(model_path):
    nodes = [
        helper.make_node("Transpose", ["A"], ["tp0"], "TransposeA"),
        helper.make_node("Transpose", ["B"], ["tp1"], "TransposeB"),
        helper.make_node("Gemm", ["tp0", "tp1"], ["output"], "Gemm", alpha=3.0, transA=1),
        helper.make_node("Identity", ["tp0"], ["output2"], "IdentityAt"),
    ]

    inputs = [
        helper.make_tensor_value_info("A", TensorProto.FLOAT, ["M", "K"]),
        helper.make_tensor_value_info("B", TensorProto.FLOAT, ["N", "K"]),
    ]

    outputs = [
        helper.make_tensor_value_info("output", TensorProto.FLOAT, ["M", "N"]),
        helper.make_tensor_value_info("output2", TensorProto.FLOAT, ["K", "M"]),
    ]

    save(model_path, nodes, inputs, outputs, [])


# (A')'B' = AB'  and  (B')'C = BC
def gemm_transpose_2outputs_from_transpose_to_2gemms(model_path):
    nodes = [
        helper.make_node("Transpose", ["A"], ["tp0"], "TransposeA"),
        helper.make_node("Transpose", ["B"], ["tp1"], "TransposeB"),
        helper.make_node("Gemm", ["tp0", "tp1"], ["output"], "Gemm1", alpha=3.0, transA=1),
        helper.make_node("Gemm", ["tp1", "C"], ["output3"], "Gemm2", alpha=3.0, transA=1),
        helper.make_node("Identity", ["tp0"], ["output2"], "IdentityAt"),
    ]

    inputs = [
        helper.make_tensor_value_info("A", TensorProto.FLOAT, ["M", "K"]),
        helper.make_tensor_value_info("B", TensorProto.FLOAT, ["N", "K"]),
        helper.make_tensor_value_info("C", TensorProto.FLOAT, ["K", "L"]),
    ]

    outputs = [
        helper.make_tensor_value_info("output", TensorProto.FLOAT, ["M", "N"]),
        helper.make_tensor_value_info("output2", TensorProto.FLOAT, ["K", "M"]),
        helper.make_tensor_value_info("output3", TensorProto.FLOAT, ["N", "L"]),
    ]

    save(model_path, nodes, inputs, outputs, [])


gemm_transpose_2outputs_from_transpose("gemm_transpose_2outputs_from_transpose.onnx")
gemm_transpose_2outputs_from_transpose_to_2gemms("gemm_transpose_2outputs_from_transpose_to_2gemms.onnx")
