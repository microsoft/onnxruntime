import onnx
from onnx import helper
from onnx import TensorProto
from onnx import OperatorSetIdProto

onnxdomain = OperatorSetIdProto()
onnxdomain.version = 12
# The empty string ("") or absence of this field implies the operator set that is defined as part of the ONNX specification.
onnxdomain.domain = ""
msdomain = OperatorSetIdProto()
msdomain.version = 1
msdomain.domain = "com.microsoft"
opsets = [onnxdomain, msdomain]


def save(model_path, nodes, inputs, outputs, initializers):
    graph = helper.make_graph(
        nodes,
        "TransposeMatMulTest",
        inputs, outputs, initializers)

    model = helper.make_model(
        graph, opset_imports=opsets, producer_name="onnxruntime-test")

    onnx.save(model, model_path)


def gen_from_transpose_scale_matmul(model_path):
    nodes = [
        helper.make_node(
            "Transpose",
            ["input_0"],
            ["transposed_input_0"]),
        helper.make_node(
            "TransposeScaleMatMul",
            ["transposed_input_0", "input_1"],
            ["output"],
            "TransposeScaleMatMul",
            "",
            msdomain.domain,
            alpha=3.0, transA=1)
    ]

    inputs = [
        helper.make_tensor_value_info(
            "input_0", TensorProto.FLOAT, ['M', 'K']),
        helper.make_tensor_value_info(
            "input_1", TensorProto.FLOAT, ['K', 'N'])
    ]

    outputs = [
        helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, ['M', 'N'])
    ]

    save(model_path, nodes, inputs, outputs, [])


gen_from_transpose_scale_matmul(
    "transpose_matmul_2d_fusion_from_transpose_scale_matmul.onnx")


def gen_invalid_default_perm(model_path):
    nodes = [
        helper.make_node(
            "Transpose",
            ["input_0"],
            ["transposed_input_0"]),
        helper.make_node(
            "MatMul",
            ["transposed_input_0", "input_1"],
            ["output"])
    ]

    inputs = [
        helper.make_tensor_value_info(
            "input_0", TensorProto.FLOAT, ['K', 'M', 3, 2]),
        helper.make_tensor_value_info(
            "input_1", TensorProto.FLOAT, [2, 3, 'K', 'N'])
    ]

    outputs = [
        helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [2, 3, 'M', 'N'])
    ]

    save(model_path, nodes, inputs, outputs, [])


gen_invalid_default_perm(
    "transpose_matmul_4d_fusion_invalid_default_perm.onnx")


def gen_with_preserved_transpose(model_path):
    nodes = [
        helper.make_node(
            "Transpose",
            ["input_0"],
            ["transposed_input_0"]),
        helper.make_node(
            "MatMul",
            ["transposed_input_0", "input_1"],
            ["output_0"]),
        helper.make_node(
            "Identity",
            ["transposed_input_0"],
            ["output_1"])
    ]

    inputs = [
        helper.make_tensor_value_info(
            "input_0", TensorProto.FLOAT, ['K', 'M']),
        helper.make_tensor_value_info(
            "input_1", TensorProto.FLOAT, ['K', 'N'])
    ]

    outputs = [
        helper.make_tensor_value_info(
            "output_0", TensorProto.FLOAT, ['M', 'N']),
        helper.make_tensor_value_info(
            "output_1", TensorProto.FLOAT, ['M', 'K'])
    ]

    save(model_path, nodes, inputs, outputs, [])


gen_with_preserved_transpose(
    "transpose_matmul_2d_fusion_with_preserved_transpose.onnx")
