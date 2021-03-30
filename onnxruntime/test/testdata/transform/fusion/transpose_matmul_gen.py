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
            "FusedMatMul",
            ["transposed_input_0", "input_1"],
            ["output"],
            "FusedMatMul",
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


def gen_transpose_fusion_with_cast(model_path):
    cast_1 = helper.make_node(
        "Cast",
        ["input_1"],
        ["casted_input_1"],
        "Cast_1",
        to = TensorProto.FLOAT16)
    transpose_0 = helper.make_node(
        "Transpose",
        ["input_0"],
        ["transposed_input_0"],
        "Transpose_0",
        perm = [0, 1, 3, 2])
    cast_0 = helper.make_node(
        "Cast",
        ["transposed_input_0"],
        ["transposed_casted_input_0"],
        "Cast_0",
        to = TensorProto.FLOAT16)
    matmul_0 = helper.make_node(
        "MatMul",
        ["transposed_casted_input_0", "casted_input_1"],
        ["output_0"],
        "MatMul_0")

    nodes = [transpose_0, cast_0, cast_1, matmul_0]

    input_0 = helper.make_tensor_value_info("input_0", TensorProto.FLOAT, [3, 2, 'N', 'N'])
    input_1 = helper.make_tensor_value_info("input_1", TensorProto.FLOAT, [3, 2, 'N', 'N'])
    inputs = [input_0, input_1]
    output_0 = helper.make_tensor_value_info("output_0", TensorProto.FLOAT16, [3, 2, 'N', 'N'])
    outputs = [output_0]
    # Testcase0: First input of MatMul is transposed
    save(model_path + "0.onnx", nodes, inputs, outputs, [])

    # Testcase1: Re-arragne nodes so that the transpose is on second input of matmul
    transpose_1 = helper.make_node(
               "Transpose",
               ["input_1"],
               ["transposed_input_1"],
               "Transpose_1",
               perm = [0, 1, 3, 2])
    cast_1.input[0] = "transposed_input_1"
    cast_1.output[0] = "transposed_casted_input_1"
    cast_0.input[0] = "input_0"
    cast_0.output[0] = "casted_input_0"
    matmul_0.input[0] = cast_0.output[0]
    matmul_0.input[1] = cast_1.output[0]
    nodes = [cast_0, transpose_1, cast_1, matmul_0]
    save(model_path + "1.onnx", nodes, inputs, outputs, [])

    # Testcase2: Create an example with two Cast-ed Transpose-ed inputs feeding a MatMul
    cast_0.input[0] = "transposed_input_0"
    cast_0.output[0] = "transposed_casted_input_0"
    matmul_0.input[0] = cast_0.output[0]
    nodes = [transpose_0, cast_0, transpose_1, cast_1, matmul_0]
    save(model_path + "2.onnx", nodes, inputs, outputs, [])

    # Testcase3: Create a second MatMul node using the outputs from the same Cast nodes as before
    # with each Cast node feeding more than one node.
    nodes.append(helper.make_node(
            "MatMul",
            ["transposed_casted_input_0", "transposed_casted_input_1"],
            ["output_1"],
            "MatMul_1"))
    output_1 = helper.make_tensor_value_info("output_1", TensorProto.FLOAT16, [3, 2, 'N', 'N'])
    outputs.append(output_1)
    save(model_path + "3.onnx", nodes, inputs, outputs, [])

    # Testcase4: The second MatMul uses transposed inputs without cast.
    nodes.pop()
    outputs.pop()
    matmul_1 = helper.make_node(
            "MatMul",
            ["transposed_input_0", "transposed_input_1"],
            ["output_1"],
            "MatMul_1")
    nodes.append(matmul_1)

    outputs.append(helper.make_tensor_value_info(
            "output_1", TensorProto.FLOAT, [3, 2, 'N', 'N']))
    save(model_path + "4.onnx", nodes, inputs, outputs, [])

    # Testcase5: Each MatMul uses outputs from a Cast and a Transpose
    input_0.type.tensor_type.elem_type = TensorProto.FLOAT16
    cast_0.attribute[0].i = TensorProto.FLOAT
    matmul_0.input[0] = "transposed_input_0"
    matmul_1.input[0] = "transposed_casted_input_0"
    output_1.type.tensor_type.elem_type = TensorProto.FLOAT
    save(model_path + "5.onnx", nodes, inputs, outputs, [])

gen_transpose_fusion_with_cast(
    "transpose_cast_matmul_4d_fusion")

def gen_transpose_fusion_invalid_datatype(model_path, datatype):
    nodes = [
        helper.make_node(
            "Transpose",
            ["input_0"],
            ["transposed_input_0"],
            perm = [0, 1, 3, 2]),
        helper.make_node(
            "MatMul",
            ["transposed_input_0", "input_1"],
            ["output"])
    ]

    inputs = [
        helper.make_tensor_value_info(
            "input_0", datatype, [2, 3, 'K', 'M']),
        helper.make_tensor_value_info(
            "input_1", datatype, [2, 3, 'K', 'N'])
    ]

    outputs = [
        helper.make_tensor_value_info(
            "output", datatype, [2, 3, 'M', 'N'])
    ]

    save(model_path, nodes, inputs, outputs, [])


gen_transpose_fusion_invalid_datatype("transpose_matmul_4d_fusion_invalid_datatype_int32.onnx", TensorProto.INT32)
gen_transpose_fusion_invalid_datatype("transpose_matmul_4d_fusion_invalid_datatype_int64.onnx", TensorProto.INT64)
