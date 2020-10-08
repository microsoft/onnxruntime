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

scale_value = 3.0


def save(model_path, nodes, inputs, outputs, initializers):
    graph = helper.make_graph(
        nodes,
        "MatMulScaleTest",
        inputs, outputs, initializers)

    model = helper.make_model(
        graph, opset_imports=opsets, producer_name="onnxruntime-test")

    onnx.save(model, model_path)


def gen(model_path,
        use_transpose_matmul,
        scale_input_0, scale_input_1, scale_output):
    matmul_op = "TransposeScaleMatMul" if use_transpose_matmul else "MatMul"
    matmul_domain = "com.microsoft" if use_transpose_matmul else ""
    matmul_attrs = {"alpha": scale_value} if use_transpose_matmul else {}

    nodes = []

    if scale_input_0:
        nodes.append(helper.make_node(
            "Mul", ["input_0", "scale"], ["scaled_input_0"], "scale input_0"))

    if scale_input_1:
        nodes.append(helper.make_node(
            "Div", ["input_1", "scale_reciprocal"], ["scaled_input_1"], "scale input_1"))

    nodes.append(helper.make_node(
        matmul_op,
        [
            "scaled_input_0" if scale_input_0 else "input_0",
            "scaled_input_1" if scale_input_1 else "input_1"
        ],
        [
            "unscaled_output" if scale_output else "output"
        ],
        matmul_op,
        "",
        matmul_domain,
        **matmul_attrs))

    if scale_output:
        nodes.append(helper.make_node(
            "Mul", ["scale", "unscaled_output"], ["output"], "scale output"))

    initializers = [
        helper.make_tensor("scale", TensorProto.FLOAT, [], [scale_value]),
        helper.make_tensor("scale_reciprocal",
                           TensorProto.FLOAT, [], [1/scale_value])
    ]

    inputs = [
        helper.make_tensor_value_info(
            "input_0", TensorProto.FLOAT, [2, 'M', 'K']),
        helper.make_tensor_value_info(
            "input_1", TensorProto.FLOAT, [2, 'K', 'N'])
    ]

    outputs = [
        helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [2, 'M', 'N'])
    ]

    save(model_path, nodes, inputs, outputs, initializers)


gen("matmul_scale_in0.onnx", False, True, False, False)
gen("matmul_scale_in0_in1.onnx", False, True, True, False)
gen("matmul_scale_in0_in1_out.onnx", False, True, True, True)
gen("matmul_scale_transposescalematmul_in0_in1_out.onnx", True, True, True, True)


UNFUSABLE_DIV_NOT_SCALE = 0
UNFUSABLE_SCALE_NOT_SCALAR = 1
UNFUSABLE_SCALE_NOT_CONSTANT = 2


def gen_unfusable(model_path, unfusable_type):
    matmul_op = "MatMul"

    if unfusable_type == UNFUSABLE_DIV_NOT_SCALE:
        scale_node = helper.make_node(
            "Div", ["scale", "input_0"], ["scaled_input_0"], "scale input_0")
    elif unfusable_type == UNFUSABLE_SCALE_NOT_SCALAR:
        scale_node = helper.make_node(
            "Mul", ["scale_non_scalar", "input_0"], ["scaled_input_0"], "scale input_0")
    elif unfusable_type == UNFUSABLE_SCALE_NOT_CONSTANT:
        scale_node = helper.make_node(
            "Mul", ["input_0", "input_0"], ["scaled_input_0"], "scale input_0")
    else:
        raise ValueError("Invalid unfusable_type: {}".format(unfusable_type))

    nodes = [
        scale_node,
        helper.make_node(
            matmul_op, ["scaled_input_0", "input_1"], ["output"], matmul_op)
    ]

    initializers = [
        helper.make_tensor("scale", TensorProto.FLOAT, [], [scale_value]),
        helper.make_tensor("scale_non_scalar", TensorProto.FLOAT,
                           [2, 1, 1], [scale_value, scale_value])
    ]

    inputs = [
        helper.make_tensor_value_info(
            "input_0", TensorProto.FLOAT, [2, 'M', 'K']),
        helper.make_tensor_value_info(
            "input_1", TensorProto.FLOAT, [2, 'K', 'N'])
    ]

    outputs = [
        helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [2, 'M', 'N'])
    ]

    save(model_path, nodes, inputs, outputs, initializers)


gen_unfusable("matmul_scale_unfusable_div_not_scale.onnx",
              UNFUSABLE_DIV_NOT_SCALE)
gen_unfusable("matmul_scale_unfusable_scale_not_scalar.onnx",
              UNFUSABLE_SCALE_NOT_SCALAR)
gen_unfusable("matmul_scale_unfusable_scale_not_constant.onnx",
              UNFUSABLE_SCALE_NOT_CONSTANT)


def gen_reused_input_scale(model_path):
    matmul_op = "MatMul"

    nodes = [
        helper.make_node(
            "Mul", ["input_0", "scale"], ["scaled_input_0"],
            "scale input_0"),
        helper.make_node(
            matmul_op, ["scaled_input_0", "input_1"], ["output_0"],
            "MatMul input_0 and input_1"),
        helper.make_node(
            matmul_op, ["scaled_input_0", "input_2"], ["output_1"],
            "MatMul input_0 and input_2")
    ]

    initializers = [
        helper.make_tensor("scale", TensorProto.FLOAT, [], [scale_value])
    ]

    inputs = [
        helper.make_tensor_value_info(
            "input_0", TensorProto.FLOAT, [2, 'M', 'K']),
        helper.make_tensor_value_info(
            "input_1", TensorProto.FLOAT, [2, 'K', 'N']),
        helper.make_tensor_value_info(
            "input_2", TensorProto.FLOAT, [2, 'K', 'N'])
    ]

    outputs = [
        helper.make_tensor_value_info(
            "output_0", TensorProto.FLOAT, [2, 'M', 'N']),
        helper.make_tensor_value_info(
            "output_1", TensorProto.FLOAT, [2, 'M', 'N'])
    ]

    save(model_path, nodes, inputs, outputs, initializers)


gen_reused_input_scale("matmul_scale_reused_input_scale.onnx")
