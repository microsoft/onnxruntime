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
        "CastPropagateTest",
        inputs, outputs, initializers)

    model = helper.make_model(
        graph, opset_imports=opsets, producer_name="onnxruntime-test")

    onnx.save(model, model_path)

def gen_propagate_cast_float16(model_path):
    nodes = [
        helper.make_node(
            "MatMul",
            ["input_0", "input_1"],
            ["product"],
            "MatMul_0"),
        helper.make_node(
            "Cast",
            ["product"],
            ["output"],
            "Cast_0",
            to = TensorProto.FLOAT16)
    ]

    inputs = [
        helper.make_tensor_value_info(
            "input_0", TensorProto.FLOAT, ['M', 'K']),
        helper.make_tensor_value_info(
            "input_1", TensorProto.FLOAT, ['K', 'N'])
    ]

    outputs = [
        helper.make_tensor_value_info(
            "output", TensorProto.FLOAT16, ['M', 'N'])
    ]

    save(model_path, nodes, inputs, outputs, [])

def gen_propagate_cast_float(model_path):
    nodes = [
        helper.make_node(
            "Cast",
            ["input_0"],
            ["cast_input_0"],
            "Cast_0",
            to = TensorProto.FLOAT),
        helper.make_node(
            "Cast",
            ["input_1"],
            ["cast_input_1"],
            "Cast_1",
            to = TensorProto.FLOAT),
        helper.make_node(
            "MatMul",
            ["cast_input_0", "cast_input_1"],
            ["product"],
            "MatMul_0"),
        helper.make_node(
            "Cast",
            ["product"],
            ["output"],
            "Cast_2",
            to = TensorProto.FLOAT16),
     ]

    inputs = [
        helper.make_tensor_value_info(
            "input_0", TensorProto.FLOAT16, ['M', 'K']),
        helper.make_tensor_value_info(
            "input_1", TensorProto.FLOAT16, ['K', 'N'])
    ]

    outputs = [
        helper.make_tensor_value_info(
            "output", TensorProto.FLOAT16, ['M', 'N'])
    ]

    save(model_path, nodes, inputs, outputs, [])

gen_propagate_cast_float16("propagate_cast_float16.onnx")
gen_propagate_cast_float("propagate_cast_float.onnx")
