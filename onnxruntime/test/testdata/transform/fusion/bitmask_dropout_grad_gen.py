import onnx
from onnx import helper
from onnx import TensorProto
from onnx import OperatorSetIdProto

onnxdomain = OperatorSetIdProto()
onnxdomain.version = 13
# The empty string ("") or absence of this field implies the operator set that is defined as part of the ONNX specification.
onnxdomain.domain = ""
msdomain = OperatorSetIdProto()
msdomain.version = 1
msdomain.domain = "com.microsoft"
opsets = [onnxdomain, msdomain]


def save(model_path, nodes, inputs, outputs, initializers):
    graph = helper.make_graph(
        nodes,
        "BitmaskDropoutGradTest",
        inputs, outputs, initializers)

    model = helper.make_model(
        graph, opset_imports=opsets, producer_name="onnxruntime-test")

    print(model_path)
    onnx.save(model, model_path)

def gen_bitmask_dropout_grad_basic(model_path):
    nodes = [
        helper.make_node(op_type="Dropout", inputs=["A"], outputs=["tp0", "tp1"]),
        helper.make_node(op_type="DropoutGrad", inputs=["tp0", "tp1"], outputs=["B"]),
    ]

    inputs = [
        helper.make_tensor_value_info("A", TensorProto.FLOAT, ['M', 'N']),
    ]

    outputs = [
        helper.make_tensor_value_info("B", TensorProto.FLOAT, ['M', 'N']),
    ]

    save(model_path, nodes, inputs, outputs, initializers=[])

def gen_bitmask_dropout_grad_multiple_mask_uses(model_path):
    nodes = [
        helper.make_node(op_type="Dropout", inputs=["A"], outputs=["tp0", "tp1"]),
        helper.make_node(op_type="DropoutGrad", inputs=["tp0", "tp1"], outputs=["B"]),
        helper.make_node(op_type="Identity", inputs=["tp1"], outputs=["C"]),
    ]

    inputs = [
        helper.make_tensor_value_info("A", TensorProto.FLOAT, ['M', 'N']),
    ]

    outputs = [
        helper.make_tensor_value_info("B", TensorProto.FLOAT, ['M', 'N']),
    ]

    save(model_path, nodes, inputs, outputs, initializers=[])

gen_bitmask_dropout_grad_basic("bitmask_dropout_grad_basic.onnx")
gen_bitmask_dropout_grad_multiple_mask_uses("bitmask_dropout_grad_multiple_mask_uses")