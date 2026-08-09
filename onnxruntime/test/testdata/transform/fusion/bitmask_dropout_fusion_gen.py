import onnx
from onnx import OperatorSetIdProto, TensorProto, helper

onnxdomain = OperatorSetIdProto()
onnxdomain.version = 13
onnxdomain.domain = ""
msdomain = OperatorSetIdProto()
msdomain.version = 1
msdomain.domain = "com.microsoft"
opsets = [onnxdomain, msdomain]


def save(model_path, nodes, inputs, outputs, initializers):
    graph = helper.make_graph(nodes, "BitmaskDropoutGradTest", inputs, outputs, initializers)
    model = helper.make_model(graph, opset_imports=opsets, producer_name="onnxruntime-test")
    onnx.save(model, model_path)


input_a = helper.make_tensor_value_info("A", TensorProto.FLOAT, ["M", "N"])
input_b = helper.make_tensor_value_info("B", TensorProto.FLOAT, ["N"])
input_c = helper.make_tensor_value_info("C", TensorProto.FLOAT, ["M", "N"])
input_dy = helper.make_tensor_value_info("dY", TensorProto.FLOAT, ["M", "N"])
output_dx = helper.make_tensor_value_info("dX", TensorProto.FLOAT, ["M", "N"])
output_y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["M", "N"])
output_mask = helper.make_tensor_value_info("MASK", TensorProto.BOOL, ["M", "N"])
ratio = helper.make_tensor("ratio", TensorProto.FLOAT, [], [0.1])
training_mode = helper.make_tensor("training_mode", TensorProto.BOOL, [], [1])


def gen_bitmask_dropout_replacement_basic(model_path):
    nodes = [
        helper.make_node(op_type="Dropout", inputs=["A", "ratio", "training_mode"], outputs=["Y", "tp1"]),
        helper.make_node(
            op_type="DropoutGrad",
            inputs=["dY", "tp1", "ratio", "training_mode"],
            outputs=["dX"],
            domain="com.microsoft",
        ),
    ]
    save(model_path, nodes, [input_a, input_dy], [output_y, output_dx], [ratio, training_mode])


def gen_bitmask_dropout_replacement_multiple_mask_uses(model_path):
    nodes = [
        helper.make_node(op_type="Dropout", inputs=["A", "ratio", "training_mode"], outputs=["Y", "tp1"]),
        helper.make_node(
            op_type="DropoutGrad",
            inputs=["dY", "tp1", "ratio", "training_mode"],
            outputs=["dX"],
            domain="com.microsoft",
        ),
        helper.make_node(op_type="Identity", inputs=["tp1"], outputs=["MASK"]),
    ]
    save(model_path, nodes, [input_a, input_dy], [output_y, output_dx, output_mask], [ratio, training_mode])


def gen_bitmask_bias_dropout_replacement_basic(model_path):
    nodes = [
        helper.make_node(
            op_type="BiasDropout",
            inputs=["A", "B", "C", "ratio", "training_mode"],
            outputs=["Y", "tp1"],
            domain="com.microsoft",
        ),
        helper.make_node(
            op_type="DropoutGrad",
            inputs=["dY", "tp1", "ratio", "training_mode"],
            outputs=["dX"],
            domain="com.microsoft",
        ),
    ]
    save(model_path, nodes, [input_a, input_b, input_c, input_dy], [output_y, output_dx], [ratio, training_mode])


def gen_bitmask_bias_dropout_fusion_basic(model_path):
    nodes = [
        helper.make_node(op_type="Add", inputs=["A", "B"], outputs=["add"]),
        helper.make_node(
            op_type="BitmaskDropout",
            inputs=["add", "ratio", "training_mode"],
            outputs=["Y", "tp1"],
            domain="com.microsoft",
        ),
        helper.make_node(
            op_type="BitmaskDropoutGrad",
            inputs=["dY", "tp1", "ratio", "training_mode"],
            outputs=["dX"],
            domain="com.microsoft",
        ),
    ]
    save(model_path, nodes, [input_a, input_b, input_dy], [output_y, output_dx], [ratio, training_mode])


def gen_bitmask_bias_dropout_fusion_residual(model_path):
    nodes = [
        helper.make_node(op_type="Add", inputs=["A", "B"], outputs=["add"]),
        helper.make_node(
            op_type="BitmaskDropout",
            inputs=["add", "ratio", "training_mode"],
            outputs=["tp0", "tp1"],
            domain="com.microsoft",
        ),
        helper.make_node(
            op_type="BitmaskDropoutGrad",
            inputs=["dY", "tp1", "ratio", "training_mode"],
            outputs=["dX"],
            domain="com.microsoft",
        ),
        helper.make_node(op_type="Add", inputs=["tp0", "C"], outputs=["Y"]),
    ]
    save(model_path, nodes, [input_a, input_b, input_c, input_dy], [output_y, output_dx], [ratio, training_mode])


gen_bitmask_dropout_replacement_basic("bitmask_dropout_replacement_basic.onnx")
gen_bitmask_dropout_replacement_multiple_mask_uses("bitmask_dropout_replacement_multiple_mask_uses.onnx")
gen_bitmask_bias_dropout_replacement_basic("bitmask_bias_dropout_replacement_basic.onnx")
gen_bitmask_bias_dropout_fusion_basic("bitmask_bias_dropout_fusion_basic.onnx")
gen_bitmask_bias_dropout_fusion_residual("bitmask_bias_dropout_fusion_residual.onnx")
