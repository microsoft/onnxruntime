import onnx
from onnx import helper, TensorProto, OperatorSetIdProto

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
input_d = helper.make_tensor_value_info("D", TensorProto.FLOAT, ["M", "N"])
output_g = helper.make_tensor_value_info("G", TensorProto.FLOAT, ["M", "N"])
output_y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["M", "N"])
output_z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, ["M", "N"])
output_mask = helper.make_tensor_value_info("MASK", TensorProto.BOOL, ["M", "N"])
ratio = helper.make_tensor("ratio", TensorProto.FLOAT, [], [0.1])
training_mode = helper.make_tensor("training_mode", TensorProto.BOOL, [], [1])


def gen_bitmask_dropout_grad_basic(model_path):
    nodes = [
        helper.make_node(op_type="Dropout", inputs=["A", "ratio", "training_mode"], outputs=["tp0", "tp1"]),
        helper.make_node(
            op_type="DropoutGrad", inputs=["C", "tp1", "ratio", "training_mode"], outputs=["G"], domain="com.microsoft"
        ),
    ]
    save(model_path, nodes, [input_a, input_c], [output_g], [ratio, training_mode])


def gen_bitmask_dropout_grad_multiple_mask_uses(model_path):
    nodes = [
        helper.make_node(op_type="Dropout", inputs=["A", "ratio", "training_mode"], outputs=["tp0", "tp1"]),
        helper.make_node(
            op_type="DropoutGrad", inputs=["C", "tp1", "ratio", "training_mode"], outputs=["G"], domain="com.microsoft"
        ),
        helper.make_node(op_type="Identity", inputs=["tp1"], outputs=["MASK"]),
    ]
    save(model_path, nodes, [input_a, input_c], [output_g, output_mask], [ratio, training_mode])


def gen_bitmask_bias_dropout_grad_basic(model_path):
    nodes = [
        helper.make_node(op_type="Add", inputs=["A", "B"], outputs=["add"]),
        helper.make_node(op_type="Dropout", inputs=["add", "ratio", "training_mode"], outputs=["tp0", "tp1"]),
        helper.make_node(
            op_type="DropoutGrad", inputs=["C", "tp1", "ratio", "training_mode"], outputs=["G"], domain="com.microsoft"
        ),
    ]
    save(model_path, nodes, [input_a, input_b, input_c], [output_g], [ratio, training_mode])


def gen_bitmask_bias_dropout_grad_basic_2(model_path):
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
            inputs=["C", "tp1", "ratio", "training_mode"],
            outputs=["G"],
            domain="com.microsoft",
        ),
    ]
    save(model_path, nodes, [input_a, input_b, input_c], [output_g], [ratio, training_mode])


def gen_bitmask_bias_dropout_grad_multiple_mask_uses(model_path):
    nodes = [
        helper.make_node(op_type="Add", inputs=["A", "B"], outputs=["add"]),
        helper.make_node(op_type="Dropout", inputs=["add", "ratio", "training_mode"], outputs=["tp0", "tp1"]),
        helper.make_node(
            op_type="DropoutGrad", inputs=["C", "tp1", "ratio", "training_mode"], outputs=["G"], domain="com.microsoft"
        ),
        helper.make_node(op_type="Identity", inputs=["tp1"], outputs=["MASK"]),
    ]
    save(model_path, nodes, [input_a, input_b, input_c], [output_g, output_mask], [ratio, training_mode])


def gen_bitmask_bias_dropout_grad_residual(model_path):
    nodes = [
        helper.make_node(op_type="Add", inputs=["A", "B"], outputs=["add"]),
        helper.make_node(op_type="Dropout", inputs=["add", "ratio", "training_mode"], outputs=["tp0", "tp1"]),
        helper.make_node(
            op_type="DropoutGrad", inputs=["C", "tp1", "ratio", "training_mode"], outputs=["G"], domain="com.microsoft"
        ),
        helper.make_node(op_type="Add", inputs=["tp0", "D"], outputs=["Y"]),
    ]
    save(model_path, nodes, [input_a, input_b, input_c, input_d], [output_g, output_y], [ratio, training_mode])


def gen_bitmask_bias_dropout_grad_residual_2(model_path):
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
            inputs=["C", "tp1", "ratio", "training_mode"],
            outputs=["G"],
            domain="com.microsoft",
        ),
        helper.make_node(op_type="Add", inputs=["tp0", "D"], outputs=["Y"]),
    ]
    save(model_path, nodes, [input_a, input_b, input_c, input_d], [output_g, output_y], [ratio, training_mode])


def gen_bitmask_bias_dropout_grad_residual_multiple_consumers(model_path):
    nodes = [
        helper.make_node(op_type="Add", inputs=["A", "B"], outputs=["add"]),
        helper.make_node(op_type="Dropout", inputs=["add", "ratio", "training_mode"], outputs=["tp0", "tp1"]),
        helper.make_node(
            op_type="DropoutGrad", inputs=["C", "tp1", "ratio", "training_mode"], outputs=["G"], domain="com.microsoft"
        ),
        helper.make_node(op_type="Add", inputs=["tp0", "D"], outputs=["Y"]),
        helper.make_node(op_type="Identity", inputs=["tp0"], outputs=["Z"]),
    ]
    save(
        model_path, nodes, [input_a, input_b, input_c, input_d], [output_g, output_y, output_z], [ratio, training_mode]
    )


gen_bitmask_dropout_grad_basic("bitmask_dropout_grad_basic.onnx")
gen_bitmask_dropout_grad_multiple_mask_uses("bitmask_dropout_grad_multiple_mask_uses.onnx")
gen_bitmask_bias_dropout_grad_basic("bitmask_bias_dropout_grad_basic.onnx")
gen_bitmask_bias_dropout_grad_basic_2("bitmask_bias_dropout_grad_basic_2.onnx")
gen_bitmask_bias_dropout_grad_multiple_mask_uses("bitmask_bias_dropout_grad_multiple_mask_uses.onnx")
gen_bitmask_bias_dropout_grad_residual("bitmask_bias_dropout_grad_residual.onnx")
gen_bitmask_bias_dropout_grad_residual_2("bitmask_bias_dropout_grad_residual_2.onnx")
gen_bitmask_bias_dropout_grad_residual_multiple_consumers("bitmask_bias_dropout_grad_residual_multiple_consumers.onnx")
