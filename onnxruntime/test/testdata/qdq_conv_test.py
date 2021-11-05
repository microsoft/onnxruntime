import onnx
from onnx import helper
from onnx import TensorProto


# Generate a basic QDQ Conv model
def GenerateModel(model_name):
    nodes = [
        helper.make_node("DequantizeLinear", ["X", "Scale", "Zero_point_uint8"], ["input_DQ"], "input_DQ"),
        helper.make_node("DequantizeLinear", ["W", "Scale", "Zero_point_uint8"], ["conv_weight_DQ"], "conv_weight_DQ"),
        helper.make_node("DequantizeLinear", ["Bias", "Scale", "Zero_point_int32"], ["conv_bias_DQ"], "conv_bias_DQ"),
        helper.make_node("Conv", ["input_DQ", "conv_weight_DQ", "conv_bias_DQ"], ["conv_output"], "conv"),
        helper.make_node("QuantizeLinear", ["conv_output", "Scale", "Zero_point_uint8"], ["Y"], "output_Q"),
    ]

    initializers = [
        helper.make_tensor('Scale', TensorProto.FLOAT, [1], [256.0]),
        helper.make_tensor('Zero_point_uint8', TensorProto.UINT8, [1], [0]),
        helper.make_tensor('Zero_point_int32', TensorProto.INT32, [1], [0]),
        helper.make_tensor('W', TensorProto.UINT8, [1, 1, 3, 3], [128] * 9),
        helper.make_tensor('Bias', TensorProto.INT32, [1], [64]),
    ]

    inputs = [
        helper.make_tensor_value_info('X', TensorProto.UINT8, [1, 1, 5, 5]),
    ]

    outputs = [
        helper.make_tensor_value_info('Y', TensorProto.UINT8, [1, 1, 3, 3]),
    ]

    graph = helper.make_graph(
        nodes,
        "QDQ_Conv_Model_Basic",
        inputs,
        outputs,
        initializers
    )

    model = helper.make_model(graph)
    onnx.save(model, model_name)


if __name__ == "__main__":
    GenerateModel('qdq_conv_model_basic.onnx')
