import onnx
from onnx import helper
from onnx import TensorProto


# Generate a basic QDQ Conv model
def GenerateModel(model_name):
    nodes = []
    initializers = []
    inputs = []
    outputs = []

    for i in range(3):
        def name(base):
            return f"{base}_{i}"

        nodes.extend([
            helper.make_node("DequantizeLinear", [name("X"), name("Scale"), name("Zero_point_uint8")], [name("input_DQ")], name("input_DQ")),
            helper.make_node("DequantizeLinear", [name("W"), name("Scale"), name("Zero_point_uint8")], [name("conv_weight_DQ")], name("conv_weight_DQ")),
            helper.make_node("DequantizeLinear", [name("Bias"), name("Scale"), name("Zero_point_int32")], [name("conv_bias_DQ")], name("conv_bias_DQ")),
            helper.make_node("Conv", [name("input_DQ"), name("conv_weight_DQ"), name("conv_bias_DQ")], [name("conv_output")], name("conv")),
            helper.make_node("QuantizeLinear", [name("conv_output"), name("Scale"), name("Zero_point_uint8")], [name("Y")], name("output_Q")),
        ])

        initializers.extend([
            helper.make_tensor(name('Scale'), TensorProto.FLOAT, [1], [256.0]),
            helper.make_tensor(name('Zero_point_uint8'), TensorProto.UINT8, [1], [0]),
            helper.make_tensor(name('Zero_point_int32'), TensorProto.INT32, [1], [0]),
            helper.make_tensor(name('W'), TensorProto.UINT8, [1, 1, 3, 3], [128] * 9),
            helper.make_tensor(name('Bias'), TensorProto.INT32, [1], [64]),
        ])

        inputs.extend([
            helper.make_tensor_value_info(name('X'), TensorProto.UINT8, [1, 1, 5, 5]),
        ])

        outputs.extend([
            helper.make_tensor_value_info(name('Y'), TensorProto.UINT8, [1, 1, 3, 3]),
        ])

    graph = helper.make_graph(
        nodes,
        "QDQ_Convs",
        inputs,
        outputs,
        initializers
    )

    model = helper.make_model(graph)
    onnx.save(model, model_name)


if __name__ == "__main__":
    GenerateModel('qdq_convs.onnx')
