import onnx
from onnx import helper
from onnx import TensorProto


# This is to test the operators without "Qlinear" support but still support uint8 input
# These operators need to be internal to a graph/partition
# def GenerateModel(model_name):
def GenerateModel(model_name):
    nodes = [
        helper.make_node("QuantizeLinear", ["X", "Scale", "Zero_point"], ["X_quantized"], "quantize"),
        helper.make_node("Concat", ["X_quantized", "X_quantized"], ["X_concat"], axis=0, name="concat"),
        helper.make_node("Transpose", ["X_concat"], ["X_transposed"], "transpose"),
        helper.make_node("DequantizeLinear", ["X_transposed", "Scale", "Zero_point"], ["Y"], "dequantize"),
    ]

    initializers = [
        helper.make_tensor('Scale', TensorProto.FLOAT, [1], [256.0]),
        helper.make_tensor('Zero_point', TensorProto.UINT8, [1], [0]),
    ]

    inputs = [
        helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 3]),
    ]

    graph = helper.make_graph(
        nodes,
        "NNAPI_Internal_uint8_Test",
        inputs,
        [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [3, 2])],
        initializers
    )

    model = helper.make_model(graph)
    onnx.save(model, model_name)


if __name__ == "__main__":
    GenerateModel('nnapi_internal_uint8_support.onnx')
