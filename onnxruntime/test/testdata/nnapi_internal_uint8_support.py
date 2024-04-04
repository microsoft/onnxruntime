import onnx
from onnx import TensorProto, helper


# This is to test the operators without "Qlinear" support but still support uint8 input
# These operators need to be internal to a graph/partition
# def GenerateModel(model_name):
def GenerateModel(model_name):  # noqa: N802
    nodes = [
        helper.make_node(
            "QuantizeLinear",
            ["X", "Scale", "Zero_point"],
            ["X_quantized"],
            "quantize_0",
        ),
        helper.make_node(
            "Concat",
            ["X_quantized", "X_quantized"],
            ["X_concat"],
            axis=-2,
            name="concat_0",
        ),
        helper.make_node(
            "MaxPool",
            ["X_concat"],
            ["X_maxpool"],
            kernel_shape=[2, 2],
            name="maxpool_0",
        ),
        helper.make_node(
            "Transpose",
            ["X_maxpool"],
            ["X_transposed"],
            perm=[0, 1, 3, 2],
            name="transpose_0",
        ),
        helper.make_node(
            "DequantizeLinear",
            ["X_transposed", "Scale", "Zero_point"],
            ["Y"],
            "dequantize_0",
        ),
    ]

    initializers = [
        helper.make_tensor("Scale", TensorProto.FLOAT, [1], [256.0]),
        helper.make_tensor("Zero_point", TensorProto.UINT8, [1], [0]),
    ]

    inputs = [
        helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 1, 3]),
    ]

    graph = helper.make_graph(
        nodes,
        "NNAPI_Internal_uint8_Test",
        inputs,
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 2, 1])],
        initializers,
    )

    model = helper.make_model(graph)
    onnx.save(model, model_name)


if __name__ == "__main__":
    GenerateModel("nnapi_internal_uint8_support.onnx")
