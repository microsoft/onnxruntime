import onnx
from onnx import TensorProto, helper


# Since NNAPI EP handles Reshape and Flatten differently,
# Please see ReshapeOpBuilder::CanSkipReshape in <repo_root>/onnxruntime/core/providers/nnapi/nnapi_builtin/builders/op_builder.cc
# We have a separated test for these skip reshape scenarios
def GenerateModel(model_name):
    nodes = [
        helper.make_node("Flatten", ["X"], ["Flatten_1_Y"], "flatten_1"),
        helper.make_node("MatMul", ["Flatten_1_Y", "MatMul_B"], ["MatMul_Y"], "matmul"),
        helper.make_node("Reshape", ["Y", "Reshape_1_shape"], ["Reshape_1_Y"], "reshape_1"),
        helper.make_node("Gemm", ["Reshape_1_Y", "Gemm_B"], ["Gemm_Y"], "gemm"),
        helper.make_node("Reshape", ["MatMul_Y", "Reshape_2_shape"], ["Reshape_2_Y"], "reshape_2"),
        helper.make_node("Flatten", ["Gemm_Y"], ["Flatten_2_Y"], "flatten_2", axis=0),
        helper.make_node("Add", ["Reshape_2_Y", "Flatten_2_Y"], ["Z"], "add"),
    ]

    initializers = [
        helper.make_tensor("Reshape_1_shape", TensorProto.INT64, [2], [3, 4]),
        helper.make_tensor("Reshape_2_shape", TensorProto.INT64, [2], [1, 6]),
        helper.make_tensor(
            "Gemm_B",
            TensorProto.FLOAT,
            [4, 2],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        ),
        helper.make_tensor("MatMul_B", TensorProto.FLOAT, [2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    ]

    inputs = [
        helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 1, 2]),
        helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 2, 2]),
    ]

    graph = helper.make_graph(
        nodes,
        "NNAPI_Reshape_Flatten_Test",
        inputs,
        [helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, 6])],
        initializers,
    )

    model = helper.make_model(graph)
    onnx.save(model, model_name)


if __name__ == "__main__":
    GenerateModel("nnapi_reshape_flatten_test.onnx")
