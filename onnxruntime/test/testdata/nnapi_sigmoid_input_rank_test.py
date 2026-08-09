import onnx
from onnx import TensorProto, helper

# Since NNAPI ANEURALNETWORKS_LOGISTIC only supports input tensor rank up to 4, we added limits in
# Sigmoid op support checker in NNAPI EP, so we don't fail hard. Added test case here.


def GenerateModel(model_name):  # noqa: N802
    node = [
        helper.make_node("Sigmoid", ["X"], ["Y"], "sigmoid"),
    ]

    graph = helper.make_graph(
        node,
        "Nnapi_sigmoid_input_rank_test",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 1, 2, 1, 2])],  # input
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 1, 2, 1, 2])],  # output
    )

    model = helper.make_model(graph)
    onnx.save(model, model_name)


if __name__ == "__main__":
    GenerateModel("nnapi_sigmoid_input_rank_test.onnx")
