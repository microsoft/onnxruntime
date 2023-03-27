import onnx
from onnx import TensorProto, helper


def GenerateModel(model_name):  # noqa: N802
    initializers = [
        helper.make_tensor(
            "weight",
            TensorProto.FLOAT,
            [1, 4],
            [1.0, 2.0, 3.0, 4.0],
        ),
    ]

    nodes = [
        helper.make_node(
            "custom op",
            ["input1", "weight"],
            ["output"],
            "custom op",
            domain="test.customop",
        ),
    ]

    graph = helper.make_graph(
        nodes,
        "custom op graph",
        [  # input
            helper.make_tensor_value_info("input1", TensorProto.FLOAT, [1, 4]),
        ],
        [  # output
            helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4]),
        ],
        initializers,
    )

    model = helper.make_model(graph)
    onnx.save(model, model_name)


if __name__ == "__main__":
    GenerateModel("test_kernel_info_get_const_input.onnx")
