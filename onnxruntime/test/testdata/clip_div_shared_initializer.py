from onnx import TensorProto, checker, helper, save

graph_proto = helper.make_graph(
    [
        helper.make_node(
            "Clip",
            inputs=["input_0", "initializer_0", "initializer_1"],
            outputs=["clip_output"],
            name="clip",
        ),
        helper.make_node(
            "Div",
            inputs=["clip_output", "initializer_1"],
            outputs=["output_0"],
            name="div",
        ),
    ],
    "Main_graph",
    [
        helper.make_tensor_value_info("input_0", TensorProto.FLOAT, [3, 2]),
    ],
    [
        helper.make_tensor_value_info("output_0", TensorProto.FLOAT, [3, 2]),
    ],
    [
        helper.make_tensor("initializer_0", TensorProto.FLOAT, [], [0.0]),
        helper.make_tensor("initializer_1", TensorProto.FLOAT, [], [6.0]),
    ],
)

model = helper.make_model(graph_proto)
checker.check_model(model, True)
save(model, "clip_div_shared_initializer.onnx")
