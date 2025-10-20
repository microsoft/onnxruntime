from onnx import TensorProto, checker, helper, save

# (A + B) * B + A
graph_proto = helper.make_graph(
    nodes=[
        helper.make_node(
            "Add",
            inputs=["A", "B"],
            outputs=["add_output"],
            name="add_0",
        ),
        helper.make_node(
            "Mul",
            inputs=["add_output", "B"],
            outputs=["mul_output"],
            name="mul_0",
        ),
        helper.make_node(
            "Add",
            inputs=["mul_output", "A"],
            outputs=["C"],
            name="add_1",
        ),
    ],
    name="Main_graph",
    inputs=[
        helper.make_tensor_value_info("A", TensorProto.FLOAT, [3, 2]),
        helper.make_tensor_value_info("B", TensorProto.FLOAT, [3, 2]),
    ],
    outputs=[
        helper.make_tensor_value_info("C", TensorProto.FLOAT, [3, 2]),
    ],
)

model = helper.make_model(graph_proto)
checker.check_model(model, True)
save(model, "add_mul_add.onnx")
