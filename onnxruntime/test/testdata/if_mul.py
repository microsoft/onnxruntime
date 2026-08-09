from onnx import TensorProto, checker, helper, save, shape_inference

# if A: C = B * 2
# else: C = B * 3

if_then_branch = helper.make_graph(
    nodes=[
        helper.make_node(
            "Mul",
            inputs=["B", "ConstTwo"],
            outputs=["if_output"],
            name="mul_0",
        ),
    ],
    name="if_then_branch",
    inputs=[
        # No explicit inputs
    ],
    outputs=[
        helper.make_tensor_value_info("if_output", TensorProto.FLOAT, [3, 2]),
    ],
)

if_else_branch = helper.make_graph(
    nodes=[
        helper.make_node(
            "Mul",
            inputs=["B", "ConstThree"],
            outputs=["if_output"],
            name="mul_1",
        ),
    ],
    name="if_else_branch",
    inputs=[
        # No explicit inputs
    ],
    outputs=[
        helper.make_tensor_value_info("if_output", TensorProto.FLOAT, [3, 2]),
    ],
)

graph_proto = helper.make_graph(
    nodes=[
        helper.make_node(
            "If",
            inputs=["A"],
            outputs=["C"],
            name="if_0",
            then_branch=if_then_branch,
            else_branch=if_else_branch,
        ),
    ],
    name="Main_graph",
    inputs=[
        helper.make_tensor_value_info("A", TensorProto.BOOL, [1]),
        helper.make_tensor_value_info("B", TensorProto.FLOAT, [3, 2]),
    ],
    outputs=[
        helper.make_tensor_value_info("C", TensorProto.FLOAT, [3, 2]),
    ],
    initializer=[
        helper.make_tensor("ConstTwo", TensorProto.FLOAT, [3, 2], [2.0] * 6),
        helper.make_tensor("ConstThree", TensorProto.FLOAT, [3, 2], [3.0] * 6),
    ],
)

model = helper.make_model(graph_proto)
model = shape_inference.infer_shapes(model)
checker.check_model(model, True)
save(model, "if_mul.onnx")
