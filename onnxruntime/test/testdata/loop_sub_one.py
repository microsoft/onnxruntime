from onnx import TensorProto, checker, helper, save, shape_inference

"""
x = A
for (int i = 0; i < MAX_ITERS; i++) {
  y = x - 1.0
  user_val = x - 1.0
  x = y
}
C = x
D = user_val (will be the concatenated result of all iterations)
"""

loop_body = helper.make_graph(
    nodes=[
        helper.make_node(
            "Sub",
            inputs=["loop_state_in", "ConstOne"],
            outputs=["loop_state_out"],
            name="sub_0",
        ),
        helper.make_node(
            "Sub",
            inputs=["loop_state_in", "ConstOne"],
            outputs=["user_defined_val"],
            name="sub_1",
        ),
    ],
    name="loop_body",
    inputs=[
        helper.make_tensor_value_info("index", TensorProto.INT64, [1]),
        helper.make_tensor_value_info("subgraph_keep_going_in", TensorProto.BOOL, [1]),
        helper.make_tensor_value_info("loop_state_in", TensorProto.FLOAT, [1]),
    ],
    outputs=[
        helper.make_tensor_value_info("subgraph_keep_going_in", TensorProto.BOOL, [1]),
        helper.make_tensor_value_info("loop_state_out", TensorProto.FLOAT, [1]),
        helper.make_tensor_value_info("user_defined_val", TensorProto.FLOAT, [1]),
    ],
    initializer=[
        helper.make_tensor("ConstOne", TensorProto.FLOAT, [1], [1.0]),
    ],
)

graph_proto = helper.make_graph(
    nodes=[
        helper.make_node(
            "Loop",
            inputs=["MAX_ITERS", "", "A"],
            outputs=["C", "D"],
            name="loop_0",
            body=loop_body,
        ),
    ],
    name="Main_graph",
    inputs=[
        helper.make_tensor_value_info("MAX_ITERS", TensorProto.INT64, [1]),
        helper.make_tensor_value_info("A", TensorProto.FLOAT, [1]),
    ],
    outputs=[
        helper.make_tensor_value_info("C", TensorProto.FLOAT, [1]),
        helper.make_tensor_value_info("D", TensorProto.FLOAT, None),
    ],
)

model = helper.make_model(graph_proto)
model = shape_inference.infer_shapes(model)
checker.check_model(model, True)
save(model, "loop_sub_one.onnx")
