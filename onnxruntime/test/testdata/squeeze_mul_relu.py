from onnx import TensorProto, checker, helper, save, shape_inference

# A --> Squeeze --> Mul --> Relu --> Mul(2x) --> C
#                   ^
#                   |
# B ----------------+
graph_proto = helper.make_graph(
    nodes=[
        helper.make_node(
            "Squeeze",
            inputs=["A"],
            outputs=["squeeze0_output"],
            name="squeeze_0",
        ),
        helper.make_node(
            "Mul",
            inputs=["squeeze0_output", "B"],
            outputs=["mul0_output"],
            name="mul_0",
        ),
        helper.make_node(
            "Relu",
            inputs=["mul0_output"],
            outputs=["relu0_output"],
            name="relu_0",
        ),
        helper.make_node(
            "Mul",
            inputs=["relu0_output", "Const2"],
            outputs=["C"],
            name="mul_1",
        ),
    ],
    name="Main_graph",
    inputs=[
        helper.make_tensor_value_info("A", TensorProto.FLOAT, [3, 1, 2]),
        helper.make_tensor_value_info("B", TensorProto.FLOAT, [3, 2]),
    ],
    outputs=[
        helper.make_tensor_value_info("C", TensorProto.FLOAT, [3, 2]),
    ],
    initializer=[
        helper.make_tensor("Const2", TensorProto.FLOAT, [3, 2], [2.0] * 6),
    ],
)

model = helper.make_model(graph_proto)
model = shape_inference.infer_shapes(model)
checker.check_model(model, True)
save(model, "squeeze_mul_relu.onnx")
