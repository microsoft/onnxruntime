from onnx import TensorProto, checker, helper, save, shape_inference

# A --> Squeeze --> Mul --> C
#                   ^
#                   |
# B ----------------+
graph_proto = helper.make_graph(
    nodes=[
        helper.make_node(
            "Squeeze",
            inputs=["A"],
            outputs=["squeeze_output"],
            name="squeeze_0",
        ),
        helper.make_node(
            "Mul",
            inputs=["squeeze_output", "B"],
            outputs=["C"],
            name="mul_0",
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
)

model = helper.make_model(graph_proto)
model = shape_inference.infer_shapes(model)
checker.check_model(model, True)
save(model, "squeeze_mul.onnx")
