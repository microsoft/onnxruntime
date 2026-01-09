from onnx import TensorProto, checker, helper, save, shape_inference

# A --> Mul --> Shape --> Relu --> Reshape(mul_output) --> B
graph_proto = helper.make_graph(
    nodes=[
        helper.make_node(
            "Mul",
            inputs=["A", "ConstTwo"],
            outputs=["mul_output"],
            name="mul_0",
        ),
        helper.make_node(
            "Shape",
            inputs=["mul_output"],
            outputs=["shape_output"],
            name="shape_0",
        ),
        helper.make_node(
            "Relu",
            inputs=["shape_output"],
            outputs=["relu_output"],
            name="relu_0",
        ),
        helper.make_node(
            "Reshape",
            inputs=["mul_output", "relu_output"],
            outputs=["B"],
            name="reshape_0",
        ),
    ],
    name="Main_graph",
    inputs=[
        helper.make_tensor_value_info("A", TensorProto.FLOAT, [3, 2]),
    ],
    outputs=[
        helper.make_tensor_value_info("B", TensorProto.FLOAT, [3, 2]),
    ],
    initializer=[
        helper.make_tensor("ConstTwo", TensorProto.FLOAT, [3, 2], [2.0] * 6),
    ],
)

model = helper.make_model(graph_proto)
model = shape_inference.infer_shapes(model)
checker.check_model(model, True)
save(model, "plugin_kernel_ep_cpu_preferred_nodes.onnx")
