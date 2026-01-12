from onnx import TensorProto, checker, helper, save, shape_inference

# Scan body: y_t = x_t * 2.0
scan_body = helper.make_graph(
    nodes=[
        helper.make_node(
            "Mul",
            inputs=["x_t", "ConstTwo"],
            outputs=["y_t"],
            name="mul_0",
        ),
    ],
    name="scan_body",
    inputs=[
        helper.make_tensor_value_info("x_t", TensorProto.FLOAT, [3]),
    ],
    outputs=[
        helper.make_tensor_value_info("y_t", TensorProto.FLOAT, [3]),
    ],
    initializer=[
        helper.make_tensor("ConstTwo", TensorProto.FLOAT, [3], [2.0] * 3),
    ],
)

# Top graph: Y = Scan(X)
graph_proto = helper.make_graph(
    nodes=[
        helper.make_node(
            "Scan",
            inputs=["X"],
            outputs=["Y"],
            name="scan_0",
            body=scan_body,
            num_scan_inputs=1,
            scan_input_axes=[1],
            scan_output_axes=[1],
        ),
    ],
    name="Main_graph",
    inputs=[
        helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 3]),
    ],
    outputs=[
        helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 3]),
    ],
)

model = helper.make_model(graph_proto)
model = shape_inference.infer_shapes(model)
checker.check_model(model, True)
save(model, "scan_mul.onnx")
