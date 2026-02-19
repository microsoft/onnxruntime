from onnx import TensorProto, checker, helper, save, shape_inference

batch_size = 1
seq_len = 64
hidden_size = 896

input_vi = helper.make_tensor_value_info(
    name="input",
    elem_type=TensorProto.FLOAT,
    shape=[batch_size, seq_len, hidden_size],
)

skip_vi = helper.make_tensor_value_info(
    name="skip",
    elem_type=TensorProto.FLOAT,
    shape=[batch_size, seq_len, hidden_size],
)

output_vi = helper.make_tensor_value_info(
    name="output",
    elem_type=TensorProto.FLOAT,
    shape=[batch_size, seq_len, hidden_size],
)

input_skip_bias_sum_vi = helper.make_tensor_value_info(
    name="input_skip_bias_sum",
    elem_type=TensorProto.FLOAT,
    shape=[batch_size, seq_len, hidden_size],
)

gamma_init = helper.make_tensor(
    name="gamma",
    data_type=TensorProto.FLOAT,
    dims=[hidden_size],
    vals=[1] * hidden_size
)

node = helper.make_node(
    op_type="SkipSimplifiedLayerNormalization",
    inputs=["input", "skip", "gamma"],
    outputs=["output", "", "", "input_skip_bias_sum"],
    domain="com.microsoft",
    epsilon=1e-6,
    name="SkipLayerNorm",
)

graph = helper.make_graph(
    nodes=[node],
    name="SkipSimplifiedLayerNormGraph",
    inputs=[input_vi, skip_vi],
    outputs=[output_vi, input_skip_bias_sum_vi],
    initializer=[gamma_init],
)

model = helper.make_model(
    graph,
    opset_imports=[
        helper.make_operatorsetid("", 17),
        helper.make_operatorsetid("com.microsoft", 1),
    ],
)

model = shape_inference.infer_shapes(model)
checker.check_model(model, True)
save(model, "skip_simplified_layer_normalization.onnx")
