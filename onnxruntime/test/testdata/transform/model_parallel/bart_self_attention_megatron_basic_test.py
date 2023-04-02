import random  # noqa: F401

import numpy as np
import onnx
from onnx import GraphProto, OperatorSetIdProto, TensorProto, helper, numpy_helper  # noqa: F401

batch = 6
hidden_size = 4
attention_head = 2
hidden_per_attention = 2

relative_attention_num_buckets = 32
input_len = 8
output_len = 8

X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [batch, input_len, hidden_size])
Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [output_len, batch, hidden_size])

q_weight_np_vals = (0.01 * np.arange(hidden_size * hidden_size, dtype=np.float32)).reshape((hidden_size, hidden_size))
q_weight_initializer = numpy_helper.from_array(q_weight_np_vals, "encoder.layers.0.self_attn.q_proj.weight")

k_weight_np_vals = (0.01 * np.arange(hidden_size * hidden_size, dtype=np.float32)).reshape((hidden_size, hidden_size))
k_weight_initializer = numpy_helper.from_array(k_weight_np_vals, "encoder.layers.0.self_attn.k_proj.weight")

v_weight_np_vals = (0.01 * np.arange(hidden_size * hidden_size, dtype=np.float32)).reshape((hidden_size, hidden_size))
v_weight_initializer = numpy_helper.from_array(v_weight_np_vals, "encoder.layers.0.self_attn.v_proj.weight")

q_bias_np_vals = 0.01 * np.arange(hidden_size, dtype=np.float32)
q_bias_initializer = numpy_helper.from_array(q_bias_np_vals, "encoder.layers.0.self_attn.q_proj.bias")

k_bias_np_vals = 0.01 * np.arange(hidden_size, dtype=np.float32)
k_bias_initializer = numpy_helper.from_array(k_bias_np_vals, "encoder.layers.0.self_attn.k_proj.bias")

v_bias_np_vals = 0.01 * np.arange(hidden_size, dtype=np.float32)
v_bias_initializer = numpy_helper.from_array(v_bias_np_vals, "encoder.layers.0.self_attn.v_proj.bias")

q_shape_initializer = numpy_helper.from_array(
    np.asarray([input_len, batch * attention_head, hidden_per_attention], dtype=np.int64),
    "q_shape",
)
k_shape_initializer = numpy_helper.from_array(
    np.asarray([-1, batch * attention_head, hidden_per_attention], dtype=np.int64),
    "k_shape",
)
v_shape_initializer = numpy_helper.from_array(
    np.asarray([-1, batch * attention_head, hidden_per_attention], dtype=np.int64),
    "v_shape",
)

mul_np_vals = np.asarray([0.1767766922712326], dtype=np.float32).reshape(())
mul_initializer = numpy_helper.from_array(mul_np_vals, "mul_const")

qk_shape_initializer = numpy_helper.from_array(
    np.asarray([batch, attention_head, input_len, input_len], dtype=np.int64),
    "qk_shape",
)

dummy_condition_initializer = numpy_helper.from_array(np.zeros((batch, input_len), dtype=bool), "dummy_cond")
inf_const_initializer = numpy_helper.from_array(np.asarray([-np.inf], dtype=np.float32), "inf_const")

where_shape_initializer = numpy_helper.from_array(
    np.asarray([batch * attention_head, input_len, input_len], dtype=np.int64),
    "where_shape",
)

dropout_np_vals = np.asarray([0.1], dtype=np.float32).reshape(())
dropout_initializer = numpy_helper.from_array(dropout_np_vals, "ratio")

dropout_mode_np_vals = np.array([False], dtype=bool).reshape(())
dropout_mode_initializer = numpy_helper.from_array(dropout_mode_np_vals, "mode")

shape_initializer3 = numpy_helper.from_array(
    np.array([input_len, batch, attention_head * hidden_per_attention], dtype=np.int64),
    "concat_shape_3",
)

dense_weight_np_vals = (0.01 * np.arange(hidden_size * hidden_size, dtype=np.float32)).reshape(
    (hidden_size, hidden_size)
)
dense_weight_initializer = numpy_helper.from_array(dense_weight_np_vals, "encoder.layers.0.self_attn.out_proj.weight")

dense_bias_np_vals = 0.01 * np.arange(hidden_size, dtype=np.float32)
dense_bias_initializer = numpy_helper.from_array(dense_bias_np_vals, "encoder.layers.0.self_attn.out_proj.bias")


transpose_ip = helper.make_node("Transpose", ["input"], ["transpose_ip"], name="transpose_ip", perm=[1, 0, 2])

transpose_q = helper.make_node(
    "Transpose",
    [q_weight_initializer.name],
    ["transpose_q"],
    name="transpose_q",
    perm=[1, 0],
)
transpose_k = helper.make_node(
    "Transpose",
    [k_weight_initializer.name],
    ["transpose_k"],
    name="transpose_k",
    perm=[1, 0],
)
transpose_v = helper.make_node(
    "Transpose",
    [v_weight_initializer.name],
    ["transpose_v"],
    name="transpose_v",
    perm=[1, 0],
)

matmul_q = helper.make_node("MatMul", ["transpose_ip", "transpose_q"], ["matmul_q"], name="matmul_q")
matmul_k = helper.make_node("MatMul", ["transpose_ip", "transpose_k"], ["matmul_k"], name="matmul_k")
matmul_v = helper.make_node("MatMul", ["transpose_ip", "transpose_v"], ["matmul_v"], name="matmul_v")


add_q = helper.make_node("Add", ["matmul_q", q_bias_initializer.name], ["add_q"], name="add_q")
add_k = helper.make_node("Add", ["matmul_k", k_bias_initializer.name], ["add_k"], name="add_k")
add_v = helper.make_node("Add", ["matmul_v", v_bias_initializer.name], ["add_v"], name="add_v")

mul_q = helper.make_node("Mul", ["add_q", "mul_const"], ["mul_q"], name="mul_q")

reshape_q = helper.make_node("Reshape", ["mul_q", q_shape_initializer.name], ["reshape_q"], name="reshape_q")
reshape_k = helper.make_node("Reshape", ["add_k", k_shape_initializer.name], ["reshape_k"], name="reshape_k")
reshape_v = helper.make_node("Reshape", ["add_v", v_shape_initializer.name], ["reshape_v"], name="reshape_v")

transpose_q2 = helper.make_node("Transpose", ["reshape_q"], ["transpose_q2"], name="transpose_q2", perm=[1, 0, 2])
transpose_k2 = helper.make_node("Transpose", ["reshape_k"], ["transpose_k2"], name="transpose_k2", perm=[1, 2, 0])
transpose_v2 = helper.make_node("Transpose", ["reshape_v"], ["transpose_v2"], name="transpose_v2", perm=[1, 0, 2])

matmul = helper.make_node("MatMul", ["transpose_q2", "transpose_k2"], ["matmul"], name="matmul")
reshape_qk = helper.make_node("Reshape", ["matmul", qk_shape_initializer.name], ["reshape_qk"], name="reshape_qk")


unsqueeze = helper.make_node(
    "Unsqueeze",
    [dummy_condition_initializer.name],
    ["unsqueeze_cond"],
    axes=[1, 2],
    name="unsqueeze_cond",
)
where = helper.make_node(
    "Where",
    ["unsqueeze_cond", inf_const_initializer.name, "reshape_qk"],
    ["where"],
    name="where",
)

reshape_where = helper.make_node(
    "Reshape",
    ["where", where_shape_initializer.name],
    ["reshape_where"],
    name="reshape_where",
)

softmax = helper.make_node("Softmax", ["reshape_where"], ["softmax"], name="softmax", axis=2)
dropout1 = helper.make_node(
    "Dropout",
    ["softmax", dropout_initializer.name, dropout_mode_initializer.name],
    ["dropout1", "dropout1_mask"],
    name="dropout1",
)

matmul2 = helper.make_node("MatMul", ["dropout1", "transpose_v2"], ["matmul2"], name="matmul2")
transpose = helper.make_node("Transpose", ["matmul2"], ["transpose"], name="transpose", perm=[1, 0, 2])
reshape = helper.make_node("Reshape", ["transpose", shape_initializer3.name], ["reshape"], name="reshape")

transpose_o_weight = helper.make_node(
    "Transpose",
    [dense_weight_initializer.name],
    ["transpose_o_weight"],
    name="transpose_o_weight",
    perm=[1, 0],
)
matmul3 = helper.make_node("MatMul", ["reshape", "transpose_o_weight"], ["matmul3"], name="matmul3")
add3 = helper.make_node("Add", ["matmul3", dense_bias_initializer.name], ["add3"], name="add3")
identity = helper.make_node("Identity", ["add3"], ["output"], name="identity")

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [
        transpose_ip,
        transpose_q,
        transpose_k,
        transpose_v,
        matmul_q,
        matmul_k,
        matmul_v,
        add_q,
        add_k,
        add_v,
        mul_q,
        reshape_q,
        reshape_k,
        reshape_v,
        transpose_q2,
        transpose_k2,
        transpose_v2,
        matmul,
        reshape_qk,
        unsqueeze,
        where,
        reshape_where,
        softmax,
        dropout1,
        matmul2,
        transpose,
        reshape,
        transpose_o_weight,
        matmul3,
        add3,
        identity,
    ],
    "self-attention-megatron-test-model",
    [X],
    [Y],
    [
        q_weight_initializer,
        k_weight_initializer,
        v_weight_initializer,
        q_bias_initializer,
        k_bias_initializer,
        v_bias_initializer,
        q_shape_initializer,
        k_shape_initializer,
        v_shape_initializer,
        mul_initializer,
        qk_shape_initializer,
        dummy_condition_initializer,
        inf_const_initializer,
        where_shape_initializer,
        dropout_initializer,
        dropout_mode_initializer,
        shape_initializer3,
        dense_weight_initializer,
        dense_bias_initializer,
    ],
)

opsets = []
onnxdomain = OperatorSetIdProto()
onnxdomain.version = 12
onnxdomain.domain = ""  # The empty string ("") or absence of this field implies the operator set that is defined as part of the ONNX specification.
opsets.append(onnxdomain)

msdomain = OperatorSetIdProto()
msdomain.version = 1
msdomain.domain = "com.microsoft"

opsets.append(msdomain)
kwargs = {}
kwargs["opset_imports"] = opsets

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name="onnx-example", **kwargs)
onnx.save(model_def, "bart_self_attention_megatron_basic_test.onnx")
