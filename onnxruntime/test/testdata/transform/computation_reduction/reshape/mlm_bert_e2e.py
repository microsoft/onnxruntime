import onnx
from onnx import OperatorSetIdProto, TensorProto, helper

# inputs and outputs
hidden = 1024
head = 16
vocab_size = 30522
inputs = [
    helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch_size", "sequence_length", hidden]),
    helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch_size", "sequence_length"]),
    helper.make_tensor_value_info("matmul1.weight", TensorProto.FLOAT16, [hidden, 1024]),
    helper.make_tensor_value_info("add1.bias", TensorProto.FLOAT16, [hidden]),
    helper.make_tensor_value_info("matmul2.weight", TensorProto.FLOAT16, [hidden, 1024]),
    helper.make_tensor_value_info("add2.bias", TensorProto.FLOAT16, [hidden]),
    helper.make_tensor_value_info("matmul3.weight", TensorProto.FLOAT16, [hidden, 1024]),
    helper.make_tensor_value_info("add3.bias", TensorProto.FLOAT16, [hidden]),
    helper.make_tensor_value_info("matmul4.weight", TensorProto.FLOAT16, [hidden, 1024]),
    helper.make_tensor_value_info("add4.bias", TensorProto.FLOAT16, [hidden]),
    helper.make_tensor_value_info("layer_norm1.weight", TensorProto.FLOAT, [hidden]),
    helper.make_tensor_value_info("layer_norm1.bias", TensorProto.FLOAT, [hidden]),
    helper.make_tensor_value_info("matmul7.weight", TensorProto.FLOAT16, [hidden, hidden * 4]),
    helper.make_tensor_value_info("add7.bias", TensorProto.FLOAT16, [hidden * 4]),
    helper.make_tensor_value_info("matmul8.weight", TensorProto.FLOAT16, [hidden * 4, hidden]),
    helper.make_tensor_value_info("add8.bias", TensorProto.FLOAT16, [hidden]),
    helper.make_tensor_value_info("layer_norm2.weight", TensorProto.FLOAT, [hidden]),
    helper.make_tensor_value_info("layer_norm2.bias", TensorProto.FLOAT, [hidden]),
    helper.make_tensor_value_info("matmul9.weight", TensorProto.FLOAT16, [hidden, hidden]),
    helper.make_tensor_value_info("add9.bias", TensorProto.FLOAT16, [hidden]),
    helper.make_tensor_value_info("layer_norm3.weight", TensorProto.FLOAT, [hidden]),
    helper.make_tensor_value_info("layer_norm3.bias", TensorProto.FLOAT, [hidden]),
    helper.make_tensor_value_info("matmul10.weight", TensorProto.FLOAT16, [hidden, vocab_size]),
    helper.make_tensor_value_info("add10.bias", TensorProto.FLOAT16, [vocab_size]),
    helper.make_tensor_value_info("labels", TensorProto.INT64, ["batch_size*sequence_length"]),
]

outputs = [
    # helper.make_tensor_value_info("output-0", TensorProto.FLOAT16, ["batch_size", "sequence_length", vocab_size]),
    helper.make_tensor_value_info("output-1", TensorProto.FLOAT, []),
]

# initializers

initializers = [
    helper.make_tensor("scalar_float_0.1", TensorProto.FLOAT, [], [0.1]),
    helper.make_tensor("scalar_float_0", TensorProto.FLOAT, [], [0.0]),
    helper.make_tensor("scalar_float16_8", TensorProto.FLOAT16, [], [8]),
    helper.make_tensor("scalar_bool_true", TensorProto.BOOL, [], [1]),
    helper.make_tensor("scalar_float_1", TensorProto.FLOAT, [], [1]),
    helper.make_tensor("scalar_float_big_num", TensorProto.FLOAT, [], [-3.4028234663852886e38]),
    helper.make_tensor("scalar_int_0", TensorProto.INT64, [], [0]),
    helper.make_tensor("scalar_int_1", TensorProto.INT64, [], [1]),
    helper.make_tensor("padding_idx", TensorProto.INT64, [], [-100]),
    helper.make_tensor("single_value_1d_int_0", TensorProto.INT64, [1], [0]),
    helper.make_tensor("single_value_1d_int_1", TensorProto.INT64, [1], [1]),
    helper.make_tensor("single_value_1d_int_2", TensorProto.INT64, [1], [2]),
    helper.make_tensor("single_value_1d_int_16", TensorProto.INT64, [1], [head]),
    helper.make_tensor("single_value_1d_int_64", TensorProto.INT64, [1], [hidden // head]),
    helper.make_tensor("single_value_1d_int_1024", TensorProto.INT64, [1], [hidden]),
    helper.make_tensor("shape1", TensorProto.INT64, [4], [0, 0, 16, 64]),
    helper.make_tensor("shape2", TensorProto.INT64, [4], [0, 0, 16, 64]),
    helper.make_tensor("shape3", TensorProto.INT64, [4], [0, 0, 16, 64]),
    helper.make_tensor("shape4", TensorProto.INT64, [3], [0, 0, 1024]),
    helper.make_tensor("shape5", TensorProto.INT64, [2], [-1, vocab_size]),
]

# nodes

nodes = [
    helper.make_node("Dropout", ["input", "scalar_float_0", "scalar_bool_true"], ["d1_out", "d1_mask"], "d1"),
    helper.make_node("Cast", ["d1_out"], ["c1_out"], name="c1", to=10),
    # attention
    ## left branch
    helper.make_node("MatMul", ["c1_out", "matmul1.weight"], ["m1_out"], "m1"),
    helper.make_node("Add", ["add1.bias", "m1_out"], ["a1_out"], "a1"),
    helper.make_node("Reshape", ["a1_out", "shape1"], ["reshape1_out"], "reshape1"),
    helper.make_node("Transpose", ["reshape1_out"], ["transpose1_out"], name="transpose1", perm=[0, 2, 1, 3]),
    ## middle branch
    helper.make_node("MatMul", ["c1_out", "matmul2.weight"], ["m2_out"], "m2"),
    helper.make_node("Add", ["add2.bias", "m2_out"], ["a2_out"], "a2"),
    helper.make_node("Reshape", ["a2_out", "shape2"], ["reshape2_out"], "reshape2"),
    helper.make_node("Transpose", ["reshape2_out"], ["transpose2_out"], name="transpose2", perm=[0, 2, 1, 3]),
    ## right banch
    helper.make_node("MatMul", ["c1_out", "matmul3.weight"], ["m3_out"], "m3"),
    helper.make_node("Add", ["add3.bias", "m3_out"], ["a3_out"], "a3"),
    helper.make_node("Reshape", ["a3_out", "shape3"], ["reshape3_out"], "reshape3"),
    helper.make_node("Transpose", ["reshape3_out"], ["transpose3_out"], name="transpose3", perm=[0, 2, 3, 1]),
    ## middle branch result computes with right branch result
    helper.make_node("MatMul", ["transpose2_out", "transpose3_out"], ["m4_out"], "m4"),
    helper.make_node("Div", ["m4_out", "scalar_float16_8"], ["div1_out"], "div1"),
    helper.make_node("Cast", ["div1_out"], ["c2_out"], name="c2", to=1),
    helper.make_node("Unsqueeze", ["attention_mask", "single_value_1d_int_1"], ["unsqueeze7_out"], "unsqueeze7"),
    helper.make_node("Unsqueeze", ["unsqueeze7_out", "single_value_1d_int_2"], ["unsqueeze8_out"], "unsqueeze8"),
    helper.make_node("Cast", ["unsqueeze8_out"], ["c3_out"], name="c3", to=1),
    helper.make_node("Sub", ["scalar_float_1", "c3_out"], ["sub1_out"], "sub1"),
    helper.make_node("Mul", ["sub1_out", "scalar_float_big_num"], ["mul1_out"], "mul1"),
    helper.make_node("Add", ["mul1_out", "c2_out"], ["a4_out"], "a4"),
    helper.make_node("Softmax", ["a4_out"], ["softmax1_out"], "softmax1", axis=-1),
    helper.make_node("Dropout", ["softmax1_out", "scalar_float_0", "scalar_bool_true"], ["d2_out", "d2_mask"], "d2"),
    helper.make_node("Cast", ["d2_out"], ["c4_out"], name="c4", to=10),
    ## left branch result computes with result of `middle branch result computes with right branch result``
    helper.make_node("MatMul", ["c4_out", "transpose1_out"], ["m5_out"], "m5"),
    helper.make_node("Transpose", ["m5_out"], ["tranpose4_out"], name="tranpose4", perm=[0, 2, 1, 3]),
    helper.make_node("Reshape", ["tranpose4_out", "shape4"], ["reshape4_out"], "reshape4"),
    ## attention output
    helper.make_node("MatMul", ["reshape4_out", "matmul4.weight"], ["m6_out"], "m6"),
    helper.make_node("Add", ["add4.bias", "m6_out"], ["a5_out"], "a5"),
    helper.make_node("Dropout", ["a5_out", "scalar_float_0", "scalar_bool_true"], ["d4_out", 'd4_mask"'], "d4"),
    helper.make_node("Cast", ["d4_out"], ["c5_out"], name="c5", to=1),
    helper.make_node("Add", ["d1_out", "c5_out"], ["a6_out"], "a6"),
    # MLP
    helper.make_node(
        "LayerNormalization",
        ["a6_out", "layer_norm1.weight", "layer_norm1.bias"],
        ["layernorm1_out", "layernorm1_mean", "layernorm1_var"],
        "layernorm1",
        axis=-1,
        epsion=0.000009999999747378752,
    ),
    helper.make_node("Cast", ["layernorm1_out"], ["c6_out"], name="c6", to=10),
    helper.make_node("MatMul", ["c6_out", "matmul7.weight"], ["m7_out"], "m7"),
    helper.make_node("BiasGelu", ["m7_out", "add7.bias"], ["biasgelu1_out"], "biasgelu1", domain="com.microsoft"),
    helper.make_node("MatMul", ["biasgelu1_out", "matmul8.weight"], ["m8_out"], "m8"),
    helper.make_node("Add", ["add8.bias", "m8_out"], ["a7_out"], "a7"),
    helper.make_node("Dropout", ["a7_out", "scalar_float_0", "scalar_bool_true"], ["d5_out", "d5_mask"], "d5"),
    helper.make_node("Cast", ["d5_out"], ["c7_out"], name="c7", to=1),
    helper.make_node("Add", ["layernorm1_out", "c7_out"], ["a8_out"], "a8"),
    helper.make_node(
        "LayerNormalization",
        ["a8_out", "layer_norm2.weight", "layer_norm2.bias"],
        ["layernorm2_out", "layernorm2_mean", "layernorm2_var"],
        "layernorm2",
        axis=-1,
        epsion=0.000009999999747378752,
    ),
    helper.make_node("Cast", ["layernorm2_out"], ["c8_out"], name="c8", to=10),
    helper.make_node("MatMul", ["c8_out", "matmul9.weight"], ["m9_out"], "m9"),
    helper.make_node("BiasGelu", ["m9_out", "add9.bias"], ["biasgelu2_out"], "biasgelu2", domain="com.microsoft"),
    helper.make_node(
        "LayerNormalization",
        ["biasgelu2_out", "layer_norm3.weight", "layer_norm3.bias"],
        ["layernorm3_out", "layernorm3_mean", "layernorm3_var"],
        "layernorm3",
        axis=-1,
        epsion=0.000009999999747378752,
    ),
    helper.make_node("Cast", ["layernorm3_out"], ["c9_out"], name="c9", to=10),
    helper.make_node("MatMul", ["c9_out", "matmul10.weight"], ["m10_out"], "m10"),
    helper.make_node("Add", ["add10.bias", "m10_out"], ["a10_out"], "a10"),
    # helper.make_node("Identity", ["a10_out"], ["output-0"], "identity_output0"),
    helper.make_node("Reshape", ["a10_out", "shape5"], ["reshape5_out"], "reshape5"),
    helper.make_node("Cast", ["reshape5_out"], ["c10_out"], name="c10", to=1),
    helper.make_node(
        "SoftmaxCrossEntropyLossInternal",
        ["c10_out", "labels", "", "padding_idx"],
        ["sce_out0", "sce_out1"],
        "sceloss0",
        domain="com.microsoft",
    ),
    helper.make_node("Identity", ["sce_out0"], ["output-1"], "identity_output1"),
]


# Shapes that cannot be inferred by onnx shape inference
value_infos = [
    helper.make_value_info(
        name="reshape1_out",
        type_proto=helper.make_tensor_type_proto(
            elem_type=TensorProto.FLOAT16, shape=["batch_size", "sequence_length", head, hidden // head]
        ),
    ),
    helper.make_value_info(
        name="reshape2_out",
        type_proto=helper.make_tensor_type_proto(
            elem_type=TensorProto.FLOAT16, shape=["batch_size", "sequence_length", head, hidden // head]
        ),
    ),
    helper.make_value_info(
        name="reshape3_out",
        type_proto=helper.make_tensor_type_proto(
            elem_type=TensorProto.FLOAT16, shape=["batch_size", "sequence_length", head, hidden // head]
        ),
    ),
    helper.make_value_info(
        name="reshape4_out",
        type_proto=helper.make_tensor_type_proto(
            elem_type=TensorProto.FLOAT16, shape=["batch_size", "sequence_length", hidden]
        ),
    ),
    helper.make_value_info(
        name="reshape5_out",
        type_proto=helper.make_tensor_type_proto(
            elem_type=TensorProto.FLOAT16, shape=["batch_size*sequence_length", vocab_size]
        ),
    ),
    helper.make_value_info(
        name="layernorm1_out",
        type_proto=helper.make_tensor_type_proto(
            elem_type=TensorProto.FLOAT, shape=["batch_size", "sequence_length", hidden]
        ),
    ),
    helper.make_value_info(
        name="layernorm2_out",
        type_proto=helper.make_tensor_type_proto(
            elem_type=TensorProto.FLOAT, shape=["batch_size", "sequence_length", hidden]
        ),
    ),
    helper.make_value_info(
        name="layernorm3_out",
        type_proto=helper.make_tensor_type_proto(
            elem_type=TensorProto.FLOAT, shape=["batch_size", "sequence_length", hidden]
        ),
    ),
    helper.make_value_info(
        name="concattraining4_out",
        type_proto=helper.make_tensor_type_proto(elem_type=TensorProto.INT64, shape=[3]),
    ),
    helper.make_value_info(
        name="biasgelu1_out",
        type_proto=helper.make_tensor_type_proto(
            elem_type=TensorProto.FLOAT16, shape=["batch_size", "sequence_length", hidden * 4]
        ),
    ),
    helper.make_value_info(
        name="biasgelu2_out",
        type_proto=helper.make_tensor_type_proto(
            elem_type=TensorProto.FLOAT16, shape=["batch_size", "sequence_length", hidden]
        ),
    ),
]

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    nodes,
    "test-model",
    inputs,
    outputs,
    initializers,
    "doc string",
    value_infos,
)


opsets = []
onnxdomain = OperatorSetIdProto()
onnxdomain.version = 14
onnxdomain.domain = ""  # The empty string ("") or absence of this field implies the operator
# set that is defined as part of the ONNX specification.
opsets.append(onnxdomain)

msdomain = OperatorSetIdProto()
msdomain.version = 1
msdomain.domain = "com.microsoft"

opsets.append(msdomain)
kwargs = {}
kwargs["opset_imports"] = opsets


model_def = helper.make_model(graph_def, producer_name="onnx-example", **kwargs)
final_model = onnx.shape_inference.infer_shapes(model_def)
onnx.save(final_model, "mlm_bert_e2e.onnx")
