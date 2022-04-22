import onnx
from onnx import OperatorSetIdProto, TensorProto, helper

# inputs/outputs
X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 1])
O1 = helper.make_tensor_value_info("O1", TensorProto.FLOAT, [2, 1])
O2 = helper.make_tensor_value_info("O2", TensorProto.FLOAT, [2, 1])
O3 = helper.make_tensor_value_info("O3", TensorProto.FLOAT, [2, 1])
O4 = helper.make_tensor_value_info("O4", TensorProto.FLOAT, [2, 1])
O5 = helper.make_tensor_value_info("O5", TensorProto.FLOAT, [2, 1])

X2 = helper.make_tensor_value_info("X2", TensorProto.FLOAT, [])

# initializers
zeroratio_float = helper.make_tensor("ratio_zero_float", TensorProto.FLOAT, [], [0.0])
zeroratio_double = helper.make_tensor("ratio_zero_double", TensorProto.DOUBLE, [], [0.0])
zeroratio_float16 = helper.make_tensor("ratio_zero_float16", TensorProto.FLOAT16, [], [0])
nonzeroratio = helper.make_tensor("ratio_nonzero", TensorProto.FLOAT, [], [0.1])
training_mode = helper.make_tensor("training_mode", TensorProto.BOOL, [], [1])

opsets = []
onnxdomain = OperatorSetIdProto()
onnxdomain.version = 12
onnxdomain.domain = ""  # The empty string ("") or absence of this field implies the operator set that is defined as part of the ONNX specification.
opsets.append(onnxdomain)

kwargs = {}
kwargs["opset_imports"] = opsets

# Create the model (ModelProto)
I1 = helper.make_node("Identity", ["X"], ["I1_out"], name="I1")
D1 = helper.make_node("Dropout", ["I1_out", "ratio_zero_float", "training_mode"], ["D1_out"], "D1")
I2 = helper.make_node("Identity", ["D1_out"], ["O1"], name="I2")

I3 = helper.make_node("Identity", ["X"], ["I3_out"], name="I3")
D2 = helper.make_node("Dropout", ["I3_out", "ratio_nonzero", "training_mode"], ["D2_out"], "D2")
I4 = helper.make_node("Identity", ["D2_out"], ["O2"], name="I4")

I5 = helper.make_node("Identity", ["X"], ["I5_out"], name="I5")
D3 = helper.make_node("Dropout", ["I5_out", "X2", "training_mode"], ["D3_out"], "D3")
I6 = helper.make_node("Identity", ["D3_out"], ["O3"], name="I6")

I7 = helper.make_node("Identity", ["X"], ["I7_out"], name="I7")
D4 = helper.make_node("Dropout", ["I7_out", "ratio_zero_double", "training_mode"], ["D4_out"], "D4")
I8 = helper.make_node("Identity", ["D4_out"], ["O4"], name="I8")

I9 = helper.make_node("Identity", ["X"], ["I9_out"], name="I9")
D5 = helper.make_node("Dropout", ["I9_out", "ratio_zero_float16", "training_mode"], ["D5_out"], "D5")
I10 = helper.make_node("Identity", ["D5_out"], ["O5"], name="I10")

graph = helper.make_graph(
    [I1, D1, I2, I3, D2, I4, I5, D3, I6, I7, D4, I8, I9, D5, I10],
    "Dropout_Elimination",  # name
    [X, X2],
    [O1, O2, O3, O4, O5],
    [zeroratio_float, zeroratio_double, zeroratio_float16, nonzeroratio, training_mode],
)

model = helper.make_model(graph, producer_name="onnx-example", **kwargs)
onnx.save(model, "dropout_ratio.onnx")
