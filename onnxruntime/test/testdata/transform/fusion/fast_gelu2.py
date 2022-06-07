import numpy as np
import onnx
from onnx import AttributeProto, GraphProto, OperatorSetIdProto, TensorProto, helper, numpy_helper

# Gelu formula: x * 0.5 * (1.0 + tanh((sqrt(2 / pi) * (x + 0.044715 * pow(x, 3)))))
has_bias = False  # change it to True to generate fast_gelu_openai_with_bias.onnx
gelu_use_graph_input = True  # change it to False to let Gelu don't have graph inputs/outputs as inputs/outputs.

X = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", "seqlen", 64])
Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", "seqlen", 64])

bias_np_vals = (0.01 * np.arange(64)).astype(np.float32).reshape((64))
bias_initializer = numpy_helper.from_array(bias_np_vals, "input_bias")

pow_np_vals = np.asarray([3]).astype(np.float32).reshape(())
pow_initializer = numpy_helper.from_array(pow_np_vals, "pow_init")

a_weight_np_vals = np.asarray([0.044714998453855515]).astype(np.float32).reshape(())
a_weight_initializer = numpy_helper.from_array(a_weight_np_vals, "mul1_init")

b_weight_np_vals = np.asarray([0.7978845834732056]).astype(np.float32).reshape(())
b_weight_initializer = numpy_helper.from_array(b_weight_np_vals, "mul2_init")

c_weight_np_vals = np.asarray([0.5]).astype(np.float32).reshape(())
c_weight_initializer = numpy_helper.from_array(c_weight_np_vals, "mul3_init")

b_bias_np_vals = np.asarray([1.0]).astype(np.float32).reshape(())
b_bias_initializer = numpy_helper.from_array(b_bias_np_vals, "add2_init")

nodes = []
gelu_input = "input"
if not gelu_use_graph_input:
    leading_identity = helper.make_node("Identity", [gelu_input], ["identity_leading"], name="identity_leading")
    gelu_input = "identity_leading"
    nodes.append(leading_identity)

mul_input_name = gelu_input
if has_bias:
    add0 = helper.make_node("Add", [gelu_input, bias_initializer.name], ["add0"], name="add0")
    mul_input_name = "add0"
    nodes.append(add0)

pow1 = helper.make_node("Pow", [mul_input_name, pow_initializer.name], ["pow1"], name="pow1")
nodes.append(pow1)

mul1 = helper.make_node("Mul", ["pow1", a_weight_initializer.name], ["mul1"], name="mul1")
nodes.append(mul1)

add1 = helper.make_node("Add", [mul_input_name, "mul1"], ["add1"], name="add1")
nodes.append(add1)

mul2 = helper.make_node("Mul", ["add1", b_weight_initializer.name], ["mul2"], name="mul2")
nodes.append(mul2)

tanh = helper.make_node("Tanh", ["mul2"], ["tanh"], name="tanh")
nodes.append(tanh)

add2 = helper.make_node("Add", ["tanh", b_bias_initializer.name], ["add2"], name="add2")
nodes.append(add2)

mul5 = helper.make_node("Mul", [mul_input_name, c_weight_initializer.name], ["mul5"], name="mul5")
nodes.append(mul5)

mul6 = helper.make_node("Mul", ["mul5", "add2"], ["mul6"], name="mul6")
ending_identity = helper.make_node("Identity", ["mul6"], ["output"], name="ending_identity")
nodes.extend([mul6, ending_identity])

initializers = []
if has_bias:
    initializers = [bias_initializer]

initializers.extend(
    [
        pow_initializer,
        a_weight_initializer,
        b_weight_initializer,
        b_bias_initializer,
        c_weight_initializer,
    ]
)
# Create the graph (GraphProto)
graph_def = helper.make_graph(nodes, "test-model", [X], [Y], initializers)

opsets = []
onnxdomain = OperatorSetIdProto()
onnxdomain.version = 10
onnxdomain.domain = ""  # The empty string ("") or absence of this field implies the operator set that is defined as part of the ONNX specification.
opsets.append(onnxdomain)

msdomain = OperatorSetIdProto()
msdomain.version = 1
msdomain.domain = "com.microsoft"

opsets.append(msdomain)
kwargs = {}
kwargs["opset_imports"] = opsets

model_def = helper.make_model(graph_def, producer_name="onnx-example", **kwargs)

file_name = "fast_gelu2"
if has_bias:
    file_name += "_with_bias"

if gelu_use_graph_input:
    file_name += "_use_graph_input"
onnx.save(model_def, file_name + ".onnx")
