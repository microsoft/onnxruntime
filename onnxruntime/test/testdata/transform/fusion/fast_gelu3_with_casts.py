import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto, OperatorSetIdProto
from onnx import numpy_helper
import numpy as np

# Gelu formula: x * 0.5 * (1.0 + tanh((sqrt(2 / pi) * (x + 0.044715 * pow(x, 3)))))

X = helper.make_tensor_value_info('input', TensorProto.FLOAT, ["batch", "seqlen", 64])
Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, ["batch", "seqlen", 64])

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
leading_identity = helper.make_node('Identity', [gelu_input], ['identity_leading'], name="identity_leading")
gelu_input = "identity_leading"
nodes.append(leading_identity)

mul_input_name = gelu_input

cast1 = helper.make_node('Cast', [mul_input_name], ['cast1'], name='cast1', to=1)
nodes.append(cast1)

pow1 = helper.make_node('Pow', ['cast1', pow_initializer.name], ['pow1'], name="pow1")
nodes.append(pow1)

mul1 = helper.make_node('Mul', ['pow1', a_weight_initializer.name], ['mul1'], name="mul1")
nodes.append(mul1)

cast2 = helper.make_node('Cast', [mul_input_name], ['cast2'], name='cast2', to=1)
nodes.append(cast2)

add1 = helper.make_node('Add', ['mul1', 'cast2'], ['add1'], name="add1")
nodes.append(add1)

mul2 = helper.make_node('Mul', ['add1', b_weight_initializer.name], ['mul2'], name="mul2")
nodes.append(mul2)

tanh = helper.make_node('Tanh', ['mul2'], ['tanh'], name="tanh")
nodes.append(tanh)

add2 = helper.make_node('Add', ['tanh', b_bias_initializer.name], ['add2'], name="add2")
nodes.append(add2)

mul5 = helper.make_node('Mul', [mul_input_name, c_weight_initializer.name], ['mul5'], name="mul5")
nodes.append(mul5)

cast3 = helper.make_node('Cast', ['mul5'], ['cast3'], name='cast3', to=1)
nodes.append(cast3)

mul6 = helper.make_node('Mul', ['cast3', 'add2'], ['mul6'], name="mul6")
ending_identity = helper.make_node('Identity', ['mul6'], ['output'], name="ending_identity")
nodes.extend([mul6, ending_identity])

initializers = []

initializers.extend(
    [pow_initializer, a_weight_initializer, b_weight_initializer, b_bias_initializer, c_weight_initializer])
# Create the graph (GraphProto)
graph_def = helper.make_graph(nodes, 'test-model', [X], [Y], initializers)

opsets = []
onnxdomain = OperatorSetIdProto()
onnxdomain.version = 13
onnxdomain.domain = ""  # The empty string ("") or absence of this field implies the operator set that is defined as part of the ONNX specification.
opsets.append(onnxdomain)

msdomain = OperatorSetIdProto()
msdomain.version = 1
msdomain.domain = "com.microsoft"

opsets.append(msdomain)
kwargs = {}
kwargs["opset_imports"] = opsets

model_def = helper.make_model(graph_def, producer_name='onnx-example', **kwargs)

onnx.save(model_def, "fast_gelu3_with_casts.onnx")
