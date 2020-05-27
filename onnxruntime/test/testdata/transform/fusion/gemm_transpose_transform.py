import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto, OperatorSetIdProto
from onnx import numpy_helper
import numpy as np

# Gelu formula: x * 0.5 * (1.0 + tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)))

has_bias = True  # change it to True to generate fast_gelu_with_bias.onnx
gelu_use_graph_input = True  # change it to False to let Gelu don't have graph inputs as inputs.

X = helper.make_tensor_value_info('input', TensorProto.FLOAT, ["batchxseqlen", 64])
B = helper.make_tensor_value_info('input2', TensorProto.FLOAT, ["batchxseqlen", 16])
Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [16, 64])

bias_np_vals = np.asarray([0.0]).astype(np.float32).reshape(())
bias_initializer = numpy_helper.from_array(bias_np_vals, "input_bias")

nodes = []
leading_identity = helper.make_node('Identity', ["input"], ['identity_leading'], name="identity_leading")
nodes.append(leading_identity)


leading_identity2 = helper.make_node('Identity', ["input2"], ['identity_leading2'], name="identity_leading2")
nodes.append(leading_identity2)

mul1 = helper.make_node('Gemm', ["identity_leading", "identity_leading2", bias_initializer.name], ['gemm1'], name="gemm1", transA=1, transB=0, alpha=1.0, beta=0.0)
nodes.append(mul1)

transpose = helper.make_node('Transpose', ['gemm1'], ['trans1'], name="trans1", perm=[1,0])
nodes.append(transpose)

output_identity = helper.make_node('Identity', ['trans1'], ['output'], name="output")
nodes.append(output_identity)

# Create the graph (GraphProto)
graph_def = helper.make_graph(nodes, 'test-model', [X, B], [Y], [bias_initializer])

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

model_def = helper.make_model(graph_def, producer_name='onnx-example', **kwargs)

onnx.save(model_def, "gemm_transpose_transform.onnx")
