import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto, OperatorSetIdProto
from onnx import numpy_helper
import numpy as np

X = helper.make_tensor_value_info('input', TensorProto.FLOAT, ["batch", "seqlen", 128])
unsqueezed_masked_lm_positions = helper.make_tensor_value_info('unsqueezed_masked_lm_positions', 
                                                               TensorProto.INT64, ["batch", "dynamic_prediction_count", 1])
Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, ["batch", "dynamic_prediction_count", 128])
nodes = []

# case 1
gelu1 = helper.make_node('Gelu', ['input'], ['gelu_1'], name="gelu_1", domain='com.microsoft')
nodes.append(gelu1)

gathernd1 = helper.make_node('GatherND', ['gelu_1', 'unsqueezed_masked_lm_positions'], ['output'], name="gathernd_1", batch_dims=1)
nodes.append(gathernd1)

graph_def = helper.make_graph(nodes, 'test-model', [X, unsqueezed_masked_lm_positions], [Y])

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

model_def = helper.make_model(graph_def, producer_name='onnx-example', **kwargs)

onnx.save(model_def, "gathernd_gelu.onnx")
