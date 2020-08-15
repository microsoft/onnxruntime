import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto, OperatorSetIdProto
from onnx import numpy_helper
import numpy as np

X = helper.make_tensor_value_info('input', TensorProto.FLOAT, ["batch", "seqlen", 128])
unsqueezed_masked_lm_positions = helper.make_tensor_value_info('unsqueezed_masked_lm_positions', 
                                                               TensorProto.INT64, ["batch", "dynamic_prediction_count", 1])
Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, ["batch", "dynamic_prediction_count", 128])
Y2 = helper.make_tensor_value_info('output2', TensorProto.FLOAT, ["batch", "dynamic_prediction_count", 128])
nodes = []

# case 1
divisor_np_val = np.random.uniform(0.0, 1.0, (128)).astype(np.float32).reshape((128))
divisor_initializer = numpy_helper.from_array(divisor_np_val, "divisor")
div1 = helper.make_node('Div', ['input', 'divisor'], ['div_1'], name="div_1")
nodes.append(div1)

gathernd1 = helper.make_node('GatherND', ['div_1', 'unsqueezed_masked_lm_positions'], ['output'], name="gathernd_1", batch_dims=1)
nodes.append(gathernd1)

# case 2
divisor2_np_val = np.random.uniform(0.0, 1.0, (128)).astype(np.float32).reshape((128))
divisor2_initializer = numpy_helper.from_array(divisor2_np_val, "divisor2")
div2 = helper.make_node('Div', ['divisor2', 'input'], ['div_2'], name="div_2")
nodes.append(div2)

gathernd2 = helper.make_node('GatherND', ['div_2', 'unsqueezed_masked_lm_positions'], ['output2'], name="gathernd_2", batch_dims=1)
nodes.append(gathernd2)

graph_def = helper.make_graph(nodes, 'test-model', [X, unsqueezed_masked_lm_positions], [Y, Y2], [divisor_initializer, divisor2_initializer])

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

onnx.save(model_def, "gathernd_div.onnx")
