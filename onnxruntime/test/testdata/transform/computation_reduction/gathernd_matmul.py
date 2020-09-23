import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto, OperatorSetIdProto
from onnx import numpy_helper
import numpy as np

X = helper.make_tensor_value_info('input', TensorProto.FLOAT, ["batch", "seqlen", 128])
unsqueezed_masked_lm_positions = helper.make_tensor_value_info('unsqueezed_masked_lm_positions', 
                                                               TensorProto.INT64, ["batch", "dynamic_prediction_count", 1])
Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, ["batch", "dynamic_prediction_count", 128])

matmul1_np_vals = np.random.uniform(0.0, 1.0, (128, 128)).astype(np.float32).reshape((128, 128))
matmul1_initializer = numpy_helper.from_array(matmul1_np_vals, "matmul1_initializer")

nodes=[]
matmul1 = helper.make_node('MatMul', ['input', matmul1_initializer.name], ['matmul1'], name="matmul_1")
nodes.append(matmul1)

gathernd1 = helper.make_node('GatherND', ['matmul1', 'unsqueezed_masked_lm_positions'], ['output'], name="gathernd_1", batch_dims=1)
nodes.append(gathernd1)

initializers=[matmul1_initializer]

graph_def = helper.make_graph(nodes, 'test-model', [X, unsqueezed_masked_lm_positions], [Y], initializers)

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

onnx.save(model_def, "gathernd_matmul.onnx")
