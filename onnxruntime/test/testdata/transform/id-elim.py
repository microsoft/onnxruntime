import onnx
from onnx import helper
from onnx import TensorProto, GraphProto, OperatorSetIdProto
from onnx import numpy_helper
import numpy as np

X1 = helper.make_tensor_value_info('x1', TensorProto.INT64, [4, 4])
X2 = helper.make_tensor_value_info('x2', TensorProto.INT64, [4, 4])
Y1 = helper.make_tensor_value_info('output1', TensorProto.INT64, [4, 4])
Y2 = helper.make_tensor_value_info('output2', TensorProto.INT64, [4, 4])

add1 = helper.make_node('Add', ['x1', 'x2'], ['add1'], name='add1')
add2 = helper.make_node('Add', ['x1', 'x2'], ['add2'], name='add2')
id1 = helper.make_node('Identity', ['add1'], ['output1'], name='id1')
id2 = helper.make_node('Identity', ['add2'], ['output2'], name='id2')

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [add1, add2, id1, id2],
    'cast_elimination_model',
    [X1, X2],
    [Y1, Y2]
)

opsets = []
onnxdomain = OperatorSetIdProto()
onnxdomain.version = 12
onnxdomain.domain = "" # The empty string ("") or absence of this field implies the operator set that is defined as part of the ONNX specification.
opsets.append(onnxdomain)

msdomain = OperatorSetIdProto()
msdomain.version = 1
msdomain.domain = 'com.microsoft'

opsets.append(msdomain)
kwargs={}
kwargs['opset_imports'] = opsets

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-example', **kwargs)
onnx.save(model_def, 'id-elim.onnx')
