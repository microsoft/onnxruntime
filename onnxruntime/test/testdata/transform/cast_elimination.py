import onnx
from onnx import helper
from onnx import TensorProto, GraphProto, OperatorSetIdProto
from onnx import numpy_helper
import numpy as np

X1 = helper.make_tensor_value_info('x1', TensorProto.INT64, [4, 4])
X2 = helper.make_tensor_value_info('x2', TensorProto.INT64, [4, 1])
X3 = helper.make_tensor_value_info('x3', TensorProto.INT64, [4, 1])
Y = helper.make_tensor_value_info('output', TensorProto.INT64, [4, 4])

less1 = helper.make_node('Less', ['x1', 'x2'], ['less1'], name='less1')
less2 = helper.make_node('Less', ['x1', 'x3'], ['less2'], name='less2')
cast1 = helper.make_node('Cast', ['less1'], ['cast1'], name='cast1', to=9)
and_node = helper.make_node('And', ['cast1', 'less2'], ['and_node'], name='and_node')
cast2 = helper.make_node('Cast', ['and_node'], ['cast2'], name='cast2', to=9)
cast3 = helper.make_node('Cast', ['cast2'], ['cast3'], name='cast3', to=1)
cast4 = helper.make_node('Cast', ['x1'], ['cast4'], name='cast4', to=7)
cast5 = helper.make_node('Cast', ['cast4'], ['cast5'], name='cast5', to=1)
matmul = helper.make_node('MatMul', ['cast3', 'cast5'], ['matmul'], name='matmul')
cast6 = helper.make_node('Cast', ['matmul'], ['cast6'], name='cast6', to=7)
cast7 = helper.make_node('Cast', ['cast6'], ['output'], name='cast7', to=7)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [less1, less2, cast1, and_node, cast2, cast3, cast4, cast5, matmul, cast6, cast7],
    'cast_elimination_model',
    [X1, X2, X3],
    [Y]
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
onnx.save(model_def, 'cast_elimination.onnx')
