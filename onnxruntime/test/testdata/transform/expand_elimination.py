import onnx
from onnx import helper
from onnx import TensorProto, GraphProto, OperatorSetIdProto
from onnx import numpy_helper
import numpy as np

X1 = helper.make_tensor_value_info('input1', TensorProto.FLOAT, [2, 1])
X2 = helper.make_tensor_value_info('input2', TensorProto.FLOAT, ['dynamic', 4])
Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 2, 4])

shape_constant1 = numpy_helper.from_array(np.array([1, 4], dtype=np.int64), name='shape_constant1')
shape_constant2 = numpy_helper.from_array(np.array([1, 1], dtype=np.int64), name='shape_constant2')
shape_constant3 = numpy_helper.from_array(np.array([2, 1], dtype=np.int64), name='shape_constant3')
shape_constant4 = numpy_helper.from_array(np.array([1, 1, 1], dtype=np.int64), name='shape_constant4')
shape_constant5 = numpy_helper.from_array(np.array([1, 4], dtype=np.int64), name='shape_constant5')
shape_constant6 = numpy_helper.from_array(np.array([2, 1], dtype=np.int64), name='shape_constant6')

identity1 = helper.make_node('Identity', ['input1'], ['identity1'], name='identity1')
expand1 = helper.make_node('Expand', ['identity1', shape_constant1.name], ['expand1'], name='expand1')
expand2 = helper.make_node('Expand', ['identity1', shape_constant2.name], ['expand2'], name='expand2')
mul1 = helper.make_node('Mul', ['expand1', 'expand2'], ['mul1'], name='mul1')  # (2, 4)
expand3 = helper.make_node('Expand', ['mul1', shape_constant3.name], ['expand3'], name='expand3')
expand4 = helper.make_node('Expand', ['identity1', shape_constant4.name], ['expand4'], name='expand4')
mul2 = helper.make_node('Mul', ['expand3', 'expand4'], ['mul2'], name='mul2')  # (1, 2, 4)
identity2 = helper.make_node('Identity', ['input2'], ['identity2'], name='identity2')
expand5 = helper.make_node('Expand', ['identity2', shape_constant5.name], ['expand5'], name='expand5')
expand6 = helper.make_node('Expand', ['identity2', shape_constant6.name], ['expand6'], name='expand6')
mul3 = helper.make_node('Mul', ['expand5', 'expand6'], ['mul3'], name='mul3')  # (dynamic=2, 4)
mul4 = helper.make_node('Mul', ['mul2', 'mul3'], ['output'], name='mul4')

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [identity1, expand1, expand2, mul1, expand3, expand4, mul2, identity2, expand5, expand6, mul3, mul4],
    'expand_elimination_model',
    [X1, X2],
    [Y],
    [shape_constant1, shape_constant2, shape_constant3, shape_constant4, shape_constant5, shape_constant6]
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
onnx.save(model_def, 'expand_elimination.onnx')
