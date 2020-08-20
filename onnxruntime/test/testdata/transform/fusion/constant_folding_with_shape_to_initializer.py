import onnx
from onnx import helper
from onnx import TensorProto, GraphProto, OperatorSetIdProto
from onnx import numpy_helper
import numpy as np

X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 4, 8])
Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 4, 16])

matmul_weight_vals = (0.01 * np.arange(2 * 4 * 4, dtype=np.float32)).reshape((2, 4, 4))
matmul_weight_initializer = numpy_helper.from_array(matmul_weight_vals, 'matmul_weight')
gather_constant_zero = numpy_helper.from_array(np.int64(0), name='gather_constant_zero')
gather_constant_one = numpy_helper.from_array(np.int64(1), name='gather_constant_one')
div_constant_two = numpy_helper.from_array(np.int64(2), name='div_constant_two')
unsqueeze_constant_16 = numpy_helper.from_array(np.int64(16), name='unsqueeze_constant_16')

shape1 = helper.make_node('Shape', ['input'], ['shape1'], name='shape1')
constant_of_shape = helper.make_node('ConstantOfShape', ['shape1'], ['constant_of_shape'], name='constant_of_shape')
transpose = helper.make_node('Transpose', ['constant_of_shape'], ['transpose'], name='transpose', perm=[0,2,1])
matmul1 = helper.make_node('MatMul', ['transpose', matmul_weight_initializer.name], ['matmul1'], name='matmul1')
matmul2 = helper.make_node('MatMul', ['matmul1', 'input'], ['matmul2'], name='matmul2')
shape2 = helper.make_node('Shape', ['matmul2'], ['shape2'], name='shape2')
gather1 = helper.make_node('Gather', ['shape2', gather_constant_zero.name], ['gather1'], name='gather1', axis=0)
gather2 = helper.make_node('Gather', ['shape2', gather_constant_one.name], ['gather2'], name='gather2', axis=0)
div = helper.make_node('Div', ['gather2', div_constant_two.name], ['div'], name='div')
unsqueeze1 = helper.make_node('Unsqueeze', ['gather1'], ['unsqueeze1'], name='unsqueeze1', axes=[0])
unsqueeze2 = helper.make_node('Unsqueeze', ['div'], ['unsqueeze2'], name='unsqueeze2', axes=[0])
unsqueeze3 = helper.make_node('Unsqueeze', [unsqueeze_constant_16.name], ['unsqueeze3'], name='unsqueeze3', axes=[0])
concat = helper.make_node('Concat', ['unsqueeze1', 'unsqueeze2', 'unsqueeze3'], ['concat'], name='concat', axis=0)
reshape = helper.make_node('Reshape', ['matmul2', 'concat'], ['output'], name='reshape')

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [shape1, constant_of_shape, transpose, matmul1, matmul2, shape2, gather1, gather2, div, unsqueeze1, unsqueeze2, unsqueeze3, concat, reshape],
    'constant_folding_with_shape_to_initializer_model',
    [X],
    [Y],
    [matmul_weight_initializer, gather_constant_zero, gather_constant_one, div_constant_two, unsqueeze_constant_16]
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
onnx.save(model_def, 'constant_folding_with_shape_to_initializer.onnx')



X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1])
Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1])

squeeze = helper.make_node('Squeeze', ['input'], ['squeeze'], name='squeeze', axes=[0])
shape = helper.make_node('Shape', ['squeeze'], ['shape'], name='shape')
constant_of_shape = helper.make_node('ConstantOfShape', ['shape'], ['constant_of_shape'], name='constant_of_shape')
add = helper.make_node('Add', ['squeeze', 'constant_of_shape'], ['add'], name='add')
unsqueeze = helper.make_node('Unsqueeze', ['add'], ['output'], name='unsqueeze', axes=[0])

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [squeeze, shape, constant_of_shape, add, unsqueeze],
    'constant_folding_with_scalar_shape_to_initializer_model',
    [X],
    [Y]
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-example', **kwargs)
onnx.save(model_def, 'constant_folding_with_scalar_shape_to_initializer.onnx')
