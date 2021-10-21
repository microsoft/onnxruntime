import onnx
from onnx import helper
from onnx import TensorProto, GraphProto, OperatorSetIdProto
from onnx import numpy_helper
import numpy as np

initial = helper.make_tensor_value_info('initial', TensorProto.FLOAT, [2])
x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 2])
y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 2])
z = helper.make_tensor_value_info('z', TensorProto.FLOAT, [3, 2])

sum_in = helper.make_tensor_value_info('sum_in', TensorProto.FLOAT, [2])
next = helper.make_tensor_value_info('next', TensorProto.FLOAT, [2])
sum_out = helper.make_tensor_value_info('sum_out', TensorProto.FLOAT, [2])
scan_out = helper.make_tensor_value_info('scan_out', TensorProto.FLOAT, [2])

add_node = helper.make_node(
    'Add',
    inputs=['sum_in', 'next'],
    outputs=['sum_out']
)
id_node = helper.make_node(
    'Identity',
    inputs=['sum_out'],
    outputs=['scan_out']
)
scan_body = helper.make_graph(
    [add_node, id_node],
    'scan_body',
    [sum_in, next],
    [sum_out, scan_out]
)
# create scan op node
scan_node = helper.make_node(
    'Scan',
    inputs=['initial', 'x'],
    outputs=['y', 'z'],
    num_scan_inputs=1,
    body=scan_body
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [scan_node],
    'test_scan9_sum',
    [initial, x],
    [y, z]
)

opsets = []
onnxdomain = OperatorSetIdProto()
onnxdomain.version = 9
onnxdomain.domain = "" # The empty string ("") or absence of this field implies the operator set that is defined as part of the ONNX specification.
opsets.append(onnxdomain)

kwargs={}
kwargs['opset_imports'] = opsets

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-example', **kwargs)
onnx.save(model_def, 'scan9_sum.onnx')
