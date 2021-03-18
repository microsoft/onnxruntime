import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto, OperatorSetIdProto
from onnx import numpy_helper
import numpy as np


X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [2, 2])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [2, 2])

add = helper.make_node(
    'Add', # node name
    ['X', 'X'], # inputs
    ['add'], # outputs
    name="add"
)

pyop = helper.make_node(
    'PyOp', # node name
    ['add'], # inputs
    ['pyop1'], # outputs
    name="pyop1",
    doc_string="",
    domain="pyopfloat",
    A='123',
    B='456',
    C='789',
    class_name="CustomFnWrapperModule",
    compute="compute",
    input_types=[1],
    module="mymodule",
    output_types=[1],
)

add2 = helper.make_node(
    'Add', # node name
    ['pyop1', 'pyop1'], # inputs
    ['Y'], # outputs
    name="Y"
)



graph_def = helper.make_graph(
    [add, pyop, add2],
    'test-model',
    [X],
    [Y],
)

opsets = []
onnxdomain = OperatorSetIdProto()
onnxdomain.version = 12
onnxdomain.domain = "" # The empty string ("") or absence of this field implies the operator set that is defined as part of the ONNX specification.
opsets.append(onnxdomain)

msdomain = OperatorSetIdProto()
msdomain.version = 1
msdomain.domain = "pyopfloat"

opsets.append(msdomain)
kwargs={}
kwargs["opset_imports"] = opsets

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-example', **kwargs)

onnx.save(model_def, 'sample.onnx')