# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import onnx
from onnx import helper
from onnx import TensorProto, GraphProto, OperatorSetIdProto
from onnx import numpy_helper
import numpy as np

X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [4, 4])
Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1])
Y.type.tensor_type.shape.Clear()

nonzero = helper.make_node('NonZero', ['input'], ['nonzero'], name='nonzero')
transpose = helper.make_node('Transpose', ['nonzero'], ['transpose'], name='transpose', perm=[1,0])
gathernd = helper.make_node('GatherND', ['input', 'transpose'], ['output'], name='gathernd')

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [nonzero, transpose, gathernd],
    'nonzero_shape_setter_model',
    [X],
    [Y]
)

opsets = []
onnxdomain = OperatorSetIdProto()
onnxdomain.version = 12
onnxdomain.domain = "" # Empty string implies the operator set that is defined as part of the ONNX specification.
opsets.append(onnxdomain)

msdomain = OperatorSetIdProto()
msdomain.version = 1
msdomain.domain = 'com.microsoft'

opsets.append(msdomain)
kwargs={}
kwargs['opset_imports'] = opsets

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-example', **kwargs)
onnx.save(model_def, 'nonzero_shape_setter.onnx')
