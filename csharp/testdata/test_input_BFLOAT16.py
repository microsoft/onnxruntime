# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import onnx
from onnx import helper
from onnx.helper import make_opsetid
from onnx import AttributeProto, TensorProto, GraphProto

input_info = helper.make_tensor_value_info('input', TensorProto.BFLOAT16, [1, 5])
output_info = helper.make_tensor_value_info('output', TensorProto.BFLOAT16, [1, 5])

# Create a node (NodeProto) - This is based on Pad-11
node_def = helper.make_node(
    'Identity', # node name
    ['input'], # inputs
    ['output'] # outputs
)

graph_def = helper.make_graph(nodes=[node_def], name='test_types_BLOAT16',
                             inputs=[input_info], outputs=[output_info])

model_def = helper.make_model(graph_def, producer_name='AIInfra',
     opset_imports=[make_opsetid('', 13)])

final_model = onnx.utils.polish_model(model_def)
onnx.save(final_model, 'test_types_BFLOAT16.onnx')

