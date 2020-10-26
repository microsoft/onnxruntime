import onnx
from onnx import helper
from onnx import TensorProto, OperatorSetIdProto

# inputs/outputs
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [2, 1])
O1 = helper.make_tensor_value_info('O1', TensorProto.FLOAT, [2, 1])
O2 = helper.make_tensor_value_info('O2', TensorProto.FLOAT, [2, 1])

# initializers
zeroratio = helper.make_tensor('ratio_zero', TensorProto.FLOAT, [], [0.0])
nonzeroratio = helper.make_tensor('ratio_nonzero', TensorProto.FLOAT, [], [0.1])
training_mode = helper.make_tensor('training_mode', TensorProto.BOOL, [], [1])

opsets = []
onnxdomain = OperatorSetIdProto()
onnxdomain.version = 12
onnxdomain.domain = "" # The empty string ("") or absence of this field implies the operator set that is defined as part of the ONNX specification.
opsets.append(onnxdomain)

kwargs={}
kwargs['opset_imports'] = opsets

# Create the model (ModelProto)
I1 = helper.make_node('Identity', ['X'], ['I1_out'], name='I1')
D1 = helper.make_node("Dropout", ["I1_out", "ratio_zero", "training_mode"], ["D1_out"], "D1")
I2 = helper.make_node('Identity', ['D1_out'], ['O1'], name='I2')

I3 = helper.make_node('Identity', ['X'], ['I3_out'], name='I3')
D2 = helper.make_node("Dropout", ["I3_out", "ratio_nonzero", "training_mode"], ["D2_out"], "D2")
I4 = helper.make_node('Identity', ['D2_out'], ['O2'], name='I4')

graph = helper.make_graph(
    [I1, D1, I2, I3, D2, I4],
    "Dropout_Elimination",  #name
    [X],
    [O1, O2],
    [zeroratio, nonzeroratio, training_mode])

model = helper.make_model(graph, producer_name='onnx-example', **kwargs)
onnx.save(model, 'dropout_ratio.onnx')