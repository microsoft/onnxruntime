import onnx
from onnx import helper
from onnx import TensorProto, OperatorSetIdProto

# inputs/outputs
A = helper.make_tensor_value_info('A', TensorProto.FLOAT, ['unk_1', 'unk_2', 3072])
B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [3072])
R = helper.make_tensor_value_info('R', TensorProto.FLOAT, ['unk_1', 'unk_2', 3072])
C = helper.make_tensor_value_info('C', TensorProto.FLOAT, ['unk_1', 'unk_2', 3072])
mask = helper.make_tensor_value_info('mask', TensorProto.BOOL, ['unk_1', 'unk_2', 3072])

# initializers
ratio = helper.make_tensor('ratio_const', TensorProto.FLOAT, [], [0.8])
training_mode = helper.make_tensor('training_mode', TensorProto.BOOL, [], [1])

opsets = []
onnxdomain = OperatorSetIdProto()
onnxdomain.version = 12
onnxdomain.domain = "" # The empty string ("") or absence of this field implies the operator set that is defined as part of the ONNX specification.
opsets.append(onnxdomain)

kwargs={}
kwargs['opset_imports'] = opsets

# Create the model (ModelProto)
bias = helper.make_node("Add", ["A", "B"], ["add0_out"], "add0")
dropout_12 = helper.make_node("Dropout", ["add0_out", "ratio_const", "training_mode"], ["C", "mask"], "dropout0")

graph = helper.make_graph(
    [bias, dropout_12],
    "Bias_Dropout_Fusion",  #name
    [A, B],
    [C],
    [ratio, training_mode])

model = helper.make_model(graph, producer_name='onnx-example', **kwargs)
onnx.save(model, 'bias_dropout_fusion1.onnx')

# Create the model (ModelProto)
bias = helper.make_node("Add", ["B", "A"], ["add0_out"], "add0")
dropout_12 = helper.make_node("Dropout", ["add0_out", "ratio_const", "training_mode"], ["C", "mask"], "dropout0")

graph = helper.make_graph(
    [bias, dropout_12],
    "Bias_Dropout_Fusion",  #name
    [A, B],
    [C],
    [ratio, training_mode])

model = helper.make_model(graph, producer_name='onnx-example', **kwargs)
onnx.save(model, 'bias_dropout_fusion2.onnx')


# Create the model (ModelProto)
bias = helper.make_node("Add", ["A", "B"], ["add0_out"], "add0")
dropout_12 = helper.make_node("Dropout", ["add0_out", "ratio_const", "training_mode"], ["dropout_out", "mask"], "dropout0")
residual = helper.make_node("Add", ["dropout_out", "R"], ["C"], "add1")

graph = helper.make_graph(
    [bias, dropout_12, residual],
    "Bias_Dropout_Fusion",  #name
    [A, B, R],
    [C],
    [ratio, training_mode])

model = helper.make_model(graph, producer_name='onnx-example', **kwargs)
onnx.save(model, 'bias_dropout_residual_fusion1.onnx')

# Create the model (ModelProto)
bias = helper.make_node("Add", ["B", "A"], ["add0_out"], "add0")
dropout_12 = helper.make_node("Dropout", ["add0_out", "ratio_const", "training_mode"], ["dropout_out", "mask"], "dropout0")
residual = helper.make_node("Add", ["R", "dropout_out"], ["C"], "add1")

graph = helper.make_graph(
    [bias, dropout_12, residual],
    "Bias_Dropout_Fusion",  #name
    [A, B, R],
    [C],
    [ratio, training_mode])

model = helper.make_model(graph, producer_name='onnx-example', **kwargs)
onnx.save(model, 'bias_dropout_residual_fusion2.onnx')

# Create the model (ModelProto)
R_mismatch = helper.make_tensor_value_info('R', TensorProto.FLOAT, [3072])

bias = helper.make_node("Add", ["B", "A"], ["add0_out"], "add0")
dropout_12 = helper.make_node("Dropout", ["add0_out", "ratio_const", "training_mode"], ["dropout_out", "mask"], "dropout0")
residual = helper.make_node("Add", ["R", "dropout_out"], ["C"], "add1")

graph = helper.make_graph(
    [bias, dropout_12, residual],
    "Bias_Dropout_Fusion",  #name
    [A, B, R_mismatch],
    [C],
    [ratio, training_mode])

model = helper.make_model(graph, producer_name='onnx-example', **kwargs)
onnx.save(model, 'bias_dropout_residual_fusion_mismatch.onnx')