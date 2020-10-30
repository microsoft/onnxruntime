import onnx
from onnx import helper
from onnx import TensorProto, OperatorSetIdProto

# inputs/outputs
A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [4])
B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [4])
C = helper.make_tensor_value_info('C', TensorProto.FLOAT, [4])
accu_out = helper.make_tensor_value_info('accu_out', TensorProto.FLOAT, [4])

# initializers
buffer = helper.make_tensor('buffer', TensorProto.FLOAT, [4], [0, 0, 0, 0])

opsets = []
onnxdomain = OperatorSetIdProto()
onnxdomain.version = 12
onnxdomain.domain = "" # The empty string ("") or absence of this field implies the operator set that is defined as part of the ONNX specification.
opsets.append(onnxdomain)

msdomain = OperatorSetIdProto()
msdomain.version = 1
msdomain.domain = "com.microsoft"
opsets.append(msdomain)

kwargs={}
kwargs['opset_imports'] = opsets

# Create the model (ModelProto)
sum_node = helper.make_node("Sum", ["A", "B", "C"], ["sum_out"], "sum")
accumulator_node = helper.make_node("InPlaceAccumulator", ["buffer", "sum_out"], ["accu_out"], "accumulator")
accumulator_node.domain = "com.microsoft"

graph = helper.make_graph(
    [sum_node, accumulator_node],
    "Sum_Accumulator_Transformation",  #name
    [A, B, C],
    [accu_out],
    [buffer])

model = helper.make_model(graph, producer_name='onnx-example', **kwargs)
onnx.save(model, 'sum_accumulator.onnx')