import onnx
from onnx import OperatorSetIdProto, TensorProto, helper

input_shape = [2, 2]
output_shape = [1]
input = helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)
output = helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)

cast1 = helper.make_node("Cast", ["input"], ["cast1"], name="cast1", to=1)
shape1 = helper.make_node("Shape", ["cast1"], ["shape1"], name="shape1")
add = helper.make_node("Add", ["cast1", "cast1"], ["add"], name="add")
cast2 = helper.make_node("Cast", ["add"], ["cast2"], name="cast2", to=1)
shape2 = helper.make_node("Shape", ["cast2"], ["shape2"], name="shape2")

# reshape = helper.make_node("Reshape", ["shape2", "shape2"], ["reshape"], name="reshape")
transpose1 = helper.make_node("Transpose", ["input"], ["transpose1"], name="transpose1")
shape3 = helper.make_node("Shape", ["transpose1"], ["shape3"], name="shape3")

transpose2 = helper.make_node("Transpose", ["shape3"], ["transpose2"], name="transpose2")
shape4 = helper.make_node("Shape", ["transpose2"], ["shape4"], name="shape4")

cast3 = helper.make_node("Cast", ["shape4"], ["output"], name="cast3", to=1)

graph_def = helper.make_graph(
    [cast1, shape1, add, cast2, shape2, transpose1, shape3, transpose2, shape4, cast3],
    "pre_shape_node_elimination.onnx",
    [input],
    [output],
)

opsets = []
onnxdomain = OperatorSetIdProto()
onnxdomain.version = 15
onnxdomain.domain = ""
opsets.append(onnxdomain)

kwargs = {}
kwargs["opset_imports"] = opsets

model_def = helper.make_model(graph_def, producer_name="onnx-example", **kwargs)
onnx.save(model_def, "pre_shape_node_elimination.onnx")
