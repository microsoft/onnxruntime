import onnx
from onnx import TensorProto, helper, shape_inference

# create output with rank but unnamed symbolic dim
output = helper.make_tensor_value_info("C", TensorProto.FLOAT, [1])
output.type.tensor_type.shape.Clear()
dim = output.type.tensor_type.shape.dim.add()
print(dim)

graph_def = helper.make_graph(
    nodes=[
        helper.make_node(op_type="Reshape", inputs=["A", "B"], outputs=["C"], name="reshape"),
    ],
    name="test-model",
    inputs=[
        # create inputs with symbolic dims
        helper.make_tensor_value_info("A", TensorProto.FLOAT, ["n", 2]),
        helper.make_tensor_value_info("B", TensorProto.INT64, ["m"]),
    ],
    outputs=[output],
    initializer=[],
)

model = helper.make_model(graph_def, opset_imports=[helper.make_operatorsetid("", 11)])
onnx.checker.check_model(model)

inferred_model = shape_inference.infer_shapes(model)
onnx.checker.check_model(inferred_model)

onnx.save_model(model, "capi_symbolic_dims.onnx")

import onnxruntime as rt

sess = rt.InferenceSession("capi_symbolic_dims.onnx")
print([i.shape for i in sess.get_inputs()])
print([i.shape for i in sess.get_outputs()])
