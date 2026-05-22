"""Generate a simple ONNX model with two dynamic-dimension inputs for testing --data_shape.

The model computes: output = (A + B), where A and B have dynamic batch and sequence dims.
  A: float32[batch, seq, 4]
  B: float32[batch, seq, 4]
  output: float32[batch, seq, 4]

Usage:
  python dynamic_shape_add.py
  onnxruntime_perf_test -m dynamic_shape_add.onnx -I -t 5 \
      --data_shape "A:[1,16,4][2,32,4] B:[1,16,4][2,32,4]"
"""

from onnx import TensorProto, checker, helper, save

graph_proto = helper.make_graph(
    nodes=[
        helper.make_node(
            "Add",
            inputs=["A", "B"],
            outputs=["output"],
            name="add_0",
        ),
    ],
    name="dynamic_shape_add",
    inputs=[
        helper.make_tensor_value_info("A", TensorProto.FLOAT, ["batch", "seq", 4]),
        helper.make_tensor_value_info("B", TensorProto.FLOAT, ["batch", "seq", 4]),
    ],
    outputs=[
        helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", "seq", 4]),
    ],
)

model = helper.make_model(graph_proto, opset_imports=[helper.make_operatorsetid("", 18)])
checker.check_model(model, True)
save(model, "dynamic_shape_add.onnx")
print("Saved dynamic_shape_add.onnx")
print(f"  Inputs: {[(i.name, [d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim]) for i in model.graph.input]}")
print(f"  Outputs: {[(o.name, [d.dim_param or d.dim_value for d in o.type.tensor_type.shape.dim]) for o in model.graph.output]}")
