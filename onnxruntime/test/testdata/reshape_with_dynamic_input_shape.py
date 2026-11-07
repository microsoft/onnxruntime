from pathlib import Path

import onnx
from onnx import TensorProto, helper

# This model contains a Reshape where:
# - X has shape [M, 512] and `M` is a dynamic dimension.
# - shape is a constant initializer with value [-1, 2048].
# CoreML MIL supports -1 in the shape and can infer the dimension at runtime.

M = "M"
K = 512
N = 2048

graph = helper.make_graph(
    [  # nodes
        helper.make_node("Reshape", ["X", "shape"], ["Y"], "Reshape"),
    ],
    "ReshapeWithDynamicInputShape",  # name
    [  # inputs
        helper.make_tensor_value_info("X", TensorProto.FLOAT, [M, K]),
    ],
    [  # outputs
        helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, N]),
    ],
    [  # initializers
        helper.make_tensor("shape", TensorProto.INT64, [2], [-1, N]),
    ],
)

opset_imports = [helper.make_operatorsetid("", 19)]
model = helper.make_model(graph, opset_imports=opset_imports)
onnx.save(model, str(Path(__file__).parent / "reshape_with_dynamic_input_shape.onnx"))
