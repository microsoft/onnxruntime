import onnx
from onnx import OperatorSetIdProto, TensorProto, helper

onnxdomain = OperatorSetIdProto()
onnxdomain.version = 12
# The empty string ("") or absence of this field implies the operator set that is defined as part of the ONNX specification.
onnxdomain.domain = ""
msdomain = OperatorSetIdProto()
msdomain.version = 1
msdomain.domain = "com.microsoft"
opsets = [onnxdomain, msdomain]

graph = helper.make_graph(
    [  # nodes
        # Add node before Gelu
        # helper.make_node("Split", ["inp", "split", "axis"], ["out1", "out2"], "split"),
        # helper.make_node("Split", ["inp", "split"], ["out1", "out2"], "split"),
        helper.make_node("Split", ["inp", "axis"], ["out1", "out2"], "split"),
        # Gelu subgraph
        helper.make_node("QuickGelu", ["out2", "alpha"], ["gelu_out"], "QuickGelu", "", msdomain.domain),
        helper.make_node("Mul", ["out1", "gelu_out"], ["out"], "mul"),
    ],
    "Split_QuickGelu_Fusion",  # name
    [  # inputs
        helper.make_tensor_value_info("inp", TensorProto.FLOAT, [76, 54, 1368]),
        # helper.make_tensor_value_info("split", TensorProto.INT64, [2]),
    ],
    [  # outputs
        helper.make_tensor_value_info("out", TensorProto.FLOAT, [76, 54, 684]),
    ],
    [  # initializers
        helper.make_tensor("axis", TensorProto.INT64, [], [-1]),
        helper.make_tensor("alpha", TensorProto.FLOAT, [], [1]),
    ],
)

model = helper.make_model(graph)
onnx.save(model, r"split_quickgelu_fusion.onnx")
