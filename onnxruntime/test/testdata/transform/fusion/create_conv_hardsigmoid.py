import onnx
from onnx import TensorProto, helper

graph = helper.make_graph(
    [  # nodes
        # fusable, const_min_negative should be replaced
        helper.make_node("Conv", ["X", "W"], ["conv0_out"], "Conv0"),
        helper.make_node("HardSigmoid", ["conv0_out"], ["hardsigmoid0_out"], "HardSigmoid0"),
    ],
    "ConvClipFusion",  # name
    [  # inputs
        helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 10, 10]),
        helper.make_tensor_value_info("W", TensorProto.FLOAT, [1, 1, 3, 3]),
    ],
    [  # outputs
        helper.make_tensor_value_info("hardsigmoid0_out", TensorProto.FLOAT, None),
    ],
)

model = helper.make_model(graph)
onnx.save(model, r"conv_hardsigmoid.onnx")
