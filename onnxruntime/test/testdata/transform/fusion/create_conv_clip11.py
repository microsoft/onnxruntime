import onnx
from onnx import helper
from onnx import TensorProto

graph = helper.make_graph(
    [  # nodes
        # fusable, min/max from constant inputs.
        helper.make_node("Conv", ["X0", "W"], ["conv0_out"], "Conv0"),
        helper.make_node("Clip", ["conv0_out", "const_min", "const_max"], ["clip0_out"], "Clip0"),

        # mutable input. no fusion.
        helper.make_node("Conv", ["X1", "W"], ["conv1_out"], "Conv1"),
        helper.make_node("Clip", ["conv1_out", "mutable_min", "const_max"], ["clip1_out"], "Clip1"),

        # fusable. default min/max.
        helper.make_node("Conv", ["X2", "W"], ["conv2_out"], "Conv2"),
        helper.make_node("Clip", ["conv2_out"], ["clip2_out"], "Clip2"),
    ],
    "ConvClipFusion",  # name
    [  # inputs
        # each Conv has a distinct X input so that the common subexpression elimination does not combine them
        helper.make_tensor_value_info('X0', TensorProto.FLOAT, [1, 1, 7]),
        helper.make_tensor_value_info('X1', TensorProto.FLOAT, [1, 1, 7]),
        helper.make_tensor_value_info('X2', TensorProto.FLOAT, [1, 1, 7]),
        helper.make_tensor_value_info('W', TensorProto.FLOAT, [1, 1, 1]),
        helper.make_tensor_value_info('mutable_min', TensorProto.FLOAT, [1]),
    ],
    [  # outputs
        helper.make_tensor_value_info('clip0_out', TensorProto.FLOAT, None),
        helper.make_tensor_value_info('clip1_out', TensorProto.FLOAT, None),
        helper.make_tensor_value_info('clip2_out', TensorProto.FLOAT, None),
    ],
    [  # initializers
        helper.make_tensor('const_min', TensorProto.FLOAT, [1], [-1.0]),
        helper.make_tensor('const_max', TensorProto.FLOAT, [1], [10.0])
    ])

model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 11)])
onnx.save(model, r'conv_clip11.onnx')
