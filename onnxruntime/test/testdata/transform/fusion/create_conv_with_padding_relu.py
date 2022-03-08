import onnx
from onnx import helper
from onnx import TensorProto

# copied and adapted from here:
# https://github.com/onnx/onnx/blob/c940fa3fea84948e46603cab2f86467291443beb/docs/Operators.md?plain=1#L3494-L3502

graph = helper.make_graph(
    [  # nodes
        # Convolution with padding
        helper.make_node('Conv', ['x', 'W'], ['y'],
                         kernel_shape=[3, 3],
                         pads=[1, 1, 1, 1]),
        helper.make_node('Relu', ['y'], ['relu_out']),
    ],
    "ConvWithPaddingReluFusion",
    [ # inputs
        helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 5, 5]),
        helper.make_tensor_value_info('W', TensorProto.FLOAT, [1, 1, 3, 3]),
    ],
    [ # outputs
        helper.make_tensor_value_info('relu_out', TensorProto.FLOAT, [1, 1, 5, 5]),
    ]
)

model = helper.make_model(graph)
onnx.save(model, r'conv_with_padding_relu.onnx')
