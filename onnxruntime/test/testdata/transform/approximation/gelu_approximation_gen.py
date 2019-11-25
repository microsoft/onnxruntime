import onnx
from onnx import helper
from onnx import TensorProto

graph = helper.make_graph(
    [ # nodes
        # Add node before Gelu
        helper.make_node("Gelu", ["A"], ["C"], "Gelu_1", domain="com.microsoft"),
    ],
    "Gelu_NoBias",  #name
    [  # inputs
        helper.make_tensor_value_info('A', TensorProto.FLOAT, ['batch', 'seq_len', 3072]),
    ],
    [  # outputs
        helper.make_tensor_value_info('C', TensorProto.FLOAT, ['batch', 'seq_len', 3072]),
    ],
    [  # initializers
    ]
)

model = helper.make_model(graph)
onnx.save(model, r'gelu.onnx')

graph = helper.make_graph(
    [ # nodes
        # Add node before Gelu
        helper.make_node("AddGeluFusion", ["A", "B"], ["C"], "AddGeluFusion_1", domain="com.microsoft"),
    ],
    "Gelu_AddBias",  #name
    [  # inputs
        helper.make_tensor_value_info('A', TensorProto.FLOAT, ['batch', 'seq_len', 3072]),
        helper.make_tensor_value_info('B', TensorProto.FLOAT, [3072]),
    ],
    [  # outputs
        helper.make_tensor_value_info('C', TensorProto.FLOAT, ['batch', 'seq_len', 3072]),
    ],
    [  # initializers
    ]
)

model = helper.make_model(graph)
onnx.save(model, r'gelu_add_bias.onnx')

graph = helper.make_graph(
    [ # nodes
        # Add node before Gelu
        helper.make_node("AddGeluFusion", ["A", "B"], ["C"], "AddGeluFusion_1", domain="com.microsoft"),
    ],
    "Gelu_Add_ShapeNotMatch",  #name
    [  # inputs
        helper.make_tensor_value_info('A', TensorProto.FLOAT, ['batch', 'seq_len', 3072]),
        helper.make_tensor_value_info('B', TensorProto.FLOAT, ['batch', 'seq_len', 3072]), # Bias shape not matched for FastGelu
    ],
    [  # outputs
        helper.make_tensor_value_info('C', TensorProto.FLOAT, ['batch', 'seq_len', 3072]),
    ],
    [  # initializers
    ]
)

model = helper.make_model(graph)
onnx.save(model, r'gelu_add_shape_not_match.onnx')


