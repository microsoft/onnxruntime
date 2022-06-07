import onnx
from onnx import TensorProto, helper

graph = helper.make_graph(
    [  # nodes
        # Add node before Gelu
        helper.make_node("Gelu", ["A"], ["C"], "Gelu_1", domain="com.microsoft"),
    ],
    "Gelu_NoBias",  # name
    [  # inputs
        helper.make_tensor_value_info("A", TensorProto.FLOAT, ["batch", "seq_len", 3072]),
    ],
    [  # outputs
        helper.make_tensor_value_info("C", TensorProto.FLOAT, ["batch", "seq_len", 3072]),
    ],
    [],  # initializers
)

model = helper.make_model(graph)
onnx.save(model, r"gelu.onnx")

graph = helper.make_graph(
    [  # nodes
        # Add node before Gelu
        helper.make_node("BiasGelu", ["A", "B"], ["C"], "AddGeluFusion_1", domain="com.microsoft"),
    ],
    "Gelu_AddBias",  # name
    [  # inputs
        helper.make_tensor_value_info("A", TensorProto.FLOAT, ["batch", "seq_len", 3072]),
        helper.make_tensor_value_info("B", TensorProto.FLOAT, [3072]),
    ],
    [  # outputs
        helper.make_tensor_value_info("C", TensorProto.FLOAT, ["batch", "seq_len", 3072]),
    ],
    [],  # initializers
)

model = helper.make_model(graph)
onnx.save(model, r"gelu_add_bias.onnx")

graph = helper.make_graph(
    [  # nodes
        # Add node before Gelu
        helper.make_node("MatMul", ["A", "B"], ["C"], "MatMul_1"),
        helper.make_node("BiasGelu", ["C", "D"], ["E"], "AddGeluFusion_1", domain="com.microsoft"),
    ],
    "MatMul_AddGeluFusion",  # name
    [  # inputs
        helper.make_tensor_value_info("A", TensorProto.FLOAT, ["batch", "seq_len", "x"]),
        helper.make_tensor_value_info("B", TensorProto.FLOAT, [128, 3072]),
        helper.make_tensor_value_info("D", TensorProto.FLOAT, [3072]),
    ],
    [  # outputs
        helper.make_tensor_value_info("E", TensorProto.FLOAT, ["batch", "seq_len", 3072]),
    ],
    [],  # initializers
)

model = helper.make_model(graph)
onnx.save(model, r"gelu_add_matmul.onnx")
