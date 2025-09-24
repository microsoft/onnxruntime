"""
Run this script to recreate the original onnx model.
Example usage:
python input_propagated_to_output.py input_propagated_to_output.onnx
"""

import sys

import numpy as np
import onnx


def order_repeated_field(repeated_proto, key_name, order):
    order = list(order)
    repeated_proto.sort(key=lambda x: order.index(getattr(x, key_name)))


def make_node(op_type, inputs, outputs, name=None, doc_string=None, domain=None, **kwargs):
    node = onnx.helper.make_node(op_type, inputs, outputs, name, doc_string, domain, **kwargs)
    if doc_string == "":
        node.doc_string = ""
    order_repeated_field(node.attribute, "name", kwargs.keys())
    return node


def make_graph(*args, doc_string=None, **kwargs):
    graph = onnx.helper.make_graph(*args, doc_string=doc_string, **kwargs)
    if doc_string == "":
        graph.doc_string = ""
    return graph


W1 = np.array(
    [
        [[[0.3258337378501892]], [[0.1461111307144165]], [[-0.4239698648452759]]],
        [[[0.14769716560840607]], [[0.20565544068813324]], [[-0.5241780877113342]]],
        [[[0.07987150549888611]], [[-0.17475983500480652]], [[0.005230882670730352]]],
    ],
    dtype=np.float32,
)

B1 = np.array(
    [-0.3170531392097473, -0.2701416313648224, -0.14249320328235626],
    dtype=np.float32,
)

W3 = np.array(
    [
        [[[0.14025720953941345]], [[0.1433156430721283]], [[-0.1403128057718277]]],
        [[[-0.07530076801776886]], [[0.11853527277708054]], [[-0.19437682628631592]]],
        [[[0.5786639451980591]], [[-0.28565627336502075]], [[0.9048876166343689]]],
    ],
    dtype=np.float32,
)

B3 = np.array(
    [-0.13307525217533112, 0.5522456169128418, 0.6449958086013794],
    dtype=np.float32,
)

W5 = np.array(
    [
        [[[-0.08959630876779556]], [[0.07607565075159073]], [[0.24446037411689758]]],
        [[[-0.06293385475873947]], [[-0.41520264744758606]], [[-0.83400559425354]]],
        [[[-0.031176576390862465]], [[-0.04187283664941788]], [[-0.439873069524765]]],
    ],
    dtype=np.float32,
)

B5 = np.array(
    [0.5949633717536926, -0.40198755264282227, -0.20182392001152039],
    dtype=np.float32,
)

model = onnx.helper.make_model(
    opset_imports=[onnx.helper.make_operatorsetid("", 14)],
    ir_version=7,
    graph=make_graph(
        name="input_propagated_to_output",
        inputs=[
            onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, shape=[1, 3, 1, 3]),
        ],
        outputs=[
            onnx.helper.make_tensor_value_info("X6", onnx.TensorProto.FLOAT, shape=[1, 3, 1, 3]),
            onnx.helper.make_tensor_value_info("X1", onnx.TensorProto.FLOAT, shape=[1, 3, 1, 3]),
            onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, shape=[1, 3, 1, 3]),
            onnx.helper.make_tensor_value_info("X2", onnx.TensorProto.FLOAT, shape=[1, 3, 1, 3]),
            onnx.helper.make_tensor_value_info("X4", onnx.TensorProto.FLOAT, shape=[1, 3, 1, 3]),
            onnx.helper.make_tensor_value_info("X3", onnx.TensorProto.FLOAT, shape=[1, 3, 1, 3]),
            onnx.helper.make_tensor_value_info("X5", onnx.TensorProto.FLOAT, shape=[1, 3, 1, 3]),
        ],
        initializer=[
            onnx.numpy_helper.from_array(W1, name="W1"),
            onnx.numpy_helper.from_array(W3, name="W3"),
            onnx.numpy_helper.from_array(W5, name="W5"),
            onnx.numpy_helper.from_array(B1, name="B1"),
            onnx.numpy_helper.from_array(B3, name="B3"),
            onnx.numpy_helper.from_array(B5, name="B5"),
        ],
        nodes=[
            make_node("Relu", inputs=["input"], outputs=["X1"], name="Relu1"),
            make_node("Conv", inputs=["X1", "W1", "B1"], outputs=["X2"], name="Conv1"),
            make_node("Relu", inputs=["X2"], outputs=["X3"], name="Relu2"),
            make_node("Conv", inputs=["X3", "W3", "B3"], outputs=["X4"], name="Conv2"),
            make_node("Conv", inputs=["X1", "W5", "B5"], outputs=["X5"], name="Conv3"),
            make_node("Add", inputs=["X4", "X5"], outputs=["X6"], name="Add"),
        ],
    ),
)

if __name__ == "__main__" and len(sys.argv) == 2:
    _, out_path = sys.argv
    onnx.save(model, out_path)
