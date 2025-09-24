"""
Run this script to recreate the original onnx model.
Example usage:
python input_propagated_to_output.py input_propagated_to_output.onnx
"""

import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "input_propagated_to_output")


def order_repeated_field(repeated_proto, key_name, order):
    order = list(order)
    repeated_proto.sort(key=lambda x: order.index(getattr(x, key_name)))


def make_node(op_type, inputs, outputs, name=None, doc_string=None, domain=None, **kwargs):
    node = helper.make_node(op_type, inputs, outputs, name, doc_string, domain, **kwargs)
    if doc_string == "":
        node.doc_string = ""
    order_repeated_field(node.attribute, "name", kwargs.keys())
    return node


def make_graph(*args, doc_string=None, **kwargs):
    graph = helper.make_graph(*args, doc_string=doc_string, **kwargs)
    if doc_string == "":
        graph.doc_string = ""
    return graph


model = helper.make_model(
    opset_imports=[helper.make_operatorsetid("", 14)],
    ir_version=7,
    graph=make_graph(
        name="input_propagated_to_output",
        inputs=[
            helper.make_tensor_value_info("input", TensorProto.FLOAT, shape=[1, 3, 1, 3]),
        ],
        outputs=[
            helper.make_tensor_value_info("X6", TensorProto.FLOAT, shape=[1, 3, 1, 3]),
            helper.make_tensor_value_info("X1", TensorProto.FLOAT, shape=[1, 3, 1, 3]),
            helper.make_tensor_value_info("input", TensorProto.FLOAT, shape=[1, 3, 1, 3]),
            helper.make_tensor_value_info("X2", TensorProto.FLOAT, shape=[1, 3, 1, 3]),
            helper.make_tensor_value_info("X4", TensorProto.FLOAT, shape=[1, 3, 1, 3]),
            helper.make_tensor_value_info("X3", TensorProto.FLOAT, shape=[1, 3, 1, 3]),
            helper.make_tensor_value_info("X5", TensorProto.FLOAT, shape=[1, 3, 1, 3]),
        ],
        initializer=[
            numpy_helper.from_array(
                np.load(os.path.join(DATA_DIR, "W1.npy")).astype("float32").reshape([3, 3, 1, 1]), name="W1"
            ),
            numpy_helper.from_array(
                np.load(os.path.join(DATA_DIR, "W3.npy")).astype("float32").reshape([3, 3, 1, 1]), name="W3"
            ),
            numpy_helper.from_array(
                np.load(os.path.join(DATA_DIR, "W5.npy")).astype("float32").reshape([3, 3, 1, 1]), name="W5"
            ),
            numpy_helper.from_array(
                np.load(os.path.join(DATA_DIR, "B1.npy")).astype("float32").reshape([3]), name="B1"
            ),
            numpy_helper.from_array(
                np.load(os.path.join(DATA_DIR, "B3.npy")).astype("float32").reshape([3]), name="B3"
            ),
            numpy_helper.from_array(
                np.load(os.path.join(DATA_DIR, "B5.npy")).astype("float32").reshape([3]), name="B5"
            ),
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
