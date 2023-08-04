"""
Run this script to recreate the original onnx model.
Example usage:
python mobilenet_v3_small_excerpt_gen.py out_model_path.onnx

The excerpt model and this script were generated from a full model by first extracting the excerpt with
onnx.utils.extract_model [1] and then generating the python script from the excerpt model with onnx2py [2].

[1]: https://github.com/onnx/onnx/blob/v1.14.0/docs/PythonAPIOverview.md#extracting-sub-model-with-inputs-outputs-tensor-names
[2]: https://github.com/microsoft/onnxconverter-common/blob/v1.13.0/onnxconverter_common/onnx2py.py
"""

import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "mobilenet_v3_small_excerpt_gen")


def clear_field(proto, field):
    proto.ClearField(field)
    return proto


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
    opset_imports=[clear_field(helper.make_operatorsetid("", 13), "domain")],
    ir_version=6,
    producer_name="onnx.utils.extract_model",
    graph=make_graph(
        name="Extracted from {torch-jit-export}",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, shape=["batch_size", 3, 224, 224])],
        outputs=[helper.make_tensor_value_info("254", TensorProto.FLOAT, shape=["batch_size", 16, 112, 112])],
        initializer=[
            numpy_helper.from_array(
                np.load(os.path.join(DATA_DIR, "const0_535.npy")).astype("float32").reshape([16, 3, 3, 3]), name="535"
            ),
            numpy_helper.from_array(
                np.load(os.path.join(DATA_DIR, "const1_536.npy")).astype("float32").reshape([16]), name="536"
            ),
        ],
        value_info=[
            helper.make_tensor_value_info("534", TensorProto.FLOAT, shape=["batch_size", 16, 112, 112]),
            helper.make_tensor_value_info("247", TensorProto.FLOAT, shape=[]),
            helper.make_tensor_value_info("248", TensorProto.FLOAT, shape=["batch_size", 16, 112, 112]),
            helper.make_tensor_value_info("249", TensorProto.FLOAT, shape=[]),
            helper.make_tensor_value_info("250", TensorProto.FLOAT, shape=[]),
            helper.make_tensor_value_info("251", TensorProto.FLOAT, shape=["batch_size", 16, 112, 112]),
            helper.make_tensor_value_info("252", TensorProto.FLOAT, shape=[]),
            helper.make_tensor_value_info("253", TensorProto.FLOAT, shape=["batch_size", 16, 112, 112]),
            helper.make_tensor_value_info("254", TensorProto.FLOAT, shape=["batch_size", 16, 112, 112]),
        ],
        nodes=[
            make_node(
                "Conv",
                inputs=["input", "535", "536"],
                outputs=["534"],
                name="Conv_0",
                dilations=[1, 1],
                group=1,
                kernel_shape=[3, 3],
                pads=[1, 1, 1, 1],
                strides=[2, 2],
            ),
            make_node(
                "Constant",
                inputs=[],
                outputs=["247"],
                name="Constant_1",
                value=numpy_helper.from_array(np.array(3.0, dtype="float32"), name=""),
            ),
            make_node("Add", inputs=["534", "247"], outputs=["248"], name="Add_2"),
            make_node(
                "Constant",
                inputs=[],
                outputs=["249"],
                name="Constant_3",
                value=numpy_helper.from_array(np.array(0.0, dtype="float32"), name=""),
            ),
            make_node(
                "Constant",
                inputs=[],
                outputs=["250"],
                name="Constant_4",
                value=numpy_helper.from_array(np.array(6.0, dtype="float32"), name=""),
            ),
            make_node("Clip", inputs=["248", "249", "250"], outputs=["251"], name="Clip_5"),
            make_node(
                "Constant",
                inputs=[],
                outputs=["252"],
                name="Constant_6",
                value=numpy_helper.from_array(np.array(6.0, dtype="float32"), name=""),
            ),
            make_node("Div", inputs=["251", "252"], outputs=["253"], name="Div_7"),
            make_node("Mul", inputs=["534", "253"], outputs=["254"], name="Mul_8"),
        ],
    ),
)

if __name__ == "__main__" and len(sys.argv) == 2:
    _, out_path = sys.argv
    onnx.save(model, out_path)
