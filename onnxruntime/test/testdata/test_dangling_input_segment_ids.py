"""
Run this script to recreate the original onnx model.
Example usage:
python test_dangling_input_segment_ids.py out_model_path.onnx
"""

import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_dangling_input_segment_ids")


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
    opset_imports=[helper.make_operatorsetid("", 14), helper.make_operatorsetid("com.microsoft", 1)],
    ir_version=7,
    graph=make_graph(
        name="embed_layernorm_graph",
        inputs=[
            helper.make_tensor_value_info("input_ids", TensorProto.INT32, shape=[1, 4]),
            helper.make_tensor_value_info("segment_ids", TensorProto.INT32, shape=[1, 4]),
        ],
        outputs=[
            helper.make_tensor_value_info("layernorm_out", TensorProto.FLOAT, shape=[1, 4, 4]),
            helper.make_tensor_value_info("mask_index_out", TensorProto.INT32, shape=[1]),
        ],
        initializer=[
            numpy_helper.from_array(
                np.load(os.path.join(DATA_DIR, "const0_word_embed.npy")).astype("float32").reshape([32, 4]),
                name="word_embed",
            ),
            numpy_helper.from_array(
                np.load(os.path.join(DATA_DIR, "const1_pos_embed.npy")).astype("float32").reshape([16, 4]),
                name="pos_embed",
            ),
            numpy_helper.from_array(
                np.array(
                    [0.6185135841369629, 0.010364261455833912, 0.5386272668838501, 0.0030179566238075495],
                    dtype="float32",
                ),
                name="gamma",
            ),
            numpy_helper.from_array(
                np.array(
                    [0.9511938095092773, 0.9054020047187805, 0.7959669232368469, 0.9152743220329285], dtype="float32"
                ),
                name="beta",
            ),
        ],
        nodes=[
            make_node(
                "EmbedLayerNormalization",
                inputs=["input_ids", "", "word_embed", "pos_embed", "", "gamma", "beta"],
                outputs=["layernorm_out", "mask_index_out"],
                domain="com.microsoft",
            )
        ],
    ),
)

if __name__ == "__main__" and len(sys.argv) == 2:
    _, out_path = sys.argv
    onnx.save(model, out_path)
