"""
Run this script to recreate the original onnx model.
Example usage:
python packed_attention_fp16.model.py out_model_path.onnx
"""

import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def clear_field(proto, field):
    proto.ClearField(field)
    return proto


def order_repeated_field(repeated_proto, key_name, order):
    order = list(order)
    repeated_proto.sort(key=lambda x: order.index(getattr(x, key_name)))


def make_node(op_type, inputs, outputs, name=None, doc_string=None, domain=None, **kwargs):
    node = helper.make_node(op_type, inputs, outputs, name, doc_string, domain, **kwargs)
    if not doc_string:
        node.doc_string = ""
    order_repeated_field(node.attribute, "name", kwargs.keys())
    return node


def make_graph(*args, doc_string=None, **kwargs):
    graph = helper.make_graph(*args, doc_string=doc_string, **kwargs)
    if not doc_string:
        graph.doc_string = ""
    return graph


model = helper.make_model(
    opset_imports=[
        clear_field(helper.make_operatorsetid("", 12), "domain"),
        helper.make_operatorsetid("com.microsoft", 1),
    ],
    ir_version=7,
    producer_name="onnxruntime.transformers",
    producer_version="1.15.0",
    graph=make_graph(
        name="torch-jit-export",
        inputs=[
            helper.make_tensor_value_info("input", TensorProto.FLOAT16, shape=["token_count", "hidden_size"]),
            helper.make_tensor_value_info("weight", TensorProto.FLOAT16, shape=["hidden_size", "hidden_size_x_3"]),
            helper.make_tensor_value_info("bias", TensorProto.FLOAT16, shape=["hidden_size_x_3"]),
            helper.make_tensor_value_info("token_offset", TensorProto.INT32, shape=["batch_size", "seq_len"]),
            helper.make_tensor_value_info("cumulative_sequence_length", TensorProto.INT32, shape=["batch_size_plus_1"]),
            helper.make_tensor_value_info(
                "rbp", TensorProto.FLOAT16, shape=["batch_size", "num_heads", "seq_len", "seq_len"]
            ),
        ],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT16, shape=["token_count", "hidden_size"])],
        nodes=[
            make_node(
                "Constant",
                inputs=[],
                outputs=["constant_0"],
                name="constant_0",
                value=numpy_helper.from_array(np.array([0], dtype="int64"), name=""),
            ),
            make_node(
                "Constant",
                inputs=[],
                outputs=["constant_1"],
                name="Constant_1",
                value=numpy_helper.from_array(np.array([1], dtype="int64"), name=""),
            ),
            make_node(
                "Constant",
                inputs=[],
                outputs=["constant_last"],
                name="constant_last",
                value=numpy_helper.from_array(np.array([-1], dtype="int64"), name=""),
            ),
            make_node(
                "Constant",
                inputs=[],
                outputs=["constant_max"],
                name="Constant_max",
                value=numpy_helper.from_array(np.array([9223372036854775807], dtype="int64"), name=""),
            ),
            make_node(
                "Slice",
                inputs=["cumulative_sequence_length", "constant_0", "constant_last"],
                outputs=["start"],
                name="slice_start",
            ),
            make_node(
                "Slice",
                inputs=["cumulative_sequence_length", "constant_1", "constant_max"],
                outputs=["end"],
                name="slice_end",
            ),
            make_node("Sub", inputs=["end", "start"], outputs=["mask_idx"]),
            make_node(
                "RestorePadding",
                inputs=["input", "token_offset"],
                outputs=["restore_padding_output"],
                name="RestorePadding_1",
                domain="com.microsoft",
            ),
            make_node(
                "Attention",
                inputs=["restore_padding_output", "weight", "bias", "mask_idx", "", "rbp"],
                outputs=["attention_outputs"],
                name="Attention_0",
                domain="com.microsoft",
                num_heads=12,
                mask_filter_value=-3.4028234663852886e38,
            ),
            make_node(
                "RemovePadding",
                inputs=["attention_outputs", "mask_idx"],
                outputs=["output", "remove_padding_token_offset", "cumulated_seq_len", "max_seq_len"],
                name="RemovePadding_0",
                domain="com.microsoft",
            ),
        ],
    ),
)

if __name__ == "__main__" and len(sys.argv) == 2:
    _, out_path = sys.argv
    onnx.save(model, out_path)
