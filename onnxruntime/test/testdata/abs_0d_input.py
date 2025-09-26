"""
Run this script to recreate the original onnx model.
Example usage:
python abs_0d_input.py out_model_path.onnx
"""

import sys

from onnx import TensorProto, helper, save


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
    opset_imports=[
        clear_field(helper.make_operatorsetid("", 21), "domain"),
        clear_field(helper.make_operatorsetid("", 1), "domain"),
        clear_field(helper.make_operatorsetid("", 1), "domain"),
        clear_field(helper.make_operatorsetid("", 21), "domain"),
    ],
    ir_version=11,
    producer_name="ort_ep_utils::OrtGraphToProto",
    producer_version="",
    model_version=0,
    graph=make_graph(
        name="OpenVINOExecutionProvider_11295571201636618024_0",
        inputs=[helper.make_tensor_value_info("absInput_1", TensorProto.FLOAT, shape=[])],
        outputs=[helper.make_tensor_value_info("absOutput_0", TensorProto.FLOAT, shape=[])],
        nodes=[make_node("Abs", inputs=["absInput_1"], outputs=["absOutput_0"], name="_0", domain="")],
    ),
)

if __name__ == "__main__" and len(sys.argv) == 2:
    _, out_path = sys.argv
    save(model, out_path)
