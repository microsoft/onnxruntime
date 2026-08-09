"""
Run this script to recreate the original onnx model.
Example usage:
python invalid_dim_param_value_repetition.py
"""

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


model = onnx.helper.make_model(
    opset_imports=[onnx.helper.make_operatorsetid("", 11)],
    ir_version=5,
    producer_name="skl2onnx",
    producer_version="1.5.9999",
    domain="ai.onnx",
    model_version=0,
    graph=make_graph(
        name="OnnxIdentity",
        inputs=[
            onnx.helper.make_tensor_value_info("X1", onnx.TensorProto.FLOAT, shape=["Symbolic", "Symbolic"]),
            onnx.helper.make_tensor_value_info("X2", onnx.TensorProto.FLOAT, shape=["Symbolic", "Symbolic"]),
        ],
        outputs=[
            onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape=[None, None]),
        ],
        initializer=[
            onnx.numpy_helper.from_array(np.array([0.10000000149011612], dtype="float32"), name="Addcst"),
        ],
        nodes=[
            # take an input. Add to create a local output buffer for O01.
            make_node("Add", inputs=["X1", "Addcst"], outputs=["O01"], name="Add1", domain=""),
            # Use Shape -> ConstantOfShape to make O01 available for reuse
            make_node("Shape", inputs=["O01"], outputs=["O02"], name="Shape1", domain=""),
            # ConstantOfShape to get back to the right rank, and ReduceSum so the value is broadcastable in the
            # the downstream Add
            make_node("ConstantOfShape", inputs=["O02"], outputs=["O03"], name="ConstantOfShape ", domain=""),
            make_node("ReduceSum", inputs=["O03"], outputs=["O04"], name="ReduceSum1", domain=""),
            # Two Add nodes with the ReduceSum output. One could be in-place, but the other needs a buffer.
            # This should trigger attempted re-use of O01, so provided X2 is larger than X1 that should break
            make_node("Add", inputs=["O04", "X2"], outputs=["O05"], name="Add2", domain=""),
            make_node("Add", inputs=["X2", "O04"], outputs=["O06"], name="Add3", domain=""),
            # concat to separate the Add outputs from graph output (which is always allocated)
            make_node("Concat", inputs=["O05", "O06"], outputs=["Y"], axis=-1, name="Concat", domain=""),
        ],
    ),
)

if __name__ == "__main__":
    onnx.save(model, "invalid_dim_param_value_repetition.onnx")
