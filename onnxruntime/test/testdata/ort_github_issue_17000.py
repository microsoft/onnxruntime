import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


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


test_graph = make_graph(
    name="test_graph",
    # model input of a sequence type to test IsSparseTensor issue
    inputs=[
        helper.make_tensor_sequence_value_info("seq_in", TensorProto.FLOAT, shape=None),
    ],
    outputs=[
        helper.make_tensor_value_info("still_has_elements", TensorProto.BOOL, shape=[]),
    ],
    initializer=[
        numpy_helper.from_array(np.array(0, dtype="int64"), name="i0"),
    ],
    nodes=[
        make_node("SequenceLength", inputs=["seq_in"], outputs=["seq_len"], name="get_seq_len"),
        make_node("Greater", inputs=["seq_len", "i0"], outputs=["has_elements"], name="get_has_elements"),
        # If node with one branch that has no nodes to test the allocation planner issue
        #   if sequence has elements:
        #     remove one
        #     output bool of whether it still has elements
        #   else:
        #     output false (gives us branch with no nodes)
        make_node(
            "If",
            name="test_if",
            inputs=["has_elements"],
            outputs=["still_has_elements"],
            then_branch=make_graph(
                name="then",
                inputs=[],
                outputs=[helper.make_tensor_value_info("then_bool_out", TensorProto.BOOL, shape=[])],
                nodes=[
                    make_node("SequenceErase", inputs=["seq_in", "i0"], outputs=["seq_less_one"]),
                    make_node("SequenceLength", inputs=["seq_less_one"], outputs=["new_seq_len"]),
                    make_node("Greater", inputs=["new_seq_len", "i0"], outputs=["then_bool_out"]),
                ],
            ),
            else_branch=make_graph(
                name="else",
                initializer=[numpy_helper.from_array(np.array(False, dtype="bool"), name="else_bool_out")],
                inputs=[],
                outputs=[helper.make_tensor_value_info("else_bool_out", TensorProto.BOOL, shape=[])],
                nodes=[],
            ),
        ),
    ],
)

# Graph with Sequence operations and an If node that has a subgraph with no nodes
model = helper.make_model(opset_imports=[helper.make_operatorsetid("ai.onnx", 14)], ir_version=7, graph=test_graph)

onnx.shape_inference.infer_shapes(model, strict_mode=True)
onnx.save(model, "ort_github_issue_17000.onnx")
