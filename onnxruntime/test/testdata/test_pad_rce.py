import onnx
from onnx import helper, TensorProto
import numpy as np

def create_pad_model():
    input_data = helper.make_tensor_value_info("input", TensorProto.UINT64, [None, None, None])
    pads = helper.make_tensor_value_info("pads", TensorProto.INT64, [None])
    constant_value = helper.make_tensor_value_info("constant_value", TensorProto.UINT64, [])

    output = helper.make_tensor_value_info("output", TensorProto.UINT64, [None, None, None, None])

    pad_node = helper.make_node(
        op_type="Pad",
        inputs=["input", "pads", "constant_value"],
        outputs=["output"],
        mode="constant"  # or reflect/edge
    )
    graph = helper.make_graph(
        nodes=[pad_node],
        name="PadModel",
        inputs=[input_data, pads, constant_value],
        outputs=[output]
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    onnx.save(model, "test_pad_rce.onnx")


if __name__ == "__main__":
    create_pad_model()