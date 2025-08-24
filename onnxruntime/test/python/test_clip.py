import numpy as np
import onnx
from onnx import TensorProto, helper

import onnxruntime as ort


def make_clip_model() -> onnx.ModelProto:
    max_node = helper.make_node(op_type="Constant", domain="ai.onnx", inputs=[], outputs=["max"], value_int=0)
    clip_node = helper.make_node(
        op_type="Clip",
        domain="ai.onnx",
        inputs=["x", "", "max"],
        outputs=["y"],
    )

    graph = helper.make_graph(
        nodes=[max_node, clip_node],
        name="TestClip",
        inputs=[helper.make_value_info("x", type_proto=helper.make_tensor_type_proto(TensorProto.INT64, [2]))],
        outputs=[helper.make_value_info("y", type_proto=helper.make_tensor_type_proto(TensorProto.INT64, [2]))],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    return model


def test_clip():
    sess = ort.InferenceSession(make_clip_model().SerializeToString())
    x = np.asarray([2147483649, 2147483649], np.int64)
    (res,) = sess.run(None, {"x": x})

    np.testing.assert_array_equal(res, np.clip(x, min=None, max=0))
