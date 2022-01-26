import numpy
from numpy.testing import assert_almost_equal
from onnx import numpy_helper, TensorProto
from onnx.helper import (
    make_model, make_node, set_model_props, make_tensor,
    make_graph, make_tensor_value_info)
import tvm
import onnxruntime
import onnxruntime.providers.stvm

if "StvmExecutionProvider" not in onnxruntime.get_available_providers():
    raise AssertionError(
        "Unable to find 'StvmExecutionProvider' in %r." % onnxruntime.get_available_providers())

X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
B = make_tensor_value_info('B', TensorProto.FLOAT, [None, None])
Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None, None])
node1 = make_node('MatMul', ['X', 'A'], ['XA'])
node2 = make_node('Add', ['XA', 'B'], ['Y'])
graph = make_graph([node1, node2], 'lr', [X, A, B], [Y])
onnx_model = make_model(graph)

a = numpy.random.randn(2, 2).astype(numpy.float32)
b = numpy.random.randn(1, 2).astype(numpy.float32)
x = numpy.random.randn(1, 2).astype(numpy.float32)

sess = onnxruntime.InferenceSession(
    onnx_model.SerializeToString(), providers=['CPUExecutionProvider'])

y = sess.run(None, {'A': a, 'B': b, 'X': x})[0]

provider_options = dict(
    target="llvm -mcpu=core-avx2",
    target_host="llvm -mcpu=core-avx2",
    opt_level=3,
    freeze_weights=True,
    tuning_file_path="",
    tuning_type="Ansor")

so = onnxruntime.SessionOptions()
so.log_severity_level = 0
so.log_verbosity_level = 0
so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL

sess = onnxruntime.InferenceSession(
    onnx_model.SerializeToString(), so,
    providers=["StvmExecutionProvider"],
    provider_options=[provider_options])

y_tvm = sess.run(None, {'A': a, 'B': b, 'X': x})[0]

assert_almost_equal(y, y_tvm)

"""
AssertionError: 
Arrays are not almost equal to 7 decimals
  (shapes (1, 2), (1, 1) mismatch)
    x: array([[2.2854233, 0.6363953]], dtype=float32)
    y: array([[1.9743297]], dtype=float32)
"""
