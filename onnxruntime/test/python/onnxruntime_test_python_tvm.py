import numpy
from numpy.testing import assert_almost_equal
from onnx import numpy_helper, TensorProto
from onnx.helper import (
    make_model, make_node, set_model_props, make_tensor,
    make_graph, make_tensor_value_info)
import onnxruntime

if "TvmExecutionProvider" not in onnxruntime.get_available_providers():
    raise AssertionError(
        "Unable to find 'TvmExecutionProvider' in %r." % onnxruntime.get_available_providers())

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
data = {'A': a, 'B': b, 'X': x}

sess = onnxruntime.InferenceSession(
    onnx_model.SerializeToString(), providers=['CPUExecutionProvider'])

y = sess.run(None, data)[0]

provider_options = dict(
    target="llvm -mcpu=core-avx2",
    target_host="llvm -mcpu=core-avx2",
    opt_level=3,
    freeze_weights=True,
    tuning_file_path="",
    tuning_type="Ansor",
    input_names=" ".join(i.name for i in sess.get_inputs()),
    input_shapes=" ".join(str(numpy.array(data[i.name].shape)) 
                          for i in sess.get_inputs()))

so = onnxruntime.SessionOptions()
so.log_severity_level = 0
so.log_verbosity_level = 0
so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL

sess = onnxruntime.InferenceSession(
    onnx_model.SerializeToString(), so,
    providers=["TvmExecutionProvider"],
    provider_options=[provider_options])

y_tvm = sess.run(None, data)[0]

assert_almost_equal(y, y_tvm)
