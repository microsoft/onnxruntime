import numpy
import unittest
import onnxruntime

from typing import Dict, List, AnyStr, Tuple, Any
from onnx import TensorProto, ModelProto
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info
from numpy.testing import assert_almost_equal


numpy.random.seed(32)


def get_model_with_undefined_shapes() -> ModelProto:
    X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
    A = make_tensor_value_info("A", TensorProto.FLOAT, [None, None])
    B = make_tensor_value_info("B", TensorProto.FLOAT, [None, None])
    Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])
    node1 = make_node("MatMul", ["X", "A"], ["XA"])
    node2 = make_node("Add", ["XA", "B"], ["Y"])
    graph = make_graph([node1, node2], "lr", [X, A, B], [Y])
    onnx_model = make_model(graph)
    return onnx_model


def get_input_data_for_model_with_undefined_shapes() -> Dict[AnyStr, numpy.ndarray]:
    a = numpy.random.randn(2, 2).astype(numpy.float32)
    b = numpy.random.randn(1, 2).astype(numpy.float32)
    x = numpy.random.randn(1, 2).astype(numpy.float32)
    data = {"A": a, "B": b, "X": x}
    return data


def get_input_names_and_shapes(data: Dict[AnyStr, numpy.ndarray]) -> Tuple[List[AnyStr], List[AnyStr]]:
    keys = list(data.keys())
    values = [data[key] for key in keys]
    return (
        list(data.keys()),
        [str(value.shape).replace(",", "").replace("(", "[").replace(")", "]") for value in values]
    )


def get_cpu_output(onnx_model: ModelProto, data: Dict[AnyStr, numpy.ndarray]) -> List[numpy.ndarray]:
    sess = onnxruntime.InferenceSession(onnx_model.SerializeToString(), providers=["CPUExecutionProvider"])
    output = sess.run(None, data)
    return output


def get_tvm_output(onnx_model: ModelProto, data: Dict[AnyStr, numpy.ndarray], provider_options: Dict[AnyStr, Any]) -> List[numpy.ndarray]:
    so = onnxruntime.SessionOptions()
    so.log_severity_level = 0
    so.log_verbosity_level = 0
    so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL

    sess = onnxruntime.InferenceSession(
        onnx_model.SerializeToString(),
        so,
        providers=["TvmExecutionProvider"],
        provider_options=[provider_options],
    )

    output = sess.run(None, data)
    return output


class TestTVM(unittest.TestCase):
    def test_accuracy_for_model_with_undefined_shapes(self):
        onnx_model = get_model_with_undefined_shapes()
        data = get_input_data_for_model_with_undefined_shapes()

        cpu_output = get_cpu_output(onnx_model, data)
        names, shapes = get_input_names_and_shapes(data)
        provider_options = dict(
            target="llvm",
            input_names=" ".join(names),
            input_shapes=" ".join(shapes),
        )
        tvm_output = get_tvm_output(onnx_model, data, provider_options)

        assert_almost_equal(cpu_output, tvm_output, decimal=5)


if __name__ == '__main__':
    if "TvmExecutionProvider" not in onnxruntime.get_available_providers():
        raise AssertionError("Unable to find 'TvmExecutionProvider' in %r." % onnxruntime.get_available_providers())
    unittest.main()
