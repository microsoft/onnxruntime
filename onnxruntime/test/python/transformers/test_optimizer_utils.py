import unittest
import numpy
from onnx import ModelProto, TensorProto, helper
from onnxruntime.python.tools.transformers.optimizer_utils import extract_external_data_from_model


class TestOptimizerUtils(unittest.TestCase):
    def test_extract_external_data_from_model(self):
        model = self._get_model_proto_with_raw_data()
        external_names, external_values = extract_external_data_from_model(model)
        self.assertEqual(list(external_names), ["inputs"])
        self.assertEqual(len(external_values), 1)
        self.assertEqual(external_values[0].numpy(), [0.0])


    def _get_model_proto_with_raw_data(self) -> ModelProto:
        input = helper.make_tensor_value_info("inputs", TensorProto.FLOAT, [None])
        output = helper.make_tensor_value_info("outputs", TensorProto.FLOAT, [None])
        raw_data = numpy.array([0.0], dtype=numpy.float32).tobytes()
        tensor = helper.make_tensor("inputs", TensorProto.FLOAT, [1], raw_data, True)
        node = helper.make_node("Identity", inputs=["inputs"], outputs=["outputs"])
        return helper.make_model(helper.make_graph([node], "graph", [input], [output], initializer=[tensor]))
