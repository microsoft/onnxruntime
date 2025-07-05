import os
import tempfile
import unittest

import numpy as np
import onnx
from op_test_utils import TestDataFeeds, check_model_correctness

from onnxruntime.quantization import QuantFormat, QuantType, quantize_static


class TestAdjustWeightScaleForInt32BiasQOperator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp_model_dir = tempfile.TemporaryDirectory(prefix="ort.qop.adj_int32_bias_")
        cls._tmp_dir_path = cls._tmp_model_dir.name

    @classmethod
    def tearDownClass(cls):
        cls._tmp_model_dir.cleanup()

    def build_conv_test_model(self, input_shape, weight_shape, onnx_float_type):
        np_float_type = onnx.helper.tensor_dtype_to_np_dtype(onnx_float_type)
        input_0 = onnx.helper.make_tensor_value_info("input_0", onnx_float_type, input_shape)
        output_0 = onnx.helper.make_tensor_value_info("output_0", onnx_float_type, None)

        tiny_value = 1e-7 if np_float_type == np.float32 else 0.007782

        # Step 1: reshape to (C_out, -1) to ensure per-channel broadcasting
        weight_data = np.full(weight_shape, tiny_value, dtype=np_float_type)
        weight_data = weight_data.reshape(weight_shape[0], -1)
        for i in range(weight_data.shape[0]):
            for j in range(weight_data.shape[1]):
                if j % 2 == 0:
                    weight_data[i, j] = -weight_data[i, j]
        # Step 2: reshape back to original shape
        weight_data = weight_data.reshape(weight_shape)
        weight = onnx.numpy_helper.from_array(weight_data, "weight")

        bias_shape = [weight_shape[0]]
        bias_data = np.ones(bias_shape, dtype=np_float_type)
        for i in range(len(bias_data)):
            bias_data[i] = 5.0 if (i % 2 == 0) else -4.5
            if np_float_type == np.float16:
                bias_data[i] = 1400 if (i % 2 == 0) else -1200
        bias = onnx.numpy_helper.from_array(bias_data, "bias")

        conv_node = onnx.helper.make_node("Conv", ["input_0", "weight", "bias"], ["output_0"], name="Conv0")
        graph = onnx.helper.make_graph([conv_node], "Convfloat", [input_0], [output_0], initializer=[weight, bias])
        opset_imports = [onnx.helper.make_opsetid("", 21)]
        model = onnx.helper.make_model(graph, opset_imports=opset_imports)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model, True)
        return model

    def test_adjust_weight_scale_for_int32_bias_qop(self):
        test_configs = [
            (onnx.TensorProto.FLOAT, True),
            (onnx.TensorProto.FLOAT, False),
            (onnx.TensorProto.FLOAT, True),
            (onnx.TensorProto.FLOAT, False),
        ]

        for float_type, per_channel in test_configs:
            with self.subTest(float_type=float_type, per_channel=per_channel):
                label = f"_f{float_type}_perchannel{per_channel}"
                float_model_path = os.path.join(self._tmp_dir_path, f"conv{label}.float.onnx")
                qop_model_path = os.path.join(self._tmp_dir_path, f"conv{label}.qop.onnx")

                input_shape = [1, 1, 128, 128]
                weight_shape = [8, 1, 1, 1]
                float_model = self.build_conv_test_model(input_shape, weight_shape, float_type)
                onnx.save_model(float_model, float_model_path)

                np_float_type = onnx.helper.tensor_dtype_to_np_dtype(float_type)
                input_rmin = 0.0
                input_scale = 0.05 if float_type == onnx.TensorProto.FLOAT else 0.01
                input_rmax = (input_scale * 255.0) + input_rmin
                input_data_list = [
                    {"input_0": np.full(input_shape, input_rmin, dtype=np_float_type)},
                    {"input_0": np.full(input_shape, (input_rmax - input_rmin) / 2.0, dtype=np_float_type)},
                    {"input_0": np.full(input_shape, input_rmax, dtype=np_float_type)},
                ]
                data_reader = TestDataFeeds(input_data_list)

                quantize_static(
                    float_model_path,
                    qop_model_path,
                    data_reader,
                    activation_type=QuantType.QInt8,
                    weight_type=QuantType.QInt8,
                    per_channel=per_channel,
                    quant_format=QuantFormat.QOperator,
                    extra_options={
                        "ActivationSymmetric": True,
                        "WeightSymmetric": True,
                    },
                )

                data_reader.rewind()
                check_model_correctness(self, float_model_path, qop_model_path, data_reader.get_next())


if __name__ == "__main__":
    unittest.main()
