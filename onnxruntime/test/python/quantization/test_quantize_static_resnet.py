# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import random
import tempfile
import unittest

import numpy as np
import onnx
from numpy.testing import assert_allclose
from onnx.numpy_helper import to_array
from resnet_code import create_model

from onnxruntime import InferenceSession
from onnxruntime import __version__ as ort_version
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.calibrate import CalibrationDataReader, CalibrationMethod


class FakeResnetCalibrationDataReader(CalibrationDataReader):
    def __init__(self, batch_size: int = 16):
        super().__init__()
        self.dataset = [
            (np.random.rand(1, 3, 32, 32).astype(np.float32), random.randint(0, 9)) for _ in range(batch_size)
        ]
        self.iterator = iter(self.dataset)

    def get_next(self) -> dict:
        try:
            return {"input": next(self.iterator)[0]}
        except Exception:
            return None


class TestStaticQuantizationResNet(unittest.TestCase):
    def test_quantize_static_resnet(self):
        kwargs = {
            "activation_type": QuantType.QUInt8,
            "weight_type": QuantType.QInt8,
            "calibrate_method": CalibrationMethod.Percentile,
            "extra_options": {
                "ActivationSymmetric": False,
                "EnableSubgraph": False,
                "ForceQuantizeNoInputCheck": False,
                "MatMulConstBOnly": False,
                "WeightSymmetric": True,
                "extra.Sigmoid.nnapi": False,
            },
            "nodes_to_exclude": None,
            "nodes_to_quantize": None,
            "op_types_to_quantize": None,
            "per_channel": True,
            "quant_format": QuantFormat.QDQ,
            "reduce_range": False,
        }

        proto = create_model()

        with tempfile.TemporaryDirectory() as temp:
            model = os.path.join(temp, "resnet_first_nodes.onnx")
            with open(model, "wb") as f:
                f.write(proto.SerializeToString())

            for per_channel in [True, False]:
                kwargs["per_channel"] = per_channel
                dataloader = FakeResnetCalibrationDataReader(16)
                with self.subTest(per_channel=per_channel):
                    qdq_file = os.path.join(
                        temp, f"preprocessed-small-qdq-{1 if per_channel else 0}-ort-{ort_version}.onnx"
                    )
                    quantize_static(
                        model_input=model,
                        model_output=qdq_file,
                        calibration_data_reader=dataloader,
                        use_external_data_format=False,
                        **kwargs,
                    )

                    # With onnxruntime==1.15.1, the initializer 'onnx::Conv_504_zero_point' is:
                    # * uint8(128) if per_channel is False
                    # * int8([0, 0, ....]) if per_channel is True
                    # With onnxruntime>1.16.0
                    # * uint8(128) if per_channel is False
                    # * uint8([128, 128, ..., 127, ...]) if per_channel is True
                    # QLinearConv : zero point of per-channel filter must be same.
                    # That's why the quantization forces a symmetric quantization into INT8.
                    # zero_point is guaranted to be zero whatever the channel is.

                    with open(qdq_file, "rb") as f:
                        onx = onnx.load(f)
                    for init in onx.graph.initializer:
                        arr = to_array(init)
                        if (
                            arr.dtype == np.int8
                            and "zero_point" not in init.name
                            and not init.name.endswith("quantized")
                        ):
                            raise AssertionError(
                                f"Initializer {init.name!r} has type {arr.dtype} and "
                                f"shape {arr.shape} but should be {np.uint8}."
                            )

                    sess = InferenceSession(qdq_file, providers=["CPUExecutionProvider"])
                    shape = (1, 3, 32, 32)
                    size = np.prod(shape)
                    dummy = (np.arange(size) / float(size)).astype(np.float32).reshape(shape)
                    got = sess.run(None, {"input": dummy})
                    self.assertEqual(got[0].shape, (1, 64, 8, 8))
                    self.assertEqual(got[0].dtype, np.float32)
                    if per_channel:
                        expected = np.array(
                            [
                                [[1.0862497091293335, 0.9609132409095764], [1.0862497091293335, 0.9191343784332275]],
                                [[0.7520190477371216, 1.0026921033859253], [1.0444709062576294, 1.0862497091293335]],
                                [[0.0, 0.0], [0.0, 0.0]],
                                [[0.0, 0.0], [0.9609132409095764, 0.7937979102134705]],
                            ],
                            dtype=np.float32,
                        )
                        assert_allclose(expected, got[0][0, :4, :2, :2], atol=0.2)
                    else:
                        expected = np.array(
                            [
                                [[1.428238868713379, 1.2602107524871826], [1.3442248106002808, 1.2182037830352783]],
                                [[0.8821475505828857, 1.0921826362609863], [1.1341897249221802, 1.1761966943740845]],
                                [[0.0, 0.0], [0.0, 0.0]],
                                [[0.0, 0.0], [1.2182037830352783, 1.050175666809082]],
                            ],
                            dtype=np.float32,
                        )
                        assert_allclose(expected, got[0][0, :4, :2, :2], atol=0.2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
