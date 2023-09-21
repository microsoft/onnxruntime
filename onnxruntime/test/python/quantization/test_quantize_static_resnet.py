#!/usr/bin/env python
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
from numpy.testing import assert_allclose

from onnxruntime import InferenceSession
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.calibrate import CalibrationDataReader, CalibrationMethod
from resnet_code import create_model


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
            # if per_channel is True, it raises an exception.
            # QLinearConv : zero point of per-channel filter must be same
            "per_channel": False,
            "quant_format": QuantFormat.QDQ,
            "reduce_range": False,
            "weight_type": QuantType.QUInt8,
        }

        dataloader = FakeResnetCalibrationDataReader(16)
        proto = create_model()

        with tempfile.TemporaryDirectory() as temp:
            model = os.path.join(temp, "resnet_first_nodes.onnx")
            with open(model, "wb") as f:
                f.write(proto.SerializeToString())
            qdq_file = os.path.join(temp, "preprocessed-small-qdq.onnx")
            quantize_static(
                model_input=model,
                model_output=qdq_file,
                calibration_data_reader=dataloader,
                use_external_data_format=False,
                **kwargs,
            )

            sess = InferenceSession(qdq_file, providers=["CPUExecutionProvider"])
            shape = (1, 3, 32, 32)
            size = np.prod(shape)
            dummy = (np.arange(size) / float(size)).astype(np.float32).reshape(shape)
            got = sess.run(None, {"input": dummy})
            self.assertEqual(got[0].shape, (1, 64, 8, 8))
            self.assertEqual(got[0].dtype, np.float32)
            expected = np.array(
                [
                    [[1.4244736433029175, 1.256888508796692], [1.3406810760498047, 1.2149922847747803]],
                    [[0.8798219561576843, 1.131199598312378], [1.131199598312378, 1.173095941543579]],
                    [[0.0, 0.0], [0.0, 0.0]],
                    [[0.0, 0.0], [1.2149922847747803, 1.0474071502685547]],
                ],
                dtype=np.float32,
            )
            print(got[0][0, :4, :2, :2].tolist())
            assert_allclose(expected, got[0][0, :4, :2, :2], atol=1e-2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
