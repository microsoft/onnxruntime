#!/usr/bin/env python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import unittest
import os
import random
import tempfile
import numpy as np
from onnxruntime import InferenceSession
from onnxruntime.quantization.calibrate import CalibrationDataReader
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.calibrate import CalibrationMethod


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
        folder = os.path.join(os.path.dirname(__file__), "..", "..", "testdata")
        model = os.path.join(folder, "resnet_first_nodes.onnx")
        if not os.path.exists(model):
            raise FileNotFoundError(f"Unable to find {model!r} in testdata.")

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
            "per_channel": True,
            "quant_format": QuantFormat.QDQ,
            "reduce_range": False,
            "weight_type": QuantType.QUInt8,
        }

        dataloader = FakeResnetCalibrationDataReader(16)

        with tempfile.TemporaryDirectory(prefix="test_calibration.") as temp:
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
            print(got[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
