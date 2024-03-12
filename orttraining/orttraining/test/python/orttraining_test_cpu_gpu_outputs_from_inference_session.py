# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# pylint: disable=W0212,C0114,C0116
import unittest

import numpy as np
import onnx

import onnxruntime
from onnxruntime.capi import _pybind_state as ORTC  # type: ignore[import]


class TestMixDeviceOutput(unittest.TestCase):
    def test_mix_device_output(self):
        A0 = onnx.helper.make_tensor_value_info("A0", onnx.TensorProto.FLOAT, [2, 3])
        B0 = onnx.helper.make_tensor_value_info("B0", onnx.TensorProto.FLOAT, [2, 3])
        C0 = onnx.helper.make_tensor_value_info("C0", onnx.TensorProto.FLOAT, [2, 3])

        A1 = onnx.helper.make_tensor_value_info("A1", onnx.TensorProto.FLOAT, [3, 2])
        B1 = onnx.helper.make_tensor_value_info("B1", onnx.TensorProto.FLOAT, [3, 2])
        C1 = onnx.helper.make_tensor_value_info("C1", onnx.TensorProto.FLOAT, [3, 2])

        n0 = onnx.helper.make_node("Add", ["A0", "B0"], ["C0"])
        n1 = onnx.helper.make_node("Sub", ["A1", "B1"], ["C1"])

        graph = onnx.helper.make_graph([n0, n1], "test", [A0, B0, A1, B1], [C0, C1])

        model = onnx.helper.make_model(graph, producer_name="test")

        opset = model.opset_import.add()
        opset.domain = ""
        opset.version = 18

        session = onnxruntime.InferenceSession(
            model.SerializeToString(),
            providers=["CUDAExecutionProvider"],
        )

        input_names = ["A0", "B0", "A1", "B1"]
        input_values = ORTC.OrtValueVector()
        for arr in [
            np.random.randn(2, 3).astype(np.float32),
            np.random.randn(2, 3).astype(np.float32),
            np.random.randn(3, 2).astype(np.float32),
            np.random.randn(3, 2).astype(np.float32),
        ]:
            input_values.push_back(
                ORTC.OrtValue.ortvalue_from_numpy(
                    arr,
                    ORTC.OrtDevice(
                        ORTC.OrtDevice.cpu(),
                        ORTC.OrtDevice.default_memory(),
                        0,
                    ),
                ),
            )
        input_devices = [
            ORTC.OrtDevice(
                ORTC.OrtDevice.cpu(),
                ORTC.OrtDevice.default_memory(),
                0,
            ),
            ORTC.OrtDevice(
                ORTC.OrtDevice.cpu(),
                ORTC.OrtDevice.default_memory(),
                0,
            ),
            ORTC.OrtDevice(
                ORTC.OrtDevice.cpu(),
                ORTC.OrtDevice.default_memory(),
                0,
            ),
            ORTC.OrtDevice(
                ORTC.OrtDevice.cpu(),
                ORTC.OrtDevice.default_memory(),
                0,
            ),
        ]
        output_names = ["C0", "C1"]
        output_devices = [
            ORTC.OrtDevice(
                ORTC.OrtDevice.cpu(),
                ORTC.OrtDevice.default_memory(),
                0,
            ),
            ORTC.OrtDevice(
                ORTC.OrtDevice.cuda(),
                ORTC.OrtDevice.default_memory(),
                0,
            ),
        ]

        output_values = ORTC.OrtValueVector()
        run_options = onnxruntime.RunOptions()
        session.run_with_ortvaluevector(
            run_options, input_names, input_values, output_names, output_values, output_devices
        )


if __name__ == "__main__":
    unittest.main()
