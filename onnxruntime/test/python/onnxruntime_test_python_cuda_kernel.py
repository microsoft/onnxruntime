# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from onnx import TensorProto, helper

import onnxruntime as onnxrt


def create_mul_model():
    graph = helper.make_graph(
        [helper.make_node("Mul", ["x", "y"], ["z"])],
        "mul_kernel",
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, ["size"]) for name in ("x", "y")],
        [helper.make_tensor_value_info("z", TensorProto.FLOAT, ["size"])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    return model


class TestCudaKernel(unittest.TestCase):
    def _run_mul(self, provider):
        self.assertIn(provider, onnxrt.get_available_providers())
        with tempfile.TemporaryDirectory() as directory:
            options = onnxrt.SessionOptions()
            options.graph_optimization_level = onnxrt.GraphOptimizationLevel.ORT_DISABLE_ALL
            options.enable_profiling = True
            options.profile_file_prefix = str(Path(directory) / "mul_profile")
            if provider == "CUDAExecutionProvider":
                options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")

            session = onnxrt.InferenceSession(
                create_mul_model().SerializeToString(), sess_options=options, providers=[provider]
            )
            try:
                session.disable_fallback()
                self.assertIn(provider, session.get_providers())
                for size in (17, 257):
                    with self.subTest(provider=provider, size=size):
                        x = np.arange(size, dtype=np.float32) - np.float32(size // 2)
                        y = (np.arange(size, dtype=np.float32) % np.float32(7) - np.float32(3)) / np.float32(4)
                        actual = session.run(["z"], {"x": x, "y": y})[0]
                        np.testing.assert_array_equal(actual, x * y)
            finally:
                profile_path = session.end_profiling()

            with open(profile_path) as profile:
                events = json.load(profile)
            mul_events = [
                event
                for event in events
                if event.get("cat") == "Node"
                and event.get("name", "").endswith("_kernel_time")
                and event.get("args", {}).get("op_name") == "Mul"
            ]
            self.assertEqual(len(mul_events), 2, "Expected a profiled Mul kernel for each inference")
            self.assertEqual({event["args"].get("provider") for event in mul_events}, {provider})

    def test_mul_cpu(self):
        self._run_mul("CPUExecutionProvider")

    def test_mul_cuda(self):
        self._run_mul("CUDAExecutionProvider")


if __name__ == "__main__":
    unittest.main()
