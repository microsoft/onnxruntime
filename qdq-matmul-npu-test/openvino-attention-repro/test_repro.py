"""Tests that run without a WinML NPU."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx

from repro import CASES, build_model, run


class ReproTest(unittest.TestCase):
    def test_cpu_controls_are_equivalent_and_weight_free(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "issue-repro"
            result = run(output, cpu_only=True)
            self.assertTrue(result["cpu_outputs_bitwise_identical"])
            self.assertFalse(result["ep_tested"])
            with np.load(output / "inputs.npz", allow_pickle=False) as inputs:
                self.assertEqual(set(inputs.files), {"scores", "mask"})
                self.assertTrue(np.all(inputs["mask"] == 0))
            for name in CASES:
                model = onnx.load(output / f"{name}.onnx")
                self.assertEqual(len(model.graph.initializer), 0 if name == "float_mask" else 2)
            with self.assertRaises(FileExistsError):
                run(output, cpu_only=True)

    def test_four_op_repro_and_two_op_float_control(self):
        self.assertEqual(
            [node.op_type for node in build_model(1).graph.node],
            ["QuantizeLinear", "DequantizeLinear", "Add", "Softmax"],
        )
        self.assertEqual([node.op_type for node in build_model(None).graph.node], ["Add", "Softmax"])


if __name__ == "__main__":
    unittest.main()
