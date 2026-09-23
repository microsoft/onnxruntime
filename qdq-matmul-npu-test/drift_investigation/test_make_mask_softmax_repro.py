import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx

from drift_investigation.make_mask_softmax_repro import write_repro


class MaskSoftmaxReproTest(unittest.TestCase):
    def test_matched_controls_at_multiple_scales(self):
        with tempfile.TemporaryDirectory() as temp:
            for scale in (0.5, 1.0, 2.0):
                folder = Path(temp) / str(scale)
                write_repro(folder, scale, 100, 8, 0.0)
                with np.load(folder / "inputs.npz", allow_pickle=False) as inputs:
                    self.assertEqual(set(inputs.files), {"scores", "mask"})
                self.assertEqual(
                    [node.op_type for node in onnx.load(folder / "qdq_mask.onnx").graph.node],
                    ["QuantizeLinear", "DequantizeLinear", "Add", "Softmax"],
                )
                self.assertEqual(
                    [node.op_type for node in onnx.load(folder / "float_mask.onnx").graph.node],
                    ["Add", "Softmax"],
                )

    def test_rejects_nonrepresentable_mask_and_existing_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "repro"
            with self.assertRaisesRegex(ValueError, "exactly representable"):
                write_repro(folder, 2.0, 100, 8, -1.0)
            write_repro(folder, 2.0, 100, 8, -2.0)
            with self.assertRaises(FileExistsError):
                write_repro(folder, 2.0, 100, 8, -2.0)


if __name__ == "__main__":
    unittest.main()
