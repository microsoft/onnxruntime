import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import onnx

from drift_investigation.make_mask_softmax_repro import make_model
from drift_investigation.vision.make_short_sequence_repro import resize_model
from drift_investigation.vision.vary_mask_qdq import main as vary_mask_main


class ReproToolsTest(unittest.TestCase):
    def test_mask_scale_variant_preserves_cpu_results(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            source, variant = folder / "source.onnx", folder / "variant.onnx"
            onnx.save(make_model(True, 1.0, 100, 8), source)
            np.savez(folder / "inputs.npz", scores=np.linspace(-1, 1, 8, dtype=np.float32).reshape(1, 1, 8),
                     mask=np.zeros((1, 1, 8), dtype=np.float32))
            with patch("sys.argv", [
                "vary_mask_qdq", str(source), "--output", str(variant),
                "--scale", "0.5", "--zero-point", "200", "--scale-initializer", "mask_scale",
                "--zero-point-initializer", "mask_zero_point", "--inputs", str(folder / "inputs.npz"),
            ]):
                vary_mask_main()
            self.assertTrue(variant.is_file())

    def test_short_sequence_resizes_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / "source.onnx", Path(temp) / "short.onnx"
            onnx.save(make_model(False, 1.0, 100, 8), source)
            resize_model(source, output, 8, 4)
            self.assertEqual(
                onnx.load(output).graph.input[0].type.tensor_type.shape.dim[-1].dim_value, 4
            )


if __name__ == "__main__":
    unittest.main()
