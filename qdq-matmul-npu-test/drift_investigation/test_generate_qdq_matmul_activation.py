"""Check activation types and all supported weight granularities in unit MatMul models."""

from contextlib import redirect_stderr
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

from generate_qdq_matmul_model import PROJECTIONS, build_model, parse_args
from run_acc import create_cpu_session


def make_model(activation_type: str, granularity: str):
    options = [
        "generate_qdq_matmul_model.py",
        "--input-shape", "1", "32", "32",
        "--weight-shape", "32", "16",
        "--weight-quantization", granularity,
        "--weight-signedness", "signed",
        "--weight-bit-width", "4",
        "--weight-symmetry", "symmetric",
        "--omit-weight-zero-point",
        "--activation-type", activation_type,
    ]
    with patch.object(sys, "argv", options):
        return build_model(parse_args())


class GenerateQdqMatmulActivationTest(unittest.TestCase):
    def test_per_tensor_channel_and_blockwise_support_both_activation_types(self):
        data = np.random.default_rng(4).standard_normal((1, 32, 32)).astype(np.float32)
        for granularity in ("per-tensor", "per-channel", "blockwise"):
            for activation_type in ("uint16", "uint8"):
                with self.subTest(granularity=granularity, activation_type=activation_type):
                    model = make_model(activation_type, granularity)
                    initializers = {
                        tensor.name: numpy_helper.to_array(tensor)
                        for tensor in model.graph.initializer
                    }
                    dtype = np.dtype(activation_type)
                    self.assertEqual(initializers["activation_zero_point"].dtype, dtype)
                    self.assertEqual(initializers["output_zero_point"].dtype, dtype)
                    expected = PROJECTIONS["visual"].activation_scale * 65535 / np.iinfo(dtype).max
                    self.assertAlmostEqual(initializers["activation_scale"].item(), expected, delta=expected * 1e-6)
                    self.assertEqual(
                        initializers["activation_zero_point"].item(),
                        round(PROJECTIONS["visual"].activation_zero_point * np.iinfo(dtype).max / 65535),
                    )
                    self.assertEqual(
                        initializers["output_zero_point"].item(),
                        round(PROJECTIONS["visual"].output_zero_point * np.iinfo(dtype).max / 65535),
                    )
                    options = ort.SessionOptions()
                    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
                    result = ort.InferenceSession(
                        model.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"]
                    ).run(None, {"input": data})[0]
                    self.assertEqual(result.shape, (1, 32, 16))
                    self.assertTrue(np.isfinite(result).all())

    def test_cpu_basic_session_avoids_unsupported_blockwise_integer_matmul_fusion(self):
        model = make_model("uint8", "blockwise")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.onnx"
            onnx.save(model, path)
            session = create_cpu_session(ort, path, log_level=3, optimization="basic")
            result = session.run(None, {"input": np.ones((1, 32, 32), dtype=np.float32)})[0]
            self.assertEqual(result.shape, (1, 32, 16))

    def test_rejects_invalid_activation_scale_and_zero_point(self):
        for extra in (
            ["--activation-type", "uint8", "--activation-zero-point", "256"],
            ["--activation-scale", "nan"],
            ["--output-zero-point", "-1"],
        ):
            with self.subTest(extra=extra), patch.object(
                sys, "argv", ["generate_qdq_matmul_model.py", *extra]
            ), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args()


if __name__ == "__main__":
    unittest.main()
