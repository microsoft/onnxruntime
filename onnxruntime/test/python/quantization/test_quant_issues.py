# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import tempfile
import unittest
import warnings


def ignore_warnings(warns):
    """
    Catches warnings.

    :param warns:   warnings to ignore
    """

    def wrapper(fct):
        if warns is None:
            raise AssertionError(f"warns cannot be None for '{fct}'.")

        def call_f(self):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", warns)
                return fct(self)

        return call_f

    return wrapper


class TestQuantIssues(unittest.TestCase):
    @ignore_warnings(DeprecationWarning)
    def test_minimal_model(self):
        folder = os.path.join(os.path.dirname(__file__), "..", "..", "testdata")
        onnx_path = os.path.join(folder, "qdq_minimal_model.onnx")
        if not os.path.exists(onnx_path):
            # The file does seem to be the same location in every CI job.
            raise unittest.SkipTest("unable to find {onnx_path!r}")

        import numpy as np

        import onnxruntime.quantization as oq

        class Mock:
            def __init__(self):
                self.i = 0

            def get_next(self):
                if self.i > 10:
                    return None
                self.i += 1
                return {"input": np.random.randint(0, 255, size=(1, 3, 32, 32), dtype=np.uint8)}

        with tempfile.TemporaryDirectory() as temp:
            preprocessed_path = os.path.join(temp, "preprocessed.onnx")
            quantized_path = os.path.join(temp, "quantized.onnx")
            oq.quant_pre_process(onnx_path, preprocessed_path, enable_symbolic_shape=False)
            oq.quantize_static(
                preprocessed_path,
                quantized_path,
                Mock(),
                calibrate_method=oq.CalibrationMethod.Percentile,
                op_types_to_quantize=["Conv", "Mul", "Gemm"],
            )
            assert os.path.exists(preprocessed_path), f"missing output {preprocessed_path!r}"
            assert os.path.exists(quantized_path), f"missing output {quantized_path!r}"


if __name__ == "__main__":
    unittest.main(verbosity=2)
