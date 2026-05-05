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

        import numpy as np  # noqa: PLC0415

        import onnxruntime.quantization as oq  # noqa: PLC0415

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
            oq.quant_pre_process(onnx_path, preprocessed_path, skip_symbolic_shape=True)
            oq.quantize_static(
                preprocessed_path,
                quantized_path,
                Mock(),
                calibrate_method=oq.CalibrationMethod.Percentile,
                op_types_to_quantize=["Conv", "Mul", "Gemm"],
            )
            assert os.path.exists(preprocessed_path), f"missing output {preprocessed_path!r}"
            assert os.path.exists(quantized_path), f"missing output {quantized_path!r}"

    def test_issue_23268_quantize_static_modelproto_no_validation_error(self):
        # Regression test for https://github.com/microsoft/onnxruntime/issues/23268.
        # Before PR #23322, save_and_reload_model_with_shape_infer() saved the caller's
        # ModelProto with save_as_external_data=True (mutating it to point at a temp
        # path), then deleted that temp dir.  Any subsequent onnx.checker.check_model()
        # call on the original proto therefore raised:
        #   onnx.onnx_cpp2py_export.checker.ValidationError
        # because the referenced external-data file no longer existed.
        # PR #23322 fixed this by deep-copying the proto before saving.
        import numpy as np  # noqa: PLC0415
        import onnx.helper as oh  # noqa: PLC0415
        import onnx.numpy_helper as onp  # noqa: PLC0415
        from onnx import TensorProto  # noqa: PLC0415

        import onnxruntime.quantization as oq  # noqa: PLC0415

        # Build a minimal Add model: output = input + weight.
        # Weight shape [32, 32] float32 => 4096 bytes, which is above the
        # 1024-byte threshold that triggers ONNX external-data serialization
        # inside save_and_reload_model_with_shape_infer.
        weight_data = np.ones((32, 32), dtype=np.float32)
        weight_initializer = onp.from_array(weight_data, name="weight")

        input_vi = oh.make_tensor_value_info("input", TensorProto.FLOAT, [32, 32])
        output_vi = oh.make_tensor_value_info("output", TensorProto.FLOAT, [32, 32])
        add_node = oh.make_node("Add", inputs=["input", "weight"], outputs=["output"])

        graph = oh.make_graph([add_node], "test_graph", [input_vi], [output_vi], [weight_initializer])
        model_proto = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
        model_proto.ir_version = 7

        class MockReader:
            def __init__(self):
                self.i = 0

            def get_next(self):
                if self.i > 0:
                    return None
                self.i += 1
                return {"input": np.ones((32, 32), dtype=np.float32)}

        with tempfile.TemporaryDirectory() as temp:
            output_path = os.path.join(temp, "quantized.onnx")
            # Before the fix this raised ValidationError because the temp dir
            # created inside save_and_reload_model_with_shape_infer was deleted
            # while the mutated proto still referenced it.
            oq.quantize_static(model_proto, output_path, MockReader())
            self.assertTrue(
                os.path.exists(output_path),
                f"Expected quantized model at {output_path!r}",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
