#!/usr/bin/env python3
"""
Test for Python 3.14+ exception handling compatibility in ONNX Runtime.
Specifically tests the fix for issue #27392 where type conversion errors
were causing segfaults instead of proper RuntimeError exceptions.
"""

import sys
import unittest

import numpy as np
from helper import get_name

import onnxruntime as onnxrt


class TestPython314ExceptionHandling(unittest.TestCase):
    """Tests for Python 3.14+ exception handling compatibility in type conversion."""

    def setUp(self):
        """Set up test environment."""
        self.python_version = sys.version_info
        self.is_python_314_plus = self.python_version >= (3, 14)

    def test_dict_vectorizer_exception_handling(self):
        """Test that dict vectorizer properly handles type errors without segfaults."""
        if not self.is_python_314_plus:
            self.skipTest("Python 3.14+ specific test")

        # This test reproduces the exact scenario from issue #27392
        sess = onnxrt.InferenceSession(
            get_name("pipeline_vectorize.onnx"),
            providers=onnxrt.get_available_providers(),
        )

        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name

        # Valid input that should work
        x_valid = {0: 25.0, 1: 5.13, 2: 0.0, 3: 0.453, 4: 5.966}

        # This should work without issues
        res = sess.run([output_name], {input_name: x_valid})
        self.assertIsNotNone(res)

        # Create invalid input with string key (this was causing segfaults)
        x_invalid = x_valid.copy()
        x_invalid["string_key"] = 5.6  # This should trigger the type error

        # This should raise RuntimeError, NOT cause a segfault
        with self.assertRaises(RuntimeError) as cm:
            sess.run([output_name], {input_name: x_invalid})

        error_msg = str(cm.exception)
        self.assertIn("Unexpected key type", error_msg)
        self.assertIn("str", error_msg)
        self.assertIn("int64_t", error_msg)

    def test_multiple_invalid_key_types(self):
        """Test various invalid key types that should raise exceptions properly."""
        if not self.is_python_314_plus:
            self.skipTest("Python 3.14+ specific test")

        sess = onnxrt.InferenceSession(
            get_name("pipeline_vectorize.onnx"),
            providers=onnxrt.get_available_providers(),
        )

        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name

        # Test different invalid key types
        invalid_inputs = [
            {"string": 1.0},  # string key
            {1.5: 1.0},  # float key
            {None: 1.0},  # None key
            {(1, 2): 1.0},  # tuple key
        ]

        for i, invalid_input in enumerate(invalid_inputs):
            with self.subTest(case=i, input_type=type(list(invalid_input.keys())[0]).__name__):
                with self.assertRaises((RuntimeError, TypeError, onnxrt.capi.onnxruntime_pybind11_state.InvalidArgument)) as cm:
                    sess.run([output_name], {input_name: invalid_input})

                # Should get a proper exception, not a segfault
                error_msg = str(cm.exception)
                # Either our type error or a different validation error is fine,
                # as long as it doesn't segfault
                self.assertTrue(len(error_msg) > 0)

    def test_exception_during_cleanup(self):
        """Test that exceptions during object cleanup don't cause issues."""
        if not self.is_python_314_plus:
            self.skipTest("Python 3.14+ specific test")

        sess = onnxrt.InferenceSession(
            get_name("pipeline_vectorize.onnx"),
            providers=onnxrt.get_available_providers(),
        )

        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name

        # Run multiple iterations to stress-test cleanup
        for i in range(10):
            invalid_input = {f"invalid_key_{i}": float(i)}

            with self.assertRaises((RuntimeError, onnxrt.capi.onnxruntime_pybind11_state.InvalidArgument)):
                sess.run([output_name], {input_name: invalid_input})

        # If we get here without segfaulting, the fix is working

    def test_backward_compatibility(self):
        """Test that the fix doesn't break functionality on older Python versions."""
        sess = onnxrt.InferenceSession(
            get_name("pipeline_vectorize.onnx"),
            providers=onnxrt.get_available_providers(),
        )

        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name

        # Valid input should work on all Python versions
        x_valid = {0: 25.0, 1: 5.13, 2: 0.0, 3: 0.453, 4: 5.966}
        res = sess.run([output_name], {input_name: x_valid})

        output_expected = np.array([[49.752754]], dtype=np.float32)
        np.testing.assert_allclose(res[0], output_expected, rtol=1e-05, atol=1e-08)

        # Invalid input should still raise RuntimeError
        x_invalid = x_valid.copy()
        x_invalid["invalid"] = 5.6

        with self.assertRaises((RuntimeError, onnxrt.capi.onnxruntime_pybind11_state.InvalidArgument)) as cm:
            sess.run([output_name], {input_name: x_invalid})

        # Error message should be consistent
        error_msg = str(cm.exception)
        self.assertIn("Unexpected key type", error_msg)


if __name__ == "__main__":
    unittest.main()
