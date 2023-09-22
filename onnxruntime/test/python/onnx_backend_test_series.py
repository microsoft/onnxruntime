# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import collections
import json
import os
import platform
import re
import sys
import unittest
from typing import Dict

import numpy as np
import onnx
import onnx.backend.test.case.test_case
import onnx.backend.test.runner
import onnx.defs

import onnxruntime.backend as backend  # pylint: disable=consider-using-from-import

pytest_plugins = ("onnx.backend.test.report",)


class OrtBackendTest(onnx.backend.test.runner.Runner):
    """ONNX test runner with ORT-specific behavior."""

    # pylint: disable=too-few-public-methods
    def __init__(
        self,
        rtol_overrides: Dict[str, float],
        atol_overrides: Dict[str, float],
    ):
        self._rtol_overrides = rtol_overrides
        self._atol_overrides = atol_overrides

        super().__init__(backend, parent_module=__name__)

    @classmethod
    def assert_similar_outputs(cls, ref_outputs, outputs, rtol, atol):
        """Asserts ref_outputs and outputs match to within the given tolerances."""

        def assert_similar_array(ref_output, output):
            np.testing.assert_equal(ref_output.dtype, output.dtype)
            if ref_output.dtype == object:
                np.testing.assert_array_equal(ref_output, output)
            else:
                np.testing.assert_allclose(ref_output, output, rtol=rtol, atol=atol)

        np.testing.assert_equal(len(ref_outputs), len(outputs))
        for i in range(len(outputs)):  # pylint: disable=consider-using-enumerate
            if isinstance(outputs[i], list):
                for j in range(len(outputs[i])):
                    assert_similar_array(ref_outputs[i][j], outputs[i][j])
            else:
                assert_similar_array(ref_outputs[i], outputs[i])

    def _add_model_test(self, model_test: onnx.backend.test.case.test_case.TestCase, kind: str) -> None:
        attrs = {}
        # TestCase changed from a namedtuple to a dataclass in ONNX 1.12.
        # We can just modify t_c.rtol and atol directly once ONNX 1.11 is no longer supported.
        if hasattr(model_test, "_asdict"):
            attrs = model_test._asdict()
        else:
            attrs = vars(model_test)
        attrs["rtol"] = self._rtol_overrides[model_test.name]
        attrs["atol"] = self._atol_overrides[model_test.name]

        super()._add_model_test(onnx.backend.test.case.test_case.TestCase(**attrs), kind)


def apply_filters(filters, category):
    opset_version = f"opset{onnx.defs.onnx_opset_version()}"
    validated_filters = []
    for f in filters[category]:
        if type(f) is list:
            opset_regex = f[0]
            filter_regex = f[1]
            opset_match = re.match(opset_regex, opset_version)
            if opset_match is not None:
                validated_filters.append(filter_regex)
        else:
            validated_filters.append(f)
    return validated_filters


def load_jsonc(basename: str):
    """Returns a deserialized object from the JSONC file in testdata/<basename>."""
    filename = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "testdata",
        basename,
    )
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found {filename!r}.")

    with open(filename, encoding="utf-8") as f:  # pylint: disable=invalid-name
        lines = f.readlines()
    lines = [x.split("//")[0] for x in lines]
    return json.loads("\n".join(lines))


def create_backend_test(test_name=None):
    """Creates an OrtBackendTest and adds its TestCase's to global scope so unittest will find them."""

    overrides = load_jsonc("onnx_backend_test_series_overrides.jsonc")
    rtol_default = overrides["rtol_default"]
    atol_default = overrides["atol_default"]
    rtol_overrides = collections.defaultdict(lambda: rtol_default)
    rtol_overrides.update(overrides["rtol_overrides"])
    atol_overrides = collections.defaultdict(lambda: atol_default)
    atol_overrides.update(overrides["atol_overrides"])

    backend_test = OrtBackendTest(rtol_overrides, atol_overrides)

    # Type not supported
    backend_test.exclude(r"(FLOAT16)")

    if test_name:
        backend_test.include(test_name + ".*")
    else:
        filters = load_jsonc("onnx_backend_test_series_filters.jsonc")
        current_failing_tests = apply_filters(filters, "current_failing_tests")

        if platform.architecture()[0] == "32bit":
            current_failing_tests += apply_filters(filters, "current_failing_tests_x86")

        if backend.supports_device("DNNL"):
            current_failing_tests += apply_filters(filters, "current_failing_tests_DNNL")

        if backend.supports_device("NNAPI"):
            current_failing_tests += apply_filters(filters, "current_failing_tests_NNAPI")

        if backend.supports_device("OPENVINO_GPU_FP32") or backend.supports_device("OPENVINO_GPU_FP16"):
            current_failing_tests += apply_filters(filters, "current_failing_tests_OPENVINO_GPU")

        if backend.supports_device("OPENVINO_CPU_FP32"):
            current_failing_tests += apply_filters(filters, "current_failing_tests_OPENVINO_CPU_FP32")

        if backend.supports_device("OPENVINO_CPU_FP16"):
            current_failing_tests += apply_filters(filters, "current_failing_tests_OPENVINO_CPU_FP16")

        if backend.supports_device("OPENVINO"):
            current_failing_tests += apply_filters(filters, "current_failing_tests_OPENVINO_opset18")

        if backend.supports_device("MIGRAPHX"):
            current_failing_tests += apply_filters(filters, "current_failing_tests_MIGRAPHX")

        # Skip these tests for a "pure" DML onnxruntime python wheel. We keep these tests enabled for instances where both DML and CUDA
        # EPs are available (Windows GPU CI pipeline has this config) - these test will pass because CUDA has higher precedence than DML
        # and the nodes are assigned to only the CUDA EP (which supports these tests)
        if backend.supports_device("DML") and not backend.supports_device("GPU"):
            current_failing_tests += apply_filters(filters, "current_failing_tests_pure_DML")

        filters = (
            current_failing_tests
            + apply_filters(filters, "tests_with_pre_opset7_dependencies")
            + apply_filters(filters, "unsupported_usages")
            + apply_filters(filters, "failing_permanently")
            + apply_filters(filters, "test_with_types_disabled_due_to_binary_size_concerns")
        )

        backend_test.exclude("(" + "|".join(filters) + ")")
        print("excluded tests:", filters)

        # exclude TRT EP temporarily and only test CUDA EP to retain previous behavior
        os.environ["ORT_ONNX_BACKEND_EXCLUDE_PROVIDERS"] = "TensorrtExecutionProvider"

    # import all test cases at global scope to make
    # them visible to python.unittest.
    globals().update(backend_test.enable_report().test_cases)


def parse_args():
    """Returns args parsed from sys.argv."""
    parser = argparse.ArgumentParser(
        os.path.basename(__file__),
        description="Run the ONNX backend tests using ONNXRuntime.",
    )

    # Add an argument to match a single test name, by adding the name to the 'include' filter.
    # Using -k with python unittest (https://docs.python.org/3/library/unittest.html#command-line-options)
    # doesn't work as it filters on the test method name (Runner._add_model_test) rather than individual
    # test case names.
    parser.add_argument(
        "-t",
        "--test-name",
        dest="test_name",
        type=str,
        help="Only run tests that match this value. Matching is regex based, and '.*' is automatically appended",
    )

    # parse just our args. python unittest has its own args and arg parsing, and that runs inside unittest.main()
    parsed, unknown = parser.parse_known_args()
    sys.argv = sys.argv[:1] + unknown

    return parsed


if __name__ == "__main__":
    args = parse_args()

    create_backend_test(args.test_name)
    unittest.main()
