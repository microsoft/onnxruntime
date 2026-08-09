# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Runs the ONNX backend test corpus against ONNX Runtime.

This module used to drive ``onnx.backend.test.runner.Runner``, which
dynamically scans the ``onnx/backend/test/data/node`` (and similar)
directories bundled with the upstream ``onnx`` package. That on-disk corpus
can disappear silently between ``onnx`` releases (see onnx#7959) and is not
shipped at all by ``onnx-light``.

Instead, this module consumes the pre-compiled backend test-case registry
exposed by ``onnx-light`` (``onnx_light.onnx.backend.collect_test_case`` /
``make_test_class``), accessed through ``onnxruntime._onnx_shim`` so this
keeps working with whichever ONNX backend implements that registry. Requires
``onnx-light`` (``pip install onnxruntime[onnx-light]``, ``USE_OPTIM_ONNX=1``);
see the module docstring of ``onnxruntime._onnx_shim`` for details.
"""

import argparse
import collections
import json
import os
import platform
import re
import sys
import unittest

import onnxruntime as ort
from onnxruntime._onnx_shim.onnx import defs

try:
    from onnxruntime._onnx_shim.onnx.backend import collect_test_case, make_test_class

    _BACKEND_IMPORT_ERROR = None
except ImportError as _exc:  # pragma: no cover - exercised only without onnx-light
    collect_test_case = None
    make_test_class = None
    _BACKEND_IMPORT_ERROR = _exc

# Minimum number of ONNX "node" test cases that must be discovered in the onnx-light backend
# test registry. This is the Python-leg twin of the C++ onnx_test_runner -m floor and the CMake
# ORT_ONNX_NODE_MIN_CASES gate; all three MUST move in lockstep if the floor is ever raised. It
# guards against the node corpus silently shrinking (e.g. a future onnx-light release dropping
# cases). Today's corpus is ~2100 node cases; 1500 leaves ample headroom.
MIN_NODE_CASES = 1500

# Legacy exclusion kept for parity with the previous onnx.backend.test.runner-based series:
# (b)float16 support was historically incomplete enough that these cases were always excluded,
# independent of any -t/--test-name filtering below. Made case-insensitive so it also covers the
# "bfloat16" cases contributed by onnx-light's registry (the classic corpus only ever had
# upper-case "FLOAT16" names).
#
# The onnx-light registry additionally contributes ~1500 "test_cc_*"-prefixed cases (its own
# C++-implemented extra corpus: training ops such as Adam/momentum, newer opset-24 Attention
# variants, bfloat16/float8 dtype coverage, etc.) that never existed in the classic onnx test
# corpus the onnx_backend_test_series_filters.jsonc allow/deny lists were curated against. They
# are excluded by default here to keep this migration's coverage equivalent to the previous
# series; vetting and enabling them against ORT is tracked separately.
ALWAYS_EXCLUDED_TESTS = [r"(?i)(FLOAT16)", r"^test_cc_"]

# Mirrors OnnxRuntimeBackend.allowReleasedOpsetsOnly (see onnxruntime/python/backend/backend.py):
# by default, models stamped with an onnx opset still under development are skipped rather than
# treated as hard failures, since ORT only guarantees behavior for officially released opsets.
_ALLOW_RELEASED_ONNX_OPSET_ONLY = os.getenv("ALLOW_RELEASED_ONNX_OPSET_ONLY", "1") == "1"
_UNRELEASED_OPSET_MESSAGE = "is under development and support for this is limited"


def _supports_device(device: str) -> bool:
    """Returns whether ONNX Runtime was compiled with support for *device*.

    Mirrors ``onnxruntime.backend.backend.OnnxRuntimeBackend.supports_device`` without
    depending on the ``onnxruntime.backend`` wrapper module, which imports upstream-``onnx``-only
    submodules (``onnx.backend.base``, ``onnx.version``) that ``onnx-light`` does not ship.
    """
    if device == "CUDA":
        device = "GPU"
    current = ort.get_device()
    return "-" + device in current or device + "-" in current or device == current


def _run_with_ort(model, *inputs):
    """Runs an onnx-light backend ``TestCase`` model with ONNX Runtime.

    This is the callable signature expected by
    ``onnxruntime._onnx_shim.onnx.backend.make_test_class``: it is invoked as
    ``rt(model, *inputs)`` and must return the list of output arrays.
    """
    excluded_providers = os.getenv("ORT_ONNX_BACKEND_EXCLUDE_PROVIDERS", default="").split(",")
    providers = [p for p in ort.get_available_providers() if p not in excluded_providers]
    model_bytes = model.SerializeToString()
    try:
        sess = ort.InferenceSession(model_bytes, providers=providers)
    except Exception as exc:
        if _ALLOW_RELEASED_ONNX_OPSET_ONLY and _UNRELEASED_OPSET_MESSAGE in str(exc):
            raise unittest.SkipTest(
                "Skipping this test as only released onnx opsets are supported. "
                f"To run this test set env variable ALLOW_RELEASED_ONNX_OPSET_ONLY to 0. {exc}"
            ) from exc
        raise
    input_names = [i.name for i in sess.get_inputs()]
    feeds = dict(zip(input_names, inputs, strict=False))
    return sess.run(None, feeds)


def _strip_device_suffix(pattern: str) -> str:
    """Strips a trailing literal ``_cpu``/``_cuda`` device suffix from a filter regex.

    The jsonc filters were curated against the previous onnx.backend.test.runner-based series,
    which generated one test per (test case, device backend) pair, named e.g.
    ``test_foo_cpu``/``test_foo_cuda``. The onnx-light registry instead exposes a single
    ``test_foo`` case run against whichever providers ``_run_with_ort`` configures, so those
    patterns would otherwise silently stop matching anything. None of the patterns anchor their
    end (no trailing ``$``), so shortening them here is safe for the ``re.search``-based matching
    ``make_test_class`` performs.
    """
    for suffix in ("_cpu", "_cuda"):
        if pattern.endswith(suffix):
            return pattern[: -len(suffix)]
    return pattern


def apply_filters(filters, category):
    opset_version = f"opset{defs.onnx_opset_version()}"
    validated_filters = []
    for f in filters[category]:
        if type(f) is list:
            opset_regex = f[0]
            filter_regex = f[1]
            opset_match = re.match(opset_regex, opset_version)
            if opset_match is not None:
                validated_filters.append(_strip_device_suffix(filter_regex))
        else:
            validated_filters.append(_strip_device_suffix(f))
    return validated_filters


def load_jsonc(basename: str):
    """Returns a deserialized object from the JSONC file in testdata/<basename>."""
    filenames = [
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "testdata", basename),
        os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", "test", "testdata", basename)),
    ]

    filtered = [f for f in filenames if os.path.exists(f)]
    if not filtered:
        raise FileNotFoundError(f"No file found in {filenames!r}.")

    filename = filtered[0]
    with open(filename, encoding="utf-8") as f:  # pylint: disable=invalid-name
        lines = f.readlines()
    lines = [x.split("//")[0] for x in lines]
    return json.loads("\n".join(lines))


def create_backend_test(test_name=None):
    """Builds the ``OnnxBackendTest`` unittest.TestCase class from the onnx-light registry."""
    if _BACKEND_IMPORT_ERROR is not None:
        raise unittest.SkipTest(
            "The onnx-light backend test registry is unavailable "
            "(onnxruntime._onnx_shim.onnx.backend.collect_test_case/make_test_class could not be "
            f"imported): {_BACKEND_IMPORT_ERROR}. Install onnx-light "
            "(pip install onnxruntime[onnx-light]) and set USE_OPTIM_ONNX=1 to run this test."
        )

    overrides = load_jsonc("onnx_backend_test_series_overrides.jsonc")
    rtol_default = overrides["rtol_default"]
    atol_default = overrides["atol_default"]
    rtol_overrides = collections.defaultdict(lambda: rtol_default)
    rtol_overrides.update(overrides["rtol_overrides"])
    atol_overrides = collections.defaultdict(lambda: atol_default)
    atol_overrides.update(overrides["atol_overrides"])

    all_tests = collect_test_case()

    # Consumption-point floor (full runs only; a targeted -t/test_name run intentionally collects
    # a subset). Fires if the node corpus failed to materialize or has otherwise silently shrunk.
    node_case_count = sum(1 for tc in all_tests.values() if tc.kind == "node")
    if not test_name and node_case_count < MIN_NODE_CASES:
        raise RuntimeError(
            f"Node test corpus collapsed -- discovered only {node_case_count} ONNX 'node' test "
            f"case(s) in the onnx-light backend registry, but at least {MIN_NODE_CASES} are "
            f"required. The registry appears missing, empty, or truncated."
        )

    # Apply the project's default tolerances (and any per-test overrides) to every discovered
    # test case, matching the previous onnx.backend.test.runner-based behavior.
    atols = {name: atol_overrides[name] for name in all_tests}
    rtols = {name: rtol_overrides[name] for name in all_tests}

    if test_name:
        include_regex = [test_name + ".*"]
        exclude_regex = list(ALWAYS_EXCLUDED_TESTS)
    else:
        filters = load_jsonc("onnx_backend_test_series_filters.jsonc")
        current_failing_tests = apply_filters(filters, "current_failing_tests")

        if platform.architecture()[0] == "32bit":
            current_failing_tests += apply_filters(filters, "current_failing_tests_x86")

        if _supports_device("DNNL"):
            current_failing_tests += apply_filters(filters, "current_failing_tests_DNNL")

        if _supports_device("NNAPI"):
            current_failing_tests += apply_filters(filters, "current_failing_tests_NNAPI")

        if _supports_device("OPENVINO_GPU"):
            current_failing_tests += apply_filters(filters, "current_failing_tests_OPENVINO_GPU")

        if _supports_device("OPENVINO_CPU"):
            current_failing_tests += apply_filters(filters, "current_failing_tests_OPENVINO_CPU_FP32")
            current_failing_tests += apply_filters(filters, "current_failing_tests_OPENVINO_CPU_FP16")

        if _supports_device("OPENVINO_NPU"):
            current_failing_tests += apply_filters(filters, "current_failing_tests_OPENVINO_NPU")

        if _supports_device("OPENVINO"):
            current_failing_tests += apply_filters(filters, "current_failing_tests_OPENVINO_opset18")

        if _supports_device("MIGRAPHX"):
            current_failing_tests += apply_filters(filters, "current_failing_tests_MIGRAPHX")

        if _supports_device("WEBGPU"):
            current_failing_tests += apply_filters(filters, "current_failing_tests_WEBGPU")

        if _supports_device("QNN"):
            current_failing_tests += apply_filters(filters, "current_failing_tests_QNN")

        # Skip these tests for a "pure" DML onnxruntime python wheel. We keep these tests enabled for instances where both DML and CUDA
        # EPs are available (Windows GPU CI pipeline has this config) - these test will pass because CUDA has higher precedence than DML
        # and the nodes are assigned to only the CUDA EP (which supports these tests)
        if _supports_device("DML") and not _supports_device("GPU"):
            current_failing_tests += apply_filters(filters, "current_failing_tests_pure_DML")

        exclude_regex = (
            ALWAYS_EXCLUDED_TESTS
            + current_failing_tests
            + apply_filters(filters, "tests_with_pre_opset7_dependencies")
            + apply_filters(filters, "unsupported_usages")
            + apply_filters(filters, "failing_permanently")
            + apply_filters(filters, "test_with_types_disabled_due_to_binary_size_concerns")
        )
        include_regex = None
        print("excluded tests:", exclude_regex)

        # exclude TRT EP temporarily and only test CUDA EP to retain previous behavior
        os.environ["ORT_ONNX_BACKEND_EXCLUDE_PROVIDERS"] = "TensorrtExecutionProvider"

    # Build the discoverable unittest.TestCase class and expose it at module scope so
    # unittest.main() (and pytest) can find it.
    globals()["OnnxBackendTest"] = make_test_class(
        _run_with_ort,
        include_regex=include_regex,
        exclude_regex=exclude_regex,
        atols=atols,
        rtols=rtols,
    )


def parse_args():
    """Returns args parsed from sys.argv."""
    parser = argparse.ArgumentParser(
        os.path.basename(__file__),
        description="Run the ONNX backend tests using ONNXRuntime.",
    )

    # Add an argument to match a single test name, by adding the name to the 'include' filter.
    # Using -k with python unittest (https://docs.python.org/3/library/unittest.html#command-line-options)
    # doesn't work as it filters on the test method name (make_test_class's dynamic test_<name> method)
    # rather than individual test case names.
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

    try:
        create_backend_test(args.test_name)
    except unittest.SkipTest as skip_exc:
        # create_backend_test() can raise this before any unittest.TestCase exists to attach the
        # skip to (e.g. the onnx-light backend registry is unavailable). Report it the same way
        # `unittest`/pytest report a skipped run, instead of letting it look like a crash.
        print(f"SKIP: {skip_exc}", file=sys.stderr)
        sys.exit(0)
    unittest.main()
