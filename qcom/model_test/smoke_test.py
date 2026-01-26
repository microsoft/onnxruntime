# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import os
from pathlib import Path

import pytest
from model_test import ModelTestCase, ModelTestDef, ModelTestSuite
from model_zoo_test import get_xfails

import onnxruntime

SMOKE_TESTS = list(
    ModelTestSuite(
        Path(os.getenv("ORT_WHEEL_SMOKE_TEST_ROOT", "HopefullyBogusPath")),
        backend_type="htp",
        rtol=None,
        atol=None,
        cosine_similarity=None,
        enable_context=True,
        enable_cpu_fallback=False,
    ).tests
)

SMOKE_TEST_IDS = [str(st) for st in SMOKE_TESTS]


@pytest.mark.parametrize("test_def", SMOKE_TESTS, ids=SMOKE_TEST_IDS)
def test_models(test_def: ModelTestDef) -> None:
    xfails = get_xfails("ORT_WHEEL_SMOKE_TEST_XFAILS")
    if test_def.model_root.name in xfails:
        pytest.xfail(xfails[test_def.model_root.name])
    ModelTestCase(test_def).run()


def test_qnn_version() -> None:
    assert onnxruntime.capi.build_and_package_info.qnn_version is not None
