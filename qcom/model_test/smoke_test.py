# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import os
from pathlib import Path

import pytest
from model_test import ModelTestCase, ModelTestDef, ModelTestSuite

SMOKE_TESTS = list(
    ModelTestSuite(
        Path(os.getenv("ORT_WHEEL_SMOKE_TEST_ROOT", "HopefullyBogusPath")),
        backend_type="htp",
        rtol=None,
        atol=None,
        enable_context=True,
        enable_cpu_fallback=False,
    ).tests
)

SMOKE_TEST_IDS = [str(st) for st in SMOKE_TESTS]


@pytest.mark.parametrize("test_def", SMOKE_TESTS, ids=SMOKE_TEST_IDS)
def test_models(test_def: ModelTestDef) -> None:
    ModelTestCase(test_def).run()
