# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import os
from pathlib import Path

import pytest
from model_test import ModelTestCase, ModelTestDef, ModelTestSuite

MODEL_ZOO_ROOTS = [Path(p) for p in os.getenv("ORT_MODEL_ZOO_TEST_ROOTS", "").split(os.pathsep) if len(p) > 0]


for model_zoo_root in MODEL_ZOO_ROOTS:
    TEST_DEFS = list(
        ModelTestSuite(
            model_zoo_root,
            backend_type="htp",
            rtol=None,
            atol=None,
            enable_context=True,
            enable_cpu_fallback=False,
        ).tests
    )

    TEST_IDS = [str(st) for st in TEST_DEFS]

    @pytest.mark.parametrize("test_def", TEST_DEFS, ids=TEST_IDS)
    def test_models(test_def: ModelTestDef) -> None:
        ModelTestCase(test_def).run()
