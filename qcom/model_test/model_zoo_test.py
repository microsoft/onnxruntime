# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import os
from pathlib import Path
from typing import cast, get_args

import pytest
from model_test import BackendT, ModelTestCase, ModelTestDef, ModelTestSuite

MODEL_ZOO_ROOTS = [Path(p) for p in os.getenv("ORT_MODEL_ZOO_TEST_ROOTS", "").split(os.pathsep) if len(p) > 0]
MODEL_ZOO_BACKEND = cast(BackendT, os.getenv("ORT_MODEL_ZOO_BACKEND", "htp"))
assert MODEL_ZOO_BACKEND in get_args(BackendT)
MODEL_ZOO_ENABLE_CONTEXT = os.getenv("ORT_MODEL_ZOO_ENABLE_CONTEXT", "1") == "1"
MODEL_ZOO_ENABLE_CPU_FALLBACK = os.getenv("ORT_MODEL_ZOO_ENABLE_CPU_FALLBACK", "0") == "1"

for model_zoo_root in MODEL_ZOO_ROOTS:
    TEST_DEFS = list(
        ModelTestSuite(
            model_zoo_root,
            backend_type=MODEL_ZOO_BACKEND,
            rtol=None,
            atol=None,
            enable_context=MODEL_ZOO_ENABLE_CONTEXT,
            enable_cpu_fallback=MODEL_ZOO_ENABLE_CPU_FALLBACK,
        ).tests
    )

    TEST_IDS = [str(st) for st in TEST_DEFS]

    @pytest.mark.parametrize("test_def", TEST_DEFS, ids=TEST_IDS)
    def test_models(test_def: ModelTestDef) -> None:
        ModelTestCase(test_def).run()
