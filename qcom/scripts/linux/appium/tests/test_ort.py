# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import NamedTuple

import pytest
from adb_utils import Adb
from ctest_plan import CTestPlan
from qdc_helpers import (
    ORT_BUILD_CONFIG,
    ORT_DEVICE_PATH,
    ORT_TEST_RESULTS_DEVICE_LOG,
    QDC_HOST_PATH,
    QNN_ADSP_LIBRARY_PATH,
    QNN_LD_LIBRARY_PATH,
    TestBase,
)


class ModelTestDef(NamedTuple):
    name: str
    backend: str
    model_root: Path
    working_dir: Path | None = None
    extra_args: Iterable[str] = []


CTEST_PLAN = CTestPlan(
    Path(QDC_HOST_PATH) / ORT_BUILD_CONFIG / "CTestTestfile.cmake",
    str(PurePosixPath(ORT_DEVICE_PATH) / ORT_BUILD_CONFIG),
)

MODEL_TEST_DEFINITIONS = [
    ModelTestDef(
        "node",
        "cpu",
        Path(ORT_DEVICE_PATH) / ORT_BUILD_CONFIG / "_deps" / "onnx-src" / "onnx" / "backend" / "test" / "data" / "node",
    ),
    ModelTestDef(
        "float32",
        "cpu",
        Path(ORT_DEVICE_PATH) / "model_tests" / "onnx_models" / "testdata" / "float32",
        Path(ORT_DEVICE_PATH) / "model_tests" / "onnx_models",
    ),
    ModelTestDef(
        "qdq",
        "htp",
        Path(ORT_DEVICE_PATH) / "model_tests" / "onnx_models" / "testdata" / "qdq",
        Path(ORT_DEVICE_PATH) / "model_tests" / "onnx_models",
    ),
    ModelTestDef(
        "qdq-with-context-cache",
        "htp",
        Path(ORT_DEVICE_PATH) / "model_tests" / "onnx_models" / "testdata" / "qdq-with-context-cache",
        Path(ORT_DEVICE_PATH) / "model_tests" / "onnx_models",
        ["-f"],
    ),
]

MODEL_TEST_IDS = [m.name for m in MODEL_TEST_DEFINITIONS]


class TestOrt(TestBase):
    @pytest.fixture(scope="session", autouse=True)
    def device_session(self):
        self.prepare_ort_tests()
        yield self
        self.copy_logs()

    @pytest.mark.parametrize(["test_cmd"], [[tp] for tp in CTEST_PLAN.plan], ids=CTEST_PLAN.names)
    def test_onnxruntime_test_suite(self, test_cmd) -> None:
        Adb().shell([self.__get_test_cmd(test_cmd)])

    @pytest.mark.parametrize("test_def", MODEL_TEST_DEFINITIONS, ids=MODEL_TEST_IDS)
    def test_onnx_models(self, test_def: ModelTestDef) -> None:
        build_root = Path(ORT_DEVICE_PATH) / ORT_BUILD_CONFIG
        runner_exe = build_root / "onnx_test_runner"

        # fmt: off
        test_cmd = [
            str(runner_exe),
            "-j", "1",
            "-e", "qnn",
            "-i", f"'backend_type|{test_def.backend}'",
            *test_def.extra_args,
            str(test_def.model_root),
        ]
        # fmt: on

        Adb().shell([self.__get_test_cmd(test_cmd, test_def.working_dir)])

    @staticmethod
    def __get_test_cmd(test_cmd: list[str], working_dir: Path | None = None) -> str:
        if working_dir is None:
            working_dir = Path(ORT_DEVICE_PATH) / ORT_BUILD_CONFIG
        test_str = " ".join(test_cmd)
        return (
            f"set -euo pipefail && "
            f"cd {working_dir} && "
            f"echo -=-=-=-=-=-=-=-=-=-=- >> {ORT_TEST_RESULTS_DEVICE_LOG} && "
            f"echo Running test: {test_str} >> {ORT_TEST_RESULTS_DEVICE_LOG} && "
            f"env ADSP_LIBRARY_PATH={QNN_ADSP_LIBRARY_PATH} LD_LIBRARY_PATH={QNN_LD_LIBRARY_PATH} "
            f"{test_str} 2>&1 | tee -a {ORT_TEST_RESULTS_DEVICE_LOG}"
        )
