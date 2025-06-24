# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

from pathlib import Path, PurePosixPath

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


def get_test_parameters() -> list[list[list[str]]]:
    test_plan = CTestPlan(
        Path(QDC_HOST_PATH) / ORT_BUILD_CONFIG / "CTestTestfile.cmake",
        str(PurePosixPath(ORT_DEVICE_PATH) / ORT_BUILD_CONFIG),
    ).plan

    return [[tp] for tp in test_plan]


class TestOrt(TestBase):
    @pytest.fixture(scope="session", autouse=True)
    def device_session(self):
        self.prepare_ort_tests()
        yield self
        self.copy_logs()

    @pytest.mark.parametrize(["test_cmd"], get_test_parameters())
    def test_onnxruntime_test_suite(self, test_cmd) -> None:
        Adb().shell([self.__get_test_cmd(test_cmd)])

    def test_onnx_models(self) -> None:
        build_root = Path(ORT_DEVICE_PATH) / ORT_BUILD_CONFIG
        runner_exe = build_root / "onnx_test_runner"

        # fmt: off
        models_dir = (
            build_root / "_deps" / "onnx-src" / "onnx" / "backend" / "test" / "data" / "node"
        )
        test_cmd = [
            str(runner_exe),
            "-j", "1",
            "-e", "qnn",
            "-i", "'backend_type|cpu'",
            str(models_dir),
        ]
        # fmt: on

        Adb().shell([self.__get_test_cmd(test_cmd)])

    @staticmethod
    def __get_test_cmd(test_cmd: list[str]) -> str:
        test_str = " ".join(test_cmd)
        return (
            f"cd {ORT_DEVICE_PATH}/{ORT_BUILD_CONFIG} && "
            f"echo -=-=-=-=-=-=-=-=-=-=- >> {ORT_TEST_RESULTS_DEVICE_LOG} && "
            f"echo Running test: {test_str} >> {ORT_TEST_RESULTS_DEVICE_LOG} && "
            f"env ADSP_LIBRARY_PATH={QNN_ADSP_LIBRARY_PATH} LD_LIBRARY_PATH={QNN_LD_LIBRARY_PATH} "
            f"{test_str} >> {ORT_TEST_RESULTS_DEVICE_LOG} 2>&1"
        )
