# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

from pathlib import Path, PurePosixPath

import pytest
from adb_utils import Adb
from ctest_plan import CTestPlan
from qdc_helpers import (
    ORT_BUILD_CONFIG,
    ORT_DEVICE_PATH,
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
        cmd = (
            f"cd {ORT_DEVICE_PATH}/{ORT_BUILD_CONFIG} && "
            f"env ADSP_LIBRARY_PATH={QNN_ADSP_LIBRARY_PATH} LD_LIBRARY_PATH={QNN_LD_LIBRARY_PATH} "
            f"{' '.join(test_cmd)}"
        )
        adb = Adb()
        adb.shell([cmd])
