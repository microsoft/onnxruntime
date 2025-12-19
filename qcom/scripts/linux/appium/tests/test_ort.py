# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import tempfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import NamedTuple

import pytest
from ctest_plan import CTestPlan
from qdc_helpers import TestBase


class ModelTestDef(NamedTuple):
    name: str
    backend: str
    model_root: Path
    working_dir: Path | None = None
    extra_args: Iterable[str] = []


CONFIG = TestBase.config()

CTEST_PLAN = CTestPlan(
    Path(CONFIG.host_build_root) / "CTestTestfile.cmake",
    str(PurePosixPath(CONFIG.device_build_root)),
)

MODEL_TEST_DEFINITIONS = [
    ModelTestDef(
        "node",
        "cpu",
        Path(CONFIG.device_runtime_path)
        / "cmake"
        / "external"
        / "onnx"
        / "onnx"
        / "backend"
        / "test"
        / "data"
        / "node",
    ),
    ModelTestDef(
        "float32",
        "cpu",
        Path(CONFIG.device_onnx_model_test_path) / "testdata" / "float32",
        Path(CONFIG.device_onnx_model_test_path),
    ),
    ModelTestDef(
        "qdq",
        "htp",
        Path(CONFIG.device_onnx_model_test_path) / "testdata" / "qdq",
        Path(CONFIG.device_onnx_model_test_path),
    ),
    ModelTestDef(
        "qdq-with-context-cache",
        "htp",
        Path(CONFIG.device_onnx_model_test_path) / "testdata" / "qdq-with-context-cache",
        Path(CONFIG.device_onnx_model_test_path),
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
        self.__assert_passes(self.__get_test_cmd(test_cmd))

    @pytest.mark.parametrize("test_def", MODEL_TEST_DEFINITIONS, ids=MODEL_TEST_IDS)
    def test_onnx_models(self, test_def: ModelTestDef) -> None:
        runner_exe = Path(CONFIG.device_build_root) / "onnx_test_runner"

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

        self.__assert_passes(self.__get_test_cmd(test_cmd, test_def.working_dir))

    def __assert_passes(self, test_cmd: str) -> None:
        self.device.shell([test_cmd])
        with tempfile.TemporaryDirectory(prefix="TestRc-") as tmpdir:
            rc_path = Path(tmpdir) / "rc.txt"
            self.device.pull(self.__rc_device_path, rc_path)
            rc = rc_path.read_text().splitlines()
            assert len(rc) == 1
            assert rc[0] == "0", f"Test command returned non-zero value {rc}."

    def __get_test_cmd(
        self,
        test_cmd: list[str],
        working_dir: Path | None = None,
    ) -> str:
        if working_dir is None:
            working_dir = Path(CONFIG.device_build_root)
        test_str = " ".join(test_cmd)
        return (
            f"cd {working_dir} && "
            f"echo -=-=-=-=-=-=-=-=-=-=- >> {CONFIG.test_results_device_log} && "
            f"echo Running test: {test_str} >> {CONFIG.test_results_device_log} && "
            f"(env ADSP_LIBRARY_PATH={CONFIG.device_adsp_library_path} LD_LIBRARY_PATH={CONFIG.device_ld_library_path} "
            f"{test_str}; echo $? > {self.__rc_device_path}) 2>&1 | tee -a {CONFIG.test_results_device_log}"
        )

    @property
    def __rc_device_path(self) -> Path:
        return Path(CONFIG.device_results_root) / "rc.txt"
