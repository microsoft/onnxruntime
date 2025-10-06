# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import functools
import operator
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from ..task import (
    CompositeTask,
    ConditionalTask,
    NoOpTask,
    PyTestTask,
    RunExecutablesTask,
    RunExecutablesWithVenvTask,
    RunInTempDirectoryTask,
    Task,
)
from ..tools import PythonExecutableArchT, get_onnx_models_root, get_python_executable
from ..util import (
    MSFT_CI_REQUIREMENTS_RELPATH,
    REPO_ROOT,
)
from .build import ConfigT, TargetPyVersionT, get_ort_version


def uv_pip_install_cmd(
    requirements: Iterable[Path] = [], packages: Iterable[Path] = [], index_url: str | None = None
) -> list[str]:
    cmd = (
        ["uv", "pip", "install", "--native-tls"]
        + [f"--requirement={r}" for r in requirements]
        + [str(p) for p in packages]
    )
    if index_url is not None:
        url_parts = urlparse(index_url)
        cmd.extend([f"--trusted-host={url_parts.hostname}", f"--index-url={index_url}"])
    return cmd


class CreateOrtVenvTask(CompositeTask):
    def __init__(self, python_executable: Path, venv_path: Path) -> None:
        super().__init__(
            group_name=None,
            tasks=[
                CreateVenvTask(python_executable=python_executable, venv_path=venv_path),
                RunExecutablesWithVenvTask(
                    group_name=f"Installing required packages into {venv_path}",
                    venv=venv_path,
                    executables_and_args=[
                        uv_pip_install_cmd(
                            requirements=[
                                REPO_ROOT / MSFT_CI_REQUIREMENTS_RELPATH,
                                REPO_ROOT / "requirements-dev.txt",
                            ]
                        ),
                        ["lintrunner", "init"],
                        uv_pip_install_cmd(
                            requirements=[
                                REPO_ROOT / "qcom" / "requirements.txt",
                            ],
                            index_url="http://ort-ep-win-01:8080",
                        ),
                    ],
                ),
            ],
        )


class CreateVenvTask(CompositeTask):
    def __init__(self, python_executable: Path, venv_path: Path) -> None:
        super().__init__(
            group_name=f"Creating virtual environment at {venv_path}",
            tasks=[
                ConditionalTask(
                    group_name=None,
                    condition=venv_path.exists,
                    true_task=NoOpTask(),
                    false_task=CompositeTask(
                        group_name=None,
                        tasks=[
                            RunExecutablesTask(
                                group_name=None,
                                executables_and_args=[
                                    [
                                        str(python_executable),
                                        "-m",
                                        "venv",
                                        str(venv_path),
                                    ],
                                ],
                            ),
                            RunExecutablesWithVenvTask(
                                group_name=None,
                                venv=venv_path,
                                executables_and_args=[
                                    [
                                        "python",
                                        "-m",
                                        "pip",
                                        "install",
                                        "pip",
                                        "--upgrade",
                                    ],
                                    ["python", "-m", "pip", "install", "uv"],
                                ],
                            ),
                        ],
                    ),
                ),
            ],
        )


# Valid Windows Portable Executable (PE) types that a wheel can target.
# Note that ARM64ec and ARM64x both require an AMD64 (not ARM64) Python interpreter.
WheelPeArchT = Literal["arm64", "arm64ec", "arm64x"]


class OrtWheelTestTask(RunInTempDirectoryTask):
    def __init__(
        self,
        group_name: str | None,
        build_venv: Path | None,
        wheel_pe_arch: WheelPeArchT,
        py_version: TargetPyVersionT,
        get_wheel: Callable[[], Path],
        test_files_or_dirs: list[str],
        get_test_env: Callable[[], Mapping[str, str]] | None = None,
    ) -> None:
        self.__build_venv = build_venv
        self.__wheel_pe_arch = wheel_pe_arch
        self.__target_py_version: TargetPyVersionT = py_version
        self.__get_wheel = get_wheel
        self.__test_files_or_dirs = test_files_or_dirs
        self.__get_test_env = get_test_env
        super().__init__(group_name, self.make_wheel_test, tmpdir_prefix="py-smoke-test-")

    @property
    def __python_exe_arch(self) -> PythonExecutableArchT:
        if self.__wheel_pe_arch in ["arm64ec", "arm64x"]:
            return "x86_64"
        elif self.__wheel_pe_arch == "arm64":
            return "arm64"
        raise ValueError(f"Unknown wheel PE arch {self.__wheel_pe_arch}.")

    def make_wheel_test(self, tmpdir: Path) -> Task:
        venv_path = tmpdir / "venv"
        python_exe = get_python_executable(self.__build_venv, self.__python_exe_arch, self.__target_py_version)
        test_env = self.__get_test_env() if self.__get_test_env is not None else None

        return CompositeTask(
            None,
            [
                CreateVenvTask(python_exe, venv_path),
                RunExecutablesWithVenvTask(
                    group_name="Installing model test requirements",
                    venv=venv_path,
                    executables_and_args=[
                        uv_pip_install_cmd(
                            requirements=[REPO_ROOT / "qcom" / "model_test" / "requirements.txt"],
                            packages=[self.__get_wheel()],
                        )
                    ],
                ),
                PyTestTask(
                    group_name="Testing wheel",
                    venv=venv_path,
                    env=test_env,
                    files_or_dirs=self.__test_files_or_dirs,
                ),
            ],
        )


class OrtWheelSmokeTestTask(OrtWheelTestTask):
    def __init__(
        self,
        group_name: str | None,
        venv: Path | None,
        wheel_pe_arch: WheelPeArchT,
        config: ConfigT,
        py_version: TargetPyVersionT,
    ) -> None:
        self.__venv = venv
        self.__wheel_pe_arch = wheel_pe_arch
        self.__config = config
        self.__py_version = py_version

        super().__init__(
            group_name,
            venv,
            wheel_pe_arch,
            py_version,
            self.__find_wheel,
            [str(REPO_ROOT / "qcom" / "model_test" / "smoke_test.py")],
            get_test_env=self.__get_test_env,
        )

    def __find_wheel(self) -> Path:
        """
        Finding the wheel is less straightforward than you might think. We have two issues to contend with:
        1. When the task is created, the wheel might not yet exist since a build to produce it hasn't yet been run.
           For that reason, this function is passed to the task.
        2. It's possible that the wheel has a date embedded in its name. For example, if we're in CI and want to run
           a wheel that was built on a different machine, the file was just emplaced here and we don't have a way to
           reliabily predict its name (e.g., if the wheel was built yesterday).
        """
        build_root = REPO_ROOT / "build" / f"windows-{self.__wheel_pe_arch}"
        package_name = "onnxruntime_qnn_qcom_internal"
        py_vsn = f"cp{self.__py_version.replace('.', '')}"
        wheel_arch = "amd64" if self.__wheel_pe_arch in ["arm64ec", "arm64x"] else self.__wheel_pe_arch
        filename_glob = f"{package_name}-{get_ort_version()}*-{py_vsn}-{py_vsn}-win_{wheel_arch}.whl"

        # The wheel has a date in its filename, was produced by a Visual Studio build, or both.
        dist_dirs = [
            build_root / self.__config / "dist",
            build_root / self.__config / self.__config / "dist",
        ]
        found_wheels = sorted(
            functools.reduce(operator.iadd, [list(d.glob(filename_glob)) for d in dist_dirs], []),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )
        if len(found_wheels) == 0:
            raise FileNotFoundError("Could not find onnxruntime wheel.")
        return found_wheels[0]

    def __get_test_env(self) -> Mapping[str, str]:
        """Get an environment that tells the smoke test where to find its model."""
        return {"ORT_WHEEL_SMOKE_TEST_ROOT": str(get_onnx_models_root(self.__venv) / "testdata" / "smoke")}


class RunLinterTask(CompositeTask):
    def __init__(self, venv_path: Path, auto_fix: bool = False) -> None:
        lintrunner_cmd = [
            "lintrunner",
            "--configs",
            f"{REPO_ROOT}/.lintrunner.toml",
            "--force-color",
            "--all-files",
            "-v",
        ] + (["-a"] if auto_fix else [])

        super().__init__(
            group_name="Run source linter",
            tasks=[
                RunExecutablesWithVenvTask(
                    group_name=None,
                    venv=venv_path,
                    executables_and_args=[lintrunner_cmd],
                )
            ],
        )
