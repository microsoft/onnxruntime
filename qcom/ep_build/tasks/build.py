# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import logging
import os
import shutil
from collections.abc import Collection, Iterable, Mapping
from pathlib import Path
from typing import Literal

from ..github import is_host_github_runner
from ..task import (
    BashScriptsWithVenvTask,
    CompositeTask,
    CopyFileTask,
    ExtractArchiveTask,
    PyTestTask,
    RemovePathsTask,
    RunExecutablesWithVenvTask,
    RunInTempDirectoryTask,
)
from ..typing import BuildConfigT, TargetArchLinuxT, TargetArchWindowsT, TargetPyVersionT
from ..util import REPO_ROOT, git_head_sha
from .docker import DOCKER_REPO_ROOT, MANYLINUX_2_34_AARCH64_TAG, DockerBuildAndTestTask
from .windows import RunPowershellScriptsTask


def get_ort_version() -> str:
    return (REPO_ROOT / "VERSION_NUMBER").read_text().strip()


class BuildEpDockerTask(CompositeTask):
    """Build ONNX Runtime for Linux inside a Docker container."""

    def __init__(
        self,
        group_name: str | None,
        target_arch: TargetArchLinuxT,
        config: BuildConfigT,
        target_py_version: TargetPyVersionT | None,
        qairt_sdk_root: Path | None,
        ccache_root: Path | None,
    ) -> None:
        dist_rel_dir = Path("build") / f"linux-{target_arch}" / config / "dist"

        super().__init__(
            group_name,
            [
                RemovePathsTask(
                    "Deleting wheels to workaround ORT build bug",
                    (REPO_ROOT / dist_rel_dir).glob("*.whl"),
                ),
                DockerBuildAndTestTask(
                    "Building ONNX Runtime inside a container",
                    ["_build_ort_linux_aarch64_manylinux_2_34"],
                    target_py_version,
                    MANYLINUX_2_34_AARCH64_TAG,
                    volumes={REPO_ROOT: DOCKER_REPO_ROOT},
                    venv_path=DOCKER_REPO_ROOT / "build" / "venv.build",
                    qairt_sdk_root=qairt_sdk_root,
                    ccache_root=ccache_root,
                ),
            ],
        )


class BuildEpLinuxTask(BashScriptsWithVenvTask):
    """Build ONNX Runtime on a Linux host."""

    def __init__(
        self,
        group_name: str | None,
        venv: Path | None,
        target_platform: Literal["android", "linux"],
        target_arch: TargetArchLinuxT,
        config: BuildConfigT,
        target_py_version: TargetPyVersionT | None,
        qairt_sdk_root: Path | None,
        mode: str,
        extra_args: Iterable[str] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        cmd = [
            str(REPO_ROOT / "qcom" / "scripts" / "linux" / "build.sh"),
            f"--target-arch={target_arch}",
            f"--target-platform={target_platform}",
            f"--config={config}",
            f"--mode={mode}",
        ]

        if target_py_version is not None:
            cmd.append(f"--target-py-version={target_py_version}")

        if qairt_sdk_root is not None:
            cmd.append(f"--qairt-sdk-root={qairt_sdk_root}")

        if extra_args is not None:
            cmd.extend(extra_args)

        super().__init__(group_name, venv, [cmd], env=ort_build_env_vars(env))


class BuildEpWindowsTask(RunPowershellScriptsTask):
    def __init__(
        self,
        group_name: str | None,
        venv: Path | None,
        target_arch: TargetArchWindowsT,
        config: BuildConfigT,
        target_py_version: TargetPyVersionT | None,
        qairt_sdk_root: Path | None,
        mode: str,
        build_as_x: bool = False,
    ) -> None:
        cmd = [
            str(REPO_ROOT / "qcom" / "scripts" / "windows" / "build.ps1"),
            "-Arch",
            target_arch,
            "-Config",
            config,
            "-Mode",
            mode,
        ]

        if venv is not None:
            cmd.extend(["-PyVEnv", str(venv).replace(" ", "` ")])
        if qairt_sdk_root is not None:
            cmd.extend(["-QairtSdkRoot", str(qairt_sdk_root).replace(" ", "` ")])

        if build_as_x:
            cmd.extend(["-BuildAsX", "1"])

        if target_py_version is not None:
            cmd.extend(["-TargetPyVersion", str(target_py_version)])

        super().__init__(group_name, [cmd], env=ort_build_env_vars())


class AdbTestsTask(RunInTempDirectoryTask):
    def __init__(
        self,
        group_name: str | None,
        venv: Path | None,
        platform: Literal["android", "linux"],
        target_arch: Literal["aarch64", "aarch64_manylinux_2_34", "aarch64_oe_gcc11_2"],
    ) -> None:
        self.__venv = venv
        self.__platform = platform
        self.__target_arch = target_arch
        super().__init__(group_name, self.make_test_task, "AdbTests-")

    # This is a pretty slow way to do this, but it's easy to implement
    # and essentially free to maintain. If you find yourself using this
    # often enough that your life would be better if we didn't roundtrip
    # through a zip file, please open a Jira and we'll invest more here.
    def make_test_task(self, tmpdir: Path) -> CompositeTask:
        # Local import to avoid circular dependency
        from ..tools import get_onnx_models_root  # noqa: PLC0415

        env = dict(os.environ)
        env["QDC_TEST_ROOT"] = str(tmpdir)

        # The QDC test harness assumes that we have ONNX model tests in a directory
        # called <mumble>/model_tests/onnx_models and that neither is a symlink.
        onnx_models_package_content = get_onnx_models_root(self.__venv)
        model_test_root = tmpdir / "model_tests"
        model_test_root.mkdir(parents=True)
        onnx_models_root = model_test_root / "onnx_models"
        logging.debug(f"Copying ONNX models to {onnx_models_root}")
        shutil.copytree(onnx_models_package_content, onnx_models_root)
        env["MODEL_TEST_ROOT"] = str(model_test_root)

        test_archive_ext = "zip" if self.__platform == "android" else "tar.bz2"

        if "ORT_TEST_CONFIG_PATH" in os.environ:
            test_config_src = Path(os.environ["ORT_TEST_CONFIG_PATH"])
        else:
            test_config_src = (
                REPO_ROOT
                / "qcom"
                / "scripts"
                / "linux"
                / "appium"
                / "configs"
                / f"{self.__platform}-{self.__target_arch}.jsonc"
            )

        return CompositeTask(
            group_name=None,
            tasks=[
                CopyFileTask(
                    "Copying test config file",
                    test_config_src,
                    tmpdir / "test_config.jsonc",
                ),
                ExtractArchiveTask(
                    "Extracting ONNX Runtime test package",
                    REPO_ROOT
                    / "build"
                    / f"onnxruntime-tests-{self.__platform}-{self.__target_arch}.{test_archive_ext}",
                    tmpdir,
                ),
                PyTestTask(
                    "Testing ONNX Runtime with a local device",
                    self.__venv,
                    ["tests"],
                    env=env,
                    cwd=REPO_ROOT / "qcom" / "scripts" / "linux" / "appium",
                ),
            ],
        )


class QdcTestsTask(RunExecutablesWithVenvTask):
    def __init__(
        self,
        group_name: str | None,
        venv: Path | None,
        platforms: Collection[Literal["android", "qualcomm_linux", "windows"]],
        extra_args: Iterable[str] | None = None,
    ) -> None:
        if "QDC_API_TOKEN" not in os.environ:
            raise RuntimeError("QDC_API_TOKEN must be set in the environment to run tests on QDC.")

        cmd = [
            "python",
            str(REPO_ROOT / "qcom" / "scripts" / "all" / "qdc_runner.py"),
            f"--log-dir={REPO_ROOT / 'build' / 'qdc-%p'}",  # %p is expanded to "android" or "windows"
        ]

        if len(platforms) > 0:
            cmd.extend(["--enable-platforms", *platforms])

        if extra_args is not None:
            cmd.extend(extra_args)

        if is_host_github_runner():
            actor = os.environ["GITHUB_ACTOR"]
            branch = os.environ["GITHUB_REF_NAME"]
            cmd.append(f"--name={actor}-{branch}")
            cmd.append(f"--on-behalf-of={actor}")

        super().__init__(group_name, venv, [cmd])


def ort_build_env_vars(old_env: Mapping[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy() if old_env is None else dict(old_env)
    if env.get("ORT_NIGHTLY_BUILD", "0") == "1":
        env["NIGHTLY_BUILD"] = "1"
        env["Build_SourceVersion"] = git_head_sha()
    return env
