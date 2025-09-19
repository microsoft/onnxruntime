# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import logging
import os
import subprocess
import sys
from collections.abc import Collection, Iterable
from pathlib import Path
from typing import Literal

from ..github import is_host_github_runner
from ..task import BashScriptsWithVenvTask, RunExecutablesWithVenvTask
from ..util import REPO_ROOT, git_head_sha, run_and_get_output
from .windows import RunPowershellScriptsTask


class BuildEpLinuxTask(BashScriptsWithVenvTask):
    """Build ONNX Runtime on a Linux host."""

    def __init__(
        self,
        group_name: str | None,
        venv: Path | None,
        target_platform: Literal["android", "linux"],
        target_arch: Literal["aarch64", "aarch64-oe-gcc11.2", "x86_64"],
        qairt_sdk_root: Path | None,
        mode: str,
    ) -> None:
        cmd = [
            str(REPO_ROOT / "qcom" / "scripts" / "linux" / "build.sh"),
            f"--target-arch={target_arch}",
            f"--target-platform={target_platform}",
            f"--mode={mode}",
        ]

        if qairt_sdk_root is not None:
            cmd.append(f"--qairt-sdk-root={qairt_sdk_root}")

        super().__init__(group_name, venv, [cmd], env=ort_build_env_vars())


TargetArchWindowsT = Literal["arm64", "arm64ec", "x86_64"]


class BuildEpWindowsTask(RunPowershellScriptsTask):
    def __init__(
        self,
        group_name: str | None,
        venv: Path | None,
        target_arch: TargetArchWindowsT,
        config: Literal["Debug", "Release", "RelWithDebInfo"],
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

        # When building for ARM64x, we only build the Python bits for ARM64ec
        if build_as_x and target_arch == "arm64":
            target_py_exe = None
        else:
            target_py_exe = self.__target_py_exe(target_arch)

        if target_py_exe is not None:
            cmd.extend(["-TargetPyExe", str(target_py_exe).replace(" ", "` ")])

        super().__init__(group_name, [cmd], env=ort_build_env_vars())

    @staticmethod
    def __target_py_exe(target_arch: TargetArchWindowsT) -> Path | None:
        if target_arch == "arm64":
            try:
                return Path(
                    run_and_get_output(["py", "-3.12-arm64", "-c", "import sys; print(sys.executable)"], quiet=True)
                )
            except subprocess.CalledProcessError:
                logging.warning(f"Could not find native Python for {target_arch}.")
                return None
        return Path(sys.executable)


class QdcTestsTask(RunExecutablesWithVenvTask):
    def __init__(
        self,
        group_name: str | None,
        venv: Path | None,
        platforms: Collection[Literal["android", "windows"]],
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


def ort_build_env_vars() -> dict[str, str]:
    env = os.environ.copy()
    if env.get("ORT_NIGHTLY_BUILD", "0") == "1":
        env["NIGHTLY_BUILD"] = "1"
        env["Build_SourceVersion"] = git_head_sha()
    return env
