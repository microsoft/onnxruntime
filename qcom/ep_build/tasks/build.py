# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import os
from collections.abc import Collection, Iterable
from pathlib import Path
from typing import Literal

from ..github import is_host_github_runner
from ..task import BashScriptsWithVenvTask, RunExecutablesWithVenvTask
from ..util import REPO_ROOT, git_head_sha
from .windows import RunPowershellScriptsTask


class BuildEpLinuxTask(BashScriptsWithVenvTask):
    """Build ONNX Runtime on a Linux host."""

    def __init__(
        self,
        group_name: str | None,
        venv: Path | None,
        target_platform: Literal["android", "linux"],
        qairt_sdk_root: Path | None,
        mode: str,
    ) -> None:
        cmd = [
            str(REPO_ROOT / "qcom" / "scripts" / "linux" / "build.sh"),
            f"--target-platform={target_platform}",
            f"--mode={mode}",
        ]

        if qairt_sdk_root is not None:
            cmd.append(f"--qairt-sdk-root={qairt_sdk_root}")

        super().__init__(group_name, venv, [cmd], env=ort_build_env_vars())


class BuildEpWindowsTask(RunPowershellScriptsTask):
    def __init__(
        self,
        group_name: str | None,
        venv: Path | None,
        target_platform: Literal["android", "windows"],
        arch: Literal["arm64", "arm64ec", "x86_64"],
        config: Literal["Debug", "Release", "RelWithDebInfo"],
        qairt_sdk_root: Path | None,
        mode: str,
    ) -> None:
        cmd = [
            str(REPO_ROOT / "qcom" / "scripts" / "windows" / "build.ps1"),
            "-Arch",
            arch,
            "-Config",
            config,
            "-TargetPlatform",
            target_platform,
            "-Mode",
            mode,
        ]

        if venv is not None:
            cmd.extend(["-PyVEnv", str(venv)])
        if qairt_sdk_root is not None:
            cmd.extend(["-QairtSdkRoot", str(qairt_sdk_root)])

        super().__init__(group_name, [cmd], env=ort_build_env_vars())


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
