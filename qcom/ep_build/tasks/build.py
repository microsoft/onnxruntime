# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import os
from collections.abc import Collection
from pathlib import Path
from typing import Literal

from ..github import is_host_github_runner
from ..task import BashScriptsWithVenvTask, RunExecutablesWithVenvTask
from ..util import REPO_ROOT
from .windows import RunPowershellScriptsTask


class BuildEpLinuxTask(BashScriptsWithVenvTask):
    """Build ONNX Runtime on a Linux host."""

    def __init__(
        self,
        group_name: str | None,
        venv: Path | None,
        target_platform: str,
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

        super().__init__(group_name, venv, [cmd])


class BuildEpWindowsTask(RunPowershellScriptsTask):
    def __init__(
        self,
        group_name: str | None,
        venv: Path | None,
        arch: str,
        qairt_sdk_root: Path | None,
        mode: str,
    ) -> None:
        cmd = [
            str(REPO_ROOT / "qcom" / "scripts" / "windows" / "build.ps1"),
            "-Arch",
            arch,
            "-Mode",
            mode,
        ]

        if venv is not None:
            cmd.extend(["-PyVEnv", str(venv)])
        if qairt_sdk_root is not None:
            cmd.extend(["-QairtSdkRoot", str(qairt_sdk_root)])

        super().__init__(group_name, [cmd])


class QdcTestsTask(RunExecutablesWithVenvTask):
    def __init__(
        self,
        group_name: str | None,
        venv: Path | None,
        platforms: Collection[Literal["android", "windows"]],
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

        if is_host_github_runner():
            actor = os.environ["GITHUB_ACTOR"]
            branch = os.environ["GITHUB_REF_NAME"]
            cmd.append(f"--name={actor}-{branch}")
            cmd.append(f"--on-behalf-of={actor}")

        super().__init__(group_name, venv, [cmd])
