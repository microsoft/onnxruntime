# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

from pathlib import Path

from ..task import BashScriptsWithVenvTask
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
