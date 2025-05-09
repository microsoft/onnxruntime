# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

from pathlib import Path

from ..task import BashScriptsTask
from ..util import REPO_ROOT
from .windows import RunPowershellScriptsTask


class BuildEpLinuxTask(BashScriptsTask):
    def __init__(
        self,
        group_name: str | None,
        qairt_sdk_root: Path,
        mode: str,
    ) -> None:
        cmd = [
            str(REPO_ROOT / "qcom" / "scripts" / "linux" / "build.sh"),
            f"--qairt_sdk_root={qairt_sdk_root}",
            f"--mode={mode}",
        ]
        super().__init__(group_name, [cmd])


class BuildEpWindowsTask(RunPowershellScriptsTask):
    def __init__(
        self,
        group_name: str | None,
        arch: str,
        qairt_sdk_root: Path,
        mode: str,
    ) -> None:
        cmd = [
            str(REPO_ROOT / "qcom" / "scripts" / "windows" / "build.ps1"),
            "-Arch",
            arch,
            "-Mode",
            mode,
            "-QairtSdkRoot",
            str(qairt_sdk_root),
        ]
        super().__init__(group_name, [cmd])


class InstallDepsWindowsTask(RunPowershellScriptsTask):
    def __init__(
        self,
        group_name: str | None,
    ) -> None:
        cmd = [
            str(REPO_ROOT / "qcom" / "scripts" / "windows" / "install_deps.ps1"),
        ]
        super().__init__(group_name, [cmd])
