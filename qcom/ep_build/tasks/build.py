# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

from pathlib import Path
from typing import Optional

from ..util import REPO_ROOT
from .windows import RunPowershellScriptsTask


class BuildEpWindowsTask(RunPowershellScriptsTask):
    def __init__(
        self,
        group_name: Optional[str],
        arch: str,
        qairt_sdk_root: Path,
        mode: str,
    ) -> None:
        cmd = [
            str(REPO_ROOT / "qcom" / "scripts" / "windows" / "build.ps1"),
            "-Arch", arch,
            "-Mode", mode,
            "-QairtSdkRoot", str(qairt_sdk_root),
        ]
        super().__init__(group_name, [cmd])


class InstallDepsWindowsTask(RunPowershellScriptsTask):
    def __init__(
        self,
        group_name: Optional[str],
    ) -> None:
        cmd = [
            str(REPO_ROOT / "qcom" / "scripts" / "windows" / "install_deps.ps1"),
        ]
        super().__init__(group_name, [cmd])
