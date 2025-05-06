# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

from collections.abc import Mapping
from pathlib import Path

from ..task import RunExecutablesTask

POWERSHELL_EXECUTABLE = "powershell.exe"


class RunPowershellScriptsTask(RunExecutablesTask):
    """
    A task that runs PowerShell scripts
    """

    def __init__(
        self,
        group_name: str | None,
        scripts_and_args: list[list[str]],
        env: Mapping[str, str] | None = None,
        cwd: Path | None = None,
    ) -> None:
        executables_and_args = [[POWERSHELL_EXECUTABLE] + s_a for s_a in scripts_and_args]  # noqa: RUF005
        super().__init__(group_name, executables_and_args, env, cwd)
