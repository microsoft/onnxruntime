# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

from pathlib import Path
from typing import List, Mapping, Optional

from ..task import RunExecutablesTask

POWERSHELL_EXECUTABLE = "powershell.exe"


class RunPowershellScriptsTask(RunExecutablesTask):
    """
    A task that runs PowerShell scripts
    """

    def __init__(
        self,
        group_name: Optional[str],
        scripts_and_args: List[List[str]],
        env: Optional[Mapping[str, str]] = None,
        cwd: Optional[Path] = None,
    ) -> None:
        executables_and_args = [[POWERSHELL_EXECUTABLE] + s_a for s_a in scripts_and_args]
        super().__init__(group_name, executables_and_args, env, cwd)
