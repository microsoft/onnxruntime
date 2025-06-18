# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

from pathlib import Path

from ..task import (
    CompositeTask,
    ConditionalTask,
    NoOpTask,
    RunExecutablesTask,
    RunExecutablesWithVenvTask,
)
from ..util import (
    MSFT_CI_REQUIREMENTS_RELPATH,
    REPO_ROOT,
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
                                ],
                            ),
                        ],
                    ),
                ),
                RunExecutablesWithVenvTask(
                    group_name=None,
                    venv=venv_path,
                    executables_and_args=[
                        [
                            "pip",
                            "install",
                            "uv",
                        ],
                        [
                            "uv",
                            "pip",
                            "install",
                            "-r",
                            f"{REPO_ROOT}/{MSFT_CI_REQUIREMENTS_RELPATH}",
                            "-r",
                            f"{REPO_ROOT}/requirements-dev.txt",
                            "--native-tls",
                        ],
                        ["lintrunner", "init"],
                        [
                            "uv",
                            "pip",
                            "install",
                            "-r",
                            f"{REPO_ROOT}/qcom/requirements.txt",
                            "--trusted-host=ort-ep-win-01",
                            "--index-url=http://ort-ep-win-01:8080",
                            "--native-tls",
                        ],
                    ],
                ),
            ],
        )


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
