# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import logging
import os
import platform
import shlex
import subprocess
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path


def default_parallelism() -> int:
    """A conservative number of processes across which to spread pytests desiring parallelism."""
    from .github import is_host_github_runner  # noqa: PLC0415

    cpu_count = os.cpu_count()
    if not cpu_count:
        return 1

    # In CI, saturate the machine
    if is_host_github_runner():
        return cpu_count

    # When running locally, leave a little CPU for other uses
    return max(1, int(cpu_count - 2))


# Convenience function for printing to the logger.
def echo(value: str) -> None:
    logging.info(value)


def get_env_bool(key: str, default: bool | None = None) -> bool | None:
    val = os.environ.get(key, None)
    if val is None:
        return default
    return str_to_bool(val)


def get_env_int(key: str, default: int | None = None) -> int | None:
    val = os.environ.get(key, None)
    if val is None:
        return default
    int_val = int(val)
    assert str(int_val) == val, f"Environment variable '{key}' is not a well-formed int."
    return int_val


def git_head_sha() -> str:
    return run_and_get_output(["git", "rev-parse", "HEAD"], quiet=True)


def have_root() -> bool:
    # mypy/pyright are generally unhappy here because these calls aren't always available.
    if is_host_windows():
        import ctypes  # noqa: PLC0415

        return ctypes.windll.shell32.IsUserAnAdmin() != 0  # type: ignore[attr-defined]
    return os.geteuid() == 0  # type:ignore[attr-defined]


def is_host_arm64() -> bool:
    return platform.processor().startswith("ARMv8")


def is_host_in_ci():
    from .github import is_host_github_runner  # noqa: PLC0415

    return is_host_github_runner()


def is_host_user_linux():
    return is_host_linux() and not is_host_in_ci()


def is_host_linux():
    return platform.uname().system == "Linux"


def is_host_mac():
    return platform.uname().system == "Darwin"


def is_host_windows():
    return platform.uname().system == "Windows"


def process_output(process: subprocess.CompletedProcess):
    return process.stdout.decode("utf-8").strip()


def run(
    command: str | list[str],
    check: bool = True,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    stdout: int | None = None,
    capture_output: bool = False,
    quiet: bool = False,
) -> subprocess.CompletedProcess:
    return run_with_venv(
        venv=None,
        command=command,
        check=check,
        env=env,
        cwd=cwd,
        stdout=stdout,
        capture_output=capture_output,
        quiet=quiet,
    )


def run_and_get_output(
    command: str | list[str],
    check: bool = True,
    cwd: Path | None = None,
    capture_stderr: bool = False,
    quiet: bool = False,
) -> str:
    return run_with_venv_and_get_output(
        venv=None,
        command=command,
        check=check,
        cwd=cwd,
        capture_stderr=capture_stderr,
        quiet=quiet,
    )


def run_with_venv(
    venv: Path | None,
    command: str | list[str],
    check: bool = True,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    stdout: int | None = None,
    capture_output: bool = False,
    quiet: bool = False,
) -> subprocess.CompletedProcess:
    if venv is None:
        full_command = command
    else:
        # `source` requires paths with forward slashes
        activate_path = str((venv / Path(VENV_ACTIVATE_RELPATH)).absolute()).replace("\\", "/")
        shell_command = f"source {activate_path} && " + (command if isinstance(command, str) else shlex.join(command))
        full_command = [BASH_EXECUTABLE, "-c", shell_command]
    if not quiet:
        echo(f"$ {full_command if isinstance(full_command, str) else shlex.join(full_command)}")
    return subprocess.run(
        full_command,
        stdout=stdout,
        capture_output=capture_output,
        shell=isinstance(full_command, str),
        check=check,
        executable=BASH_EXECUTABLE if isinstance(full_command, str) else None,
        env=env,
        cwd=cwd,
    )


def run_with_venv_and_get_output(
    venv: Path | None,
    command: str | list[str],
    check: bool = True,
    cwd: Path | None = None,
    capture_stderr: bool = False,
    quiet: bool = False,
) -> str:
    return process_output(
        run_with_venv(
            venv,
            command,
            stdout=subprocess.PIPE if not capture_stderr else None,
            check=check,
            cwd=cwd,
            capture_output=capture_stderr,
            quiet=quiet,
        )
    )


def str_to_bool(word: str) -> bool:
    return word.lower() in ["1", "true", "yes"]


def timestamp_brief() -> str:
    return datetime.now().strftime("%Y%m%d.%H%M%S")


class Colors:
    GREEN = "\033[0;32m" if not is_host_windows() else ""
    GREY = "\033[0;37m" if not is_host_windows() else ""
    RED = "\033[0;31m" if not is_host_windows() else ""
    RED_BOLD = "\033[0;1;31m" if not is_host_windows() else ""
    RED_REVERSED_VIDEO = "\033[0;7;31m" if not is_host_windows() else ""
    YELLOW = "\033[0;33m" if not is_host_windows() else ""
    OFF = "\033[0m" if not is_host_windows() else ""


if is_host_windows():
    BASH_EXECUTABLE = str(Path(os.environ["ProgramW6432"], "Git/bin/bash.exe"))  # noqa: SIM112
    assert os.path.isfile(BASH_EXECUTABLE), f"Bash executable not found in {BASH_EXECUTABLE}."
else:
    BASH_EXECUTABLE = run_and_get_output(["which", "bash"], quiet=True)


DEFAULT_PYTHON = Path("python.exe") if is_host_windows() else Path("python3.10")
"""Different python distributions have different executable names. Use this for a reasonable default."""

MSFT_CI_REQUIREMENTS_RELPATH = (
    f"tools/ci_build/github/{'windows' if is_host_windows() else 'linux'}/python/requirements.txt"
)

REPO_ROOT = Path(run_and_get_output(["git", "rev-parse", "--show-toplevel"], quiet=True))


VENV_ACTIVATE_RELPATH = "Scripts/activate" if is_host_windows() else "bin/activate"
"""Where to find the bash script to source to activate a virtual environment."""
