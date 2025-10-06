# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import logging
import os
from pathlib import Path
from typing import Literal

from .tasks.build import TargetPyVersionT
from .util import (
    REPO_ROOT,
    is_host_windows,
    process_output,
    run_with_venv,
    run_with_venv_and_get_output,
)

PACKAGE_MANAGER = REPO_ROOT / "qcom" / "scripts" / "all" / "package_manager.py"


def get_onnx_models_root(package_manager_venv: Path | None) -> Path:
    return get_package_content_dir(package_manager_venv, "onnx_models")


def get_package_bin_dir(package_manager_venv: Path | None, package: str) -> Path:
    install_package(package_manager_venv, package)
    return Path(_package_action(package_manager_venv, package, "print-bin-dir"))


def get_package_content_dir(package_manager_venv: Path | None, package: str) -> Path:
    install_package(package_manager_venv, package)
    return Path(_package_action(package_manager_venv, package, "print-content-dir"))


PythonExecutableArchT = Literal["arm64", "x86_64"]


def get_python_executable(
    package_manager_venv: Path | None,
    arch: PythonExecutableArchT,
    version: TargetPyVersionT,
) -> Path:
    if not is_host_windows():
        raise NotImplementedError("Not available on this platform")

    py_package = f"python_{version.replace('.', '')}_windows_{arch}"

    def _query_launcher():
        py_vsn = version if arch == "x86_64" else f"{version}-{arch}"
        get_exe_cmd = ["py", f"-{py_vsn}", "-c", "import sys; print(sys.executable)"]
        get_exe_result = run_with_venv(None, get_exe_cmd, check=False, capture_output=True)
        return (get_exe_result.returncode, process_output(get_exe_result))

    rc, py_exe = _query_launcher()
    # See https://docs.python.org/3/using/windows.html#return-codes
    # 101 --> Failed to launch Python
    # 103 --> Unable to locate the requested version
    if rc == 0:
        return Path(py_exe)
    elif rc == 101:
        logging.info("Looks like this python version has been clobbered")
        repair_package(package_manager_venv, py_package)
    elif rc == 103:
        logging.info("This version has not been installed")
        install_package(package_manager_venv, py_package)
    else:
        raise RuntimeError(f"Unknown return code {rc} from py launcher")

    # We've hopefully installed/repaired Python at this point
    rc, py_exe = _query_launcher()
    if rc != 0:
        raise RuntimeError("Could not find Python executable, even after attempting to install/repair it.")
    return Path(py_exe)


def get_tools_dir() -> Path:
    tools_dir = Path(os.getenv("ORT_BUILD_TOOLS_PATH", REPO_ROOT / "build" / "Tools"))
    tools_dir.mkdir(parents=True, exist_ok=True)
    return tools_dir


def install_package(package_manager_venv: Path | None, package: str) -> None:
    _package_action(package_manager_venv, package, "install")


def repair_package(package_manager_venv: Path | None, package: str) -> None:
    _package_action(package_manager_venv, package, "repair")


def _package_action(
    package_manager_venv: Path | None,
    package: str,
    action: str,
) -> str:
    return run_with_venv_and_get_output(
        package_manager_venv,
        ["python", str(PACKAGE_MANAGER), f"--{action}", f"--package={package}", f"--package-root={get_tools_dir()}"],
    )
