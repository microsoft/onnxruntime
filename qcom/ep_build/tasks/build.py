# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import os
from collections.abc import Collection, Generator, Iterable, Mapping
from pathlib import Path
from typing import Literal

from ..github import is_host_github_runner
from ..task import BashScriptsWithVenvTask, CompositeTask, RemovePathsTask, RunExecutablesWithVenvTask
from ..typing import BuildConfigT, TargetArchLinuxT, TargetArchWindowsT, TargetPyVersionT
from ..util import REPO_ROOT, git_head_sha
from .docker import DOCKER_REPO_ROOT, MANYLINUX_2_34_AARCH64_TAG, DockerBuildAndTestTask, DockerRunTask
from .windows import RunPowershellScriptsTask


def get_ort_version() -> str:
    return (REPO_ROOT / "VERSION_NUMBER").read_text().strip()


class BuildEpDockerTask(CompositeTask):
    """Build ONNX Runtime for Linux inside a Docker container."""

    def __init__(
        self,
        group_name: str | None,
        target_arch: TargetArchLinuxT,
        config: BuildConfigT,
        target_py_version: TargetPyVersionT | None,
        qairt_sdk_root: Path | None,
        ccache_root: Path | None,
    ) -> None:
        dist_rel_dir = Path("build") / f"linux-{target_arch}" / config / "dist"

        def non_manylinux_wheels() -> Generator[Path, None, None]:
            dist_dir = REPO_ROOT / dist_rel_dir
            return (whl for whl in dist_dir.glob("*.whl") if "manylinux" not in whl.name)

        def auditwheel_cmd() -> list[str]:
            whls = [whl.name for whl in non_manylinux_wheels()]
            assert len(whls) > 0, f"No wheels found in {dist_rel_dir}."
            return ["auditwheel", "repair", "-w", ".", "--plat", "manylinux_2_34_aarch64", *whls]

        super().__init__(
            group_name,
            [
                DockerBuildAndTestTask(
                    "Building ONNX Runtime inside a container",
                    ["_build_ort_linux_aarch64_manylinux_2_34"],
                    target_py_version,
                    MANYLINUX_2_34_AARCH64_TAG,
                    volumes={REPO_ROOT: DOCKER_REPO_ROOT},
                    venv_path=DOCKER_REPO_ROOT / "build" / "venv.build",
                    qairt_sdk_root=qairt_sdk_root,
                    ccache_root=ccache_root,
                ),
                DockerRunTask(
                    "Repairing ONNX Runtime wheel",
                    MANYLINUX_2_34_AARCH64_TAG,
                    auditwheel_cmd,
                    working_dir=DOCKER_REPO_ROOT / dist_rel_dir,
                    volumes={REPO_ROOT: DOCKER_REPO_ROOT},
                ),
                RemovePathsTask(
                    "Deleting non-repaired wheel",
                    non_manylinux_wheels,
                ),
            ],
        )


class BuildEpLinuxTask(BashScriptsWithVenvTask):
    """Build ONNX Runtime on a Linux host."""

    def __init__(
        self,
        group_name: str | None,
        venv: Path | None,
        target_platform: Literal["android", "linux"],
        target_arch: TargetArchLinuxT,
        config: BuildConfigT,
        target_py_version: TargetPyVersionT | None,
        qairt_sdk_root: Path | None,
        mode: str,
        extra_args: Iterable[str] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        cmd = [
            str(REPO_ROOT / "qcom" / "scripts" / "linux" / "build.sh"),
            f"--target-arch={target_arch}",
            f"--target-platform={target_platform}",
            f"--config={config}",
            f"--mode={mode}",
        ]

        if target_py_version is not None:
            cmd.append(f"--target-py-version={target_py_version}")

        if qairt_sdk_root is not None:
            cmd.append(f"--qairt-sdk-root={qairt_sdk_root}")

        if extra_args is not None:
            cmd.extend(extra_args)

        super().__init__(group_name, venv, [cmd], env=ort_build_env_vars(env))


class BuildEpWindowsTask(RunPowershellScriptsTask):
    def __init__(
        self,
        group_name: str | None,
        venv: Path | None,
        target_arch: TargetArchWindowsT,
        config: BuildConfigT,
        target_py_version: TargetPyVersionT | None,
        qairt_sdk_root: Path | None,
        mode: str,
        build_as_x: bool = False,
    ) -> None:
        cmd = [
            str(REPO_ROOT / "qcom" / "scripts" / "windows" / "build.ps1"),
            "-Arch",
            target_arch,
            "-Config",
            config,
            "-Mode",
            mode,
        ]

        if venv is not None:
            cmd.extend(["-PyVEnv", str(venv).replace(" ", "` ")])
        if qairt_sdk_root is not None:
            cmd.extend(["-QairtSdkRoot", str(qairt_sdk_root).replace(" ", "` ")])

        if build_as_x:
            cmd.extend(["-BuildAsX", "1"])

        if target_py_version is not None:
            cmd.extend(["-TargetPyVersion", str(target_py_version)])

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


def ort_build_env_vars(old_env: Mapping[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy() if old_env is None else dict(old_env)
    if env.get("ORT_NIGHTLY_BUILD", "0") == "1":
        env["NIGHTLY_BUILD"] = "1"
        env["Build_SourceVersion"] = git_head_sha()
    return env
