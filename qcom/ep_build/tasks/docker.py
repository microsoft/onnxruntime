# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import functools
import operator
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from ..task import RunExecutablesTask
from ..typing import TargetPyVersionT
from ..util import DEFAULT_PYTHON_LINUX

DOCKER_BUILD_USER = "ortqnnep"
DOCKER_REPO_ROOT = Path("/ort")
MANYLINUX_2_34_AARCH64_TAG = "ort-manylinux_2_34_aarch64"


def _argify(arg: str, sep: str, args: Mapping[Any, Any]) -> list[str]:
    return functools.reduce(operator.iadd, [[arg, f"{k}{sep}{args[k]}"] for k in args], [])


class DockerBuildTask(RunExecutablesTask):
    def __init__(
        self,
        group_name: str | None,
        dockerfile: Path,
        image_name: str,
        build_args: Mapping[str, str] | None = None,
    ) -> None:
        build_args_list: list[str] = ["--build-arg", f"BUILD_USER={DOCKER_BUILD_USER}"]
        if build_args is not None:
            build_args_list.extend(_argify("--build-arg", "=", build_args))
        # fmt: off
        super().__init__(
            group_name,
            [
                [
                    "docker", "build", "--platform", "linux/aarch64",
                    "--file", str(dockerfile), "--tag", image_name,
                    *build_args_list,
                    str(dockerfile.parent),
                ]
            ],
        )
        # fmt: on


class DockerRunTask(RunExecutablesTask):
    def __init__(
        self,
        group_name: str | None,
        image_name: str,
        command: Iterable[str] | Callable[[], Iterable[str]],
        working_dir: Path | None = None,
        volumes: Mapping[Path, Path] | None = None,
        env: Mapping[str, str] | None = None,
        remove: bool = True,
    ) -> None:
        def make_cmd() -> list[list[str]]:
            cmd = ["docker", "run", "--platform", "linux/aarch64", "--user", DOCKER_BUILD_USER]
            if working_dir is not None:
                cmd.extend(["--workdir", str(working_dir)])
            if volumes is not None:
                cmd.extend(_argify("--volume", ":", {k.absolute(): v for k, v in volumes.items()}))
            if env is not None:
                cmd.extend(_argify("--env", "=", env))
            if remove:
                cmd.append("--rm")
            cmd.append(image_name)
            cmd.extend(command if isinstance(command, Iterable) else command())
            return [cmd]

        super().__init__(group_name, make_cmd)


class DockerBuildAndTestTask(DockerRunTask):
    """Run build_and_test.py inside a docker container."""

    def __init__(
        self,
        group_name: str | None,
        tasks: Iterable[str],
        target_py_version: TargetPyVersionT | None,
        image_name: str,
        volumes: Mapping[Path, Path] | None = None,
        remove: bool = True,
        venv_path: Path | None = None,
        qairt_sdk_root: Path | None = None,
        ccache_root: Path | None = None,
    ) -> None:
        # deferred to avoid circular import
        from ..tools import get_package_dir, get_tools_dir  # noqa:PLC0415

        cmd = [
            str(DEFAULT_PYTHON_LINUX),
            str(DOCKER_REPO_ROOT / "qcom" / "build_and_test.py"),
            f"--target-py-version={target_py_version}",
        ]
        if venv_path is not None:
            cmd.append(f"--venv-path={venv_path}")
        cmd.extend(tasks)

        volumes_with_caches = dict({} if volumes is None else volumes)
        volumes_with_caches[get_package_dir()] = Path("/ort_caches/packages")
        volumes_with_caches[get_tools_dir()] = Path("/ort_caches/tools")

        env: dict[str, str] = {
            "ORT_BUILD_PACKAGE_CACHE_PATH": "/ort_caches/packages",
            "ORT_BUILD_TOOLS_PATH": "/ort_caches/tools",
            "ORT_BUILD_PRUNE_PACKAGES": "0",
        }

        if qairt_sdk_root is not None:
            volumes_with_caches[qairt_sdk_root] = Path("/ort_caches/qairt")
            cmd.append("--qairt-sdk-root=/ort_caches/qairt")

        if ccache_root is not None:
            volumes_with_caches[ccache_root] = Path("/ort_caches/ccache")
            cmd.append("--docker-ccache-root=/ort_caches/ccache")

        super().__init__(
            group_name,
            image_name,
            cmd,
            DOCKER_REPO_ROOT,
            volumes_with_caches,
            env,
            remove,
        )
