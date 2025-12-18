# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import shlex
import subprocess
from collections.abc import Iterable
from pathlib import Path

from .device_base import DeviceBase


class SshDevice(DeviceBase):
    def __init__(self, user: str, host: str, port: int | None = None, ssh_options: Iterable[str] | None = None) -> None:
        self.__user = user
        self.__host = host
        self.__port = port if port is not None else 22
        self.__ssh_options = ssh_options

    @property
    def connection_str(self) -> str:
        return f"{self.__user}@{self.__host}"

    def push(self, local: Path, remote: Path) -> None:
        self.__rsync(str(local), f"{self.connection_str}:{remote}")

    def pull(self, remote: Path, local: Path) -> None:
        self.__rsync(f"{self.connection_str}:{remote}", str(local))

    def shell(
        self,
        command: list[str],
        check: bool = True,
        capture_output: bool = False,
    ) -> list[str] | None:
        args = [*self.__ssh_cmd, self.connection_str, *command]
        print(f"$ {shlex.join(args)}")
        res = subprocess.run(args, check=check, capture_output=capture_output)
        if capture_output:
            return res.stdout.decode("utf-8").split("\n")
        return None

    def __rsync(self, src: str, dest: str) -> None:
        # We avoid -a because it can cause directories to be created that we cannot write to.
        # fmt: off
        args = [
            "rsync",
            "-e", shlex.join(self.__ssh_cmd),
            "-lrt", "--compress", "--verbose",
            src, dest,
        ]
        # fmt: on

        print(f"$ {shlex.join(args)}")
        subprocess.run(args, check=True)

    @property
    def __ssh_cmd(self) -> list[str]:
        # fmt: off
        cmd = [
            "ssh",
            "-p", str(self.__port),
        ]
        # fmt: on

        if self.__ssh_options is not None:
            for opt in self.__ssh_options:
                cmd.extend(["-o", opt])

        return cmd
