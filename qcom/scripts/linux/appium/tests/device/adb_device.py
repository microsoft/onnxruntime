# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import shlex
import subprocess
from pathlib import Path

from .device_base import DeviceBase


class AdbDevice(DeviceBase):
    def __init__(self, adb_exe: Path = Path("adb"), serial_id: str | None = None, host: str | None = None) -> None:
        self.__adb_exe = adb_exe
        self.__serial_id = serial_id
        self.__host = host

    def run_adb(self, adb_args: list[str], check: bool, capture_output: bool = False) -> list[str] | None:
        res = self.__run_adb(adb_args, check, capture_output=capture_output)
        if capture_output:
            return res.stdout.decode("utf-8").split("\n")
        return None

    def logcat(self, regex: str | None) -> str:
        cmd = ["logcat", "-d"]
        if regex is not None:
            cmd += [f"--regex={regex}"]
        res = self.__run_adb(cmd, check=True, capture_output=False)
        return res.stdout.decode("utf-8")

    def push(self, local: Path, remote: Path) -> None:
        self.__do("push", [str(local), str(remote)], check=True)

    def pull(self, remote: Path, local: Path) -> None:
        self.__do("pull", [str(remote), str(local)], check=True)

    def shell(
        self,
        command: list[str],
        check: bool = True,
        capture_output: bool = False,
    ) -> list[str] | None:
        return self.__do("shell", command, check=check, capture_output=capture_output)

    def __do(
        self,
        verb: str,
        extra_args: list[str],
        check: bool,
        capture_output: bool = False,
    ) -> list[str] | None:
        return self.run_adb([verb, *extra_args], check=check, capture_output=capture_output)

    def __run_adb(
        self, adb_args: list[str], check: bool, capture_output: bool = True
    ) -> subprocess.CompletedProcess[bytes]:
        args: list[str] = [str(self.__adb_exe)]
        if self.__host is None:
            args.append("-d")
        else:
            args += ["-H", self.__host]
        if self.__serial_id is not None:
            args += ["-s", self.__serial_id]
        args += adb_args
        print(f"$ {shlex.join(args)}")
        return subprocess.run(args, check=check, capture_output=capture_output)
