# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import shlex
import subprocess
from pathlib import Path


class Adb:
    _DEVICE_IS_ROOTED: bool | None = None
    _SU_USES_C_ARG: bool | None = None

    def __init__(self, adb_exe: Path = Path("adb"), serial_id: str | None = None) -> None:
        self.__adb_exe = adb_exe
        self.__serial_id = serial_id

    @property
    def device_is_rooted(self) -> bool:
        if self._DEVICE_IS_ROOTED is None:
            self._DEVICE_IS_ROOTED = self.__device_is_rooted()
        return self._DEVICE_IS_ROOTED

    @property
    def su_uses_c_arg(self) -> bool:
        if self._SU_USES_C_ARG is None:
            self._SU_USES_C_ARG = self.__su_uses_c_arg()
        return self._SU_USES_C_ARG

    def run_adb(self, adb_args: list[str], check: bool, capture_output: bool = False) -> list[str] | None:
        res = self.__run_adb(adb_args, check, capture_output=capture_output)
        if capture_output:
            return res.stdout.decode("utf-8").split("\n")
        return None

    def install(self, package: Path) -> None:
        self.__do("install", [str(package)], check=True)

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
        command: list[str] | None = None,
        check: bool = True,
        capture_output: bool = False,
    ) -> list[str] | None:
        if command is None:
            command = []
        return self.__do("shell", command, check=check, capture_output=capture_output)

    def sudo(self, command: list[str] | None, check: bool = True) -> list[str] | None:
        if command is None:
            command = []
        if not self.device_is_rooted:
            raise RuntimeError("sudo is only available on rooted devices.")
        if self.su_uses_c_arg:
            cmd = ["su", "-c", shlex.join(command)]
        else:
            cmd = ["su", "root", shlex.join(command)]
        return self.shell(cmd, check=check)

    def uninstall(self, package: Path) -> None:
        self.__do("uninstall", [str(package)], check=False)

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
        args: list[str] = [str(self.__adb_exe), "-d"]
        if self.__serial_id is not None:
            args += ["-s", self.__serial_id]
        args += adb_args
        print(f"$ {shlex.join(args)}")
        return subprocess.run(args, check=check, capture_output=capture_output)

    def __device_is_rooted(self) -> bool:
        res = self.__run_adb(["shell", "which", "su"], False)
        return res.returncode == 0

    def __su_uses_c_arg(self) -> bool:
        """
        Try to figure out if we're running a version of su that has a -c option.
        The alternative, seen on automotive devices, has the following usage message:

        usage: su [WHO [COMMAND...]]
        Switch to WHO (default 'root') and run the given COMMAND (default sh).

        WHO is a comma-separated list of user, group, and supplementary groups
        that order.
        """
        proc = subprocess.run(
            ["adb", "shell", "su -h"],
            check=False,
            capture_output=True,
        )
        return not proc.stderr.decode("utf-8").startswith("usage: su [WHO [COMMAND...]]")
