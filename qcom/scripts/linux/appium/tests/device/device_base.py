# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import abc
from pathlib import Path
from urllib.parse import urlparse


class DeviceBase(abc.ABC):
    @abc.abstractmethod
    def push(self, local: Path, remote: Path) -> None:
        pass

    @abc.abstractmethod
    def pull(self, remote: Path, local: Path) -> None:
        pass

    @abc.abstractmethod
    def shell(
        self,
        command: list[str],
        check: bool = True,
        capture_output: bool = False,
    ) -> list[str] | None:
        pass


def device_from_url(url: str) -> DeviceBase:
    # local to avoid circular import
    from .adb_device import AdbDevice  # noqa:PLC0415
    from .ssh_device import SshDevice  # noqa:PLC0415

    url_parts = urlparse(url)
    if url_parts.scheme == "adb":
        # adb:// (not currently configurable)
        if url_parts.hostname is not None:
            # adb://serial@host --> adb -H host -s serial
            assert url_parts.username is not None, "Serial number must not be None"
            return AdbDevice(serial_id=url_parts.username, host=url_parts.hostname)
        else:
            # adb://
            return AdbDevice()
    elif url_parts.scheme == "ssh":
        # ssh://user@host[:port]/[?key=val[&key=val...]]
        # Example: ssh://root@192.168.0.115:22/?StrictHostKeychecking=no&UserKnownHostsFile=/dev/null
        assert url_parts.username is not None, "Username must not be None"
        assert url_parts.hostname is not None, "Hostname must not be None"
        ssh_options = None if len(url_parts.query) == 0 else url_parts.query.split("&")
        return SshDevice(url_parts.username, url_parts.hostname, url_parts.port, ssh_options)
    else:
        raise NotImplementedError(f"Unknown device URL scheme {url_parts.scheme}.")
