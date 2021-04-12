# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import collections
import contextlib
import logging
import os
import shutil
import signal
import subprocess
import time
import typing

from ..run import run
from ..platform import is_windows


_log = logging.getLogger("util.android")


SdkToolPaths = collections.namedtuple(
    "SdkToolPaths", ["emulator", "adb", "sdkmanager", "avdmanager"])


def get_sdk_tool_paths(sdk_root: str):
    def filename(name, windows_extension):
        if is_windows():
            return "{}.{}".format(name, windows_extension)
        else:
            return name

    def resolve_path(dirnames, basename):
        dirnames.insert(0, "")
        for dirname in dirnames:
            path = shutil.which(os.path.join(dirname, basename))
            if path is not None:
                path = os.path.realpath(path)
                _log.debug("Found {} at {}".format(basename, path))
                return path
        _log.warning("Failed to resolve path for {}".format(basename))
        return None

    return SdkToolPaths(
        emulator=resolve_path(
            [os.path.join(sdk_root, "emulator")],
            filename("emulator", "exe")),
        adb=resolve_path(
            [os.path.join(sdk_root, "platform-tools")],
            filename("adb", "exe")),
        sdkmanager=resolve_path(
            [os.path.join(sdk_root, "tools", "bin"),
             os.path.join(sdk_root, "cmdline-tools", "tools", "bin")],
            filename("sdkmanager", "bat")),
        avdmanager=resolve_path(
            [os.path.join(sdk_root, "tools", "bin"),
             os.path.join(sdk_root, "cmdline-tools", "tools", "bin")],
            filename("avdmanager", "bat")))


def create_virtual_device(
        sdk_tool_paths: SdkToolPaths,
        system_image_package_name: str,
        avd_name: str):
    run(sdk_tool_paths.sdkmanager, "--install", system_image_package_name,
        input=b"y")

    run(sdk_tool_paths.avdmanager, "create", "avd",
        "--name", avd_name,
        "--package", system_image_package_name,
        "--force",
        input=b"no")


_process_creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if is_windows() else 0


def _start_process(*args) -> subprocess.Popen:
    _log.debug("Starting process - args: {}".format([*args]))
    return subprocess.Popen([*args], creationflags=_process_creationflags)


_stop_signal = signal.CTRL_BREAK_EVENT if is_windows() else signal.SIGTERM


def _stop_process(proc: subprocess.Popen):
    _log.debug("Stopping process - args: {}".format(proc.args))
    proc.send_signal(_stop_signal)

    try:
        proc.wait(30)
    except subprocess.TimeoutExpired:
        _log.warning("Timeout expired, forcibly stopping process...")
        proc.kill()


def _stop_process_with_pid(pid: int):
    # not attempting anything fancier than just sending _stop_signal for now
    _log.debug("Stopping process - pid: {}".format(pid))
    os.kill(pid, _stop_signal)


def start_emulator(
        sdk_tool_paths: SdkToolPaths,
        avd_name: str,
        extra_args: typing.Optional[typing.Sequence[str]] = None) -> subprocess.Popen:
    with contextlib.ExitStack() as emulator_stack, \
         contextlib.ExitStack() as waiter_stack:
        emulator_args = [
            sdk_tool_paths.emulator, "-avd", avd_name,
            "-memory", "4096",
            "-timezone", "America/Los_Angeles",
            "-no-snapshot",
            "-no-audio",
            "-no-boot-anim",
            "-no-window"]
        if extra_args is not None:
            emulator_args += extra_args

        emulator_process = emulator_stack.enter_context(
            _start_process(*emulator_args))
        emulator_stack.callback(_stop_process, emulator_process)

        waiter_process = waiter_stack.enter_context(
            _start_process(
                sdk_tool_paths.adb, "wait-for-device", "shell",
                "while [[ -z $(getprop sys.boot_completed) ]]; do sleep 1; done; input keyevent 82"))
        waiter_stack.callback(_stop_process, waiter_process)

        # poll subprocesses
        sleep_interval_seconds = 1
        while True:
            waiter_ret, emulator_ret = waiter_process.poll(), emulator_process.poll()

            if emulator_ret is not None:
                # emulator exited early
                raise RuntimeError("Emulator exited early with return code: {}".format(emulator_ret))

            if waiter_ret is not None:
                if waiter_ret == 0:
                    break
                raise RuntimeError("Waiter process exited with return code: {}".format(waiter_ret))

            time.sleep(sleep_interval_seconds)

        # emulator is ready now
        emulator_stack.pop_all()
        return emulator_process


def stop_emulator(emulator_proc_or_pid: typing.Union[subprocess.Popen, int]):
    if isinstance(emulator_proc_or_pid, subprocess.Popen):
        _stop_process(emulator_proc_or_pid)
    elif isinstance(emulator_proc_or_pid, int):
        _stop_process_with_pid(emulator_proc_or_pid)
    else:
        raise ValueError("Expected either a PID or subprocess.Popen instance.")
