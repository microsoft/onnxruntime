# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import collections
import contextlib
import logging
import os
import shutil
import signal
import subprocess

from .run import run
from .platform import is_windows


_log = logging.getLogger("util.android")


AndroidSdkToolPaths = collections.namedtuple(
    "AndroidSdkToolPaths", ["emulator", "adb", "sdkmanager", "avdmanager"])


def get_android_sdk_tool_paths(sdk_root: str):
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

    return AndroidSdkToolPaths(
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


@contextlib.contextmanager
def running_android_emulator(
        sdk_tool_paths: AndroidSdkToolPaths,
        system_image_package_name: str,
        emulator_name: str = "ort_android_emulator"):
    run(sdk_tool_paths.sdkmanager, "--install", system_image_package_name,
        input=b"y")

    run(sdk_tool_paths.avdmanager, "create", "avd",
        "--name", emulator_name,
        "--package", system_image_package_name,
        "--force",
        input=b"no")

    with contextlib.ExitStack() as context_stack:
        emulator_creationflags = \
            subprocess.CREATE_NEW_PROCESS_GROUP if is_windows() else 0

        emulator_process = context_stack.enter_context(
            subprocess.Popen(
                [sdk_tool_paths.emulator, "-avd", emulator_name,
                 "-partition-size", "2047",
                 "-memory", "4096",
                 "-timezone", "America/Los_Angeles",
                 "-no-snapshot",
                 "-no-audio",
                 "-no-boot-anim",
                 "-no-window",
                 ],
                creationflags=emulator_creationflags))

        def close_emulator():
            _log.debug("Closing emulator...")
            if is_windows():
                emulator_process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                emulator_process.terminate()

            try:
                emulator_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                _log.warning("Timeout expired, forcibly closing emulator...")
                emulator_process.kill()

        context_stack.callback(close_emulator)

        run(sdk_tool_paths.adb, "wait-for-device", "shell",
            "while [[ -z $(getprop sys.boot_completed) ]]; do sleep 1; done; input keyevent 82")

        yield
