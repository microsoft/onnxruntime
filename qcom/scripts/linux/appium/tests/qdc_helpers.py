# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

from pathlib import Path

from adb_utils import Adb

ORT_BUILD_CONFIG = "Release"

# this is where our zip file is extracted on the QDC host.
QDC_HOST_PATH = "/qdc/appium"

# this is where we will copy our files on the Android device.
ORT_DEVICE_PATH = "/data/local/tmp/onnxruntime"

# Glob matching test result files on the device
ORT_TEST_RESULTS_DEVICE_GLOB = f"{ORT_DEVICE_PATH}/{ORT_BUILD_CONFIG}/onnxruntime_*.results.xml"

# all files in this folder will be uploaded back to QDC.
QDC_LOG_PATH = "/data/local/tmp/QDC_logs"

QNN_ADSP_LIBRARY_PATH = "\\;".join([f"{ORT_DEVICE_PATH}/lib/hexagon-v{arch}/unsigned" for arch in [66, 68, 73, 75, 79]])
QNN_LD_LIBRARY_PATH = f"{ORT_DEVICE_PATH}/{ORT_BUILD_CONFIG}:{ORT_DEVICE_PATH}/lib/aarch64-android"


class TestBase:
    def clean_device(self):
        adb = Adb()

        # Clean-up device. QDC doesn't do any clean-up.
        for path in [ORT_DEVICE_PATH, QDC_LOG_PATH]:
            adb.shell(["rm", "-rf", path])
            adb.shell(["mkdir", "-p", path])

    def prepare_device(self):
        adb = Adb()

        # disable SE Linux; this enables tombstone to capture symbols.
        adb.sudo(["setenforce", "0"])

    def prepare_ort_tests(self):
        self.clean_device()
        self.prepare_device()

        adb = Adb()

        # Push binaries from QDC_HOST_PATH to /data/local/tmp
        for item in Path(QDC_HOST_PATH).iterdir():
            adb.push(item, Path(ORT_DEVICE_PATH))

        # Builds sometimes come from Windows, where executable bits are not set.
        adb.shell(
            [
                "find",
                f"{ORT_DEVICE_PATH}/lib",
                "-type",
                "f",
                "-exec",
                "chmod",
                "+x",
                "{}",
                "\\;",
            ],
        )
        adb.shell([f"sh -c 'chmod +x {ORT_DEVICE_PATH}/{ORT_BUILD_CONFIG}/*'"])

    def copy_logs(self):
        adb = Adb()
        adb.shell(
            [f"sh -c 'cp {ORT_TEST_RESULTS_DEVICE_GLOB} {QDC_LOG_PATH}'"],
        )
