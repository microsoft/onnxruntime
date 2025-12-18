# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

from pathlib import Path

from device import DeviceBase, device_from_url
from ort_test_config import OrtTestConfig, default_test_config


class TestBase:
    @staticmethod
    def config() -> OrtTestConfig:
        return default_test_config()

    @property
    def device(self) -> DeviceBase:
        return device_from_url(self.config().device_url)

    def clean_device(self):
        # Clean-up device. QDC doesn't do any clean-up.
        for path in [self.config().device_path, self.config().qdc_log_path]:
            self.device.shell(["rm", "-rf", path])
            self.device.shell(["mkdir", "-p", path])

    def prepare_ort_tests(self):
        self.clean_device()

        # Push binaries from qdc_host_path to /data/local/tmp
        for item in Path(self.config().qdc_host_path).iterdir():
            self.device.push(item, Path(self.config().device_path))

        # Push test models
        self.device.shell(["mkdir", "-p", f"{self.config().device_path}/model_tests"])
        for item in Path(self.config().model_test_path).iterdir():
            # We're playing games with .resolve() and .name because adb push doesn't follow symlinks.
            self.device.push(item.resolve(), Path(self.config().device_path) / "model_tests" / item.name)

        # Builds sometimes come from Windows, where executable bits are not set.
        if (Path(self.config().host_build_root) / "lib").exists():
            # fmt: off
            self.device.shell(
                [
                    "find", f"{self.config().device_path}/lib",
                    "-type", "f",
                    "-exec", "chmod", "+x", "{}",
                    "\\;",
                ],
            )
            # fmt: on
        self.device.shell([f"sh -c 'chmod +x {self.config().device_build_root}/*'"])

    def copy_logs(self):
        self.device.shell(
            [f"sh -c 'cp {self.config().test_results_device_glob} {self.config().qdc_log_path}'"],
        )
