# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import jsonc  # type: ignore[import-untyped]

DEFAULT_HOST_ROOT = "/qdc/appium"

BuildConfigT = Literal["Debug", "RelWithDebInfo", "Release"]
BuildTargetT = Literal["android-aarch64", "linux-aarch64_oe_gcc11_2"]


@dataclass
class OrtTestConfig:
    build_target: BuildTargetT
    build_config: BuildConfigT = "Release"
    device_home: str = "/data/local/tmp"

    device_url: str = "adb://"

    # this is where our zip file is extracted on the QDC host.
    qdc_host_path: str = DEFAULT_HOST_ROOT

    # directory containing model test suites from ONNX.
    host_onnx_model_test_path = f"{qdc_host_path}/model_tests/onnx_models"

    # If true, remove previously uploaded builds during setup.
    clean_build: bool = True

    # If True, remove host_onnx_model_test_path during setup.
    clean_onnx_model_tests: bool = False

    # A list of test executables run by ctest to skip.
    _skip_ctests: list[str] | None = None

    @property
    def build_root_relpath(self) -> str:
        """Where to find the build's executables relative to the test archive root."""
        return f"build/{self.build_target}/{self.build_config}"

    @property
    def device_adsp_library_path(self) -> str:
        """
        The QNN EP build handles skel libs differently depending on the platform.
        The correct implementations copy them into the config directory; others
        just get the files copied into lib/hexagon-v*/unsigned by the test artifact
        archiver.
        """
        return f"{self.device_runtime_path}/build/{self.build_target}/{self.build_config}\\;" + "\\;".join(
            f"{self.device_runtime_path}/lib/hexagon-v{arch}/unsigned" for arch in [66, 68, 73, 75, 79, 81]
        )

    @property
    def device_build_root(self) -> str:
        """Where to find the build's executables on the device."""
        return f"{self.device_runtime_path}/{self.build_root_relpath}"

    @property
    def device_ld_library_path(self) -> str:
        """The EP build doesn't set rpath correctly on all platforms."""
        if self.build_target == "android-aarch64":
            return f"{self.device_build_root}:{self.device_runtime_path}/lib/aarch64-android"
        return ""

    @property
    def device_results_root(self) -> str:
        """Where to put results files on the device."""
        return self.device_build_root

    @property
    def device_onnx_model_test_path(self) -> str:
        """Where to put ONNX test models on the device."""
        return f"{self.device_home}/onnx_model_tests"

    @property
    def device_runtime_path(self) -> str:
        """This is where we will copy the contents of the test archive onto the test device."""
        return f"{self.device_home}/onnxruntime"

    @property
    def host_build_root(self) -> str:
        """Where to find the build's executables on the host."""
        return f"{self.qdc_host_path}/{self.build_root_relpath}"

    @property
    def qdc_log_path(self) -> str:
        """All files in this folder will be uploaded back to QDC."""
        return f"{self.device_home}/QDC_logs"

    @property
    def skip_ctests(self) -> list[str]:
        return [] if self._skip_ctests is None else self._skip_ctests

    @skip_ctests.setter
    def skip_ctests(self, value: Iterable[str] | str | None) -> None:
        if isinstance(value, str):
            value = value.split(",") if len(value) > 0 else []
        if isinstance(value, Iterable):
            self._skip_ctests = list(value)
        else:
            self._skip_ctests = []

    @property
    def test_results_device_glob(self) -> str:
        """Glob matching test result files on the device."""
        return f"{self.device_results_root}/onnxruntime_*.results.*"

    @property
    def test_results_device_log(self) -> str:
        """Path to the on-device test log; this should match the glob in test_results_device_glob."""
        return f"{self.device_results_root}/onnxruntime_test.results.txt"


def default_test_config() -> OrtTestConfig:
    # User-specified config path always wins
    if "ORT_TEST_CONFIG_PATH" in os.environ:
        logging.info("Loading config found in ORT_TEST_CONFIG_PATH")
        return parse_test_config(Path(os.environ["ORT_TEST_CONFIG_PATH"]))

    # No? Okay, maybe build_and_test.py slipped it into our test archive.
    default_config_path = Path(DEFAULT_HOST_ROOT) / "test_config.jsonc"
    if default_config_path.exists():
        logging.debug(f"Loading default config in {default_config_path}.")
        return parse_test_config(default_config_path)
    else:
        logging.warning(f"Default config file not found in {default_config_path}.")

    raise ValueError("Could not find a suitable test config file.")


def _parse_test_config_obj(config_obj: dict) -> OrtTestConfig:
    props = [
        "build_config",
        "clean_build",
        "clean_onnx_model_tests",
        "device_home",
        "device_url",
        "host_onnx_model_test_path",
        "qdc_host_path",
        "skip_ctests",
    ]

    build_target = config_obj["build_target"]
    test_config = OrtTestConfig(build_target)
    logging.debug(f"build_target: {build_target}")

    for prop in props:
        if prop in config_obj:
            logging.debug(f"{prop} --> {config_obj[prop]}")
            setattr(test_config, prop, config_obj[prop])

    return test_config


def parse_test_config(config_json_path: Path) -> OrtTestConfig:
    config = jsonc.load(config_json_path.open("rt"))
    assert isinstance(config, dict), "Config JSON must represent a dict."
    return _parse_test_config_obj(config)
