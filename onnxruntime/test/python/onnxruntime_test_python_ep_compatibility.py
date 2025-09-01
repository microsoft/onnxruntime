# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import platform
import sys
import unittest

from onnxruntime.capi.onnxruntime_pybind11_state import (
    OrtCompiledModelCompatibility,
    get_ep_devices,
    get_model_compatibility_for_ep_devices,
)

# handle change from python 3.8 and on where loading a dll from the current directory needs to be explicitly allowed.
if platform.system() == "Windows" and sys.version_info.major >= 3 and sys.version_info.minor >= 8:  # noqa: YTT204
    os.add_dll_directory(os.getcwd())


class TestEpCompatibility(unittest.TestCase):
    def test_invalid_args(self):
        # empty devices
        with self.assertRaises(RuntimeError):
            get_model_compatibility_for_ep_devices([], "info")
        # None compatibility info should raise TypeError before native call
        with self.assertRaises(TypeError):
            get_model_compatibility_for_ep_devices(get_ep_devices(), None)  # type: ignore[arg-type]

    def test_basic_smoke(self):
        devices = list(get_ep_devices())
        if not devices:
            self.skipTest("No EP devices available in this build")

        # Always select CPUExecutionProvider; skip if not present.
        cpu_devices = [d for d in devices if getattr(d, "ep_name", None) == "CPUExecutionProvider"]
        if not cpu_devices:
            self.skipTest("CPUExecutionProvider not available in this build")
        selected = [cpu_devices[0]]

        # API requires all devices belong to the same EP; we pass only one.
        status = get_model_compatibility_for_ep_devices(selected, "arbitrary-compat-string")
        self.assertEqual(status, OrtCompiledModelCompatibility.EP_NOT_APPLICABLE)


if __name__ == "__main__":
    unittest.main()
