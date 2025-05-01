# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from __future__ import annotations

import os
import platform
import sys
import tempfile
import unittest

import onnxruntime as onnxrt

# handle change from python 3.8 and on where loading a dll from the current directory needs to be explicitly allowed.
if platform.system() == "Windows" and sys.version_info.major >= 3 and sys.version_info.minor >= 8:  # noqa: YTT204
    os.add_dll_directory(os.getcwd())

available_providers = list(onnxrt.get_available_providers())


class TestAutoEP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp_model_dir = tempfile.TemporaryDirectory(prefix="ort.autoep_")

        # Note: swap with the commented line if you want to see the models in local test dir.
        cls._tmp_dir_path = cls._tmp_model_dir.name
        # cls._tmp_dir_path = "."

    @classmethod
    def tearDownClass(cls):
        cls._tmp_model_dir.cleanup()

    def test_register_unregister_cuda_ep_library(self):
        if "CUDAExecutionProvider" not in available_providers:
            self.skipTest("Skipping test because it needs CUDA EP")

        cuda_ep_lib = None
        if sys.platform == "win32":
            cuda_ep_lib = "onnxruntime_providers_cuda.dll"
        elif sys.platform == "linux":
            cuda_ep_lib = "libonnxruntime_providers_cuda.so"
        else:
            self.skipTest("Skipping test because it can only run on Windows or Linux")

        onnxrt.register_execution_provider_library("CUDAExecutionProvider", cuda_ep_lib)
        onnxrt.unregister_execution_provider_library("CUDAExecutionProvider")
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main(verbosity=1)
