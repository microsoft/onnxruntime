# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from __future__ import annotations

import os
import tempfile
import unittest

import onnxruntime as onnxrt
from onnxruntime.capi.onnxruntime_pybind11_state import Fail


class AutoEpTestCase(unittest.TestCase):
    """
    Base class for TestCase classes that need to register and unregister EP libraries.
    Because EP libraries are registered with the ORT environment and all unit tests share
    the same environment, this class tracks which libraries have already been registered
    so that they are not erroneously registered or unregistered.

    Derived classes must use 'self.register_execution_provider_library()' and
    'self.unregister_execution_provider_library()' to benefit from these utilities.
    """

    @classmethod
    def setUpClass(cls):
        cls._tmp_model_dir = tempfile.TemporaryDirectory(prefix="ort.autoep_")

        # Note: swap with the commented line if you want to see the models in local test dir.
        cls._tmp_dir_path = cls._tmp_model_dir.name
        # cls._tmp_dir_path = "."

        # Track registered EP libraries across all tests.
        cls._registered_providers = set()

    @classmethod
    def tearDownClass(cls):
        cls._tmp_model_dir.cleanup()

    def register_execution_provider_library(self, ep_registration_name: str, ep_lib_path: os.PathLike | str):
        if ep_registration_name in self._registered_providers:
            return  # Already registered

        try:
            onnxrt.register_execution_provider_library(ep_registration_name, ep_lib_path)
        except Fail as onnxruntime_error:
            if "already registered" in str(onnxruntime_error):
                pass  # Allow register to fail if the EP library was previously registered.
            else:
                raise onnxruntime_error

        # Add this EP library to set of registered EP libraries.
        # If the unit test itself does not unregister the library, tearDown() will try.
        self._registered_providers.add(ep_registration_name)

    def unregister_execution_provider_library(self, ep_registration_name: str):
        if ep_registration_name not in self._registered_providers:
            return  # Not registered

        try:
            onnxrt.unregister_execution_provider_library(ep_registration_name)
        except Fail as onnxruntime_error:
            if "was not registered" in str(onnxruntime_error):
                pass  # Allow unregister to fail if the EP library was never registered.
            else:
                raise onnxruntime_error

        self._registered_providers.remove(ep_registration_name)
