# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from __future__ import annotations

import os
import platform
import sys
import unittest
from collections.abc import Sequence

import onnx
from autoep_helper import AutoEpTestCase
from helper import get_name

import onnxruntime as onnxrt
from onnxruntime.capi.onnxruntime_pybind11_state import ModelRequiresCompilation

# handle change from python 3.8 and on where loading a dll from the current directory needs to be explicitly allowed.
if platform.system() == "Windows" and sys.version_info.major >= 3 and sys.version_info.minor >= 8:  # noqa: YTT204
    os.add_dll_directory(os.getcwd())

available_providers = list(onnxrt.get_available_providers())


class TestCompileApi(AutoEpTestCase):
    def test_compile_with_files_prefer_npu_policy(self):
        """
        Tests compiling a model (to/from files) using an EP selection policy (PREFER_NPU).
        """
        if "QNNExecutionProvider" not in available_providers:
            self.skipTest("Skipping test because it needs to run on QNN EP")

        if sys.platform != "win32":
            self.skipTest("Skipping test because provider selection policies are only supported on Windows")

        ep_lib_path = "onnxruntime_providers_qnn.dll"
        ep_name = "QNNExecutionProvider"
        self.register_execution_provider_library(ep_name, ep_lib_path)

        input_model_path = get_name("nhwc_resize_scales_opset18.onnx")
        output_model_path = os.path.join(self._tmp_dir_path, "model.compiled0.onnx")

        session_options = onnxrt.SessionOptions()
        session_options.set_provider_selection_policy(onnxrt.OrtExecutionProviderDevicePolicy.PREFER_NPU)

        model_compiler = onnxrt.ModelCompiler(
            session_options,
            input_model_path,
            embed_compiled_data_into_model=True,
            external_initializers_file_path=None,
        )
        model_compiler.compile_to_file(output_model_path)
        self.assertTrue(os.path.exists(output_model_path))
        self.unregister_execution_provider_library(ep_name)

    def test_compile_with_ep_selection_delegate(self):
        """
        Tests compiling a model (to/from files) using an EP selection delegate callback.
        """
        if sys.platform != "win32":
            self.skipTest("Skipping test because provider selection policies are only supported on Windows")

        input_model_path = get_name("nhwc_resize_scales_opset18.onnx")
        output_model_path = os.path.join(self._tmp_dir_path, "model.compiled.delegate.onnx")

        # User's custom EP selection function.
        def my_delegate(
            ep_devices: Sequence[onnxrt.OrtEpDevice],
            model_metadata: dict[str, str],
            runtime_metadata: dict[str, str],
            max_selections: int,
        ) -> Sequence[onnxrt.OrtEpDevice]:
            self.assertGreater(len(ep_devices), 0)
            self.assertGreater(len(model_metadata), 0)
            self.assertGreater(max_selections, 0)

            # Select the first and last devices (if there are more than one)
            selected_devices = [ep_devices[0]]
            if max_selections > 2 and len(ep_devices) > 1:
                selected_devices.append(ep_devices[-1])  # ORT CPU EP is always last

            return selected_devices

        session_options = onnxrt.SessionOptions()
        session_options.set_provider_selection_policy_delegate(my_delegate)

        model_compiler = onnxrt.ModelCompiler(
            session_options,
            input_model_path,
            embed_compiled_data_into_model=True,
            external_initializers_file_path=None,
        )
        model_compiler.compile_to_file(output_model_path)
        self.assertTrue(os.path.exists(output_model_path))

    def test_compile_with_input_and_output_files(self):
        """
        Tests compiling a model (to/from files) using explicit EP.
        """
        provider = None
        provider_options = dict()
        if "QNNExecutionProvider" in available_providers:
            provider = "QNNExecutionProvider"
            provider_options["backend_type"] = "htp"
        # TODO(adrianlizarraga): Allow test to run for other compiling EPs (e.g., OpenVINO)

        input_model_path = get_name("nhwc_resize_scales_opset18.onnx")
        output_model_path = os.path.join(self._tmp_dir_path, "model.compiled1.onnx")

        session_options = onnxrt.SessionOptions()
        if provider:
            session_options.add_provider(provider, provider_options)

        model_compiler = onnxrt.ModelCompiler(
            session_options,
            input_model_path,
            embed_compiled_data_into_model=True,
            external_initializers_file_path=None,
        )
        model_compiler.compile_to_file(output_model_path)
        self.assertTrue(os.path.exists(output_model_path))

    def test_compile_to_file_with_input_model_in_buffer(self):
        """
        Tests compiling an input model that is stored in a buffer. The output is saved to a file.
        """
        provider = None
        provider_options = dict()
        if "QNNExecutionProvider" in available_providers:
            provider = "QNNExecutionProvider"
            provider_options["backend_type"] = "htp"
        # TODO(adrianlizarraga): Allow test to run for other compiling EPs (e.g., OpenVINO)

        input_onnx_model = onnx.load(get_name("nhwc_resize_scales_opset18.onnx"))
        input_model_bytes = input_onnx_model.SerializeToString()
        output_model_path = os.path.join(self._tmp_dir_path, "model.compiled2.onnx")

        session_options = onnxrt.SessionOptions()
        if provider:
            session_options.add_provider(provider, provider_options)

        model_compiler = onnxrt.ModelCompiler(
            session_options,
            input_model_bytes,
            embed_compiled_data_into_model=True,
            external_initializers_file_path=None,
        )
        model_compiler.compile_to_file(output_model_path)
        self.assertTrue(os.path.exists(output_model_path))

    def test_compile_from_buffer_to_buffer(self):
        """
        Tests compiling an input model that is stored in a buffer. The output is stored in a buffer too.
        """
        provider = None
        provider_options = dict()
        if "QNNExecutionProvider" in available_providers:
            provider = "QNNExecutionProvider"
            provider_options["backend_type"] = "htp"
        # TODO(adrianlizarraga): Allow test to run for other compiling EPs (e.g., OpenVINO)

        input_onnx_model = onnx.load(get_name("nhwc_resize_scales_opset18.onnx"))
        input_model_bytes = input_onnx_model.SerializeToString()

        session_options = onnxrt.SessionOptions()
        if provider:
            session_options.add_provider(provider, provider_options)

        model_compiler = onnxrt.ModelCompiler(
            session_options,
            input_model_bytes,
            embed_compiled_data_into_model=True,
            external_initializers_file_path=None,
        )
        output_model_bytes = model_compiler.compile_to_bytes()
        self.assertTrue(isinstance(output_model_bytes, bytes))
        self.assertGreater(len(output_model_bytes), 0)

    def test_fail_load_uncompiled_model_and_then_compile(self):
        """
        Tests compiling scenario:
         - Load uncompiled model into session that disables JIT compilation.
         - Expect an error (ModelRequiresCompilation)
         - Compile model and retry creating an inference session successfully.
        """
        if "QNNExecutionProvider" not in available_providers:
            self.skipTest("Skipping test because it needs to run on a compiling EP")

        input_model_path = get_name("nhwc_resize_scales_opset18.onnx")

        session_options = onnxrt.SessionOptions()
        session_options.add_session_config_entry("session.disable_model_compile", "1")  # Disable JIT model compilation!
        session_options.add_provider("QNNExecutionProvider", {"backend_type": "htp"})

        # Session creation should fail with error ORT_MODEL_REQUIRES_COMPILATION because the input model
        # is not compiled and we disabled JIT compilation for this session.
        with self.assertRaises(ModelRequiresCompilation) as context:
            onnxrt.InferenceSession(
                input_model_path,
                sess_options=session_options,
                enable_fallback=False,
            )
        self.assertIn("needs to compile", str(context.exception))

        # Try to compile the model now.
        compiled_model_path = os.path.join(self._tmp_dir_path, "model.compiled3.onnx")
        model_compiler = onnxrt.ModelCompiler(
            session_options,
            input_model_path,
            embed_compiled_data_into_model=True,
            external_initializers_file_path="external_weights.bin",
            external_initializers_size_threshold=128,
        )
        model_compiler.compile_to_file(compiled_model_path)

        self.assertTrue(os.path.exists(compiled_model_path))
        self.assertEqual(session_options.get_session_config_entry("session.disable_model_compile"), "1")
        self.assertTrue(session_options.has_providers())

        # Creating the session with the compiled model should not fail.
        sess = onnxrt.InferenceSession(compiled_model_path, sess_options=session_options)
        self.assertIsNotNone(sess)


if __name__ == "__main__":
    unittest.main(verbosity=1)
