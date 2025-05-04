# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from __future__ import annotations

import os
import platform
import sys
import tempfile
import unittest

import onnx
from helper import get_name

import onnxruntime as onnxrt
from onnxruntime.capi.onnxruntime_pybind11_state import ModelRequiresCompilation

# handle change from python 3.8 and on where loading a dll from the current directory needs to be explicitly allowed.
if platform.system() == "Windows" and sys.version_info.major >= 3 and sys.version_info.minor >= 8:  # noqa: YTT204
    os.add_dll_directory(os.getcwd())

available_providers = list(onnxrt.get_available_providers())


class TestCompileApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp_model_dir = tempfile.TemporaryDirectory(prefix="ort.compile_api_")

        # Note: swap with the commented line if you want to see the models in local test dir.
        cls._tmp_dir_path = cls._tmp_model_dir.name
        # cls._tmp_dir_path = "."

    @classmethod
    def tearDownClass(cls):
        cls._tmp_model_dir.cleanup()

    def test_compile_with_files_prefer_npu_policy(self):
        if "QNNExecutionProvider" not in available_providers:
            self.skipTest("Skipping test because it needs to run on QNN EP")

        if sys.platform != "win32":
            self.skipTest("Skipping test because provider selection policies are only supported on Windows")

        ep_lib_path = "onnxruntime_providers_qnn.dll"
        ep_registration_name = "QNNExecutionProvider"
        #onnxrt.register_execution_provider_library(ep_registration_name, os.path.realpath(ep_lib_path))
        onnxrt.register_execution_provider_library(ep_registration_name, ep_lib_path)

        session_options = onnxrt.SessionOptions()
        session_options.log_severity_level = 1
        session_options.logid = "TestCompileWithFiles"
        session_options.graph_optimization_level = onnxrt.GraphOptimizationLevel.ORT_ENABLE_BASIC
        session_options.set_provider_selection_policy(onnxrt.OrtExecutionProviderDevicePolicy.PREFER_NPU)

        input_model_path = get_name("nhwc_resize_scales_opset18.onnx")
        output_model_path = os.path.join(self._tmp_dir_path, "model.compiled.onnx")

        model_compiler = onnxrt.ModelCompiler(
            session_options,
            input_model_path,
            embed_compiled_data_into_model=True,
            external_initializers_file_path=None,
        )
        model_compiler.compile_to_file(output_model_path)
        self.assertTrue(os.path.exists(output_model_path))

    def test_compile_with_files(self):
        providers = None
        provider_options = None
        if "QNNExecutionProvider" in available_providers:
            providers = ["QNNExecutionProvider"]
            provider_options = [{"backend_type": "htp"}]
        # TODO(adrianlizarraga): Allow test to run for other compiling EPs (e.g., OpenVINO)

        so = onnxrt.SessionOptions()
        so.log_severity_level = 1
        so.logid = "TestCompileWithFiles"
        so.graph_optimization_level = onnxrt.GraphOptimizationLevel.ORT_ENABLE_BASIC

        input_model_path = get_name("nhwc_resize_scales_opset18.onnx")
        output_model_path = os.path.join(self._tmp_dir_path, "model.compiled.onnx")

        model_compile_options = onnxrt.ModelCompilationOptions(so)
        model_compile_options.set_providers(providers=providers, provider_options=provider_options)
        model_compile_options.set_input_model(input_model_path)
        model_compile_options.set_output_model_path(output_model_path)
        model_compile_options.set_ep_context_embed_mode(True)

        onnxrt.compile_model(model_compile_options)

        self.assertTrue(os.path.exists(output_model_path))

    def test_compile_with_input_model_in_buffer(self):
        providers = None
        provider_options = None
        if "QNNExecutionProvider" in available_providers:
            providers = ["QNNExecutionProvider"]
            provider_options = [{"backend_type": "htp"}]
        # TODO(adrianlizarraga): Allow test to run for other compiling EPs (e.g., OpenVINO)

        so = onnxrt.SessionOptions()
        so.log_severity_level = 1
        so.logid = "TestCompileWithInputBuffer"
        so.graph_optimization_level = onnxrt.GraphOptimizationLevel.ORT_ENABLE_BASIC

        input_onnx_model = onnx.load(get_name("nhwc_resize_scales_opset18.onnx"))
        input_model_bytes = input_onnx_model.SerializeToString()
        output_model_path = os.path.join(self._tmp_dir_path, "model.compiled2.onnx")

        model_compile_options = onnxrt.ModelCompilationOptions(so)
        model_compile_options.set_providers(providers=providers, provider_options=provider_options)
        model_compile_options.set_input_model(input_model_bytes)
        model_compile_options.set_output_model_path(output_model_path)
        model_compile_options.set_ep_context_embed_mode(True)

        onnxrt.compile_model(model_compile_options)

        self.assertTrue(os.path.exists(output_model_path))

    def test_fail_load_uncompiled_model_and_then_compile(self):
        if "QNNExecutionProvider" not in available_providers:
            self.skipTest("Skipping test because it needs to run on a compiling EP")

        providers = ["QNNExecutionProvider"]
        provider_options = [{"backend_type": "htp"}]
        input_model_path = get_name("nhwc_resize_scales_opset18.onnx")

        so = onnxrt.SessionOptions()
        so.add_session_config_entry("session.disable_model_compile", "1")  # Disable JIT model compilation!

        # Session creation should fail with error ORT_MODEL_REQUIRES_COMPILATION because the input model
        # is not compiled and we disabled JIT compilation for this session.
        with self.assertRaises(ModelRequiresCompilation) as context:
            onnxrt.InferenceSession(
                input_model_path,
                sess_options=so,
                providers=providers,
                provider_options=provider_options,
                enable_fallback=False,
            )
        self.assertIn("needs to compile", str(context.exception))

        # Try to compile the model now.
        compiled_model_path = os.path.join(self._tmp_dir_path, "model.compiled3.onnx")
        model_compile_options = onnxrt.ModelCompilationOptions(so)
        model_compile_options.set_providers(providers=providers, provider_options=provider_options)
        model_compile_options.set_input_model(input_model_path)
        model_compile_options.set_output_model_path(compiled_model_path)
        model_compile_options.set_ep_context_embed_mode(True)
        model_compile_options.set_output_model_external_initializers_file(
            "external_weights.bin",
            external_initializers_size_threshold=128,
        )

        onnxrt.compile_model(model_compile_options)

        self.assertTrue(os.path.exists(compiled_model_path))
        self.assertEqual(so.get_session_config_entry("session.disable_model_compile"), "1")

        # Creating the session with the compiled model should not fail.
        sess = onnxrt.InferenceSession(
            compiled_model_path, sess_options=so, providers=providers, provider_options=provider_options
        )
        self.assertIsNotNone(sess)


if __name__ == "__main__":
    unittest.main(verbosity=1)
