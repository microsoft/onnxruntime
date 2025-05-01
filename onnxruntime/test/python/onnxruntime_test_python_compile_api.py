# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from __future__ import annotations

import os
import platform
import sys
import tempfile
import unittest

from helper import get_name

import onnx
import onnxruntime as onnxrt
from onnxruntime.capi import _pybind_state as C
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

    def test_compile_with_files(self):
        providers = None
        provider_options = None
        if "QNNExecutionProvider" in available_providers:
            providers = ["QNNExecutionProvider"]
            provider_options = [{"backend_type": "htp"}]
    
        so = onnxrt.SessionOptions()
        so.log_severity_level = 1
        so.logid = "TestCompileWithFiles"
        so.graph_optimization_level = onnxrt.GraphOptimizationLevel.ORT_ENABLE_BASIC

        input_model_path = get_name("nhwc_resize_scales_opset18.onnx")
        output_model_path = os.path.join(self._tmp_dir_path, "model.compiled.onnx")

        model_compile_options = onnxrt.ModelCompilationOptions(so)
        model_compile_options.set_input_model(input_model_path)
        model_compile_options.set_output_model_path(output_model_path)
        model_compile_options.set_ep_context_embed_mode(True)
        
        onnxrt.compile_model(model_compile_options,
                             providers=providers,
                             provider_options=provider_options)

        self.assertTrue(os.path.exists(output_model_path))

    def test_compile_with_input_model_in_buffer(self):
        providers = None
        provider_options = None
        if "QNNExecutionProvider" in available_providers:
            providers = ["QNNExecutionProvider"]
            provider_options = [{"backend_type": "htp"}]
    
        so = onnxrt.SessionOptions()
        so.log_severity_level = 1
        so.logid = "TestCompileWithInputBuffer"
        so.graph_optimization_level = onnxrt.GraphOptimizationLevel.ORT_ENABLE_BASIC

        input_onnx_model = onnx.load(get_name("nhwc_resize_scales_opset18.onnx"))
        input_model_bytes = input_onnx_model.SerializeToString()
        output_model_path = os.path.join(self._tmp_dir_path, "model.compiled2.onnx")

        model_compile_options = onnxrt.ModelCompilationOptions(so)
        model_compile_options.set_input_model(input_model_bytes)
        model_compile_options.set_output_model_path(output_model_path)
        model_compile_options.set_ep_context_embed_mode(True)
        
        onnxrt.compile_model(model_compile_options,
                             providers=providers,
                             provider_options=provider_options)

        self.assertTrue(os.path.exists(output_model_path))

    def test_fail_load_uncompiled_model_and_then_compile(self):
        providers = None
        provider_options = None
        if "QNNExecutionProvider" in available_providers:
            providers = ["QNNExecutionProvider"]
            provider_options = [{"backend_type": "htp"}]
    
        input_model_path = get_name("nhwc_resize_scales_opset18.onnx")

        so = onnxrt.SessionOptions()
        so.add_session_config_entry("session.disable_model_compile", "1")  # Disable JIT model compilation!

        # Session creation should fail with error ORT_MODEL_REQUIRES_COMPILATION because the input model
        # is not compiled and we disabled JIT compilation for this session.
        with self.assertRaises(ModelRequiresCompilation) as context:
            sess = onnxrt.InferenceSession(input_model_path,
                                           sess_options=so,
                                           providers=providers,
                                           provider_options=provider_options,
                                           enable_fallback=False)
        self.assertIn("QNNExecutionProvider", str(context.exception))

        # Try to compile the model now.
        compiled_model_path = os.path.join(self._tmp_dir_path, "model.compiled3.onnx")
        model_compile_options = onnxrt.ModelCompilationOptions(so)
        model_compile_options.set_input_model(input_model_path)
        model_compile_options.set_output_model_path(compiled_model_path)
        model_compile_options.set_ep_context_embed_mode(True)
        
        onnxrt.compile_model(model_compile_options,
                             providers=providers,
                             provider_options=provider_options)

        self.assertTrue(os.path.exists(compiled_model_path))
        self.assertEqual(so.get_session_config_entry("session.disable_model_compile"), "1")

        # Creating the session with the compiled model should not fail.
        sess = onnxrt.InferenceSession(compiled_model_path,
                                       sess_options=so,
                                       providers=providers,
                                       provider_options=provider_options)
        self.assertIsNotNone(sess)


if __name__ == "__main__":
    unittest.main(verbosity=1)
