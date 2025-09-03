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
from onnxruntime.capi.onnxruntime_pybind11_state import Fail, ModelRequiresCompilation

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

    def test_compile_flags_error_if_no_compiled_nodes(self):
        """
        Tests specifying an additional flag (OrtCompileApiFlags.ERROR_IF_NO_NODES_COMPILED) that
        makes compiling return an error if no compiled nodes are generated (e.g., by using CPU EP).
        """
        input_model_path = get_name("nhwc_resize_scales_opset18.onnx")
        output_model_path = os.path.join(self._tmp_dir_path, "model.compiled1.onnx")

        session_options = onnxrt.SessionOptions()
        model_compiler = onnxrt.ModelCompiler(
            session_options,
            input_model_path,
            embed_compiled_data_into_model=True,
            external_initializers_file_path=None,
            flags=onnxrt.OrtCompileApiFlags.ERROR_IF_NO_NODES_COMPILED,
        )

        # Compiling should raise a Fail exception and the output model should not be generated
        with self.assertRaises(Fail) as context:
            model_compiler.compile_to_file(output_model_path)
        self.assertIn("Unable to compile any nodes", str(context.exception))
        self.assertFalse(os.path.exists(output_model_path))

    def test_compile_flags_error_if_output_file_exists(self):
        """
        Tests specifying an additional flag (OrtCompileApiFlags.ERROR_IF_OUTPUT_FILE_EXISTS) that
        makes compiling return an error the output model file already exists.
        """
        input_model_path = get_name("nhwc_resize_scales_opset18.onnx")
        output_model_path = os.path.join(self._tmp_dir_path, "model.compiled1.onnx")

        # Compile the first time (should be fine)
        session_options = onnxrt.SessionOptions()
        model_compiler = onnxrt.ModelCompiler(
            session_options,
            input_model_path,
            embed_compiled_data_into_model=True,
            external_initializers_file_path=None,
            flags=onnxrt.OrtCompileApiFlags.ERROR_IF_OUTPUT_FILE_EXISTS,
        )

        model_compiler.compile_to_file(output_model_path)
        self.assertTrue(os.path.exists(output_model_path))  # Output model was generated

        # Compiling again should raise a Fail exception saying that the model already exists.
        with self.assertRaises(Fail) as context:
            model_compiler.compile_to_file(output_model_path)
        self.assertIn("exists already", str(context.exception))

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

    def test_compile_graph_optimization_level(self):
        """
        Tests compiling a model with no optimizations (default) vs all optimizations.
        """
        input_model_path = get_name("test_cast_back_to_back_non_const_mixed_types_origin.onnx")
        output_model_path_0 = os.path.join(self._tmp_dir_path, "cast.disable_all.compiled.onnx")
        output_model_path_1 = os.path.join(self._tmp_dir_path, "cast.enable_all.compiled.onnx")

        # Local function that compiles a model with a given graph optimization level and returns
        # the count of operator types in the compiled model.
        def compile_and_get_op_counts(
            output_model_path: str,
            graph_opt_level: onnxrt.GraphOptimizationLevel | None,
        ) -> dict[str, int]:
            session_options = onnxrt.SessionOptions()
            if graph_opt_level is not None:
                model_compiler = onnxrt.ModelCompiler(
                    session_options,
                    input_model_path,
                    graph_optimization_level=graph_opt_level,
                )
            else:
                # graph optimization level defaults to ORT_DISABLE_ALL if not provided.
                model_compiler = onnxrt.ModelCompiler(session_options, input_model_path)

            model_compiler.compile_to_file(output_model_path)
            self.assertTrue(os.path.exists(output_model_path))

            model: onnx.ModelProto = onnx.load(get_name(output_model_path))
            op_counts = {}
            for node in model.graph.node:
                if node.op_type not in op_counts:
                    op_counts[node.op_type] = 1
                else:
                    op_counts[node.op_type] += 1

            return op_counts

        # Compile model on CPU with no graph optimizations (default).
        # Model should have 9 Casts
        op_counts_0 = compile_and_get_op_counts(output_model_path_0, graph_opt_level=None)
        self.assertEqual(op_counts_0["Cast"], 9)

        # Compile model on CPU with ALL graph optimizations.
        # Model should have less casts (optimized out)
        op_counts_1 = compile_and_get_op_counts(
            output_model_path_1, graph_opt_level=onnxrt.GraphOptimizationLevel.ORT_ENABLE_BASIC
        )
        self.assertEqual(op_counts_1["Cast"], 8)

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
