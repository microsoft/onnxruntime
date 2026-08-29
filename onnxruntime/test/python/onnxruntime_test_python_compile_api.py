# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from __future__ import annotations

import gc
import os
import platform
import subprocess
import sys
import threading
import unittest
import weakref
from collections.abc import Sequence

import onnx
from autoep_helper import AutoEpTestCase
from helper import get_name, get_shared_library_filename_for_platform

import onnxruntime as onnxrt
from onnxruntime.capi.onnxruntime_inference_collection import ModelCompiler
from onnxruntime.capi.onnxruntime_pybind11_state import Fail, InvalidArgument, ModelRequiresCompilation

# handle change from python 3.8 and on where loading a dll from the current directory needs to be explicitly allowed.
if platform.system() == "Windows" and sys.version_info.major >= 3 and sys.version_info.minor >= 8:  # noqa: YTT204
    os.add_dll_directory(os.getcwd())

available_providers = list(onnxrt.get_available_providers())


class TestCompileApi(AutoEpTestCase):
    def test_callback_finalizers_can_reenter_configuration(self):
        child_code = r"""
import sys

import onnxruntime as ort

events = []


class ReadCallback:
    def __init__(self, options):
        self.options = options

    def __call__(self, _name, _output):
        pass

    def __del__(self):
        self.options.clear_ep_context_data_read_func()
        events.append("read")


class SelectionCallback:
    def __init__(self, options):
        self.options = options

    def __call__(self, _devices, _model_metadata, _runtime_metadata, _max_selections):
        return []

    def __del__(self):
        self.options.set_provider_selection_policy(ort.OrtExecutionProviderDevicePolicy.PREFER_GPU)
        events.append("selection")


class WriteCallback:
    def __init__(self, compiler):
        self.compiler = compiler

    def __call__(self, _name, _data):
        pass

    def __del__(self):
        self.compiler.clear_ep_context_data_write_func()
        events.append("write")


read_options = ort.SessionOptions()
read_callback = ReadCallback(read_options)
read_options.set_ep_context_data_read_func(read_callback, 1)
del read_callback
read_options.clear_ep_context_data_read_func()

selection_options = ort.SessionOptions()
selection_callback = SelectionCallback(selection_options)
selection_options.set_provider_selection_policy_delegate(selection_callback)
del selection_callback
selection_options.set_provider_selection_policy(ort.OrtExecutionProviderDevicePolicy.PREFER_GPU)

compiler = ort.ModelCompiler(ort.SessionOptions(), sys.argv[1], embed_compiled_data_into_model=True)
write_callback = WriteCallback(compiler)
compiler.set_ep_context_data_write_func(write_callback)
del write_callback
compiler.clear_ep_context_data_write_func()

assert events == ["read", "selection", "write"], events
"""
        result = subprocess.run(
            [sys.executable, "-c", child_code, get_name("mul_1.onnx")],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")

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

    def test_compile_shared_resources_plugin_ep(self):
        """
        Test compiling two example models using weight sharing (via example plugin EP)
        """
        ep_lib_path = get_shared_library_filename_for_platform("example_plugin_ep")
        try:
            ep_lib_path = get_name(ep_lib_path)
        except FileNotFoundError:
            self.skipTest(f"Skipping test because EP library '{ep_lib_path}' cannot be found")

        ep_name = "example_ep"
        self.register_execution_provider_library(ep_name, os.path.realpath(ep_lib_path))

        ep_device = next((d for d in onnxrt.get_ep_devices() if d.ep_name == ep_name), None)
        self.assertIsNotNone(ep_device)

        input_models = [get_name("add_mul_add.onnx"), get_name("mul_1.onnx")]
        output_models = [
            os.path.join(self._tmp_dir_path, "output_model_0_ctx.onnx"),
            os.path.join(self._tmp_dir_path, "output_model_1_ctx.onnx"),
        ]

        num_models = len(input_models)
        session_options = onnxrt.SessionOptions()

        # Set option that tells EP to share resources (e.g., weights) across sessions. The example plugin EP
        # doesn't actually do anything special, but we do this to test the API
        session_options.add_session_config_entry("ep.share_ep_contexts", "1")
        session_options.add_provider_for_devices([ep_device], {})

        # Compile individual models
        for i in range(num_models):
            if i == num_models - 1:
                # Tell EP that this is the last session that will be sharing resources.
                session_options.add_session_config_entry("ep.stop_share_ep_contexts", "1")

            model_compiler = onnxrt.ModelCompiler(
                session_options,
                input_models[i],
                embed_compiled_data_into_model=False,
            )
            model_compiler.compile_to_file(output_models[i])
            self.assertTrue(os.path.exists(output_models[i]))

        self.unregister_execution_provider_library(ep_name)

    def test_external_ep_context_data_callbacks(self):
        ep_lib_path = get_shared_library_filename_for_platform("example_plugin_ep")
        try:
            ep_lib_path = get_name(ep_lib_path)
        except FileNotFoundError:
            self.skipTest(f"Skipping test because EP library '{ep_lib_path}' cannot be found")

        ep_name = "python_ep_context"
        self.register_execution_provider_library(ep_name, os.path.realpath(ep_lib_path))
        try:
            ep_device = next((device for device in onnxrt.get_ep_devices() if device.ep_name == ep_name), None)
            self.assertIsNotNone(ep_device)
            assert ep_device is not None

            input_model_path = get_name("mul_1.onnx")
            write_count = 0
            read_count = 0
            context_name = ""
            context_payload = b""
            retained_write_data = None
            retained_read_output = None
            retained_failed_write_data = None
            retained_failed_read_output = None

            write_failure_options = onnxrt.SessionOptions()
            write_failure_options.add_provider_for_devices([ep_device], {})
            write_failure_compiler: ModelCompiler = onnxrt.ModelCompiler(
                write_failure_options, input_model_path, embed_compiled_data_into_model=False
            )

            def fail_write(_name: str, data: onnxrt.OrtEpContextData):
                nonlocal retained_failed_write_data
                retained_failed_write_data = data
                raise ValueError("synthetic Python EPContext write failure")

            write_failure_compiler.set_ep_context_data_write_func(fail_write)
            with self.assertRaises(Fail) as context:
                write_failure_compiler.compile_to_bytes()
            self.assertIn("synthetic Python EPContext write failure", str(context.exception))
            assert retained_failed_write_data is not None
            with self.assertRaises(RuntimeError):
                retained_failed_write_data.read()

            compile_options = onnxrt.SessionOptions()
            compile_options.add_provider_for_devices([ep_device], {})
            compiler: ModelCompiler = onnxrt.ModelCompiler(
                compile_options, input_model_path, embed_compiled_data_into_model=False
            )
            compiler_ref = weakref.ref(compiler)

            def write_context(name: str, data: onnxrt.OrtEpContextData):
                nonlocal write_count, context_name, context_payload, retained_write_data
                write_count += 1
                context_name = name
                retained_write_data = data
                model_compiler = compiler_ref()
                self.assertIsNotNone(model_compiler)
                assert model_compiler is not None
                with self.assertRaises(RuntimeError) as context:
                    model_compiler.clear_ep_context_data_write_func()
                self.assertIn("while compilation is in progress", str(context.exception))
                midpoint = data.size // 2
                context_payload = data.read(0, midpoint) + data.read(midpoint)

            compiler.set_ep_context_data_write_func(write_context)
            compiled_model = compiler.compile_to_bytes()

            self.assertEqual(write_count, 1)
            self.assertTrue(context_name)
            self.assertTrue(context_payload)
            assert retained_write_data is not None
            self.assertFalse(os.path.exists(context_name))
            with self.assertRaises(RuntimeError):
                retained_write_data.read()

            missing_output_options = onnxrt.SessionOptions()
            missing_output_options.set_ep_context_data_read_func(lambda _name, _output: None, len(context_payload))
            missing_output_options.add_provider_for_devices([ep_device], {})
            with self.assertRaises(InvalidArgument) as context:
                onnxrt.InferenceSession(compiled_model, sess_options=missing_output_options)
            self.assertIn("must allocate its output buffer", str(context.exception))
            self.assertFalse(os.path.exists(context_name))

            oversized_options = onnxrt.SessionOptions()
            oversized_options.set_ep_context_data_read_func(
                lambda _name, output: output.allocate(len(context_payload) + 1), len(context_payload)
            )
            oversized_options.add_provider_for_devices([ep_device], {})
            with self.assertRaises(Fail) as context:
                onnxrt.InferenceSession(compiled_model, sess_options=oversized_options)
            self.assertIn("configured maximum size", str(context.exception))
            self.assertFalse(os.path.exists(context_name))

            failure_options = onnxrt.SessionOptions()

            def fail_read(_name: str, output: onnxrt.OrtEpContextDataBuffer):
                nonlocal retained_failed_read_output
                retained_failed_read_output = output
                output.allocate(len(context_payload))
                raise ValueError("synthetic Python EPContext read failure")

            failure_options.set_ep_context_data_read_func(fail_read, len(context_payload))
            failure_options.add_provider_for_devices([ep_device], {})
            with self.assertRaises(Fail) as context:
                onnxrt.InferenceSession(compiled_model, sess_options=failure_options)
            self.assertIn("synthetic Python EPContext read failure", str(context.exception))
            self.assertFalse(os.path.exists(context_name))
            assert retained_failed_read_output is not None
            with self.assertRaises(RuntimeError):
                retained_failed_read_output.write(b"x")

            load_options = onnxrt.SessionOptions()

            def read_context(name: str, output: onnxrt.OrtEpContextDataBuffer):
                nonlocal read_count, retained_read_output
                read_count += 1
                self.assertEqual(name, context_name)
                retained_read_output = output
                output.allocate(len(context_payload))
                midpoint = len(context_payload) // 2
                output.write(context_payload[:midpoint])
                output.write(context_payload[midpoint:], midpoint)

            load_options.set_ep_context_data_read_func(read_context, len(context_payload))
            load_options.add_provider_for_devices([ep_device], {})
            session = onnxrt.InferenceSession(compiled_model, sess_options=load_options)
            self.assertIsNotNone(session)
            self.assertEqual(read_count, 1)
            assert retained_read_output is not None
            with self.assertRaises(RuntimeError):
                retained_read_output.write(b"x")

            propagated_load_options = session.get_session_options()
            propagated_load_options.add_provider_for_devices([ep_device], {})
            load_options.clear_ep_context_data_read_func()
            del load_options, session
            gc.collect()

            propagated_session = onnxrt.InferenceSession(compiled_model, sess_options=propagated_load_options)
            self.assertIsNotNone(propagated_session)
            self.assertEqual(read_count, 2)

            compile_read_count = 0
            source_compile_options = onnxrt.SessionOptions()

            def read_context_for_compile(name: str, output: onnxrt.OrtEpContextDataBuffer):
                nonlocal compile_read_count
                compile_read_count += 1
                self.assertEqual(name, context_name)
                output.allocate(len(context_payload))
                output.write(context_payload)

            read_context_for_compile_ref = weakref.ref(read_context_for_compile)
            source_compile_options.set_ep_context_data_read_func(read_context_for_compile, len(context_payload))
            source_compile_options.add_provider_for_devices([ep_device], {})
            retained_compiler: ModelCompiler = onnxrt.ModelCompiler(
                source_compile_options, input_model_path, embed_compiled_data_into_model=True
            )
            source_compile_options.clear_ep_context_data_read_func()
            del source_compile_options, read_context_for_compile
            gc.collect()

            self.assertIsNotNone(read_context_for_compile_ref())
            self.assertTrue(retained_compiler.compile_to_bytes())
            self.assertEqual(compile_read_count, 0)
            self.assertFalse(os.path.exists(context_name))

            del retained_compiler
            gc.collect()
            self.assertIsNone(read_context_for_compile_ref())

            concurrent_options = onnxrt.SessionOptions()
            concurrent_options.add_provider_for_devices([ep_device], {})
            concurrent_compiler: ModelCompiler = onnxrt.ModelCompiler(
                concurrent_options, input_model_path, embed_compiled_data_into_model=False
            )
            callback_started = threading.Event()
            release_callback = threading.Event()
            compile_errors: list[BaseException] = []
            concurrent_compiler_ref = weakref.ref(concurrent_compiler)

            def blocking_write(_name: str, _data: onnxrt.OrtEpContextData):
                callback_started.set()
                self.assertTrue(release_callback.wait(timeout=10))

            def compile_in_thread():
                try:
                    model_compiler = concurrent_compiler_ref()
                    if model_compiler is None:
                        raise RuntimeError("ModelCompiler was released before compilation started")
                    model_compiler.compile_to_bytes()
                except BaseException as exception:
                    compile_errors.append(exception)

            concurrent_compiler.set_ep_context_data_write_func(blocking_write)
            compile_thread = threading.Thread(target=compile_in_thread)
            compile_thread.start()
            try:
                self.assertTrue(callback_started.wait(timeout=10))
                with self.assertRaises(Fail) as context:
                    concurrent_compiler.compile_to_bytes()
                self.assertIn("Compilation is already in progress", str(context.exception))
            finally:
                release_callback.set()
                compile_thread.join(timeout=10)

            self.assertFalse(compile_thread.is_alive())
            self.assertEqual(compile_errors, [])

            del propagated_session, concurrent_compiler, compiler, write_failure_compiler
            gc.collect()
        finally:
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
        delegate_ref = weakref.ref(my_delegate)

        model_compiler = onnxrt.ModelCompiler(
            session_options,
            input_model_path,
            embed_compiled_data_into_model=True,
            external_initializers_file_path=None,
        )
        del session_options, my_delegate
        gc.collect()
        self.assertIsNotNone(delegate_ref())

        model_compiler.compile_to_file(output_model_path)
        self.assertTrue(os.path.exists(output_model_path))

        del model_compiler
        gc.collect()
        self.assertIsNone(delegate_ref())

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

    def test_compile_from_file_to_stream(self):
        """
        Tests compiling a model (from files) to an output stream using a custom write functor.
        """
        provider = None
        provider_options = dict()
        if "QNNExecutionProvider" in available_providers:
            provider = "QNNExecutionProvider"
            provider_options["backend_type"] = "htp"

        input_model_path = get_name("nhwc_resize_scales_opset18.onnx")
        output_model_path = os.path.join(self._tmp_dir_path, "model.compiled.stream.onnx")

        with open(output_model_path, "wb") as output_fd:
            # User's custom write functor. Writes the model to a file.
            def my_write_func(buffer: bytes):
                self.assertGreater(len(buffer), 0)
                output_fd.write(buffer)

            session_options = onnxrt.SessionOptions()
            if provider:
                session_options.add_provider(provider, provider_options)

            model_compiler = onnxrt.ModelCompiler(
                session_options,
                input_model_path,
                embed_compiled_data_into_model=True,
                external_initializers_file_path=None,
            )
            model_compiler.compile_to_stream(my_write_func)

        self.assertTrue(os.path.exists(output_model_path))

    def test_compile_to_stream_that_raises_exception(self):
        """
        Tests compiling a model to an output stream that always raises an exception.
        """
        input_model_path = get_name("nhwc_resize_scales_opset18.onnx")

        # User's custom write functor that raises an exception.
        test_py_error_message = "My Python Error"

        def my_write_func(buffer: bytes):
            self.assertGreater(len(buffer), 0)
            raise ValueError(test_py_error_message)

        session_options = onnxrt.SessionOptions()
        model_compiler = onnxrt.ModelCompiler(
            session_options,
            input_model_path,
            embed_compiled_data_into_model=True,
            external_initializers_file_path=None,
        )

        # Try to compile and expect ORT to raise a Fail exception that contains our message.
        with self.assertRaises(Fail) as context:
            model_compiler.compile_to_stream(my_write_func)
        self.assertIn(test_py_error_message, str(context.exception))

    def test_compile_with_basic_initializer_location_func(self):
        """
        Tests compiling a model using a custom initializer handler that stores initializers
        in an external file.
        """
        input_model_path = get_name("conv_qdq_external_ini.onnx")
        output_model_path = os.path.join(self._tmp_dir_path, "conv_qdq.init_handler.onnx")
        initializer_file_path = os.path.join(self._tmp_dir_path, "conv_qdq.init_handler.bin")

        if os.path.exists(output_model_path):
            os.remove(output_model_path)

        if os.path.exists(initializer_file_path):
            os.remove(initializer_file_path)

        with open(initializer_file_path, "wb") as ext_init_file:

            def store_large_initializer_externally(
                initializer_name: str,
                initializer_value: onnxrt.OrtValue,
                external_info: onnxrt.OrtExternalInitializerInfo | None,
            ) -> onnxrt.OrtExternalInitializerInfo | None:
                self.assertTrue(initializer_name)  # Should have valid name
                byte_size = initializer_value.tensor_size_in_bytes()

                if byte_size < 64:
                    return None  # Store small initializer within compiled model.

                # Else, write initializer to new external file.
                value_np = initializer_value.numpy()
                file_offset = ext_init_file.tell()
                ext_init_file.write(value_np.tobytes())
                return onnxrt.OrtExternalInitializerInfo(initializer_file_path, file_offset, byte_size)

            session_options = onnxrt.SessionOptions()
            model_compiler = onnxrt.ModelCompiler(
                session_options,
                input_model_path,
                embed_compiled_data_into_model=True,
                external_initializers_file_path=None,
                get_initializer_location_func=store_large_initializer_externally,
            )
            model_compiler.compile_to_file(output_model_path)

        self.assertTrue(os.path.exists(output_model_path))
        self.assertTrue(os.path.exists(initializer_file_path))

    def test_compile_with_initializer_func_that_reuses(self):
        """
        Tests compiling a model using a custom initializer handler that reuses external initializer files.
        """
        input_model_path = get_name("conv_qdq_external_ini.onnx")
        output_model_path = os.path.join(self._tmp_dir_path, "conv_qdq.init_handler_reuse.onnx")

        if os.path.exists(output_model_path):
            os.remove(output_model_path)

        # Function that reuses external initializer files for the compiled model.
        def reuse_external_initializers(
            initializer_name: str,
            initializer_value: onnxrt.OrtValue,
            external_info: onnxrt.OrtExternalInitializerInfo | None,
        ) -> onnxrt.OrtExternalInitializerInfo | None:
            self.assertTrue(initializer_name)  # Should have valid name
            self.assertNotEqual(initializer_value.data_ptr(), 0)
            self.assertGreater(initializer_value.tensor_size_in_bytes(), 0)
            if external_info is not None:
                # Original initializer is stored externally.
                # Make the initializer in the compiled model use the same external file
                return external_info

            return None  # Otherwise, make a copy of the initializer and store it within compiled model.

        session_options = onnxrt.SessionOptions()
        model_compiler = onnxrt.ModelCompiler(
            session_options,
            input_model_path,
            embed_compiled_data_into_model=True,
            external_initializers_file_path=None,
            get_initializer_location_func=reuse_external_initializers,
        )
        model_compiler.compile_to_file(output_model_path)
        self.assertTrue(os.path.exists(output_model_path))

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
