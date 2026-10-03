# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from __future__ import annotations

import copy
import ctypes
import gc
import importlib.util
import os
import pathlib
import platform
import queue
import sys
import threading
import time
import unittest
import weakref

import numpy as np
from helper import get_name

import onnxruntime as onnxrt
from onnxruntime.capi import _pybind_state as C
from onnxruntime.capi.onnxruntime_inference_collection import (
    _GRAPH_ANNOTATION_SKIP,
    _graph_annotation_id,
    _has_foreign_webgpu_context,
)
from onnxruntime.capi.onnxruntime_pybind11_state import Fail, OrtValueVector, RunOptions

# handle change from python 3.8 and on where loading a dll from the current directory needs to be explicitly allowed.
if platform.system() == "Windows" and sys.version_info.major >= 3 and sys.version_info.minor >= 8:  # noqa: YTT204
    os.add_dll_directory(os.getcwd())

available_providers = [
    (
        ep,
        {"enable_cann_subgraph": True},
    )
    if ep == "CANNExecutionProvider"
    else ep
    for ep in onnxrt.get_available_providers()
]

# TVM EP doesn't support:
# * calling Run() on different threads using the same session object
# * symbolic inputs
# * string inputs
# * byte type inputs
# * object type inputs
# * void type inputs
# * SequenceConstruct operator
# * custom operators
# * testSequenceInsert
# * testSequenceLength
available_providers_without_tvm = [
    ep for ep in available_providers if (ep[0] if isinstance(ep, tuple) else ep) not in {"TvmExecutionProvider"}
]

available_providers_without_tvm_and_tensorrt = [
    ep
    for ep in available_providers_without_tvm
    if (ep[0] if isinstance(ep, tuple) else ep) not in {"TensorrtExecutionProvider"}
]


def device_ortvalue_from_numpy(session, array, device_type, device_id=0, vendor_id=-1):
    """Allocate a device OrtValue from the session allocator and upload host data into it.

    This is the supported composition: allocate with Session.create_ortvalue_from_shape_and_type,
    then upload with the environment-level onnxruntime.copy_tensors. For the default WebGPU context
    the session allocator and the environment data transfer both resolve context 0.
    """
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    device_value = session.create_ortvalue_from_shape_and_type(
        array.shape, array.dtype, device_type, device_id, vendor_id
    )
    onnxrt.copy_tensors([onnxrt.OrtValue.ortvalue_from_numpy(array)], [device_value])
    return device_value


class TestInferenceSession(unittest.TestCase):
    def run_model(self, session_object, run_options):
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        input_name = session_object.get_inputs()[0].name
        res = session_object.run([], {input_name: x}, run_options=run_options)
        output_expected = np.array([[1.0, 4.0], [9.0, 16.0], [25.0, 36.0]], dtype=np.float32)
        np.testing.assert_allclose(res[0], output_expected, rtol=1e-05, atol=1e-08)

    def run_model_with_input(self, session_object, input_name, input_value, iter_num, queue):
        for _ in range(iter_num):
            predict = session_object.run(None, {input_name: input_value})[0]
            queue.put(max(predict.flatten().tolist()))

    def load_cuda_lib(self):
        cuda_lib = None
        if sys.platform == "win32":
            cuda_lib = "cuda.dll"
        elif sys.platform == "linux":
            cuda_lib = "libcuda.so"
        elif sys.platform == "darwin":
            cuda_lib = "libcuda.dylib"

        if cuda_lib is not None:
            try:
                return ctypes.CDLL(cuda_lib)
            except OSError:
                pass
        return None

    def cuda_device_count(self, cuda_lib):
        if cuda_lib is None:
            return -1
        num_device = ctypes.c_int()
        cuda_lib.cuInit(0)
        result = cuda_lib.cuDeviceGetCount(ctypes.byref(num_device))
        if result != 0:
            error_str = ctypes.c_char_p()
            cuda_lib.cuGetErrorString(result, ctypes.byref(error_str))
            print(f"cuDeviceGetCount failed with error code {result}: {error_str.value.decode()}")
            return -1
        return num_device.value

    def test_tvm_imported(self):
        if "TvmExecutionProvider" not in onnxrt.get_available_providers():
            return
        import tvm  # noqa: PLC0415

        self.assertTrue(tvm is not None)

    def test_get_version_string(self):
        self.assertIsNot(onnxrt.get_version_string(), None)

    def test_get_build_info(self):
        self.assertIsNot(onnxrt.get_build_info(), None)
        self.assertIn("Build Info", onnxrt.get_build_info())

    def test_model_serialization(self):
        try:
            so = onnxrt.SessionOptions()
            so.log_severity_level = 1
            so.logid = "TestModelSerialization"
            so.optimized_model_filepath = "./PythonApiTestOptimizedModel.onnx"
            onnxrt.InferenceSession(get_name("mul_1.onnx"), sess_options=so)
            self.assertTrue(os.path.isfile(so.optimized_model_filepath))
            os.remove(so.optimized_model_filepath)
        except Fail as onnxruntime_error:
            if (
                str(onnxruntime_error) == "[ONNXRuntimeError] : 1 : FAIL : Unable to serialize model as it contains"
                " compiled nodes. Please disable any execution providers which generate compiled nodes."
            ):
                pass
            else:
                raise onnxruntime_error

    def test_model_serialization_with_external_initializers(self):
        try:
            so = onnxrt.SessionOptions()
            so.log_severity_level = 1
            so.logid = "TestModelSerializationWithExternalInitializers"
            so.optimized_model_filepath = "./model_with_external_initializers.onnx"
            external_initializers_file = "external_initializers.bin"
            so.add_session_config_entry(
                "session.optimized_model_external_initializers_file_name", external_initializers_file
            )
            so.add_session_config_entry("session.optimized_model_external_initializers_min_size_in_bytes", "100")
            onnxrt.InferenceSession(get_name("mnist.onnx"), sess_options=so)
            self.assertTrue(os.path.isfile(so.optimized_model_filepath))
            self.assertTrue(os.path.isfile(external_initializers_file))
            os.remove(so.optimized_model_filepath)
            os.remove(external_initializers_file)
        except Fail as onnxruntime_error:
            if (
                str(onnxruntime_error) == "[ONNXRuntimeError] : 1 : FAIL : Unable to serialize model as it contains"
                " compiled nodes. Please disable any execution providers which generate compiled nodes."
            ):
                pass
            else:
                raise onnxruntime_error

    def test_model_serialization_with_external_initializers_to_directory(self):
        try:
            so = onnxrt.SessionOptions()
            so.log_severity_level = 1
            so.logid = "TestModelSerializationWithExternalInitializersToDirectory"
            directory = "./testdata/"
            so.optimized_model_filepath = os.path.join(directory, "model_with_external_initializers_in_dir.onnx")
            external_initializers_file = "external_initializers_in_dir.bin"
            so.add_session_config_entry(
                "session.optimized_model_external_initializers_file_name", external_initializers_file
            )
            so.add_session_config_entry("session.optimized_model_external_initializers_min_size_in_bytes", "100")
            onnxrt.InferenceSession(get_name("mnist.onnx"), sess_options=so)
            self.assertTrue(os.path.isfile(so.optimized_model_filepath))
            self.assertTrue(os.path.isfile(os.path.join(directory, external_initializers_file)))
            os.remove(so.optimized_model_filepath)
            os.remove(os.path.join(directory, external_initializers_file))
        except Fail as onnxruntime_error:
            if (
                str(onnxruntime_error) == "[ONNXRuntimeError] : 1 : FAIL : Unable to serialize model as it contains"
                " compiled nodes. Please disable any execution providers which generate compiled nodes."
            ):
                pass
            else:
                raise onnxruntime_error

    def test_model_serialization_with_original_external_initializers_to_directory(self):
        try:
            so = onnxrt.SessionOptions()
            so.log_severity_level = 1
            so.logid = "TestModelSerializationWithOriginalExternalInitializersToDirectory"
            directory = "./testdata/"
            so.optimized_model_filepath = os.path.join(directory, "model_opt_with_ext_data.onnx")
            external_initializers_file = "model_opt_with_ext_data.bin"
            so.add_session_config_entry(
                "session.optimized_model_external_initializers_file_name", external_initializers_file
            )
            so.add_session_config_entry("session.optimized_model_external_initializers_min_size_in_bytes", "20")
            onnxrt.InferenceSession(get_name("model_with_orig_ext_data.onnx"), sess_options=so)
            self.assertTrue(os.path.isfile(so.optimized_model_filepath))
            self.assertTrue(os.path.isfile(os.path.join(directory, external_initializers_file)))
            os.remove(so.optimized_model_filepath)
            os.remove(os.path.join(directory, external_initializers_file))
        except Fail as onnxruntime_error:
            if (
                str(onnxruntime_error) == "[ONNXRuntimeError] : 1 : FAIL : Unable to serialize model as it contains"
                " compiled nodes. Please disable any execution providers which generate compiled nodes."
            ):
                pass
            else:
                raise onnxruntime_error

    def test_model_serialization_with_original_external_initializers_to_current_directory(self):
        optimized_model_filepath = "model_opt_with_ext_data_1.onnx"
        external_initializers_file = "model_opt_with_ext_data_1.bin"
        optimized_model_filepath_2 = "model_opt_with_ext_data_2.onnx"
        external_initializers_file_2 = "model_opt_with_ext_data_2.bin"

        so = onnxrt.SessionOptions()
        so.log_severity_level = 1
        so.logid = "TestModelSerializationWithOriginalExternalInitializersToCurrentDirectory"
        so.optimized_model_filepath = optimized_model_filepath

        so.add_session_config_entry(
            "session.optimized_model_external_initializers_file_name", external_initializers_file
        )

        so.add_session_config_entry("session.optimized_model_external_initializers_min_size_in_bytes", "100")
        session1 = onnxrt.InferenceSession(get_name("model_with_orig_ext_data.onnx"), sess_options=so)
        del session1
        self.assertTrue(os.path.isfile(optimized_model_filepath))

        so2 = onnxrt.SessionOptions()
        so2.log_severity_level = 1
        so2.logid = "TestModelSerializationWithExternalInitializersInCurrentDirectory"
        so2.optimized_model_filepath = optimized_model_filepath_2
        so2.add_session_config_entry(
            "session.optimized_model_external_initializers_file_name", external_initializers_file_2
        )
        so2.add_session_config_entry("session.optimized_model_external_initializers_min_size_in_bytes", "10")

        # verify that we can load the optimized model with external data in current directory and save
        # optimized model with external data to current directory.
        session2 = onnxrt.InferenceSession(optimized_model_filepath, sess_options=so2)
        del session2
        self.assertTrue(os.path.isfile(optimized_model_filepath_2))
        self.assertTrue(os.path.isfile(external_initializers_file_2))

        # Remove model 1 to make sure optimized model 2 can be loaded independently from model 1
        os.remove(optimized_model_filepath)

        session3 = onnxrt.InferenceSession(optimized_model_filepath_2, sess_options=onnxrt.SessionOptions())
        del session3

        os.remove(optimized_model_filepath_2)
        os.remove(external_initializers_file_2)

    def test_get_providers(self):
        self.assertTrue("CPUExecutionProvider" in onnxrt.get_available_providers())
        # get_all_providers() returns the default EP order from highest to lowest.
        # CPUExecutionProvider should always be last.
        self.assertTrue(onnxrt.get_all_providers()[-1] == "CPUExecutionProvider")
        sess = onnxrt.InferenceSession(get_name("mul_1.onnx"), providers=onnxrt.get_available_providers())
        self.assertTrue("CPUExecutionProvider" in sess.get_providers())

    def test_enabling_and_disabling_telemetry(self):
        onnxrt.disable_telemetry_events()

        # no-op on non-Windows builds
        # may be no-op on certain Windows builds based on build configuration
        onnxrt.enable_telemetry_events()

    def test_deserialization_from_path_object(self):
        # path object is allowed
        onnxrt.InferenceSession(pathlib.Path(get_name("mul_1.onnx")), providers=available_providers)

    def test_set_providers(self):
        if "CUDAExecutionProvider" in onnxrt.get_available_providers():
            sess = onnxrt.InferenceSession(get_name("mul_1.onnx"), providers=["CUDAExecutionProvider"])
            # confirm that CUDA Provider is in list of registered providers.
            self.assertTrue("CUDAExecutionProvider" in sess.get_providers())
            # reset the session and register only CPU Provider.
            sess.set_providers(["CPUExecutionProvider"])
            # confirm only CPU Provider is registered now.
            self.assertEqual(["CPUExecutionProvider"], sess.get_providers())

    @unittest.skipUnless(
        "TensorrtExecutionProvider" in onnxrt.get_available_providers(),
        "TensorRT execution provider is not available",
    )
    def test_tensorrt_inference_without_cpu_fallback(self):
        session_options = onnxrt.SessionOptions()
        session_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        sess = onnxrt.InferenceSession(
            get_name("mul_1.onnx"),
            sess_options=session_options,
            providers=["TensorrtExecutionProvider"],
        )
        self.run_model(sess, None)

    def test_set_providers_with_options(self):
        if "TensorrtExecutionProvider" in onnxrt.get_available_providers():
            sess = onnxrt.InferenceSession(get_name("mul_1.onnx"), providers=["TensorrtExecutionProvider"])
            self.assertIn("TensorrtExecutionProvider", sess.get_providers())

            options = sess.get_provider_options()
            option = options["TensorrtExecutionProvider"]
            self.assertIn("device_id", option)
            self.assertIn("trt_max_partition_iterations", option)
            self.assertIn("trt_min_subgraph_size", option)
            self.assertIn("trt_max_workspace_size", option)
            self.assertIn("trt_dump_subgraphs", option)
            self.assertIn("trt_engine_cache_enable", option)
            self.assertIn("trt_engine_cache_path", option)
            self.assertIn("trt_force_sequential_engine_build", option)

            max_partition_iterations = option["trt_max_partition_iterations"]
            new_max_partition_iterations = int(max_partition_iterations) + 1
            min_subgraph_size = option["trt_min_subgraph_size"]
            new_min_subgraph_size = int(min_subgraph_size) + 1
            ori_max_workspace_size = option["trt_max_workspace_size"]
            new_max_workspace_size = int(ori_max_workspace_size) // 2

            option = {}
            option["trt_max_partition_iterations"] = new_max_partition_iterations
            option["trt_min_subgraph_size"] = new_min_subgraph_size
            option["trt_max_workspace_size"] = new_max_workspace_size
            dump_subgraphs = "true"
            option["trt_dump_subgraphs"] = dump_subgraphs
            engine_cache_enable = "true"
            option["trt_engine_cache_enable"] = engine_cache_enable
            engine_cache_path = "./engine_cache"
            option["trt_engine_cache_path"] = engine_cache_path
            force_sequential_engine_build = "true"
            option["trt_force_sequential_engine_build"] = force_sequential_engine_build
            option["user_compute_stream"] = "1"
            sess.set_providers(["TensorrtExecutionProvider"], [option])

            options = sess.get_provider_options()
            option = options["TensorrtExecutionProvider"]
            self.assertEqual(
                option["trt_max_partition_iterations"],
                str(new_max_partition_iterations),
            )
            self.assertEqual(option["trt_min_subgraph_size"], str(new_min_subgraph_size))
            self.assertEqual(option["trt_max_workspace_size"], str(new_max_workspace_size))
            self.assertEqual(option["trt_dump_subgraphs"], "1")
            self.assertEqual(option["trt_engine_cache_enable"], "1")
            self.assertEqual(option["trt_engine_cache_path"], str(engine_cache_path))
            self.assertEqual(option["trt_force_sequential_engine_build"], "1")
            self.assertEqual(option["user_compute_stream"], "1")
            self.assertEqual(option["has_user_compute_stream"], "1")

            session_options = C.get_default_session_options()

            # TRT plugins registered as custom op domain should only be added once in session option regardless of number of session creation
            sess1 = onnxrt.InferenceSession(
                get_name("mul_1.onnx"), session_options, providers=["TensorrtExecutionProvider"]
            )
            sess2 = onnxrt.InferenceSession(
                get_name("mul_1.onnx"), session_options, providers=["TensorrtExecutionProvider"]
            )
            self.assertIn("TensorrtExecutionProvider", sess1.get_providers())
            self.assertIn("TensorrtExecutionProvider", sess2.get_providers())

            # We currently disable following test code since that not all test machines/GPUs have nvidia int8 capability

            """
            int8_use_native_calibration_table = "false"
            option['trt_int8_use_native_calibration_table'] = int8_use_native_calibration_table
            int8_enable = "true"
            option['trt_int8_enable'] = int8_enable
            calib_table_name = '/home/onnxruntime/table.flatbuffers' # this file is not existed
            option['trt_int8_calibration_table_name'] = calib_table_name
            with self.assertRaises(RuntimeError):
                sess.set_providers(['TensorrtExecutionProvider'], [option])
            """

            try:
                import torch  # noqa: PLC0415

                if torch.cuda.is_available():
                    s = torch.cuda.Stream()
                    option["user_compute_stream"] = str(s.cuda_stream)
                    sess.set_providers(["TensorrtExecutionProvider"], [option])
                    options = sess.get_provider_options()
                    self.assertEqual(options["TensorrtExecutionProvider"]["user_compute_stream"], str(s.cuda_stream))
                    self.assertEqual(options["TensorrtExecutionProvider"]["has_user_compute_stream"], "1")
            except ImportError:
                print("torch is not installed, skip testing setting user_compute_stream from torch cuda stream")

        if "CUDAExecutionProvider" in onnxrt.get_available_providers():
            cuda_success = 0

            def run_base_test1():
                sess = onnxrt.InferenceSession(get_name("mul_1.onnx"), providers=["CUDAExecutionProvider"])
                self.assertTrue("CUDAExecutionProvider" in sess.get_providers())

                option1 = {"device_id": 0}
                sess.set_providers(["CUDAExecutionProvider"], [option1])
                self.assertEqual(
                    ["CUDAExecutionProvider", "CPUExecutionProvider"],
                    sess.get_providers(),
                )
                option2 = {"device_id": -1}
                with self.assertRaises(RuntimeError):
                    sess.set_providers(["CUDAExecutionProvider"], [option2])
                sess.set_providers(["CUDAExecutionProvider", "CPUExecutionProvider"], [option1, {}])
                self.assertEqual(
                    ["CUDAExecutionProvider", "CPUExecutionProvider"],
                    sess.get_providers(),
                )

            def run_base_test2():
                sess = onnxrt.InferenceSession(get_name("mul_1.onnx"), providers=["CUDAExecutionProvider"])
                self.assertIn("CUDAExecutionProvider", sess.get_providers())

                # test get/set of "gpu_mem_limit" configuration.
                options = sess.get_provider_options()
                self.assertIn("CUDAExecutionProvider", options)
                option = options["CUDAExecutionProvider"]
                self.assertIn("gpu_mem_limit", option)
                ori_mem_limit = option["gpu_mem_limit"]
                new_mem_limit = int(ori_mem_limit) // 2
                option["gpu_mem_limit"] = new_mem_limit
                sess.set_providers(["CUDAExecutionProvider"], [option])
                options = sess.get_provider_options()
                self.assertEqual(
                    options["CUDAExecutionProvider"]["gpu_mem_limit"],
                    str(new_mem_limit),
                )

                option["gpu_mem_limit"] = ori_mem_limit
                sess.set_providers(["CUDAExecutionProvider"], [option])
                options = sess.get_provider_options()
                self.assertEqual(options["CUDAExecutionProvider"]["gpu_mem_limit"], ori_mem_limit)

                def test_get_and_set_option_with_values(option_name, option_values):
                    provider_options = sess.get_provider_options()
                    self.assertIn("CUDAExecutionProvider", provider_options)
                    cuda_options = options["CUDAExecutionProvider"]
                    self.assertIn(option_name, cuda_options)
                    for option_value in option_values:
                        cuda_options[option_name] = option_value
                        sess.set_providers(["CUDAExecutionProvider"], [cuda_options])
                        new_provider_options = sess.get_provider_options()
                        self.assertEqual(
                            new_provider_options.get("CUDAExecutionProvider", {}).get(option_name),
                            str(option_value),
                        )

                test_get_and_set_option_with_values("enable_cuda_graph", ["1", "0"])

                test_get_and_set_option_with_values("arena_extend_strategy", ["kNextPowerOfTwo", "kSameAsRequested"])

                test_get_and_set_option_with_values("cudnn_conv_algo_search", ["DEFAULT", "EXHAUSTIVE", "HEURISTIC"])

                test_get_and_set_option_with_values("enable_cudnn", ["1", "0"])

                test_get_and_set_option_with_values("do_copy_in_default_stream", [0, 1])

                test_get_and_set_option_with_values("tunable_op_enable", ["1", "0"])

                test_get_and_set_option_with_values("tunable_op_tuning_enable", ["1", "0"])

                test_get_and_set_option_with_values("tunable_op_max_tuning_duration_ms", ["-1", "1"])

                test_get_and_set_option_with_values("use_tf32", ["1", "0"])

                test_get_and_set_option_with_values("sdpa_kernel", ["0", "1", "2"])

                option["gpu_external_alloc"] = "0"
                option["gpu_external_free"] = "0"
                option["gpu_external_empty_cache"] = "0"
                sess.set_providers(["CUDAExecutionProvider"], [option])
                options = sess.get_provider_options()
                self.assertEqual(options["CUDAExecutionProvider"]["gpu_external_alloc"], "0")
                self.assertEqual(options["CUDAExecutionProvider"]["gpu_external_free"], "0")
                self.assertEqual(options["CUDAExecutionProvider"]["gpu_external_empty_cache"], "0")

                option["user_compute_stream"] = "0"
                sess.set_providers(["CUDAExecutionProvider"], [option])
                options = sess.get_provider_options()
                self.assertEqual(options["CUDAExecutionProvider"]["user_compute_stream"], "0")

                try:
                    import torch  # noqa: PLC0415

                    if torch.cuda.is_available():
                        s = torch.cuda.Stream()
                        option["user_compute_stream"] = str(s.cuda_stream)
                        sess.set_providers(["CUDAExecutionProvider"], [option])
                        options = sess.get_provider_options()
                        self.assertEqual(options["CUDAExecutionProvider"]["user_compute_stream"], str(s.cuda_stream))
                        self.assertEqual(options["CUDAExecutionProvider"]["has_user_compute_stream"], "1")
                except ImportError:
                    print("torch is not installed, skip testing setting user_compute_stream from torch cuda stream")

                #
                # Note: Tests that throw an exception leave an empty session due to how set_providers currently works,
                #       so run them last. Each set_providers call will attempt to re-create a session, so it's
                #       fine for a test that fails to run immediately after another one that fails.
                #       Alternatively a valid call to set_providers could be used to recreate the underlying session
                #       after a failed call.
                #
                option["arena_extend_strategy"] = "wrong_value"
                with self.assertRaises(RuntimeError):
                    sess.set_providers(["CUDAExecutionProvider"], [option])

                option["gpu_mem_limit"] = -1024
                with self.assertRaises(RuntimeError):
                    sess.set_providers(["CUDAExecutionProvider"], [option])

                option["gpu_mem_limit"] = 1024.1024
                with self.assertRaises(RuntimeError):
                    sess.set_providers(["CUDAExecutionProvider"], [option])

                option["gpu_mem_limit"] = "wrong_value"
                with self.assertRaises(RuntimeError):
                    sess.set_providers(["CUDAExecutionProvider"], [option])

            def set_device_id_test(i, cuda_lib):
                device = ctypes.c_int()
                result = ctypes.c_int()
                error_str = ctypes.c_char_p()

                sess = onnxrt.InferenceSession(get_name("mul_1.onnx"), providers=["CPUExecutionProvider"])
                option = {"device_id": i}
                sess.set_providers(["CUDAExecutionProvider"], [option])
                self.assertEqual(
                    ["CUDAExecutionProvider", "CPUExecutionProvider"],
                    sess.get_providers(),
                )
                result = cuda_lib.cuCtxGetDevice(ctypes.byref(device))
                if result != cuda_success:
                    cuda_lib.cuGetErrorString(result, ctypes.byref(error_str))
                    print(f"cuCtxGetDevice failed with error code {result}: {error_str.value.decode()}")

                self.assertEqual(result, cuda_success)
                self.assertEqual(i, device.value)

            def run_advanced_test(cuda_lib):
                num_device = self.cuda_device_count(cuda_lib)
                if num_device < 0:
                    return

                # Configure session to be ready to run on all available cuda devices
                for i in range(num_device):
                    set_device_id_test(i, cuda_lib)

                sess = onnxrt.InferenceSession(get_name("mul_1.onnx"), providers=["CPUExecutionProvider"])

                # configure session with invalid option values and that should fail
                with self.assertRaises(RuntimeError):
                    option = {"device_id": num_device}
                    sess.set_providers(["CUDAExecutionProvider"], [option])
                    option = {"device_id": "invalid_value"}
                    sess.set_providers(["CUDAExecutionProvider"], [option])

                # configure session with invalid option should fail
                with self.assertRaises(RuntimeError):
                    option = {"invalid_option": 123}
                    sess.set_providers(["CUDAExecutionProvider"], [option])

            run_base_test1()
            run_base_test2()
            cuda = self.load_cuda_lib()
            if cuda is not None:
                print("run advanced_test")
                run_advanced_test(cuda)

    def test_invalid_set_providers(self):
        with self.assertRaises(RuntimeError) as context:
            sess = onnxrt.InferenceSession(get_name("mul_1.onnx"), providers=["CPUExecutionProvider"])
            sess.set_providers(["InvalidProvider"])
        self.assertTrue("Unknown Provider Type: InvalidProvider" in str(context.exception))

    def test_session_providers(self):
        if "CUDAExecutionProvider" in onnxrt.get_available_providers():
            # create session from scratch, but constrain it to only use the CPU.
            sess = onnxrt.InferenceSession(get_name("mul_1.onnx"), providers=["CPUExecutionProvider"])
            self.assertEqual(["CPUExecutionProvider"], sess.get_providers())

    def test_get_and_set_tuning_results(self):
        def get_tuning_results_for_ep(sess, ep):  # without the outer list
            tuning_results = sess.get_tuning_results()
            self.assertGreaterEqual(len(tuning_results), 1)
            tuning_results_for_this_ep = [t for t in tuning_results if t.get("ep") == ep]
            self.assertEqual(len(tuning_results_for_this_ep), 1)
            return tuning_results_for_this_ep[0]

        probe_op_sig = "probe_but_not_an_op_signature"
        probe_params_sig = "probe_but_not_an_params_signature"
        probe_value = 10000000

        def copy_tuning_results_with_probe(tr):
            tr = copy.deepcopy(tr)
            tr["results"][probe_op_sig] = {probe_params_sig: probe_value}
            return tr

        def assert_tuning_results_loaded(sess, ep):
            tr = get_tuning_results_for_ep(sess, ep)
            self.assertIn(probe_op_sig, tr["results"])
            self.assertEqual(tr["results"][probe_op_sig], {probe_params_sig: probe_value})

        def assert_tuning_results_not_loaded(sess, ep):
            tr = get_tuning_results_for_ep(sess, ep)
            self.assertNotIn(probe_op_sig, tr["results"])

        def do_test_get_and_set_tuning_results(ep):
            sess = onnxrt.InferenceSession(get_name("mul_1.onnx"), providers=[ep])
            tuning_results = get_tuning_results_for_ep(sess, ep)

            self.assertIn("ep", tuning_results)
            self.assertIn("results", tuning_results)
            self.assertIn("validators", tuning_results)
            self.assertIn("ORT_VERSION", tuning_results["validators"])
            self.assertNotIn("NOT_A_VALIDATOR_KEY", tuning_results["validators"])

            # invalid EP will be rejected
            invalid_unknown_ep = copy_tuning_results_with_probe(tuning_results)
            invalid_unknown_ep["ep"] = "UnknownEP"
            sess.set_tuning_results([invalid_unknown_ep])
            with self.assertRaises(RuntimeError) as context:
                sess.set_tuning_results([invalid_unknown_ep], error_on_invalid=True)
            self.assertIn("Cannot find execution provider UnknownEP", str(context.exception))
            assert_tuning_results_not_loaded(sess, ep)

            # missing validator key will be rejected
            mismatched_validator_key_missing = copy_tuning_results_with_probe(tuning_results)
            mismatched_validator_key_missing["validators"].pop("ORT_VERSION")
            sess.set_tuning_results([mismatched_validator_key_missing])
            with self.assertRaises(RuntimeError) as context:
                sess.set_tuning_results([mismatched_validator_key_missing], error_on_invalid=True)
            self.assertIn("ORT_VERSION", str(context.exception))
            self.assertIn("is not provided for validation", str(context.exception))
            assert_tuning_results_not_loaded(sess, ep)

            mismatched_validator_key_extra = copy_tuning_results_with_probe(tuning_results)
            mismatched_validator_key_extra["validators"]["NOT_A_VALIDATOR_KEY"] = "NOT_USED"
            sess.set_tuning_results([mismatched_validator_key_extra])
            with self.assertRaises(RuntimeError) as context:
                sess.set_tuning_results([mismatched_validator_key_extra], error_on_invalid=True)
            self.assertIn("NOT_A_VALIDATOR_KEY", str(context.exception))
            self.assertIn("is unable to consume it", str(context.exception))
            assert_tuning_results_not_loaded(sess, ep)

            validation_failure = copy_tuning_results_with_probe(tuning_results)
            validation_failure["validators"]["ORT_VERSION"] = "This is not a proper ORT_VERSION value!"
            sess.set_tuning_results([validation_failure])
            with self.assertRaises(RuntimeError) as context:
                sess.set_tuning_results([validation_failure], error_on_invalid=True)
            self.assertIn("Failed to load TuningResults", str(context.exception))
            self.assertIn("version mismatch", str(context.exception))
            assert_tuning_results_not_loaded(sess, ep)

            loadable = copy_tuning_results_with_probe(tuning_results)
            sess.set_tuning_results([loadable], error_on_invalid=True)
            assert_tuning_results_loaded(sess, ep)

        if "CUDAExecutionProvider" in onnxrt.get_available_providers():
            do_test_get_and_set_tuning_results("CUDAExecutionProvider")

    def test_run_model_with_optional_sequence_input(self):
        sess = onnxrt.InferenceSession(get_name("identity_opt.onnx"))
        x = [np.array([1, 2, 3, 4, 5]).astype(np.float32)]
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name
        res = sess.run([output_name], {input_name: x})
        np.testing.assert_allclose(res[0], x)

    def test_run_model(self):
        sess = onnxrt.InferenceSession(get_name("mul_1.onnx"), providers=available_providers)
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)

        inputs = sess.get_inputs()
        self.assertEqual(len(inputs), 1)
        self.assertEqual(inputs[0].name, "X")
        self.assertEqual(inputs[0].shape, [3, 2])

        input_meminfos = sess.get_input_memory_infos()
        self.assertEqual(len(input_meminfos), 1)
        self.assertIsNotNone(input_meminfos[0])

        input_epdevices = sess.get_input_epdevices()
        # The entry my be None (null) but it should be present
        self.assertEqual(len(input_epdevices), 1)

        outputs = sess.get_outputs()
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0].name, "Y")
        self.assertEqual(outputs[0].shape, [3, 2])

        output_meminfos = sess.get_output_memory_infos()
        self.assertEqual(len(output_meminfos), 1)
        self.assertIsNotNone(output_meminfos[0])

        res = sess.run([outputs[0].name], {inputs[0].name: x})
        output_expected = np.array([[1.0, 4.0], [9.0, 16.0], [25.0, 36.0]], dtype=np.float32)
        np.testing.assert_allclose(res[0], output_expected, rtol=1e-05, atol=1e-08)

    def test_run_async(self):
        event = threading.Event()
        allow_callback = threading.Event()
        output_expected = np.array([[1.0, 4.0], [9.0, 16.0], [25.0, 36.0]], dtype=np.float32)
        input_ref = None

        class MyData:
            def __init__(self, id):
                self.__id = id

            def get_id(self):
                return self.__id

        my_data = MyData(123456)

        def callback(res: np.ndarray, data: MyData, err: str) -> None:
            self.assertTrue(allow_callback.wait(10))
            self.assertIsNotNone(input_ref())
            self.assertEqual(len(err), 0)
            self.assertEqual(len(res), 1)
            self.assertEqual(data.get_id(), 123456)
            np.testing.assert_allclose(res[0], output_expected, rtol=1e-05, atol=1e-08)
            event.set()

        so = onnxrt.SessionOptions()
        so.intra_op_num_threads = 2

        sess = onnxrt.InferenceSession(get_name("mul_1.onnx"), so, providers=available_providers)

        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        input_ref = weakref.ref(x)
        run_options = onnxrt.RunOptions()
        sess.run_async(["Y"], {"X": x}, callback, my_data, run_options)
        del x
        del run_options
        del sess
        gc.collect()
        allow_callback.set()

        event.wait(10)  # timeout in 10 sec
        self.assertTrue(event.is_set())
        deadline = time.monotonic() + 10
        while input_ref() is not None and time.monotonic() < deadline:
            gc.collect()
            time.sleep(0.01)
        self.assertIsNone(input_ref())

    def test_run_model_from_bytes(self):
        with open(get_name("mul_1.onnx"), "rb") as f:
            content = f.read()
        sess = onnxrt.InferenceSession(content, providers=available_providers)
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        input_name = sess.get_inputs()[0].name
        self.assertEqual(input_name, "X")
        input_shape = sess.get_inputs()[0].shape
        self.assertEqual(input_shape, [3, 2])
        output_name = sess.get_outputs()[0].name
        self.assertEqual(output_name, "Y")
        output_shape = sess.get_outputs()[0].shape
        self.assertEqual(output_shape, [3, 2])
        res = sess.run([output_name], {input_name: x})
        output_expected = np.array([[1.0, 4.0], [9.0, 16.0], [25.0, 36.0]], dtype=np.float32)
        np.testing.assert_allclose(res[0], output_expected, rtol=1e-05, atol=1e-08)

    def test_run_model2(self):
        sess = onnxrt.InferenceSession(get_name("matmul_1.onnx"), providers=available_providers)
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        input_name = sess.get_inputs()[0].name
        self.assertEqual(input_name, "X")
        input_shape = sess.get_inputs()[0].shape
        self.assertEqual(input_shape, [3, 2])
        output_name = sess.get_outputs()[0].name
        self.assertEqual(output_name, "Y")
        output_shape = sess.get_outputs()[0].shape
        self.assertEqual(output_shape, [3, 1])
        res = sess.run([output_name], {input_name: x})
        output_expected = np.array([[5.0], [11.0], [17.0]], dtype=np.float32)
        np.testing.assert_allclose(res[0], output_expected, rtol=1e-05, atol=1e-08)

    def test_run_model2_contiguous(self):
        sess = onnxrt.InferenceSession(get_name("matmul_1.onnx"), providers=available_providers)
        x = np.array([[2.0, 1.0], [4.0, 3.0], [6.0, 5.0]], dtype=np.float32)[:, [1, 0]]
        input_name = sess.get_inputs()[0].name
        self.assertEqual(input_name, "X")
        input_shape = sess.get_inputs()[0].shape
        self.assertEqual(input_shape, [3, 2])
        output_name = sess.get_outputs()[0].name
        self.assertEqual(output_name, "Y")
        output_shape = sess.get_outputs()[0].shape
        self.assertEqual(output_shape, [3, 1])
        res = sess.run([output_name], {input_name: x})
        output_expected = np.array([[5.0], [11.0], [17.0]], dtype=np.float32)
        np.testing.assert_allclose(res[0], output_expected, rtol=1e-05, atol=1e-08)
        xcontiguous = np.ascontiguousarray(x)
        rescontiguous = sess.run([output_name], {input_name: xcontiguous})
        np.testing.assert_allclose(rescontiguous[0], output_expected, rtol=1e-05, atol=1e-08)

    def test_run_model_multiple_threads(self):
        # Skip this test for a "pure" DML onnxruntime python wheel.
        # We keep this test enabled for instances where both DML and CUDA EPs are available
        # (Windows GPU CI pipeline has this config) - this test will pass because CUDA has higher precedence
        # than DML and the nodes are assigned to only the CUDA EP (which supports this test).
        if "DmlExecutionProvider" in available_providers and "CUDAExecutionProvider" not in available_providers:
            print(
                "Skipping testRunModelMultipleThreads as the DML EP does not support calling Run()"
                " on different threads using the same session object."
            )
        else:
            so = onnxrt.SessionOptions()
            so.log_verbosity_level = 1
            so.logid = "MultiThreadsTest"
            sess = onnxrt.InferenceSession(
                get_name("mul_1.onnx"),
                sess_options=so,
                providers=available_providers_without_tvm,
            )
            ro1 = onnxrt.RunOptions()
            ro1.logid = "thread1"
            t1 = threading.Thread(target=self.run_model, args=(sess, ro1))
            ro2 = onnxrt.RunOptions()
            ro2.logid = "thread2"
            t2 = threading.Thread(target=self.run_model, args=(sess, ro2))
            t1.start()
            t2.start()
            t1.join()
            t2.join()

        if "CUDAExecutionProvider" in available_providers:
            cuda_options = {
                "gpu_mem_limit": 2 * 1024 * 1024 * 1024,
                "arena_extend_strategy": "kSameAsRequested",
            }
            model_path = "../models/zoo/opset7/ResNet18-v2/resnet18-v2-7.onnx"
            if not os.path.exists(model_path):
                print("cannot find resnet18-v2-7.onnx")
                return
            session = onnxrt.InferenceSession(model_path, providers=[("CUDAExecutionProvider", cuda_options)])
            [thread_num, iter_num] = [4, 20]
            q = queue.Queue()
            input_name = session.get_inputs()[0].name
            input_value = np.random.rand(1, 3, 224, 224).astype(np.float32)
            workers = [
                threading.Thread(target=self.run_model_with_input, args=(session, input_name, input_value, iter_num, q))
                for idx in range(thread_num)
            ]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join()

            result = q.get()
            while q.qsize() > 0:
                self.assertEqual(result, q.get())

    def test_list_as_input(self):
        sess = onnxrt.InferenceSession(get_name("mul_1.onnx"), providers=available_providers)
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        input_name = sess.get_inputs()[0].name
        res = sess.run([], {input_name: x.tolist()})
        output_expected = np.array([[1.0, 4.0], [9.0, 16.0], [25.0, 36.0]], dtype=np.float32)
        np.testing.assert_allclose(res[0], output_expected, rtol=1e-05, atol=1e-08)

    def test_string_list_as_input(self):
        sess = onnxrt.InferenceSession(get_name("identity_string.onnx"), providers=available_providers_without_tvm)
        x = np.array(["this", "is", "identity", "test"], dtype=str).reshape((2, 2))
        x_name = sess.get_inputs()[0].name
        res = sess.run([], {x_name: x.tolist()})
        np.testing.assert_equal(res[0], x)

    def test_run_device(self):
        device = onnxrt.get_device()
        self.assertTrue("CPU" in device or "GPU" in device)

    def test_run_model_symbolic_input(self):
        sess = onnxrt.InferenceSession(get_name("matmul_2.onnx"), providers=available_providers_without_tvm)
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        input_name = sess.get_inputs()[0].name
        self.assertEqual(input_name, "X")
        input_shape = sess.get_inputs()[0].shape
        # Input X has an unknown dimension.
        self.assertEqual(input_shape, ["None", 2])
        output_name = sess.get_outputs()[0].name
        self.assertEqual(output_name, "Y")
        output_shape = sess.get_outputs()[0].shape
        # Output X has an unknown dimension.
        self.assertEqual(output_shape, ["None", 1])
        res = sess.run([output_name], {input_name: x})
        output_expected = np.array([[5.0], [11.0], [17.0]], dtype=np.float32)
        np.testing.assert_allclose(res[0], output_expected, rtol=1e-05, atol=1e-08)

    def test_boolean_inputs(self):
        sess = onnxrt.InferenceSession(get_name("logicaland.onnx"), providers=available_providers)
        a = np.array([[True, True], [False, False]], dtype=bool)
        b = np.array([[True, False], [True, False]], dtype=bool)

        # input1:0 is first in the protobuf, and input:0 is second
        # and we maintain the original order.
        a_name = sess.get_inputs()[0].name
        self.assertEqual(a_name, "input1:0")
        a_shape = sess.get_inputs()[0].shape
        self.assertEqual(a_shape, [2, 2])
        a_type = sess.get_inputs()[0].type
        self.assertEqual(a_type, "tensor(bool)")

        b_name = sess.get_inputs()[1].name
        self.assertEqual(b_name, "input:0")
        b_shape = sess.get_inputs()[1].shape
        self.assertEqual(b_shape, [2, 2])
        b_type = sess.get_inputs()[0].type
        self.assertEqual(b_type, "tensor(bool)")

        output_name = sess.get_outputs()[0].name
        self.assertEqual(output_name, "output:0")
        output_shape = sess.get_outputs()[0].shape
        self.assertEqual(output_shape, [2, 2])
        output_type = sess.get_outputs()[0].type
        self.assertEqual(output_type, "tensor(bool)")

        output_expected = np.array([[True, False], [False, False]], dtype=bool)
        res = sess.run([output_name], {a_name: a, b_name: b})
        np.testing.assert_equal(res[0], output_expected)

    def test_string_input1(self):
        sess = onnxrt.InferenceSession(get_name("identity_string.onnx"), providers=available_providers_without_tvm)
        x = np.array(["this", "is", "identity", "test"], dtype=str).reshape((2, 2))

        x_name = sess.get_inputs()[0].name
        self.assertEqual(x_name, "input:0")
        x_shape = sess.get_inputs()[0].shape
        self.assertEqual(x_shape, [2, 2])
        x_type = sess.get_inputs()[0].type
        self.assertEqual(x_type, "tensor(string)")

        output_name = sess.get_outputs()[0].name
        self.assertEqual(output_name, "output:0")
        output_shape = sess.get_outputs()[0].shape
        self.assertEqual(output_shape, [2, 2])
        output_type = sess.get_outputs()[0].type
        self.assertEqual(output_type, "tensor(string)")

        res = sess.run([output_name], {x_name: x})
        np.testing.assert_equal(res[0], x)

    def test_string_input2(self):
        sess = onnxrt.InferenceSession(get_name("identity_string.onnx"), providers=available_providers_without_tvm)
        x = np.array(["Olá", "你好", "여보세요", "hello"], dtype=str).reshape((2, 2))

        x_name = sess.get_inputs()[0].name
        self.assertEqual(x_name, "input:0")
        x_shape = sess.get_inputs()[0].shape
        self.assertEqual(x_shape, [2, 2])
        x_type = sess.get_inputs()[0].type
        self.assertEqual(x_type, "tensor(string)")

        output_name = sess.get_outputs()[0].name
        self.assertEqual(output_name, "output:0")
        output_shape = sess.get_outputs()[0].shape
        self.assertEqual(output_shape, [2, 2])
        output_type = sess.get_outputs()[0].type
        self.assertEqual(output_type, "tensor(string)")

        res = sess.run([output_name], {x_name: x})
        np.testing.assert_equal(res[0], x)

    def test_input_bytes(self):
        sess = onnxrt.InferenceSession(get_name("identity_string.onnx"), providers=available_providers_without_tvm)
        x = np.array([b"this", b"is", b"identity", b"test"]).reshape((2, 2))

        x_name = sess.get_inputs()[0].name
        self.assertEqual(x_name, "input:0")
        x_shape = sess.get_inputs()[0].shape
        self.assertEqual(x_shape, [2, 2])
        x_type = sess.get_inputs()[0].type
        self.assertEqual(x_type, "tensor(string)")

        output_name = sess.get_outputs()[0].name
        self.assertEqual(output_name, "output:0")
        output_shape = sess.get_outputs()[0].shape
        self.assertEqual(output_shape, [2, 2])
        output_type = sess.get_outputs()[0].type
        self.assertEqual(output_type, "tensor(string)")

        res = sess.run([output_name], {x_name: x})
        np.testing.assert_equal(res[0].astype("|S8"), x)

    def test_input_object(self):
        sess = onnxrt.InferenceSession(get_name("identity_string.onnx"), providers=available_providers_without_tvm)
        x = np.array(["this", "is", "identity", "test"], object).reshape((2, 2))

        x_name = sess.get_inputs()[0].name
        self.assertEqual(x_name, "input:0")
        x_shape = sess.get_inputs()[0].shape
        self.assertEqual(x_shape, [2, 2])
        x_type = sess.get_inputs()[0].type
        self.assertEqual(x_type, "tensor(string)")

        output_name = sess.get_outputs()[0].name
        self.assertEqual(output_name, "output:0")
        output_shape = sess.get_outputs()[0].shape
        self.assertEqual(output_shape, [2, 2])
        output_type = sess.get_outputs()[0].type
        self.assertEqual(output_type, "tensor(string)")

        res = sess.run([output_name], {x_name: x})
        np.testing.assert_equal(res[0], x)

    def test_input_void(self):
        sess = onnxrt.InferenceSession(get_name("identity_string.onnx"), providers=available_providers_without_tvm)
        # numpy 1.20+ doesn't automatically pad the bytes based entries in the array when dtype is np.void,
        # so we use inputs where that is the case
        x = np.array([b"must", b"have", b"same", b"size"], dtype=np.void).reshape((2, 2))

        x_name = sess.get_inputs()[0].name
        self.assertEqual(x_name, "input:0")
        x_shape = sess.get_inputs()[0].shape
        self.assertEqual(x_shape, [2, 2])
        x_type = sess.get_inputs()[0].type
        self.assertEqual(x_type, "tensor(string)")

        output_name = sess.get_outputs()[0].name
        self.assertEqual(output_name, "output:0")
        output_shape = sess.get_outputs()[0].shape
        self.assertEqual(output_shape, [2, 2])
        output_type = sess.get_outputs()[0].type
        self.assertEqual(output_type, "tensor(string)")

        res = sess.run([output_name], {x_name: x})

        expr = np.array([["must", "have"], ["same", "size"]], dtype=object)
        np.testing.assert_equal(res[0], expr)

    def test_raise_wrong_num_inputs(self):
        with self.assertRaises(ValueError) as context:
            sess = onnxrt.InferenceSession(get_name("logicaland.onnx"), providers=onnxrt.get_available_providers())
            a = np.array([[True, True], [False, False]], dtype=bool)
            sess.run([], {"input:0": a})
        self.assertIn(
            "Required inputs (['input1:0']) are missing from input feed (['input:0'])", str(context.exception)
        )

    def test_model_meta(self):
        model_path = "../models/opset8/test_squeezenet/model.onnx"
        if not os.path.exists(model_path):
            return
        sess = onnxrt.InferenceSession(model_path, providers=onnxrt.get_available_providers())
        modelmeta = sess.get_modelmeta()
        self.assertEqual("onnx-caffe2", modelmeta.producer_name)
        self.assertEqual("squeezenet_old", modelmeta.graph_name)
        self.assertEqual("", modelmeta.domain)
        self.assertEqual("", modelmeta.description)
        self.assertEqual("", modelmeta.graph_description)

    def test_profiler_with_session_options(self):
        so = onnxrt.SessionOptions()
        so.enable_profiling = True
        sess = onnxrt.InferenceSession(
            get_name("mul_1.onnx"),
            sess_options=so,
            providers=available_providers,
        )
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        sess.run([], {"X": x})
        profile_file = sess.end_profiling()

        tags = ["pid", "dur", "ts", "ph", "X", "name", "args"]
        with open(profile_file) as f:
            lines = f.readlines()
            self.assertTrue("[" in lines[0])
            for i in range(1, len(lines) - 1):
                for tag in tags:
                    self.assertTrue(tag in lines[i])
            self.assertTrue("]" in lines[-1])

        os.remove(profile_file)

    def test_profiler_get_start_time_ns(self):
        def get_single_session_profiling_start_time():
            so = onnxrt.SessionOptions()
            so.enable_profiling = True
            sess = onnxrt.InferenceSession(
                get_name("mul_1.onnx"),
                sess_options=so,
                providers=onnxrt.get_available_providers(),
            )
            start_time = sess.get_profiling_start_time_ns()
            os.remove(sess.end_profiling())
            return start_time

        # Get 1st profiling's start time
        start_time_1 = get_single_session_profiling_start_time()
        # Get 2nd profiling's start time
        start_time_2 = get_single_session_profiling_start_time()
        # Get 3rd profiling's start time
        start_time_3 = get_single_session_profiling_start_time()

        # Chronological profiling's start time
        self.assertTrue(start_time_1 <= start_time_2 <= start_time_3)

    def test_graph_optimization_level(self):
        opt = onnxrt.SessionOptions()
        # default should be all optimizations optimization
        self.assertEqual(opt.graph_optimization_level, onnxrt.GraphOptimizationLevel.ORT_ENABLE_ALL)
        opt.graph_optimization_level = onnxrt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        self.assertEqual(
            opt.graph_optimization_level,
            onnxrt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        )
        sess = onnxrt.InferenceSession(get_name("logicaland.onnx"), sess_options=opt, providers=available_providers)
        a = np.array([[True, True], [False, False]], dtype=bool)
        b = np.array([[True, False], [True, False]], dtype=bool)

        sess.run([], {"input1:0": a, "input:0": b})

    def test_sequence_length(self):
        sess = onnxrt.InferenceSession(get_name("sequence_length.onnx"), providers=available_providers_without_tvm)
        x = [
            np.array([1.0, 0.0, 3.0, 44.0, 23.0, 11.0], dtype=np.float32).reshape((2, 3)),
            np.array([1.0, 0.0, 3.0, 44.0, 23.0, 11.0], dtype=np.float32).reshape((2, 3)),
        ]

        x_name = sess.get_inputs()[0].name
        self.assertEqual(x_name, "X")
        x_type = sess.get_inputs()[0].type
        self.assertEqual(x_type, "seq(tensor(float))")

        output_name = sess.get_outputs()[0].name
        self.assertEqual(output_name, "Y")
        output_type = sess.get_outputs()[0].type
        self.assertEqual(output_type, "tensor(int64)")

        output_expected = np.array(2, dtype=np.int64)
        res = sess.run([output_name], {x_name: x})
        self.assertEqual(output_expected, res[0])

    def test_sequence_construct(self):
        sess = onnxrt.InferenceSession(
            get_name("sequence_construct.onnx"),
            providers=available_providers_without_tvm,
        )

        self.assertEqual(sess.get_inputs()[0].type, "tensor(int64)")
        self.assertEqual(sess.get_inputs()[1].type, "tensor(int64)")

        self.assertEqual(sess.get_inputs()[0].name, "tensor1")
        self.assertEqual(sess.get_inputs()[1].name, "tensor2")

        output_name = sess.get_outputs()[0].name
        self.assertEqual(output_name, "output_sequence")
        output_type = sess.get_outputs()[0].type
        self.assertEqual(output_type, "seq(tensor(int64))")

        output_expected = [
            np.array([1, 0, 3, 44, 23, 11], dtype=np.int64).reshape((2, 3)),
            np.array([1, 2, 3, 4, 5, 6], dtype=np.int64).reshape((2, 3)),
        ]

        res = sess.run(
            [output_name],
            {
                "tensor1": np.array([1, 0, 3, 44, 23, 11], dtype=np.int64).reshape((2, 3)),
                "tensor2": np.array([1, 2, 3, 4, 5, 6], dtype=np.int64).reshape((2, 3)),
            },
        )

        np.testing.assert_array_equal(res[0], output_expected)

    def test_sequence_insert(self):
        opt = onnxrt.SessionOptions()
        opt.execution_mode = onnxrt.ExecutionMode.ORT_SEQUENTIAL
        sess = onnxrt.InferenceSession(
            get_name("sequence_insert.onnx"),
            sess_options=opt,
            providers=available_providers_without_tvm,
        )

        self.assertEqual(sess.get_inputs()[0].type, "seq(tensor(int64))")
        self.assertEqual(sess.get_inputs()[1].type, "tensor(int64)")

        self.assertEqual(sess.get_inputs()[0].name, "input_seq")
        self.assertEqual(sess.get_inputs()[1].name, "tensor")

        output_name = sess.get_outputs()[0].name
        self.assertEqual(output_name, "output_sequence")
        output_type = sess.get_outputs()[0].type
        self.assertEqual(output_type, "seq(tensor(int64))")

        output_expected = [np.array([1, 0, 3, 44, 23, 11], dtype=np.int64).reshape((2, 3))]
        res = sess.run(
            [output_name],
            {
                "tensor": np.array([1, 0, 3, 44, 23, 11], dtype=np.int64).reshape((2, 3)),
                "input_seq": [],
            },
        )
        np.testing.assert_array_equal(res[0], output_expected)

    def test_ort_execution_mode(self):
        opt = onnxrt.SessionOptions()
        self.assertEqual(opt.execution_mode, onnxrt.ExecutionMode.ORT_SEQUENTIAL)
        opt.execution_mode = onnxrt.ExecutionMode.ORT_PARALLEL
        self.assertEqual(opt.execution_mode, onnxrt.ExecutionMode.ORT_PARALLEL)

    def test_loading_session_options_from_model(self):
        try:
            os.environ["ORT_LOAD_CONFIG_FROM_MODEL"] = str(1)
            sess = onnxrt.InferenceSession(
                get_name("model_with_valid_ort_config_json.onnx"),
                providers=onnxrt.get_available_providers(),
            )
            session_options = sess.get_session_options()

            self.assertEqual(session_options.inter_op_num_threads, 5)  # from the ORT config

            self.assertEqual(session_options.intra_op_num_threads, 2)  # from the ORT config

            self.assertEqual(
                session_options.execution_mode, onnxrt.ExecutionMode.ORT_SEQUENTIAL
            )  # default option (not from the ORT config)

            self.assertEqual(
                session_options.graph_optimization_level,
                onnxrt.GraphOptimizationLevel.ORT_ENABLE_ALL,
            )  # from the ORT config

            self.assertEqual(session_options.enable_profiling, True)  # from the ORT config

            os.remove(sess.end_profiling())

        except Exception:
            raise

        finally:
            # Make sure the usage of the feature is disabled after this test
            os.environ["ORT_LOAD_CONFIG_FROM_MODEL"] = str(0)

    def test_session_options_add_free_dimension_override_by_denotation(self):
        so = onnxrt.SessionOptions()
        so.add_free_dimension_override_by_denotation("DATA_BATCH", 3)
        so.add_free_dimension_override_by_denotation("DATA_CHANNEL", 5)
        sess = onnxrt.InferenceSession(
            get_name("abs_free_dimensions.onnx"),
            sess_options=so,
            providers=onnxrt.get_available_providers(),
        )
        input_name = sess.get_inputs()[0].name
        self.assertEqual(input_name, "x")
        input_shape = sess.get_inputs()[0].shape
        # Free dims with denotations - "DATA_BATCH" and "DATA_CHANNEL" have values assigned to them.
        self.assertEqual(input_shape, [3, 5, 5])

    def test_session_options_add_free_dimension_override_by_name(self):
        so = onnxrt.SessionOptions()
        so.add_free_dimension_override_by_name("Dim1", 4)
        so.add_free_dimension_override_by_name("Dim2", 6)
        sess = onnxrt.InferenceSession(
            get_name("abs_free_dimensions.onnx"),
            sess_options=so,
            providers=onnxrt.get_available_providers(),
        )
        input_name = sess.get_inputs()[0].name
        self.assertEqual(input_name, "x")
        input_shape = sess.get_inputs()[0].shape
        # "Dim1" and "Dim2" have values assigned to them.
        self.assertEqual(input_shape, [4, 6, 5])

    def test_session_options_add_config_entry(self):
        so = onnxrt.SessionOptions()
        key = "CONFIG_KEY"
        val = "CONFIG_VAL"
        so.add_session_config_entry(key, val)
        self.assertEqual(so.get_session_config_entry(key), val)

    def test_invalid_session_options_config_entry(self):
        so = onnxrt.SessionOptions()
        invalide_key = "INVALID_KEY"
        with self.assertRaises(RuntimeError) as context:
            so.get_session_config_entry(invalide_key)
        self.assertTrue(
            "SessionOptions does not have configuration with key: " + invalide_key in str(context.exception)
        )

    def test_session_options_add_initializer(self):
        # Create an initializer and add it to a SessionOptions instance
        so = onnxrt.SessionOptions()
        # This initializer is different from the actual initializer in the model for "W"
        ortvalue_initializer = onnxrt.OrtValue.ortvalue_from_numpy(
            np.array([[2.0, 1.0], [4.0, 3.0], [6.0, 5.0]], dtype=np.float32)
        )
        # The user should manage the life cycle of this OrtValue and should keep it in scope
        # as long as any session that is going to be reliant on it is in scope
        so.add_initializer("W", ortvalue_initializer)

        # Create an InferenceSession that only uses the CPU EP and validate that it uses the
        # initializer provided via the SessionOptions instance (overriding the model initializer)
        # We only use the CPU EP because the initializer we created is on CPU and we want the model to use that
        sess = onnxrt.InferenceSession(get_name("mul_1.onnx"), sess_options=so, providers=["CPUExecutionProvider"])
        res = sess.run(
            ["Y"],
            {"X": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)},
        )
        self.assertTrue(
            np.array_equal(
                res[0],
                np.array([[2.0, 2.0], [12.0, 12.0], [30.0, 30.0]], dtype=np.float32),
            )
        )

    def test_session_options_add_external_initializers(self):
        # Create an external initializer data in OrtValue
        # This initializer will replace the initializer with external data reference in the graph
        ortvalue_initializer = onnxrt.OrtValue.ortvalue_from_numpy(np.array([0, 0, 1, 1]).astype(np.int64))
        so = onnxrt.SessionOptions()
        so.add_external_initializers(["Pads_not_on_disk"], [ortvalue_initializer])
        # This should not throw
        onnxrt.InferenceSession(
            get_name("model_with_external_initializer_come_from_user.onnx"),
            sess_options=so,
            providers=["CPUExecutionProvider"],
        )

    def test_session_options_add_external_initializers_from_files_in_memory(self):
        # Provide external initializer file content directly from memory
        # The model references an external file named "Pads_not_on_disk.bin" for the initializer
        pads_bytes = np.array([0, 0, 1, 1], dtype=np.int64).tobytes()

        so = onnxrt.SessionOptions()
        so.add_external_initializers_from_files_in_memory(
            ["Pads_not_on_disk.bin"],
            [pads_bytes],
            [len(pads_bytes)],
        )

        # This should not throw
        onnxrt.InferenceSession(
            get_name("model_with_external_initializer_come_from_user.onnx"),
            sess_options=so,
            providers=["CPUExecutionProvider"],
        )

    def test_register_custom_ops_library(self):
        if sys.platform.startswith("win"):
            shared_library = os.path.abspath("custom_op_library.dll")
            if not os.path.exists(shared_library):
                raise FileNotFoundError(f"Unable to find '{shared_library}'")

        elif sys.platform.startswith("darwin"):
            shared_library = "libcustom_op_library.dylib"
            if not os.path.exists(shared_library):
                raise FileNotFoundError(f"Unable to find '{shared_library}'")

        else:
            shared_library = "./libcustom_op_library.so"
            if not os.path.exists(shared_library):
                raise FileNotFoundError(f"Unable to find '{shared_library}'")

        this = os.path.dirname(__file__)
        custom_op_model = os.path.join(this, "testdata", "custom_op_library", "custom_op_test.onnx")
        if not os.path.exists(custom_op_model):
            raise FileNotFoundError(f"Unable to find '{custom_op_model}'")

        so1 = onnxrt.SessionOptions()
        so1.register_custom_ops_library(shared_library)

        # Model loading successfully indicates that the custom op node could be resolved successfully
        sess1 = onnxrt.InferenceSession(
            custom_op_model, sess_options=so1, providers=available_providers_without_tvm_and_tensorrt
        )
        # Run with input data
        input_name_0 = sess1.get_inputs()[0].name
        input_name_1 = sess1.get_inputs()[1].name
        output_name = sess1.get_outputs()[0].name
        input_0 = np.ones((3, 5)).astype(np.float32)
        input_1 = np.zeros((3, 5)).astype(np.float32)
        res = sess1.run([output_name], {input_name_0: input_0, input_name_1: input_1})
        output_expected = np.ones((3, 5)).astype(np.float32)
        np.testing.assert_allclose(res[0], output_expected, rtol=1e-05, atol=1e-08)

        # Create an alias of SessionOptions instance
        # We will use this alias to construct another InferenceSession
        so2 = so1

        # Model loading successfully indicates that the custom op node could be resolved successfully
        onnxrt.InferenceSession(
            custom_op_model, sess_options=so2, providers=available_providers_without_tvm_and_tensorrt
        )

        # Create another SessionOptions instance with the same shared library referenced
        so3 = onnxrt.SessionOptions()
        so3.register_custom_ops_library(shared_library)
        onnxrt.InferenceSession(
            custom_op_model, sess_options=so3, providers=available_providers_without_tvm_and_tensorrt
        )

    def test_ort_value(self):
        providers_to_test = available_providers
        numpy_arr_input = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        numpy_arr_output = np.array([[1.0, 4.0], [9.0, 16.0], [25.0, 36.0]], dtype=np.float32)

        def test_session_with_ortvalue_input(ortvalue, providers):
            sess = onnxrt.InferenceSession(get_name("mul_1.onnx"), providers=providers)
            res = sess.run(["Y"], {"X": ortvalue})

            if "QNNExecutionProvider" in providers:
                # QNN runs float32 with fp16 precision, so relax accuracy expectations
                np.testing.assert_allclose(numpy_arr_output, res[0], rtol=1e-04, atol=1e-06)
            else:
                self.assertTrue(np.array_equal(res[0], numpy_arr_output))

            vect = sess._sess.run_with_ort_values({"X": ortvalue._get_c_value()}, ["Y"], RunOptions())
            self.assertIsInstance(vect, OrtValueVector)

        ortvalue1 = onnxrt.OrtValue.ortvalue_from_numpy(numpy_arr_input)
        self.assertEqual(ortvalue1.device_name(), "cpu")
        self.assertEqual(ortvalue1.shape(), [3, 2])
        self.assertEqual(ortvalue1.data_type(), "tensor(float)")
        self.assertEqual(ortvalue1.is_tensor(), True)
        # Assumes float32 and shape {3, 2} as above
        self.assertEqual(ortvalue1.tensor_size_in_bytes(), 4 * 2 * 3)
        self.assertTrue(np.array_equal(ortvalue1.numpy(), numpy_arr_input))

        # Pass in the constructed OrtValue to a session via Run() and check results
        test_session_with_ortvalue_input(ortvalue1, providers_to_test)

        # The constructed OrtValue should still be valid after being used in a session
        self.assertTrue(np.array_equal(ortvalue1.numpy(), numpy_arr_input))

        # test ort_value creation on top of the bytes
        float_tensor_data_type = 1  # TensorProto_DataType_FLOAT
        ort_value_with_type = onnxrt.OrtValue.ortvalue_from_numpy_with_onnx_type(
            numpy_arr_input, float_tensor_data_type
        )
        self.assertTrue(ort_value_with_type.is_tensor())
        self.assertEqual(float_tensor_data_type, ort_value_with_type.element_type())
        self.assertEqual([3, 2], ort_value_with_type.shape())

        if "CUDAExecutionProvider" in providers_to_test:
            ortvalue2 = onnxrt.OrtValue.ortvalue_from_numpy(numpy_arr_input, "cuda", 0)
            self.assertEqual(ortvalue2.device_name(), "cuda")
            self.assertEqual(ortvalue2.shape(), [3, 2])
            self.assertEqual(ortvalue2.data_type(), "tensor(float)")
            self.assertEqual(ortvalue2.is_tensor(), True)
            self.assertTrue(np.array_equal(ortvalue2.numpy(), numpy_arr_input))

            # Pass in the constructed OrtValue to a session via Run() and check results
            test_session_with_ortvalue_input(ortvalue2, providers_to_test)

            # The constructed OrtValue should still be valid after being used in a session
            self.assertTrue(np.array_equal(ortvalue2.numpy(), numpy_arr_input))

    def test_ort_value_gh_issue9799(self):
        if "CUDAExecutionProvider" in onnxrt.get_available_providers():
            session = onnxrt.InferenceSession(
                get_name("identity_9799.onnx"),
                providers=onnxrt.get_available_providers(),
            )

            for seq_length in range(40, 200):
                inps = np.ones((seq_length, 16, 7, 5, 3, 3)).astype(np.float32)
                ort_val = onnxrt.OrtValue.ortvalue_from_numpy(inps, "cuda", 0)
                upstreams_onnxrt = {"input": ort_val}
                outs = session.run(output_names=["output"], input_feed=upstreams_onnxrt)[0]
                self.assertTrue(np.allclose(inps, outs))

    @unittest.skipIf(not hasattr(C.OrtValue, "from_dlpack"), "dlpack not enabled in this build")
    def test_ort_value_dlpack(self):
        # Tests originally from orttraining/orttraining/test/python/orttraining_test_ortvalue.py testOrtValueDlPack_float32
        numpy_arr_input = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        ortvalue = onnxrt.OrtValue.ortvalue_from_numpy(numpy_arr_input)
        self.assertEqual(numpy_arr_input.shape, tuple(ortvalue.shape()))
        ptr = ortvalue._ortvalue.data_ptr()

        dlp = ortvalue._ortvalue.to_dlpack()
        self.assertFalse(C.is_dlpack_uint8_tensor(dlp))
        ortvalue2 = C.OrtValue.from_dlpack(dlp, False)
        self.assertEqual(ptr, ortvalue2.data_ptr())
        new_array = ortvalue2.numpy()
        np.testing.assert_equal(numpy_arr_input, new_array)

        dlp = ortvalue._ortvalue.__dlpack__()
        self.assertFalse(C.is_dlpack_uint8_tensor(dlp))
        ortvalue2 = C.OrtValue.from_dlpack(dlp, False)
        self.assertEqual(ptr, ortvalue2.data_ptr())
        new_array = ortvalue2.numpy()
        np.testing.assert_equal(numpy_arr_input, new_array)

        device = ortvalue._ortvalue.__dlpack_device__()
        self.assertEqual((1, 0), device)

    @unittest.skipIf(not hasattr(C.OrtValue, "from_dlpack"), "dlpack not enabled in this build")
    def test_ort_value_dlpack_zero_size(self):
        # Zero-size tensors are vacuously contiguous; from_dlpack must accept them.
        # Regression test: OrtValue.from_dlpack was incorrectly rejecting zero-size tensors.
        zero_size_shapes = [
            (1, 8, 0, 128),  # zero in the middle (KV-cache use case)
            (0,),  # 1-D zero-size
            (0, 4),  # zero leading dimension
            (4, 0),  # zero trailing dimension
        ]
        for shape in zero_size_shapes:
            with self.subTest(shape=shape):
                arr = np.zeros(shape, dtype=np.float32)
                # Test via numpy __dlpack__ protocol
                dlp = arr.__dlpack__()
                ortvalue = C.OrtValue.from_dlpack(dlp, False)
                self.assertEqual(list(shape), list(ortvalue.shape()))
                # Test round-trip: OrtValue -> dlpack -> OrtValue
                ort_input = onnxrt.OrtValue.ortvalue_from_numpy(arr)
                dlp2 = ort_input._ortvalue.to_dlpack()
                ortvalue2 = C.OrtValue.from_dlpack(dlp2, False)
                self.assertEqual(list(shape), list(ortvalue2.shape()))

    def test_ort_value_array_protocol(self):
        """Test that OrtValue supports numpy's __array__ protocol."""
        numpy_arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        ortvalue = onnxrt.OrtValue.ortvalue_from_numpy(numpy_arr)

        # np.asarray should work via __array__ and share memory (zero-copy)
        result = np.asarray(ortvalue)
        np.testing.assert_equal(numpy_arr, result)
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(ortvalue.data_ptr(), result.ctypes.data)

        # np.array should also work
        result2 = np.array(ortvalue)
        np.testing.assert_equal(numpy_arr, result2)

        # same dtype should still share memory (no unnecessary copy)
        result_same = np.asarray(ortvalue, dtype=np.float32)
        np.testing.assert_equal(numpy_arr, result_same)
        self.assertEqual(ortvalue.data_ptr(), result_same.ctypes.data)

        # dtype conversion via __array__
        result_f64 = np.asarray(ortvalue, dtype=np.float64)
        np.testing.assert_equal(numpy_arr.astype(np.float64), result_f64)
        self.assertEqual(result_f64.dtype, np.float64)

        # Integer tensor
        int_arr = np.array([1, 2, 3], dtype=np.int64)
        ortvalue_int = onnxrt.OrtValue.ortvalue_from_numpy(int_arr)
        result_int = np.asarray(ortvalue_int)
        np.testing.assert_equal(int_arr, result_int)
        self.assertEqual(result_int.dtype, np.int64)

        # Boolean tensor
        bool_arr = np.array([True, False, True], dtype=np.bool_)
        ortvalue_bool = onnxrt.OrtValue.ortvalue_from_numpy(bool_arr)
        result_bool = np.asarray(ortvalue_bool)
        np.testing.assert_equal(bool_arr, result_bool)
        self.assertEqual(result_bool.dtype, np.bool_)

    @unittest.skipUnless(
        importlib.util.find_spec("onnx") is not None,
        "onnx package is required to build the test model",
    )
    def test_run_output_does_not_alias_input_passthrough(self):
        """Test that session.run() returns independent numpy arrays when a model
        input passes through as a model output. Reproduces the dangling-pointer
        corruption described in https://github.com/microsoft/onnxruntime/issues/21922
        """
        import onnx  # noqa: PLC0415

        # Build a model where 'input_0' is both a graph input and a graph output,
        # plus a computed output (input_0 + 10).
        inp_shape = [1, 2, 2, 2]
        input_0 = onnx.helper.make_tensor_value_info("input_0", onnx.TensorProto.FLOAT, inp_shape)
        output_plus10 = onnx.helper.make_tensor_value_info("plus_10", onnx.TensorProto.FLOAT, inp_shape)
        ten_const = onnx.numpy_helper.from_array(np.array(10, dtype=np.float32), "ten_const")
        add_node = onnx.helper.make_node("Add", ["input_0", "ten_const"], ["plus_10"], name="Add0")
        graph = onnx.helper.make_graph(
            [add_node],
            "PassthroughTest",
            [input_0],
            [output_plus10, input_0],
            initializer=[ten_const],
        )
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 21)])
        model = onnx.shape_inference.infer_shapes(model)

        sess_options = onnxrt.SessionOptions()
        sess_options.graph_optimization_level = onnxrt.GraphOptimizationLevel.ORT_DISABLE_ALL
        session = onnxrt.InferenceSession(
            model.SerializeToString(),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        num_runs = 7
        all_run_outputs = []
        for run_index in range(num_runs):
            input_data = np.full(inp_shape, float(run_index), dtype=np.float32)
            outputs = session.run(None, {"input_0": input_data})

            # Immediately after run, outputs must be correct
            np.testing.assert_array_equal(
                outputs[0],
                np.full(inp_shape, run_index + 10.0, dtype=np.float32),
                err_msg=f"Run {run_index}: 'plus_10' wrong immediately after run",
            )
            np.testing.assert_array_equal(
                outputs[1],
                np.full(inp_shape, float(run_index), dtype=np.float32),
                err_msg=f"Run {run_index}: 'input_0' wrong immediately after run",
            )

            # The pass-through output must NOT alias the input buffer —
            # it must be an independent copy so it survives across runs.
            self.assertFalse(
                np.shares_memory(outputs[1], input_data),
                f"Run {run_index}: 'input_0' unexpectedly aliases the input buffer",
            )
            all_run_outputs.append(outputs)

        # After all runs, every saved output must still hold its original value.
        for run_index, outputs in enumerate(all_run_outputs):
            np.testing.assert_array_equal(
                outputs[0],
                np.full(inp_shape, run_index + 10.0, dtype=np.float32),
                err_msg=f"Run {run_index}: 'plus_10' corrupted after loop",
            )
            np.testing.assert_array_equal(
                outputs[1],
                np.full(inp_shape, float(run_index), dtype=np.float32),
                err_msg=f"Run {run_index}: 'input_0' corrupted after loop (issue #21922)",
            )

    def test_run_session_owned_output_is_zero_copy(self):
        """Verify that session-allocated CPU outputs (the common case) expose
        a backing base object instead of owning a separate numpy buffer."""
        sess = onnxrt.InferenceSession(get_name("mul_1.onnx"), providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        result = sess.run(None, {input_name: x})
        # The output should be a numpy array; for a session-owned buffer
        # it should not be a separately-allocated copy.  We verify that by
        # checking the array is backed by another object via ``output.base``.
        output = result[0]
        self.assertIsNotNone(output.base, "Session-owned output should have a backing base object")

    @unittest.skipIf(not hasattr(C.OrtValue, "from_dlpack"), "dlpack not enabled in this build")
    def test_ort_value_dlpack_protocol(self):
        """Test that OrtValue exposes __dlpack__ and __dlpack_device__ protocols."""
        numpy_arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        ortvalue = onnxrt.OrtValue.ortvalue_from_numpy(numpy_arr)

        # __dlpack_device__ should return (device_type, device_id) for CPU
        device = ortvalue.__dlpack_device__()
        self.assertEqual((1, 0), device)

        # __dlpack__ should return a capsule that can be consumed by from_dlpack
        dlp = ortvalue.__dlpack__()
        ortvalue2 = onnxrt.OrtValue.from_dlpack(dlp)
        np.testing.assert_equal(numpy_arr, ortvalue2.numpy())

    @unittest.skipIf(not hasattr(C.OrtValue, "from_dlpack"), "dlpack not enabled in this build")
    def test_ort_value_from_dlpack_protocol_object(self):
        """Test OrtValue.from_dlpack with objects implementing __dlpack__ protocol."""
        numpy_arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

        # numpy arrays support __dlpack__ protocol since numpy 1.22
        if hasattr(numpy_arr, "__dlpack__"):
            ortvalue = onnxrt.OrtValue.from_dlpack(numpy_arr)
            np.testing.assert_equal(numpy_arr, ortvalue.numpy())
            self.assertEqual(list(numpy_arr.shape), list(ortvalue.shape()))

        # Round-trip: numpy -> OrtValue -> OrtValue (via __dlpack__)
        ortvalue_src = onnxrt.OrtValue.ortvalue_from_numpy(numpy_arr)
        ortvalue_dst = onnxrt.OrtValue.from_dlpack(ortvalue_src)
        np.testing.assert_equal(numpy_arr, ortvalue_dst.numpy())
        # Verify shared memory (no copy)
        self.assertEqual(ortvalue_src.data_ptr(), ortvalue_dst.data_ptr())

    @unittest.skipIf(not hasattr(C.OrtValue, "from_dlpack"), "dlpack not enabled in this build")
    def test_ort_value_from_dlpack_bool(self):
        """Test that from_dlpack auto-detects boolean tensors."""
        bool_arr = np.array([True, False, True, False], dtype=np.bool_)
        ortvalue_src = onnxrt.OrtValue.ortvalue_from_numpy(bool_arr)

        # Round-trip through DLPack should preserve bool dtype
        ortvalue_dst = onnxrt.OrtValue.from_dlpack(ortvalue_src)
        result = ortvalue_dst.numpy()
        np.testing.assert_equal(bool_arr, result)

        # Ensure uint8 is NOT falsely detected as bool
        uint8_arr = np.array([1, 2, 255], dtype=np.uint8)
        ortvalue_uint8 = onnxrt.OrtValue.ortvalue_from_numpy(uint8_arr)
        ortvalue_uint8_dst = onnxrt.OrtValue.from_dlpack(ortvalue_uint8)
        result_uint8 = ortvalue_uint8_dst.numpy()
        np.testing.assert_equal(uint8_arr, result_uint8)
        self.assertEqual(result_uint8.dtype, np.uint8)

    def test_sparse_tensor_coo_format(self):
        cpu_device = onnxrt.OrtDevice.make("cpu", 0)
        shape = [9, 9]
        values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        # Linear indices
        indices = np.array([3, 5, 15], dtype=np.int64)
        sparse_tensor = onnxrt.SparseTensor.sparse_coo_from_numpy(shape, values, indices, cpu_device)
        self.assertEqual(sparse_tensor.format(), onnxrt.OrtSparseFormat.ORT_SPARSE_COO)
        self.assertEqual(sparse_tensor.dense_shape(), shape)
        self.assertEqual(sparse_tensor.data_type(), "sparse_tensor(float)")
        self.assertEqual(sparse_tensor.device_name(), "cpu")

        # Get Data View on a numeric type.
        values_ret = sparse_tensor.values()
        self.assertFalse(values_ret.flags.writeable)
        indices_ret = sparse_tensor.as_coo_view().indices()
        self.assertFalse(indices_ret.flags.writeable)
        # Run GC to test that values_ret still exhibits expected data
        gc.collect()
        self.assertTrue(np.array_equal(values, values_ret))
        self.assertTrue(np.array_equal(indices, indices_ret))

        # Test new Ortvalue interfaces
        ort_value = onnxrt.OrtValue.ort_value_from_sparse_tensor(sparse_tensor)
        sparse_tensor = ort_value.as_sparse_tensor()
        values_ret = sparse_tensor.values()
        self.assertFalse(values_ret.flags.writeable)
        indices_ret = sparse_tensor.as_coo_view().indices()
        self.assertFalse(indices_ret.flags.writeable)
        gc.collect()

        # Test string data on cpu only, need to subst values only
        str_values = np.array(["xyz", "yxz", "zyx"], dtype=str)
        str_sparse_tensor = onnxrt.SparseTensor.sparse_coo_from_numpy(shape, str_values, indices, cpu_device)
        self.assertEqual(str_sparse_tensor.format(), onnxrt.OrtSparseFormat.ORT_SPARSE_COO)
        self.assertEqual(str_sparse_tensor.dense_shape(), shape)
        self.assertEqual(str_sparse_tensor.data_type(), "sparse_tensor(string)")
        self.assertEqual(str_sparse_tensor.device_name(), "cpu")

        # Get string values back
        str_values_ret = str_sparse_tensor.values()
        self.assertTrue(np.array_equal(str_values, str_values_ret))
        # Check indices
        str_indices_ret = str_sparse_tensor.as_coo_view().indices()
        gc.collect()
        self.assertFalse(str_indices_ret.flags.writeable)
        self.assertTrue(np.array_equal(indices, str_indices_ret))

        cuda_device = onnxrt.OrtDevice.make("cuda", 0)
        if "CUDAExecutionProvider" in onnxrt.get_available_providers():
            # Test to_cuda
            copy_on_cuda = sparse_tensor.to_cuda(cuda_device)
            self.assertEqual(copy_on_cuda.dense_shape(), shape)
            self.assertEqual(copy_on_cuda.data_type(), "sparse_tensor(float)")
            self.assertEqual(copy_on_cuda.device_name(), "cuda")

            # Test that gpu copy would fail to copy to cuda
            with self.assertRaises(RuntimeError):
                copy_on_cuda.to_cuda(cuda_device)
            # Test that string tensor copy would fail
            with self.assertRaises(RuntimeError):
                str_sparse_tensor.to_cuda(cuda_device)
        else:
            # No cuda available
            with self.assertRaises(RuntimeError):
                sparse_tensor.to_cuda(cuda_device)

    def test_sparse_tensor_csr_format(self):
        cpu_device = onnxrt.OrtDevice.make("cpu", 0)
        shape = [9, 9]
        values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        inner_indices = np.array([1, 1, 1], dtype=np.int64)
        outer_indices = np.array([0, 1, 2, 3, 3, 3, 3, 3, 3, 3], dtype=np.int64)
        sparse_tensor = onnxrt.SparseTensor.sparse_csr_from_numpy(
            shape, values, inner_indices, outer_indices, cpu_device
        )
        self.assertEqual(sparse_tensor.format(), onnxrt.OrtSparseFormat.ORT_SPARSE_CSRC)
        self.assertEqual(sparse_tensor.dense_shape(), shape)
        self.assertEqual(sparse_tensor.data_type(), "sparse_tensor(float)")
        self.assertEqual(sparse_tensor.device_name(), "cpu")

        # Test CSR(C) indices
        inner_indices_ret = sparse_tensor.as_csrc_view().inner()
        outer_indices_ret = sparse_tensor.as_csrc_view().outer()
        self.assertFalse(inner_indices_ret.flags.writeable)
        self.assertFalse(outer_indices_ret.flags.writeable)
        gc.collect()
        self.assertTrue(np.array_equal(inner_indices, inner_indices_ret))
        self.assertTrue(np.array_equal(outer_indices, outer_indices_ret))

        # Test with strings
        str_values = np.array(["xyz", "yxz", "zyx"], dtype=str)
        str_sparse_tensor = onnxrt.SparseTensor.sparse_csr_from_numpy(
            shape, str_values, inner_indices, outer_indices, cpu_device
        )
        self.assertEqual(str_sparse_tensor.format(), onnxrt.OrtSparseFormat.ORT_SPARSE_CSRC)
        self.assertEqual(str_sparse_tensor.dense_shape(), shape)
        self.assertEqual(str_sparse_tensor.data_type(), "sparse_tensor(string)")
        self.assertEqual(str_sparse_tensor.device_name(), "cpu")

        if "CUDAExecutionProvider" in onnxrt.get_available_providers():
            cuda_device = onnxrt.OrtDevice.make("cuda", 0)
            cuda_sparse_tensor = sparse_tensor.to_cuda(cuda_device)
            self.assertEqual(cuda_sparse_tensor.device_name(), "cuda")
            self.assertEqual(cuda_sparse_tensor.format(), onnxrt.OrtSparseFormat.ORT_SPARSE_CSRC)
            self.assertEqual(cuda_sparse_tensor.dense_shape(), shape)
            self.assertEqual(cuda_sparse_tensor.data_type(), "sparse_tensor(float)")

    def test_run_model_with_cuda_copy_stream(self):
        available_providers = onnxrt.get_available_providers()

        if "CUDAExecutionProvider" not in available_providers:
            print("Skipping testRunModelWithCudaCopyStream when CUDA is not available")
        else:
            # adapted from issue #4829 for a race condition when copy is not on default stream
            # note:
            # 1. if there are intermittent failure in this test, something is wrong
            # 2. it's easier to repro on slower GPU (like M60, Geforce 1070)

            # to repro #4829, set the CUDA EP do_copy_in_default_stream option to False
            providers = [
                ("CUDAExecutionProvider", {"do_copy_in_default_stream": True}),
                "CPUExecutionProvider",
            ]

            session = onnxrt.InferenceSession(get_name("issue4829.onnx"), providers=providers)
            shape = np.array([2, 2], dtype=np.int64)
            for _iteration in range(100000):
                session.run(output_names=["output"], input_feed={"shape": shape})

    def test_ort_device(self):
        cpu_device = onnxrt.OrtDevice.make("cpu", 0)
        self.assertEqual(cpu_device.device_id(), 0)
        self.assertEqual(cpu_device.device_type(), 0)
        self.assertEqual(cpu_device.device_vendor_id(), onnxrt.OrtDeviceVendorId.NONE)
        self.assertEqual(cpu_device.device_mem_type(), 0)

        cuda_device = onnxrt.OrtDevice.make("cuda", 0)
        self.assertEqual(cuda_device.device_vendor_id(), onnxrt.OrtDeviceVendorId.NVIDIA)
        self.assertEqual(onnxrt.OrtDeviceVendorId.NVIDIA, 0x10DE)

        webgpu_device = onnxrt.OrtDevice.make("webgpu", 0)
        self.assertEqual(webgpu_device.device_vendor_id(), onnxrt.OrtDeviceVendorId.NONE)

    def test_ort_memory_info(self):
        cpu_memory_info = onnxrt.OrtMemoryInfo(
            "Cpu",
            onnxrt.OrtAllocatorType.ORT_ARENA_ALLOCATOR,
            0,
            onnxrt.OrtMemType.DEFAULT,
        )
        self.assertEqual(cpu_memory_info.name, "Cpu")
        self.assertEqual(cpu_memory_info.device_id, 0)
        self.assertEqual(cpu_memory_info.mem_type, onnxrt.OrtMemType.DEFAULT)
        self.assertEqual(cpu_memory_info.allocator_type, onnxrt.OrtAllocatorType.ORT_ARENA_ALLOCATOR)
        self.assertEqual(cpu_memory_info.device_mem_type, onnxrt.OrtDeviceMemoryType.DEFAULT)
        self.assertEqual(cpu_memory_info.device_vendor_id, 0)

    def test_ort_memory_info_create_v2(self):
        cpu_memory_info = onnxrt.OrtMemoryInfo.create_v2(
            "Test",
            onnxrt.OrtMemoryInfoDeviceType.CPU,
            0,  # vendor_id
            0,  # device_id
            onnxrt.OrtDeviceMemoryType.DEFAULT,
            128,  # alignment
            onnxrt.OrtAllocatorType.ORT_ARENA_ALLOCATOR,
        )
        self.assertEqual(cpu_memory_info.name, "Test")
        self.assertEqual(cpu_memory_info.device_id, 0)
        self.assertEqual(cpu_memory_info.mem_type, onnxrt.OrtMemType.DEFAULT)
        self.assertEqual(cpu_memory_info.allocator_type, onnxrt.OrtAllocatorType.ORT_ARENA_ALLOCATOR)
        self.assertEqual(cpu_memory_info.device_mem_type, onnxrt.OrtDeviceMemoryType.DEFAULT)
        self.assertEqual(cpu_memory_info.device_vendor_id, 0)

    def test_shared_allocator_using_create_and_register_allocator(self):
        # Create and register an arena based allocator

        # To create an OrtArenaCfg using non-default parameters, use one of below templates:
        # ort_arena_cfg = onnxrt.OrtArenaCfg(0, -1, -1, -1) - Note: doesn't expose initial_growth_chunk_size_bytes/max_power_of_two_extend_bytes option
        # ort_arena_cfg = onnxrt.OrtArenaCfg({"max_mem": -1, ""arena_extend_strategy": 1, etc..})
        ort_memory_info = onnxrt.OrtMemoryInfo(
            "Cpu",
            onnxrt.OrtAllocatorType.ORT_ARENA_ALLOCATOR,
            0,
            onnxrt.OrtMemType.DEFAULT,
        )
        # Use this option if using non-default OrtArenaCfg : onnxrt.create_and_register_allocator(ort_memory_info, ort_arena_cfg)
        onnxrt.create_and_register_allocator(ort_memory_info, None)

        # Create a session that will use the registered arena based allocator
        so1 = onnxrt.SessionOptions()
        so1.log_severity_level = 1
        so1.add_session_config_entry("session.use_env_allocators", "1")
        onnxrt.InferenceSession(
            get_name("mul_1.onnx"),
            sess_options=so1,
            providers=onnxrt.get_available_providers(),
        )

        # Create a session that will NOT use the registered arena based allocator
        so2 = onnxrt.SessionOptions()
        so2.log_severity_level = 1
        onnxrt.InferenceSession(
            get_name("mul_1.onnx"),
            sess_options=so2,
            providers=onnxrt.get_available_providers(),
        )

        if "CUDAExecutionProvider" in available_providers:
            cuda_mem_info = onnxrt.OrtMemoryInfo(
                "Cuda",
                onnxrt.OrtAllocatorType.ORT_ARENA_ALLOCATOR,
                0,
                onnxrt.OrtMemType.DEFAULT,
            )
            ort_arena_cfg = onnxrt.OrtArenaCfg(0, -1, -1, -1)
            onnxrt.create_and_register_allocator_v2("CUDAExecutionProvider", cuda_mem_info, {}, ort_arena_cfg)
            so3 = onnxrt.SessionOptions()
            so3.log_severity_level = 1
            so3.add_session_config_entry("session.use_env_allocators", "1")
            onnxrt.InferenceSession(
                get_name("mul_1.onnx"),
                sess_options=so3,
                providers=onnxrt.get_available_providers(),
            )

    def test_session_scoped_cpu_ortvalue(self):
        """A session allocator value is usable only with the session that created it.

        Population goes through onnxruntime.copy_tensors rather than a session-specific update path,
        so this covers allocation and provenance only; the copy itself is covered by the WebGPU
        graph-capture flow, where a device data transfer is always registered.
        """
        session = onnxrt.InferenceSession(get_name("mul_1.onnx"), providers=["CPUExecutionProvider"])
        input_metadata = session.get_inputs()[0]
        input_value = np.arange(np.prod(input_metadata.shape), dtype=np.float32).reshape(input_metadata.shape)

        session_value = session.create_ortvalue_from_shape_and_type(input_metadata.shape, np.float32, "cpu")
        self.assertFalse(session_value._is_webgpu_buffer)
        self.assertIs(session_value._session, session._sess)
        self.assertEqual(session_value.shape(), list(input_metadata.shape))

        other_session = onnxrt.InferenceSession(get_name("mul_1.onnx"), providers=["CPUExecutionProvider"])
        other_session_value = other_session.create_ortvalue_from_shape_and_type(input_metadata.shape, np.float32, "cpu")
        with self.assertRaisesRegex(ValueError, "same session"):
            session_value.update_inplace(other_session_value)
        with self.assertRaisesRegex(ValueError, "session that created it"):
            other_session.run(None, {input_metadata.name: session_value})
        with self.assertRaisesRegex(ValueError, "session that created it"):
            other_session.run_with_ort_values(None, {input_metadata.name: session_value})
        with self.assertRaisesRegex(ValueError, "session that created it"):
            session.io_binding().bind_ortvalue_input(input_metadata.name, other_session_value)
        with self.assertRaisesRegex(ValueError, "session that created it"):
            other_session.run_with_iobinding(session.io_binding())

        reset_session = onnxrt.InferenceSession(get_name("mul_1.onnx"), providers=["CPUExecutionProvider"])
        stale_value = reset_session.create_ortvalue_from_shape_and_type(input_metadata.shape, np.float32, "cpu")
        stale_binding = reset_session.io_binding()
        reset_session.set_providers(["CPUExecutionProvider"])
        with self.assertRaisesRegex(ValueError, "session that created it"):
            reset_session.run(None, {input_metadata.name: stale_value})
        with self.assertRaisesRegex(ValueError, "session that created it"):
            reset_session.run_with_iobinding(stale_binding)

        scalar_value = session.create_ortvalue_from_shape_and_type([], np.float32, "cpu")
        self.assertEqual(scalar_value.shape(), [])
        # A plain run with host inputs is unaffected by any of the above.
        np.testing.assert_array_equal(
            session.run(None, {input_metadata.name: input_value})[0],
            session.run(None, {input_metadata.name: input_value})[0],
        )

    def test_is_webgpu_buffer_is_read_only(self):
        """OrtValue._is_webgpu_buffer must always come from the native value and never be settable.

        Graph-capture validation and the OrtValue copy paths refuse host memory based on it, so a
        settable attribute would let a CPU tensor pass itself off as a WebGPU device tensor.
        """
        cpu_value = onnxrt.OrtValue.ortvalue_from_numpy(np.zeros((2, 2), dtype=np.float32))
        self.assertFalse(cpu_value._is_webgpu_buffer)
        self.assertEqual(cpu_value._is_webgpu_buffer, cpu_value._get_c_value()._is_webgpu_buffer())

        with self.assertRaises(AttributeError):
            cpu_value._is_webgpu_buffer = True
        self.assertFalse(cpu_value._is_webgpu_buffer)

        # A spoofed value must not be able to satisfy graph-capture validation either.
        session = onnxrt.InferenceSession(get_name("mul_1.onnx"), providers=["CPUExecutionProvider"])
        io_binding = session.io_binding()
        io_binding.bind_ortvalue_input(session.get_inputs()[0].name, cpu_value)
        with self.assertRaisesRegex(ValueError, "requires fixed WebGPU device OrtValues"):
            io_binding._validate_capture_bindings()

    def test_device_ortvalue_provenance_rules(self):
        """Session-scoped OrtValues are usable only with the session that created them."""
        session = onnxrt.InferenceSession(get_name("mul_1.onnx"), providers=["CPUExecutionProvider"])
        other_session = onnxrt.InferenceSession(get_name("mul_1.onnx"), providers=["CPUExecutionProvider"])
        input_metadata = session.get_inputs()[0]
        input_value = np.arange(np.prod(input_metadata.shape), dtype=np.float32).reshape(input_metadata.shape)

        # Copy CPU storage so in-place updates remain isolated.
        unowned = onnxrt.OrtValue.ortvalue_from_numpy(input_value.copy())
        owned = session.create_ortvalue_from_shape_and_type(input_metadata.shape, np.float32, "cpu")
        foreign = other_session.create_ortvalue_from_shape_and_type(input_metadata.shape, np.float32, "cpu")

        # Shared-allocator and locally owned values are both accepted.
        session._validate_ortvalue_ownership([unowned, owned])

        with self.assertRaises(ValueError) as raised:
            session._validate_ortvalue_ownership([foreign])
        self.assertIn("session that created it", str(raised.exception))

        io_binding = session.io_binding()
        io_binding.bind_ortvalue_input(input_metadata.name, unowned)
        io_binding.bind_ortvalue_input(input_metadata.name, owned)
        with self.assertRaisesRegex(ValueError, "must be bound to the session that created it"):
            io_binding.bind_ortvalue_input(input_metadata.name, foreign)
        with self.assertRaisesRegex(ValueError, "must be bound to the session that created it"):
            io_binding.bind_ortvalue_output(session.get_outputs()[0].name, foreign)

        # Shared values remain writable through the environment transfer.
        updated_input = input_value + 1.0
        unowned.update_inplace(updated_input)
        np.testing.assert_array_equal(unowned.numpy(), updated_input)
        unowned.update_inplace(onnxrt.OrtValue.ortvalue_from_numpy(input_value.copy()))
        np.testing.assert_array_equal(unowned.numpy(), input_value)

        with self.assertRaisesRegex(ValueError, "same session"):
            owned.update_inplace(foreign)

    def test_ortvalue_ownership_allows_matching_webgpu_context(self):
        """run() and IOBinding share one rule: a WebGPU buffer from another session is
        accepted only when the WebGPU contexts match.

        Context ids are injected because a second WebGPU context needs caller-supplied Dawn
        handles, which Python cannot reach.
        """

        class WebGpuValue(onnxrt.OrtValue):
            _is_webgpu_buffer = True

        class FakeSession:
            def __init__(self, context_id):
                self._context_id = context_id

            def webgpu_context_id(self):
                return self._context_id

        class FakeOwner:
            def __init__(self, session):
                self._sess = session

        session = onnxrt.InferenceSession(get_name("mul_1.onnx"), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        source = onnxrt.OrtValue.ortvalue_from_numpy(np.zeros(session.get_inputs()[0].shape, dtype=np.float32))

        def owned_by(context_id):
            value = WebGpuValue(source._get_c_value())
            value._session = FakeSession(context_id)
            return value

        target = FakeOwner(FakeSession(0))
        onnxrt.InferenceSession._validate_ortvalue_ownership(target, [owned_by(0)])
        with self.assertRaisesRegex(ValueError, "must be used with the session that created it"):
            onnxrt.InferenceSession._validate_ortvalue_ownership(target, [owned_by(1)])

        # The binding paths apply the same rule, so they must accept the same value.
        io_binding = session.io_binding()
        io_binding._session = FakeSession(0)
        io_binding.bind_ortvalue_input(input_name, owned_by(0))
        io_binding.bind_ortvalue_output(output_name, owned_by(0))
        with self.assertRaisesRegex(ValueError, "must be bound to the session that created it"):
            io_binding.bind_ortvalue_input(input_name, owned_by(1))
        with self.assertRaisesRegex(ValueError, "must be bound to the session that created it"):
            io_binding.bind_ortvalue_output(output_name, owned_by(1))

        # update_inplace shares the rule, so a same-context source must copy through.
        payload = np.arange(np.prod(session.get_inputs()[0].shape), dtype=np.float32).reshape(
            session.get_inputs()[0].shape
        )
        destination = WebGpuValue(onnxrt.OrtValue.ortvalue_from_numpy(payload.copy())._get_c_value())
        destination._session = FakeSession(0)
        destination.update_inplace(owned_by(0))
        np.testing.assert_array_equal(destination.numpy(), np.zeros_like(payload))
        with self.assertRaisesRegex(ValueError, "must originate from the same session"):
            destination.update_inplace(owned_by(1))

    def test_foreign_webgpu_context_predicate(self):
        """Unit-test the copy_tensors context predicate without needing a second WebGPU context.

        A real custom context requires caller-supplied Dawn instance and device handles, which are
        not reachable from Python, so the context id is injected here instead. This is the only
        coverage of the rejection branch that runs on a CPU-only machine.
        """

        class FakeValue:
            def __init__(self, is_webgpu, session):
                self._is_webgpu_buffer = is_webgpu
                self._session = session

        default_session = object()
        custom_session = object()
        no_webgpu_session = object()
        context_ids = {id(default_session): 0, id(custom_session): 1, id(no_webgpu_session): -1}

        def context_id_getter(session):
            return context_ids[id(session)]

        cpu_unowned = FakeValue(False, None)
        cpu_owned = FakeValue(False, custom_session)
        webgpu_unowned = FakeValue(True, None)
        webgpu_default = FakeValue(True, default_session)
        webgpu_custom = FakeValue(True, custom_session)
        webgpu_no_ep = FakeValue(True, no_webgpu_session)

        # Only a WebGPU value that can be attributed to a non-default context is rejected.
        for allowed in (cpu_unowned, cpu_owned, webgpu_unowned, webgpu_default, webgpu_no_ep):
            self.assertFalse(_has_foreign_webgpu_context([allowed], context_id_getter))
        self.assertTrue(_has_foreign_webgpu_context([webgpu_custom], context_id_getter))

        # One offending value anywhere in the batch is enough, in either direction.
        self.assertTrue(_has_foreign_webgpu_context([webgpu_default, webgpu_custom], context_id_getter))
        self.assertTrue(_has_foreign_webgpu_context([cpu_unowned, webgpu_custom], context_id_getter))
        self.assertFalse(_has_foreign_webgpu_context([], context_id_getter))

    def test_webgpu_context_id_reports_default_for_cpu_session(self):
        """A session with no WebGPU EP reports -1, which the copy_tensors predicate treats as unknown."""
        session = onnxrt.InferenceSession(get_name("mul_1.onnx"), providers=["CPUExecutionProvider"])
        self.assertEqual(session._sess.webgpu_context_id(), -1)

    def test_graph_annotation_id_run_option(self):
        """gpu_graph_id defaults to 0, accepts integer strings, and uses -1 to skip capture."""
        self.assertEqual(_GRAPH_ANNOTATION_SKIP, -1)
        # No run options, and run options without the entry, both mean the default graph.
        self.assertEqual(_graph_annotation_id(None), 0)
        self.assertEqual(_graph_annotation_id(onnxrt.RunOptions()), 0)

        for value, expected in (("-1", -1), ("0", 0), ("1", 1), ("7", 7)):
            run_options = onnxrt.RunOptions()
            run_options.add_run_config_entry("gpu_graph_id", value)
            self.assertEqual(_graph_annotation_id(run_options), expected)

        invalid_options = onnxrt.RunOptions()
        invalid_options.add_run_config_entry("gpu_graph_id", "not-an-int")
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            _graph_annotation_id(invalid_options)

    def test_reset_session_releases_captured_graphs(self):
        """set_providers() must not leave captured-graph bookkeeping pointing at the replaced session.

        The dict is keyed by graph annotation id and holds a weakref to the IOBinding that captured
        it, so a stale entry would reject a binding from the replacement session and a later
        release_captured_graph() would act on the replacement session while unpinning the old
        binding.
        """
        session = onnxrt.InferenceSession(get_name("mul_1.onnx"), providers=["CPUExecutionProvider"])
        input_metadata = session.get_inputs()[0]
        output_name = session.get_outputs()[0].name
        input_value = np.arange(np.prod(input_metadata.shape), dtype=np.float32).reshape(input_metadata.shape)

        stale_binding = session.io_binding()
        stale_binding.bind_cpu_input(input_metadata.name, input_value)
        stale_binding.bind_output(output_name)
        # Reproduce exactly what run_with_iobinding() records once a graph has been captured.
        # Capture itself needs a WebGPU device, but the bookkeeping this exercises does not.
        stale_signature = stale_binding._capture_signature()
        session._captured_graph_bindings[0] = (weakref.ref(stale_binding), stale_signature)
        stale_binding._pinned_graph_ids.add(0)

        session.set_providers(["CPUExecutionProvider"])

        # Both sides of the bookkeeping must be cleared, and the old binding usable again.
        self.assertEqual(session._captured_graph_bindings, {})
        self.assertEqual(stale_binding._pinned_graph_ids, set())
        stale_binding.clear_binding_inputs()
        stale_binding.clear_binding_outputs()

        # The same graph id must be capturable again through the replacement session.
        fresh_binding = session.io_binding()
        fresh_binding.bind_cpu_input(input_metadata.name, input_value)
        fresh_binding.bind_output(output_name)
        session.run_with_iobinding(fresh_binding)
        np.testing.assert_allclose(
            fresh_binding.copy_outputs_to_cpu()[0],
            session.run(None, {input_metadata.name: input_value})[0],
        )
        session.release_captured_graph()

    def test_run_with_ortvaluevector_is_gated_only_on_capture(self):
        """run_with_ortvaluevector must be gated on graph capture, not on provider identity.

        It is the only run API that used a provider-name check; run(), run_with_ort_values() and
        run_async() all gate on _validate_graph_capture_run_api. A raw vector is unsafe only while
        capture is armed, because replay re-issues the buffers recorded at capture.
        """
        session = onnxrt.InferenceSession(get_name("mul_1.onnx"), providers=["CPUExecutionProvider"])
        input_metadata = session.get_inputs()[0]
        output_metadata = session.get_outputs()[0]
        input_value = np.arange(np.prod(input_metadata.shape), dtype=np.float32).reshape(input_metadata.shape)

        feeds = OrtValueVector()
        feeds.push_back(onnxrt.OrtValue.ortvalue_from_numpy(input_value)._get_c_value())
        fetches = OrtValueVector()
        session.run_with_ortvaluevector(
            onnxrt.RunOptions(),
            [input_metadata.name],
            feeds,
            [output_metadata.name],
            fetches,
            [onnxrt.OrtDevice.make("cpu", 0)._get_c_device()],
        )
        np.testing.assert_allclose(
            fetches[0].numpy(),
            session.run(None, {input_metadata.name: input_value})[0],
        )

    @unittest.skipIf(
        "WebGpuExecutionProvider" not in onnxrt.get_available_providers(),
        "WebGpuExecutionProvider is not available",
    )
    def test_webgpu_run_with_ortvaluevector_without_capture(self):
        """A WebGPU session that is not capturing must accept raw OrtValue vectors.

        The previous guard rejected every WebGPU session regardless of capture state or of whether
        the vectors were even device-backed, which broke a pre-existing CPU-only use.
        """
        so = onnxrt.SessionOptions()
        so.enable_mem_pattern = False

        try:
            session = onnxrt.InferenceSession(
                get_name("mul_1.onnx"),
                sess_options=so,
                providers=["WebGpuExecutionProvider"],
            )
        except RuntimeError as error:
            if "Failed to get a WebGPU" in str(error):
                self.skipTest(str(error))
            raise

        self.assertIn("WebGpuExecutionProvider", session.get_providers())
        self.assertFalse(session._sess.is_webgpu_graph_capture_enabled())

        input_metadata = session.get_inputs()[0]
        output_metadata = session.get_outputs()[0]
        input_value = np.arange(np.prod(input_metadata.shape), dtype=np.float32).reshape(input_metadata.shape)
        reference_session = onnxrt.InferenceSession(get_name("mul_1.onnx"), providers=["CPUExecutionProvider"])
        expected = reference_session.run(None, {input_metadata.name: input_value})[0]

        feeds = OrtValueVector()
        feeds.push_back(onnxrt.OrtValue.ortvalue_from_numpy(input_value)._get_c_value())
        fetches = OrtValueVector()
        session.run_with_ortvaluevector(
            onnxrt.RunOptions(),
            [input_metadata.name],
            feeds,
            [output_metadata.name],
            fetches,
            [onnxrt.OrtDevice.make("cpu", 0)._get_c_device()],
        )
        np.testing.assert_allclose(fetches[0].numpy(), expected, rtol=1e-5, atol=1e-5)

    @unittest.skipIf(
        "WebGpuExecutionProvider" not in onnxrt.get_available_providers(),
        "WebGpuExecutionProvider is not available",
    )
    def test_webgpu_graph_capture_session_ortvalues(self):
        so = onnxrt.SessionOptions()
        so.enable_mem_pattern = False
        so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        so.add_session_config_entry("ep.webgpuexecutionprovider.enableGraphCapture", "1")

        try:
            session = onnxrt.InferenceSession(
                get_name("mul_1.onnx"),
                sess_options=so,
                providers=["WebGpuExecutionProvider"],
            )
        except RuntimeError as error:
            if "Failed to get a WebGPU" in str(error):
                self.skipTest(str(error))
            raise

        input_metadata = session.get_inputs()[0]
        output_metadata = session.get_outputs()[0]
        input_value = np.arange(np.prod(input_metadata.shape), dtype=np.float32).reshape(input_metadata.shape)
        reference_session = onnxrt.InferenceSession(get_name("mul_1.onnx"), providers=["CPUExecutionProvider"])

        # Step 1: allocate fixed device tensors from the session allocator.
        gpu_input = session.create_ortvalue_from_shape_and_type(input_metadata.shape, np.float32, "webgpu")
        gpu_output = session.create_ortvalue_from_shape_and_type(output_metadata.shape, np.float32, "webgpu")
        # Step 2: upload the host input with the environment-level copy_tensors.
        onnxrt.copy_tensors([onnxrt.OrtValue.ortvalue_from_numpy(input_value)], [gpu_input])
        self.assertTrue(gpu_input._is_webgpu_buffer)
        self.assertEqual(gpu_input.device_name(), "webgpu")
        gpu_alias_input = device_ortvalue_from_numpy(
            session,
            input_value,
            "gpu",
            vendor_id=onnxrt.OrtDeviceVendorId.NONE,
        )
        self.assertTrue(gpu_alias_input._is_webgpu_buffer)
        # data_ptr() returns the opaque WGPUBuffer handle, which is what interop callers want.
        self.assertNotEqual(gpu_input.data_ptr(), 0)
        # numpy() reads back through the environment-registered WebGPU data transfer.
        np.testing.assert_allclose(gpu_input.numpy(), input_value)
        with self.assertRaisesRegex(RuntimeError, "DLPack export"):
            gpu_input.__dlpack__()
        with self.assertRaisesRegex(RuntimeError, "DLPack export"):
            gpu_input.__dlpack_device__()
        with self.assertRaisesRegex(ValueError, "graph capture requires"):
            session.run(None, {input_metadata.name: gpu_input})
        with self.assertRaisesRegex(ValueError, "graph capture requires"):
            session.run_with_ort_values(None, {input_metadata.name: gpu_input})
        with self.assertRaisesRegex(ValueError, "graph capture requires"):
            session.run_async(
                None,
                {input_metadata.name: input_value},
                lambda *_: None,
                None,
            )
        with self.assertRaisesRegex(ValueError, "graph capture requires"):
            session.run_with_ortvaluevector(None, [], None, [], None, None)
        # Default-context WebGPU values copy in both directions through the shared data transfer.
        onnxrt.copy_tensors([gpu_input], [gpu_output])
        np.testing.assert_allclose(gpu_output.numpy(), input_value)

        raw_vector = OrtValueVector()
        # A standalone vector has no parent to keep the session alive; the IOBinding one does.
        raw_vector.push_back(onnxrt.OrtValue.ortvalue_from_numpy(input_value)._get_c_value())
        with self.assertRaisesRegex(RuntimeError, "standalone OrtValueVector"):
            raw_vector.push_back(gpu_input._get_c_value())

        unsafe_io_binding = session.io_binding()
        global_cpu_input = onnxrt.OrtValue.ortvalue_from_numpy(input_value)
        global_cpu_output = onnxrt.OrtValue.ortvalue_from_numpy(np.zeros(output_metadata.shape, dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "WebGPU source and destination"):
            gpu_input.update_inplace(global_cpu_input)
        with self.assertRaisesRegex(ValueError, "WebGPU source and destination"):
            global_cpu_input.update_inplace(gpu_input)
        # Shared-allocator WebGPU values have no session owner but remain bindable.
        shared_provenance_source = device_ortvalue_from_numpy(session, input_value, "webgpu")
        unowned_webgpu_input = onnxrt.OrtValue(shared_provenance_source._get_c_value())
        self.assertIsNone(unowned_webgpu_input._session)
        self.assertTrue(unowned_webgpu_input._is_webgpu_buffer)
        scratch_webgpu_value = session.create_ortvalue_from_shape_and_type(input_metadata.shape, np.float32, "webgpu")
        onnxrt.copy_tensors([unowned_webgpu_input], [scratch_webgpu_value])
        onnxrt.copy_tensors([scratch_webgpu_value], [unowned_webgpu_input])
        unsafe_io_binding.bind_ortvalue_input(input_metadata.name, unowned_webgpu_input)
        unsafe_io_binding.clear_binding_inputs()

        # Host upload into a shared-allocator device value goes through copy_tensors.
        onnxrt.copy_tensors([onnxrt.OrtValue.ortvalue_from_numpy(input_value)], [unowned_webgpu_input])
        np.testing.assert_allclose(unowned_webgpu_input.numpy(), input_value)

        # WebGPU -> CPU readback through the shared data transfer is supported for the default context.
        onnxrt.copy_tensors([unowned_webgpu_input], [global_cpu_input])
        np.testing.assert_allclose(global_cpu_input.numpy(), input_value)
        del unowned_webgpu_input
        del shared_provenance_source
        del scratch_webgpu_value
        # Run-time validation keeps gpu_graph_id=-1 usable with ordinary bindings.
        unsafe_io_binding.bind_cpu_input(input_metadata.name, input_value)
        unsafe_io_binding.bind_output(output_metadata.name)
        with self.assertRaisesRegex(ValueError, "requires fixed WebGPU device OrtValues"):
            session.run_with_iobinding(unsafe_io_binding)
        skip_capture_options = onnxrt.RunOptions()
        skip_capture_options.add_run_config_entry("gpu_graph_id", "-1")
        session.run_with_iobinding(unsafe_io_binding, skip_capture_options)
        np.testing.assert_allclose(
            unsafe_io_binding.copy_outputs_to_cpu()[0],
            reference_session.run(None, {input_metadata.name: input_value})[0],
            rtol=1e-5,
            atol=1e-5,
        )
        unsafe_io_binding.clear_binding_inputs()
        unsafe_io_binding.clear_binding_outputs()
        unsafe_io_binding.bind_ortvalue_input(input_metadata.name, global_cpu_input)
        unsafe_io_binding.bind_ortvalue_output(output_metadata.name, global_cpu_output)
        with self.assertRaisesRegex(ValueError, "requires fixed WebGPU device OrtValues"):
            session.run_with_iobinding(unsafe_io_binding)
        unsafe_io_binding.clear_binding_inputs()
        unsafe_io_binding.clear_binding_outputs()

        # gpu_graph_id=-1 keeps convenience APIs on the uncaptured path.
        with self.assertRaisesRegex(ValueError, "requires fixed device OrtValues"):
            session.run(None, {input_metadata.name: input_value})
        np.testing.assert_allclose(
            session.run(None, {input_metadata.name: input_value}, skip_capture_options)[0],
            reference_session.run(None, {input_metadata.name: input_value})[0],
            rtol=1e-5,
            atol=1e-5,
        )

        io_binding = session.io_binding()
        io_binding.bind_ortvalue_input(input_metadata.name, gpu_input)
        io_binding.bind_ortvalue_output(output_metadata.name, gpu_output)

        def run_and_copy_output(current_session, current_io_binding, run_options=None):
            current_session.run_with_iobinding(current_io_binding, run_options)
            return current_io_binding.copy_outputs_to_cpu()[0]

        expected = reference_session.run(None, {input_metadata.name: input_value})[0]
        np.testing.assert_allclose(run_and_copy_output(session, io_binding), expected)
        raw_outputs = io_binding._iobinding.get_outputs()
        with self.assertRaisesRegex(RuntimeError, "DLPack export"):
            raw_outputs.dlpack_at(0)
        with self.assertRaisesRegex(RuntimeError, "DLPack export"):
            raw_outputs.to_dlpacks(None)
        indexed_raw_output = raw_outputs[0]
        del raw_outputs
        bound_output = io_binding.get_outputs()[0]
        self.assertIs(bound_output._session, session._sess)
        np.testing.assert_allclose(bound_output.numpy(), expected)
        vector_outputs = io_binding.get_outputs_as_ortvaluevector()
        self.assertTrue(vector_outputs[0]._is_webgpu_buffer())
        del vector_outputs

        updated_input = input_value + 10.0
        onnxrt.copy_tensors([onnxrt.OrtValue.ortvalue_from_numpy(updated_input)], [gpu_input])
        updated_expected = reference_session.run(None, {input_metadata.name: updated_input})[0]
        alternate_gpu_output = device_ortvalue_from_numpy(
            session, np.zeros(output_metadata.shape, dtype=np.float32), "webgpu"
        )
        alternate_io_binding = session.io_binding()
        alternate_io_binding.bind_ortvalue_input(input_metadata.name, gpu_input)
        alternate_io_binding.bind_ortvalue_output(output_metadata.name, alternate_gpu_output)

        # Replay rejects a different IOBinding because it targets the captured buffers.
        with self.assertRaisesRegex(ValueError, "captured with a different IOBinding"):
            session.run_with_iobinding(alternate_io_binding)

        # Captured bindings remain immutable until release.
        with self.assertRaisesRegex(ValueError, "still reference this IOBinding"):
            io_binding.bind_ortvalue_output(output_metadata.name, alternate_gpu_output)
        with self.assertRaisesRegex(ValueError, "still reference this IOBinding"):
            io_binding.clear_binding_inputs()
        with self.assertRaisesRegex(ValueError, "still reference this IOBinding"):
            io_binding.clear_binding_outputs()

        np.testing.assert_allclose(run_and_copy_output(session, io_binding), updated_expected)
        np.testing.assert_array_equal(
            alternate_io_binding.copy_outputs_to_cpu()[0],
            np.zeros(output_metadata.shape, dtype=np.float32),
        )

        ortvalue_update_input = input_value + 20.0
        onnxrt.copy_tensors([device_ortvalue_from_numpy(session, ortvalue_update_input, "webgpu")], [gpu_input])
        ortvalue_update_expected = reference_session.run(None, {input_metadata.name: ortvalue_update_input})[0]
        np.testing.assert_allclose(run_and_copy_output(session, io_binding), ortvalue_update_expected)

        session.release_captured_graph()

        graph_one_run_options = onnxrt.RunOptions()
        graph_one_run_options.add_run_config_entry("gpu_graph_id", "1")
        np.testing.assert_allclose(
            run_and_copy_output(session, io_binding, graph_one_run_options),
            ortvalue_update_expected,
        )
        repeated_capture_input = input_value + 30.0
        onnxrt.copy_tensors([onnxrt.OrtValue.ortvalue_from_numpy(repeated_capture_input)], [gpu_input])
        repeated_capture_expected = reference_session.run(None, {input_metadata.name: repeated_capture_input})[0]
        np.testing.assert_allclose(
            run_and_copy_output(session, io_binding, graph_one_run_options),
            repeated_capture_expected,
        )
        session.release_captured_graph(1)

        # Skipped runs use current inputs instead of replaying captured commands.
        no_capture_run_options = onnxrt.RunOptions()
        no_capture_run_options.add_run_config_entry("gpu_graph_id", "-1")
        for offset in (40.0, 50.0):
            uncaptured_input = input_value + offset
            onnxrt.copy_tensors([onnxrt.OrtValue.ortvalue_from_numpy(uncaptured_input)], [gpu_input])
            np.testing.assert_allclose(
                run_and_copy_output(session, io_binding, no_capture_run_options),
                reference_session.run(None, {input_metadata.name: uncaptured_input})[0],
            )

        io_binding.clear_binding_inputs()
        io_binding.clear_binding_outputs()
        alternate_io_binding.clear_binding_inputs()
        alternate_io_binding.clear_binding_outputs()
        self.assertEqual(bound_output.shape(), output_metadata.shape)
        del io_binding
        del alternate_io_binding
        del session
        gc.collect()
        self.assertTrue(indexed_raw_output._is_webgpu_buffer())
        del indexed_raw_output
        gc.collect()

    @unittest.skipIf(
        "WebGpuExecutionProvider" not in onnxrt.get_available_providers(),
        "WebGpuExecutionProvider is not available",
    )
    def test_webgpu_graph_capture_across_set_providers(self):
        """A real capture must not outlive the session handle that set_providers() replaces."""
        so = onnxrt.SessionOptions()
        so.enable_mem_pattern = False
        so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        so.add_session_config_entry("ep.webgpuexecutionprovider.enableGraphCapture", "1")

        try:
            session = onnxrt.InferenceSession(
                get_name("mul_1.onnx"),
                sess_options=so,
                providers=["WebGpuExecutionProvider"],
            )
        except RuntimeError as error:
            if "Failed to get a WebGPU" in str(error):
                self.skipTest(str(error))
            raise

        input_metadata = session.get_inputs()[0]
        output_metadata = session.get_outputs()[0]
        input_value = np.arange(np.prod(input_metadata.shape), dtype=np.float32).reshape(input_metadata.shape)
        reference_session = onnxrt.InferenceSession(get_name("mul_1.onnx"), providers=["CPUExecutionProvider"])
        expected = reference_session.run(None, {input_metadata.name: input_value})[0]

        def capture(current_session):
            io_binding = current_session.io_binding()
            io_binding.bind_ortvalue_input(
                input_metadata.name,
                device_ortvalue_from_numpy(current_session, input_value, "webgpu"),
            )
            io_binding.bind_ortvalue_output(
                output_metadata.name,
                current_session.create_ortvalue_from_shape_and_type(output_metadata.shape, np.float32, "webgpu"),
            )
            current_session.run_with_iobinding(io_binding)
            return io_binding

        first_binding = capture(session)
        np.testing.assert_allclose(first_binding.copy_outputs_to_cpu()[0], expected, rtol=1e-5, atol=1e-5)
        self.assertEqual(set(session._captured_graph_bindings), {0})
        self.assertEqual(first_binding._pinned_graph_ids, {0})

        session.set_providers(["WebGpuExecutionProvider"])

        # The replaced handle's capture must be gone from both sides of the bookkeeping.
        self.assertEqual(session._captured_graph_bindings, {})
        self.assertEqual(first_binding._pinned_graph_ids, set())
        first_binding.clear_binding_inputs()
        first_binding.clear_binding_outputs()

        # Graph 0 must be capturable again through the replacement session.
        second_binding = capture(session)
        np.testing.assert_allclose(second_binding.copy_outputs_to_cpu()[0], expected, rtol=1e-5, atol=1e-5)
        session.release_captured_graph()
        second_binding.clear_binding_inputs()
        second_binding.clear_binding_outputs()

    @unittest.skipIf(
        "WebGpuExecutionProvider" not in onnxrt.get_available_providers(),
        "WebGpuExecutionProvider is not available",
    )
    def test_webgpu_raw_output_vector_keeps_session_alive(self):
        """A device OrtValue reached through the raw vector must survive its session.

        This pins the pybind keepalive chain that makes get_outputs_as_ortvaluevector() safe:

            OrtValue --keep_alive<0,1> on OrtValueVector.__getitem__-->
            OrtValueVector --reference_internal on SessionIOBinding.get_outputs-->
            SessionIOBinding --keep_alive<1,2> on SessionIOBinding.__init__--> InferenceSession

        Without the full chain, freeing the buffer after the session is gone would call
        GpuBufferAllocator::Free() through a buffer-manager getter that captured the dead EP.
        """
        so = onnxrt.SessionOptions()
        so.enable_mem_pattern = False

        try:
            session = onnxrt.InferenceSession(
                get_name("mul_1.onnx"),
                sess_options=so,
                providers=["WebGpuExecutionProvider"],
            )
        except RuntimeError as error:
            if "Failed to get a WebGPU" in str(error):
                self.skipTest(str(error))
            raise

        input_metadata = session.get_inputs()[0]
        output_metadata = session.get_outputs()[0]
        input_value = np.arange(np.prod(input_metadata.shape), dtype=np.float32).reshape(input_metadata.shape)

        io_binding = session.io_binding()
        io_binding.bind_cpu_input(input_metadata.name, input_value)
        io_binding.bind_output(output_metadata.name, "webgpu")
        session.run_with_iobinding(io_binding)

        held = io_binding.get_outputs_as_ortvaluevector()[0]
        self.assertTrue(held._is_webgpu_buffer())

        del io_binding
        del session
        gc.collect()

        # The value is still valid with no session in scope ...
        self.assertTrue(held._is_webgpu_buffer())
        # ... and releasing the WebGPU buffer afterwards must not fault.
        del held
        gc.collect()

    def test_memory_arena_shrinkage(self):
        if (
            platform.architecture()[0] == "32bit"
            or "ppc" in platform.machine()
            or "powerpc" in platform.machine()
            or "powerpc" in platform.processor()
        ):
            # on x86 or ppc builds, the CPU allocator does not use an arena
            print("Skipping testMemoryArenaShrinkage in 32bit or powerpc platform.")
        else:
            x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)

            sess1 = onnxrt.InferenceSession(get_name("mul_1.onnx"), providers=["CPUExecutionProvider"])
            input_name = sess1.get_inputs()[0].name

            # Shrink CPU memory after execution
            ro1 = onnxrt.RunOptions()
            ro1.add_run_config_entry("memory.enable_memory_arena_shrinkage", "cpu:0")
            self.assertEqual(
                ro1.get_run_config_entry("memory.enable_memory_arena_shrinkage"),
                "cpu:0",
            )
            sess1.run([], {input_name: x}, ro1)

            available_providers = onnxrt.get_available_providers()
            if "CUDAExecutionProvider" in available_providers:
                sess2 = onnxrt.InferenceSession(get_name("mul_1.onnx"), providers=available_providers)
                input_name = sess2.get_inputs()[0].name

                # Shrink CPU and GPU memory after execution
                ro2 = onnxrt.RunOptions()
                ro2.add_run_config_entry("memory.enable_memory_arena_shrinkage", "cpu:0;gpu:0")
                self.assertEqual(
                    ro2.get_run_config_entry("memory.enable_memory_arena_shrinkage"),
                    "cpu:0;gpu:0",
                )
                sess2.run([], {input_name: x}, ro2)

    def test_check_and_normalize_provider_args(self):
        from onnxruntime.capi.onnxruntime_inference_collection import check_and_normalize_provider_args  # noqa: PLC0415

        valid_providers = ["a", "b", "c"]

        def check_success(providers, provider_options, expected_providers, expected_provider_options):
            (
                actual_providers,
                actual_provider_options,
            ) = check_and_normalize_provider_args(providers, provider_options, valid_providers)
            self.assertEqual(actual_providers, expected_providers)
            self.assertEqual(actual_provider_options, expected_provider_options)

        check_success(None, None, [], [])

        check_success(["a"], None, ["a"], [{}])

        check_success(["a", "b"], None, ["a", "b"], [{}, {}])

        check_success([("a", {1: 2}), "b"], None, ["a", "b"], [{"1": "2"}, {}])

        check_success(["a", "b"], [{1: 2}, {}], ["a", "b"], [{"1": "2"}, {}])

        with self.assertWarns(UserWarning):
            check_success(["a", "b", "a"], [{"x": 1}, {}, {"y": 2}], ["a", "b"], [{"x": "1"}, {}])

        def check_failure(providers, provider_options):
            with self.assertRaises(ValueError):
                check_and_normalize_provider_args(providers, provider_options, valid_providers)

        # disable this test
        # provider not valid
        # check_failure(["d"], None)

        # providers not sequence
        check_failure(3, None)

        # providers value invalid
        check_failure([3], None)

        # provider_options not sequence
        check_failure(["a"], 3)

        # provider_options value invalid
        check_failure(["a"], ["not dict"])

        # providers and provider_options length mismatch
        check_failure(["a", "b"], [{1: 2}])

        # provider options unsupported mixed specification
        check_failure([("a", {1: 2})], [{3: 4}])

    def test_create_allocator(self):
        def verify_allocator(allocator, expected_config):
            for key, val in expected_config.items():
                if key == "max_mem":
                    self.assertEqual(allocator.max_mem, val)
                elif key == "arena_extend_strategy":
                    self.assertEqual(allocator.arena_extend_strategy, val)
                elif key == "initial_chunk_size_bytes":
                    self.assertEqual(allocator.initial_chunk_size_bytes, val)
                elif key == "max_dead_bytes_per_chunk":
                    self.assertEqual(allocator.max_dead_bytes_per_chunk, val)
                elif key == "initial_growth_chunk_size_bytes":
                    self.assertEqual(allocator.initial_growth_chunk_size_bytes, val)
                elif key == "max_power_of_two_extend_bytes":
                    self.assertEqual(allocator.max_power_of_two_extend_bytes, val)
                else:
                    raise ValueError("Invalid OrtArenaCfg option: " + key)

        # Verify ordered parameter initialization
        ort_arena_cfg = onnxrt.OrtArenaCfg(8, 0, 4, 2)
        expected_allocator = {
            "max_mem": 8,
            "arena_extend_strategy": 0,
            "initial_chunk_size_bytes": 4,
            "max_dead_bytes_per_chunk": 2,
        }
        verify_allocator(ort_arena_cfg, expected_allocator)

        # Verify key-value pair initialization
        expected_kvp_allocator = {
            "max_mem": 16,
            "arena_extend_strategy": 1,
            "initial_chunk_size_bytes": 8,
            "max_dead_bytes_per_chunk": 4,
            "initial_growth_chunk_size_bytes": 2,
        }
        ort_arena_cfg_kvp = onnxrt.OrtArenaCfg(expected_kvp_allocator)
        verify_allocator(ort_arena_cfg_kvp, expected_kvp_allocator)

        # Verify key-value pair initialization
        expected_kvp_allocator = {
            "max_mem": 32,
            "arena_extend_strategy": 11,
            "initial_chunk_size_bytes": 18,
            "max_dead_bytes_per_chunk": 14,
            "initial_growth_chunk_size_bytes": 12,
            "max_power_of_two_extend_bytes": 17,
        }
        ort_arena_cfg_kvp = onnxrt.OrtArenaCfg(expected_kvp_allocator)
        verify_allocator(ort_arena_cfg_kvp, expected_kvp_allocator)

    def test_multiple_devices(self):
        if "CUDAExecutionProvider" in onnxrt.get_available_providers():
            cuda_lib = self.load_cuda_lib()
            cuda_devices = self.cuda_device_count(cuda_lib)
            if cuda_devices <= 1:
                return

            # https://github.com/microsoft/onnxruntime/issues/18432. Make sure device Id is properly set
            # Scenario 1, 3 sessions created with different device Id under IOBinding
            sessions = []
            for i in range(3):
                sessions.append(
                    onnxrt.InferenceSession(
                        get_name("mnist.onnx"), providers=[("CUDAExecutionProvider", {"device_id": i % 2})]
                    )
                )

            for i in range(3):
                binding = sessions[i].io_binding()
                image = np.ones([1, 1, 28, 28], np.float32)
                image_on_gpu = onnxrt.OrtValue.ortvalue_from_numpy(image, "cuda", i % 2)

                binding.bind_ortvalue_input("Input3", image_on_gpu)
                binding.bind_output(name="Plus214_Output_0", device_type="cuda", device_id=i % 2)

                binding.synchronize_inputs()
                sessions[i].run_with_iobinding(binding)
                binding.synchronize_outputs()

            # Scenario 2, 2 normal sessions created with different device Id
            device0_session = onnxrt.InferenceSession(
                get_name("mnist.onnx"), providers=[("CUDAExecutionProvider", {"device_id": 0})]
            )
            device1_session = onnxrt.InferenceSession(
                get_name("mnist.onnx"), providers=[("CUDAExecutionProvider", {"device_id": 1})]
            )
            image = {
                "Input3": np.ones([1, 1, 28, 28], np.float32),
            }
            device0_session.run(output_names=["Plus214_Output_0"], input_feed=image)
            device1_session.run(output_names=["Plus214_Output_0"], input_feed=image)
            device0_session.run(output_names=["Plus214_Output_0"], input_feed=image)

    def test_adater_export_read(self):
        adapter_version = 1
        model_version = 1
        file_path = pathlib.Path(os.path.realpath(__file__)).parent
        file_path = str(file_path / "test_adapter.onnx_adapter")

        float_data_type = 1
        int64_data_type = 7
        val = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        param_1 = np.array(val).astype(np.float32).reshape(5, 2)
        param_2 = np.array(val).astype(np.int64).reshape(2, 5)

        ort_val_1 = onnxrt.OrtValue.ortvalue_from_numpy_with_onnx_type(param_1, float_data_type)
        ort_val_2 = onnxrt.OrtValue.ortvalue_from_numpy_with_onnx_type(param_2, int64_data_type)

        params = {"param_1": ort_val_1, "param_2": ort_val_2}

        adapter_format = onnxrt.AdapterFormat()
        adapter_format.set_adapter_version(adapter_version)
        adapter_format.set_model_version(model_version)
        adapter_format.set_parameters(params)

        adapter_format.export_adapter(file_path)

        adapter_format_read = onnxrt.AdapterFormat.read_adapter(file_path)
        os.remove(file_path)

        self.assertEqual(adapter_version, adapter_format_read.get_adapter_version())
        self.assertEqual(model_version, adapter_format_read.get_model_version())

        actual_params = adapter_format_read.get_parameters()
        self.assertCountEqual(params, actual_params)
        for key, value in actual_params.items():
            self.assertIn(key, params)
            expected_val = params.get(key)
            self.assertTrue(value.is_tensor())
            self.assertEqual(expected_val.element_type(), value.element_type())
            self.assertEqual(expected_val.shape(), value.shape())
            np.testing.assert_allclose(value.numpy(), expected_val.numpy())

    def test_adapter_read_modify_export(self):
        # Verify that an instance created by read_adapter can have its
        # parameters replaced and re-exported (single-source architecture).
        adapter_version = 1
        model_version = 1
        file_path = pathlib.Path(os.path.realpath(__file__)).parent
        original_path = str(file_path / "test_adapter_read_modify_original.onnx_adapter")
        modified_path = str(file_path / "test_adapter_read_modify_new.onnx_adapter")

        float_data_type = 1
        original_data = np.array([1, 2, 3, 4]).astype(np.float32).reshape(2, 2)
        modified_data = np.array([10, 20, 30, 40, 50, 60]).astype(np.float32).reshape(3, 2)

        ort_original = onnxrt.OrtValue.ortvalue_from_numpy_with_onnx_type(original_data, float_data_type)
        ort_modified = onnxrt.OrtValue.ortvalue_from_numpy_with_onnx_type(modified_data, float_data_type)

        # Write original adapter
        adapter_format = onnxrt.AdapterFormat()
        adapter_format.set_adapter_version(adapter_version)
        adapter_format.set_model_version(model_version)
        adapter_format.set_parameters({"param": ort_original})
        adapter_format.export_adapter(original_path)

        try:
            # Read, replace parameters, and re-export.
            adapter_read = onnxrt.AdapterFormat.read_adapter(original_path)
            adapter_read.set_parameters({"new_param": ort_modified})
            adapter_read.export_adapter(modified_path)

            # Verify the re-exported file has the NEW parameters.
            adapter_verify = onnxrt.AdapterFormat.read_adapter(modified_path)
            params = adapter_verify.get_parameters()
            self.assertIn("new_param", params)
            self.assertNotIn("param", params)
            self.assertEqual(params["new_param"].shape(), [3, 2])
            np.testing.assert_allclose(params["new_param"].numpy(), modified_data)
        finally:
            if os.path.exists(original_path):
                os.remove(original_path)
            if os.path.exists(modified_path):
                os.remove(modified_path)

    def test_adapter_parameters_keep_alive(self):
        # Regression test: AdapterFormat.read_adapter returned OrtValue views
        # over storage owned by the C AdapterFormat with nothing keeping the
        # parent alive. The natural pattern below dropped the parent and left
        # the dict with dangling pointers, causing a use-after-free on the
        # next access. read_adapter now pins the owning C AdapterFormat on
        # every OrtValue it produces (pybind11 add_patient via
        # keep_alive_impl), so the dict and any individual value keep the
        # backing adapter alive on their own. Mirrors the strong-ref pattern
        # used by the SparseTensor view bindings.
        adapter_version = 1
        model_version = 1
        file_path = pathlib.Path(os.path.realpath(__file__)).parent
        file_path = str(file_path / "test_adapter_keep_alive.onnx_adapter")

        float_data_type = 1
        val = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        param_1 = np.array(val).astype(np.float32).reshape(5, 2)

        ort_val_1 = onnxrt.OrtValue.ortvalue_from_numpy_with_onnx_type(param_1, float_data_type)

        adapter_format = onnxrt.AdapterFormat()
        adapter_format.set_adapter_version(adapter_version)
        adapter_format.set_model_version(model_version)
        adapter_format.set_parameters({"param_1": ort_val_1})
        adapter_format.export_adapter(file_path)

        try:
            # Drop the AdapterFormat temporary; only `params` keeps a reference.
            params = onnxrt.AdapterFormat.read_adapter(file_path).get_parameters()
            gc.collect()

            self.assertIn("param_1", params)
            value = params["param_1"]
            self.assertTrue(value.is_tensor())
            self.assertEqual(value.shape(), [5, 2])
            np.testing.assert_allclose(value.numpy(), param_1)

            # Also drop the dict; an individual OrtValue must keep the adapter
            # alive on its own (we pin it on every value in get_parameters()).
            single_value = params["param_1"]
            del params
            gc.collect()
            np.testing.assert_allclose(single_value.numpy(), param_1)
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

    def test_adapter_export_rejects_string_tensors(self):
        # Regression test: export_adapter previously serialized Tensor::DataRaw()
        # for SizeInBytes() bytes regardless of element type. For string tensors
        # that copied the std::string object representation (heap pointers and
        # uninitialized padding) into the adapter file, leaking runtime addresses
        # (ASLR bypass) and producing an unloadable adapter. Export must reject
        # string-typed parameters.
        #
        # There is no public Python API to construct a string-typed OrtValue
        # directly (ortvalue_from_numpy rejects non-numeric arrays), so we
        # obtain one by running a tiny Constant model whose only output is a
        # string tensor.
        from onnx import TensorProto, helper  # noqa: PLC0415

        const_node = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["str_out"],
            value=helper.make_tensor("v", TensorProto.STRING, dims=[2], vals=[b"hello", b"world"]),
        )
        graph = helper.make_graph(
            [const_node],
            "string_const",
            inputs=[],
            outputs=[helper.make_tensor_value_info("str_out", TensorProto.STRING, [2])],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        sess = onnxrt.InferenceSession(model.SerializeToString(), providers=onnxrt.get_available_providers())
        ort_val_str = sess.run_with_ort_values(["str_out"], {})[0]

        adapter_format = onnxrt.AdapterFormat()
        adapter_format.set_adapter_version(1)
        adapter_format.set_model_version(1)
        adapter_format.set_parameters({"str_param": ort_val_str})

        file_path = pathlib.Path(os.path.realpath(__file__)).parent
        file_path = str(file_path / "test_adapter_string_reject.onnx_adapter")

        try:
            with self.assertRaises(Exception) as ctx:
                adapter_format.export_adapter(file_path)
            self.assertIn("STRING", str(ctx.exception))
            self.assertFalse(os.path.exists(file_path), "adapter file must not be created when export is rejected")
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

    def test_run_with_adapter(self):
        model_path = get_name("lora/two_params_lora_model.onnx")
        file_path = os.getcwd() + "/" + get_name("lora/two_params_lora_model.onnx_adapter")
        adapter_path = os.path.abspath(file_path)

        expected_output = np.array(
            [
                [154.0, 176.0, 198.0, 220.0],
                [154.0, 176.0, 198.0, 220.0],
                [154.0, 176.0, 198.0, 220.0],
                [154.0, 176.0, 198.0, 220.0],
            ],
            dtype=np.float32,
        )

        adapter = onnxrt.LoraAdapter()
        adapter.Load(adapter_path)

        run_options = onnxrt.RunOptions()
        run_options.add_active_adapter(adapter)
        session = onnxrt.InferenceSession(model_path)

        inputs = {"input_x": np.ones((4, 4), dtype=np.float32)}

        outputs = session.run(None, inputs, run_options)
        self.assertEqual(len(outputs), 1)
        self.assertTrue(np.allclose(outputs[0], expected_output))

    def test_run_base_model(self):
        model_path = get_name("lora/two_params_lora_model.onnx")

        expected_output = np.array(
            [[28.0, 32.0, 36.0, 40.0], [28.0, 32.0, 36.0, 40.0], [28.0, 32.0, 36.0, 40.0], [28.0, 32.0, 36.0, 40.0]],
            dtype=np.float32,
        )

        run_options = onnxrt.RunOptions()
        session = onnxrt.InferenceSession(model_path)

        inputs = {"input_x": np.ones((4, 4), dtype=np.float32)}

        outputs = session.run(None, inputs, run_options)
        self.assertEqual(len(outputs), 1)
        self.assertTrue(np.allclose(outputs[0], expected_output))

    def test_get_graph_provider_assignment_info(self):
        """
        Tests querying for information about the nodes assigned to the CPU EP.
        """

        # Create session options that enables recording EP graph partitioning info.
        session_options = onnxrt.SessionOptions()
        session_options.add_session_config_entry("session.record_ep_graph_assignment_info", "1")

        session = onnxrt.InferenceSession(get_name("add_mul_add.onnx"), sess_options=session_options)

        # Query session for information on each subgraph assigned to an EP.
        ep_subgraphs = session.get_provider_graph_assignment_info()

        # Check that all 3 nodes are assigned to CPU EP (each in its own subgraph)
        self.assertEqual(len(ep_subgraphs), 3)
        for ep_subgraph in ep_subgraphs:
            self.assertEqual(ep_subgraph.ep_name, "CPUExecutionProvider")
            self.assertEqual(len(ep_subgraph.get_nodes()), 1)

        # Serialize each node to an identifier (concatenates domain, operator type, and node name)
        node_ids: list[str] = [f"{n.domain}:{n.op_type}/{n.name}" for s in ep_subgraphs for n in s.get_nodes()]

        # Should have 1 Mul and 2 Adds.
        self.assertEqual(len(node_ids), 3)
        self.assertIn(":Add/add_0", node_ids)
        self.assertIn(":Add/add_1", node_ids)
        self.assertIn(":Mul/mul_0", node_ids)

    def test_get_graph_provider_assignment_info_not_enabled(self):
        """
        Tests querying for information about the nodes assigned to the CPU EP when
        the corresponding config entry is disabled.
        """

        # Do not enable "session.record_ep_graph_assignment_info"
        session = onnxrt.InferenceSession(get_name("add_mul_add.onnx"))

        # Expect failure
        with self.assertRaises(Fail) as context:
            session.get_provider_graph_assignment_info()
        self.assertIn(
            "Session configuration entry 'session.record_ep_graph_assignment_info' must be set to \"1\"",
            str(context.exception),
        )

    def test_tree_ensemble_logistic(self):
        try:
            import onnx  # noqa: PLC0415
        except ImportError:
            # onnx is not installed on ARM build.
            self.skipTest("onnx is not installed")
        # issue https://github.com/microsoft/onnxruntime/issues/27533
        x = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [None, 3])
        label_out = onnx.helper.make_tensor_value_info("label", onnx.TensorProto.INT64, [None])
        prob_out = onnx.helper.make_tensor_value_info("probs", onnx.TensorProto.FLOAT, [None, 2])

        def make_model(
            nodes_modes,
            nodes_values,
            nodes_truenodeids,
            nodes_falsenodeids,
            class_treeids,
            class_nodeids,
            class_weights,
            **node_kwargs,
        ):
            """Build a minimal TreeEnsembleClassifier ONNX model."""
            n_nodes = len(nodes_modes)
            if "base_values" not in node_kwargs:
                node_kwargs["base_values"] = [-0.405]  # logit(0.4)
            node = onnx.helper.make_node(
                "TreeEnsembleClassifier",
                inputs=["X"],
                outputs=["label", "probs"],
                domain="ai.onnx.ml",
                nodes_treeids=[0] * n_nodes,
                nodes_nodeids=list(range(n_nodes)),
                nodes_featureids=[0] * n_nodes,
                nodes_values=nodes_values,
                nodes_modes=nodes_modes,
                nodes_truenodeids=nodes_truenodeids,
                nodes_falsenodeids=nodes_falsenodeids,
                nodes_missing_value_tracks_true=[0] * n_nodes,
                nodes_hitrates=[1.0] * n_nodes,
                class_treeids=class_treeids,
                class_nodeids=class_nodeids,
                class_ids=[0] * len(class_weights),
                class_weights=class_weights,
                classlabels_int64s=[0, 1],
                post_transform="LOGISTIC",
                **node_kwargs,
            )
            graph = onnx.helper.make_graph([node], "test", [x], [label_out, prob_out])
            return onnx.helper.make_model(
                graph,
                opset_imports=[
                    onnx.helper.make_opsetid("", 15),
                    onnx.helper.make_opsetid("ai.onnx.ml", 3),
                ],
            )

        test_input = {"X": np.array([[0.1, 0.0, 0.0]], dtype=np.float32)}

        # Case 1: Tree with a real split (root splits on feature 0 at 0.5)
        model_split = make_model(
            nodes_modes=["BRANCH_LT", "LEAF", "LEAF"],
            nodes_values=[0.5, 0.0, 0.0],
            nodes_truenodeids=[1, 0, 0],
            nodes_falsenodeids=[2, 0, 0],
            class_treeids=[0, 0],
            class_nodeids=[1, 2],
            class_weights=[0.3, -0.3],  # mixed positive/negative
        )
        sess_split = onnxrt.InferenceSession(model_split.SerializeToString())
        result_split = sess_split.run(None, test_input)
        # x[0]=0.1 < 0.5, so left leaf (weight=0.3), aggregate = -0.405 + 0.3 = -0.105
        expected_p1 = 1 / (1 + np.exp(0.105))  # sigmoid(-0.105)
        with self.subTest(case="Case 1: Tree with a real split"):
            np.testing.assert_allclose(result_split[1][0][1], expected_p1, atol=1e-5)

        # Case 2: Leaf-only tree (single LEAF node, no splits)
        model_leaf = make_model(
            nodes_modes=["LEAF"],
            nodes_values=[0.0],
            nodes_truenodeids=[0],
            nodes_falsenodeids=[0],
            class_treeids=[0],
            class_nodeids=[0],
            class_weights=[0.0],  # non-negative weight
        )
        sess_leaf = onnxrt.InferenceSession(model_leaf.SerializeToString())
        result_leaf = sess_leaf.run(None, test_input)
        # aggregate = -0.405 + 0 = -0.405
        expected_p1_leaf = 1 / (1 + np.exp(0.405))  # sigmoid(-0.405) ≈ 0.400
        with self.subTest(case="Case 2: Leaf-only tree (single LEAF node, no splits)"):
            np.testing.assert_allclose(result_leaf[1][0][1], expected_p1_leaf, atol=1e-5)

        # Case 3: Same leaf-only tree but with a negative weight (workaround)
        model_leaf_neg = make_model(
            nodes_modes=["LEAF"],
            nodes_values=[0.0],
            nodes_truenodeids=[0],
            nodes_falsenodeids=[0],
            class_treeids=[0],
            class_nodeids=[0],
            class_weights=[-0.405],  # negative weight (move base_values into weight)
            base_values=[0.0],  # zero base
        )
        sess_leaf_neg = onnxrt.InferenceSession(model_leaf_neg.SerializeToString())
        result_leaf_neg = sess_leaf_neg.run(None, test_input)
        with self.subTest(case="Case 3: Same leaf-only tree but with a negative weight"):
            np.testing.assert_allclose(result_leaf_neg[1][0][1], expected_p1_leaf, atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=1)
