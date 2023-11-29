# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from __future__ import annotations

import copy
import ctypes
import gc
import os
import pathlib
import platform
import queue
import sys
import threading
import unittest

import numpy as np
from helper import get_name

import onnxruntime as onnxrt
from onnxruntime.capi.onnxruntime_pybind11_state import Fail, OrtValueVector, RunOptions

# handle change from python 3.8 and on where loading a dll from the current directory needs to be explicitly allowed.
if platform.system() == "Windows" and sys.version_info.major >= 3 and sys.version_info.minor >= 8:  # noqa: YTT204
    os.add_dll_directory(os.getcwd())

available_providers = [provider for provider in onnxrt.get_available_providers()]

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
    provider for provider in onnxrt.get_available_providers() if provider not in {"TvmExecutionProvider"}
]

available_providers_without_tvm_and_tensorrt = [
    provider
    for provider in onnxrt.get_available_providers()
    if provider not in {"TvmExecutionProvider", "TensorrtExecutionProvider"}
]


class TestInferenceSession(unittest.TestCase):
    def run_model(self, session_object, run_options):
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        input_name = session_object.get_inputs()[0].name
        res = session_object.run([], {input_name: x}, run_options=run_options)
        output_expected = np.array([[1.0, 4.0], [9.0, 16.0], [25.0, 36.0]], dtype=np.float32)
        np.testing.assert_allclose(output_expected, res[0], rtol=1e-05, atol=1e-08)

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
            print("cuDeviceGetCount failed with error code %d: %s" % (result, error_str.value.decode()))
            return -1
        return num_device.value

    def test_tvm_imported(self):
        if "TvmExecutionProvider" not in onnxrt.get_available_providers():
            return
        import tvm

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
            so.add_session_config_entry("session.optimized_model_external_initializers_min_size_in_bytes", "100")
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

        # TODO(anyone): Set this to 100 will cause test error since some tensor below the threshold
        # still refers to the original external data file. We shall fix this issue so that the
        # optimized model only refers to one external data file.
        so.add_session_config_entry("session.optimized_model_external_initializers_min_size_in_bytes", "10")
        session1 = onnxrt.InferenceSession(get_name("model_with_orig_ext_data.onnx"), sess_options=so)
        del session1
        self.assertTrue(os.path.isfile(optimized_model_filepath))
        self.assertTrue(os.path.isfile(external_initializers_file))

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
        os.remove(external_initializers_file)

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

            from onnxruntime.capi import _pybind_state as C

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

                test_get_and_set_option_with_values("arena_extend_strategy", ["kNextPowerOfTwo", "kSameAsRequested"])

                test_get_and_set_option_with_values("cudnn_conv_algo_search", ["DEFAULT", "EXHAUSTIVE", "HEURISTIC"])

                test_get_and_set_option_with_values("do_copy_in_default_stream", [0, 1])

                test_get_and_set_option_with_values("tunable_op_enable", ["1", "0"])

                test_get_and_set_option_with_values("tunable_op_tuning_enable", ["1", "0"])

                test_get_and_set_option_with_values("tunable_op_max_tuning_duration_ms", ["-1", "1"])

                option["gpu_external_alloc"] = "0"
                option["gpu_external_free"] = "0"
                option["gpu_external_empty_cache"] = "0"
                sess.set_providers(["CUDAExecutionProvider"], [option])
                options = sess.get_provider_options()
                self.assertEqual(options["CUDAExecutionProvider"]["gpu_external_alloc"], "0")
                self.assertEqual(options["CUDAExecutionProvider"]["gpu_external_free"], "0")
                self.assertEqual(options["CUDAExecutionProvider"]["gpu_external_empty_cache"], "0")
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

        if "ROCMExecutionProvider" in onnxrt.get_available_providers():

            def run_rocm_options_test():
                sess = onnxrt.InferenceSession(get_name("mul_1.onnx"), providers=["ROCMExecutionProvider"])
                self.assertIn("ROCMExecutionProvider", sess.get_providers())
                options = sess.get_provider_options()

                def test_get_and_set_option_with_values(option_name, option_values):
                    provider_options = sess.get_provider_options()
                    self.assertIn("ROCMExecutionProvider", provider_options)
                    rocm_options = options["ROCMExecutionProvider"]
                    self.assertIn(option_name, rocm_options)
                    for option_value in option_values:
                        rocm_options[option_name] = option_value
                        sess.set_providers(["ROCMExecutionProvider"], [rocm_options])
                        new_provider_options = sess.get_provider_options()
                        self.assertEqual(
                            new_provider_options.get("ROCMExecutionProvider", {}).get(option_name),
                            str(option_value),
                        )

                test_get_and_set_option_with_values("tunable_op_enable", ["1", "0"])

                test_get_and_set_option_with_values("tunable_op_tuning_enable", ["1", "0"])

                test_get_and_set_option_with_values("tunable_op_max_tuning_duration_ms", ["-1", "1"])

            run_rocm_options_test()

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

        if "ROCMExecutionProvider" in onnxrt.get_available_providers():
            do_test_get_and_set_tuning_results("ROCMExecutionProvider")

    def test_run_model(self):
        sess = onnxrt.InferenceSession(get_name("mul_1.onnx"), providers=available_providers)
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
        np.testing.assert_allclose(output_expected, res[0], rtol=1e-05, atol=1e-08)

    def test_run_async(self):
        event = threading.Event()
        output_expected = np.array([[1.0, 4.0], [9.0, 16.0], [25.0, 36.0]], dtype=np.float32)

        class MyData:
            def __init__(self, id):
                self.__id = id

            def get_id(self):
                return self.__id

        my_data = MyData(123456)

        def callback(res: np.ndarray, data: MyData, err: str) -> None:
            self.assertEqual(len(err), 0)
            self.assertEqual(len(res), 1)
            self.assertEqual(data.get_id(), 123456)
            np.testing.assert_allclose(output_expected, res[0], rtol=1e-05, atol=1e-08)
            event.set()

        so = onnxrt.SessionOptions()
        so.intra_op_num_threads = 2

        sess = onnxrt.InferenceSession(get_name("mul_1.onnx"), so, providers=available_providers)

        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        sess.run_async(["Y"], {"X": x}, callback, my_data)

        event.wait(10)  # timeout in 10 sec
        self.assertTrue(event.is_set())

    def test_run_model_from_bytes(self):
        with open(get_name("mul_1.onnx"), "rb") as f:
            content = f.read()
        sess = onnxrt.InferenceSession(content, providers=onnxrt.get_available_providers())
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
        np.testing.assert_allclose(output_expected, res[0], rtol=1e-05, atol=1e-08)

    def test_run_model2(self):
        sess = onnxrt.InferenceSession(get_name("matmul_1.onnx"), providers=onnxrt.get_available_providers())
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
        np.testing.assert_allclose(output_expected, res[0], rtol=1e-05, atol=1e-08)

    def test_run_model2_contiguous(self):
        sess = onnxrt.InferenceSession(get_name("matmul_1.onnx"), providers=onnxrt.get_available_providers())
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
        np.testing.assert_allclose(output_expected, res[0], rtol=1e-05, atol=1e-08)
        xcontiguous = np.ascontiguousarray(x)
        rescontiguous = sess.run([output_name], {input_name: xcontiguous})
        np.testing.assert_allclose(output_expected, rescontiguous[0], rtol=1e-05, atol=1e-08)

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
        sess = onnxrt.InferenceSession(get_name("mul_1.onnx"), providers=onnxrt.get_available_providers())
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        input_name = sess.get_inputs()[0].name
        res = sess.run([], {input_name: x.tolist()})
        output_expected = np.array([[1.0, 4.0], [9.0, 16.0], [25.0, 36.0]], dtype=np.float32)
        np.testing.assert_allclose(output_expected, res[0], rtol=1e-05, atol=1e-08)

    def test_string_list_as_input(self):
        sess = onnxrt.InferenceSession(get_name("identity_string.onnx"), providers=available_providers_without_tvm)
        x = np.array(["this", "is", "identity", "test"], dtype=str).reshape((2, 2))
        x_name = sess.get_inputs()[0].name
        res = sess.run([], {x_name: x.tolist()})
        np.testing.assert_equal(x, res[0])

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
        np.testing.assert_allclose(output_expected, res[0], rtol=1e-05, atol=1e-08)

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
        np.testing.assert_equal(output_expected, res[0])

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
        np.testing.assert_equal(x, res[0])

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
        np.testing.assert_equal(x, res[0])

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
        np.testing.assert_equal(x, res[0].astype("|S8"))

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
        np.testing.assert_equal(x, res[0])

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
        np.testing.assert_equal(expr, res[0])

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
            providers=onnxrt.get_available_providers(),
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

        np.testing.assert_array_equal(output_expected, res[0])

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
        np.testing.assert_array_equal(output_expected, res[0])

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

    def test_register_custom_ops_library(self):
        if sys.platform.startswith("win"):
            shared_library = "custom_op_library.dll"
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
        np.testing.assert_allclose(output_expected, res[0], rtol=1e-05, atol=1e-08)

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
        numpy_arr_input = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        numpy_arr_output = np.array([[1.0, 4.0], [9.0, 16.0], [25.0, 36.0]], dtype=np.float32)

        def test_session_with_ortvalue_input(ortvalue):
            sess = onnxrt.InferenceSession(get_name("mul_1.onnx"), providers=onnxrt.get_available_providers())
            res = sess.run(["Y"], {"X": ortvalue})
            self.assertTrue(np.array_equal(res[0], numpy_arr_output))
            vect = sess._sess.run_with_ort_values({"X": ortvalue._get_c_value()}, ["Y"], RunOptions())
            self.assertIsInstance(vect, OrtValueVector)

        ortvalue1 = onnxrt.OrtValue.ortvalue_from_numpy(numpy_arr_input)
        self.assertEqual(ortvalue1.device_name(), "cpu")
        self.assertEqual(ortvalue1.shape(), [3, 2])
        self.assertEqual(ortvalue1.data_type(), "tensor(float)")
        self.assertEqual(ortvalue1.is_tensor(), True)
        self.assertTrue(np.array_equal(ortvalue1.numpy(), numpy_arr_input))

        # Pass in the constructed OrtValue to a session via Run() and check results
        test_session_with_ortvalue_input(ortvalue1)

        # The constructed OrtValue should still be valid after being used in a session
        self.assertTrue(np.array_equal(ortvalue1.numpy(), numpy_arr_input))

        if "CUDAExecutionProvider" in onnxrt.get_available_providers():
            ortvalue2 = onnxrt.OrtValue.ortvalue_from_numpy(numpy_arr_input, "cuda", 0)
            self.assertEqual(ortvalue2.device_name(), "cuda")
            self.assertEqual(ortvalue2.shape(), [3, 2])
            self.assertEqual(ortvalue2.data_type(), "tensor(float)")
            self.assertEqual(ortvalue2.is_tensor(), True)
            self.assertTrue(np.array_equal(ortvalue2.numpy(), numpy_arr_input))

            # Pass in the constructed OrtValue to a session via Run() and check results
            test_session_with_ortvalue_input(ortvalue2)

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

    def test_memory_arena_shrinkage(self):
        if platform.architecture()[0] == "32bit" or "ppc" in platform.machine() or "powerpc" in platform.machine():
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
        from onnxruntime.capi.onnxruntime_inference_collection import check_and_normalize_provider_args

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

    def test_register_custom_e_ps_library(self):
        from onnxruntime.capi import _pybind_state as C

        available_eps = C.get_available_providers()
        # skip amd gpu build
        if "kRocmExecutionProvider" in available_eps:
            return
        if sys.platform.startswith("win"):
            shared_library = "test_execution_provider.dll"

        elif sys.platform.startswith("darwin"):
            # exclude for macos
            return

        else:
            shared_library = "./libtest_execution_provider.so"

        if not os.path.exists(shared_library):
            raise FileNotFoundError(f"Unable to find '{shared_library}'")

        this = os.path.dirname(__file__)
        custom_op_model = os.path.join(this, "testdata", "custom_execution_provider_library", "test_model.onnx")
        if not os.path.exists(custom_op_model):
            raise FileNotFoundError(f"Unable to find '{custom_op_model}'")

        session_options = C.get_default_session_options()
        sess = C.InferenceSession(session_options, custom_op_model, True, True)
        sess.initialize_session(
            ["my_ep"],
            [
                {
                    "shared_lib_path": shared_library,
                    "device_id": "1",
                    "some_config": "val",
                }
            ],
            set(),
        )
        print("Create session with customize execution provider successfully!")

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

    def test_iobinding_multiple_devices(self):
        if "CUDAExecutionProvider" in onnxrt.get_available_providers():
            cuda_lib = self.load_cuda_lib()
            cuda_devices = self.cuda_device_count(cuda_lib)
            if cuda_devices <= 1:
                return
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


if __name__ == "__main__":
    unittest.main(verbosity=1)
