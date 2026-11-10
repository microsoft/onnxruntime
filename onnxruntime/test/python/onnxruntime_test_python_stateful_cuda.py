# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Explicit CUDA-plugin tests; normal value access must not copy device tensors."""

import gc
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnxruntime_test_python_stateful import STATEFUL_AVAILABLE, ddim_config
from onnxruntime_test_python_stateful_package import ddim_chain, denoiser_model

import onnxruntime as ort
from onnxruntime import stateful as sf


class TestStatefulCudaValues(unittest.TestCase):
    def test_native_sampling_readonly_copies_and_unregistered_library_lifetime(self):
        self._worker("--lifetime-worker", 90)

    def test_native_package_policies_standalone_devices_and_ddim(self):
        self._worker("--pipeline-worker", 180)

    def _worker(self, mode, timeout):
        if not STATEFUL_AVAILABLE:
            if os.environ.get("ORT_STATEFUL_REQUIRE_CUDA_PLUGIN") == "1":
                self.fail("Qualification requires stateful execution")
            self.skipTest("Stateful execution is unavailable in this build")
        if "cuda-plugin-ep=1" not in ort.get_build_info():
            if os.environ.get("ORT_STATEFUL_REQUIRE_CUDA_PLUGIN") == "1":
                self.fail("Qualification requires the actual CUDA-plugin Python build")
            self.skipTest("Native CUDA greedy requires the CUDA-plugin build")
        command = [sys.executable]
        coverage = sys.modules.get("coverage")
        if coverage is not None and coverage.Coverage.current() is not None:
            command.extend(["-m", "coverage", "run", "--parallel-mode", f"--include={Path(sf.__file__).resolve()}"])
        command.extend([str(Path(__file__).resolve()), mode])
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(Path(ort.__file__).resolve().parent.parent)
        subprocess.run(command, env=environment, check=True, timeout=timeout)

    def _cuda_options(self, device, stream=None):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        options.add_session_config_entry("session.record_ep_graph_assignment_info", "1")
        if stream is not None:
            options.add_session_config_entry("ep.cuda.user_compute_stream", str(stream.get_handle()))
        options.add_provider_for_devices([device], {})
        return options

    def _write_package(self, root, models, chain):
        manifest = {"schema_version": "1.1", "components": {}, "chains": {"pipeline": chain}}
        for component, model in models.items():
            for variant in ("cpu", "cuda"):
                current = onnx.load_from_string(model.SerializeToString())
                if variant == "cpu":
                    initializer = current.graph.initializer[0]
                    data = numpy_helper.to_array(initializer).copy()
                    data = np.roll(data, 1, axis=-1) if data.ndim else np.asarray(data * 2, dtype=data.dtype)
                    initializer.CopyFrom(numpy_helper.from_array(data, initializer.name))
                relative = f"{component}/{variant}"
                destination = root / relative
                destination.mkdir(parents=True)
                onnx.save_model(current, destination / "model.onnx")
                manifest["components"].setdefault(component, {"variants": {}})["variants"][variant] = {
                    "variant_directory": relative,
                    "ep": "CUDAExecutionProvider" if variant == "cuda" else "CPUExecutionProvider",
                    "device": "gpu" if variant == "cuda" else "cpu",
                    "executor_info": {"ort": {"model_file": "model.onnx"}},
                }
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def _decoder(self, dtype, element):
        scores = np.array([[0, 4, 1, -2], [0, 1, 5, -2], [-1, 1, 2, 6], [6, 2, 1, 0]], dtype=dtype)
        return helper.make_model(
            helper.make_graph(
                [
                    helper.make_node("Gather", ["scores", "token"], ["logits"], axis=0),
                    helper.make_node("Cast", ["token"], ["key_value"], to=element),
                    helper.make_node("Reshape", ["key_value", "key_shape"], ["new_key"]),
                    helper.make_node("Concat", ["past", "new_key"], ["present"], axis=2),
                ],
                "stateful_cuda_decoder",
                [
                    helper.make_tensor_value_info("token", TensorProto.INT64, [1, "S"]),
                    helper.make_tensor_value_info("past", element, [1, 1, "P", 1]),
                ],
                [
                    helper.make_tensor_value_info("logits", element, [1, "S", 4]),
                    helper.make_tensor_value_info("present", element, [1, 1, "T", 1]),
                ],
                [
                    numpy_helper.from_array(scores, "scores"),
                    numpy_helper.from_array(np.array([1, 1, -1, 1], dtype=np.int64), "key_shape"),
                ],
            ),
            opset_imports=[helper.make_opsetid("", 18)],
            ir_version=10,
        )

    def _run_pipeline(self):
        devices = [
            device
            for device in ort.get_ep_devices()
            if device.ep_name == "CUDAExecutionProvider"
            and device.memory_info(ort.OrtDeviceMemoryType.DEFAULT).device_id == 0
        ]
        self.assertTrue(devices)
        device = devices[0]
        stream = device.create_sync_stream()
        self._streams = [stream]
        options = self._cuda_options(device, stream)
        requested = sf.ProviderSelection(sf.ProviderMode.REQUIRE, "CUDAExecutionProvider", sf.DeviceType.GPU, 0)
        gpu_policy = {
            "mode": "require_provider",
            "provider_name": "CUDAExecutionProvider",
            "device_type": "gpu",
            "device_id": 0,
        }
        for dtype, element, cache_type in (
            (np.float32, TensorProto.FLOAT, sf.ElementType.FLOAT32),
            (np.float16, TensorProto.FLOAT16, sf.ElementType.FLOAT16),
        ):
            model = self._decoder(dtype, element)
            session = ort.InferenceSession(model.SerializeToString(), options)
            self.assertTrue(
                all(graph.ep_name == "CUDAExecutionProvider" for graph in session.get_provider_graph_assignment_info())
            )
            caps = sf.query_provider_capabilities(session, "CUDAExecutionProvider")
            self.assertTrue(caps.operations & (1 << sf.ProviderOperation.SAMPLE_GREEDY))
            with sf.Sampler(provider=requested, session=session) as sampler:
                with self.assertRaisesRegex(ort.capi.onnxruntime_pybind11_state.Fail, "device mismatch"):
                    sampler.sample(np.ones((1, 4), dtype=dtype))
                other = next(
                    (
                        candidate
                        for candidate in ort.get_ep_devices()
                        if candidate.ep_name == "CUDAExecutionProvider"
                        and candidate.memory_info(ort.OrtDeviceMemoryType.DEFAULT).device_id == 1
                    ),
                    None,
                )
                if other is not None:
                    other_session = ort.InferenceSession(model.SerializeToString(), self._cuda_options(other))
                    wrong_device = other_session.create_ortvalue_from_shape_and_type((1, 4), dtype, "cuda", 1)
                    with self.assertRaisesRegex(ort.capi.onnxruntime_pybind11_state.Fail, "device mismatch"):
                        sampler.sample(wrong_device)
                self.assertEqual(sampler.get_dispatch_info().cpu_calls, 0)
                self.assertEqual(sampler.get_dispatch_info().provider_calls, 0)
            cache = sf.KVCacheManager(
                (sf.CacheSpec("key", cache_type, (1, 1, 0, 1), 2),), provider=requested, session=session
            )
            value = session.create_ortvalue_from_shape_and_type((1, 1, 1, 1), dtype, "cuda", 0)
            ort.copy_tensors([ort.OrtValue.ortvalue_from_numpy(np.ones((1, 1, 1, 1), dtype=dtype))], [value])
            cache.update("key", value, sf.CacheUpdate.SHARE)
            old = cache.get_view("key").get_info().value
            cache.update("key", value, sf.CacheUpdate.APPEND)
            cache.reserve("key", 4)
            current = cache.get_view("key").get_info()
            self.assertEqual(current.length, 2)
            self.assertEqual(current.value.get_info().device_type, sf.DeviceType.GPU)
            np.testing.assert_array_equal(current.value.copy_to_cpu().numpy(), np.ones((1, 1, 2, 1), dtype=dtype))
            cache.reset()
            cache.close()
            np.testing.assert_array_equal(old.copy_to_cpu().numpy(), np.ones((1, 1, 1, 1), dtype=dtype))
            for gpu_cache in (False, True):
                for gpu_sampler in (False, True):
                    for budget in (0, 1, 3):
                        with tempfile.TemporaryDirectory() as directory:
                            bindings = [{"cache": "key", "input": "past", "output": "present", "update": "share"}]
                            prefill = {
                                "id": "prefill",
                                "component": "prefill_model",
                                "executor_info": {
                                    "ort": {
                                        "kind": "session_loop",
                                        "inputs": {"token": {"kind": "external_sequence", "name": "prompt"}},
                                        "loop": {
                                            "finite_input_names": ["token"],
                                            "nonempty_input": "token",
                                            "exit_conditions": [{"kind": "end_of_inputs"}],
                                            "kv_caches": bindings,
                                            "cache_provider": gpu_policy if gpu_cache else {"mode": "cpu"},
                                        },
                                    }
                                },
                            }
                            decode = {
                                "id": "decode",
                                "component": "decode_model",
                                "executor_info": {
                                    "ort": {
                                        "kind": "session_loop",
                                        "loop": {
                                            "exit_conditions": [{"kind": "max_generated_tokens", "limit": budget}],
                                            "kv_caches": bindings,
                                            "cache_provider": gpu_policy if gpu_cache else {"mode": "cpu"},
                                            "generation": {
                                                "phase": "decode",
                                                "input_ids": "token",
                                                "logits_output": "logits",
                                                "initial_logits": {
                                                    "kind": "model_output",
                                                    "step": "prefill",
                                                    "output": "logits",
                                                },
                                            },
                                            "sampler": {
                                                "algorithm": "greedy",
                                                "provider": gpu_policy if gpu_sampler else {"mode": "cpu"},
                                            },
                                        },
                                    }
                                },
                            }
                            chain = {
                                "steps": [prefill, decode],
                                "executor_info": {
                                    "ort": {
                                        "kv_caches": {
                                            "key": {
                                                "element_type": "float32" if dtype == np.float32 else "float16",
                                                "shape": [1, 1, 0, 1],
                                                "sequence_axis": 2,
                                                "initial_capacity": 0,
                                            }
                                        },
                                        "outputs": {
                                            "tokens": {"kind": "generated_token_ids", "step": "decode"},
                                            "prefill_logits": {
                                                "kind": "model_output",
                                                "step": "prefill",
                                                "output": "logits",
                                            },
                                        },
                                    }
                                },
                            }
                            root = Path(directory) / "pkg"
                            self._write_package(root, {"prefill_model": model, "decode_model": model}, chain)
                            with sf.ModelPackage(root) as package:
                                definition = package.create_chain(
                                    "pipeline", options, step_session_options={"decode": options}
                                )
                            execution = definition.create_execution(
                                sequences={"prompt": [np.array([[0, 1]], dtype=np.int64)]}
                            )
                            definition.close()
                            execution.step()
                            sampled = execution.step()
                            self.assertFalse(sampled.get_info().executed)
                            self.assertEqual(sampled.get_info().invocations, 1)
                            result = sampled if sampled.get_info().completed else execution.run()
                            expected = np.array([2, 3, 0][:budget], dtype=np.int64)
                            np.testing.assert_array_equal(result.get_outputs()["tokens"].numpy(), expected)
                            self.assertEqual(result.get_info().invocations, max(1, budget))
                            dispatch = execution.get_dispatch_info(
                                sf.ProviderFeature.SAMPLER, sf.ProviderOperation.SAMPLE_GREEDY, 1
                            )
                            self.assertEqual(dispatch.provider_calls, budget if gpu_sampler else 0)
                            self.assertEqual(dispatch.cpu_calls, 0 if gpu_sampler else budget)
                            logits = result.get_outputs()["prefill_logits"]
                            self.assertEqual(
                                logits.get_info().device_type, sf.DeviceType.GPU if gpu_sampler else sf.DeviceType.CPU
                            )
                            if gpu_sampler:
                                with sf.Sampler(provider=requested, session=session) as sampler:
                                    self.assertEqual(sampler.sample(logits), 2)
                                    self.assertEqual(sampler.get_dispatch_info().provider_calls, 1)
                            state = execution.get_state().get_outputs(sf.OutputGroup.STATE_VALUES)["key"]
                            self.assertEqual(state.get_info().shape[2], 2 + max(0, budget - 1))
                            self.assertEqual(
                                state.get_info().device_type, sf.DeviceType.GPU if gpu_cache else sf.DeviceType.CPU
                            )
                            execution.reset()
                            np.testing.assert_array_equal(execution.run().get_outputs()["tokens"].numpy(), expected)
                            execution.close()
        for mode in sf.GuidanceMode:
            with tempfile.TemporaryDirectory() as directory:
                policy = ddim_config(latent_shape=(1, 1, 1, 3), guidance_mode=mode)
                root = Path(directory) / "pkg"
                self._write_package(root, {"denoiser": denoiser_model(policy)}, ddim_chain(policy))
                with sf.ModelPackage(root) as package:
                    definition = package.create_chain("pipeline", options)
                execution = definition.create_execution(
                    {
                        "init": np.array([0.125, -0.25, 1], dtype=np.float32).reshape(1, 1, 1, 3),
                        "positive": np.full((1, 1, 1, 1), 0.25, dtype=np.float32),
                        "negative": np.full((1, 1, 1, 1), -0.5, dtype=np.float32),
                    }
                )
                definition.close()
                result = execution.run()
                value = result.get_outputs()["latents"]
                expected = np.array([-8.022068, -8.661748, -6.5294814], dtype=np.float32)
                np.testing.assert_allclose(value.numpy().ravel(), expected, rtol=1e-4, atol=1e-4)
                self.assertEqual(value.get_info().device_type, sf.DeviceType.CPU)
                self.assertEqual(result.get_info().invocations, 2 if mode == sf.GuidanceMode.BATCHED else 4)
                execution.close()

    def _run_value_lifetime(self):
        bundled = ort._get_cuda_plugin_ep_library_path()
        library = os.environ.get("ORT_STATEFUL_TEST_CUDA_LIBRARY") or bundled
        self.assertIsNotNone(library)
        self.assertTrue(Path(library).is_file())
        registration = "CUDAExecutionProvider" if bundled else "m13-stateful-cuda"
        registered = False
        try:
            if not bundled:
                ort.register_execution_provider_library(registration, library)
            registered = True
            retained = []
            for dtype, element_type in ((np.float32, TensorProto.FLOAT), (np.float16, TensorProto.FLOAT16)):
                with self.subTest(dtype=dtype):
                    devices = [
                        device
                        for device in ort.get_ep_devices()
                        if device.ep_name == "CUDAExecutionProvider"
                        and device.memory_info(ort.OrtDeviceMemoryType.DEFAULT).device_id == 0
                    ]
                    self.assertTrue(devices, "Required CUDA plugin ordinal0 was not discovered")
                    options = ort.SessionOptions()
                    options.intra_op_num_threads = 1
                    options.inter_op_num_threads = 1
                    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
                    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
                    options.add_session_config_entry("session.record_ep_graph_assignment_info", "1")
                    options.add_provider_for_devices([devices[0]], {})
                    scores = np.array([[0, 4, 1, -2], [0, 1, 5, -2]], dtype=dtype)
                    model = helper.make_model(
                        helper.make_graph(
                            [helper.make_node("Gather", ["scores", "token"], ["logits"], axis=0)],
                            "stateful_cuda_readonly",
                            [helper.make_tensor_value_info("token", TensorProto.INT64, [1, "S"])],
                            [helper.make_tensor_value_info("logits", element_type, [1, "S", 4])],
                            [numpy_helper.from_array(scores, "scores")],
                        ),
                        opset_imports=[helper.make_opsetid("", 18)],
                        ir_version=10,
                    )
                    session = ort.InferenceSession(model.SerializeToString(), options)
                    assignment = session.get_provider_graph_assignment_info()
                    self.assertTrue(assignment)
                    self.assertTrue(all(graph.ep_name == "CUDAExecutionProvider" for graph in assignment))
                    provider = sf.ProviderSelection(
                        sf.ProviderMode.REQUIRE, "CUDAExecutionProvider", sf.DeviceType.GPU, 0
                    )
                    config = sf.LoopConfig(
                        ("logits",),
                        (sf.ExitCondition(sf.ExitKind.END_OF_INPUTS),),
                        finite_input_names=("token",),
                        nonempty_input="token",
                        generation=sf.GenerationConfig(sf.GenerationPhase.PREFILL, "token", "logits"),
                        sampler_provider=provider,
                    )
                    with sf.SessionLoop(session, config) as loop:
                        execution = loop.create_execution(batches=[{"token": np.array([[0, 1]], dtype=np.int64)}])
                    result = execution.run()
                    self.assertEqual(result.get_info().generated_token, 2)
                    dispatch = execution.get_dispatch_info(
                        sf.ProviderFeature.SAMPLER, sf.ProviderOperation.SAMPLE_GREEDY
                    )
                    self.assertEqual(dispatch.origin, sf.DispatchOrigin.PROVIDER)
                    self.assertEqual(dispatch.provider_calls, 1)
                    self.assertEqual(dispatch.cpu_calls, 0)
                    self.assertEqual(dispatch.cpu_delegations, 0)
                    value = result.get_outputs()["logits"]
                    self.assertEqual(value.get_info().device_type, sf.DeviceType.GPU)
                    self.assertEqual(value.get_info().device_id, 0)
                    self.assertEqual(value.get_info().element_type, element_type)
                    with self.assertRaises(ValueError):
                        value.numpy()
                    clone = value.to_ortvalue()
                    self.assertEqual(clone.device_name().lower(), "cuda")
                    zeros = ort.OrtValue.ortvalue_from_numpy(np.zeros((1, 2, 4), dtype=dtype))
                    ort.copy_tensors([zeros], [clone])
                    np.testing.assert_array_equal(value.copy_to_cpu().numpy(), scores[np.newaxis, ...])
                    result.close()
                    execution.close()
                    del session, options, devices, clone, zeros
                    gc.collect()
                    retained.append((value, scores[np.newaxis, ...]))
            ort.unregister_execution_provider_library(registration)
            registered = False
            for value, expected in retained:
                np.testing.assert_array_equal(value.copy_to_cpu().numpy(), expected)
                clone = value.to_ortvalue()
                self.assertEqual(clone.device_name().lower(), "cuda")
                value.close()
                del clone
        finally:
            if registered:
                ort.unregister_execution_provider_library(registration)


if __name__ == "__main__":
    if sys.argv[1:] == ["--lifetime-worker"]:
        TestStatefulCudaValues()._run_value_lifetime()
    elif sys.argv[1:] == ["--pipeline-worker"]:
        worker = TestStatefulCudaValues()
        worker._run_pipeline()
        gc.collect()
        worker._streams.clear()
    else:
        unittest.main()
