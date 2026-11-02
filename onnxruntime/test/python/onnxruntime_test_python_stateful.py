# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Public stateful Python tests against the locally built runtime."""

import dataclasses
import gc
import unittest
import weakref
from types import SimpleNamespace
from unittest import mock

import numpy as np
from onnx import TensorProto, helper, numpy_helper

import onnxruntime as ort
from onnxruntime import OrtMemoryInfoDeviceType
from onnxruntime import stateful as sf


def ddim_config(**changes):
    fields = {
        "latent_shape": (1, 4, 2, 2),
        "num_inference_steps": 2,
        "guidance_scale": 7.5,
        "latent_input": "sample",
        "timestep_input": "time",
        "conditioning_input": "conditioning",
        "prediction_output": "epsilon",
        "initialization_input": "init",
        "positive_conditioning_input": "positive",
        "negative_conditioning_input": "negative",
        "latent_output": "latents",
    }
    fields.update(changes)
    return sf.DdimConfig(**fields)


class TestStatefulConfig(unittest.TestCase):
    def test_collection_ownership_and_frozen_records(self):
        shape = [1, 1, 0, 2]
        cache = sf.CacheSpec("key", sf.ElementType.FLOAT32, shape, 2, 4)
        shape[0] = 9
        self.assertEqual(cache.shape, (1, 1, 0, 2))
        binding = sf.CacheBinding(cache, "past", "present", sf.CacheUpdate.COPY)
        ids = [2, 3]
        eos = sf.ExitCondition(sf.ExitKind.EOS, eos_token_ids=ids)
        ids.append(4)
        self.assertEqual(eos.eos_token_ids, (2, 3))
        outputs, finite = ["present"], ["input"]
        exits, feedback, caches = [eos], [sf.FeedbackBinding("x", "y")], [binding]
        config = sf.LoopConfig(
            outputs,
            exits,
            finite_input_names=finite,
            feedback=feedback,
            kv_caches=caches,
            generated_token_output="token",
            nonempty_input="input",
        )
        for collection in (outputs, finite, exits, feedback, caches):
            collection.clear()
        self.assertEqual(config.output_names, ("present",))
        self.assertEqual(config.finite_input_names, ("input",))
        self.assertEqual(config.exit_conditions, (eos,))
        self.assertEqual(config.feedback, (sf.FeedbackBinding("x", "y"),))
        self.assertEqual(config.kv_caches, (binding,))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            config.output_names = ("changed",)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            cache.shape = (1, 1, 0, 1)

    def test_all_typed_binding_records_and_policy_configuration(self):
        source = sf.ValueSource(sf.SourceKind.EXTERNAL_INPUT, "prompt")
        initial = sf.GenerationBootstrap(
            sf.ValueSource(sf.SourceKind.COMPONENT_OUTPUT, "logits", 0), source, logits_position=1
        )
        self.assertEqual(initial.prompt, source)
        generation = sf.GenerationConfig(
            sf.GenerationPhase.DECODE, "input_ids", "logits", "mask", "positions", logits_position=0
        )
        provider = sf.ProviderSelection(
            sf.ProviderMode.REQUIRE, "CUDAExecutionProvider", sf.DeviceType.GPU, device_id=1
        )
        loop = sf.LoopConfig(
            ("logits",),
            (sf.ExitCondition(sf.ExitKind.MAX_GENERATED_TOKENS, limit=3),),
            generation=generation,
            cache_provider=provider,
            sampler_provider=provider,
        )
        self.assertIs(loop.generation, generation)
        self.assertEqual(loop.cache_provider.device_id, 1)
        self.assertEqual(sf.SamplerConfig(logits_position=2).logits_position, 2)
        self.assertIsNone(sf.SamplerConfig().logits_position)
        self.assertIsNone(sf.GenerationConfig(sf.GenerationPhase.PREFILL, "input_ids", "logits").attention_mask)
        self.assertIsNone(sf.GenerationBootstrap(initial.logits, initial.prompt).logits_position)
        inputs = [sf.InputBinding("input_ids", source)]
        states = [sf.StateBinding("keys", "present")]
        component = sf.ChainComponent(
            "decode",
            execution=None,
            inputs=inputs,
            state_outputs=states,
            cache_provider=provider,
            sampler_provider=provider,
            generation_bootstrap=initial,
        )
        inputs.clear()
        states.clear()
        self.assertEqual(len(component.inputs), 1)
        self.assertEqual(len(component.state_outputs), 1)
        components = [component]
        selected = [sf.ChainOutput("tokens", 0, kind=sf.OutputKind.GENERATED_TOKEN_IDS)]
        chain = sf.ChainConfig(components, selected)
        components.clear()
        selected.clear()
        self.assertEqual(len(chain.components), 1)
        self.assertEqual(len(chain.outputs), 1)
        self.assertIsNone(sf.ChainComponent("plain", execution=None).generation_bootstrap)
        self.assertEqual(sf.ChainOutput("result", 0, "output").output_name, "output")

    def test_ddim_shape_capture_and_explicit_formats(self):
        shape = [2, 4, 2, 3]
        config = ddim_config(latent_shape=shape, guidance_mode=sf.GuidanceMode.SEQUENTIAL)
        shape.clear()
        self.assertEqual(config.latent_shape, (2, 4, 2, 3))
        for format_ in sf.TimestepFormat:
            policy = dataclasses.replace(config, timestep_format=format_)
            loop = sf.LoopConfig(("latents",), (sf.ExitCondition(sf.ExitKind.MAX_ITERATIONS, limit=2),), ddim=policy)
            self.assertEqual(loop.ddim.timestep_format, format_)
        self.assertEqual(ddim_config(guidance_scale=1).guidance_scale, 1)
        for enum in (
            sf.ExitKind,
            sf.SourceKind,
            sf.OutputKind,
            sf.OutputGroup,
            sf.CacheUpdate,
            sf.ElementType,
            sf.GenerationPhase,
            sf.SamplingAlgorithm,
            sf.ProviderMode,
            sf.DeviceType,
            sf.ProviderFeature,
            sf.ProviderOperation,
            sf.GuidanceMode,
            sf.TimestepFormat,
        ):
            self.assertTrue(all(isinstance(value.value, int) for value in enum))
        for value in sf.DeviceType:
            self.assertEqual(value.value, getattr(OrtMemoryInfoDeviceType, value.name).value)

    def test_invalid_primitive_values_and_configuration_types(self):
        with self.assertRaises(TypeError):
            sf.ExitCondition(1)
        for name in (None, 42):
            with self.subTest(name=name), self.assertRaises(TypeError):
                sf.FeedbackBinding(name, "output")
        for name in ("", "a\0b"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                sf.FeedbackBinding(name, "output")
        with self.assertRaises(UnicodeEncodeError):
            sf.FeedbackBinding("\ud800", "output")
        for limit in (True, 1.5, "1"):
            with self.subTest(limit=limit), self.assertRaises(TypeError):
                sf.ExitCondition(sf.ExitKind.MAX_ITERATIONS, limit=limit)
        for limit in (-(1 << 63) - 1, 1 << 63):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                sf.ExitCondition(sf.ExitKind.MAX_ITERATIONS, limit=limit)
        constructors = (
            lambda: sf.InputBinding("x", object()),
            lambda: sf.CacheBinding(object(), "x", "y"),
            lambda: sf.CacheSpec("key", sf.ElementType.FLOAT32, "shape", 0),
            lambda: sf.CacheSpec("key", sf.ElementType.FLOAT32, (1, "x"), 0),
            lambda: sf.CacheSpec("key", sf.ElementType.FLOAT32, (1, True), 0),
            lambda: sf.LoopConfig("out", ()),
            lambda: sf.LoopConfig(("",), ()),
            lambda: sf.LoopConfig(("out",), (object(),)),
            lambda: sf.LoopConfig(("out",), (), generation=object()),
            lambda: sf.LoopConfig(("out",), (), ddim=object()),
            lambda: sf.LoopConfig(("out",), (), cache_provider=None),
            lambda: sf.GenerationBootstrap(object(), object()),
            lambda: sf.ChainComponent("stage", None, cache_provider=object()),
            lambda: sf.ChainComponent("stage", None, generation_bootstrap=object()),
            lambda: ddim_config(guidance_scale=True),
            lambda: ddim_config(guidance_mode=0),
            lambda: ddim_config(timestep_format=sf.GuidanceMode.BATCHED),
            lambda: sf.SamplerConfig(algorithm="greedy"),
            lambda: sf.ProviderSelection(mode=sf.ExitKind.END_OF_INPUTS),
            lambda: sf.ExitCondition(sf.ExitKind.END_OF_INPUTS, limit=1),
            lambda: sf.ExitCondition(sf.ExitKind.MAX_ITERATIONS, limit=-1),
            lambda: sf.ExitCondition(sf.ExitKind.END_OF_INPUTS, output_name="unused"),
            lambda: sf.ExitCondition(sf.ExitKind.MAX_ITERATIONS, eos_token_ids=(1,)),
            lambda: sf.SamplerConfig(logits_position=-1),
            lambda: sf.GenerationConfig(sf.GenerationPhase.PREFILL, "input_ids", "logits", logits_position=-1),
            lambda: sf.GenerationBootstrap(
                sf.ValueSource(sf.SourceKind.COMPONENT_OUTPUT, "logits"),
                sf.ValueSource(sf.SourceKind.EXTERNAL_INPUT, "prompt"),
                logits_position=-1,
            ),
        )
        for constructor in constructors:
            with self.subTest(constructor=constructor), self.assertRaises((TypeError, ValueError)):
                constructor()
        with self.assertRaises(TypeError):
            sf.ProviderSelection(unrecognized_option="x")


def feedback_session():
    graph = helper.make_graph(
        [
            helper.make_node("Add", ["x", "past"], ["present"]),
            helper.make_node("Greater", ["present", "limit"], ["done"]),
        ],
        "stateful_python_feedback",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("past", TensorProto.FLOAT, []),
        ],
        [
            helper.make_tensor_value_info("present", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("done", TensorProto.BOOL, []),
        ],
        [numpy_helper.from_array(np.array(5, dtype=np.float32), "limit")],
    )
    return tiny_session(graph)


def feedback_config():
    return sf.LoopConfig(
        ("present",),
        (sf.ExitCondition(sf.ExitKind.END_OF_INPUTS),),
        finite_input_names=("x",),
        feedback=(sf.FeedbackBinding("past", "present"),),
    )


def scalar(value):
    return np.array(value, dtype=np.float32)


def tiny_session(graph):
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])


def ddim_session(policy):
    batch = policy.latent_shape[0] * (2 if policy.guidance_mode == sf.GuidanceMode.BATCHED else 1)
    shape = [batch, *policy.latent_shape[1:]]
    integer = policy.timestep_format in (sf.TimestepFormat.INT64_SCALAR, sf.TimestepFormat.INT64_VECTOR)
    vector = policy.timestep_format in (sf.TimestepFormat.INT64_VECTOR, sf.TimestepFormat.FLOAT32_VECTOR)
    time_type = TensorProto.INT64 if integer else TensorProto.FLOAT
    nodes = []
    time = "time"
    if integer:
        nodes.append(helper.make_node("Cast", ["time"], ["float_time"], to=TensorProto.FLOAT))
        time = "float_time"
    nodes.extend(
        [
            helper.make_node("Mul", ["sample", "latent_scale"], ["scaled_latent"]),
            helper.make_node("Mul", [time, "time_scale"], ["scaled_time"]),
            helper.make_node("Add", ["scaled_latent", "scaled_time"], ["time_prediction"]),
            helper.make_node("Add", ["time_prediction", "conditioning"], ["epsilon"]),
        ]
    )
    return tiny_session(
        helper.make_graph(
            nodes,
            "stateful_python_ddim",
            [
                helper.make_tensor_value_info("sample", TensorProto.FLOAT, shape),
                helper.make_tensor_value_info("time", time_type, [1] if vector else []),
                helper.make_tensor_value_info("conditioning", TensorProto.FLOAT, [batch, 1, 1, 1]),
            ],
            [helper.make_tensor_value_info("epsilon", TensorProto.FLOAT, shape)],
            [
                numpy_helper.from_array(scalar(0.125), "latent_scale"),
                numpy_helper.from_array(scalar(0.0001), "time_scale"),
            ],
        )
    )


class TestStatefulStandalone(unittest.TestCase):
    def test_cache_share_copy_append_reserve_reset_and_retained_views(self):
        for dtype, element in ((np.float32, sf.ElementType.FLOAT32), (np.float16, sf.ElementType.FLOAT16)):
            with self.subTest(dtype=dtype):
                manager = sf.KVCacheManager((sf.CacheSpec("key", element, (1, 1, 0, 2), 2, 2),))
                initial = manager.get_view("key").get_info()
                self.assertEqual(initial.length, 0)
                self.assertGreaterEqual(initial.capacity, 2)
                self.assertEqual(initial.value.get_info().shape, (1, 1, 0, 2))
                borrowed = np.array([[[[1, 2]]]], dtype=dtype)
                with self.assertRaises(ort.capi.onnxruntime_pybind11_state.Fail):
                    manager.update("key", borrowed, sf.CacheUpdate.SHARE)
                self.assertFalse(manager.requires_reset())
                shared = ort.OrtValue.ortvalue_from_shape_and_type((1, 1, 1, 2), dtype)
                shared.update_inplace(borrowed)
                weak_shared = weakref.ref(shared)
                manager.update("key", shared, sf.CacheUpdate.SHARE)
                old = manager.get_view("key")
                self.assertEqual(old.get_info().length, 1)
                del shared
                gc.collect()
                self.assertIsNotNone(weak_shared())
                copied = np.array([[[[3, 4]]]], dtype=dtype)
                manager.update("key", copied, sf.CacheUpdate.COPY)
                copied[...] = -1
                copy_view = manager.get_view("key").get_info()
                np.testing.assert_array_equal(copy_view.value.numpy(), np.array([[[[3, 4]]]], dtype=dtype))
                manager.update("key", np.array([[[[5, 6]]]], dtype=dtype), sf.CacheUpdate.APPEND)
                appended = manager.get_view("key").get_info()
                expected = np.array([[[[3, 4], [5, 6]]]], dtype=dtype)
                self.assertEqual(appended.length, 2)
                np.testing.assert_array_equal(appended.value.numpy(), expected)
                manager.reserve("key", 4)
                self.assertGreaterEqual(manager.get_view("key").get_info().capacity, 4)
                stats = manager.get_stats()
                self.assertGreater(stats.allocations, 0)
                self.assertGreater(stats.bytes_copied, 0)
                self.assertEqual(stats.shared_updates, 1)
                dispatch = manager.get_dispatch_info(sf.ProviderOperation.CACHE_SHARE)
                self.assertEqual(dispatch.origin, sf.DispatchOrigin.CPU)
                self.assertEqual(dispatch.cpu_calls, 1)
                manager.reset()
                self.assertFalse(manager.requires_reset())
                self.assertEqual(manager.get_view("key").get_info().length, 0)
                manager.close()
                np.testing.assert_array_equal(appended.value.numpy(), expected)
                np.testing.assert_array_equal(old.get_info().value.numpy(), np.array([[[[1, 2]]]], dtype=dtype))
                np.testing.assert_array_equal(copy_view.value.numpy(), np.array([[[[3, 4]]]], dtype=dtype))
                old.close()
                appended.value.close()
                copy_view.value.close()
                initial.value.close()
                with self.assertRaises(RuntimeError):
                    manager.reset()

    def test_standalone_sampler_native_ties_rows_failures_and_diagnostics(self):
        for dtype in (np.float32, np.float16):
            with self.subTest(dtype=dtype), sf.Sampler() as sampler:
                logits = np.array([[2, 4, 4, 1]], dtype=dtype)
                self.assertEqual(sampler.sample(logits), 1)
                value = ort.OrtValue.ortvalue_from_numpy(logits)
                self.assertEqual(sampler.sample(value), 1)
                info = sampler.get_dispatch_info()
                self.assertEqual(info.origin, sf.DispatchOrigin.CPU)
                self.assertEqual(info.cpu_calls, 2)
                self.assertEqual(info.provider_calls, 0)
                with self.assertRaises(ort.capi.onnxruntime_pybind11_state.Fail):
                    sampler.sample(np.array([[np.nan, 0]], dtype=dtype))
                self.assertEqual(sampler.sample(logits), 1)
            with self.assertRaises(RuntimeError):
                sampler.sample(logits)
        rows = np.array([[[9, 1, 1], [0, 2, 8]]], dtype=np.float32)
        with sf.Sampler(sf.SamplerConfig(logits_position=0)) as first, sf.Sampler() as last:
            self.assertEqual(first.sample(rows), 0)
            self.assertEqual(last.sample(rows), 2)

    def test_standalone_configuration_and_operation_errors_remain_explicit(self):
        with self.assertRaises(TypeError):
            sf.Sampler(object())
        with self.assertRaises(TypeError):
            sf.Sampler(provider=object())
        with self.assertRaises(TypeError):
            sf.KVCacheManager((), provider=object())
        with self.assertRaises(TypeError):
            sf.KVCacheManager((), session=object())
        uninitialized = object.__new__(ort.InferenceSession)
        uninitialized._sess = None
        with self.assertRaises(ValueError):
            sf.Sampler(session=uninitialized)
        with self.assertRaises(TypeError):
            sf.query_provider_capabilities(None, "CPUExecutionProvider")
        caps = sf.query_provider_capabilities(feedback_session(), "CPUExecutionProvider")
        self.assertEqual(caps.version, 1)
        self.assertIsInstance(caps.operations, int)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            caps.version = 2
        with self.assertRaises(ort.capi.onnxruntime_pybind11_state.InvalidArgument):
            sf.query_provider_capabilities(feedback_session(), "missing")
        with sf.KVCacheManager((sf.CacheSpec("key", sf.ElementType.FLOAT32, (1, 1, 0, 2), 2),)) as manager:
            with self.assertRaises(ort.capi.onnxruntime_pybind11_state.Fail):
                manager.get_view("missing")
            with self.assertRaises(ort.capi.onnxruntime_pybind11_state.InvalidArgument):
                manager.get_dispatch_info(sf.ProviderOperation.SAMPLE_GREEDY)
            with self.assertRaises(ort.capi.onnxruntime_pybind11_state.Fail):
                manager.reserve("key", -1)
            with self.assertRaises(TypeError):
                manager.update("key", np.zeros((1, 1, 1, 2), dtype=np.float32), kind="copy")


class TestStatefulExecution(unittest.TestCase):
    def test_native_children_are_destroyed_before_transitive_dependencies(self):
        events = []

        class Observed:
            def __init__(self, name):
                self.name = name

            def __del__(self):
                events.append(self.name)

        dependency = Observed("dependency")
        native = Observed("parent")
        parent = sf._Resource(native, (dependency,))
        child = sf._Resource(Observed("child"), (sf._Lease(native, (dependency,)),))
        del native, dependency
        parent.close()
        self.assertEqual(events, [])
        child.close()
        self.assertEqual(events, ["child", "parent", "dependency"])

    def test_feature_unavailability_keeps_configuration_usable(self):
        session = feedback_session()
        with (
            mock.patch.object(sf._C, "_stateful", SimpleNamespace(available=False)),
            self.assertRaises(ort.capi.onnxruntime_pybind11_state.NotImplemented),
        ):
            config = feedback_config()
            sf.SessionLoop(session, config)

    def test_loop_owns_inputs_session_results_and_independent_copies(self):
        session = feedback_session()
        original = session._sess
        loop = sf.SessionLoop(session, feedback_config())
        session.set_providers(["CPUExecutionProvider"])
        self.assertIsNot(session._sess, original)
        del original, session
        past = scalar(0)
        weak_past = weakref.ref(past)
        execution = loop.create_execution({"past": past}, batches=[{"x": scalar(x)} for x in (2, 3, 4)])
        del past
        loop.close()
        loop.close()
        gc.collect()
        self.assertIsNotNone(weak_past())
        first = execution.step()
        first_info = first.get_info()
        self.assertTrue(first_info.executed)
        self.assertEqual((first_info.iterations, first_info.invocations), (1, 1))
        value = first.get_outputs()["present"]
        first.close()
        self.assertEqual(value.get_info().shape, ())
        self.assertEqual(value.get_info().device_type, sf.DeviceType.CPU)
        self.assertEqual(value.numpy().item(), 2)
        converted = value.to_ortvalue()
        converted.update_inplace(scalar(99))
        self.assertEqual(value.numpy().item(), 2)
        array = value.numpy()
        array[...] = -1
        self.assertEqual(value.numpy().item(), 2)
        explicit = value.copy_to_cpu()
        explicit.update_inplace(scalar(77))
        self.assertEqual(value.numpy().item(), 2)
        final = execution.run()
        self.assertEqual(final.get_outputs()["present"].numpy().item(), 9)
        self.assertEqual(final.get_info().invocations, 3)
        staging = execution.get_staging_info()
        self.assertGreater(staging.cpu_snapshot_copy_bytes, 0)
        self.assertGreater(staging.cpu_input_request_bytes, 0)
        state = execution.get_state()
        self.assertEqual(state.get_info().kind, sf.ResultKind.STATE_SNAPSHOT)
        self.assertEqual(state.get_outputs(sf.OutputGroup.STATE_VALUES)["past"].numpy().item(), 9)
        completed = execution.step()
        self.assertFalse(completed.get_info().executed)
        self.assertTrue(completed.get_info().completed)
        self.assertEqual(dict(completed.get_outputs()), {})
        execution.reset()
        self.assertEqual(execution.run().get_outputs()["present"].numpy().item(), 9)
        self.assertEqual(value.numpy().item(), 2)
        execution.close()
        final.close()
        state.close()
        completed.close()
        gc.collect()
        self.assertEqual(value.numpy().item(), 2)
        self.assertFalse(hasattr(value, "data_ptr"))
        self.assertFalse(hasattr(value, "update_inplace"))
        self.assertFalse(hasattr(value, "__dlpack__"))
        self.assertFalse(hasattr(value, "__array__"))
        value.close()
        with self.assertRaises(RuntimeError):
            value.numpy()
        with self.assertRaises(RuntimeError):
            first.get_info()
        self.assertEqual(first_info.iterations, 1)

    def test_cancel_reset_busy_close_and_native_diagnostics(self):
        loop = sf.SessionLoop(feedback_session(), feedback_config())
        execution = loop.create_execution({"past": scalar(0)}, batches=[{"x": scalar(2)}])
        execution.cancel()
        with self.assertRaises(ort.capi.onnxruntime_pybind11_state.Fail):
            execution.step()
        self.assertTrue(execution.requires_reset())
        with self.assertRaises(ort.capi.onnxruntime_pybind11_state.Fail):
            execution.get_state()
        with self.assertRaises(ort.capi.onnxruntime_pybind11_state.Fail):
            execution.run()
        execution.reset()
        self.assertFalse(execution.requires_reset())
        with execution._operation():
            with self.assertRaises(RuntimeError):
                execution.step()
            with self.assertRaises(RuntimeError):
                execution.close()
            with self.assertRaises(RuntimeError):
                execution.requires_reset()
            execution.cancel()
        execution.reset()
        with self.assertRaises(ort.capi.onnxruntime_pybind11_state.Fail):
            execution.get_dispatch_info(sf.ProviderFeature.CACHE, sf.ProviderOperation.CACHE_INITIALIZE)
        with self.assertRaises(ValueError):
            execution.get_staging_info(-1)
        with self.assertRaises(ValueError):
            execution.get_dispatch_info(sf.ProviderFeature.CACHE, sf.ProviderOperation.CACHE_INITIALIZE, -1)
        self.assertEqual(execution.run().get_outputs()["present"].numpy().item(), 2)
        execution.close()
        with self.assertRaises(RuntimeError):
            execution.cancel()
        with self.assertRaises(RuntimeError):
            execution.__enter__()
        loop.close()

    def test_result_delivery_failure_after_advancement_requires_reset(self):
        with sf.SessionLoop(feedback_session(), feedback_config()) as loop:
            execution = loop.create_execution({"past": scalar(0)}, batches=[{"x": scalar(2)}, {"x": scalar(3)}])
        first = execution.step().get_outputs()["present"]
        with (
            mock.patch.object(sf, "StepResult", side_effect=MemoryError("controlled result allocation failure")),
            self.assertRaises(MemoryError),
        ):
            execution.step()
        self.assertTrue(execution.requires_reset())
        # Native counters include the initial feedback binding overwritten on the second Step.
        self.assertEqual(execution.get_staging_info().cpu_input_request_bytes, 20)
        self.assertEqual(execution.get_staging_info().cpu_snapshot_copy_bytes, 8)
        with self.assertRaises(ort.capi.onnxruntime_pybind11_state.Fail):
            execution.step()
        self.assertEqual(first.numpy().item(), 2)
        execution.reset()
        self.assertEqual(execution.run().get_outputs()["present"].numpy().item(), 5)
        self.assertEqual(first.numpy().item(), 2)
        execution.close()

    def test_invalid_inputs_are_not_silently_copied_or_coerced(self):
        loop = sf.SessionLoop(feedback_session(), feedback_config())
        invalid = (
            [1, 2],
            np.ones((2, 2), dtype=np.float32)[:, 0],
            np.array("x"),
            np.array([object()], dtype=object),
            np.array(1, dtype=">f4"),
        )
        for value in invalid:
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                loop.create_execution({"past": value}, batches=[{"x": scalar(2)}])
        with self.assertRaises(TypeError):
            loop.create_execution([scalar(0)])
        with self.assertRaises(TypeError):
            loop.create_execution({"past": scalar(0)}, batches="x")
        with self.assertRaises(TypeError):
            loop.create_execution({"past": scalar(0)}, batches=[None])
        with self.assertRaises(ValueError):
            loop.create_execution({"": scalar(0)})
        with self.assertRaises(TypeError):
            sf.SessionLoop(object(), feedback_config())
        with self.assertRaises(TypeError):
            sf.SessionLoop(feedback_session(), object())
        with loop.create_execution({"past": scalar(0)}, batches=[{"x": scalar(2)}]) as execution:
            self.assertEqual(execution.run().get_outputs()["present"].numpy().item(), 2)
        with self.assertRaises(RuntimeError):
            execution.reset()

    def test_programmatic_cache_bindings_in_loops_and_session_components(self):
        session = tiny_session(
            helper.make_graph(
                [helper.make_node("Concat", ["past", "x"], ["present"], axis=0)],
                "managed_cache",
                [
                    helper.make_tensor_value_info("x", TensorProto.FLOAT, [1]),
                    helper.make_tensor_value_info("past", TensorProto.FLOAT, ["P"]),
                ],
                [helper.make_tensor_value_info("present", TensorProto.FLOAT, ["T"])],
            )
        )
        binding = sf.CacheBinding(sf.CacheSpec("key", sf.ElementType.FLOAT32, (0,), 0), "past", "present")
        first = np.array([2], dtype=np.float32)
        second = np.array([3], dtype=np.float32)
        with sf.SessionLoop(
            session,
            sf.LoopConfig(
                ("present",),
                (sf.ExitCondition(sf.ExitKind.END_OF_INPUTS),),
                finite_input_names=("x",),
                kv_caches=(binding,),
            ),
        ) as loop:
            execution = loop.create_execution(batches=[{"x": first}, {"x": second}])
        retained = execution.step().get_outputs()["present"]
        np.testing.assert_array_equal(execution.run().get_outputs()["present"].numpy(), [2, 3])
        np.testing.assert_array_equal(
            execution.get_state().get_outputs(sf.OutputGroup.STATE_VALUES)["key"].numpy(), [2, 3]
        )
        execution.reset()
        np.testing.assert_array_equal(execution.run().get_outputs()["present"].numpy(), [2, 3])
        execution.close()
        np.testing.assert_array_equal(retained.numpy(), [2])
        components = tuple(
            sf.ChainComponent(
                name,
                session,
                (sf.InputBinding("x", sf.ValueSource(sf.SourceKind.EXTERNAL_INPUT, name)),),
                kv_caches=(binding,),
            )
            for name in ("first", "second")
        )
        with sf.SessionChain(sf.ChainConfig(components, (sf.ChainOutput("final", 1, "present"),))) as chain:
            execution = chain.create_execution({"first": first, "second": second})
        np.testing.assert_array_equal(execution.run().get_outputs()["final"].numpy(), [2, 3])
        execution.close()

    def test_chain_finite_sources_named_state_and_retained_value_feeds(self):
        with sf.SessionLoop(feedback_session(), feedback_config()) as loop:
            component = sf.ChainComponent(
                "accumulate",
                loop,
                inputs=(
                    sf.InputBinding("x", sf.ValueSource(sf.SourceKind.EXTERNAL_SEQUENCE, "xs")),
                    sf.InputBinding("past", sf.ValueSource(sf.SourceKind.STATE, "sum")),
                ),
                state_outputs=(sf.StateBinding("sum", "present"),),
            )
            config = sf.ChainConfig((component,), (sf.ChainOutput("final", 0, "present"),))
            chain = sf.SessionChain(config)
        initial = ort.OrtValue.ortvalue_from_numpy(scalar(0))
        first = chain.create_execution(sequences={"xs": [scalar(2), scalar(3)]}, initial_state={"sum": initial})
        chain.close()
        result = first.run()
        self.assertEqual(result.get_info().kind, sf.ResultKind.CHAIN)
        self.assertEqual(result.get_info().invocations, 2)
        self.assertEqual(result.get_info().components_completed, 1)
        self.assertEqual(result.get_info().component_name, "accumulate")
        self.assertEqual(result.get_outputs()["final"].numpy().item(), 5)
        state = first.get_state().get_outputs(sf.OutputGroup.STATE_VALUES)["sum"]
        first.close()
        with sf.SessionLoop(feedback_session(), feedback_config()) as consumer:
            second = consumer.create_execution({"past": state}, batches=[{"x": scalar(4)}])
        state.close()
        self.assertEqual(second.run().get_outputs()["present"].numpy().item(), 9)
        second.close()
        with self.assertRaises(TypeError):
            sf.SessionChain(object())
        with self.assertRaises(TypeError):
            sf.SessionChain(sf.ChainConfig((sf.ChainComponent("invalid", None),), config.outputs))

    def test_ordinary_session_components_and_invalid_sequence_inputs(self):
        session = tiny_session(
            helper.make_graph(
                [helper.make_node("Add", ["x", "x"], ["y"])],
                "stateful_double",
                [helper.make_tensor_value_info("x", TensorProto.FLOAT, [])],
                [helper.make_tensor_value_info("y", TensorProto.FLOAT, [])],
            )
        )
        components = (
            sf.ChainComponent(
                "first", session, (sf.InputBinding("x", sf.ValueSource(sf.SourceKind.EXTERNAL_INPUT, "a")),)
            ),
            sf.ChainComponent(
                "second", session, (sf.InputBinding("x", sf.ValueSource(sf.SourceKind.COMPONENT_OUTPUT, "y", 0)),)
            ),
        )
        with sf.SessionChain(sf.ChainConfig(components, (sf.ChainOutput("final", 1, "y"),))) as chain:
            with self.assertRaises(TypeError):
                chain.create_execution(sequences=[])
            with self.assertRaises(TypeError):
                chain.create_execution({"a": scalar(3)}, sequences={"invalid": "sequence"})
            execution = chain.create_execution({"a": scalar(3)})
        first = execution.step()
        self.assertEqual(first.get_outputs(sf.OutputGroup.COMPONENT_OUTPUTS)["y"].numpy().item(), 6)
        final = execution.run()
        self.assertEqual(final.get_info().invocations, 2)
        self.assertEqual(final.get_outputs()["final"].numpy().item(), 12)
        execution.close()
        uninitialized = object.__new__(ort.InferenceSession)
        uninitialized._sess = None
        with self.assertRaises(ValueError):
            sf.SessionLoop(uninitialized, feedback_config())
        with self.assertRaises(ValueError):
            sf.SessionChain(
                sf.ChainConfig((sf.ChainComponent("uninitialized", uninitialized),), (sf.ChainOutput("out", 0, "y"),))
            )

    def test_bootstrap_budget_eos_history_and_sampler_diagnostics(self):
        scores = np.array([[0, 4, 1, -2], [0, 1, 5, -2], [-1, 1, 2, 6], [6, 2, 1, 0]], dtype=np.float32)
        session = tiny_session(
            helper.make_graph(
                [helper.make_node("Gather", ["scores", "token"], ["logits"], axis=0)],
                "stateful_decoder",
                [helper.make_tensor_value_info("token", TensorProto.INT64, [1, "S"])],
                [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, "S", 4])],
                [numpy_helper.from_array(scores, "scores")],
            )
        )
        prefill = sf.SessionLoop(
            session,
            sf.LoopConfig(
                ("logits",),
                (sf.ExitCondition(sf.ExitKind.END_OF_INPUTS),),
                finite_input_names=("token",),
                nonempty_input="token",
            ),
        )
        prompt = np.array([[0, 1]], dtype=np.int64)
        for budget, eos in ((0, False), (1, False), (3, False), (3, True)):
            with self.subTest(budget=budget, eos=eos):
                exits = [sf.ExitCondition(sf.ExitKind.MAX_GENERATED_TOKENS, budget)]
                if eos:
                    exits.append(sf.ExitCondition(sf.ExitKind.EOS, eos_token_ids=(2,)))
                decode = sf.SessionLoop(
                    session,
                    sf.LoopConfig(
                        ("logits",),
                        exits,
                        generation=sf.GenerationConfig(sf.GenerationPhase.DECODE, "token", "logits", logits_position=0),
                    ),
                )
                bootstrap = sf.GenerationBootstrap(
                    sf.ValueSource(sf.SourceKind.COMPONENT_OUTPUT, "logits", 0),
                    sf.ValueSource(sf.SourceKind.EXTERNAL_SEQUENCE, "prompt"),
                    logits_position=1,
                )
                chain = sf.SessionChain(
                    sf.ChainConfig(
                        (
                            sf.ChainComponent(
                                "prefill",
                                prefill,
                                (sf.InputBinding("token", sf.ValueSource(sf.SourceKind.EXTERNAL_SEQUENCE, "prompt")),),
                            ),
                            sf.ChainComponent("decode", decode, generation_bootstrap=bootstrap),
                        ),
                        (sf.ChainOutput("tokens", 1, kind=sf.OutputKind.GENERATED_TOKEN_IDS),),
                    )
                )
                decode.close()
                execution = chain.create_execution(sequences={"prompt": [prompt]})
                chain.close()
                initial = execution.step()
                self.assertTrue(initial.get_info().executed)
                self.assertEqual(initial.get_info().invocations, 1)
                sampled = execution.step()
                self.assertFalse(sampled.get_info().executed)
                self.assertEqual(sampled.get_info().iterations, 0)
                self.assertEqual(sampled.get_info().invocations, 1)
                self.assertEqual(sampled.get_info().generated_token, 2 if budget else None)
                final = sampled if sampled.get_info().completed else execution.run()
                count = min(budget, 1) if eos else budget
                expected = np.array([2, 3, 0][:count], dtype=np.int64)
                value = final.get_outputs()["tokens"]
                np.testing.assert_array_equal(value.numpy(), expected)
                self.assertEqual(final.get_info().invocations, max(1, count))
                info = execution.get_dispatch_info(sf.ProviderFeature.SAMPLER, sf.ProviderOperation.SAMPLE_GREEDY, 1)
                self.assertEqual(info.cpu_calls, count)
                self.assertEqual(info.provider_calls, 0)
                self.assertEqual(info.operation, sf.ProviderOperation.SAMPLE_GREEDY)
                execution.reset()
                np.testing.assert_array_equal(execution.run().get_outputs()["tokens"].numpy(), expected)
                np.testing.assert_array_equal(value.numpy(), expected)
                execution.close()
        prefill.close()

    def test_ddim_native_reference_both_guidance_modes_and_timestep_formats(self):
        for mode in sf.GuidanceMode:
            for format_ in sf.TimestepFormat:
                policy = ddim_config(latent_shape=(2, 1, 1, 3), guidance_mode=mode, timestep_format=format_)
                config = sf.LoopConfig(
                    ("latents",), (sf.ExitCondition(sf.ExitKind.MAX_ITERATIONS, limit=2),), ddim=policy
                )
                with sf.SessionLoop(ddim_session(policy), config) as loop:
                    for seeded in (False, True):
                        with self.subTest(mode=mode, format=format_, seeded=seeded):
                            initial = (
                                np.array(0, dtype=np.uint64)
                                if seeded
                                else np.tile(np.array([0.125, -0.25, 1], dtype=np.float32), 2).reshape(2, 1, 1, 3)
                            )
                            execution = loop.create_execution(
                                {
                                    "init": initial,
                                    "positive": np.full((2, 1, 1, 1), 0.25, dtype=np.float32),
                                    "negative": np.full((2, 1, 1, 1), -0.5, dtype=np.float32),
                                }
                            )
                            first = execution.step().get_outputs()["latents"]
                            before = first.numpy()
                            result = execution.run()
                            self.assertEqual(result.get_info().iterations, 2)
                            self.assertEqual(result.get_info().invocations, 2 if mode == sf.GuidanceMode.BATCHED else 4)
                            final = result.get_outputs()["latents"]
                            self.assertEqual(final.get_info().shape, (2, 1, 1, 3))
                            if not seeded:
                                expected = np.tile(
                                    np.array([-8.022068, -8.661748, -6.5294814], dtype=np.float32), 2
                                ).reshape(2, 1, 1, 3)
                                np.testing.assert_allclose(final.numpy(), expected, rtol=1e-4, atol=1e-4)
                            execution.reset()
                            np.testing.assert_array_equal(
                                execution.run().get_outputs()["latents"].numpy(), final.numpy()
                            )
                            np.testing.assert_array_equal(first.numpy(), before)
                            execution.close()
                invalid = dataclasses.replace(policy, latent_shape=(1, 2, 3))
                with self.assertRaises(ort.capi.onnxruntime_pybind11_state.InvalidArgument):
                    sf.SessionLoop(
                        ddim_session(policy),
                        sf.LoopConfig(("latents",), (sf.ExitCondition(sf.ExitKind.MAX_ITERATIONS, 2),), ddim=invalid),
                    )


if __name__ == "__main__":
    unittest.main()
