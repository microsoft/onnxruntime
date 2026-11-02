# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Read-only Python Model Package factories use the existing native execution engine."""

import dataclasses
import gc
import json
import tempfile
import unittest
import weakref
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnxruntime_test_python_stateful import ddim_config, scalar

import onnxruntime as ort
from onnxruntime import stateful as sf


def double_model():
    return helper.make_model(
        helper.make_graph(
            [helper.make_node("Add", ["x", "x"], ["y"])],
            "package_double",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [])],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=10,
    )


def package_files(root, models, chains=None):
    root = Path(root)
    manifest = {"schema_version": "1.1", "components": {}}
    for name, model in models.items():
        relative = f"{name}/cpu"
        directory = root / relative
        directory.mkdir(parents=True)
        onnx.save_model(model, directory / "model.onnx")
        manifest["components"][name] = {
            "variants": {
                "cpu": {
                    "variant_directory": relative,
                    "ep": "CPUExecutionProvider",
                    "device": "cpu",
                    "executor_info": {"ort": {"model_file": "model.onnx"}},
                }
            }
        }
    if chains is not None:
        manifest["chains"] = chains
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def single_chain():
    return {
        "steps": [
            {
                "id": "double",
                "component": "model",
                "executor_info": {
                    "ort": {
                        "kind": "session",
                        "inputs": {"x": {"kind": "external_input", "name": "input"}},
                    }
                },
            }
        ],
        "executor_info": {"ort": {"outputs": {"output": {"kind": "model_output", "step": "double", "output": "y"}}}},
    }


def options():
    result = ort.SessionOptions()
    result.intra_op_num_threads = 1
    result.inter_op_num_threads = 1
    return result


def denoiser_model(policy):
    batch = policy.latent_shape[0] * (2 if policy.guidance_mode == sf.GuidanceMode.BATCHED else 1)
    shape = [batch, *policy.latent_shape[1:]]
    graph = helper.make_graph(
        [
            helper.make_node("Mul", ["sample", "latent_scale"], ["scaled_latent"]),
            helper.make_node("Mul", ["time", "time_scale"], ["scaled_time"]),
            helper.make_node("Add", ["scaled_latent", "scaled_time"], ["time_prediction"]),
            helper.make_node("Add", ["time_prediction", "conditioning"], ["epsilon"]),
        ],
        "package_ddim",
        [
            helper.make_tensor_value_info("sample", TensorProto.FLOAT, shape),
            helper.make_tensor_value_info("time", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("conditioning", TensorProto.FLOAT, [batch, 1, 1, 1]),
        ],
        [helper.make_tensor_value_info("epsilon", TensorProto.FLOAT, shape)],
        [
            numpy_helper.from_array(scalar(0.125), "latent_scale"),
            numpy_helper.from_array(scalar(0.0001), "time_scale"),
        ],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)


def ddim_chain(policy):
    config = dataclasses.asdict(policy)
    config["guidance_mode"] = "batched" if policy.guidance_mode == sf.GuidanceMode.BATCHED else "sequential"
    config["timestep_format"] = "float32_scalar"
    return {
        "steps": [
            {
                "id": "denoise",
                "component": "denoiser",
                "executor_info": {
                    "ort": {
                        "kind": "session_loop",
                        "inputs": {
                            name: {"kind": "external_input", "name": name} for name in ("init", "positive", "negative")
                        },
                        "loop": {
                            "ddim": config,
                            "exit_conditions": [{"kind": "max_iterations", "limit": policy.num_inference_steps}],
                        },
                    }
                },
            }
        ],
        "executor_info": {
            "ort": {"outputs": {"latents": {"kind": "model_output", "step": "denoise", "output": "latents"}}}
        },
    }


class TestStatefulPackage(unittest.TestCase):
    def test_inspection_selected_component_and_adopted_ordinary_session(self):
        with tempfile.TemporaryDirectory() as directory:
            root = package_files(Path(directory) / "pkg", {"model": double_model()})
            package = sf.ModelPackage(root)
            info = package.get_info()
            self.assertEqual(info.schema_version_major, 1)
            self.assertEqual(info.components[0].name, "model")
            self.assertEqual(info.components[0].variants[0].name, "cpu")
            self.assertEqual(info.components[0].variants[0].execution_provider, "CPUExecutionProvider")
            with self.assertRaises(dataclasses.FrozenInstanceError):
                info.schema_version_major = 3
            resolved = package.resolve_string_ref("model/cpu/model.onnx")
            self.assertEqual(Path(resolved), root / "model/cpu/model.onnx")
            self.assertEqual(Path(package.resolve_string_ref("missing", must_exist=False)), root / "missing")
            component = package.select_component("model", options())
            package.close()
            selected = component.get_info()
            self.assertEqual(selected.variant_name, "cpu")
            self.assertEqual(Path(selected.variant_folder), root / "model/cpu")
            session = component.create_session()
            self.assertIsInstance(session, ort.InferenceSession)
            component.close()
            gc.collect()
            np.testing.assert_array_equal(session.run(None, {"x": scalar(3)})[0], scalar(6))
            self.assertFalse(session._enable_fallback)
            self.assertEqual(Path(session._model_path), root / "model/cpu/model.onnx")
            self.assertEqual(session.get_inputs()[0].name, "x")
            self.assertEqual(session.get_outputs()[0].name, "y")
            session.set_providers(["CPUExecutionProvider"])
            np.testing.assert_array_equal(session.run(None, {"x": scalar(4)})[0], scalar(8))
            self.assertEqual(info.components[0].name, "model")

    def test_named_chain_options_owned_sessions_and_parent_release(self):
        with tempfile.TemporaryDirectory() as directory:
            root = package_files(Path(directory) / "pkg", {"model": double_model()}, {"double": single_chain()})
            with sf.ModelPackage(str(root)) as package:
                chain = package.create_chain("double", options(), step_session_options={"double": options()})
            with chain:
                execution = chain.create_execution({"input": scalar(3)})
            result = execution.run()
            value = result.get_outputs()["output"]
            self.assertEqual(result.get_info().invocations, 1)
            self.assertEqual(value.numpy().item(), 6)
            execution.reset()
            self.assertEqual(execution.run().get_outputs()["output"].numpy().item(), 6)
            execution.close()
            result.close()
            gc.collect()
            self.assertEqual(value.numpy().item(), 6)

    def test_native_ddim_package_reference_and_reset_for_both_guidance_modes(self):
        for mode in sf.GuidanceMode:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                policy = ddim_config(latent_shape=(1, 1, 1, 3), guidance_mode=mode)
                root = package_files(
                    Path(directory) / "pkg", {"denoiser": denoiser_model(policy)}, {"denoise": ddim_chain(policy)}
                )
                with sf.ModelPackage(root) as package:
                    chain = package.create_chain("denoise")
                for seeded in (False, True):
                    with (
                        self.subTest(seeded=seeded),
                        chain.create_execution(
                            {
                                "init": np.array(0, dtype=np.uint64)
                                if seeded
                                else np.array([0.125, -0.25, 1], dtype=np.float32).reshape(1, 1, 1, 3),
                                "positive": np.full((1, 1, 1, 1), 0.25, dtype=np.float32),
                                "negative": np.full((1, 1, 1, 1), -0.5, dtype=np.float32),
                            }
                        ) as execution,
                    ):
                        result = execution.run()
                        self.assertEqual(result.get_info().invocations, 2 if mode == sf.GuidanceMode.BATCHED else 4)
                        value = result.get_outputs()["latents"]
                        if not seeded:
                            expected = np.array([-8.022068, -8.661748, -6.5294814], dtype=np.float32)
                            np.testing.assert_allclose(value.numpy().ravel(), expected, rtol=1e-4, atol=1e-4)
                        execution.reset()
                        np.testing.assert_array_equal(execution.run().get_outputs()["latents"].numpy(), value.numpy())
                chain.close()

    def test_options_callback_snapshots_survive_options_and_component_release(self):
        class Delegate:
            def __init__(self):
                self.calls = 0

            def __call__(self, devices, model_metadata, runtime_metadata, maximum):
                self.calls += 1
                return [device for device in devices if device.ep_name == "CPUExecutionProvider"][:maximum]

        with tempfile.TemporaryDirectory() as directory:
            root = package_files(Path(directory) / "pkg", {"model": double_model()})
            callback = Delegate()
            reference = weakref.ref(callback)
            selected_options = options()
            selected_options.set_provider_selection_policy_delegate(callback)
            with sf.ModelPackage(root) as package:
                component = package.select_component("model", selected_options)
            session = component.create_session(selected_options)
            self.assertGreater(callback.calls, 0)
            component.close()
            del component, selected_options, callback
            gc.collect()
            self.assertIsNotNone(reference())
            np.testing.assert_array_equal(session.run(None, {"x": scalar(7)})[0], scalar(14))
            with sf.SessionLoop(
                session,
                sf.LoopConfig(("y",), (sf.ExitCondition(sf.ExitKind.MAX_ITERATIONS, 1),)),
            ) as loop:
                execution = loop.create_execution({"x": scalar(2)})
            del session
            gc.collect()
            self.assertEqual(execution.run().get_outputs()["y"].numpy().item(), 4)
            execution.close()

    def test_context_read_callback_snapshot_survives_adoption_and_option_cloning(self):
        class Reader:
            def __call__(self, _name, _output):
                raise AssertionError("An ordinary CPU model must not request EPContext data")

        for override in (False, True):
            with self.subTest(override=override), tempfile.TemporaryDirectory() as directory:
                root = package_files(Path(directory) / "pkg", {"model": double_model()})
                reader = Reader()
                reference = weakref.ref(reader)
                selected = options()
                provided = options()
                configured = provided if override else selected
                configured.set_ep_context_data_read_func(reader, 16)
                with sf.ModelPackage(root) as package, package.select_component("model", selected) as component:
                    session = component.create_session(provided if override else None)
                cloned = session.get_session_options()
                configured.clear_ep_context_data_read_func()
                del package, component, selected, provided, configured, reader
                gc.collect()
                self.assertIsNotNone(reference())
                np.testing.assert_array_equal(session.run(None, {"x": scalar(3)})[0], scalar(6))
                del session
                gc.collect()
                self.assertEqual(reference() is not None, override)
                cloned.clear_ep_context_data_read_func()
                del cloned
                gc.collect()
                self.assertIsNone(reference())

    def test_argument_and_native_factory_errors_are_explicit(self):
        with self.assertRaises(TypeError):
            sf.ModelPackage(b"not-a-text-path")
        with tempfile.TemporaryDirectory() as directory:
            root = package_files(Path(directory) / "pkg", {"model": double_model()}, {"double": single_chain()})
            with sf.ModelPackage(root) as package:
                for name in ("", "missing"):
                    with (
                        self.subTest(name=name),
                        self.assertRaises((ValueError, ort.capi.onnxruntime_pybind11_state.InvalidArgument)),
                    ):
                        package.create_chain(name)
                with self.assertRaises(TypeError):
                    package.create_chain("double", object())
                with self.assertRaises(TypeError):
                    package.create_chain("double", step_session_options=[])
                with self.assertRaises(TypeError):
                    package.create_chain("double", step_session_options={"double": None})
                with self.assertRaises(TypeError):
                    package.select_component("model", object())
                with self.assertRaises(TypeError):
                    package.resolve_string_ref("file", base_dir=42)
                with self.assertRaises(TypeError):
                    package.resolve_string_ref("file", must_exist=1)
                with self.assertRaises(ort.capi.onnxruntime_pybind11_state.InvalidArgument):
                    package.create_chain("double", step_session_options={"unknown": options()})
                with package.select_component("model") as component, self.assertRaises(TypeError):
                    component.create_session(object())
                with self.assertRaises(RuntimeError):
                    component.get_info()


if __name__ == "__main__":
    unittest.main()
