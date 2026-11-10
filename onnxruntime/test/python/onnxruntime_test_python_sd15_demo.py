# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Offline CPU SD pipeline fixtures, adapters and checked image-output tests."""

import hashlib
import importlib.util
import io
import json
import os
import runpy
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

_HAS_PILLOW = importlib.util.find_spec("PIL") is not None
Image = importlib.import_module("PIL.Image") if _HAS_PILLOW else None
_HAS_REFERENCE = _HAS_PILLOW and all(
    importlib.util.find_spec(name) is not None for name in ("diffusers", "torch", "transformers")
)

_PATH = Path(
    os.environ.get("ORT_STATEFUL_SD15_DEMO", Path(__file__).resolve().parents[3] / "samples/python/sd15_demo.py")
)
_SPEC = importlib.util.spec_from_file_location("sd15_demo", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
demo = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = demo
with mock.patch.object(sys, "path", [str(_PATH.parent), *sys.path]):
    _SPEC.loader.exec_module(demo)
models = sys.modules["sd15_models"]


def _constant(name, data):
    return helper.make_node("Constant", [], [name], value=numpy_helper.from_array(np.asarray(data)))


def _save(path, nodes, inputs, outputs):
    graph = helper.make_graph(nodes, "tiny_sd15", inputs, outputs)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, path)


def _info(name, dtype, shape):
    return helper.make_tensor_value_info(name, dtype, shape)


class FakeTokenizer:
    def encode(self, text):
        return np.full((1, 77), 2 if text else 1, dtype=np.int32)


@contextmanager
def toy_assets(unsafe=False):
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        originals = root / "originals"
        _save(
            originals / "text_encoder/model.onnx",
            [
                _constant("axis", np.array([2], dtype=np.int64)),
                _constant("shape", np.array([1, 77, 768], dtype=np.int64)),
                helper.make_node("Cast", ["input_ids"], ["ids_float"], to=TensorProto.FLOAT),
                helper.make_node("Unsqueeze", ["ids_float", "axis"], ["expanded"]),
                helper.make_node("Expand", ["expanded", "shape"], ["hidden"]),
                helper.make_node("Cast", ["hidden"], ["last_hidden_state"], to=TensorProto.FLOAT16),
            ],
            [_info("input_ids", TensorProto.INT32, [1, 77])],
            [_info("last_hidden_state", TensorProto.FLOAT16, [1, 77, 768])],
        )
        _save(
            originals / "unet/model.onnx",
            [
                _constant("sample_scale", np.array(0.1, dtype=np.float32)),
                _constant("time_scale", np.array(0.0001, dtype=np.float32)),
                _constant("axes", np.array([1, 2], dtype=np.int64)),
                _constant("last_axis", np.array([3], dtype=np.int64)),
                helper.make_node("Cast", ["sample"], ["sample_float"], to=TensorProto.FLOAT),
                helper.make_node("Cast", ["timestep"], ["time_float"], to=TensorProto.FLOAT),
                helper.make_node("Cast", ["encoder_hidden_states"], ["condition_float"], to=TensorProto.FLOAT),
                helper.make_node("Mul", ["sample_float", "sample_scale"], ["scaled_sample"]),
                helper.make_node("Mul", ["time_float", "time_scale"], ["scaled_time"]),
                helper.make_node("ReduceMean", ["condition_float", "axes"], ["condition_mean"], keepdims=1),
                helper.make_node("Unsqueeze", ["condition_mean", "last_axis"], ["condition_bias"]),
                helper.make_node("Add", ["scaled_sample", "scaled_time"], ["time_prediction"]),
                helper.make_node("Add", ["time_prediction", "condition_bias"], ["prediction"]),
                helper.make_node("Cast", ["prediction"], ["out_sample"], to=TensorProto.FLOAT16),
            ],
            [
                _info("sample", TensorProto.FLOAT16, ["B", 4, 64, 64]),
                _info("timestep", TensorProto.FLOAT16, [1]),
                _info("encoder_hidden_states", TensorProto.FLOAT16, ["B", 77, 768]),
            ],
            [_info("out_sample", TensorProto.FLOAT16, ["B", 4, 64, 64])],
        )
        _save(
            originals / "vae_decoder/model.onnx",
            [
                _constant("starts", np.array([0], dtype=np.int64)),
                _constant("ends", np.array([3], dtype=np.int64)),
                _constant("axes", np.array([1], dtype=np.int64)),
                _constant("scales", np.array([1, 1, 8, 8], dtype=np.float32)),
                helper.make_node("Cast", ["latent_sample"], ["latents_float"], to=TensorProto.FLOAT),
                helper.make_node("Slice", ["latents_float", "starts", "ends", "axes"], ["rgb_latents"]),
                helper.make_node("Resize", ["rgb_latents", "", "scales"], ["large"], mode="nearest"),
                helper.make_node("Tanh", ["large"], ["pixels"]),
                helper.make_node("Cast", ["pixels"], ["sample"], to=TensorProto.FLOAT16),
            ],
            [_info("latent_sample", TensorProto.FLOAT16, [1, 4, 64, 64])],
            [_info("sample", TensorProto.FLOAT16, [1, 3, 512, 512])],
        )
        _save(
            originals / "safety_checker/model.onnx",
            [
                helper.make_node("Identity", ["images"], ["out_images"]),
                _constant("has_nsfw_concepts", np.array([unsafe], dtype=np.bool_)),
            ],
            [
                _info("clip_input", TensorProto.FLOAT16, [1, 3, 224, 224]),
                _info("images", TensorProto.FLOAT16, [1, 512, 512, 3]),
            ],
            [
                _info("out_images", TensorProto.FLOAT16, [1, 512, 512, 3]),
                _info("has_nsfw_concepts", TensorProto.BOOL, [1]),
            ],
        )
        for folder in ("tokenizer", "feature_extractor"):
            (originals / folder).mkdir()
            (originals / folder / "config.json").write_text("{}")
        (originals / "feature_extractor/preprocessor_config.json").write_text(
            json.dumps(
                {
                    "do_resize": True,
                    "size": {"shortest_edge": 224},
                    "resample": 3,
                    "do_center_crop": True,
                    "crop_size": {"height": 224, "width": 224},
                    "do_rescale": True,
                    "rescale_factor": 1 / 255,
                    "do_normalize": True,
                    "image_mean": [0.48145466, 0.4578275, 0.40821073],
                    "image_std": [0.26862954, 0.26130258, 0.27577711],
                    "do_convert_rgb": True,
                }
            )
        )
        (originals / "LICENSE.txt").write_text("fixture license")
        asset_list = tuple(
            demo.Asset(
                str(path.relative_to(originals)), path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest()
            )
            for path in originals.rglob("*")
            if path.is_file()
        )
        original_manifest = demo.manifest

        def cpu_manifest(*args):
            value = original_manifest(*args)
            for component in value["components"].values():
                for variant in component["variants"].values():
                    variant["ep"] = "CPUExecutionProvider"
                    variant["device"] = "cpu"
            return value

        def cpu_options(*_args, **_kwargs):
            options = demo.ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            return options

        with (
            mock.patch.object(demo, "ASSETS", asset_list),
            mock.patch.object(demo, "session_options", side_effect=cpu_options),
            mock.patch.object(demo, "manifest", side_effect=cpu_manifest),
        ):
            yield root


@unittest.skipUnless(_HAS_PILLOW, "SD image fixtures require optional Pillow; no dependency is installed implicitly")
class TestSD15Models(unittest.TestCase):
    def test_safety_image_axes_do_not_alias_clip_feature_dimensions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, destination = root / "source.onnx", root / "adapted.onnx"
            _save(
                source,
                [helper.make_node("Identity", ["images"], ["out_images"])],
                [
                    _info("clip_input", TensorProto.FLOAT16, [1, 3, "height", "width"]),
                    _info("images", TensorProto.FLOAT16, [1, "height", "width", 3]),
                ],
                [_info("out_images", TensorProto.FLOAT16, [1, "out_height", "out_width", 3])],
            )
            report = models.boundary_model(source, destination, independent_image_axes=True)
            self.assertTrue(report["independent_image_axes"])
            wrapped = onnx.load(destination)
            self.assertEqual(wrapped.graph.input[0].type.tensor_type.shape.dim[2].dim_param, "height")
            self.assertEqual(wrapped.graph.input[1].type.tensor_type.shape.dim[1].dim_param, "__sd15_image_height")
            options = demo.ort.SessionOptions()
            options.intra_op_num_threads = 1
            session = demo.ort.InferenceSession(str(destination), options, providers=["CPUExecutionProvider"])
            image = np.zeros((1, 32, 32, 3), dtype=np.float32)
            actual = session.run(None, {"clip_input": np.zeros((1, 3, 14, 14), dtype=np.float32), "images": image})
            np.testing.assert_array_equal(actual[0], image)
            invalid = onnx.load(source)
            invalid.graph.input[1].name = "not_images"
            onnx.save(invalid, source)
            with self.assertRaisesRegex(ValueError, "rank-4 images"):
                models.boundary_model(source, destination, independent_image_axes=True)

    def test_boundary_declarations_custom_domain_identity_and_cleanup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, destination = root / "source.onnx", root / "result/model.onnx"
            _save(
                source,
                [helper.make_node("Identity", ["x"], ["y"])],
                [_info("x", TensorProto.FLOAT16, [1])],
                [_info("y", TensorProto.FLOAT16, [1])],
            )
            original = onnx.load(source)
            original.graph.value_info.extend(
                [_info("x", TensorProto.FLOAT16, [1]), _info("y", TensorProto.FLOAT16, [1])]
            )
            original.graph.node[0].domain = "com.microsoft"
            onnx.save(original, source)
            report = models.boundary_model(source, destination)
            self.assertTrue(report["preserved"])
            wrapped = onnx.load(destination)
            self.assertIn("com.microsoft", [entry.domain for entry in wrapped.opset_import])
            self.assertEqual([value.name for value in wrapped.graph.value_info], ["__sd15_input_x", "__sd15_output_y"])
            previous = destination.read_bytes()
            with (
                mock.patch.object(models.onnx, "save_model", side_effect=OSError("cannot write")),
                self.assertRaisesRegex(OSError, "cannot write"),
            ):
                models.boundary_model(source, destination)
            self.assertEqual(destination.read_bytes(), previous)
            self.assertEqual(list(destination.parent.iterdir()), [destination])
            with (
                mock.patch.object(
                    models, "initializer_fingerprints", side_effect=[{"original": "expected"}, {"original": "modified"}]
                ),
                self.assertRaisesRegex(ValueError, "initializer changed"),
            ):
                models.boundary_model(source, destination)
            original.graph.initializer.append(numpy_helper.from_array(np.ones(1), "__sd15_unused"))
            onnx.save(original, source)
            with self.assertRaisesRegex(ValueError, "Reserved"):
                models.boundary_model(source, destination)

    def test_fixed_point_bicubic_matches_pillow_and_onnx(self):
        assert Image is not None
        rng = np.random.default_rng(42)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "preprocess.onnx"
            models.safety_preprocess_model(path, image_size=32, crop_size=14)
            options = demo.ort.SessionOptions()
            options.intra_op_num_threads = 1
            session = demo.ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
            sample = rng.uniform(-2, 2, (1, 3, 32, 32)).astype(np.float32)
            images, features = session.run(None, {"sample": sample})
            expected_images = np.clip(sample / 2 + 0.5, 0, 1).transpose(0, 2, 3, 1)
            resized = np.asarray(
                Image.fromarray(np.round(expected_images[0] * 255).astype(np.uint8)).resize(
                    (14, 14), Image.Resampling.BICUBIC
                )
            )
            expected = (resized.astype(np.float64) / 255).astype(np.float32).transpose(2, 0, 1)[None]
            expected = expected - np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)[None, :, None, None]
            expected /= np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)[None, :, None, None]
            np.testing.assert_array_equal(images, expected_images)
            np.testing.assert_array_equal(features, expected)
        for source, target in ((0, 1), (1, 0), (True, 1), (1, 2.0)):
            with self.subTest(source=source, target=target), self.assertRaises(ValueError):
                models.bicubic_coefficients(source, target)

    def test_boundary_preserves_initializers_and_rejects_reserved_values(self):
        with toy_assets() as root:
            source = root / "originals/unet/model.onnx"
            original = onnx.load(source)
            fingerprints = models.initializer_fingerprints(original)
            destination = root / "wrapper.onnx"
            report = models.boundary_model(source, destination)
            self.assertTrue(report["preserved"])
            self.assertEqual(fingerprints, models.initializer_fingerprints(onnx.load(destination)))
            original.graph.node[0].output[0] = "__sd15_collision"
            onnx.save_model(original, source)
            with self.assertRaisesRegex(ValueError, "Reserved"):
                models.boundary_model(source, destination)


class TestSD15Precision(unittest.TestCase):
    def test_fixed_rmse_limits_and_report_only_maximum(self):
        zero = np.zeros(10000, dtype=np.float64)
        for image, limit in ((False, 0.01), (True, 0.005)):
            with self.subTest(image=image, kind="rmse"):
                metrics = demo._check_precision(np.full(zero.shape, limit), zero, image=image)
                self.assertLessEqual(metrics["rmse"], limit)
                with self.assertRaisesRegex(AssertionError, "RMSE"):
                    demo._check_precision(np.full(zero.shape, limit + 1e-7), zero, image=image)
            maximum = zero.copy()
            maximum[0] = 0.2
            with self.subTest(image=image, kind="maximum"):
                metrics = demo._check_precision(maximum, zero, image=image)
                self.assertEqual(metrics["max_absolute_error"], 0.2)
                maximum[0] = 0.4
                metrics = demo._check_precision(maximum, zero, image=image)
                self.assertEqual(metrics["max_absolute_error"], 0.4)
                self.assertLessEqual(metrics["rmse"], limit)

    def test_invalid_metric_inputs_fail_explicitly(self):
        value = np.zeros(1, dtype=np.float32)
        for actual, expected, message in (
            ([], value, "NumPy"),
            (value, np.zeros(2, dtype=np.float32), "shapes"),
            (value[:0], value[:0], "nonempty"),
            (value.astype(np.int32), value, "floating-point"),
            (value, value.astype(np.int32), "floating-point"),
            (value + np.nan, value, "finite"),
            (value, value + np.inf, "finite"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(AssertionError, message):
                demo._check_precision(actual, expected)


@unittest.skipUnless(_HAS_PILLOW, "SD image fixtures require optional Pillow; no dependency is installed implicitly")
class TestSD15Pipeline(unittest.TestCase):
    def test_typed_package_supplied_latents_isolation_cancel_and_reset(self):
        qualification = importlib.import_module("onnxruntime_test_python_sd15_demo_assets")

        with toy_assets() as root:
            package = demo.build_package(root, steps=20)
            initial = np.zeros((1, 4, 64, 64), dtype=np.float32)
            with (
                demo.create_execution(
                    package, "positive", adapter=FakeTokenizer(), initial_latents=initial
                ) as packaged,
                qualification.programmatic_execution(
                    package, "positive", adapter=FakeTokenizer(), initial_latents=initial, demo_module=demo
                ) as typed,
            ):
                initial.fill(99)
                for _ in range(2):
                    with packaged.step():
                        pass
                with packaged.step() as update:
                    retained = update.get_outputs(demo.sf.OutputGroup.COMPONENT_OUTPUTS)["latents"]
                    snapshot = retained.numpy()
                packaged.cancel()
                with self.assertRaises(demo.ort.capi.onnxruntime_pybind11_state.Fail):
                    packaged.run()
                self.assertTrue(packaged.requires_reset())
                with self.assertRaises(demo.ort.capi.onnxruntime_pybind11_state.Fail):
                    packaged.step()
                packaged.reset()
                self.assertFalse(packaged.requires_reset())
                with packaged.run() as first, typed.run() as second:
                    self.assertEqual(first.get_info().invocations, 25)
                    for name in ("image", "unsafe", "latents"):
                        np.testing.assert_array_equal(
                            first.get_outputs()[name].numpy(), second.get_outputs()[name].numpy()
                        )
                np.testing.assert_array_equal(retained.numpy(), snapshot)
            np.testing.assert_array_equal(retained.numpy(), snapshot)
            retained.close()

    @unittest.skipUnless(_HAS_REFERENCE, "Independent reference requires optional Diffusers/Torch/Transformers")
    def test_native_cpu_ddim_matches_independent_scheduler_and_safety_reference(self):
        with toy_assets() as root:
            package = demo.build_package(root, steps=2)
            report = demo.validate_generation(package, "positive", root / "reference.png", adapter=FakeTokenizer())
            self.assertFalse(report["unsafe"])
            self.assertTrue(report["reset_replay_verified"])
            self.assertEqual(report["updates"], 2)
            self.assertEqual(report["invocations"], 7)
            self.assertTrue(Path(report["output"]).is_file())
            self.assertTrue(report["supplied_initialization_verified"])
            self.assertEqual(len(report["update_errors"]), 2)
            self.assertLessEqual(report["maximum_latent_rmse"], 0.01)
            self.assertLessEqual(report["image_errors"]["rmse"], 0.005)
            for name in ("reset_replay_errors", "supplied_initialization_errors"):
                self.assertLessEqual(report[name]["latents"]["rmse"], 0.01)
                self.assertEqual(len(report[name]["image_tensor_sha256"]), 64)

    @unittest.skipUnless(_HAS_REFERENCE, "Independent reference requires optional Diffusers/Torch/Transformers")
    def test_reference_sequential_guidance(self):
        with toy_assets() as root:
            package = demo.build_package(root, steps=1, sequential=True)
            with mock.patch.object(demo, "LocalTokenizer", return_value=FakeTokenizer()):
                report = demo.validate_generation(package, "positive", root / "sequential.png")
            self.assertEqual(report["invocations"], 7)

    @unittest.skipUnless(_HAS_REFERENCE, "Independent reference requires optional Diffusers/Torch/Transformers")
    def test_reference_rejects_missing_stages_counts_and_checkpoints(self):
        pixels = np.zeros((1, 512, 512, 3), dtype=np.float32)
        latents = np.zeros((1, 4, 64, 64), dtype=np.float32)

        def value(data):
            return SimpleNamespace(numpy=lambda: data)

        outputs = {"image": value(pixels), "unsafe": value(np.array([False])), "latents": value(latents)}
        for component, completed, invocations, message in (
            ("idle", False, 0, "did not complete"),
            ("idle", True, 0, "update count"),
            ("denoise", True, 0, "invocation accounting"),
            ("denoise", True, 6, "omitted a required"),
        ):
            with self.subTest(message=message), toy_assets() as root:
                package = demo.build_package(root, steps=1)
                execution = mock.MagicMock()
                execution.__enter__.return_value = execution
                result = execution.step.return_value.__enter__.return_value
                result.get_info.return_value = SimpleNamespace(
                    component_name=component, completed=completed, invocations=invocations
                )
                result.get_outputs.return_value = outputs
                scheduler = mock.MagicMock()
                scheduler.timesteps = [1]
                scheduler.step.return_value.prev_sample.numpy.return_value = latents
                with (
                    mock.patch.object(demo.sf.SessionChain, "create_execution", return_value=execution),
                    mock.patch.object(demo, "_scheduler", return_value=scheduler),
                    mock.patch.object(demo, "_prediction", return_value=latents),
                    self.assertRaisesRegex(AssertionError, message),
                ):
                    demo.validate_generation(package, "positive", root / "unused.png", adapter=FakeTokenizer())
                self.assertFalse((root / "unused.png").exists())

    def test_package_rollback_and_pinned_supporting_file_validation(self):
        with toy_assets() as root:
            package = demo.build_package(root, steps=1)
            previous = (package / "manifest.json").read_bytes()
            with (
                mock.patch.object(demo.sf, "ModelPackage", side_effect=ValueError("incompatible")),
                self.assertRaisesRegex(ValueError, "incompatible"),
            ):
                demo.build_package(root, steps=2)
            self.assertEqual((package / "manifest.json").read_bytes(), previous)
            (package / "manifest.json").unlink()
            with (
                mock.patch.object(demo.sf, "ModelPackage", side_effect=ValueError("incompatible")),
                self.assertRaisesRegex(ValueError, "incompatible"),
            ):
                demo.build_package(root, steps=2)
            self.assertFalse((package / "manifest.json").exists())
            target = package / "tokenizer/config.json"
            target.unlink()
            target.write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "Incorrect size"):
                demo.build_package(root, steps=1)

    def test_tokenizer_local_padding_bounds_and_adapter_validation(self):
        tokenizer = mock.Mock()
        tokenizer.encode.return_value = [1, 2]
        tokenizer.pad.return_value = {"input_ids": np.ones((1, 77), dtype=np.int64)}
        module = SimpleNamespace(CLIPTokenizer=SimpleNamespace(from_pretrained=mock.Mock(return_value=tokenizer)))
        with mock.patch.object(demo.importlib, "import_module", return_value=module):
            adapter = demo.LocalTokenizer(Path("/local/tokenizer"))
        module.CLIPTokenizer.from_pretrained.assert_called_once_with("/local/tokenizer", local_files_only=True)
        self.assertEqual(adapter.encode("").dtype, np.int32)
        self.assertEqual(adapter.encode("text").shape, (1, 77))
        with self.assertRaisesRegex(TypeError, "text"):
            adapter.encode(None)
        tokenizer.encode.return_value = list(range(78))
        with self.assertRaisesRegex(ValueError, "exceeds 77"):
            adapter.encode("long")
        for invalid in ([], np.zeros((1, 77), dtype=np.float32), np.zeros((1, 1), dtype=np.int32)):
            with self.subTest(value=type(invalid)), self.assertRaisesRegex(ValueError, "CLIP adapter"):
                demo._execution_inputs(
                    Path("/unused"), "valid", "", 42, SimpleNamespace(encode=lambda _, result=invalid: result)
                )
        with toy_assets() as root:
            package = demo.build_package(root, steps=1)
            data = json.loads((package / "manifest.json").read_text())
            data["additional_metadata"]["sd15_demo"]["revision"] = "wrong"
            (package / "manifest.json").write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "provenance"):
                demo.configuration(package)
            data["additional_metadata"]["sd15_demo"]["revision"] = demo.REVISION
            data["additional_metadata"]["sd15_demo"]["ddim"]["guidance_mode"] = "unknown"
            (package / "manifest.json").write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "DDIM policy"):
                demo.configuration(package)
            data["additional_metadata"]["sd15_demo"]["ddim"]["guidance_mode"] = "batched"
            for option in (None, "DEFAULT"):
                data["additional_metadata"]["sd15_demo"]["cuda_conv_algo_search"] = option
                (package / "manifest.json").write_text(json.dumps(data))
                with self.subTest(option=option), self.assertRaisesRegex(ValueError, "rebuild"):
                    demo.configuration(package)

    def test_device_selection_rejects_invalid_missing_and_ambiguous_ordinals(self):
        for ordinal in (-1, True):
            with self.subTest(ordinal=ordinal), self.assertRaisesRegex(ValueError, "device_id"):
                demo.session_options(True, ordinal)
        self.assertIsInstance(demo.session_options(False), demo.ort.SessionOptions)
        with self.assertRaisesRegex(ValueError, "CPU adapters"):
            demo.session_options(False, 1)
        device = SimpleNamespace(ep_name="CUDAExecutionProvider", memory_info=lambda _: SimpleNamespace(device_id=0))
        with (
            mock.patch.object(demo.ort, "get_ep_devices", return_value=[]),
            self.assertRaisesRegex(ValueError, "unavailable"),
        ):
            demo.session_options(True)
        with (
            mock.patch.object(demo.ort, "get_ep_devices", return_value=[device, device]),
            self.assertRaisesRegex(ValueError, "ambiguous"),
        ):
            demo.session_options(True)
        options = mock.Mock()
        with (
            mock.patch.object(demo.ort, "get_ep_devices", return_value=[device]),
            mock.patch.object(demo.ort, "SessionOptions", return_value=options),
        ):
            self.assertIs(demo.session_options(True), options)
        options.add_provider_for_devices.assert_called_once_with([device], {"cudnn_conv_algo_search": "HEURISTIC"})

    def test_asset_entry_points_and_cli_remain_explicit_and_offline(self):
        with toy_assets() as root:
            with mock.patch.object(demo, "acquire_file", return_value="verified") as acquire:
                self.assertEqual(len(demo.acquire_assets(root)), len(demo.ASSETS))
            self.assertTrue(any("stable-diffusion-license" in call.args[2] for call in acquire.call_args_list))
            commands = (
                ("acquire", "acquire_assets", {}),
                ("build", "build_package", root / "package"),
                ("run", "generate", {"ok": True}),
                ("validate", "validate_generation", {"ok": True}),
            )
            for command, function, response in commands:
                args = ["--assets", str(root), command]
                if command in ("run", "validate"):
                    args += ["--prompt", "positive", "--output", str(root / "image.png")]
                with (
                    self.subTest(command=command),
                    mock.patch.object(demo, function, return_value=response),
                    mock.patch.object(sys, "stdout", io.StringIO()),
                ):
                    demo.main(args)
        with (
            mock.patch.object(sys, "path", [str(_PATH.parent), *sys.path]),
            mock.patch.object(sys, "argv", [str(_PATH), "acquire"]),
            mock.patch("stateful_demo_assets.acquire_file", return_value="verified"),
            mock.patch.object(sys, "stdout", io.StringIO()),
        ):
            runpy.run_path(str(_PATH), run_name="__main__")

    def test_cpu_native_pipeline_image_seed_state_and_replay(self):
        assert Image is not None
        for sequential in (False, True):
            with self.subTest(sequential=sequential), toy_assets() as root:
                package = demo.build_package(root, steps=2, sequential=sequential)
                execution = demo.create_execution(package, "positive", adapter=FakeTokenizer())
                with execution:
                    first = execution.run()
                    pixels = first.get_outputs()["image"].numpy()
                    unsafe = first.get_outputs()["unsafe"].numpy()
                    self.assertFalse(unsafe[0])
                    self.assertEqual(pixels.shape, (1, 512, 512, 3))
                    output = demo.save_image(pixels, unsafe, root / "image.png")
                    with Image.open(output) as image:
                        self.assertEqual(image.size, (512, 512))
                        self.assertEqual(image.mode, "RGB")
                    execution.reset()
                    replay = execution.run()
                    np.testing.assert_array_equal(replay.get_outputs()["image"].numpy(), pixels)
                    self.assertEqual(replay.get_info().invocations, 5 + (4 if sequential else 2))
                    first.close()
                    replay.close()

    def test_safety_block_and_failed_run_never_write_images(self):
        with toy_assets(unsafe=True) as root:
            package = demo.build_package(root, steps=1)
            output = root / "blocked.png"
            with self.assertRaisesRegex(ValueError, "blocked"):
                demo.generate(package, "positive", output, adapter=FakeTokenizer())
            self.assertFalse(output.exists())
        with toy_assets() as root:
            package = demo.build_package(root, steps=1)
            with mock.patch.object(demo, "LocalTokenizer", return_value=FakeTokenizer()):
                result = demo.generate(package, "positive", root / "valid.png")
            self.assertFalse(result["unsafe"])
            self.assertTrue(Path(result["output"]).is_file())
            with self.assertRaises(FileExistsError):
                demo.generate(package, "positive", root / "valid.png", adapter=FakeTokenizer())

    def test_nonfinite_model_prediction_stops_before_decode_and_requires_reset(self):
        with toy_assets() as root:
            package = demo.build_package(root, steps=2)
            initial = np.full((1, 4, 64, 64), 1e10, dtype=np.float32)
            with demo.create_execution(
                package, "positive", adapter=FakeTokenizer(), initial_latents=initial
            ) as execution:
                for expected in ("positive", "negative"):
                    with execution.step() as result:
                        self.assertEqual(result.get_info().component_name, expected)
                with self.assertRaises(demo.ort.capi.onnxruntime_pybind11_state.Fail):
                    execution.run()
                self.assertTrue(execution.requires_reset())
                with self.assertRaises(demo.ort.capi.onnxruntime_pybind11_state.Fail):
                    execution.get_state()
                execution.reset()
                self.assertFalse(execution.requires_reset())
            self.assertFalse((root / "unused.png").exists())

    def test_policy_seed_prompt_and_image_errors_are_explicit(self):
        for steps, guidance, sequential in (
            (0, 7.5, False),
            (1000, 1, False),
            (True, 1, False),
            (1, float("nan"), False),
            (1, -1, False),
            (1, True, False),
            (1, 1, 1),
        ):
            with self.subTest(steps=steps, guidance=guidance), self.assertRaises((ValueError, TypeError)):
                demo.ddim_policy(steps, guidance, sequential)
        with toy_assets() as root:
            package = demo.build_package(root, steps=1)
            for prompt, negative, seed in (("", "", 42), ("x", None, 42), ("x", "", -1), ("x", "", True)):
                with self.subTest(prompt=prompt, seed=seed), self.assertRaises((ValueError, TypeError)):
                    demo.create_execution(package, prompt, negative=negative, seed=seed, adapter=FakeTokenizer())
            with self.assertRaisesRegex(ValueError, "Initial latents"):
                demo.create_execution(
                    package, "x", adapter=FakeTokenizer(), initial_latents=np.zeros(1, dtype=np.float32)
                )
        pixels = np.zeros((1, 512, 512, 3), dtype=np.float32)
        for images, unsafe in (
            (pixels, np.array([0], dtype=np.int32)),
            (pixels, np.array([True])),
            (pixels[0], np.array([False])),
            (pixels + np.nan, np.array([False])),
            (pixels + 2, np.array([False])),
        ):
            with self.subTest(unsafe=unsafe), self.assertRaises(ValueError):
                demo.save_image(images, unsafe, "/unused/output.png")


if __name__ == "__main__":
    unittest.main()
