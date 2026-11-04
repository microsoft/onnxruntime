# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Explicit local CUDA qualification; never downloads or saves unchecked images."""

import ctypes
import gc
import json
import os
import sys
import time
import unittest
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from onnxruntime_test_python_sd15_demo import demo
from PIL import Image

import onnxruntime as ort
from onnxruntime import stateful as sf
from onnxruntime.capi.onnxruntime_pybind11_state import Fail

PROMPT = "A peaceful mountain lake at sunrise, soft golden light, landscape painting"


def assets_root():
    value = os.environ.get("ORT_STATEFUL_SD15_ASSETS")
    if not value or not Path(value).is_dir():
        raise ValueError("Set ORT_STATEFUL_SD15_ASSETS to verified local assets; no download is performed")
    return Path(value)


def sequential_package(root):
    source = root / "package"
    destination = root / "qualification" / f"sequential-{time.time_ns()}"
    destination.mkdir(parents=True)
    for path in source.rglob("*"):
        target = destination / path.relative_to(source)
        if path.is_dir():
            target.mkdir(exist_ok=True)
        elif path.name != "manifest.json":
            target.hardlink_to(path)
    data = json.loads((source / "manifest.json").read_text())
    data["additional_metadata"]["sd15_demo"]["ddim"]["guidance_mode"] = "sequential"
    data["chains"]["image"]["steps"][2]["executor_info"]["ort"]["loop"]["ddim"]["guidance_mode"] = "sequential"
    (destination / "manifest.json").write_text(json.dumps(data, indent=2))
    return destination


def programmatic_execution(package, prompt, *, adapter=None, initial_latents=None, demo_module=demo):
    """Construct the same stages through typed APIs, independently of manifest bindings."""
    demo = demo_module
    config = demo.configuration(package)
    policy = config["ddim"]
    inputs = demo._execution_inputs(package, prompt, "", 42, adapter, initial_latents)
    options = demo.session_options(True, config["device_id"])
    sessions = {
        name: ort.InferenceSession(
            str(package / "components" / name / "cuda/model.onnx"), options, enable_fallback=False
        )
        for name in demo.COMPONENTS
    }
    sessions["preprocess"] = ort.InferenceSession(
        str(package / "components/preprocess/cpu/model.onnx"),
        demo.session_options(False),
        providers=["CPUExecutionProvider"],
        enable_fallback=False,
    )

    def external(port, name):
        return sf.InputBinding(port, sf.ValueSource(sf.SourceKind.EXTERNAL_INPUT, name))

    def previous(port, index, name):
        return sf.InputBinding(port, sf.ValueSource(sf.SourceKind.COMPONENT_OUTPUT, name, index))

    with ExitStack() as stack:
        denoise = stack.enter_context(
            sf.SessionLoop(
                sessions["unet"],
                sf.LoopConfig(
                    ("latents",),
                    (sf.ExitCondition(sf.ExitKind.MAX_ITERATIONS, policy["num_inference_steps"]),),
                    ddim=demo._typed_policy(policy),
                ),
            )
        )
        components = (
            sf.ChainComponent("positive", sessions["text_encoder"], (external("input_ids", "positive_ids"),)),
            sf.ChainComponent("negative", sessions["text_encoder"], (external("input_ids", "negative_ids"),)),
            sf.ChainComponent(
                "denoise",
                denoise,
                (
                    external("init", "init"),
                    previous("positive", 0, "last_hidden_state"),
                    previous("negative", 1, "last_hidden_state"),
                ),
            ),
            sf.ChainComponent("decode", sessions["vae_decoder"], (previous("latent_sample", 2, "latents"),)),
            sf.ChainComponent("preprocess", sessions["preprocess"], (previous("sample", 3, "sample"),)),
            sf.ChainComponent(
                "safety",
                sessions["safety_checker"],
                (previous("clip_input", 4, "clip_input"), previous("images", 4, "images")),
            ),
        )
        chain = stack.enter_context(
            sf.SessionChain(
                sf.ChainConfig(
                    components,
                    (
                        sf.ChainOutput("image", 5, "out_images"),
                        sf.ChainOutput("unsafe", 5, "has_nsfw_concepts"),
                        sf.ChainOutput("latents", 2, "latents"),
                    ),
                )
            )
        )
        return chain.create_execution(inputs)


def qualify_components(root):
    package = root / "package"
    options = demo.session_options(True)
    originals = {
        name: ort.InferenceSession(str(root / "originals" / name / "model.onnx"), options, enable_fallback=False)
        for name in demo.COMPONENTS
    }
    wrappers = {
        name: ort.InferenceSession(
            str(package / "components" / name / "cuda/model.onnx"), options, enable_fallback=False
        )
        for name in demo.COMPONENTS
    }
    placements = {}
    for name, session in wrappers.items():
        placements[name] = [
            {"ep": group.ep_name, "op": node.op_type}
            for group in session.get_provider_graph_assignment_info()
            for node in group.get_nodes()
        ]
        assert any(value["ep"] == "CUDAExecutionProvider" for value in placements[name]), name
    required = {"NhwcConv", "GroupNorm", "SkipGroupNorm", "MultiHeadAttention"}
    for operator in required:
        assigned = [value["ep"] for value in placements["unet"] if value["op"] == operator]
        assert assigned and set(assigned) == {"CUDAExecutionProvider"}, (operator, assigned)

    tokenizer = demo.LocalTokenizer(package / "tokenizer")
    ids = tokenizer.encode(PROMPT)
    original_text = originals["text_encoder"].run(None, {"input_ids": ids})
    wrapped_text = wrappers["text_encoder"].run(None, {"input_ids": ids})
    errors = {}
    for original, wrapped in zip(original_text, wrapped_text, strict=True):
        np.testing.assert_array_equal(wrapped, original.astype(np.float32))
    errors["text_encoder"] = 0.0
    rng = np.random.default_rng(42)
    errors["unet"] = {}
    for batch in (1, 2):
        sample = rng.standard_normal((batch, 4, 64, 64)).astype(np.float16)
        conditioning = np.repeat(original_text[0], batch, axis=0)
        timestep = np.array([951], dtype=np.float16)
        feeds = {"sample": sample, "encoder_hidden_states": conditioning, "timestep": timestep}
        original = originals["unet"].run(None, feeds)[0].astype(np.float32)
        wrapped = wrappers["unet"].run(None, {key: value.astype(np.float32) for key, value in feeds.items()})[0]
        np.testing.assert_allclose(wrapped, original, rtol=2e-3, atol=2e-3)
        errors["unet"][str(batch)] = float(np.max(np.abs(wrapped - original)))
    latent = rng.standard_normal((1, 4, 64, 64)).astype(np.float32)
    scaled = (latent * np.float32(1 / 0.18215)).astype(np.float16)
    original = originals["vae_decoder"].run(None, {"latent_sample": scaled})[0].astype(np.float32)
    wrapped = wrappers["vae_decoder"].run(None, {"latent_sample": latent})[0]
    repeat = originals["vae_decoder"].run(None, {"latent_sample": scaled})[0].astype(np.float32)
    np.testing.assert_allclose(repeat, original, rtol=2e-2, atol=2e-2)
    np.testing.assert_allclose(wrapped, original, rtol=2e-2, atol=2e-2)
    errors["vae_original_repeat"] = float(np.max(np.abs(repeat - original)))
    errors["vae_decoder"] = float(np.max(np.abs(wrapped - original)))
    normalized = np.clip(original / 2 + 0.5, 0, 1).transpose(0, 2, 3, 1)
    processor = demo.importlib.import_module("transformers").CLIPImageProcessor.from_pretrained(
        str(package / "feature_extractor"), local_files_only=True
    )
    pil = Image.fromarray(np.round(normalized[0] * 255).astype(np.uint8))
    features = processor(images=pil, return_tensors="np").pixel_values
    original = originals["safety_checker"].run(
        None, {"clip_input": features.astype(np.float16), "images": normalized.astype(np.float16)}
    )
    wrapped = wrappers["safety_checker"].run(None, {"clip_input": features, "images": normalized})
    np.testing.assert_array_equal(wrapped[0], original[0].astype(np.float32))
    np.testing.assert_array_equal(wrapped[1], original[1])
    errors["safety_checker"] = 0.0
    return {"maximum_errors": errors, "placement": placements}


class TestSD15RealAssets(unittest.TestCase):
    def setUp(self):
        if not os.environ.get("ORT_STATEFUL_SD15_ASSETS"):
            self.skipTest("Explicit local CUDA assets are required; no downloads are performed")
        self.assertIn("cuda-plugin-ep=1", ort.get_build_info())
        self.root = assets_root()

    def test_originals_wrappers_real_image_reference_and_typed_parity(self):
        demo.verify_assets(self.root)
        components = qualify_components(self.root)
        output = self.root / "qualification" / f"cuda-{time.time_ns()}"
        output.mkdir(parents=True)
        report = demo.validate_generation(self.root / "package", PROMPT, output / "sd15-seed42.png")
        with Image.open(report["output"]) as image:
            self.assertEqual(image.size, (512, 512))
            self.assertEqual(image.mode, "RGB")
            self.assertEqual(image.format, "PNG")
        with programmatic_execution(self.root / "package", PROMPT) as execution, execution.run() as result:
            typed_image = result.get_outputs()["image"].numpy()
            typed_unsafe = result.get_outputs()["unsafe"].numpy()
            np.testing.assert_array_equal(typed_unsafe, [False])
            with demo.create_execution(self.root / "package", PROMPT) as packaged, packaged.run() as paired:
                typed_errors = {
                    "latents": demo._check_precision(
                        result.get_outputs()["latents"].numpy(), paired.get_outputs()["latents"].numpy()
                    ),
                    "image": demo._check_precision(typed_image, paired.get_outputs()["image"].numpy(), image=True),
                }
                np.testing.assert_array_equal(typed_unsafe, paired.get_outputs()["unsafe"].numpy())
            typed_path = demo.save_image(typed_image, typed_unsafe, output / "sd15-typed.png")
        with Image.open(report["output"]) as expected:
            expected_pixels = np.asarray(expected, dtype=np.float32) / 255
        demo._check_precision(typed_image[0], expected_pixels, image=True)
        self.assertEqual(report["updates"], 20)
        self.assertEqual(report["invocations"], 25)
        (output / "qualification.json").write_text(
            json.dumps(
                {
                    "configuration": demo.configuration(self.root / "package"),
                    "components": components,
                    "generation": report,
                    "typed_parity_verified": True,
                    "typed_errors": typed_errors,
                    "typed_image_sha256": demo.digest(typed_path),
                },
                indent=2,
            )
        )
        print(f"SD15 qualification report: {output / 'qualification.json'}")

    def test_sequential_real_reference_and_safety(self):
        package = sequential_package(self.root)
        report = demo.validate_generation(package, PROMPT, package / "sd15-sequential.png")
        self.assertEqual(report["updates"], 20)
        self.assertEqual(report["invocations"], 45)
        self.assertFalse(report["unsafe"])
        (package / "qualification.json").write_text(json.dumps(report, indent=2))
        print(f"SD15 sequential report: {package / 'qualification.json'}")


def cuda_worker(mode):
    package = Path(os.environ.get("ORT_STATEFUL_SD15_PACKAGE", assets_root() / "package"))
    config = demo.configuration(package)
    execution = demo.create_execution(package, PROMPT)
    runtime = None
    if mode == "--cuda-trace-worker":
        with execution.run():
            pass
        execution.reset()
        runtime = ctypes.CDLL(os.environ.get("ORT_STATEFUL_CUDART", "libcudart.so.13"))
        for name in ("cudaProfilerStart", "cudaProfilerStop", "cudaDeviceSynchronize"):
            getattr(runtime, name).argtypes = []
            getattr(runtime, name).restype = ctypes.c_int
        for _ in range(2):
            with execution.step():
                pass
        assert runtime.cudaProfilerStart() == 0
    with execution:
        if runtime is not None:
            for _ in range(config["ddim"]["num_inference_steps"]):
                with execution.step() as step:
                    assert step.get_info().component_name == "denoise"
            assert runtime.cudaDeviceSynchronize() == 0
            assert runtime.cudaProfilerStop() == 0
        with execution.run() as result:
            image = result.get_outputs()["image"].numpy()
            unsafe = result.get_outputs()["unsafe"].numpy()
            retained = result.get_outputs()["latents"]
            retained_data = retained.numpy()
            assert image.shape == (1, 512, 512, 3) and np.isfinite(image).all()
            assert unsafe.dtype == np.bool_ and unsafe.shape == (1,)
            verdict = bool(unsafe[0])
            if config["ddim"]["num_inference_steps"] == 20:
                assert not verdict
            info = result.get_info()
            expected = 5 + config["ddim"]["num_inference_steps"] * (
                2 if config["ddim"]["guidance_mode"] == "sequential" else 1
            )
            assert info.invocations == expected
        execution.reset()
        with execution.run() as replay:
            demo._check_precision(replay.get_outputs()["image"].numpy(), image, image=True)
            np.testing.assert_array_equal(replay.get_outputs()["unsafe"].numpy(), unsafe)
        execution.reset()
        execution.cancel()
        try:
            execution.run()
        except Fail:
            assert execution.requires_reset()
        else:
            raise AssertionError("Cancellation did not stop the native chain")
        execution.reset()
    np.testing.assert_array_equal(retained.numpy(), retained_data)
    retained.close()
    del execution, image, unsafe
    gc.collect()
    print(
        json.dumps(
            {
                "mode": mode,
                "invocations": info.invocations,
                "updates": config["ddim"]["num_inference_steps"],
                "unsafe": verdict,
            }
        )
    )


if __name__ == "__main__":
    if sys.argv[1:] in (["--cuda-sanitizer-worker"], ["--cuda-trace-worker"]):
        cuda_worker(sys.argv[1])
    else:
        unittest.main()
