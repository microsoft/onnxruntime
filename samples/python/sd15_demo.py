# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Local CUDA SD1.5 through a native DDIM chain and mandatory safety checker."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import tempfile
import time
from pathlib import Path

import numpy as np
from sd15_models import boundary_model, safety_preprocess_model
from stateful_demo_assets import Asset, acquire_file, digest, verify_asset

import onnxruntime as ort
from onnxruntime import stateful as sf

REPOSITORY = "tlwu/stable-diffusion-v1-5-onnxruntime"
REVISION = "f22c0e90205a7a2c08bf3e8927b39dc84e4064c2"
LICENSE_REVISION = "14d42d09bffd871b1666a084fc954a50cff72ac0"
DEFAULT_ASSETS = Path.home() / "onnxruntime-stateful/sd15"
COMPONENTS = ("text_encoder", "unet", "vae_decoder", "safety_checker")
ASSETS = (
    Asset("text_encoder/model.onnx", 246178495, "26dc00399f19eecda97623110c39fb259aa59cab531f48e1aac47dc865d062c7"),
    Asset("unet/model.onnx", 1719631467, "3cf9daa9dd717e07c0875e811f6cdbf75b452fa5fa9b275c79810f7c555c5680"),
    Asset("vae_decoder/model.onnx", 99072656, "675910f5a5a2a633eb813dd7692efbd6df613fbd72835e94603e80c0818f3a29"),
    Asset("safety_checker/model.onnx", 608658360, "f73334378e98c3dad7b50fad5b8f1dd765f9a345afb49c8cb8f2d19bf5115e96"),
    Asset("README.md", 1181, "24e167f373c99766ce57119f719db17a11c5b504", True),
    Asset("model_index.json", 633, "b4e8fe5cf5cde1cf6edb7b619d498b26fb4f19dd", True),
    Asset("scheduler/scheduler_config.json", 374, "2a50ebda2676b32e672c2eb76707708b3bc5f407", True),
    Asset("feature_extractor/preprocessor_config.json", 518, "0d9d33b883843d1b370da781f3943051067e1b2c", True),
    Asset("tokenizer/merges.txt", 524619, "76e821f1b6f0a9709293c3b6b51ed90980b3166b", True),
    Asset("tokenizer/special_tokens_map.json", 472, "2c2130b544c0c5a72d5d00da071ba130a9800fb2", True),
    Asset("tokenizer/tokenizer_config.json", 704, "1bf819b6621086dc92428e2c9c8bbab39211fd55", True),
    Asset("tokenizer/vocab.json", 1059962, "469be27c5c010538f845f518c4f5e8574c78f7c8", True),
    Asset("LICENSE.txt", 14385, "0e609df0d8cd3b5d11a1ea962a56b604b70846a5", True),
)


def acquire_assets(root: str | Path) -> dict:
    directory = Path(root) / "originals"
    result = {}
    for asset in ASSETS:
        url = (
            f"https://huggingface.co/spaces/CompVis/stable-diffusion-license/resolve/{LICENSE_REVISION}/license.txt"
            if asset.name == "LICENSE.txt"
            else f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{asset.name}"
        )
        result[asset.name] = acquire_file(directory / asset.name, asset, url + "?download=true")
    return result


def verify_assets(root: Path) -> dict:
    result = {}
    for asset in ASSETS:
        path = root / "originals" / asset.name
        verify_asset(path, asset)
        result[asset.name] = {"bytes": asset.size, "sha256": digest(path) if asset.git_blob else asset.digest}
    return result


def session_options(cuda: bool, device_id: int = 0) -> ort.SessionOptions:
    if type(device_id) is not int or device_id < 0:
        raise ValueError("device_id must be a nonnegative integer")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.add_session_config_entry("session.record_ep_graph_assignment_info", "1")
    if not cuda:
        if device_id != 0:
            raise ValueError("CPU adapters use ordinal 0")
        return options
    devices = [
        value
        for value in ort.get_ep_devices()
        if value.ep_name == "CUDAExecutionProvider"
        and (info := value.memory_info(ort.OrtDeviceMemoryType.DEFAULT)) is not None
        and info.device_id == device_id
    ]
    if len(devices) != 1:
        raise ValueError(f"CUDA-plugin device {device_id} is unavailable or ambiguous")
    options.add_provider_for_devices(devices, {"cudnn_conv_algo_search": "HEURISTIC"})
    return options


def ddim_policy(steps=20, guidance=7.5, sequential=False) -> dict:
    if type(steps) is not int or not 1 <= steps <= 999:
        raise ValueError("DDIM steps must be an integer in [1,999]")
    if type(guidance) not in (int, float) or not np.isfinite(guidance) or guidance < 0:
        raise ValueError("Guidance must be a finite nonnegative number")
    if type(sequential) is not bool:
        raise TypeError("sequential must be bool")
    return {
        "profile": "sd15_ddim_v1",
        "latent_shape": [1, 4, 64, 64],
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "guidance_mode": "sequential" if sequential else "batched",
        "timestep_format": "float32_vector",
        "latent_input": "sample",
        "timestep_input": "timestep",
        "conditioning_input": "encoder_hidden_states",
        "prediction_output": "out_sample",
        "initialization_input": "init",
        "positive_conditioning_input": "positive",
        "negative_conditioning_input": "negative",
        "latent_output": "latents",
    }


def _external(name):
    return {"kind": "external_input", "name": name}


def _output(step, name):
    return {"kind": "model_output", "step": step, "output": name}


def _session_step(name, component, inputs):
    return {"id": name, "component": component, "executor_info": {"ort": {"kind": "session", "inputs": inputs}}}


def manifest(policy, hashes, device_id):
    components = {}
    for name in (*COMPONENTS, "preprocess"):
        cpu = name == "preprocess"
        variant = "cpu" if cpu else "cuda"
        components[name] = {
            "variants": {
                variant: {
                    "variant_directory": f"components/{name}/{variant}",
                    "ep": "CPUExecutionProvider" if cpu else "CUDAExecutionProvider",
                    "device": "cpu" if cpu else "gpu",
                    "executor_info": {"ort": {"model_file": "model.onnx"}},
                }
            }
        }
    denoise = {
        "id": "denoise",
        "component": "unet",
        "executor_info": {
            "ort": {
                "kind": "session_loop",
                "inputs": {
                    "init": _external("init"),
                    "positive": _output("positive", "last_hidden_state"),
                    "negative": _output("negative", "last_hidden_state"),
                },
                "loop": {
                    "ddim": policy,
                    "exit_conditions": [{"kind": "max_iterations", "limit": policy["num_inference_steps"]}],
                },
            }
        },
    }
    steps = [
        _session_step("positive", "text_encoder", {"input_ids": _external("positive_ids")}),
        _session_step("negative", "text_encoder", {"input_ids": _external("negative_ids")}),
        denoise,
        _session_step("decode", "vae_decoder", {"latent_sample": _output("denoise", "latents")}),
        _session_step("preprocess", "preprocess", {"sample": _output("decode", "sample")}),
        _session_step(
            "safety",
            "safety_checker",
            {
                "clip_input": _output("preprocess", "clip_input"),
                "images": _output("preprocess", "images"),
            },
        ),
    ]
    return {
        "schema_version": "1.1",
        "components": components,
        "chains": {
            "image": {
                "steps": steps,
                "executor_info": {
                    "ort": {
                        "outputs": {
                            "image": _output("safety", "out_images"),
                            "unsafe": _output("safety", "has_nsfw_concepts"),
                            "latents": _output("denoise", "latents"),
                        }
                    }
                },
            }
        },
        "additional_metadata": {
            "sd15_demo": {
                "repository": REPOSITORY,
                "revision": REVISION,
                "license_revision": LICENSE_REVISION,
                "source_scheduler": "PNDMScheduler",
                "effective_scheduler": "sd15_ddim_v1",
                "device_id": device_id,
                "cuda_conv_algo_search": "HEURISTIC",
                "ddim": policy,
                "assets": hashes,
            }
        },
    }


def build_package(root: str | Path, *, steps=20, guidance=7.5, sequential=False, device_id=0) -> Path:
    policy = ddim_policy(steps, guidance, sequential)
    options = session_options(True, device_id)
    root = Path(root)
    hashes = verify_assets(root)
    package = root / "package"
    wrappers = {}
    for name in COMPONENTS:
        destination = package / "components" / name / "cuda/model.onnx"
        wrappers[name] = boundary_model(
            root / "originals" / name / "model.onnx",
            destination,
            latent_scale=name == "vae_decoder",
            independent_image_axes=name == "safety_checker",
        )
    preprocess = package / "components/preprocess/cpu/model.onnx"
    safety_preprocess_model(preprocess)
    wrappers["preprocess"] = {"sha256": digest(preprocess), "profile": "pillow13_clip512_224_v1"}
    for asset in ASSETS:
        if asset.name.endswith(".onnx"):
            continue
        destination = package / asset.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            destination.hardlink_to(root / "originals" / asset.name)
        verify_asset(destination, asset)
    data = manifest(policy, hashes, device_id)
    data["additional_metadata"]["sd15_demo"]["wrappers"] = wrappers
    # Native factories need the canonical manifest name; keep rollback deterministic.
    path = package / "manifest.json"
    previous = path.read_bytes() if path.exists() else None
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    valid = False
    try:
        with (
            sf.ModelPackage(package) as opened,
            opened.create_chain("image", options, step_session_options={"preprocess": session_options(False)}),
        ):
            valid = True
    finally:
        if not valid:
            if previous is None:
                path.unlink()
            else:
                path.write_bytes(previous)
    return package


class LocalTokenizer:
    def __init__(self, path: Path):
        module = importlib.import_module("transformers")
        self.tokenizer = module.CLIPTokenizer.from_pretrained(str(path), local_files_only=True)

    def encode(self, text: str) -> np.ndarray:
        if not isinstance(text, str):
            raise TypeError("Prompt must be text")
        ids = self.tokenizer.encode(text, add_special_tokens=True, truncation=False)
        if len(ids) > 77:
            raise ValueError("CLIP prompt exceeds 77 tokens including special tokens")
        padded = self.tokenizer.pad({"input_ids": ids}, padding="max_length", max_length=77, return_tensors="np")[
            "input_ids"
        ]
        return np.asarray(padded, dtype=np.int32).reshape(1, 77)


def configuration(package: Path) -> dict:
    value = json.loads((package / "manifest.json").read_text(encoding="utf-8"))["additional_metadata"]["sd15_demo"]
    if (
        value["repository"] != REPOSITORY
        or value["revision"] != REVISION
        or value["effective_scheduler"] != "sd15_ddim_v1"
    ):
        raise ValueError("Unexpected SD1.5 package provenance or scheduler")
    expected = ddim_policy(
        value["ddim"]["num_inference_steps"],
        value["ddim"]["guidance_scale"],
        value["ddim"]["guidance_mode"] == "sequential",
    )
    if value["ddim"] != expected:
        raise ValueError("Unexpected SD1.5 DDIM policy")
    if value.get("cuda_conv_algo_search") != "HEURISTIC":
        raise ValueError("Unexpected CUDA convolution-search policy; rebuild the SD1.5 package")
    return value


def _execution_inputs(package, prompt, negative, seed, adapter, initial_latents=None):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Positive prompt must be nonempty text")
    if not isinstance(negative, str):
        raise TypeError("Negative prompt must be text")
    if type(seed) is not int or not 0 <= seed < 1 << 64:
        raise ValueError("Seed must be an unsigned64-bit integer")
    if adapter is None:
        adapter = LocalTokenizer(Path(package) / "tokenizer")
    positive = adapter.encode(prompt)
    negative_ids = adapter.encode(negative)
    for value in (positive, negative_ids):
        if not isinstance(value, np.ndarray) or value.dtype != np.int32 or value.shape != (1, 77):
            raise ValueError("CLIP adapter must return INT32 [1,77] arrays")
    initialization = np.array(seed, dtype=np.uint64)
    if initial_latents is not None:
        if (
            not isinstance(initial_latents, np.ndarray)
            or initial_latents.dtype != np.float32
            or initial_latents.shape != (1, 4, 64, 64)
            or not np.isfinite(initial_latents).all()
        ):
            raise ValueError("Initial latents must be finite FP32 [1,4,64,64]")
        initialization = initial_latents
    return {"positive_ids": positive, "negative_ids": negative_ids, "init": initialization}


def create_execution(
    package: str | Path, prompt: str, *, negative="", seed=42, adapter=None, initial_latents: np.ndarray | None = None
) -> sf.StatefulExecution:
    inputs = _execution_inputs(package, prompt, negative, seed, adapter, initial_latents)
    package = Path(package)
    config = configuration(package)
    with (
        sf.ModelPackage(package) as opened,
        opened.create_chain(
            "image",
            session_options(True, config["device_id"]),
            step_session_options={"preprocess": session_options(False)},
        ) as chain,
    ):
        return chain.create_execution(inputs)


def save_image(images: np.ndarray, unsafe: np.ndarray, destination: str | Path) -> Path:
    if not isinstance(unsafe, np.ndarray) or unsafe.dtype != np.bool_ or unsafe.shape != (1,):
        raise ValueError("Safety checker must return one boolean verdict")
    if bool(unsafe[0]):
        raise ValueError("Safety checker blocked this image; no output was saved")
    if not isinstance(images, np.ndarray) or images.shape != (1, 512, 512, 3) or images.dtype != np.float32:
        raise ValueError("Image must be FP32 NHWC [1,512,512,3]")
    if not np.isfinite(images).all() or images.min() < 0 or images.max() > 1:
        raise ValueError("Image values must be finite and within [0,1]")
    image_module = importlib.import_module("PIL.Image")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".png-", dir=destination.parent) as temporary:
        pending = Path(temporary) / "image.png"
        image_module.fromarray(np.round(images[0] * 255).astype(np.uint8)).save(pending, format="PNG")
        with image_module.open(pending) as encoded:
            encoded.verify()
        destination.hardlink_to(pending)
    return destination


def generate(package: str | Path, prompt: str, output: str | Path, *, negative="", seed=42, adapter=None) -> dict:
    started = time.perf_counter()
    execution = create_execution(package, prompt, negative=negative, seed=seed, adapter=adapter)
    initialized = time.perf_counter()
    with execution, execution.run() as result:
        completed = time.perf_counter()
        outputs = result.get_outputs()
        image = outputs["image"].numpy()
        unsafe = outputs["unsafe"].numpy()
        path = save_image(image, unsafe, output)
        info = result.get_info()
    return {
        "output": str(path),
        "sha256": digest(path),
        "unsafe": bool(unsafe[0]),
        "seed": seed,
        "invocations": info.invocations,
        "setup_seconds": initialized - started,
        "execution_seconds": completed - initialized,
        "output_seconds": time.perf_counter() - completed,
    }


def _typed_policy(value):
    fields = dict(value)
    fields["guidance_mode"] = (
        sf.GuidanceMode.SEQUENTIAL if fields["guidance_mode"] == "sequential" else sf.GuidanceMode.BATCHED
    )
    fields["timestep_format"] = sf.TimestepFormat.FLOAT32_VECTOR
    return sf.DdimConfig(**fields)


def _scheduler(steps):
    module = importlib.import_module("diffusers")
    scheduler = module.DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
        prediction_type="epsilon",
        timestep_spacing="leading",
        thresholding=False,
    )
    scheduler.set_timesteps(steps)
    return scheduler


def _prediction(session, latents, positive, negative, timestep, policy):
    time_value = np.array([timestep], dtype=np.float32)
    if policy["guidance_mode"] == "batched":
        raw = session.run(
            ["out_sample"],
            {
                "sample": np.concatenate((latents, latents)),
                "timestep": time_value,
                "encoder_hidden_states": np.concatenate((negative, positive)),
            },
        )[0]
        unconditioned, conditioned = np.split(raw, 2)
    else:
        unconditioned = session.run(
            ["out_sample"],
            {
                "sample": latents,
                "timestep": time_value,
                "encoder_hidden_states": negative,
            },
        )[0]
        conditioned = session.run(
            ["out_sample"],
            {
                "sample": latents,
                "timestep": time_value,
                "encoder_hidden_states": positive,
            },
        )[0]
    return unconditioned + np.float32(policy["guidance_scale"]) * (conditioned - unconditioned)


def _error_metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    if not isinstance(actual, np.ndarray) or not isinstance(expected, np.ndarray):
        raise AssertionError("Precision checks require NumPy arrays")
    if actual.shape != expected.shape or not actual.size:
        raise AssertionError("Precision checks require matching nonempty shapes")
    if not np.issubdtype(actual.dtype, np.floating) or not np.issubdtype(expected.dtype, np.floating):
        raise AssertionError("Precision checks require floating-point tensors")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise AssertionError("Precision checks require finite tensors")
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    return {
        "max_absolute_error": float(np.max(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference * difference))),
    }


def _check_precision(actual, expected, *, image=False, label="FP16 comparison"):
    metrics = _error_metrics(actual, expected)
    rmse_limit = 0.005 if image else 0.01
    if metrics["rmse"] > rmse_limit:
        raise AssertionError(f"{label}: {metrics}; required RMSE<={rmse_limit}")
    return metrics


def validate_generation(
    package: str | Path, prompt: str, output: str | Path, *, negative="", seed=42, adapter=None
) -> dict:
    """Compare native updates to ordinary CUDA Sessions and the pinned CPU DDIM reference."""
    image_module = importlib.import_module("PIL.Image")
    package = Path(package)
    inputs = _execution_inputs(package, prompt, negative, seed, adapter)
    config = configuration(package)
    policy = config["ddim"]
    options = session_options(True, config["device_id"])
    sessions = {
        name: ort.InferenceSession(
            str(package / "components" / name / "cuda/model.onnx"), options, enable_fallback=False
        )
        for name in COMPONENTS
    }
    positive = sessions["text_encoder"].run(["last_hidden_state"], {"input_ids": inputs["positive_ids"]})[0]
    negative_condition = sessions["text_encoder"].run(["last_hidden_state"], {"input_ids": inputs["negative_ids"]})[0]
    typed = _typed_policy(policy)
    loop_config = sf.LoopConfig(
        ("latents",), (sf.ExitCondition(sf.ExitKind.MAX_ITERATIONS, policy["num_inference_steps"]),), ddim=typed
    )
    with (
        sf.SessionLoop(sessions["unet"], loop_config) as loop,
        loop.create_execution(
            {"init": np.array(seed, dtype=np.uint64), "positive": positive, "negative": negative_condition}
        ) as initialization,
        initialization.get_state() as initial_state,
    ):
        initial = initial_state.get_outputs(sf.OutputGroup.STATE_VALUES)["latents"].numpy()
    scheduler = _scheduler(policy["num_inference_steps"])
    torch = importlib.import_module("torch")
    latents = initial.copy()
    previous_native = initial.copy()
    maximum_error = 0.0
    maximum_rmse = 0.0
    update_errors = []
    updates = 0
    saved_snapshot = None
    snapshot_value = None
    with (
        sf.ModelPackage(package) as opened,
        opened.create_chain("image", options, step_session_options={"preprocess": session_options(False)}) as chain,
        chain.create_execution(inputs) as execution,
    ):
        native_decoded = None
        native_preprocessed = None
        for _ in range(policy["num_inference_steps"] + 6):
            with execution.step() as result:
                info = result.get_info()
                values = result.get_outputs(sf.OutputGroup.COMPONENT_OUTPUTS)
                if info.component_name in ("positive", "negative"):
                    expected = positive if info.component_name == "positive" else negative_condition
                    np.testing.assert_allclose(values["last_hidden_state"].numpy(), expected, rtol=2e-3, atol=2e-3)
                elif info.component_name == "denoise":
                    timestep = scheduler.timesteps[updates]
                    local_prediction = _prediction(
                        sessions["unet"], previous_native, positive, negative_condition, int(timestep), policy
                    )
                    local = scheduler.step(
                        torch.from_numpy(local_prediction), timestep, torch.from_numpy(previous_native), eta=0
                    ).prev_sample.numpy()
                    prediction = _prediction(
                        sessions["unet"],
                        latents,
                        positive,
                        negative_condition,
                        int(timestep),
                        policy,
                    )
                    latents = scheduler.step(
                        torch.from_numpy(prediction), timestep, torch.from_numpy(latents), eta=0
                    ).prev_sample.numpy()
                    actual = values["latents"].numpy()
                    np.testing.assert_allclose(
                        actual, local, rtol=2e-3, atol=2e-2, err_msg=f"Same-input DDIM update {updates + 1}"
                    )
                    metrics = _check_precision(actual, latents, label=f"Closed-loop DDIM update {updates + 1}")
                    local_metrics = _error_metrics(actual, local)
                    update_errors.append(
                        {
                            "update": updates + 1,
                            "timestep": int(timestep),
                            "local": local_metrics,
                            "closed_loop": metrics,
                        }
                    )
                    maximum_error = max(maximum_error, metrics["max_absolute_error"])
                    maximum_rmse = max(maximum_rmse, metrics["rmse"])
                    previous_native = actual.copy()
                    updates += 1
                    if saved_snapshot is None:
                        saved_snapshot = values["latents"]
                        snapshot_value = actual.copy()
                elif info.component_name == "decode":
                    native_decoded = values["sample"].numpy()
                elif info.component_name == "preprocess":
                    native_preprocessed = {name: value.numpy() for name, value in values.items()}
                if info.completed:
                    final_images = result.get_outputs()["image"].numpy()
                    final_unsafe = result.get_outputs()["unsafe"].numpy()
                    final_latents = result.get_outputs()["latents"].numpy()
                    break
        else:
            raise AssertionError("Native SD chain did not complete its declared stages")
        if updates != policy["num_inference_steps"]:
            raise AssertionError("Native DDIM update count does not match the definition")
        expected_invocations = 5 + updates * (2 if policy["guidance_mode"] == "sequential" else 1)
        if info.invocations != expected_invocations:
            raise AssertionError("DDIM/ordinary-stage invocation accounting mismatch")
        if native_decoded is None or native_preprocessed is None or saved_snapshot is None:
            raise AssertionError("Native chain omitted a required image or latent checkpoint")
        matching_decoded = sessions["vae_decoder"].run(["sample"], {"latent_sample": final_latents})[0]
        np.testing.assert_allclose(native_decoded, matching_decoded, rtol=2e-2, atol=2e-2)
        reference_decoded = sessions["vae_decoder"].run(["sample"], {"latent_sample": latents})[0]
        normalized = np.clip(reference_decoded / 2 + 0.5, 0, 1).transpose(0, 2, 3, 1)
        processor = importlib.import_module("transformers").CLIPImageProcessor.from_pretrained(
            str(package / "feature_extractor"), local_files_only=True
        )
        pil = image_module.fromarray(np.round(normalized[0] * 255).astype(np.uint8))
        features = processor(images=pil, return_tensors="np").pixel_values
        native_normalized = np.clip(native_decoded / 2 + 0.5, 0, 1).transpose(0, 2, 3, 1)
        native_pil = image_module.fromarray(np.round(native_normalized[0] * 255).astype(np.uint8))
        native_features = processor(images=native_pil, return_tensors="np").pixel_values
        np.testing.assert_array_equal(native_preprocessed["images"], native_normalized)
        np.testing.assert_array_equal(native_preprocessed["clip_input"], native_features)
        reference_images, reference_unsafe = sessions["safety_checker"].run(
            None, {"clip_input": features, "images": normalized}
        )
        np.testing.assert_array_equal(final_unsafe, reference_unsafe)
        image_errors = _check_precision(final_images, reference_images, image=True, label="Closed-loop image")
        np.testing.assert_array_equal(saved_snapshot.numpy(), snapshot_value)
        execution.reset()
        with execution.run() as replay:
            replay_image = replay.get_outputs()["image"].numpy()
            replay_errors = {
                "latents": _check_precision(
                    replay.get_outputs()["latents"].numpy(), final_latents, label="Reset/replay latents"
                ),
                "image": _check_precision(replay_image, final_images, image=True, label="Reset/replay image"),
                "image_tensor_sha256": hashlib.sha256(replay_image.tobytes()).hexdigest(),
            }
            np.testing.assert_array_equal(replay.get_outputs()["unsafe"].numpy(), final_unsafe)
        with chain.create_execution({**inputs, "init": initial}) as supplied, supplied.run() as supplied_result:
            supplied_image = supplied_result.get_outputs()["image"].numpy()
            supplied_errors = {
                "latents": _check_precision(
                    supplied_result.get_outputs()["latents"].numpy(),
                    final_latents,
                    label="Supplied-initialization latents",
                ),
                "image": _check_precision(
                    supplied_image, final_images, image=True, label="Supplied-initialization image"
                ),
                "image_tensor_sha256": hashlib.sha256(supplied_image.tobytes()).hexdigest(),
            }
            np.testing.assert_array_equal(supplied_result.get_outputs()["unsafe"].numpy(), final_unsafe)
        path = save_image(final_images, final_unsafe, output)
    return {
        "output": str(path),
        "sha256": digest(path),
        "unsafe": bool(final_unsafe[0]),
        "updates": updates,
        "invocations": info.invocations,
        "maximum_latent_error": maximum_error,
        "maximum_latent_rmse": maximum_rmse,
        "update_errors": update_errors,
        "image_errors": image_errors,
        "reset_replay_errors": replay_errors,
        "supplied_initialization_errors": supplied_errors,
        "initial_latents_sha256": hashlib.sha256(initial.tobytes()).hexdigest(),
        "reset_replay_verified": True,
        "supplied_initialization_verified": True,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("acquire")
    build = commands.add_parser("build")
    build.add_argument("--steps", type=int, default=20)
    build.add_argument("--guidance", type=float, default=7.5)
    build.add_argument("--sequential", action="store_true")
    build.add_argument("--device-id", type=int, default=0)
    run = commands.add_parser("run")
    run.add_argument("--prompt", required=True)
    run.add_argument("--negative", default="")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--output", type=Path, required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--prompt", required=True)
    validate.add_argument("--negative", default="")
    validate.add_argument("--seed", type=int, default=42)
    validate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "acquire":
        print(json.dumps(acquire_assets(args.assets), indent=2))
    elif args.command == "build":
        print(
            build_package(
                args.assets,
                steps=args.steps,
                guidance=args.guidance,
                sequential=args.sequential,
                device_id=args.device_id,
            )
        )
    elif args.command == "run":
        print(
            json.dumps(
                generate(args.assets / "package", args.prompt, args.output, negative=args.negative, seed=args.seed),
                indent=2,
            )
        )
    else:
        print(
            json.dumps(
                validate_generation(
                    args.assets / "package", args.prompt, args.output, negative=args.negative, seed=args.seed
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
