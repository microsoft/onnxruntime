# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Explicit local Phi-3 assets and native stateful Model Packages."""

from __future__ import annotations

import argparse
import importlib
import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import onnx
import stateful_demo_assets as asset_io
from stateful_demo_assets import Asset, verify_asset
from stateful_demo_assets import digest as _digest

import onnxruntime as ort
from onnxruntime import stateful as sf

REVISION = "5f5f794c1c23c9d5ee142af85df02a6cc52d6945"
REPOSITORY = "microsoft/Phi-3-mini-4k-instruct-onnx"
DEFAULT_ASSETS = Path.home() / "onnxruntime-stateful/phi3-mini-4k-instruct"


@dataclass(frozen=True)
class Profile:
    repository_directory: str
    filename: str
    cache_dtype: str
    assets: tuple[Asset, ...]


_TOKENIZER_ASSETS = (
    Asset("added_tokens.json", 306, "178968dec606c790aa335e9142f6afec37288470", True),
    Asset("config.json", 919, "01de35834a12bca5fc9150cc2a8351135f442757", True),
    Asset("special_tokens_map.json", 599, "c6a944b4d49ce5d79030250ed6bdcbb1a65dfda1", True),
    Asset("tokenizer.json", 1937869, "88ec145f4e7684c009bc6d55df24bb82c7d3c379", True),
    Asset("tokenizer.model", 499723, "9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347"),
    Asset("tokenizer_config.json", 3441, "67aa82cddb4d66391ddf31ff99f059239bd2d1e7", True),
    Asset("LICENSE", 12501, "02c8af47f059cad687da5e5ea1739f30f4ee5871", True, True),
)
_CPU_MODEL = "phi3-mini-4k-instruct-cpu-int4-rtn-block-32-acc-level-4.onnx"
_CUDA_MODEL = "phi3-mini-4k-instruct-cuda-fp16.onnx"
PROFILES = {
    "cpu": Profile(
        "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4",
        _CPU_MODEL,
        "FLOAT",
        (
            Asset(_CPU_MODEL, 231335, "385cd1b908a0d2f8634e86d30236f6dbb7ae660eb3943fd1ef5bdc3847326480"),
            Asset(_CPU_MODEL + ".data", 2722861056, "5db30ce699aee1123cf9045742488db5928006fa618a42cb3c0840322a85ad0f"),
            Asset("genai_config.json", 1576, "5e5a61eea6e20bda5b011053a889535029e4b9c1", True),
            *_TOKENIZER_ASSETS,
        ),
    ),
    "cuda": Profile(
        "cuda/cuda-fp16",
        _CUDA_MODEL,
        "FLOAT16",
        (
            Asset(_CUDA_MODEL, 171231, "1862be84aabfb6ed6a4b8ab0cb8ed1ffe7383dd866dcb927b3daa9a5d1cb33db"),
            Asset(
                _CUDA_MODEL + ".data", 7642945536, "756684acf7d46a361958be08f42ef095ac1e2163843506c16f68400642e1813e"
            ),
            Asset("genai_config.json", 1727, "93ddf3fad60fcdab0f02a7bb393a535960ea7ac6", True),
            *_TOKENIZER_ASSETS,
        ),
    ),
}


def _profile(target: str) -> Profile:
    if not isinstance(target, str):
        raise TypeError("target must be cpu or cuda")
    if target not in PROFILES:
        raise ValueError(f"Unsupported target: {target}")
    return PROFILES[target]


def variant_directory(assets: str | Path, target: str) -> Path:
    _profile(target)
    return Path(assets) / target / "decoder" / target


def acquire_assets(assets: str | Path, target: str) -> dict[str, str]:
    """Explicitly acquire only pinned allowlisted assets; never overwrite invalid files."""
    profile = _profile(target)
    directory = variant_directory(assets, target)
    directory.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for asset in profile.assets:
        destination = directory / asset.name
        relative = asset.name if asset.repository_root else f"{profile.repository_directory}/{asset.name}"
        url = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{relative}?download=true"
        hashes[asset.name] = asset_io.acquire_file(destination, asset, url)
    return hashes


def _positive(value, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _name(pattern: str, layer: int) -> str:
    if not isinstance(pattern, str) or pattern.count("%d") != 1 or "%" in pattern.replace("%d", ""):
        raise ValueError("Cache name pattern must contain exactly one %d")
    return pattern.replace("%d", str(layer))


def _tensor_info(value) -> dict:
    tensor = value.type.tensor_type
    return {
        "name": value.name,
        "dtype": onnx.TensorProto.DataType.Name(tensor.elem_type),
        "shape": [
            dimension.dim_value if dimension.HasField("dim_value") else dimension.dim_param
            for dimension in tensor.shape.dim
        ],
    }


def _check_tensor(value: dict, dtype: str, shape: tuple) -> None:
    actual = value["shape"]
    if value["dtype"] != dtype or len(actual) != len(shape):
        raise ValueError(f"Unsupported tensor type/rank: {value}")
    for axis, (observed, expected) in enumerate(zip(actual, shape, strict=True)):
        if expected is None:
            if not isinstance(observed, str) or not observed:
                raise ValueError(f"Expected a dynamic sequence dimension: {value}")
        elif observed != expected and not (axis == 0 and expected == 1 and isinstance(observed, str) and observed):
            raise ValueError(f"Unsupported tensor shape: {value}")


def inspect_assets(assets: str | Path, target: str) -> dict:
    """Inspect actual graph interfaces without loading external tensor payloads."""
    return inspect_variant(variant_directory(assets, target), target)


def inspect_variant(directory: Path, target: str) -> dict:
    """Verify a relocatable package variant's actual graph and external data."""
    profile = _profile(target)
    for asset in profile.assets:
        verify_asset(directory / asset.name, asset)
    config = json.loads((directory / "genai_config.json").read_text(encoding="utf-8"))["model"]
    decoder = config["decoder"]
    layers = _positive(decoder["num_hidden_layers"], "num_hidden_layers")
    heads = _positive(decoder["num_key_value_heads"], "num_key_value_heads")
    head_size = _positive(decoder["head_size"], "head_size")
    vocabulary = _positive(config["vocab_size"], "vocab_size")
    context = _positive(config["context_length"], "context_length")
    eos = config["eos_token_id"]
    if (
        not isinstance(eos, list)
        or not eos
        or any(type(token) is not int or not 0 <= token < vocabulary for token in eos)
    ):
        raise ValueError("Invalid explicit EOS token set")
    if decoder["filename"] != profile.filename:
        raise ValueError("Pinned decoder filename does not match model metadata")
    model = onnx.load(directory / profile.filename, load_external_data=False)
    inputs = {value.name: _tensor_info(value) for value in model.graph.input}
    outputs = {value.name: _tensor_info(value) for value in model.graph.output}
    names = decoder["inputs"]
    output_names = decoder["outputs"]
    expected_inputs = {names["input_ids"], names["attention_mask"]}
    expected_outputs = {output_names["logits"]}
    _check_tensor(inputs[names["input_ids"]], "INT64", (1, None))
    _check_tensor(inputs[names["attention_mask"]], "INT64", (1, None))
    _check_tensor(outputs[output_names["logits"]], profile.cache_dtype, (1, None, vocabulary))
    caches = []
    for layer in range(layers):
        for kind in ("key", "value"):
            source = _name(names[f"past_{kind}_names"], layer)
            destination = _name(output_names[f"present_{kind}_names"], layer)
            _check_tensor(inputs[source], profile.cache_dtype, (1, heads, None, head_size))
            _check_tensor(outputs[destination], profile.cache_dtype, (1, heads, None, head_size))
            expected_inputs.add(source)
            expected_outputs.add(destination)
            caches.append(
                {"name": f"{kind}_{layer}", "input": source, "output": destination, "shape": [1, heads, 0, head_size]}
            )
    if set(inputs) != expected_inputs or set(outputs) != expected_outputs:
        raise ValueError("Unsupported additional or missing model ports")
    external = set()
    weight_size = (directory / (profile.filename + ".data")).stat().st_size
    for initializer in model.graph.initializer:
        if initializer.data_location == onnx.TensorProto.EXTERNAL:
            reference = {item.key: item.value for item in initializer.external_data}
            location = reference["location"]
            if location != profile.filename + ".data":
                raise ValueError(f"Unexpected external data reference: {location}")
            offset = int(reference.get("offset", "0"))
            length = int(reference["length"])
            if offset < 0 or length < 0 or offset + length > weight_size:
                raise ValueError(f"External tensor range exceeds verified weights: {initializer.name}")
            external.add(location)
    if external != {profile.filename + ".data"}:
        raise ValueError("Expected the pinned external weight file")
    return {
        "target": target,
        "filename": profile.filename,
        "cache_dtype": profile.cache_dtype,
        "input_ids": names["input_ids"],
        "attention_mask": names["attention_mask"],
        "logits": output_names["logits"],
        "vocabulary": vocabulary,
        "context_length": context,
        "eos_token_ids": eos,
        "caches": caches,
        "inputs": inputs,
        "outputs": outputs,
    }


def session_options(target: str, device_id: int = 0) -> ort.SessionOptions:
    """Select the actual requested EP device, not an ordinary-session fallback."""
    _profile(target)
    if type(device_id) is not int or device_id < 0:
        raise ValueError("device_id must be a nonnegative integer")
    options = ort.SessionOptions()
    # Keep independently owned stage Sessions low-contention and reproducible.
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.add_session_config_entry("session.record_ep_graph_assignment_info", "1")
    if target == "cpu":
        if device_id != 0:
            raise ValueError("CPU execution uses device ordinal 0")
        return options
    provider = "CUDAExecutionProvider"
    device = next(
        (
            value
            for value in ort.get_ep_devices()
            if value.ep_name == provider
            and (memory := value.memory_info(ort.OrtDeviceMemoryType.DEFAULT)) is not None
            and memory.device_id == device_id
        ),
        None,
    )
    if device is None:
        raise ValueError(f"Requested {provider} device {device_id} is unavailable")
    options.add_provider_for_devices([device], {})
    return options


def build_package(assets: str | Path, target: str, *, max_new_tokens: int = 16, device_id: int = 0) -> Path:
    """Declare one portable native chain; generation limits are definition-time policy."""
    if type(max_new_tokens) is not int or max_new_tokens < 0:
        raise ValueError("max_new_tokens must be a nonnegative integer")
    if type(device_id) is not int or device_id < 0:
        raise ValueError("device_id must be a nonnegative integer")
    report = inspect_assets(assets, target)
    if max_new_tokens > report["context_length"]:
        raise ValueError("Generation budget exceeds model context")
    provider = (
        {"mode": "cpu"}
        if target == "cpu"
        else {
            "mode": "require_provider",
            "provider_name": "CUDAExecutionProvider",
            "device_type": "gpu",
            "device_id": device_id,
        }
    )
    bindings = [
        {"cache": cache["name"], "input": cache["input"], "output": cache["output"], "update": "share"}
        for cache in report["caches"]
    ]
    prefill = {
        "id": "prefill",
        "component": "decoder",
        "executor_info": {
            "ort": {
                "kind": "session_loop",
                "inputs": {
                    report["input_ids"]: {"kind": "external_sequence", "name": "prompt"},
                    report["attention_mask"]: {"kind": "external_input", "name": "attention_mask"},
                },
                "loop": {
                    "finite_input_names": [report["input_ids"]],
                    "nonempty_input": report["input_ids"],
                    "exit_conditions": [{"kind": "end_of_inputs"}],
                    "kv_caches": bindings,
                    "cache_provider": provider,
                },
            }
        },
    }
    decode = {
        "id": "decode",
        "component": "decoder",
        "executor_info": {
            "ort": {
                "kind": "session_loop",
                "loop": {
                    "exit_conditions": [
                        {"kind": "max_generated_tokens", "limit": max_new_tokens},
                        {"kind": "eos", "token_ids": report["eos_token_ids"]},
                    ],
                    "kv_caches": bindings,
                    "cache_provider": provider,
                    "generation": {
                        "phase": "decode",
                        "input_ids": report["input_ids"],
                        "attention_mask": report["attention_mask"],
                        "logits_output": report["logits"],
                        "initial_logits": {
                            "kind": "model_output",
                            "step": "prefill",
                            "output": report["logits"],
                            "prompt_input": report["input_ids"],
                        },
                    },
                    "sampler": {"algorithm": "greedy", "provider": provider},
                },
            }
        },
    }
    manifest = {
        "schema_version": "1.1",
        "components": {
            "decoder": {
                "variants": {
                    target: {
                        "variant_directory": f"decoder/{target}",
                        "ep": "CPUExecutionProvider" if target == "cpu" else "CUDAExecutionProvider",
                        "device": "cpu" if target == "cpu" else "gpu",
                        "executor_info": {"ort": {"model_file": report["filename"]}},
                    }
                }
            }
        },
        "chains": {
            "generate": {
                "steps": [prefill, decode],
                "executor_info": {
                    "ort": {
                        "kv_caches": {
                            cache["name"]: {
                                "element_type": "float32" if target == "cpu" else "float16",
                                "shape": cache["shape"],
                                "sequence_axis": 2,
                                "initial_capacity": 0,
                            }
                            for cache in report["caches"]
                        },
                        "outputs": {
                            "tokens": {"kind": "generated_token_ids", "step": "decode"},
                            "prefill_logits": {"kind": "model_output", "step": "prefill", "output": report["logits"]},
                        },
                    }
                },
            }
        },
        "additional_metadata": {
            "phi3_demo": {
                "repository": REPOSITORY,
                "revision": REVISION,
                "target": target,
                "device_id": device_id,
                "max_new_tokens": max_new_tokens,
                "assets": {
                    asset.name: {
                        "bytes": asset.size,
                        "sha256": _digest(variant_directory(assets, target) / asset.name)
                        if asset.git_blob
                        else asset.digest,
                    }
                    for asset in _profile(target).assets
                },
            }
        },
    }
    root = Path(assets) / target
    with tempfile.TemporaryDirectory(prefix=".validate-", dir=root) as temporary:
        candidate = Path(temporary)
        variant = candidate / "decoder" / target
        variant.mkdir(parents=True)
        for asset in _profile(target).assets:
            (variant / asset.name).hardlink_to(variant_directory(assets, target) / asset.name)
        manifest_path = candidate / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        with (
            sf.ModelPackage(candidate) as package,
            package.create_chain("generate", session_options(target, device_id)),
        ):
            manifest_path.replace(root / "manifest.json")
    return root


class TextAdapter(Protocol):
    """Application-owned text boundary, separate from runtime execution policy."""

    def encode(self, text: str, *, raw: bool = False) -> list[int]: ...

    def decode(self, tokens: list[int]) -> str: ...


def _validate_prompt(text: str, raw: bool) -> None:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Prompt must be nonempty text")
    if type(raw) is not bool:
        raise TypeError("raw must be bool")


class LocalTokenizer:
    """Demo-only local tokenizer; downloaded Python implementations are never loaded."""

    def __init__(self, directory: str | Path):
        module = importlib.import_module("transformers")
        self.tokenizer = module.AutoTokenizer.from_pretrained(
            str(directory), local_files_only=True, trust_remote_code=False, use_fast=True
        )

    def encode(self, text: str, *, raw: bool = False) -> list[int]:
        _validate_prompt(text, raw)
        if raw:
            return self.tokenizer.encode(text, add_special_tokens=False)
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True
        )

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=True)


def package_configuration(root: Path) -> dict:
    """Read this demo's metadata while native factories remain the schema authority."""
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    config = manifest["additional_metadata"]["phi3_demo"]
    _profile(config["target"])
    if config["revision"] != REVISION or config["repository"] != REPOSITORY:
        raise ValueError("Package provenance does not match the approved Phi-3 revision")
    exits = manifest["chains"]["generate"]["steps"][1]["executor_info"]["ort"]["loop"]["exit_conditions"]
    budgets = [value["limit"] for value in exits if value["kind"] == "max_generated_tokens"]
    if (
        budgets != [config["max_new_tokens"]]
        or type(config["max_new_tokens"]) is not int
        or config["max_new_tokens"] < 0
    ):
        raise ValueError("Demo metadata and declared native generation budget disagree")
    return config


def _prepare_generation(root: Path, prompt: str, adapter: TextAdapter | None, raw: bool):
    _validate_prompt(prompt, raw)
    config = package_configuration(root)
    report = inspect_variant(root / "decoder" / config["target"], config["target"])
    if adapter is None:
        adapter = LocalTokenizer(root / "decoder" / config["target"])
    tokens = adapter.encode(prompt, raw=raw)
    if not isinstance(tokens, (list, tuple)) or not tokens:
        raise ValueError("Tokenizer must return a nonempty integer token sequence")
    if any(type(token) is not int or not 0 <= token < report["vocabulary"] for token in tokens):
        raise ValueError("Tokenizer returned an invalid token ID")
    if len(tokens) + config["max_new_tokens"] > report["context_length"]:
        raise ValueError("Prompt plus declared generation budget exceeds model context")
    return config, report, adapter, np.array([tokens], dtype=np.int64)


def _native_execution(root: Path, config: dict, ids: np.ndarray) -> sf.StatefulExecution:
    mask = np.ones(ids.shape, dtype=np.int64)
    with (
        sf.ModelPackage(root) as package,
        package.create_chain("generate", session_options(config["target"], config["device_id"])) as chain,
    ):
        execution = chain.create_execution({"attention_mask": mask}, sequences={"prompt": [ids]})
    return execution


def create_execution(
    package_root: str | Path, prompt: str, adapter: TextAdapter | None = None, *, raw: bool = False
) -> tuple[sf.StatefulExecution, TextAdapter, int]:
    """Return a native context, its adapter and prompt length; the caller closes the context."""
    root = Path(package_root)
    config, _, adapter, ids = _prepare_generation(root, prompt, adapter, raw)
    return _native_execution(root, config, ids), adapter, ids.shape[1]


def generate(package_root: str | Path, prompt: str, adapter: TextAdapter | None = None, *, raw: bool = False) -> dict:
    """Run the native chain and render only its generated continuation."""
    started = time.perf_counter()
    execution, adapter, prompt_length = create_execution(package_root, prompt, adapter, raw=raw)
    initialized = time.perf_counter()
    with execution, execution.run() as result:
        completed = time.perf_counter()
        tokens = result.get_outputs()["tokens"].numpy().tolist()
        info = result.get_info()
        dispatch = execution.get_dispatch_info(sf.ProviderFeature.SAMPLER, sf.ProviderOperation.SAMPLE_GREEDY, 1)
        return {
            "text": adapter.decode(tokens),
            "token_ids": tokens,
            "prompt_tokens": prompt_length,
            "generated_tokens": info.generated_tokens,
            "invocations": info.invocations,
            "exit_reasons": [reason.name for reason in info.exit_reasons],
            "sampler_origin": dispatch.origin.name,
            "sampler_provider_calls": dispatch.provider_calls,
            "sampler_cpu_calls": dispatch.cpu_calls,
            "setup_seconds": initialized - started,
            "generation_seconds": completed - initialized,
        }


def provider_placement(session: ort.InferenceSession, target: str, layers: int) -> list[dict]:
    """Copy actual graph assignment metadata; require CUDA attention and matrix compute."""
    groups = [
        {
            "provider": group.ep_name,
            "nodes": [{"name": node.name, "op_type": node.op_type} for node in group.get_nodes()],
        }
        for group in session.get_provider_graph_assignment_info()
    ]
    if target == "cpu" and any(group["provider"] != "CPUExecutionProvider" for group in groups):
        raise AssertionError("CPU qualification requires CPU-only model execution")
    if target == "cuda":
        gpu = [node for group in groups if group["provider"] == "CUDAExecutionProvider" for node in group["nodes"]]
        cpu = [node for group in groups if group["provider"] != "CUDAExecutionProvider" for node in group["nodes"]]
        if sum(node["op_type"] == "GroupQueryAttention" for node in gpu) != layers:
            raise AssertionError("CUDA qualification requires every attention layer on CUDA")
        if not any(node["op_type"] in ("MatMul", "Gemm") for node in gpu):
            raise AssertionError("CUDA qualification did not place matrix compute on CUDA")
        if any(node["op_type"] in ("GroupQueryAttention", "MatMul", "MatMulNBits", "Gemm") for node in cpu):
            raise AssertionError("Unexpected CPU fallback for attention/matrix compute")
    return groups


def validate_generation(
    package_root: str | Path, prompt: str, adapter: TextAdapter | None = None, *, raw: bool = False
) -> dict:
    """Differential validation against ordinary Session.run; readbacks are intentional here."""
    root = Path(package_root)
    config, report, adapter, ids = _prepare_generation(root, prompt, adapter, raw)
    reference = ort.InferenceSession(
        str(root / "decoder" / config["target"] / report["filename"]),
        session_options(config["target"], config["device_id"]),
        providers=["CPUExecutionProvider"] if config["target"] == "cpu" else None,
        enable_fallback=False,
    )
    placement = provider_placement(reference, config["target"], len(report["caches"]) // 2)
    output_names = [value.name for value in reference.get_outputs()]
    dtype = np.float32 if config["target"] == "cpu" else np.float16
    past = {cache["input"]: np.empty(cache["shape"], dtype=dtype) for cache in report["caches"]}
    rtol, atol = (1e-4, 1e-4) if config["target"] == "cpu" else (1e-2, 2e-2)
    tokens = []
    invocations = 0
    maximum_error = 0.0
    logits = None
    snapshot = {}
    snapshot_values = {}
    cache_names = {cache["name"] for cache in report["caches"]}
    with _native_execution(root, config, ids) as execution:
        for _ in range(config["max_new_tokens"] + 2):
            with execution.step() as result:
                info = result.get_info()
                if info.executed:
                    current_ids = ids if not invocations else np.array([[tokens[-1]]], dtype=np.int64)
                    feeds = {
                        report["input_ids"]: current_ids,
                        report["attention_mask"]: np.ones((1, ids.shape[1] + len(tokens)), dtype=np.int64),
                        **past,
                    }
                    expected = dict(zip(output_names, reference.run(None, feeds), strict=True))
                    actual = result.get_outputs(sf.OutputGroup.COMPONENT_OUTPUTS)
                    if set(actual) != set(expected):
                        raise AssertionError("Native/reference component output collections differ")
                    for name, value in actual.items():
                        copied = value.copy_to_cpu().numpy()
                        np.testing.assert_allclose(
                            copied,
                            expected[name],
                            rtol=rtol,
                            atol=atol,
                            equal_nan=False,
                            err_msg=f"Checkpoint mismatch: {name}",
                        )
                        error = np.max(np.abs(copied.astype(np.float32) - expected[name].astype(np.float32)), initial=0)
                        maximum_error = max(maximum_error, float(error))
                    past = {cache["input"]: expected[cache["output"]] for cache in report["caches"]}
                    logits = expected[report["logits"]]
                    invocations += 1
                    if invocations == 1:
                        snapshot = execution.get_state().get_outputs(sf.OutputGroup.STATE_VALUES)
                        if set(snapshot) != cache_names:
                            raise AssertionError("Native prefill cache snapshot names differ from the model")
                        snapshot_values = {name: value.copy_to_cpu().numpy() for name, value in snapshot.items()}
                if info.generated_token is not None:
                    if logits is None:
                        raise AssertionError("Native chain generated a token before prefill")
                    token = int(np.argmax(logits[0, -1]))
                    if token != info.generated_token:
                        raise AssertionError(
                            f"Native/reference greedy token mismatch: {info.generated_token} != {token}"
                        )
                    tokens.append(token)
                if info.completed:
                    final_tokens = result.get_outputs()["tokens"].numpy()
                    break
        else:
            raise AssertionError("Native chain did not complete within its declared token budget")
        if info.invocations != invocations or invocations != max(1, len(tokens)):
            raise AssertionError("Bootstrap/model invocation accounting differs from the reference")
        for name, value in snapshot.items():
            np.testing.assert_array_equal(value.copy_to_cpu().numpy(), snapshot_values[name])
        state = execution.get_state().get_outputs(sf.OutputGroup.STATE_VALUES)
        expected_length = ids.shape[1] + max(0, len(tokens) - 1)
        expected_device = sf.DeviceType.CPU if config["target"] == "cpu" else sf.DeviceType.GPU
        if set(state) != cache_names or any(
            value.get_info().shape[2] != expected_length or value.get_info().device_type != expected_device
            for value in state.values()
        ):
            raise AssertionError("Native cache length/device does not match executed tokens")
        np.testing.assert_array_equal(final_tokens, tokens)
        dispatch = execution.get_dispatch_info(sf.ProviderFeature.SAMPLER, sf.ProviderOperation.SAMPLE_GREEDY, 1)
        execution.reset()
        with execution.run() as replay:
            np.testing.assert_array_equal(replay.get_outputs()["tokens"].numpy(), tokens)
            if replay.get_info().invocations != invocations:
                raise AssertionError("Reset/replay invocation accounting changed")
        for name, value in snapshot.items():
            np.testing.assert_array_equal(value.copy_to_cpu().numpy(), snapshot_values[name])
        return {
            "target": config["target"],
            "prompt_tokens": ids.shape[1],
            "token_ids": tokens,
            "text": adapter.decode(tokens),
            "invocations": invocations,
            "rtol": rtol,
            "atol": atol,
            "maximum_absolute_checkpoint_error": maximum_error,
            "retained_cache_snapshots_unchanged": True,
            "reset_replay_verified": True,
            "cache_count": len(state),
            "cache_length": expected_length,
            "sampler_origin": dispatch.origin.name,
            "sampler_provider_calls": dispatch.provider_calls,
            "sampler_cpu_calls": dispatch.cpu_calls,
            "placement": placement,
        }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--target", choices=tuple(PROFILES), required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("acquire", help="Explicitly download and verify pinned assets")
    commands.add_parser("inspect", help="Verify and inspect existing local assets")
    build = commands.add_parser("build", help="Construct an offline portable native package")
    build.add_argument("--max-new-tokens", type=int, default=16)
    build.add_argument("--device-id", type=int, default=0)
    run = commands.add_parser("run", help="Run the already-declared package offline")
    run.add_argument("--prompt", required=True)
    run.add_argument("--raw", action="store_true")
    validate = commands.add_parser("validate", help="Compare native execution with an ordinary Session reference")
    validate.add_argument("--prompt", required=True)
    validate.add_argument("--raw", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "acquire":
        print(json.dumps(acquire_assets(args.assets, args.target), indent=2))
    elif args.command == "inspect":
        print(json.dumps(inspect_assets(args.assets, args.target), indent=2))
    elif args.command == "build":
        print(build_package(args.assets, args.target, max_new_tokens=args.max_new_tokens, device_id=args.device_id))
    elif args.command == "run":
        print(json.dumps(generate(args.assets / args.target, args.prompt, raw=args.raw), indent=2))
    else:
        print(json.dumps(validate_generation(args.assets / args.target, args.prompt, raw=args.raw), indent=2))


if __name__ == "__main__":
    main()
