#!/usr/bin/env python3
"""Run a fixed-shape ONNX model with CPU or a Windows ML NPU provider."""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import time
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from gemma_synthetic_data import (
    DEFAULT_PADDING_FRACTION,
    HIDDEN_STATE_CLIP,
    HIDDEN_STATE_STD,
    make_additive_attention_mask,
    make_hidden_states,
)


PROVIDER_NAMES = {
    "cpu": "CPUExecutionProvider",
    "vitisai": "VitisAIExecutionProvider",
    "qnn": "QNNExecutionProvider",
    "openvino": "OpenVINOExecutionProvider",
}
NPU_PROVIDERS = ("vitisai", "qnn", "openvino")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Path to a fixed-shape ONNX model.")
    parser.add_argument(
        "--provider",
        choices=tuple(PROVIDER_NAMES),
        default="vitisai",
        help="Execution provider; default: vitisai.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of measured inference iterations.",
    )
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=10,
        help="Number of unmeasured warmup iterations.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Input generation seed.")
    parser.add_argument(
        "--hidden-state-std",
        type=float,
        default=HIDDEN_STATE_STD,
        help="Standard deviation for bounded synthetic floating-point inputs.",
    )
    parser.add_argument(
        "--hidden-state-clip",
        type=float,
        default=HIDDEN_STATE_CLIP,
        help="Absolute bound for synthetic floating-point inputs.",
    )
    parser.add_argument(
        "--padding-fraction",
        type=float,
        default=DEFAULT_PADDING_FRACTION,
        help="Trailing fraction masked as padded for an attention_mask input.",
    )
    parser.add_argument(
        "--no-cpu-fallback",
        action="store_true",
        help="Require every graph node to run without CPU fallback.",
    )
    parser.add_argument(
        "--provider-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Provider-specific option; may be supplied multiple times.",
    )
    parser.add_argument(
        "--log-severity-level",
        type=int,
        choices=range(5),
        default=2,
        help="ORT logging level: 0 verbose through 4 fatal.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.model.is_file():
        raise ValueError(f"model does not exist: {args.model}")
    if args.iterations <= 0:
        raise ValueError("--iterations must be greater than zero")
    if args.warmup_iterations < 0:
        raise ValueError("--warmup-iterations cannot be negative")
    if not np.isfinite(args.hidden_state_std) or args.hidden_state_std <= 0:
        raise ValueError("--hidden-state-std must be positive and finite")
    if not np.isfinite(args.hidden_state_clip) or args.hidden_state_clip <= 0:
        raise ValueError("--hidden-state-clip must be positive and finite")
    if not 0.0 <= args.padding_fraction < 1.0:
        raise ValueError("--padding-fraction must be in the range [0, 1)")


def parse_provider_options(raw_options: list[str]) -> dict[str, str]:
    options: dict[str, str] = {}
    for raw_option in raw_options:
        key, separator, value = raw_option.partition("=")
        if not separator or not key:
            raise ValueError(
                f"invalid provider option {raw_option!r}; expected KEY=VALUE"
            )
        if key in options:
            raise ValueError(f"provider option {key!r} was specified more than once")
        options[key] = value
    return options


def preload_packaged_onnxruntime() -> None:
    ort_spec = importlib.util.find_spec("onnxruntime")
    if ort_spec is None or ort_spec.origin is None:
        raise RuntimeError(
            "onnxruntime-windowsml is not installed in this Python environment"
        )

    ort_dll = Path(ort_spec.origin).parent / "capi" / "onnxruntime.dll"
    if not ort_dll.is_file():
        raise RuntimeError(f"packaged ONNX Runtime DLL does not exist: {ort_dll}")

    # Prevent Windows from resolving the in-box ONNX Runtime DLL first.
    ctypes.WinDLL(str(ort_dll))


def remove_conflicting_winrt_runtime_dll() -> None:
    site_packages = Path(
        str(metadata.distribution("winrt-runtime").locate_file(""))
    )
    runtime_dll = site_packages / "winrt" / "msvcp140.dll"
    if runtime_dll.exists():
        runtime_dll.unlink()


def get_provider_library(provider_name: str) -> Path:
    remove_conflicting_winrt_runtime_dll()

    try:
        import winui3.microsoft.windows.ai.machinelearning as winml
        from winui3.microsoft.windows.applicationmodel.dynamicdependency.bootstrap import (
            InitializeOptions,
            initialize,
        )
    except ImportError as error:
        raise RuntimeError(
            "Windows App SDK projections are unavailable; install "
            "requirements-winml.txt"
        ) from error

    with initialize(options=InitializeOptions.ON_NO_MATCH_SHOW_UI):
        catalog = winml.ExecutionProviderCatalog.get_default()
        provider = next(
            (
                candidate
                for candidate in catalog.find_all_providers()
                if candidate.name == provider_name
            ),
            None,
        )
        if provider is None:
            raise RuntimeError(
                f"{provider_name} is not offered on this system; check the NPU "
                "driver and Windows ML execution-provider installation"
            )

        result = provider.ensure_ready_async().get()
        if result.status != winml.ExecutionProviderReadyResultState.SUCCESS:
            raise RuntimeError(
                f"{provider_name} is unavailable: status={result.status}, "
                f"reason={result.diagnostic_text}, "
                f"error_code={result.extended_error.value}"
            )
        return Path(provider.library_path)


def load_onnxruntime() -> Any:
    preload_packaged_onnxruntime()

    import onnxruntime as ort

    return ort


def register_provider(provider_name: str) -> Any:
    preload_packaged_onnxruntime()
    provider_library = get_provider_library(provider_name)

    import onnxruntime as ort

    ort.register_execution_provider_library(provider_name, str(provider_library))
    print(f"Registered {provider_name} from {provider_library}")
    return ort


def find_npu_device(ort: Any, provider_name: str) -> Any:
    for device in ort.get_ep_devices():
        if (
            device.ep_name == provider_name
            and device.device.type == ort.OrtHardwareDeviceType.NPU
        ):
            return device
    raise RuntimeError(f"{provider_name} registered but exposed no NPU device")


def numpy_dtype(onnx_type: str) -> np.dtype:
    type_map = {
        "tensor(float)": np.dtype(np.float32),
        "tensor(float16)": np.dtype(np.float16),
        "tensor(double)": np.dtype(np.float64),
        "tensor(int64)": np.dtype(np.int64),
        "tensor(int32)": np.dtype(np.int32),
        "tensor(uint16)": np.dtype(np.uint16),
        "tensor(uint8)": np.dtype(np.uint8),
        "tensor(bool)": np.dtype(np.bool_),
    }
    if onnx_type not in type_map:
        raise ValueError(f"unsupported model input type: {onnx_type}")
    return type_map[onnx_type]


def make_inputs(
    session: Any,
    seed: int,
    hidden_state_std: float = HIDDEN_STATE_STD,
    hidden_state_clip: float = HIDDEN_STATE_CLIP,
    padding_fraction: float = DEFAULT_PADDING_FRACTION,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    inputs: dict[str, np.ndarray] = {}
    for model_input in session.get_inputs():
        if any(not isinstance(dimension, int) for dimension in model_input.shape):
            raise ValueError(
                f"input {model_input.name!r} has a dynamic shape: {model_input.shape}"
            )

        dtype = numpy_dtype(model_input.type)
        shape = tuple(model_input.shape)
        if model_input.name == "attention_mask" and np.issubdtype(
            dtype, np.floating
        ):
            value = make_additive_attention_mask(shape, padding_fraction).astype(
                dtype, copy=False
            )
        elif np.issubdtype(dtype, np.floating):
            value = make_hidden_states(
                rng, shape, hidden_state_std, hidden_state_clip
            ).astype(dtype, copy=False)
        elif dtype == np.dtype(np.bool_):
            value = rng.integers(0, 2, model_input.shape, dtype=np.uint8).astype(dtype)
        else:
            value = rng.integers(0, 10, model_input.shape, dtype=dtype)
        inputs[model_input.name] = value
    return inputs


def main() -> None:
    args = parse_args()
    validate_args(args)
    provider_name = PROVIDER_NAMES[args.provider]
    provider_options = parse_provider_options(args.provider_option)

    ort = (
        load_onnxruntime()
        if args.provider == "cpu"
        else register_provider(provider_name)
    )
    session_options = ort.SessionOptions()
    session_options.log_severity_level = args.log_severity_level
    if args.provider == "cpu":
        session = ort.InferenceSession(
            str(args.model.resolve()),
            sess_options=session_options,
            providers=[provider_name],
            provider_options=[provider_options],
        )
    else:
        session_options.add_provider_for_devices(
            [find_npu_device(ort, provider_name)],
            provider_options,
        )
        if args.no_cpu_fallback:
            session_options.add_session_config_entry(
                "session.disable_cpu_ep_fallback", "1"
            )
        session = ort.InferenceSession(
            str(args.model.resolve()),
            sess_options=session_options,
        )
    model_path = args.model.resolve()
    inputs = make_inputs(
        session,
        args.seed,
        args.hidden_state_std,
        args.hidden_state_clip,
        args.padding_fraction,
    )

    for _ in range(args.warmup_iterations):
        session.run(None, inputs)

    start = time.perf_counter()
    outputs = []
    for _ in range(args.iterations):
        outputs = session.run(None, inputs)
    average_latency_ms = (
        (time.perf_counter() - start) * 1_000.0 / args.iterations
    )

    print(f"Model:             {model_path}")
    print(f"Provider:          {provider_name}")
    print(f"Provider options:  {provider_options or '{}'}")
    print(
        "Synthetic inputs: "
        f"std={args.hidden_state_std:g}, clip={args.hidden_state_clip:g}, "
        f"padding={args.padding_fraction:.1%}"
    )
    print(f"Average latency:   {average_latency_ms:.3f} ms")
    for model_output, value in zip(session.get_outputs(), outputs):
        print(f"Output {model_output.name}: shape={value.shape}, dtype={value.dtype}")


if __name__ == "__main__":
    main()
