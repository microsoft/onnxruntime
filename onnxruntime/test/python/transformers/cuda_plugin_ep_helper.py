# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import logging
import os
import sys
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

import torch

import onnxruntime as onnxrt

CUDA_PLUGIN_EP_NAME = "CudaPluginExecutionProvider"
enable_debug_print = False
logger = logging.getLogger(__name__)


class _CudaPluginRegistrationState:
    attempted = False
    registered = False


def should_test_with_cuda_plugin_ep(default_value: bool = True) -> bool:
    return os.getenv("ORT_TEST_CUDA_PLUGIN_EP", "1" if default_value else "0") == "1"


def _get_package_root(package_name: str, directory_name: str | None = None):
    root_directory_name = directory_name or package_name
    try:
        dist = distribution(package_name)
        files = dist.files or []

        for file in files:
            if file.name.endswith("__init__.py") and root_directory_name in file.parts:
                return file.locate().parent

        if not directory_name:
            for file in files:
                if file.name.endswith("__init__.py"):
                    return file.locate().parent
    except PackageNotFoundError:
        # Some test environments only have an in-tree build, not an installed wheel.
        pass

    return None


def _is_cuda_plugin_ep_built() -> bool:
    build_info = onnxrt.get_build_info()
    if ", cuda-plugin-ep=" in build_info:
        return True

    ep_lib_path = os.environ.get("ORT_CUDA_PLUGIN_PATH", "")
    if ep_lib_path and os.path.exists(ep_lib_path):
        return True

    detected_path = _get_default_cuda_plugin_ep_path()
    return bool(detected_path and os.path.exists(detected_path))


def _get_cuda_plugin_library_name() -> str:
    if sys.platform == "win32":
        return "onnxruntime_providers_cuda_plugin.dll"

    if sys.platform == "darwin":
        return "libonnxruntime_providers_cuda_plugin.dylib"

    return "libonnxruntime_providers_cuda_plugin.so"


def _get_default_cuda_plugin_ep_path() -> str | None:
    library_name = _get_cuda_plugin_library_name()

    # 1) Match currently imported onnxruntime module first to avoid ABI mismatch.
    loaded_onnxruntime_root = Path(onnxrt.__file__).resolve().parent
    loaded_candidate = loaded_onnxruntime_root / "capi" / library_name
    if loaded_candidate.exists():
        return str(loaded_candidate)

    # 2) Installed wheel location.
    for package_name in ("onnxruntime-gpu", "onnxruntime"):
        package_root = _get_package_root(package_name, "onnxruntime")
        if package_root:
            candidate = os.path.join(str(package_root), "capi", library_name)
            if os.path.exists(candidate):
                return candidate

    # 3) In-tree build location fallback. Search under the repo build dir so we
    # can handle different platforms/configurations without hard-coding Release/.so.
    # This assumes that user only builds in one configuration.
    # Recommend to use ORT_CUDA_PLUGIN_PATH if building in multiple configurations.
    repo_root = Path(__file__).resolve().parents[4]
    build_root = repo_root / "build"
    if not build_root.exists():
        return None

    matches = [path for path in build_root.rglob(library_name) if "CMakeFiles" not in path.parts]
    if matches:

        def _sort_key(path: Path) -> tuple[int, int, str]:
            path_str = str(path)
            if "Release" in path.parts:
                config_rank = 0
            elif "RelWithDebInfo" in path.parts:
                config_rank = 1
            elif "Debug" in path.parts:
                config_rank = 2
            else:
                config_rank = 3

            return (config_rank, len(path.parts), path_str)

        return str(sorted(matches, key=_sort_key)[0])

    return None


def ensure_cuda_plugin_ep_registered(default_test_with_cuda_plugin_ep: bool = True) -> bool:
    if _CudaPluginRegistrationState.registered:
        return True

    if not should_test_with_cuda_plugin_ep(default_test_with_cuda_plugin_ep):
        return False

    if not _is_cuda_plugin_ep_built():
        return False

    ep_lib_path = os.environ.get("ORT_CUDA_PLUGIN_PATH", "")
    if not ep_lib_path:
        detected_path = _get_default_cuda_plugin_ep_path()
        ep_lib_path = detected_path if detected_path else ""

    if not ep_lib_path or not os.path.exists(ep_lib_path):
        if enable_debug_print:
            print(f"CUDA Plugin EP library not found: {ep_lib_path}")
        return False

    _CudaPluginRegistrationState.attempted = True

    try:
        onnxrt.register_execution_provider_library(CUDA_PLUGIN_EP_NAME, ep_lib_path)
        _CudaPluginRegistrationState.registered = True
    except Exception as e:
        if "already registered" in str(e).lower():
            _CudaPluginRegistrationState.registered = True
        else:
            try:
                providers = {device.ep_name for device in onnxrt.get_ep_devices()}
            except Exception:
                providers = set()

            _CudaPluginRegistrationState.registered = CUDA_PLUGIN_EP_NAME in providers

            if enable_debug_print and not _CudaPluginRegistrationState.registered:
                print(f"Failed to register CUDA Plugin EP from {ep_lib_path}: {e}")

    return _CudaPluginRegistrationState.registered


def resolve_cuda_plugin_ep(ep: str, default_test_with_cuda_plugin_ep: bool = True) -> str:
    # Keep all existing test call-sites unchanged: they pass CUDA EP,
    # and we transparently route to plugin EP when it is built and loadable.
    if ep == "CUDAExecutionProvider" and ensure_cuda_plugin_ep_registered(default_test_with_cuda_plugin_ep):
        if _is_plugin_provider_type_available():
            return CUDA_PLUGIN_EP_NAME

        if enable_debug_print:
            print(f"{CUDA_PLUGIN_EP_NAME} is not exposed in available provider types. Falling back to {ep}.")
    return ep


def get_cuda_provider_name() -> str | None:
    if not torch.cuda.is_available():
        return None

    resolved_provider = resolve_cuda_plugin_ep("CUDAExecutionProvider")
    available_providers = onnxrt.get_available_providers()

    if resolved_provider in available_providers:
        return resolved_provider

    if "CUDAExecutionProvider" in available_providers:
        return "CUDAExecutionProvider"

    return None


def _is_plugin_provider_type_available() -> bool:
    try:
        return CUDA_PLUGIN_EP_NAME in onnxrt.get_available_providers()
    except Exception as e:
        logger.warning("Failed to query available providers while checking %s availability: %s", CUDA_PLUGIN_EP_NAME, e)
        return False
