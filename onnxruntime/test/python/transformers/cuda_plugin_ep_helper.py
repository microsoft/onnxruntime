# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
# Copyright 2020 The HuggingFace Inc. team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# -------------------------------------------------------------------------
import os
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

import onnxruntime as onnxrt
from onnxruntime import get_build_info


class _CudaPluginRegistrationState:
    attempted = False
    registered = False


CUDA_PLUGIN_EP_NAME = "CudaPluginExecutionProvider"
enable_debug_print = False


def _should_use_cuda_plugin_ep() -> bool:
    return os.getenv("ORT_TEST_GQA_USE_CUDA_PLUGIN_EP", "0") == "1"


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
        pass

    return None


def _is_cuda_plugin_ep_built() -> bool:
    build_info = get_build_info()
    return ", cuda-plugin-ep=" in build_info


def _get_default_cuda_plugin_ep_path() -> str | None:
    # 1) Match currently imported onnxruntime module first to avoid ABI mismatch.
    loaded_onnxruntime_root = Path(onnxrt.__file__).resolve().parent
    loaded_candidate = loaded_onnxruntime_root / "capi" / "libonnxruntime_providers_cuda_plugin.so"
    if loaded_candidate.exists():
        return str(loaded_candidate)

    # 2) Installed wheel location.
    for package_name in ("onnxruntime-gpu", "onnxruntime"):
        package_root = _get_package_root(package_name, "onnxruntime")
        if package_root:
            candidate = os.path.join(str(package_root), "capi", "libonnxruntime_providers_cuda_plugin.so")
            if os.path.exists(candidate):
                return candidate

    # 3) In-tree build location fallback only if running with in-tree onnxruntime.
    loaded_path_str = str(loaded_onnxruntime_root)
    if "build/cuda/Release" not in loaded_path_str:
        return None

    repo_root = Path(__file__).resolve().parents[4]
    candidate = str(repo_root / "build" / "cuda" / "Release" / "libonnxruntime_providers_cuda_plugin.so")
    if os.path.exists(candidate):
        return candidate

    return None


def ensure_cuda_plugin_ep_registered() -> bool:
    if _CudaPluginRegistrationState.attempted:
        return _CudaPluginRegistrationState.registered

    _CudaPluginRegistrationState.attempted = True

    if not _should_use_cuda_plugin_ep():
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

    try:
        onnxrt.register_execution_provider_library(CUDA_PLUGIN_EP_NAME, ep_lib_path)
        _CudaPluginRegistrationState.registered = True
    except Exception as e:
        if enable_debug_print:
            print(f"Failed to register CUDA Plugin EP from {ep_lib_path}: {e}")
        _CudaPluginRegistrationState.registered = False

    return _CudaPluginRegistrationState.registered


def resolve_cuda_plugin_ep(ep: str) -> str:
    # Keep all existing test call-sites unchanged: they pass CUDA EP,
    # and we transparently route to plugin EP when it is built and loadable.
    if ep == "CUDAExecutionProvider" and ensure_cuda_plugin_ep_registered():
        if _is_plugin_provider_type_available():
            return CUDA_PLUGIN_EP_NAME

        if enable_debug_print:
            print(f"{CUDA_PLUGIN_EP_NAME} is not exposed in available provider types. Falling back to {ep}.")
    return ep


def _is_plugin_provider_type_available() -> bool:
    try:
        return CUDA_PLUGIN_EP_NAME in onnxrt.get_available_providers()
    except Exception:
        return False
