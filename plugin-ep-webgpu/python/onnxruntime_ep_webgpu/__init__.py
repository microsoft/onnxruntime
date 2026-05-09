"""ONNX Runtime WebGPU Plugin Execution Provider Python Package.

Provides helper functions to locate the plugin EP shared library and
retrieve the EP name for registration with ONNX Runtime.
"""

from __future__ import annotations

import pathlib

__all__ = [
    "get_ep_name",
    "get_ep_names",
    "get_library_path",
]

_module_dir = pathlib.Path(__file__).parent


def get_library_path() -> str:
    """Return the path to the WebGPU plugin EP shared library."""
    candidate_paths = [
        _module_dir / "onnxruntime_providers_webgpu.dll",
        _module_dir / "libonnxruntime_providers_webgpu.so",
        _module_dir / "libonnxruntime_providers_webgpu.dylib",
    ]
    paths = [p for p in candidate_paths if p.is_file()]
    if len(paths) != 1:
        raise RuntimeError(
            f"Expected exactly one WebGPU plugin EP library in {_module_dir}, "
            f"found {len(paths)}: {[p.name for p in paths]}"
        )
    return str(paths[0])


def get_ep_name() -> str:
    """Return the WebGPU Execution Provider name."""
    return "WebGpuExecutionProvider"


def get_ep_names() -> list[str]:
    """Return a list of EP names provided by this plugin."""
    return [get_ep_name()]
