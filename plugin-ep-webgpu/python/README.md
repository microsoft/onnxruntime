# WebGPU Plugin EP Python Package

This directory contains the packaging source for the `onnxruntime-ep-webgpu` Python package.

## Prerequisites

- Python 3.11+
- Pre-built WebGPU plugin EP binaries (from CI or a local build)

Install build dependencies:

```bash
pip install -r requirements-build-wheel.txt
```

## Building the wheel

Wheels are built via `build_wheel.py`. Running `pip install` or `pip wheel` directly against this directory is not
supported — the source tree contains `pyproject.toml.in` (a template), not a real `pyproject.toml`.

```bash
python build_wheel.py \
  --binary_dir <path-to-built-binaries> \
  --version <PEP-440-version> \
  --output_dir <output-directory>
```

Example:

```bash
python build_wheel.py \
  --binary_dir ./build/Release \
  --version 0.1.0.dev20260429 \
  --output_dir ./dist
```

The script combines the pre-built plugin EP binaries with the package source to produce a platform-specific wheel.

## Testing

Install the wheel and dependencies in a clean environment, then run the smoke test:

```bash
python -m venv test_venv
source test_venv/bin/activate  # or test_venv\Scripts\Activate.ps1 on Windows
pip install onnx numpy
pip install dist/onnxruntime_ep_webgpu-*.whl  # pulls in onnxruntime>=1.24.4
python test/test_webgpu_plugin_ep.py
```

The wheel declares a runtime dependency on the minimum compatible `onnxruntime` package, so pip will install (or
verify) a compatible core runtime automatically.

The test validates import, EP registration, device discovery, and inference (requires WebGPU-capable hardware for the
inference portion). Set the environment variable `ORT_TEST_VERBOSE=1` to print additional diagnostic information
(environment, available providers, discovered devices, etc.).

## Versioning

The package version is derived from `plugin-ep-webgpu/VERSION_NUMBER` by the packaging pipeline, which produces a
PEP 440 version string.
