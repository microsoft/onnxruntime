# WebGPU Plugin EP Python Package — Build & Test

This directory contains the packaging source for the `onnxruntime-ep-webgpu` Python package.

## Prerequisites

- Python 3.9+
- Pre-built WebGPU plugin EP binaries (from CI or a local build)

Install build dependencies:

```bash
pip install -r requirements-build-wheel.txt
```

## Building the wheel

```bash
python build_wheel.py \
  --binary_dir <path-to-built-binaries> \
  --version <PEP-440-version> \
  --output_dir <output-directory>
```

Example:

```bash
python build_wheel.py \
  --binary_dir /build/Release \
  --version 1.26.0.dev20260410 \
  --output_dir ./dist
```

The script will:
1. Copy plugin binaries into the package directory
2. Stamp the version in `pyproject.toml`
3. Build the wheel
4. Run `auditwheel repair` on Linux for manylinux compliance
5. Verify the wheel was produced
6. Clean up copied binaries and restore `pyproject.toml`

## Testing

Install the wheel and dependencies in a clean environment, then run the smoke test:

```bash
python -m venv test_venv
source test_venv/bin/activate  # or test_venv\Scripts\Activate.ps1 on Windows
pip install onnxruntime onnx numpy
pip install dist/onnxruntime_ep_webgpu-*.whl
python test/test_webgpu_plugin_ep.py
```

The test validates import, EP registration, device discovery, and inference (requires WebGPU-capable hardware for the inference portion).

## Versioning

The package version is derived from `plugin-ep-webgpu/VERSION_NUMBER` by the CI pipeline (`set-plugin-build-variables-step.yml`), which produces a PEP 440 version string:
- **Release**: `X.Y.Z`
- **Dev**: `X.Y.Z.devYYYYMMDD`
