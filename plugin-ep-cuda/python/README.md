# CUDA Plugin EP Python Package

This directory contains the packaging source for the CUDA plugin EP Python packages:

- `onnxruntime-ep-cuda12` for CUDA 12.x builds
- `onnxruntime-ep-cuda13` for CUDA 13.x builds

Both distributions install the same import module, `onnxruntime_ep_cuda`.

## Building the wheel

Wheels are built via `build_wheel.py`. Running `pip install` or `pip wheel` directly against this directory is not
supported because the source tree contains `pyproject.toml.in` instead of a concrete `pyproject.toml`.

```bash
python build_wheel.py \
  --binary_dir <path-to-built-binaries> \
  --version <PEP-440-version> \
  --package_name <onnxruntime-ep-cuda12-or-onnxruntime-ep-cuda13> \
  --output_dir <output-directory>
```

The script combines pre-built CUDA plugin EP binaries with the package source to produce a platform-specific wheel.
