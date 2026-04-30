# CUDA Plugin EP Python Package

This directory contains the packaging source for the `onnxruntime-ep-cuda` Python package.

## Building the wheel

Wheels are built via `build_wheel.py`. Running `pip install` or `pip wheel` directly against this directory is not supported because the source tree contains `pyproject.toml.in` instead of a concrete `pyproject.toml`.

```bash
python build_wheel.py \
  --binary_dir <path-to-built-binaries> \
  --version <PEP-440-version> \
  --output_dir <output-directory>
```

The script combines pre-built CUDA plugin EP binaries with the package source to produce a platform-specific wheel.