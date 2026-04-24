# WebGPU Plugin Execution Provider

Packaging sources for the ONNX Runtime WebGPU plugin Execution Provider (EP), distributed as a standalone artifact
that plugs into an existing ONNX Runtime installation rather than being built into the main `onnxruntime` binary.

For more information about plugin EPs, see the documentation [here](https://onnxruntime.ai/docs/execution-providers/plugin-ep-libraries/).

## Contents

- [`VERSION_NUMBER`](VERSION_NUMBER) — Base plugin EP version consumed by the CI pipeline. The pipeline derives the
  final package version (release, dev) from this via
  [`tools/ci_build/github/azure-pipelines/templates/set-plugin-build-variables-step.yml`](../tools/ci_build/github/azure-pipelines/templates/set-plugin-build-variables-step.yml).
- [`python/`](python/) — Sources and build script for the `onnxruntime-ep-webgpu` Python wheel. See
  [`python/README.md`](python/README.md) for build and test instructions.

## How it fits together

The plugin EP is built as a shared library (`onnxruntime_providers_webgpu.{dll,so,dylib}`) by the main ONNX Runtime
build (`--use_webgpu shared_lib`). The resulting binaries are then packaged into:

- A Python wheel (`onnxruntime-ep-webgpu`), built from [`python/`](python/).
- A universal package published to the internal ORT-Nightly feed for Windows (x64 / arm64), Linux x64, and macOS
  arm64.

Packaging is driven by the `WebGPU Plugin EP Packaging Pipeline`
([`tools/ci_build/github/azure-pipelines/plugin-webgpu-pipeline.yml`](../tools/ci_build/github/azure-pipelines/plugin-webgpu-pipeline.yml)),
and post-build smoke tests run in the companion `WebGPU Plugin EP Test Pipeline`
([`tools/ci_build/github/azure-pipelines/plugin-webgpu-test-pipeline.yml`](../tools/ci_build/github/azure-pipelines/plugin-webgpu-test-pipeline.yml)).

## Usage

Once installed, the plugin EP is registered at runtime:

```python
import onnxruntime as ort
import onnxruntime_ep_webgpu as webgpu_ep

ort.register_execution_provider_library("webgpu", webgpu_ep.get_library_path())

devices = [d for d in ort.get_ep_devices() if d.ep_name == webgpu_ep.get_ep_name()]
sess_options = ort.SessionOptions()
sess_options.add_provider_for_devices(devices, {})
session = ort.InferenceSession("model.onnx", sess_options=sess_options)
```

See [`python/onnxruntime_ep_webgpu/README.md`](python/onnxruntime_ep_webgpu/README.md) for the user-facing package
documentation (this README is bundled into the wheel).
