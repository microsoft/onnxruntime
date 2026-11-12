# Example Plugin Execution Provider Implementations

This directory contains example code for plugin execution provider (EP) implementations. These EPs are intended to be
used for testing and as reference examples.

## Purpose

- **Demonstration**: Provides reference implementations of custom EPs as plugins, showing how to integrate with ONNX
  Runtime's plugin EP infrastructure.
- **Testing**: Used by ONNX Runtime's unit test suite to validate plugin loading, registration, and EP behaviors.

## Directory Structure

- `example_plugin_ep/`
  Contains a basic compiling plugin execution provider.

- `example_plugin_ep_virt_gpu/`
  Contains a compiling plugin execution provider that registers its own virtual hardware device. Virtual devices can be
  used for cross compiling models for different targets.

- `example_plugin_ep_kernel_registry/`
  Contains a basic plugin execution provider that registers operator kernels with ONNX Runtime, as opposed to compiling
  nodes.

- `plugin_ep_utils.h`
  Common utilities for the example plugin execution provider implementations.

## Usage

- **For Plugin EP Developers**: Use these files as a reference to create your own custom EP plugin.
- **For ORT Developers**: The files are used by ONNX Runtime's unit tests to ensure plugin support works as expected.
  Update them as needed to test updated or additional functionality.

## Notes

- This code is for demonstration and testing purposes. It is not optimized for production use.
- The API usage shown here reflects ONNX Runtime's plugin EP interfaces as of the current version.

## Telemetry

A plugin EP that emits its own telemetry should honor the host application's telemetry settings. Call
`OrtEpApi::GetTelemetryEventsEnabled` (ONNX Runtime 1.33 and later) before each event and skip the event when it
reports `false`. Do not cache the result, because the application can disable telemetry at any time. Treat a host that
predates the API as having telemetry disabled. The EP owner chooses the EP's telemetry provider and destination. See
[Disabling Telemetry](../../../../docs/Privacy.md#disabling-telemetry).

---

For more information, see the ONNX Runtime documentation on
[plugin execution providers](https://onnxruntime.ai/docs/execution-providers/plugin-ep-libraries/).
