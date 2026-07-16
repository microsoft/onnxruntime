# ONNX Runtime Telemetry

This document captures the lightweight design for ONNX Runtime telemetry across supported platforms. It is intended to
make the build defaults, platform behavior, and opt-out semantics explicit.

## Goals

- Use ONNX Runtime telemetry to improve product quality in official builds, subject to user consent and applicable
  privacy requirements.
- Keep Windows telemetry on the existing ETW/TraceLogging path.
- Add non-Windows telemetry through the cross-platform 1DS SDK (`cpp_client_telemetry`) for Linux, macOS, Android, and
  iOS.
- Avoid telemetry from local unit-test binaries and CI runs.
- Avoid committing or forwarding a plaintext pipeline telemetry token.

## Platform behavior

| Platform | Provider | Network path | Device ID behavior |
|---|---|---|---|
| Windows | ETW/TraceLogging | ORT emits ETW events; collection depends on an external trace session and user consent. | Existing Windows telemetry behavior. |
| Linux | 1DS SDK | 1DS sends events to Microsoft's telemetry backend over HTTPS. | ORT supplies a hashed persistent UUID from the user cache directory. |
| macOS | 1DS SDK | 1DS sends events to Microsoft's telemetry backend over HTTPS. | ORT supplies a hashed persistent UUID from Application Support. |
| Android | 1DS SDK | 1DS sends events to Microsoft's telemetry backend over HTTPS. | ORT does not override device ID; 1DS uses platform behavior. |
| iOS | 1DS SDK | 1DS sends events to Microsoft's telemetry backend over HTTPS. | ORT does not override device ID; 1DS uses platform behavior. |
| WebAssembly | None | Not supported. | No telemetry provider is built. |

## Build behavior

- `onnxruntime_USE_TELEMETRY` is the CMake switch that includes telemetry support.
- `build.py` and direct CMake builds require an explicit `--use_telemetry` / `-Donnxruntime_USE_TELEMETRY=ON`.
- `build.sh` enables telemetry for native builds and omits it for WebAssembly builds.
- Official non-Windows package builds pass `--use_telemetry` in the relevant build settings or pipeline templates.

## Runtime suppression

- CI and ORT unit-test runs hard-suppress the POSIX 1DS provider so the SDK is not initialized.
- `ORT_RUNNING_UNIT_TESTS=1` is set by ORT test entry points before creating the ORT environment.
- `ORT_TELEMETRY_DISABLED=1` (and other truthy values accepted by the shared environment helper) disables
  non-essential telemetry on supported providers.
- The public telemetry enable/disable APIs continue to gate telemetry events through the common telemetry interface.

## Token handling

- Non-Windows 1DS telemetry uses the encoded in-repo tenant token by default.
- `ONNXRUNTIME_TELEMETRY_TENANT_TOKEN` remains as an explicit CMake/environment override hook for callers that need a
  different token.
- Official pipelines do not define or forward `ONNXRUNTIME_TELEMETRY_TENANT_TOKEN`; this avoids exposing a plaintext
  token in Azure pipeline variables or Docker environment passthrough.
- The tenant token is embedded in client binaries and should not be treated as a secret authorization credential.

## Data shape

- POSIX telemetry mirrors the existing ONNX Runtime telemetry event model where practical.
- Error strings are scrubbed before upload to reduce accidental path or username disclosure.
- Process and hardware metadata is limited to coarse runtime, OS, process, CPU, memory, architecture, and device-class
  fields needed for quality analysis.

## Non-goals

- Replacing Windows ETW telemetry with the 1DS SDK.
- Adding ORT-specific Android or iOS device ID overrides.
- Adding WebAssembly telemetry.
- Introducing plaintext telemetry tokens in CI variables.
