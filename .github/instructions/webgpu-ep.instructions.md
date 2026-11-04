---
description: "Implementation and review invariants for the native WebGPU execution provider: shader cache keys, dispatch and bounds, untrusted tensor data, Dawn callbacks, concurrency, portability, registration, and tests."
applyTo: "onnxruntime/core/providers/webgpu/**,onnxruntime/contrib_ops/webgpu/**,onnxruntime/test/providers/webgpu/**,onnxruntime/test/webgpu/**,include/onnxruntime/core/providers/webgpu/**,onnxruntime/core/session/plugin_ep/ep_factory_webgpu.*"
---

# Native WebGPU Execution Provider

Each `Program` generates WGSL that is compiled once per program cache key and reused for every later dispatch with
that key. The EP runs on native Dawn (D3D12, Vulkan, Metal), in browsers through WebAssembly, and as the
`plugin-ep-webgpu` plugin EP. For the review procedure, regression history, and validation commands, load the
`webgpu-ep-review` skill.

## Shader Cache Key

- Anything that changes the generated WGSL must change the cache key, through `CacheHint(...)`, an input or output
  `ProgramTensorMetadataDependency` (`Type`, `Rank`, `Shape`), the component count, the workgroup size, or
  overridable constants. A missing component makes a later call reuse the wrong pipeline and silently compute wrong
  results; the nightly shader-key validation finds this only after merge.
- Encode keyed values exactly. `std::to_string(float)` keeps six decimals, so distinct attribute values can collide.
- Pass per-run values (sizes, strides, offsets, scalar attributes) as uniforms through
  `WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES` and `AddUniformVariables`. Formatting them into WGSL compiles a new
  pipeline for every distinct value.

## Dispatch and Bounds

- Address linear work with `global_idx`, `workgroup_idx`, and `local_idx`, and bound every invocation. Dispatches
  larger than `maxComputeWorkgroupsPerDimension` are reshaped and padded by
  `ProgramManager::NormalizeDispatchGroupSize`, which breaks shaders that index with `global_id` or `workgroup_id`.
- A zero dispatch is an error. Return before `RunProgram` when the output is empty.
- `workgroupBarrier`, `storageBarrier`, and subgroup operations must run in uniform control flow. An early `return`,
  including `GuardAgainstOutOfBoundsWorkgroupSizes`, before a barrier is invalid; predicate the work instead.
- Bounds-check partial tiles on loads and stores. Do not rely on robust buffer access; it is configurable through
  `enableRobustness`.
- Stay within adapter limits: 16 KiB of workgroup storage and 256 invocations per workgroup are the guaranteed
  minimums. Access tensors that can exceed `maxStorageBufferBindingSize` through the shader variable accessors
  (`GetByOffset`, `SetByOffset`, `GetByIndices`) because the EP splits them into binding segments.
- Do not bind one buffer to two bindings of a dispatch when either binding is writable; WebGPU rejects it.

## Untrusted Model Data

- Indices, positions, sequence lengths, and offsets read from tensor data on the GPU cannot be validated on the host
  without a readback. The shader must clamp or skip out-of-range values after negative-index wrapping.
- Validate attributes and host-visible inputs in `ComputeInternal` with `ORT_RETURN_IF` or `ORT_RETURN_IF_NOT`,
  including empty and rank-0 shapes.
- Narrow `int64_t` sizes, dimensions, pads, axes, and offsets to 32-bit uniforms only after a range check or with
  `onnxruntime::narrow`. `static_cast` and `gsl::narrow_cast` truncate silently; an int64-to-int32 truncation in Pad
  caused an out-of-bounds read. Use `SafeInt` for byte sizes.
- Declare every input that the kernel reads on the host with `InputMemoryType(OrtMemTypeCPUInput, index)`.

## Host Interaction, Errors, and Concurrency

- Do not read GPU data back to the host in `ComputeInternal` to make decisions; it stalls the queue and breaks graph
  capture. Keep data-dependent work on the device, for example with `SetIndirectDispatchTensor`. Issue clears and
  copies through the context APIs that graph capture records, not through raw Dawn calls.
- Never throw from a Dawn callback. Store a `Status` in the callback context and return it after the wait.
- Sessions can share one `WebGpuContext`, and with it one program cache, and run concurrently. Kernels must not keep
  per-run mutable state in members, statics, or globals, and per-session options that change WGSL must be keyed.

## Portability

- Gate optional features (`shader-f16`, subgroups, subgroup matrix, subgroup size control) on device support and keep
  a working fallback. Vendor-specific paths need a tested generic fallback: the Windows WebGPU CI tests run on NVIDIA,
  macOS on Apple GPUs, and the Linux WebGPU job only builds.
- Changes must build for native, WebAssembly, and the plugin EP; `plugin-ep-webgpu/paths.txt` lists its sources.

## Registration, Documentation, and Tests

- Register ONNX kernels in `webgpu_execution_provider.cc` and contrib kernels in
  `onnxruntime/contrib_ops/webgpu/webgpu_contrib_kernels.cc`. When adding an opset version, end the previous
  registration at the new version minus one. Type constraints must match what the shader implements.
- Registration changes alter `docs/OperatorKernels.md`. Regenerate it; the Windows GPU kernel documentation workflow
  validates it with WebGPU enabled.
- Every behavior change needs a test that runs on WebGPU, either an `OpTester` test or a test under
  `onnxruntime/test/providers/webgpu/`. Cover each shader variant the change introduces. Excluding
  `kWebGpuExecutionProvider` from a test requires a stated reason.
