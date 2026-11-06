# Registration

**Owns:** The advertised operator contract and its registration/documentation
consistency. Execution checks belong in the kernel and shader references.
**Update when:** Registration conventions, supported types/domains/opsets, or
kernel-document generation change.

- Register ONNX kernels in
  `onnxruntime/core/providers/webgpu/webgpu_execution_provider.cc` and contrib
  kernels in `onnxruntime/contrib_ops/webgpu/webgpu_contrib_kernels.cc`.
- Opset ranges must be contiguous and non-overlapping. A new version changes
  the prior registration to a range ending at the new version minus one.
  Update both the forward declaration and `BuildKernelCreateInfo` entry.
- Check layout-sensitive registrations in `kMSInternalNHWCDomain` for the
  preferred NHWC layout; changing one registration usually requires the other.
- Type constraints must match the shader. WGSL has no 64-bit integer type:
  a new int64 path must state/test its range assumption and follow
  `GetOpTypeConstraints` and `enableInt64`.
- Declare **every host-read input** with
  `InputMemoryType(OrtMemTypeCPUInput, index)` in every registration and opset
  range for the operator.
- Regenerate `docs/OperatorKernels.md` when registration changes. The
  `windows_gpu_doc_gen.yml` workflow validates it with WebGPU enabled.
- `js/web/docs/webgpu-operators.md` describes deprecated JSEP, despite its name.
  Use the native registry and `docs/OperatorKernels.md` to establish native EP
  support; a JSEP operator entry does not prove a native kernel exists.
- Verify tests reach the changed domain, opset, and types on WebGPU using
  [validation](validation.md).
