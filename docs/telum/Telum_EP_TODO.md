# Telum EP: Multi-Phase TODO (Tight Checklist)

This is the actionable checklist derived from `docs/telum/Telum_EP_Integration_Review.md`.

Each phase has:

- **Deliverables**: what "done" means
- **Tasks**: concrete file-level work
- **Acceptance**: how to prove it works

---

## Phase 0: Hygiene, Consistency, And Non-Telum Safety

### Deliverables

- Telum docs match the actual build knobs and runtime APIs.
- Telum EP "supported ops" and type checks match the kernel registry.
- Non-Telum builds are unchanged and still build/test.

### Tasks

- [x] Fix zDNN ztensor descriptor lifetime
  - [x] Heap-allocate `zdnn_tensor_desc` in `TensorConverter` (zDNN stores pointers in `zdnn_ztensor`)
  - [x] Free descriptors in `TelumKernel::ZTensorGuard`

- [x] Update `onnxruntime/core/providers/telum/README.md`
  - [x] Use correct CMake flag: `-Donnxruntime_USE_TELUM=ON`
  - [x] Document required `ZDNN_ROOT`
  - [x] Document actual selection APIs:
    - Python provider list
    - C API `SessionOptionsAppendExecutionProvider("TelumExecutionProvider", ...)`
  - [x] Document "disable CPU EP fallback" as the way to prove you ran on Telum

- [x] Remove dead / misleading scaffolding
  - [x] Remove Telum-specific graph transformer files that were not wired into the optimizer pipeline
  - [x] Do not override `IExecutionProvider::Compile(...)` unless Telum actually compiles fused subgraphs

- [x] Align declared support vs actual kernels
  - [x] Keep `supported_ops_`, per-op gating, and kernel registry synchronized
  - [x] Ensure `GetCapability()` logs when a matching kernel is missing

- [x] Resolve BFLOAT16 policy mismatch
  - [x] Add `BFloat16` type constraints across kernels
  - [x] Implement required CPU post-processing paths (e.g., Gemm alpha/beta paths)
  - [x] Add BF16 test coverage for Telum kernels

- [x] Git hygiene
  - [x] Ensure new Telum files are added and committed:
    - `onnxruntime/core/providers/telum/telum_kernel_registry.{h,cc}`
    - `onnxruntime/core/providers/telum/telum_provider_factory_creator.{h,cc}`

### Acceptance

- [ ] macOS (or other non-s390x) builds still configure and compile with Telum OFF (default).
- [ ] `GetAvailableProviders()` does not list Telum when Telum is OFF.

---

## Phase 1: Transformer-Critical Ops (Softmax, LayerNormalization) + Tests

### Deliverables

- Telum EP can execute:
  - `Softmax` (axis == last dim only)
  - `LayerNormalization` (axis == last dim only, static shapes)
- Tests exist and force Telum execution (CPU fallback disabled).

### Tasks

- [x] Extend Softmax coverage
  - [x] Add negative tests that prove Telum rejects axis != last dim (with CPU fallback disabled)
  - [x] Validate behavior on real attention-like shapes (large dims, multiple batches)

- [x] Extend LayerNormalization coverage
  - [x] Add more shape tests (rank 1, rank 4+ coercion)
  - [x] Add tests that omit mean and/or inv_std_dev outputs (optional outputs)
  - [x] Add a test that relies on ONNX default `axis`/`epsilon` attributes (no explicit attrs)

- [x] Add an end-to-end multi-op test (mini transformer block)
  - [x] Build a small ONNX model: `Gemm -> Gelu(MS) -> Softmax -> LayerNormalization`
  - [x] Assert **all nodes** are assigned to Telum (no CPU islands)
  - [x] Compare outputs to CPU EP (tight tolerance)


### Acceptance

- [ ] On s390x/z16 with zDNN installed:
  - [ ] `onnxruntime_telum_test` passes
  - [ ] Softmax and LayerNorm tests fail if `session.disable_cpu_ep_fallback=1` and Telum doesn't claim the node

---

## Phase 2: Broadcast Patterns Needed By Real Transformer Exports

### Deliverables

- Telum EP can keep common graphs on Telum without fragmenting into CPU islands due to broadcast-only limitations.

### Tasks

- [ ] Extend elementwise broadcast coverage (already supported for rank <= 4)
  - [ ] Add tests for common transformer broadcast patterns (mask add, bias add, scalar ops)
  - [ ] Decide whether to support rank > 4 broadcasting (CPU path) or explicitly reject
  - [ ] Consider adding an option to log when a Telum elementwise op took the CPU-broadcast path (debug aid)

### Acceptance

- [ ] A representative transformer model graph partitions with large contiguous Telum regions when CPU fallback is disabled.

---

## Phase 3: Performance (Prepacking / Transform Reuse)

### Deliverables

- No repeated zDNN transforms for constant weights/bias per-inference for MatMul/Gemm (and other ops as added).

### Tasks

- [x] Implement `PrePack(...)` for:
  - [x] `Gemm` weight matrix (input B)
  - [x] `Gemm` bias vector (input C) for the fusable subset (alpha==beta==1, bias vector)
  - [x] `MatMul` constant operand (typically B)
- [ ] Add kernel-side caching of other repeatedly-created transformed tensors (optional)
  - [ ] `MatMul`/`Gemm` zero-bias ztensors when no fused bias is present (avoid per-inference transform)
  - [ ] Other op-specific scratch buffers if needed
- [ ] Add targeted perf/regression tests or microbenchmarks (where possible)

### Acceptance

- [ ] Profile shows transforms for constant weights happen once per session initialization, not per inference.

---

## Phase 4: Packaging / CI / Developer Experience

### Deliverables

- Regressions are caught automatically on s390x.
- Docs for building and running Telum tests are copy-pastable.

### Tasks

- [x] Add s390x build-and-test CI job (GitHub Actions self-hosted runner)
- [ ] Add docs page under `docs/execution_providers/` or similar pointing to Telum EP
- [ ] Document a reproducible build environment (container or VM setup) for zDNN + ORT + tests

### Acceptance

- [ ] A PR that breaks Telum compilation or tests is caught before merge.
