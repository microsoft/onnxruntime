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

- [ ] Update `onnxruntime/core/providers/telum/README.md`
  - [ ] Use correct CMake flag: `-Donnxruntime_USE_TELUM=ON`
  - [ ] Document required `ZDNN_ROOT`
  - [ ] Document actual selection APIs:
    - Python provider list
    - C API `SessionOptionsAppendExecutionProvider("TelumExecutionProvider", ...)`
  - [ ] Remove or gate claims for unimplemented ops (Softmax/LayerNormalization) until implemented
  - [ ] Document "disable CPU EP fallback" as the way to prove you ran on Telum

- [ ] Align declared support vs actual kernels
  - [ ] In `onnxruntime/core/providers/telum/telum_execution_provider.cc`:
    - [ ] Remove ops from `supported_ops_` that have no kernels OR implement those kernels in Phase 1.
  - [ ] Ensure `GetCapability()` logs include a reason when a kernel is missing (optional but strongly preferred).

- [ ] Resolve BFLOAT16 policy mismatch
  - [ ] Either:
    - [ ] Add `BFloat16` type constraints and implement required CPU post-processing paths
    - [ ] Or reject BF16 in `ValidateDataTypes()` and document it

- [ ] Git hygiene
  - [ ] Add and commit new Telum files:
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

- [ ] Extend Softmax coverage
  - [ ] Add negative tests that prove Telum rejects axis != last dim (with CPU fallback disabled)
  - [ ] Validate behavior on real attention-like shapes (large dims, multiple batches)

- [ ] Extend LayerNormalization coverage
  - [ ] Add more shape tests (rank 1, rank 4+ coercion)
  - [ ] Add tests that omit mean and/or inv_std_dev outputs (optional outputs)

### Acceptance

- [ ] On s390x/z16 with zDNN installed:
  - [ ] `onnxruntime_telum_test` passes
  - [ ] Softmax and LayerNorm tests fail if `session.disable_cpu_ep_fallback=1` and Telum doesn't claim the node

---

## Phase 2: Broadcast Patterns Needed By Real Transformer Exports

### Deliverables

- Telum EP can keep common graphs on Telum without fragmenting into CPU islands due to broadcast-only limitations.

### Tasks

- [ ] Implement limited broadcast support for elementwise ops
  - [ ] Patterns to support (minimal set):
    - [ ] scalar
    - [ ] bias vector `[H]` and `[1,H]` across `[..., H]`
  - [ ] Strategy:
    - [ ] if shapes match: use zDNN elementwise
    - [ ] else: do broadcast on CPU inside the Telum kernel
  - [ ] Update `onnxruntime/core/providers/telum/telum_execution_provider.cc` gating accordingly
  - [ ] Add tests for each broadcast pattern

### Acceptance

- [ ] A representative transformer model graph partitions with large contiguous Telum regions when CPU fallback is disabled.

---

## Phase 3: Performance (Prepacking / Transform Reuse)

### Deliverables

- No repeated zDNN transforms for constant weights/bias per-inference for MatMul/Gemm (and other ops as added).

### Tasks

- [ ] Implement `PrePack(...)` for:
  - [ ] `Gemm` weight matrix (and bias when fusable)
  - [ ] `MatMul` constant operand (typically B)
- [ ] Add kernel-side caching of transformed ztensors (with correct lifetime management)
- [ ] Add targeted perf/regression tests or microbenchmarks (where possible)

### Acceptance

- [ ] Profile shows transforms for constant weights happen once per session initialization, not per inference.

---

## Phase 4: Packaging / CI / Developer Experience

### Deliverables

- Regressions are caught automatically on s390x.
- Docs for building and running Telum tests are copy-pastable.

### Tasks

- [ ] Add s390x build-and-test CI job (GitHub Actions, internal CI, or documented external runner)
- [ ] Add docs page under `docs/execution_providers/` or similar pointing to Telum EP
- [ ] Document a reproducible build environment (container or VM setup) for zDNN + ORT + tests

### Acceptance

- [ ] A PR that breaks Telum compilation or tests is caught before merge.
