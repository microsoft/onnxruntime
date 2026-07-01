# Strided Tensor Support in ONNX Runtime — Investigation & Design

Author: dmitrism
Status: Draft for discussion
Scope: Design only. No code changes proposed in this document.

---

## 1. Motivation

Customers want to feed **strided (non‑contiguous) tensors** as inputs to ORT
inference sessions so they don't have to materialize a contiguous copy of a
data subset (e.g. a sub‑volume / view of a larger buffer, a broadcasted view,
or a transposed view of a host/device tensor).

The intended consumers of such tensors are nodes that have been **claimed and
fused (compiled) by a special EP** — typically a plugin EP that knows how to
walk the strides natively (or that lowers the fused subgraph to a backend
which already handles strides, e.g. CUDA / DML / a custom IHV runtime).

The mainline ORT kernels would generally **not** support strided inputs. So
the system must be able to:

1. Accept strided inputs at the public API boundary.
2. Carry stride information through the graph to the kernels that opt in.
3. **Detect and reject** strided values flowing into kernels that do not
   support them — preferably during session initialization, not at run time.

---

## 2. Current State of Strided Tensor Support in ORT

### 2.1 The `Tensor` class already understands strides

`include/onnxruntime/core/framework/tensor.h` and
`onnxruntime/core/framework/tensor.cc`:

- `Tensor` constructors accept an optional `gsl::span<const int64_t> strides`
  and a `ptrdiff_t offset` (byte offset within `p_data`).
- Members under `#ifdef ENABLE_STRIDED_TENSORS`:
  - `mutable TensorShapeVector strides_;`
  - `bool is_contiguous_ = true;`
  - `gsl::span<const int64_t> Strides() const;` — lazily computes packed
    strides from shape if `strides_` is empty.
  - `bool IsContiguous() const noexcept;`
  - `void SetShapeAndStrides(const TensorShape& new_shape,
    gsl::span<const int64_t> new_strides);`
  - `bool CheckIsContiguous() const;`
- `NumStorageElements()` / `SizeInBytes()` use strides to compute the
  *minimum* backing buffer size (`GetSizeFromStrides`).
- Strided + sub‑byte element types are explicitly disallowed
  (`!dtype_->HasSubElems()` enforced in `Init` / `SetShapeAndStrides`).

`ByteOffset()` / `SetByteOffset()` are **always** available (not gated on
`ENABLE_STRIDED_TENSORS`) and are used by `view.cc` etc. for sub‑buffer
reuse — they are orthogonal to strides.

### 2.2 The build is gated on training

`cmake/CMakeLists.txt`:

```cmake
if (onnxruntime_ENABLE_TRAINING)
  add_compile_definitions(ENABLE_TRAINING_CORE)
  add_compile_definitions(ENABLE_STRIDED_TENSORS)
  ...
endif()
```

**Strided tensor support is currently a training‑only feature.** Inference
builds compile out the strides_ member, the `Strides()` / `IsContiguous()` /
`SetShapeAndStrides` methods, the kernel‑def metadata, the planner
integration, and the strided copy path.

### 2.3 Kernel‑def metadata for strides

`include/onnxruntime/core/framework/kernel_def_builder.h` (under
`ENABLE_STRIDED_TENSORS`):

```cpp
class KernelDef {
  std::vector<int> may_strided_inputs_;
  std::vector<std::pair<int, int>> may_strided_output_map_;  // <input_idx, output_idx>
public:
  const std::vector<int>& MayStridedInput() const;
  const std::vector<std::pair<int, int>>& MayStridedOutput() const;
};

class KernelDefBuilder {
  KernelDefBuilder& MayStridedInput(int input_index);
  KernelDefBuilder& MayStridedOutput(int input_index, int output_index);
};
```

Semantics:
- `MayStridedInput(i)` — the kernel **can** consume a strided i‑th input.
- `MayStridedOutput(i, j)` — the j‑th output **may be a strided view** that
  shares storage with the i‑th input (i.e. the kernel produces a view, not a
  copy).

Today only a handful of kernels declare these flags, all on CUDA/training:

| Kernel | Declaration |
|---|---|
| `Expand` (CUDA) | `MayStridedOutput(0, 0)` |
| `GatherElements` (CUDA) | `MayStridedInput(1)` (indices) |
| `GatherElementsGrad` (CUDA, training) | `MayStridedInput(2)` |

There is **no `MayStridedInput` / `MayStridedOutput` API** exposed via the
plugin‑EP `OrtKernelDefBuilder` C API
(`include/onnxruntime/ep/adapter/kernel_def_builder.h` and
`include/onnxruntime/core/session/onnxruntime_cxx_api.h::KernelDefBuilder`).
Plugin EPs literally cannot declare stride support today.

### 2.4 Allocation planner integration

`onnxruntime/core/framework/allocation_planner.cc` (`PlannerImpl`,
~lines 427–475) walks `MayStridedOutput` of the producer and
`MayStridedInput` of every consumer; if **all** consumers tolerate strided
input *and* the value is not a graph output / external output, it sets
`is_strided_tensor = true` and reuses the producer input's buffer (the
"strided alias" plan). The planner test
`PlannerTest.MayStridedTest2` exercises this.

### 2.5 Cross‑kernel transport

When a kernel writes a strided output, it calls
`output_tensor->SetShapeAndStrides(shape, strides)` — see
`onnxruntime/core/providers/cuda/tensor/expand.cc::FuncExpand`. The
downstream kernel (which declared `MayStridedInput`) receives a `Tensor`
whose `Strides()` / `IsContiguous()` reflect the producer's view.

### 2.6 Strided copy fallback

`onnxruntime/core/framework/copy.h` defines `StridedCopy` /
`DispatchStridedCopy<EnabledTypes>`. `CPUDataTransfer::CopyTensor` (and the
CUDA equivalent) automatically dispatch to the strided path when either
src or dst is non‑contiguous:

```cpp
#ifdef ENABLE_STRIDED_TENSORS
  if (!src.IsContiguous() || !dst.IsContiguous()) {
    return DispatchStridedCopy<element_type_lists::All>(...);
  }
#endif
```

### 2.7 Provider bridge

`provider_interfaces.h` / `provider_wrappedtypes.h` /
`provider_bridge_ort.cc` already proxy `Tensor::Strides()`,
`IsContiguous()`, `SetShapeAndStrides`, and the
`KernelDefBuilder::MayStrided*` methods across the provider DLL boundary —
again under `ENABLE_STRIDED_TENSORS`.

### 2.8 Public API exposure — **none**

This is the critical gap.

- `OrtApi::CreateTensorWithDataAsOrtValue` and
  `CreateTensorWithDataAndDeleterAsOrtValue` accept only
  `(OrtMemoryInfo*, void* data, size_t data_len, int64_t* shape,
  size_t shape_len, ONNXTensorElementDataType)`. **No strides, no byte
  offset.** Internally they go through `CreateTensorImpl`, which calls the
  contiguous `Tensor` constructor.
- `OrtApi::GetTensorTypeAndShape` / `OrtTensorTypeAndShapeInfo` expose
  shape and element type only. There is no `GetTensorStrides`.
- DLPack import (`onnxruntime/core/dlpack/dlpack_converter.cc`) explicitly
  rejects non‑contiguous tensors via `IsContiguousTensor` — even though
  DLPack's whole point is to carry strides.
- The C# `OrtValue.CreateTensorValueWithData` and
  `PinAsTensor` paths refuse `reverseStride` and assume contiguous layout.
- The Python binding has no strides argument on
  `OrtValue.ortvalue_from_numpy` (it always copies / wraps a contiguous
  array; non‑C‑contiguous numpy arrays are made contiguous first).

So: strided tensors exist **inside** the engine for training, but a
customer driving ORT through `OrtApi` / `Ort::Value` cannot create one,
and ORT cannot tell them about one in an output.

### 2.9 What plugin / fused EPs see today

A plugin EP fuses a subgraph and gets called with `OrtValue*` inputs and
outputs from the C API. The `OrtValue` is always built from a contiguous
`Tensor`. Even if we somehow planted a strided `Tensor` inside an
`OrtValue`, the plugin EP has no API to query its strides — `OrtValue` /
`OrtTensorTypeAndShapeInfo` don't expose them.

---

## 3. What "Full" Support Would Require

Goal: a customer can hand ORT a strided `OrtValue` as a session input; the
session feeds it to a fused node owned by a stride‑aware plugin EP; if any
node on the path between the input and that fused node is not stride‑aware,
**either** ORT inserts an explicit "make contiguous" copy **or** session
initialization fails with a clear diagnostic.

The work splits into seven concerns.

### 3.1 Build flag

`ENABLE_STRIDED_TENSORS` is currently coupled to `ENABLE_TRAINING`.
Decouple it:

- Introduce `onnxruntime_ENABLE_STRIDED_TENSORS` CMake option,
  default `OFF` for inference, `ON` whenever training is on.
- In `cmake/CMakeLists.txt`, `add_compile_definitions(ENABLE_STRIDED_TENSORS)`
  when **either** training or the new option is set.
- All `#ifdef ENABLE_STRIDED_TENSORS` blocks already exist and don't need
  to move — they just become available in inference builds when the option
  is on.

This is intentionally an **opt‑in** flag, not a runtime toggle, because it
adds ~3 fields to `Tensor` and a branch in the data‑transfer hot path.

### 3.2 Public C API surface

New API (additive entries appended to the current `OrtApi` table; `ORT_API_VERSION` is bumped/finalized during release preparation per `docs/Versioning.md`):

```c
// Create a strided tensor. strides are in *elements* (not bytes), one
// per dimension, matching shape_len. byte_offset is added to p_data
// before reads/writes. Pass strides == nullptr for a contiguous tensor
// (equivalent to the existing API).
OrtStatus* CreateStridedTensorWithDataAsOrtValue(
    const OrtMemoryInfo* info,
    void* p_data,
    size_t p_data_len,
    const int64_t* shape, size_t shape_len,
    const int64_t* strides,         // may be NULL
    int64_t byte_offset,            // 0 if not used
    ONNXTensorElementDataType type,
    OrtValue** out);

// Read back. *strides_out points into OrtValue-owned memory and is
// valid for the lifetime of the OrtValue. If the tensor is contiguous,
// *strides_out may be NULL and the caller can compute packed strides.
OrtStatus* GetTensorStrides(
    const OrtValue* value,
    const int64_t** strides_out,
    size_t* strides_len_out,
    int* is_contiguous_out);

OrtStatus* GetTensorByteOffset(
    const OrtValue* value, int64_t* byte_offset_out);
```

C++ wrappers (`Ort::Value::CreateStridedTensor`, `GetTensorStrides`,
`IsContiguous`) and the analogous C# / Python wrappers follow.

DLPack import (`dlpack_converter.cc`) should be updated to **preserve**
strides instead of rejecting non‑contiguous tensors, when the build has
`ENABLE_STRIDED_TENSORS`.

### 3.3 Plugin‑EP `OrtKernelDefBuilder` extension

Add to `OrtEpApi`:

```c
OrtStatus* KernelDefBuilder_MayStridedInput(
    OrtKernelDefBuilder*, int input_index);
OrtStatus* KernelDefBuilder_MayStridedOutput(
    OrtKernelDefBuilder*, int input_index, int output_index);
```

Plus the matching `Ort::KernelDefBuilder::MayStridedInput/Output` C++
wrappers and the
`onnxruntime::ep::adapter::KernelDefBuilder` adapter methods.

Crucially, the **fused node** that a plugin EP creates by claiming a
subgraph synthesizes its own kernel via the EP API. We need:

- A way for the EP, when emitting a fused kernel def, to declare which
  *fused inputs* tolerate strides. The natural place is on the
  `OrtKernelDefBuilder` it already uses to describe the fused node.
- Default for fused kernels: **no inputs are strided‑tolerant** (safe).
- For non‑fused (per‑op) plugin kernels, same `MayStridedInput` /
  `MayStridedOutput` API.

### 3.4 Tensor → OrtValue plumbing in session input feeding

Currently, when a user feeds an `OrtValue` to `Run`, ORT calls
`IExecutionFrame::SetExternalInputs` etc. The `OrtValue` holds a `Tensor`,
and the `Tensor` is passed through unchanged. So if we can construct a
strided `Tensor` inside the `OrtValue` (3.2), it already arrives at the
first kernel intact. No plumbing changes required between the API
boundary and the kernel — assuming the planner doesn't intervene
(see 3.5 and 3.6).

### 3.5 Type/shape inference and graph metadata

ONNX has no concept of strides on a `TypeProto.Tensor`. We do **not** want
to extend ONNX. Strides are a runtime/storage attribute, not a logical
type.

Implication: graph‑level shape inference is unchanged. The strided‑ness
of an edge is a **runtime** property, derived from:

1. The user marking a session input as strided (via the new C API).
2. The producer kernel's `MayStridedOutput` flag — the kernel may emit a
   strided output if the planner decides to alias.

We will track strided‑ness at the **edge** level inside the allocation
planner during session init, similar to how `MayStridedTest2` already
verifies. The output of the planner (`SequentialExecutionPlan`) gains
an "edge i is strided" flag (already exists via `is_strided_tensor` on
`AllocPlanPerValue`).

### 3.6 Static enforcement at session initialization

This is the core safety property the customer asked for: *"detect and
enforce the fact that strided inputs/outputs must match ops
expectations."*

Add a new pass — call it `StridedEdgePropagation` — that runs after
kernel resolution and before / during allocation planning:

1. Mark every session input that the user declared strided (or that
   *could* be strided — see 3.6.1) as `strided=true`.
2. For each node in topological order:
   a. Let `K` be its resolved kernel.
   b. For each input edge `e_i`:
      - If `e_i.strided` and `i ∉ K.MayStridedInput()` → handle per
        strategy below.
   c. For each output `j`:
      - If there exists `(i, j) ∈ K.MayStridedOutput()` and `e_i.strided`,
        and the planner chose to alias, then `out_j.strided = true`.
      - Otherwise `out_j.strided = false` (kernel produces a contiguous
        result).

Strategies for an unsupported strided input on edge `e_i`:

- **Strict mode** (default for now): fail session init with a clear
  diagnostic listing the offending node, kernel, EP, and input index.
- **Auto‑contiguify mode** (opt‑in via session option
  `ort_strided_inputs_auto_contiguify=1`): insert a synthesized
  `Identity`‑like "MakeContiguous" node owned by the consumer's EP
  (or by the fallback CPU EP if it has to bounce). This reuses the
  existing `DispatchStridedCopy` path — it already handles
  strided→contiguous on CPU and CUDA.

#### 3.6.1 How does the planner know an input *could* be strided?

Two options:

(a) **Declarative on the input feed** — the customer must tell the session
at `Run` time that an input is strided. But the planner runs at session
initialization, before any `Run`. So we'd need the customer to declare it
on `SessionOptions` (e.g. `AddStridedInput("input_name")`) or via a new
`OrtRunOptions` flag combined with on‑demand re‑planning. The latter is
expensive.

(b) **Conservative** — assume *any* session input *might* be strided and
rely on the static check to either:
- prove every consumer accepts strides, or
- insert a MakeContiguous (auto‑contiguify), or
- fail (strict).

Option (a) is preferred because it gives a deterministic plan and lets us
fail fast. Recommended API:

```c
OrtStatus* SessionOptionsAddStridedInput(
    OrtSessionOptions* options, const char* input_name);
```

This is purely an init‑time hint; at `Run` the actual `OrtValue` is still
free to be contiguous (the kernel doesn't care).

### 3.7 Output strides — surfacing to the customer

If a fused EP node produces a strided output that is also a graph output,
the user must see the strides. Two sub‑cases:

1. **User‑provided output buffer** (`Run` with pre‑bound outputs / IoBinding):
   ORT must verify the user's `OrtValue` strides match the kernel's
   declared output strides. If the user passed a contiguous buffer but the
   producer wants to emit a strided view, ORT either copies (if
   `MayStridedOutput` is just an option) or fails.

2. **ORT‑allocated output**: today the planner allocates contiguous storage.
   A strided view would alias an input, which we already handle. The
   `OrtValue` returned from `Run` will carry the strided `Tensor`. Once
   3.2 ships, the customer can call `GetTensorStrides` to read them.

---

## 4. Suggested Phasing

The total work is large. I suggest four stages, each independently
shippable and testable.

### Stage 1 — Decouple build flag and lock semantics (small)

- New CMake option `onnxruntime_ENABLE_STRIDED_TENSORS`.
- `ENABLE_STRIDED_TENSORS` defined when training **or** the new option.
- No public‑API or kernel changes.
- Add unit tests that build & run the existing `TensorTest.Strided` on
  inference builds with the new option.
- Documentation: write a one‑pager for EP authors explaining what the
  feature is and why it's opt‑in.

**Deliverable:** strided tensors compile into inference builds. No
externally observable behavior change yet.

### Stage 2 — Public C API for strided OrtValue (medium)

- `CreateStridedTensorWithDataAsOrtValue`.
- `GetTensorStrides`, `GetTensorByteOffset`, `IsTensorContiguous`.
- C++ inline wrappers in `onnxruntime_cxx_api.h`.
- C# / Python parity for read‑only consumption initially; full creation
  parity in Stage 4.
- Update DLPack importer to preserve strides under `ENABLE_STRIDED_TENSORS`.
- Tests: round‑trip a strided NumPy view through the C API on CPU; verify
  identical `data_ptr + offset` and strides on the way out (using a no‑op
  passthrough EP or `Identity` kernel marked `MayStridedInput`).

**Deliverable:** customers can construct strided `OrtValue`s. They will
fail at the first non‑stride‑aware kernel — but that failure is now the
*only* missing piece, not the API.

### Stage 3 — Static enforcement & MakeContiguous insertion (medium)

- New `StridedEdgePropagation` pass in session init (see 3.6).
- New `SessionOptionsAddStridedInput` C API.
- New session option `ort_strided_inputs_auto_contiguify` (default `0` =
  strict).
- Re‑use `DispatchStridedCopy` for the auto‑inserted "MakeContiguous" node.
  This already supports CPU and CUDA. For other devices it falls back to
  device→CPU→strided‑copy→CPU→device, which is correct but slow; that's
  acceptable for the auto‑contiguify path because users opted in.
- Tests:
  - Strict: a strided input feeding `Add` (no MayStridedInput) ⇒ session
    init fails with a clear error.
  - Auto: same model with the session option set ⇒ session init succeeds,
    `Run` returns numerically correct results.
  - Mixed: strided input → `Identity` (added MayStridedInput, see Stage 5)
    → fused plugin EP node that consumes the stride directly ⇒ no copy
    inserted, profiler confirms a single fused call.

**Deliverable:** ORT guarantees correctness — a strided input either flows
end‑to‑end through stride‑aware kernels, gets copied at a known boundary,
or is rejected at init.

### Stage 4 — Plugin‑EP API for stride declarations (medium)

- `OrtEpApi::KernelDefBuilder_MayStridedInput` /
  `KernelDefBuilder_MayStridedOutput`.
- Adapter wrappers in `include/onnxruntime/ep/adapter/kernel_def_builder.h`
  and `Ort::KernelDefBuilder`.
- For **fused** nodes specifically: extend the fused‑kernel‑def emit path
  so an EP author can mark fused‑input indices as stride‑tolerant when
  claiming a subgraph. This is what unlocks the customer scenario.
- Sample plugin EP that consumes a strided input directly (e.g. a trivial
  fused "ScaleAdd" that walks strides) plus an end‑to‑end test.
- C# / Python OrtValue creation parity for strided tensors.

**Deliverable:** the customer's stated scenario works end‑to‑end —
strided input → fused node owned by a plugin EP → output, no extra copies.

### Stage 5 — Optional: opportunistic stride support in select contrib kernels (small, ongoing)

Some kernels are nearly free to make stride‑aware because they already
go through a generic `DispatchStridedCopy` or `Expand`‑like loop. Good
candidates:

- `Identity` (CPU & CUDA) — trivially `MayStridedInput(0)` /
  `MayStridedOutput(0, 0)`.
- `Reshape` — only valid on contiguous; should explicitly *not* claim
  `MayStridedInput`. Mention here so we don't accidentally do it.
- `Cast` (CPU element‑wise) — straightforward to add a strided variant.
- `Transpose` — can become a pure metadata op (`MayStridedOutput(0, 0)`)
  for the cases where the permutation can be expressed as new strides.
  This is a meaningful perf win and a useful contrib‑kernel example for
  EP authors.
- `Slice` — same idea as `Transpose` for non‑step slices.

These are individually small PRs and each one shrinks the set of models
that hit a forced "MakeContiguous" insertion at Stage 3. They are not
required for the customer scenario but make it cheap to extend.

---

## 5. Risks & Open Questions

1. **ABI / header change**. New `OrtApi` entries are additive but we still
   need to bump the `OrtApi` version. Standard process; no real risk.

2. **Planner complexity**. The strided alias path in
   `allocation_planner.cc` is already non‑trivial. Stage 3's pass needs to
   compose with it correctly; in particular the producer's
   `MayStridedOutput` decision currently depends on *all* consumers
   accepting strides — that logic stays, but now also has to consider
   strided graph inputs as "producers" with all their dimensions strided.
   Risk: medium. Mitigation: the existing `MayStridedTest*` planner tests
   are a good starting point; extend them.

3. **Sub‑byte types**. `Tensor::Init` already enforces
   `is_contiguous_ || !dtype_->HasSubElems()`. We keep that invariant —
   strided int4/float4 will be rejected at API boundary with a clear error.

4. **Memory aliasing / lifetime**. A user‑supplied strided buffer is
   external memory; its lifetime is the user's problem (same as today's
   `CreateTensorWithDataAsOrtValue`). No new lifetime model needed.

5. **Negative strides / overlapping strides**. PyTorch supports negative
   strides; ORT's existing strided code path does **not** test these.
   Recommendation: at the API boundary, accept only non‑negative strides
   in Stage 2; revisit if a customer asks.

6. **Per‑Run strided‑ness vs. session option**. We chose to declare
   strided inputs at session‑init time (3.6.1, option a). If a customer
   needs to switch a given input between strided and contiguous between
   `Run` calls, they pay nothing extra in the contiguous case — the
   `is_contiguous_` flag short‑circuits the strided code paths — but
   they still go through the StridedEdgePropagation plan (which is
   correct in either case). So this is not actually a limitation.

7. **Fused‑node output strides**. If a fused node's output is consumed by
   another (non‑fused) ORT kernel, that kernel's `MayStridedInput` decides
   whether a copy is needed. This works because the fused node's
   `KernelDef` carries `MayStridedOutput` and feeds the same
   StridedEdgePropagation pass.

---

## 6. Summary

ORT already has a working internal strided‑tensor implementation, used by
the training stack on CUDA. The pieces needed to expose it to inference
customers and to fused plugin EPs are:

| Piece | Today | After |
|---|---|---|
| `Tensor` strides + offset | Yes (training only) | Yes (opt‑in for inference) |
| Strided copy / data transfer | Yes (training only) | Yes (opt‑in for inference) |
| Allocation planner alias path | Yes (training only) | Yes (opt‑in for inference) |
| Public C API to create / read strided OrtValue | **No** | New `CreateStridedTensorWithDataAsOrtValue`, `GetTensorStrides` |
| Plugin‑EP `MayStridedInput/Output` | **No** | New `OrtEpApi` entries |
| Static check that all consumers accept strides | **No** | New `StridedEdgePropagation` pass |
| Auto‑insertion of MakeContiguous | **No** | Reuses `DispatchStridedCopy`, opt‑in session option |
| DLPack carries strides | Strides are rejected | Strides preserved |

Estimated stages: 5, each independently shippable. The minimum viable
customer scenario (strided input straight into a stride‑aware fused
plugin‑EP node) is delivered after Stages 1 + 2 + 3 + 4. Stage 5 is a
quality‑of‑life follow‑up.
