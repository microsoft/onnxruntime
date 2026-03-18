# ScanVarLen: Design and Implementation

This document describes the `com.microsoft.Scan` contrib operator ("ScanVarLen"),
a variant of the ONNX Scan op that supports variable-length scan outputs.

## 1. Operator Schema

ScanVarLen is registered in the `com.microsoft` domain (opset 1). Its schema
mirrors ONNX Scan v9 with two differences:

- An optional `output_lengths` input is prepended before the variadic inputs.
- The `scan_output_directions` attribute is omitted (always forward).

```
Inputs:
  0: output_lengths      — Optional tensor(int64). One entry per scan output
                           specifying the total concat-axis size, enabling
                           pre-allocation of the final output tensor.
  1: initial_state_and_scan_inputs — Variadic (all tensor types)

Outputs:
  0: final_state_and_scan_outputs  — Variadic (all tensor types)

Attributes:
  body                  — Subgraph executed each iteration (required)
  num_scan_inputs       — Number of scan inputs M (required)
  scan_input_directions — Optional list of M flags (0=forward, 1=reverse)
  scan_input_axes       — Optional list of M axis indices for scan inputs
  scan_output_axes      — Optional list of K axis indices for scan outputs
```

**Concatenation semantics.** Unlike standard Scan (which stacks per-iteration
outputs with a new sequence dimension), ScanVarLen concatenates them along
the axis specified by `scan_output_axes` (default 0). The subgraph output
has the variable dimension at this axis; it may vary across iterations while
all other dimensions must match. For example, with `scan_output_axes=1` and
a 3D subgraph output `[D0, D1_i, D2]`, the final output is `[D0, Σ D1_i, D2]`.

This is conceptually consistent with standard Scan v9, where each iteration
contributes a slice of size 1 along the scan axis (and the dimension does not
exist in the subgraph output). In ScanVarLen the dimension already exists and
each iteration's contribution may be larger than 1.

## 2. Shape Inference

`VarLenScanInferenceFunction` (in `contrib_defs.cc`) performs type and shape
inference. It validates that `num_scan_inputs` is non-negative and does not
exceed the variadic input count, then:

- **Loop state variables**: type and shape are propagated directly from the
  corresponding input to the output, with subgraph output shapes merged in.
- **Scan outputs**: the element type comes from the subgraph output. The shape
  is built in a single pass — the subgraph output shape with the concat-axis
  dimension (at the position given by `scan_output_axes`) replaced by an unknown
  dimension, since the total size depends on all iterations.

## 3. Kernel Class Hierarchy

ScanVarLen reuses the `Scan<9>` implementation via subclassing:

```
Scan<9>  (core/providers/cpu/controlflow/scan.h)
  │
  │  Protected members:
  │    use_var_len_output_        (false for Scan<9>)
  │    num_non_variadic_inputs_   (0 for Scan<9>)
  │
  └── ScanVarLen  (contrib_ops/cpu/controlflow/scan_var_len.h)
        use_var_len_output_ = true
        num_non_variadic_inputs_ = 1
```

`ScanVarLen` is a one-line class that calls a protected constructor on `Scan<9>`,
setting the two flags before `Init()` runs. It shares `Scan<9>::Compute` and all
infrastructure; the flags control which execution path is taken at runtime.

## 4. Input Offset

The non-variadic `output_lengths` input sits at position 0, shifting all variadic
inputs by 1. This offset (`input_offset_ = num_non_variadic_inputs`) is applied
wherever `ScanImpl` accesses variadic inputs from `OpKernelContext`:

- `ValidateSubgraphInput`, `ValidateInput`, `SetupInputs`, `CreateLoopStateVariables`
- `OutputIterator::Initialize` (for reading loop state input shapes)
- `CreateFeedsFetchesManager` (feed name list starts after non-variadic inputs)

For standard `Scan<9>`, the offset is 0 and all these paths are unchanged.

## 5. Execution Paths

ScanVarLen always has `use_var_len_output_ = true`. `ScanImpl::Execute`
dispatches between two paths based on whether `output_lengths` is provided:

```
output_lengths not provided
  → AllocateLoopStateOutputs (loop state only)
  → IterateSequenceVarLen (collects per-iteration scan outputs)
  → ConcatenateScanOutputs (merges into final output)

output_lengths provided
  → AllocatePreAllocOutputs (loop state only)
  → ExecutePreAlloc (allocates scan outputs on first iteration using
    output_lengths, then writes directly each iteration)
```

(Standard `Scan<9>` takes a separate path that pre-allocates all outputs
via `OutputIterator` and uses `IterateSequence` with sliced writes. That
path requires identical shapes across iterations and is unaffected by the
ScanVarLen changes.)

### 5.1 VarLen Path (collect and concatenate)

Only loop state outputs are pre-allocated. For scan outputs, each iteration's
subgraph allocates fresh memory. The resulting `OrtValue`s are collected into
`vector<vector<OrtValue>>`. After all iterations, `ConcatenateScanOutputs`
validates dimension consistency, computes the total concat-axis size, allocates
the final output, and copies the data using axis-aware strided copies (which
reduce to simple sequential `memcpy` when the concat axis is 0).

### 5.2 PreAlloc Path (direct write)

Only loop state outputs are pre-allocated initially. On the first iteration,
the subgraph output shape is used together with `output_lengths[i]` to allocate
the final scan output tensor with the concat-axis dimension set to
`output_lengths[i]`. Subsequent iterations write data directly into the
correct position in the buffer using strided copies. A bounds check on each
write guards against `output_lengths` being too small, and a post-loop check
verifies that the total concat-axis dimension equals `output_lengths[i]`
(catching the case where it is too large). This avoids per-iteration
allocation overhead while still supporting variable-length outputs.

## 6. Key Shared Infrastructure

| Component | Role | Shared? |
|-----------|------|---------|
| `scan::detail::Info` | Counts of inputs, outputs, loop state vars, scan I/O | Yes — extended with `num_non_variadic_inputs` |
| `OutputIterator` | Manages sliced writes to pre-allocated output buffers | Yes — extended with `input_offset` |
| `LoopStateVariable` | Double-buffered loop state cycling across iterations | Yes — unchanged |
| `IterateSequence` | Standard iteration loop with OutputIterator | Scan<9> only |
| `IterateSequenceVarLen` | Iteration loop that collects per-iteration outputs | ScanVarLen only |
| `CreateFeedsFetchesManager` | Maps node inputs to subgraph feed names | Yes — respects input offset |

## 7. File Map

| File | Contents |
|------|----------|
| `core/providers/cpu/controlflow/scan.h` | `Scan<>` template, `Info` struct, protected members |
| `core/providers/cpu/controlflow/scan_9.cc` | `ScanImpl` class with all execution paths |
| `core/providers/cpu/controlflow/scan_utils.h` | `OutputIterator`, `AllocateOutput`, `Info` |
| `core/providers/cpu/controlflow/scan_utils.cc` | Iteration functions, `OutputIterator` impl |
| `contrib_ops/cpu/controlflow/scan_var_len.h` | `ScanVarLen` kernel class (one-line subclass) |
| `contrib_ops/cpu/controlflow/scan_var_len.cc` | Kernel registration |
| `core/graph/contrib_ops/contrib_defs.cc` | Schema + `VarLenScanInferenceFunction` |
| `test/providers/cpu/controlflow/scan_test.cc` | All Scan and ScanVarLen tests |

## 8. Test Cases

Nine `ScanVarLen` tests cover the new operator (all in `scan_test.cc`):

**VarLen path (no output_lengths):**
- Three shape/type inference variants: shape in main graph vs. subgraph,
  with and without type info in the subgraph
- Scalar (0-dim) loop state variable
- Outer scope access (implicit inputs)
- Input reversal (`scan_input_directions`)

**PreAlloc path (with output_lengths):**
- Basic pre-allocation
- Scalar loop state with pre-allocation

**Output axis:**
- `OutputAxis1`: concatenation along axis 1 of a 2D subgraph output,
  verifying the strided copy logic for non-zero axes

The existing 29 standard Scan tests (14 Scan8 + 15 Scan9) continue to pass,
confirming the shared code changes are backward-compatible.
