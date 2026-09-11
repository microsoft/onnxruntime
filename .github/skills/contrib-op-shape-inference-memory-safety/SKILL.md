---
name: contrib-op-shape-inference-memory-safety
description: "Audit and fix out-of-range output writes in ONNX Runtime operator shape-inference functions. Use when reviewing or fixing a contrib (or standard) op TypeAndShapeInference where a getNumOutputs() guard precedes a write to a higher output index - optional trailing outputs make a smaller output count schema-valid, so getOutputType(index) can run one past the declared outputs at Graph::Resolve."
---

# Contrib-Op Shape-Inference Output-Index Safety

Reusable method for finding and fixing the bug class where an operator's
`TypeAndShapeInference` function guards an output write with `getNumOutputs() > N` but then
writes an output index **greater than** `N`. For a node that declares fewer outputs, the
written index is past the end of the inference context's output vector.

> **Scope**: schema-level shape inference in `onnxruntime/core/graph/contrib_ops/*.cc` and
> `shape_inference_functions.cc`. This runs once during `Graph::Resolve` (model-load time),
> **EP-agnostic** - there is no per-EP (CPU/CUDA/ROCm) kernel duplicate of this code to
> chase. Op *kernels* allocate outputs via the bounds-safe `OpKernelContext::Output(index)`
> and are a separate concern.

## 1. The pattern

```cpp
// onnxruntime/core/graph/contrib_ops/bert_defs.cc  (before)
propagateElemTypeFromInputToOutput(ctx, 0, 0);
if (ctx.getNumOutputs() > 1) {                       // guard says "> 1"
  propagateElemTypeFromInputToOutput(ctx, 0, 1);
  propagateElemTypeFromInputToOutput(ctx, 0, 2);     // but writes index 2
}
```

The guard `getNumOutputs() > 1` admits a node with **exactly 2 outputs** (indices 0, 1), yet
the body writes index **2**. The implication "`> 1` ⇒ index 2 exists" is false: `> 1` only
guarantees indices 0 and 1.

### Why a smaller output count is valid

Trailing outputs declared `OpSchema::Optional` **lower `min_output`**. ONNX derives
`min_output` = number of required outputs, `max_output` = total declared. The model checker
(`checker::check_node`) only enforces `min_output <= N <= max_output`.

| Op | Output decls | min / max | A 2-output node? |
|---|---|---|---|
| `DecoderAttention` | out (req), new_key_cache (Opt), new_value_cache (Opt) | 1 / 3 | passes checker |
| `MultiHeadAttention` | out (req), present_key (Opt), present_value (Opt), qk (Opt) | 1 / 4 | passes checker |
| `DecoderMaskedMultiHeadAttention` | out (req) + 3 Optional | 1 / 4 | passes checker |

So a node with `output=['out','present_key']` is schema-valid, passes the checker, and then
reaches the index-2 write. **A passing checker is not a guarantee the index is in range.**

## 2. The sink (why the write is not caught)

```cpp
// onnxruntime/core/graph/graph.cc  -  InferenceContextImpl
const TypeProto* getInputType(size_t index) const override {
  return node_.InputDefs().at(index)->TypeAsProto();   // .at()  -> bounds-checked
}
TypeProto* getOutputType(size_t index) override {
  return &node_output_types_[index];                   // operator[]  -> NOT bounds-checked
}
```

- `node_output_types_` is sized to `node.OutputDefs().size()` in the `InferenceContextImpl`
  ctor, so for a 2-output node it has 2 elements; `getOutputType(2)` returns one past the end.
- `getInputType` uses `.at()` (would throw on a bad index); `getOutputType` uses raw
  `operator[]` (no check) - the asymmetry is the root cause.
- The call runs at `Graph::Resolve` → `InferAndVerifyTypeMatch` → `RunInferencing`. The
  surrounding `ORT_TRY/ORT_CATCH(const std::exception&)` only catches *thrown*
  `fail_shape_inference`; a raw out-of-range `operator[]` does not throw, so the catch does
  not help.
- Because this is schema-level inference, it is **EP-independent** - no CUDA/ROCm copy.

## 3. Audit technique — always sweep siblings

Do not stop at the reported function. Grep **every** shape-inference guard and compare its
threshold against the **highest output index written before the next guard**.

```bash
git grep -n 'getNumOutputs' -- \
  onnxruntime/core/graph/contrib_ops/*.cc \
  onnxruntime/core/graph/contrib_ops/shape_inference_functions.cc
```

For each `if (ctx.getNumOutputs() > N)` block, find the largest `index` passed to
`propagateElemTypeFromInputToOutput(ctx, _, index)` / `updateOutputShape(ctx, index, _)` /
`getOutputType(index)` inside it. **Rule: the guard must require strictly more outputs than
the highest index written** (write index `k` ⇒ guard must ensure `getNumOutputs() > k`).

Correct exemplars already in the tree to copy:

| Exemplar | Pattern | Why it is safe |
|---|---|---|
| `BaseGroupQueryAttention...` | `if (getNumOutputs() >= 3)` then writes idx 2 | guard covers highest index |
| `PagedAttention...` | nested `> 1` + inner `if (getNumOutputs() != 3) fail_shape_inference` | fails before any write |
| `EmbedLayerNormalizationShapeInference` | `> 2` then writes idx 2 | fixed by PR #28176 (precedent) |
| `SkipLayerNormalizationShapeInference` | each idx `k` guarded by `> k` | per-index guard |

> **Gotcha — conditional writes can hide a vacuous audit.** A write may sit behind an inner
> condition (e.g. `hasInputShape(past_key_index)` before writing index 2). The site is still
> a bug, but you can only *observe* it when that inner condition is also satisfied. Keep this
> in mind both for the audit and for tests (§5).

## 4. Fix patterns

**Point fix (required): raise the guard to cover the highest index written.**

```cpp
// before
if (ctx.getNumOutputs() > 1) { ... writes idx 2 ... }
// after
if (ctx.getNumOutputs() > 2) {  // both present_key (idx 1) AND present_value (idx 2)
  ...
}
```

Justify the threshold with the op's output semantics. For these attention ops the two trailing
outputs - `present_key` (idx 1) and `present_value` (idx 2) for `MultiHeadAttention`,
`new_key_cache` / `new_value_cache` for `DecoderAttention` (see the §1 table for each op's
exact output names) - are a **both-or-neither pair**: there is no valid configuration that
emits one without the other, so requiring all three outputs before populating indices 1 and 2
is behavior-preserving. (`PagedAttention` encodes the same invariant via its nested `!= 3`
check.)

**Defense-in-depth (recommended): bound the sink** so a future author cannot reintroduce the
class.

```cpp
// onnxruntime/core/graph/graph.cc  -  InferenceContextImpl::getOutputType
TypeProto* getOutputType(size_t index) override {
  if (index >= node_output_types_.size()) {
    fail_type_inference("output index ", index, " is out of range; node has ",
                        node_output_types_.size(), " outputs");
  }
  return &node_output_types_[index];
}
```

This mirrors `getInputType`'s `.at()` and the existing bounds checks in the sibling
`DataPropagationContextImpl`. Placing it at the base layer transitively protects the NHWC and
quantization wrapper contexts. After the point fix this branch is unreachable through a normal
model (the guard already prevents the out-of-range index), so it is pure defense-in-depth. Its
failure mode is build-dependent: with exceptions enabled, `fail_type_inference` raises
`InferenceError` (a `std::exception`), caught by the existing `ORT_CATCH(const std::exception&)`
around `RunInferencing` and surfaced as a clean load-time error; under `ORT_NO_EXCEPTIONS` it is
**not** compiled out - ONNX's no-exceptions path prints the message to `std::cerr` and calls
`abort()`, a deterministic fail-fast (consistent with `getInputType`'s `.at()`, which likewise
terminates under no-exceptions). Either way the result is a controlled failure rather than an
out-of-range write.

## 5. Test recipe

Tests live in `onnxruntime/test/contrib_ops/*.cc` and are **auto-globbed** into the
`onnxruntime_provider_test` target by `cmake/onnxruntime_unittests.cmake`
(`test/contrib_ops/*.cc` pattern) - **no cmake edit needed** for a new file. See the
`ort-test` skill for the executable taxonomy (`onnxruntime_provider_test` vs
`onnxruntime_test_all`).

Rules that make the regression test actually guard the fix:

1. **Drive through `Model` + `Graph::Resolve`**, not ONNX's standalone `TestShapeInference`.
   Only the full resolve path constructs the real `InferenceContextImpl` and hits the
   `getOutputType` sink described in §2. A standalone ONNX shape-inference helper uses a
   different context and **bypasses** the sink, so it cannot reproduce the bug.
2. **Negative tests must be NON-VACUOUS** - they must actually enter the write branch on
   pre-fix source. If a write is gated by an inner condition (§3 gotcha), satisfy it: e.g. for
   `MultiHeadAttention`/`DecoderMaskedMultiHeadAttention`, supply a **shaped `past_key`**
   (and `past_sequence_length` / `past_present_share_buffer` as the op requires) so the
   index-2 block runs. A negative test that only supplies `query` skips the block and passes
   even on pre-fix source - regression-proof in name only.
3. **Add positive (all-outputs) cases**: a node with every output present must still infer the
   trailing output types - proves the tightened guard did not over-restrict.
4. **Keep tests throw-free post-fix** so they are valid under `ORT_NO_EXCEPTIONS`. Any case
   that is *expected* to `fail_shape_inference` (throws) must be excluded with
   `#ifndef ORT_NO_EXCEPTIONS`. The "2 outputs must not go out of range" case is throw-free
   after the point fix and is safe in all builds.

**Verify the negative test is non-vacuous (sanitizer A/B)** - the most reliable way to prove a
negative test enters the previously-out-of-range branch: build the test at the **pre-fix**
commit with **AddressSanitizer** and confirm it flags the out-of-range output access; then
confirm it is clean after the fix.

```bash
# Functional run (any Debug build):
cmake --build build/Linux/Debug --target onnxruntime_provider_test -j"$(nproc)"
./build/Linux/Debug/onnxruntime_provider_test \
  --gtest_filter='AttentionOptionalOutputsShapeInferenceTest.*'

# A/B proof (isolated worktree at the pre-fix commit, CPU-only Debug + sanitizer):
git worktree add --detach ../ort-prefix-check <fix_commit>~1
# copy the new test file in, then:
python3 tools/ci_build/build.py --build_dir build/asan --config Debug --parallel \
  --skip_tests --enable_address_sanitizer --skip_submodule_sync \
  --cmake_generator Ninja --target onnxruntime_provider_test
# Pre-fix: the negative tests fail (the sanitizer flags the out-of-range output access).
# Post-fix (cherry-pick the guard fix): all tests pass, no sanitizer report.
```

## 6. Process / wording conventions

- Run **`lintrunner -a`** before pushing so the `CLANGFORMAT` / Python-format gate passes. See
  the `ort-lint` skill.
- Use **correctness/robustness framing** in code, comments, commit messages, and the PR body
  - describe the change as fixing an optional-output guard, not as a security fix. This
  matches repo convention (compare `python-kwargs-setattr-security`) and keeps the PR neutral.

## 7. Audit checklist (per-operator review)

When reviewing or hardening any operator implementation or its shape inference:

- [ ] Read the op's spec - ONNX standard op page, or for a contrib op its `OpSchema`
      registration (`.Input/.Output/.Attr`, and `Optional`/`Variadic` markers). A local ONNX
      checkout has the standard-op spec pages; contrib ops are defined only in ORT.
- [ ] Enumerate **all** inputs, attributes, and outputs, noting which are optional and the
      resulting `min/max` input and output counts.
- [ ] Validate every input/attribute before indexing into it, to avoid out-of-range reads
      (which can cascade into worse failures). Match each output-index write to a guard that
      guarantees the index is in range (§3 rule).
- [ ] Prefer `ORT_RETURN_IF` / `ORT_RETURN_IF_NOT` for validation; use `ORT_ENFORCE` in
      constructors. In shape inference use `fail_shape_inference` / `fail_type_inference`.
- [ ] Use `SafeInt<>` / `narrow<>()` for index and size arithmetic and casts to avoid overflow
      or truncation that yields a wrong index. See `core/common/safeint.h` and
      `docs/Coding_Conventions_and_Standards.md`.
- [ ] Ensure tests build and pass under **no-exceptions** builds; `#ifndef ORT_NO_EXCEPTIONS`
      around any case expected to throw.
- [ ] Exclude EPs known not to support the op, with a comment explaining why.
- [ ] Check whether **other EPs (notably CUDA/ROCm)** implement the same op and whether the
      same issue exists there. (For *shape inference* specifically, the logic is EP-agnostic
      and single-source - confirm there is no kernel-side analogue.)

## References

- **PR #28176** - "Fix ... in EmbedLayerNormalizationShapeInference": the precedent that fixed
  the identical `> 1` → `> 2` primitive in one site; the sibling attention sites were missed,
  motivating the sweep in §3.
- **PR #29268** - this fix: guards corrected in `DecoderAttention` / `MultiHeadAttention` /
  `DecoderMaskedMultiHeadAttention` shape inference, plus the `getOutputType` bounds check and
  non-vacuous regression tests.
- Sibling skill: **`ort-test`** (test executables, `--gtest_filter`, contrib-op test layout);
  **`ort-lint`** (`lintrunner -a`); **`ort-build`** (build flags, ASan).
