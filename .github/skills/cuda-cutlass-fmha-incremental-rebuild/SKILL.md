---
name: cuda-cutlass-fmha-incremental-rebuild
description: >
  Use when rebuilding ONNX Runtime CUDA after editing CUTLASS fused-MHA headers
  (onnxruntime/contrib_ops/cuda/bert/cutlass_fmha/*.h such as kernel_forward.h or
  fmha_launch_template.h), or when a header edit "passed" an incremental build but
  test behavior did not change. Explains the nvcc depfile gotcha that produces stale
  Memory-Efficient-Attention (MEA) kernels and binaries, and how to force a correct
  recompile. Also covers disk-space frugality on shared GPU dev boxes.
---

# Incremental rebuilds silently use STALE CUTLASS fused-MHA kernels

> The **general** false-green principles (stale binary, wrong-artifact mtime) are summarised
> in the `ort-test` skill's "False-green taxonomy". This skill is the CUDA/CUTLASS-specific
> detail.

## The gotcha (verification-integrity bug)

`nvcc`-generated depfiles do **not** track the CUTLASS fused-MHA headers under
`onnxruntime/contrib_ops/cuda/bert/cutlass_fmha/` (e.g. `kernel_forward.h`,
`fmha_launch_template.h`). These headers are `#include`d by the `fmha_sm*.cu`
translation units, but the build system does not record that dependency.

Consequence: after you edit one of those headers, an **incremental** `build.sh`:

- does **not** recompile `fmha_sm*.cu`,
- reports `[100%] Built target ...` and exits 0,
- leaves the recompiled artifacts — the `fmha_sm*.cu.o` objects and the
  `libonnxruntime_providers_cuda.so` they link into — **unchanged** (same `mtime` as
  the pre-edit build).

(Do **not** use the gtest test-exe mtime as the stale symptom: in the shared-provider
build the exe `dlopen`s the `.so` and is **not** relinked, so its mtime stays old even
after a *correct* rebuild — see "How to confirm" below. The reliable diagnostic signal
is the `fmha_sm*.cu.o` / `.so` mtime.)

So your "successful" rebuild is running the **old** kernel. Tests that should now
pass (or fail) reflect the previous code, not your edit. This silently invalidates
any FAIL→PASS / PASS→FAIL verification.

## The fix — force recompile the .cu units

Before rebuilding after editing any `cutlass_fmha/*.h` header:

```bash
touch onnxruntime/contrib_ops/cuda/bert/cutlass_fmha/*.cu
```

Then run the normal build command. This forces the `fmha_sm*.cu` translation units
(and downstream binaries) to recompile against your header change.

## How to confirm the rebuild was real (don't trust "[100%] Built")

Confirm that the artifact which actually **links** the recompiled `fmha_sm*.cu.o`
is newer than your header edit.

⚠️ **Do NOT just check the test EXE mtime — it can falsely flag a good build as
stale.** In the shared-provider build configuration (the default here), the CUDA
execution provider is a **shared module**: the recompiled `fmha_sm*.cu.o` link into
`libonnxruntime_providers_cuda.so`, and the `onnxruntime_provider_test` executable
**dlopens** that `.so` — it is **not relinked**. So after a *correct* rebuild the
test exe `mtime` stays **old** while the `.so` advances. Checking the exe alone
would wrongly conclude the build was stale.

Check the right artifact for your link mode:

- **Shared-provider build (default):** the `.so` that links the recompiled `.o` —
  `build/<dir>/<cfg>/libonnxruntime_providers_cuda.so`
- **Statically-linked provider:** the test exe itself (`onnxruntime_provider_test`)

Safest check — `stat` both the recompiled object and the `.so`, and confirm BOTH
are newer than the header edit:

```bash
stat -c '%y %n' onnxruntime/contrib_ops/cuda/bert/cutlass_fmha/kernel_forward.h
# in your build dir, e.g. build/Debug_quickbuild/Debug/:
stat -c '%y %n' libonnxruntime_providers_cuda.so
# and the actual recompiled object (path varies by build dir):
find . -name 'fmha_sm80.cu.o' -exec stat -c '%y %n' {} +
```

If the `.so` (and the `fmha_sm*.cu.o`) timestamps are older than (or equal to) the
header edit, the build was stale — `touch` the `.cu` files and rebuild. The most
reliable signal of all is behavioral: a test that was failing now passes (a stale
binary cannot flip its result).

## Related: pick the right test binary

This is the **CUDA/CUTLASS instance of false-green mode 1** (zero-match / wrong binary) —
see the `ort-test` skill's "False-green taxonomy" for the general principle. In short:
attention/MEA/Flash boundary gtests (e.g. `FlashStructuralEmptyRows*`,
`Attention_Causal_NonPadKVSeqLen_MEA_*`) live in **`onnxruntime_provider_test`**, which CI
runs; `onnxruntime_test_all` does not contain them and gives a false green. Verify the
MEA/Flash boundary fix against `onnxruntime_provider_test`.

## Related: disk frugality on shared GPU dev boxes

Full ORT CUDA builds are large (test binaries ~1 GB each; a build dir can reach
tens of GB). On a shared box, `/home` filling to 100% makes builds fail in
non-obvious places — e.g. `git submodule sync` reporting `No space left on device`
or a `config.lock` error, not an obvious "disk full" at the compile step.

Before a big rebuild, check free space and clean only clearly-stale, regenerable
build directories (old dated experiment dirs). Never delete another agent's active
build dir or anything ambiguous:

```bash
df -h /home
du -sh build/* | sort -h
```
