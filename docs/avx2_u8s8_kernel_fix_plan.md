# Plan: Correct Full-Range AVX2 U8/S8 MLAS Kernels

## Issue

The non-VNNI AVX2 U8/S8 kernels use `VPMADDUBSW` followed by `VPMADDWD`.
`VPMADDUBSW` sums adjacent U8-by-S8 products into signed INT16 and saturates
before the result is widened:

```text
255 * 127 + 255 * 127 = 64,770 -> clamped to 32,767
```

This silently produces incorrect `MatMulInteger` results for valid full-range
inputs. The HYDRA reproducer demonstrates that:

- the non-VNNI AVX2 result exactly matches the pair-saturation model;
- safe widened AVX2, AVX-VNNI, AVX512-VNNI, and scalar implementations agree
  exactly; and
- the initial error propagates into materially different model routing scores.

The affected AVX2 paths are:

- QGEMM in
  [`amd64/QgemmU8X8KernelAvx2.asm`](../onnxruntime/core/mlas/lib/amd64/QgemmU8X8KernelAvx2.asm)
  and
  [`x86_64/QgemmU8X8KernelAvx2.S`](../onnxruntime/core/mlas/lib/x86_64/QgemmU8X8KernelAvx2.S).
- QGEMV for `M=1` in
  [`amd64/QgemvU8S8KernelAvx2.asm`](../onnxruntime/core/mlas/lib/amd64/QgemvU8S8KernelAvx2.asm)
  and
  [`x86_64/QgemvU8S8KernelAvx2.S`](../onnxruntime/core/mlas/lib/x86_64/QgemvU8S8KernelAvx2.S).

AVX512 without VNNI is out of scope for this fix due to their age. The relevant physical CPU families are:
- 1st-generation Xeon Scalable (Skylake-SP, 2017);
- Xeon W-2100/W-3100 (Skylake-W, 2017-2019);
- Xeon D-2100 (Skylake-D, 2018);
- 7th- and 9th-generation Core X (Skylake-X, 2017-2018); and
- Core i3-8121U (Cannon Lake, limited-volume release in 2018).

## Proposed Fix

Split each unsigned activation byte into independently safe components:

```text
a_low  = a & 0x7f
a_high = a & 0x80
a      = a_low + a_high
```

Compute both components separately:

```text
dot(a, b) = dot(a_low, b) + dot(a_high, b)
```

Each adjacent pair then fits in signed INT16:

```text
a_low:  [-32,512, 32,258]
a_high: [-32,768, 32,512]
```

The assembly sequence is conceptually:

```asm
low16  = vpmaddubsw(a & 0x7f, b)
high16 = vpmaddubsw(a & 0x80, b)
low32  = vpmaddwd(low16, ones)
high32 = vpmaddwd(high16, ones)
acc32 += low32 + high32
```

Implement this decomposition in both the Windows and Unix QGEMM kernels.
Apply the same fix to QGEMV, or initially route non-VNNI `M=1` operations
through the corrected QGEMM path.

The change must preserve:

- existing packed-A and packed-B layouts;
- prepacked-weight compatibility;
- zero-point and row/column-sum corrections;
- existing QGEMM dispatch structures; and
- the current AVX-VNNI and AVX512-VNNI fast paths.

## Why Existing Tests Do Not Reveal the Issue

The current tests avoid the operand range that can saturate:

- The CPU `MatMulInteger` randomized U8 activation data is restricted to
  `[0, 127]` in
  [`matmul_integer_test.cc`](../onnxruntime/test/providers/cpu/math/matmul_integer_test.cc).
  For two U8-by-S8 products, that restriction bounds the intermediate sum to
  `[-32,512, 32,258]`, which always fits in INT16.
- The generic MLAS test-buffer generator produces byte values only in
  `[21, 63]` in
  [`test_util.h`](../onnxruntime/test/mlas/unittest/test_util.h). QGEMM shape,
  packing, threading, and zero-point coverage therefore still uses values that
  cannot expose U8/S8 pair saturation.
- The explicit MLAS boundary-value fixture covers S8-by-U8, not the affected
  U8-by-S8 orientation. The first operand's signedness determines whether the
  unsafe `VPMADDUBSW` path is used.

Dispatch can hide the problem independently of the input data. Public MLAS or
operator tests on a VNNI-capable CI machine select an AVX-VNNI or AVX512-VNNI
kernel and bypass the non-VNNI AVX2 implementation. On a non-VNNI machine, the
restricted test values still keep `VPMADDUBSW` within range. As a result, the
existing suite can pass in both hardware configurations without exercising the
failing arithmetic.

## Testing

Add tests that force the non-VNNI AVX2 implementation. Tests that only call
the public MLAS entry point can silently exercise VNNI on capable CI machines
and would not validate this fix.

Cover:

- known overflowing pairs, including `255*127`, `255*-128`, and the HYDRA
  `39,100` pair;
- `M=1` to exercise QGEMV and `M>1` to exercise QGEMM;
- K values around packing boundaries: `1`, `2`, `3`, `4`, `5`, `255`, `256`,
  `257`, and `768`;
- N tails around vector boundaries;
- packed and unpacked weights;
- scalar and per-column zero points;
- threaded and non-threaded execution; and
- exact comparison with a widened scalar implementation.

Expand the existing `MatMulInteger` random activation range from `[0, 127]` to
the full `[0, 255]`. The current range in
[`matmul_integer_test.cc`](../onnxruntime/test/providers/cpu/math/matmul_integer_test.cc)
cannot expose the saturation bug.

Finally, rerun the HYDRA one-node and full-model reproductions with non-VNNI
AVX2 forced. Require the first divergent node to change from the saturating
hash to the established scalar/VNNI hash, and require the final routing scores
to match the VNNI result.

## Expected Performance Cost

The corrected AVX2 inner loop will perform approximately twice the
multiply/reduction work, plus masking. A noticeable regression in the
non-VNNI AVX2 U8/S8 kernel is expected. The actual impact must be benchmarked
across representative M/N/K sizes, packed-weight usage, and thread counts.

The cost is justified because:

- the current implementation silently returns incorrect results for valid
  inputs;
- the error is data-dependent and can materially change model decisions;
- VNNI-capable machines retain their existing fast paths without regression;
- packed formats and memory traffic remain unchanged;
- the decomposition should remain faster than scalar fallback or full
  byte-to-INT16 expansion; and
- `reduce_range` is a model-level mitigation, not a correctness fix for
  full-range `MatMulInteger`.

Correctness must be the default. The safe AVX2 implementation can be optimized
further after benchmarks identify the dominant additional instructions.

### End-to-End Performance Summary

A Windows Release benchmark forced only the MLAS U8/S8 QGEMM and QGEMV
dispatch pointers to AVX2 on an Intel Core i9-12900K. Unrelated operators
retained their normal dispatch. The HYDRA model ran sequentially with one
intra-op and one inter-op thread.

The benchmark used deterministic token IDs sampled from the valid model
vocabulary range `[0, 50368)` at exact sequence lengths from 1 through 512.
Inputs were identical between builds, corrected outputs were deterministic, and
blocks alternated in baseline/fixed/fixed/baseline order.

| Tokens | Existing AVX2 P50 | Corrected AVX2 P50 | P50 increase |
|---:|---:|---:|---:|
| 1 | 15.39 ms | 16.43 ms | 6.8% |
| 2 | 17.72 ms | 19.27 ms | 8.7% |
| 4 | 20.30 ms | 22.12 ms | 9.0% |
| 8 | 26.94 ms | 30.97 ms | 15.0% |
| 11 | 29.61 ms | 35.56 ms | 20.1% |
| 16 | 36.37 ms | 45.94 ms | 26.3% |
| 32 | 59.50 ms | 79.56 ms | 33.7% |
| 64 | 106.37 ms | 143.28 ms | 34.7% |
| 128 | 201.46 ms | 279.81 ms | 38.9% |
| 256 | 415.53 ms | 574.49 ms | 38.3% |
| 512 | 933.27 ms | 1262.70 ms | 35.3% |

The end-to-end cost rises from approximately 7-9% for 1-4 tokens to 34-39% for
32-512 tokens as U8/S8 GEMM accounts for a larger share of execution. The
longest lengths used fewer iterations because of their cost, so additional
M/N/K, thread-count, and native non-VNNI measurements are required for final
performance acceptance.

For correctness, the forced baseline reproduced the known saturating
`MatMulInteger` hash. The corrected AVX2 output matched the scalar/VNNI INT32
output byte-for-byte. Final router outputs are FP32 and need not be
byte-identical across different ORT versions or other dispatched kernels.
