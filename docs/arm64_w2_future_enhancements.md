# ARM64 W2 Kernel: Future Performance Enhancements

This document captures two follow-up perf enhancements for the ARM64 W2
(`SQNBIT_CompInt8`, BlkBitWidth=2) NEON kernel that landed on
`hari/asg_perf_2` (the templated R{1,2}×C{1,4,8} DotProd tile grid). Both
are deferred to separate PRs because they each touch the packed-B layout
or add a new translation unit, and they are independent of the
correctness-and-tiling work that already shipped.

Companion docs:

- `docs/arm64_w2_non_lut_kernel_plan.md` — original plan and scope.
- `docs/arm64_w2_non_lut_agent_handoff.md` — implementation handoff
  (some claims pre-date the templated grid; see plan doc for the current
  authoritative ARM64 W2 layout).

The W2 DotProd kernel as of the templated-grid commit is the recipe
shipped in the table below. Both enhancements below extend it; neither
replaces it. The DotProd TU stays the correctness baseline for every
ARM64 host without FEAT_I8MM and is reused by the I8MM dispatch for R1
under the new SMMLA layout (see §1.3).

| Path | A | B in-kernel | Dot | A-sign correction | B-zp correction |
|---|---|---|---|---|---|
| ARM DotProd W4 | int8 | int8 after in-kernel `vsubq_s8(B, zp_b)` | SDOT | none | folded into unpack |
| ARM DotProd W8 | uint8 (+128 offset) | uint8 raw | UDOT | extra 128-correction SGEMM | post-kernel BlkSum SGEMM |
| ARM I8MM W8 | int8 | uint8 raw | USDOT (mixed-sign) | none | post-kernel BlkSum SGEMM |
| **ARM DotProd W2 (current)** | int8 | int8 after in-kernel `vsubq_s8(B, 2)` during 2-bit unpack | SDOT | none | folded into unpack |


## 1. SMMLA-based W2 I8MM TU

### 1.1 Motivation

FEAT_I8MM offers two new int8 instructions on top of DotProd:

- `USDOT` — unsigned-A × signed-B 4-way dot, **same throughput as SDOT**
  on every implementing core. ARM W8 uses it to skip the 128-offset
  correction SGEMM that the W8 DotProd path needs. W2 cannot benefit
  from USDOT alone because the in-kernel `vsubq_s8(B, 2)` step is
  mandatory anyway (unpacking 2-bit weights into int8 lanes).
- `SMMLA` (`vsmmlaq_s32`) — signed 2×8 × 8×2 → 2×2 int32
  matrix-multiply-accumulate. **2× the int8 ops/cycle of SDOT.** This
  is the actual reason to ship a W2 I8MM TU.

This enhancement adds a SMMLA-based TU that targets the R2+ tiles of
the grid where two M-rows are processed together.

### 1.2 B-pack layout

SMMLA's right-hand operand is laid out as
`[B_col0[k=0..7], B_col1[k=0..7]]` — two columns interleaved at an
8-K-element boundary, total 16 bytes per SMMLA. The current W2 packed-B
layout (shared with the AVX-512 reference, single-column per int8x16
vec) is incompatible with this. The I8MM path forks the layout the same
way W8 already does via `Q8BitGemmPackQuantBDataSize<8, IsI8MM>`:

- DotProd-only host → current 1-col-per-vec pack + DotProd TU (no change).
- I8MM-capable host → new 2-col-interleaved pack at 8-K boundary + new
  SMMLA TU.

`Q2BitGemmPackQuantBDataSize` and `SQ2BitGemmPackQuantBDataAndBlkSum`
in `qnbitgemm_kernel_neon.cpp` become I8MM-aware (pick layout based on
`InitializeWithI8MMSupport` at dispatch construction).

### 1.3 R1 viability under the SMMLA layout

The concern: the SMMLA-friendly B-pack layout must also work for the R1
tiles (M=1 batch) that share the same packed buffer on an I8MM host.

Resolution: R1 uses **SMMLA with the A row replicated into both SMMLA
row slots** (e.g. via `vdupq_lane_s64` after loading the A row). SMMLA
then computes a 2×2 int32 block where the top two int32 lanes are the
two dots we want (one per output column) and the bottom two are
duplicates that get discarded. Throughput is identical to 2× SDOT on
the current DotProd layout — no win, no loss for R1.

Alternatives considered and rejected:

- Deinterleave B in-kernel via `vuzp1q_s8` / `vuzp2q_s8` then SDOT —
  adds 2 insts per 16 K-bytes of B and ends up slower than R1 on the
  DotProd layout. Defeats the point of the new pack.
- Keep both B layouts in memory side-by-side — 2× the packed-B
  footprint with no perf benefit (only one layout is ever live on a
  given host).

So the dispatch model is:

| Host capability | B pack | R1 path | R2 path |
|---|---|---|---|
| DotProd only | 1-col-per-vec | SDOT (current) | SDOT (current) |
| FEAT_I8MM | 2-col-interleaved, 8-K boundary | SMMLA + replicated A row | SMMLA (real 2× win) |

M=1 workloads on I8MM hosts therefore land at the same speed as DotProd
hosts; the real SMMLA throughput win is M≥2 only. That is expected and
matches the W8 I8MM precedent (R1 on I8MM is no faster than R1 on
DotProd for W8 either).

### 1.4 Files to add / touch

New:

- `onnxruntime/core/mlas/lib/sqnbitgemm_kernel_neon_int8_i8mm_2bit.cpp`
  — SMMLA W2 TU. Compiled with `-march=armv8.2-a+i8mm` on Linux/macOS
  ARM64 (no flag on Windows ARM64). Mirrors the structure of
  `sqnbitgemm_kernel_neon_int8_i8mm.cpp` (the W8 I8MM TU).
- Pack helper alongside the existing portable W2 pack in
  `sqnbitgemm_kernel_avx512_2bit.{h,cpp}` (TU is portable C++ despite
  the name) that produces the 2-col-interleaved-at-8-K layout.

Touched:

- `onnxruntime/core/mlas/lib/qnbitgemm_kernel_neon.cpp` — make
  `Q2BitGemmPackQuantBDataSize`, `SQ2BitGemmPackQuantBDataAndBlkSum`,
  and the W2 kernel pointer I8MM-aware.
- `onnxruntime/core/mlas/lib/qnbitgemm_kernel_neon.h` — forward decl
  for `SQ2BitGemmKernel_BlkSum_CompInt8_NeonI8MM`.
- `cmake/onnxruntime_mlas.cmake` — add the new TU to both ARM64
  source lists with the `+i8mm` flag entry (mirrors the existing W8
  I8MM TU entries).
- `onnxruntime/test/mlas/unittest/test_sqnbitgemm_2bit_gemm.cpp` —
  add `NeonI8MM_*` test variants gated on `HasArmNeon_I8MM()`.

### 1.5 Expected gains

Inner-loop dot throughput per cycle, holding scalar overhead constant:

- R1: 1× SDOT-equivalent (no change). On par with current DotProd R1.
- R2 C1: ~2× the current DotProd R2 C1 inner loop.
- R2 C4 / R2 C8: same 2× factor at the inner loop; outer-loop scale /
  BlkSum gather is shared with the DotProd kernel, so end-to-end speedup
  is < 2× and shape-dependent. Microbench across (M=2/4/8, N=4/8/16,
  K=512/1024/2048, BlkLen=64) is the right way to quantify before
  shipping.


## 2. Lane-indexed SDOT for the C8 tile

### 2.1 Motivation

The current C8 tile (`Q2Int8GemmRxC_DotProd<BlkLen, NRows, 8>`) issues
one full `vdotq_s32(acc, A_vec, B_vec)` per (M-row, N-col) pair, which
reloads the A vec for each of the 8 N-cols. `vdotq_laneq_s32` lets one
A vec source four 4-lane dots against four different B vecs by indexing
a lane group of A. For C8 (= 2 lane groups of 4 N-cols), that halves
the A loads in the inner loop:

```
// Current C8 inner step (8x full SDOT, 8x A reloads):
acc[m][n] = vdotq_s32(acc[m][n], A[m], B[n]);   // n = 0..7

// Lane-indexed equivalent (8x lane-SDOT, 2x A loads via lane groups):
A0 = A[m];                                       // lanes [0..3] feed n=0..3
acc[m][0] = vdotq_laneq_s32(acc[m][0], B[0], A0, 0);
acc[m][1] = vdotq_laneq_s32(acc[m][1], B[1], A0, 1);
acc[m][2] = vdotq_laneq_s32(acc[m][2], B[2], A0, 2);
acc[m][3] = vdotq_laneq_s32(acc[m][3], B[3], A0, 3);
// lanes [4..7] of A0 reused for n=4..7 (one more vec load on A)
```

The exact win depends on the M-row count of the tile (R1 vs R2 share
the A loads, so R2 C8 amortises further), but on cores that don't
forward the A-vec load across cycles it's a measurable inner-loop win.

### 2.2 B-pack layout change (C8 only)

`vdotq_laneq_s32` requires the four B vecs that share a single A vec to
be laid out so that their `k` dimensions align lane-for-lane with the
A vec. In practice that means **4-N-interleaved B-pack for the C8 tile
only** — N-cols 0..3 of a K-block are stored as four adjacent int8x16
vecs in B; the next four cols 4..7 are the next four vecs; etc.

C1 and C4 keep the current 1-col-per-vec layout (no benefit to changing
those — C4 already uses one A vec per 4-N-col group via 4× full SDOT
and a lane-indexed C4 would not reduce A loads).

So this is the first time we'd have a tile-specific B-pack within the
same W2 TU. Options:

a. **Tile-conditional pack at prepack time** — emit both layouts; C8
   path consumes the interleaved buffer, C1/C4 paths consume the
   current buffer. Doubles W2 packed-B memory. Rejected unless C8 wins
   are large enough to justify.
b. **Single interleaved-layout pack, C1/C4 deinterleave in-kernel via
   `vuzp1q_s8` / `vuzp2q_s8`** — same packed-B size, C1/C4 take a small
   inner-loop hit. Acceptable if the hit is small relative to the C8
   win on a balanced (M, N) workload.
c. **Single interleaved-layout pack, C1/C4 read with strided gather** —
   uses `vld4q_s8` to gather 4 strided lanes per load. Same packed-B
   size, C1/C4 load cost similar to (b), no shuffle insts.

Recommended starting point: **option (c)**, with microbench validation
on a (C1, C4, C8) mix workload before committing. If the C1/C4 hit
exceeds the C8 win on representative shapes, fall back to (a) with the
2× B-pack footprint accepted as the cost.

### 2.3 Interaction with the SMMLA enhancement (§1)

The two enhancements compose cleanly:

- DotProd host: current B-pack OR new C8-interleaved B-pack (depending
  on the §2.2 decision). Both R1/R2 use SDOT / lane-SDOT.
- I8MM host: SMMLA-layout B-pack (2-col-interleaved at 8-K boundary,
  §1.2). Lane-indexed SDOT is **not relevant on this host** — SMMLA
  already provides the M-row amortisation in a different way.

So the C8 lane-SDOT work only affects the DotProd TU and the DotProd
pack helper. It does not touch the SMMLA TU.

### 2.4 Files to touch

- `onnxruntime/core/mlas/lib/sqnbitgemm_kernel_avx512_2bit.{h,cpp}` —
  the portable W2 pack lives here. Add a C8-interleaved layout helper
  (or switch the single layout per §2.2).
- `onnxruntime/core/mlas/lib/sqnbitgemm_kernel_neon_int8_2bit.cpp` —
  change `Q2Int8GemmRxC_DotProd<BlkLen, NRows, 8>` to use
  `vdotq_laneq_s32`. C1 / C4 specialisations unchanged unless
  option (b)/(c) is picked, in which case they pick up a small
  deinterleave step in their inner loops.
- `onnxruntime/test/mlas/unittest/test_sqnbitgemm_2bit_gemm.cpp` —
  the existing 12 R*xC4 tile-coverage shapes plus the C8 entries in
  `kSimdShapes*` already exercise the C8 path; the
  `PackUnpackRoundTrip` tests cover any pack-layout change. No new
  tests needed unless a new public layout helper is exposed.

### 2.5 Expected gains

Inner-loop A-load count for a C8 K-step:

- Current: 1 A load + 8 full SDOTs.
- Lane-indexed: 1 A load + 8 lane-SDOTs.

The SDOT-vs-lane-SDOT instruction throughput is identical on
implementing cores; the win is purely the A-load reduction in
**multi-row tiles** (R2 C8 reuses the same A vec across both M-rows
under both schemes, so the gain there is the C1/C4 pack-layout overhead
being absorbed by the C8 path, not raw A-load savings). On R1 C8 the
savings are exactly the load reduction.

Quantify with a microbench at C8-heavy shapes (N ≥ 8, mixed M, K=1024)
before deciding which pack option from §2.2 to ship.


## 3. Sequencing

Both enhancements are independent and can land in either order. Suggested
order:

1. **SMMLA W2 I8MM TU first.** Larger expected win on R2+ tiles; same
   B-pack-fork pattern as W8 I8MM (low review-novelty risk). Validate
   with the existing `NeonDotProd_*` test pattern reused as
   `NeonI8MM_*`.
2. **Lane-indexed C8 SDOT second.** Smaller, more localised change but
   touches the portable W2 pack (review touches AVX-512 file). Land
   after §1 so the pack-layout-fork code added by §1 is the reference
   for any further fork in §2 option (a).

Either way, microbench the inner-loop change in isolation before
committing — both enhancements have shape regimes where the win is
small or negative, and a microbench harness around a fixed (M, N, K,
BlkLen) grid is the cheapest way to discover those before shipping.
