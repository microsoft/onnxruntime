# Small-M INT4 SIMT prototype: A100 results and Apple feasibility

## Decision

**Proceed with the padded SIMT architecture for large MLP projections, not as a
blanket replacement for MatMulNBits.** At M=4, the preferred diagnostic
configuration achieves **1.73x gate/up and 1.57x down**, or **1.67x for the three
MLP projections together**, relative to the unchanged regular FP16 kernel.
M=2 MLP improves 1.28x; M=8 improves 2.07x. The N=1024 projection still regresses
16-25%, so it must not be included in a future universal small-M selection rule.

The literal initial BN16/BK512 implementation only achieved 1.19x combined M=4
MLP speedup. A single, source-motivated change to the shared-A layout was
decisive: insert one unused `vec4<f16>` after each block32. This changes the
logical K-lane address stride from 64 to 72 bytes without changing B loads,
dequantization, arithmetic, barriers, or thread ownership. BN16 then reached
1.55x combined MLP speedup. The already-tested BN8 alternative, with the same
padding, reached 1.67x and improved the long-K down projection.

Production sources, shaders, selectors, objects, and libraries are unchanged.
The implementation is under `tools/python/webgpu_small_m_int4/`; only its
standalone executable has the explicit `simt` selection. No commits or pushes
were made. No speculative-decoding end-to-end benchmark was run.

## Architecture

The requested initial configuration is preserved as the unpadded control:
128 threads, BN16, BK512, eight logical K lanes per output column, and
BM=M at M=2/4/8. M=1 uses BM1, M=3 uses BM4, and M=5/7 use BM8 with guarded
activation loads and output writes.

For the preferred **BN8/BK512, A-block-padding=1** configuration:

| Component | Ownership / behavior |
|---|---|
| Thread mapping | `column_local = local_idx / 16`, `k_lane = local_idx % 16` |
| Workgroup tile | BM rows by eight output columns; all 128 threads participate |
| K ownership | Each lane handles one block32 per BK512 iteration |
| Accumulators | BM separately emitted FP32 scalar registers per thread |
| B | One `vec4<u32>` load per block32: 16 bytes containing 32 unsigned INT4 values |
| Scales / zero | Block32 FP16 scale and implicit zero 8, or packed explicit uint8 zero |
| Dequantization | Nibble extraction, FP16 zero subtraction and scale multiplication, then promotion to FP32 |
| A staging | Cooperative `vec4<f16>` loads; nine vector slots per block32, of which eight hold data |
| Reuse | Each dequantized B value is reused across every row in the workgroup |
| Reduction | Shared FP32 partials, one final barrier, then 16 partials summed per valid output |
| Epilogue | Optional FP32 bias addition and one final FP16 conversion |
| Dispatches | One; no A prepack, global scratch, atomics, or separate reduction dispatch |

BN16 instead has eight K lanes, processes two blocks per lane per outer
iteration, and reduces eight partials per output. Both configurations use the
same number of shared partials: `BM * 128` FP32 values.

The FP32 accumulation is intentional, including when the provider's
`enableMatmulFp32Accumulation` option is false. A separate unchanged
regular-FP32 control was measured. Weight dequantization is still FP16, not
silently upgraded to an FP32 weight multiply.

Supported fixtures cover block32/64/128, the existing single-scale convention,
byte-padded explicit zero points, bias, K/N tails, vector and higher-rank A,
and broadcasting of shared 2D weights. Constant weight-matrix indexing is
implemented but not separately exercised with a nonzero index. Per-row
indirect weights, activation-row replay, INT2/INT8, and FP32 A are explicitly
rejected rather than silently falling back.

### Difference from previous row-reuse experiments

This is not just an extra M loop in the original N/K mapping. The original
regular path has a one-row by eight-column workgroup and BK1024 traversal.
The initial candidate changes N ownership to BN16, reduces logical K lanes
to eight, stages the whole BM-by-BK tile, keeps all row accumulators as FP32
scalars, and does a single cooperative final reduction. The BN8 tuning
variant still differs in K ownership, BK512 traversal, register accumulation,
and shared-A layout; it does not restore the original geometry.

Merely sharing B across rows was insufficient: the unpadded versions largely
reproduced the earlier 1.16-1.22x class of improvement. Correcting shared-A
addressing, rather than another accumulation/reduction micro-optimization,
produced the materially different result.

## Benchmark method

- Checkout: `ed8f79bcef8b073d83270a45d68046a72a5166bd`.
- GPU: NVIDIA A100-SXM4-80GB, 108 SMs; Vulkan; driver 580.159.04.
- Existing static build: `build/webgpu-pipeline-lifetime/Release`.
- All 272 matching real matrices are used: 32 attention K/V, 48 linear QKV,
  64 gate, 64 up, and 64 down. A complete rotation has 10,620 MiB of packed
  weights plus scales, much larger than A100 L2.
- Weights come from
  `/workspace/models/qwen3.8-27b-int4-webgpu-paged/model.onnx`.
  The current fused export does not contain the requested standalone N=1024
  projections, hence this original export supplies the complete shape set.
- M=2/4 MLP activations are the captured target activations under
  `build/mlp-up-epilogue-20261021/current-wall`.
  **M=8 repeats the M=4 capture twice; it is not an actual M=8 target capture.**
  Attention activations are deterministic synthetic FP16.
- All modes use identical inputs and weights for each condition. Inputs,
  initializers, and bound outputs are GPU-resident during execution.
- Full runs use 32 warmup rotations and 15 measured rotations. Mode order
  rotates with M. Each number is the median across rotations of the mean GPU
  time per projection in its group, not the best individual weight matrix.
- Native GPU timestamps are used. All automatic-path dispatches, including
  its A prepack, are summed. SIMT's staging and final reduction are inside
  its sole timed dispatch.
- Real CPU readbacks fence completed calls outside shader timestamp intervals;
  upload/readback and host launch costs are not reported as shader latency.
  These are isolated GPU costs, not wall-clock decoding latency.
- CPU fallback is disabled, and CPU-provider records and GPU program names
  are checked. Forced SIMT emits only `DiagnosticSmallMSimt`; forced regular
  emits only `MatMulNBits`.
- Sampled SM/memory clocks during every full benchmark measurement window
  were 1410/1593 MHz. No clock settings or permissions were changed.

### Preferred padded BN8 results

Times are **microseconds per projection**. Speedup is matched regular/SIMT.
`Auto` is the unchanged automatic path from the separate full-default run
with the same matrices/activations; it is not the go/no-go baseline.

| M | Projection | K x N | Regular | Padded SIMT | Speedup | Auto |
|---|---|---|---:|---:|---:|---:|
| 2 | K/V | 5120 x 1024 | 32.0 | 37.0 | 0.86x | 32.1 |
| 2 | Linear QKV | 5120 x 10240 | 86.2 | 68.4 | 1.26x | 86.2 |
| 2 | Gate | 5120 x 17408 | 131.2 | 99.9 | 1.31x | 131.2 |
| 2 | Up | 5120 x 17408 | 131.5 | 100.0 | 1.32x | 131.5 |
| 2 | Down | 17408 x 5120 | 141.4 | 115.9 | 1.22x | 141.5 |
| 4 | K/V | 5120 x 1024 | 35.2 | 44.1 | 0.80x | 353.5 |
| 4 | Linear QKV | 5120 x 10240 | 149.1 | 91.0 | 1.64x | 374.5 |
| 4 | Gate | 5120 x 17408 | 237.1 | 137.2 | 1.73x | 435.9 |
| 4 | Up | 5120 x 17408 | 237.1 | 137.2 | 1.73x | 435.9 |
| 4 | Down | 17408 x 5120 | 246.8 | 157.1 | 1.57x | 1218.8 |
| 8 | K/V | 5120 x 1024 | 46.2 | 54.2 | 0.85x | 356.4 |
| 8 | Linear QKV | 5120 x 10240 | 272.4 | 133.4 | 2.04x | 376.2 |
| 8 | Gate | 5120 x 17408 | 448.8 | 210.8 | 2.13x | 437.2 |
| 8 | Up | 5120 x 17408 | 448.7 | 210.6 | 2.13x | 437.2 |
| 8 | Down | 17408 x 5120 | 455.1 | 231.3 | 1.97x | 1221.0 |

The three MLP roles have equal matrix counts. Their combined matched
speedups are **1.280x / 1.671x / 2.073x at M=2/4/8**.
There are no M=2 or M=8 MLP regressions. The N=1024 regressions are real:
**15.7% / 25.1% / 17.3%** more time at M=2/4/8.

The M=4 padded-BN8 p10-p90 intervals across rotation means are:

| Projection | Regular, us | SIMT, us |
|---|---:|---:|
| Linear QKV | 148.946-149.150 | 90.746-91.121 |
| Gate | 236.938-237.209 | 137.062-137.238 |
| Up | 237.019-237.253 | 137.012-137.228 |
| Down | 246.656-246.966 | 156.938-157.259 |
| K/V | 35.069-35.425 | 43.981-44.212 |

These are descriptive intervals, not independent-run confidence intervals.
All rotation samples and clock samples are retained in JSON.

### Initial architecture and bounded tuning

Only two additional BN/BK geometries were screened: BN8/BK512 and
BN16/BK1024. After identifying the shared-A stride issue, both already-used
BN8/BN16 BK512 geometries were measured with padding. No broad sweep was run.

M=4 latencies in microseconds:

| Projection | Unpadded BN16/BK512 | Unpadded BN8/BK512 | Unpadded BN16/BK1024* | Padded BN16/BK512 | Padded BN8/BK512 |
|---|---:|---:|---:|---:|---:|
| Linear QKV | 116.0 | 120.5 | 116.0 | 92.4 | 91.0 |
| Gate | 197.7 | 193.3 | 196.5 | 139.1 | 137.2 |
| Up | 197.6 | 193.8 | 197.0 | 139.1 | 137.2 |
| Down | 211.1 | 192.1 | 196.5 | 185.5 | 157.1 |
| K/V | 56.1 | 44.5 | 55.0 | 59.8 | 44.1 |

All columns except * use all matching matrices. * is a four-matrix-per-role
screen with 128 warmups and 15 repeats; it is not a full all-layer result.
Each run's own regular control is retained in its result JSON.

The original unpadded BN16 run also measured regular-FP32 accumulation:

| M | Linear QKV | Gate | Up | Down | K/V |
|---|---:|---:|---:|---:|---:|
| 2 | 91.3 | 140.2 | 140.3 | 152.7 | 32.5 |
| 4 | 160.6 | 255.8 | 255.7 | 265.3 | 36.5 |
| 8 | 295.6 | 486.9 | 487.0 | 493.2 | 48.8 |

Thus the preferred SIMT does not gain by using lower-precision accumulation
than its controls. Its decision baseline remains the faster regular FP16 path.
The much slower current automatic subgroup-matrix path must not be used to
inflate the claimed target-verification benefit.

## Workgroups, resources, and limiting factors

Calculated workgroup counts per shared-weight batch:

| K x N | Regular M=2 / 4 / 8 | Initial BN16, all M | Preferred BN8, all M | BK512 iterations |
|---|---:|---:|---:|---:|
| 5120 x 1024 | 256 / 512 / 1024 | 64 | 128 | 10 |
| 5120 x 10240 | 2560 / 5120 / 10240 | 640 | 1280 | 10 |
| 5120 x 17408 | 4352 / 8704 / 17408 | 1088 | 2176 | 10 |
| 17408 x 5120 | 1280 / 2560 / 5120 | 320 | 640 | 34 |

The candidate has two workgroup barriers per K tile and one final reduction
barrier: 21 or 69 explicit barriers for these K values. The regular BK1024
path has five or 17 outer iterations. The BK1024 screen halves candidate
outer iterations but does not provide the large gain obtained by A padding.

Theoretical source-level B/scale decode work is one pass per M tile,
versus M passes for one-row workgroups. Actual DRAM savings are not assumed
to equal M: caches can service repeated regular-kernel loads.

Compiler measurements use the existing isolated
`VK_KHR_pipeline_executable_properties` interposer, not hardware stall counters.
BN8 and BN16 have identical register/shared-memory counts within each layout:

| Program / layout | M | Registers per thread | Shared bytes | Logical FP32 row accumulators |
|---|---:|---:|---:|---:|
| SIMT unpadded | 2 | 32 | 3072 | 2 |
| SIMT unpadded | 4 | 40 | 6144 | 4 |
| SIMT unpadded | 8 | 48 | 12288 | 8 |
| SIMT padded | 2 | 32 | 3328 | 2 |
| SIMT padded | 4 | 32 | 6656 | 4 |
| SIMT padded | 8 | 40 | 13312 | 8 |
| Regular FP16 | all | 33 | 2560 | not the candidate layout |
| Regular FP32 | all | 32 | 3072 | not the candidate layout |

All these reported stack sizes are zero. Padded BN8 binary sizes are
11648/14336/19584 bytes at M=2/4/8. The driver's local-memory statistic is
implausible (`0x1000000000` bytes/thread), so it is not evidence of spills
or their absence. No valid spill measurement is claimed.

**Evidence-based attribution:**

- **Shared-A layout/compiler allocation was a major initial limitation.**
  With the same BN16, BK512, workgroup count, dequantization, FP32 math, and
  synchronization, padding changes M=4 gate/up from about 197.7 to 139.1 us.
  All 77 output fixtures remain bitwise identical between padded and
  unpadded versions at each BN. Register allocation also drops 40 to 32
  at M=4 and 48 to 40 at M=8.
- The usual A100 32-bank, four-byte bank model predicts serialization for
  the original 64-byte stride: K lanes repeatedly hit the same bank pairs
  at different addresses. The 72-byte stride spreads those addresses.
  This is a source-derived explanation supported by the ablation, **not
  a measured bank-conflict counter**. The simultaneous compiler register
  change prevents assigning the entire gain uniquely to bank conflicts.
- **Parallelism matters for long-K/moderate-N.** At M=4, padded BN8 doubles
  down-projection workgroups from 320 to 640 and improves 185.5 to 157.1 us,
  with the same measured registers/shared bytes. It slightly increases
  some M=8 gate/up times relative to padded BN16, but wins their combined
  MLP time because down improves.
- **Narrow-N grid underfill remains.** Initial N=1024 has only 64 groups
  for 108 SMs. BN8 supplies 128, still far fewer concurrent warps than the
  regular multi-row grid. The remaining regression is not solved by
  shared weight reuse.
- Halving K-loop/barrier count without fixing A layout gave only modest
  gains, so synchronization count alone does not explain the initial
  failure. B unpack/dequantization is unchanged by the successful padding
  ablation; there is no evidence that optimizing it alone would reproduce
  that gain. Reduction overhead is included but was not separately timed.

Hardware performance counters were unavailable under the environment's
admin-only profiling policy and missing `CAP_SYS_ADMIN`. No permissions
were changed. Occupancy percentages, cache hit rates, DRAM bytes, and
individual stall categories were not measured.

## Correctness and numerical behavior

**77 GPU fixtures passed for each of unpadded BN16, unpadded BN8, padded
BN16, and padded BN8**, each compared with regular and regular-FP32.
**Ten host reference tests passed.**

Coverage includes M=1/2/3/4/5/7/8; K=1/7/31/32/33/63/95/96/100/511/512/
513/544/1056/5120/17408; N boundaries around 8/16/32; every nibble; explicit
nonzero zeros; odd zero-point block counts and per-column byte padding;
block64/128; flat scales; single scale; bias; rank3/rank4/vector A;
cancellation; large/small magnitudes; and zeros.

The N=1048577 fixture also exceeds the one-dimensional dispatch limit and
tests the extra padded groups introduced by dispatch normalization. The
logical-workgroup guard precedes buffer reads and explicit barriers.

The reference decodes **bytes**, not the GPU's u32/vector unpack sequence.
It computes both a true FP64 weight multiply/accumulation and FP64
accumulation after independently rounding dequantized weights to FP16.
The candidate is checked against an association-aware FP32 forward-error
bound plus final FP16 rounding; bitwise equality to regular is not required.

Worst errors over the synthetic fixtures:

| Comparison | Maximum absolute error | Maximum relative L2 |
|---|---:|---:|
| SIMT vs FP64 | 0.4655612 | 0.0005894032 |
| SIMT vs FP64, FP16-rounded weights | 0.4655612 | 0.0003417742 |
| Regular FP16 vs FP64, finite cases | 1.6582892 | 0.0013242300 |
| Regular-FP32 vs FP64 | 0.4655612 | 0.0005894032 |

The maximum absolute candidate error is **normal final FP16 rounding**:
reference `-1033.4655611515045`, candidate `-1033.0`, exactly equal to the
FP16-rounded reference. Small-product maximum absolute error is
`2.38418579e-7`. Preferred padded BN8 versus regular-FP32 has maximum
synthetic relative L2 `5.20262943e-7`.

Cancellation fixtures deliberately distinguish accumulation range:

- K=5120: regular FP16 produces 68 positive infinities. Each lane's partial
  can remain finite, but the serial FP16 final sum overflows before the
  cancelling negative contributions.
- K=17408: regular FP16 produces 68 NaNs; lane partials can already become
  opposite-signed infinities.
- SIMT and regular-FP32 produce exact zero in both fixtures.

Each product is exactly +/-224, making the finite mathematical sum zero.
These are existing FP16 accumulation limitations, not discrepancies hidden
by a relaxed tolerance. The prototype does not change that existing kernel.

**Real weights:** every output of all 272 matrices at all three M values
is finite and compared with regular. FP64 references additionally cover
first/middle/last matrices per role: 15 matrices per M, 45 in each full
configuration, not all 272.

For the preferred padded BN8:

| M | SIMT vs FP64 max absolute | SIMT vs FP64 max relative L2 | SIMT vs regular max absolute | SIMT vs regular max relative L2 |
|---|---:|---:|---:|---:|
| 2 | 0.0580122 | 0.0002759151 | 0.125 | 0.0012294529 |
| 4 | 0.0580122 | 0.0002543152 | 0.125 | 0.0012316483 |
| 8 | 0.0580122 | 0.0002543152 | 0.125 | 0.0012316483 |

On the sampled real matrices the full-weight and FP16-weight references
coincide. Non-power-of-two synthetic scales separately exercise
dequantization rounding. Sampled regular FP16 maximum absolute error is
0.1830122, and maximum relative L2 is 0.0011342130.

For context, the unchanged automatic M=4/8 subgroup-matrix path uses FP16
result fragments and reaches sampled maximum absolute error 5.0580122 and
relative L2 0.0181883350. It is not a numerical baseline for FP32 accumulation.
No existing automatic shader was modified to change this behavior.

The unchanged regular and regular-FP32 output files remain bitwise identical
across all 77 fixtures before and after the isolated ShaderHelper/padding
changes. Padded and unpadded SIMT outputs are also bitwise identical at
each fixed BN across all 77 fixtures.

## Apple M4/M5 compatibility

Exact ordinary ORT Dawn dependency:

```text
tag:     v20261013.193955
commit:  f63db59722e8c1c0da9a3bcea0e40fb5c817a059
SHA1:    f62c128edcc33e0ba6b090d6550db10c4f237a42
archive: https://github.com/google/dawn/archive/refs/tags/v20261013.193955.zip
```

The checksum-verified pinned Linux Tint used here has SHA256:

```text
cbb929d54465c5585b844366762ac9dd5acf776206ff792313cbf73b2ca23cfd
```

**Source-confirmed / validated on Linux:**

- The actual executed candidate WGSL contains only `enable f16;`, no
  subgroup-matrix extension, subgroup builtins, subgroup operations, or
  subgroup-size requirement.
- An isolated copied ShaderHelper suppresses its otherwise automatic
  unused subgroup extension/builtin emissions only for this program.
  Other programs retain their generated behavior.
- **56 distinct actual padded-variant shaders** passed pinned Tint WGSL
  resolution and MSL generation. The three exact final timed M=2/4/8
  BN8 shaders were additionally rechecked. Across all initial/tuned/padded
  experiments, 113 distinct actual shaders passed both phases.
- Padded shared memory is 3328/6656/13312 bytes at M=2/4/8. The maximum is
  below even the 16-KiB WebGPU minimum. Workgroup size 128 also stays below
  the standard minimum supported maximum of 256 invocations.
- Pinned Dawn's Metal `PhysicalDeviceMTL.mm:744` enables `ShaderF16`.
  Its Apple-family limits table at lines 885-886 provides up to 1024 threads
  and 32768 bytes for the relevant newer Apple families; lines 973-977
  expose those compute limits.

**Likely compatible, not device-proven:** ordinary half/float arithmetic,
storage buffers, shared arrays, uniform workgroup barriers, and integer
unpack operations have an MSL lowering and fit the source-confirmed resource
budget. No unsupported A100 16x8x16 mixed-precision matrix fragment is needed.

**Requires Apple hardware:** Metal SDK compilation, actual Dawn/ORT device
creation and pipeline compilation, binding/validation behavior, FP16
rounding/denorm and cancellation behavior, numerical tolerances, compiler
register allocation, occupancy, and speed. Successful MSL emission is not
Metal SDK validation or runtime correctness. Linux has no Metal SDK/xcrun;
no Apple GPU was accessed. **No A100 speedup is extrapolated to Apple.**

The benchmark executable and monitoring are Linux/Vulkan/NVIDIA-specific.
Running the same experiment on Apple needs a Metal runner and monitoring
adaptation, not merely copying the Linux executable.

## Estimated target-verification effect

This is a **what-if GPU-time projection**, not measured end-to-end performance,
wall-clock latency, or speculative acceptance. Assume only the three MLP
projections use padded BN8 and every other operator remains unchanged.

For each role and M:

```text
saved_time = historical_regular_time * (1 - measured_SIMT / measured_regular)
```

Historical controlled target totals were 88.8008 ms at M=4 and 57.7862 ms at
M=2, a 31.0146-ms widening penalty. Historical MLP projection totals were
46.1746 and 25.8622 ms.

| Quantity | M=2 | M=4 |
|---|---:|---:|
| Estimated gate saving, ms | 1.9887 | 6.3685 |
| Estimated up saving, ms | 2.0243 | 6.4187 |
| Estimated down saving, ms | 1.6441 | 5.7559 |
| Estimated total MLP saving, ms | 5.6571 | 18.5431 |
| Estimated resulting target GPU time, ms | 52.1291 | 70.2577 |

The estimated M=4 reduction is **18.54 ms / 20.9%** of target GPU time.
The estimated M=2-to-M=4 widening penalty becomes **18.13 ms**, down
**12.89 ms / 41.5%**. This excludes any benefit or regression in attention,
does not apply the prototype to N=1024, and does not assume improvements
in launch overhead, graph capture, activation fusion, or acceptance rate.

## Reproduction and recorded results

Run from `/workspace/onnxruntime`, activate the existing `.venv`, and use
fresh experiment output directories: the harness deliberately refuses to
overwrite an existing result directory. Commands below identify the
recorded runs; use a new `--output` name to repeat them.

```bash
source .venv/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

# Passed: compile copied MatMulNBits/ShaderHelper objects and standalone runner.
python tools/python/webgpu_small_m_int4/build.py \
  --output build/webgpu-small-m-simt-pad

# Passed: 10 tests.
python -m pytest -q tools/python/webgpu_small_m_int4/test_reference.py

# Passed: 77 fixtures for each command, with both regular controls.
python tools/python/webgpu_small_m_int4/experiment.py \
  --executable build/webgpu-small-m-simt-pad/diagnostic \
  --a-block-padding 1 correctness \
  --output build/webgpu-small-m-simt/correctness-pad
python tools/python/webgpu_small_m_int4/experiment.py \
  --executable build/webgpu-small-m-simt-pad/diagnostic \
  --bn 8 --a-block-padding 1 correctness \
  --output build/webgpu-small-m-simt/correctness-pad-bn8

# Passed: complete all-272-matrix M2/M4/M8 benchmark plus FP64 samples.
python tools/python/webgpu_small_m_int4/experiment.py \
  --executable build/webgpu-small-m-simt-pad/diagnostic \
  --bn 8 --a-block-padding 1 benchmark \
  --modes simt regular --warmup 32 --repeat 15 --reference \
  --output build/webgpu-small-m-simt/full-pad-bn8

# Passed: same full run for the specified BN16 geometry with A padding.
python tools/python/webgpu_small_m_int4/experiment.py \
  --executable build/webgpu-small-m-simt-pad/diagnostic \
  --a-block-padding 1 benchmark \
  --modes regular simt --warmup 32 --repeat 15 --reference \
  --output build/webgpu-small-m-simt/full-pad-bn16

# Passed: original unpadded BN16 baseline, including regular-FP32 and auto.
python tools/python/webgpu_small_m_int4/experiment.py benchmark \
  --modes regular regular-f32 simt auto --warmup 32 --repeat 15 --reference \
  --output build/webgpu-small-m-simt/full-default

# Passed: compiler statistics, not clean latency measurements.
python tools/python/webgpu_small_m_int4/experiment.py \
  --executable build/webgpu-small-m-simt-pad/diagnostic \
  --bn 8 --a-block-padding 1 resources \
  --output build/webgpu-small-m-simt/resources-pad-bn8
python tools/python/webgpu_small_m_int4/experiment.py \
  --executable build/webgpu-small-m-simt-pad/diagnostic \
  --a-block-padding 1 resources \
  --output build/webgpu-small-m-simt/resources-pad-bn16

# Passed: 56 distinct padded shaders resolved and emitted MSL.
python tools/python/webgpu_small_m_int4/validate_metal.py \
  --tint /root/.copilot/session-state/f35f1f98-881e-41bd-b257-9c4c9ca8c6fa/files/apple-tint-build/tint \
  --dump build/webgpu-small-m-simt/correctness-pad/simt.wgsl \
    build/webgpu-small-m-simt/correctness-pad-bn8/simt.wgsl \
    build/webgpu-small-m-simt/full-pad-bn16/m2/simt.wgsl \
    build/webgpu-small-m-simt/full-pad-bn16/m4/simt.wgsl \
    build/webgpu-small-m-simt/full-pad-bn16/m8/simt.wgsl \
    build/webgpu-small-m-simt/ablate-pad-bn8/m4/simt.wgsl \
  --output build/webgpu-small-m-simt/metal-pad

# Passed: exact final timed M2/M4/M8 shaders, also included in the 56 above.
python tools/python/webgpu_small_m_int4/validate_metal.py \
  --tint /root/.copilot/session-state/f35f1f98-881e-41bd-b257-9c4c9ca8c6fa/files/apple-tint-build/tint \
  --dump build/webgpu-small-m-simt/full-pad-bn8/m2/simt.wgsl \
    build/webgpu-small-m-simt/full-pad-bn8/m4/simt.wgsl \
    build/webgpu-small-m-simt/full-pad-bn8/m8/simt.wgsl \
  --output build/webgpu-small-m-simt/metal-final

# Passed.
ruff check tools/python/webgpu_small_m_int4
ruff format --check tools/python/webgpu_small_m_int4
clang-format --dry-run --Werror \
  tools/python/webgpu_small_m_int4/diagnostic.h \
  tools/python/webgpu_small_m_int4/runner.cc
git diff --check
```

Ruff emits the repository's existing warning about removed rule UP038.
The final finite-real-output assertion was additionally exercised with
`final-harness-smoke` (one matrix per group, M=4, four warmups, three repeats);
its short-run timings are not used in any performance claim.

The initial executable's SHA256 is
`eaea564e12b0e69be013da70769eb6967fa9e9a6bc6a208e33a46d7ec559d4b0`;
the padding-capable executable's is
`2b2504357124933e6ac8719215ebf466c8443da9dd57846b1379f373276273e2`.
Both build manifests record compile/link commands and source hashes.
`build/webgpu-small-m-simt/diagnostic-initial.h` preserves the exact initial
header matching its manifest; the current header matches the padding build.

Important raw artifact directories under `build/webgpu-small-m-simt/`:

| Directory | Evidence |
|---|---|
| `correctness-f16-only`, `correctness-bn8` | Original unpadded 77-fixture results and outputs |
| `correctness-pad`, `correctness-pad-bn8` | Padded 77-fixture results, actual WGSL, outputs |
| `full-default` | Full regular / regular-FP32 / unpadded BN16 / automatic timings and references |
| `full-bn8` | Full unpadded BN8 comparison |
| `full-pad-bn16`, `full-pad-bn8` | Full padded all-layer comparisons and references |
| `tune-bn16-bk1024` | Bounded four-matrix BK1024 screen |
| `resources-default`, `resources-bn8`, `resources-pad-bn16`, `resources-pad-bn8` | Compiler register/shared-memory reports |
| `metal-default`, `metal-variants`, `metal-pad`, `metal-final` | Actual WGSL, emitted MSL, compiler commands and results |

Generated models use private regular external-data files; source models are
never changed. Each full three-M model set costs approximately 32 GiB.
Generated weights, profiles, dumps, and binaries are intentionally outside
the source change.

The recorded `full-default` command uses the preserved initial executable.
For a fresh build containing only `webgpu-small-m-simt-pad/diagnostic`, add
`--executable build/webgpu-small-m-simt-pad/diagnostic --a-block-padding 0`
before `benchmark` to reproduce the same unpadded configuration.

## Focused next step

1. Retain padded BN8/BK512 as the portable large-MLP candidate and BN16 as
   its control. Do not enable either automatically in production.
2. A diagnostic full-model replay restricted to the measured large shapes
   is now justified. Confirm the actual paths and target GPU-time savings,
   then compare logits and speculative acceptance; isolated numerical
   accuracy does not establish acceptance equivalence.
3. Keep N=1024 on regular in such a future experiment. Any narrow-N work
   needs a different parallelism strategy, not an assumption that weight
   sharing always wins.
4. On real Apple M4/M5 hardware, run correctness first, including subnormals,
   cancellation, tails, and explicit zeros, then compare padded/unpadded
   SIMT using the same real projection shapes. Measure Metal compiler
   allocation and runtime rather than transferring A100 conclusions.

No production integration or further parameter sweep is warranted before
those diagnostic checks.
