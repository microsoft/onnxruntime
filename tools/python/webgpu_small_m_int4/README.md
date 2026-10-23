# Experimental small-M INT4 SIMT

The kernel is built into the WebGPU contrib provider, but is **explicitly opt-in**.
There is no automatic SIMT selection or device-specific policy. Unset
`ORT_WEBGPU_SMALL_M_DIAGNOSTIC` (or set it to `auto`) to retain ORT's normal
subgroup-matrix, DP4A, wide-tile, and regular selection.

## Source layout

| Location | Contents |
| --- | --- |
| `onnxruntime/contrib_ops/webgpu/quantization/matmul_nbits_small_m_simt.{h,cc}` | Kernel configuration, validation, dynamic WGSL generation, and dispatch |
| `onnxruntime/contrib_ops/webgpu/quantization/matmul_nbits.cc` | Explicit diagnostic routing; normal selection is unchanged |
| `onnxruntime/core/providers/webgpu/` | Per-program subgroup-declaration opt-out, including cache identity |
| `onnxruntime/test/contrib_ops/matmul_nbits_small_m_simt_test.cc` | Standard provider-target correctness, routing, defaults, and rejection tests |
| `onnxruntime/test/webgpu/matmul_nbits_benchmark.cc` | Optional C API benchmark executable |
| `onnxruntime/test/webgpu/vulkan_pipeline_stats.cc` | Optional Linux/glibc Vulkan compiler-statistics interposer |
| This directory | Python references, benchmark orchestration, and offline compiler diagnostics |

The shader remains dynamically generated in C++, like other WebGPU programs.
Its scalar per-row accumulator unrolling, FP16 dequantization, FP32 accumulation,
barriers, reduction, and final FP16 conversion are preserved. There are no copied
production translation units, hand-extracted compiler commands, or generated
production C++ files. CMake's existing WebGPU contrib glob includes the kernel.

SIMT does not use subgroup operations. Its program opts out of unused subgroup
extensions and builtins through `ProgramBase`, rather than a shader-name special
case. Other programs retain their default declarations and cache keys.

## Configuration and supported inputs

`ORT_WEBGPU_SMALL_M_DIAGNOSTIC=regular` forces regular `MatMulNBits`;
`simt` explicitly requires `MatMulNBitsSmallMSimt`. Unsupported forced inputs
fail rather than silently selecting another kernel. Invalid modes/settings fail.
These process-wide diagnostic controls must not be changed concurrently with
inference.

The original defaults are 128 threads, BN16, BK512, and an unpadded shared-A
block. BM is M for M=1/2/4/8, BM4 for M=3, and BM8 for M=5/6/7.
Each thread has BM scalar FP32 accumulators and reuses dequantized weights across
rows. The workgroup stages A as `vec4<f16>` and guards K, N, partial-M, and
normalized-dispatch tails. No prepack or global scratch is needed.

| Variable | Default | Accepted values |
| --- | --- | --- |
| `ORT_WEBGPU_SMALL_M_BN` | 16 | 8, 16, 32 |
| `ORT_WEBGPU_SMALL_M_BK` | 512 | 256, 512, 1024 |
| `ORT_WEBGPU_SMALL_M_A_BLOCK_PADDING` | 0 | 0: 64-byte stride; 1: 72-byte stride |

Padding adds one unused `vec4<f16>` per staged block32. It changes neither thread
ownership nor arithmetic. At BK512, M=2/4/8 use 3/6/12 KiB unpadded and
3.25/6.5/13 KiB padded. Storage is checked against the actual device limit.

Supported inputs are FP16 A/scales/bias/output, uint8-packed INT4, M=1–8,
positive K/N, implicit zero 8 or byte-padded uint8 zero points, block sizes
divisible by 32, the existing single-scale convention, tails, and higher-rank A.
Constant weight indexing is implemented but nonzero indices remain separately
unverified. Indirect per-row weights and activation-row replay are rejected.
INT2/INT8 and FP32 inputs are unsupported. SIMT always accumulates in FP32,
independent of `enableMatmulFp32Accumulation`; regular/automatic paths retain
their own accumulation behavior.

## Build and standard tests

From the repository root, activate an existing virtual environment. The following
Windows example builds the normal D3D12 shared provider, core, and tests:

```powershell
python tools\ci_build\build.py --update --build --build_dir b `
  --config RelWithDebInfo --use_webgpu shared_lib --build_shared_lib `
  --skip_tests --parallel 6 `
  --cmake_extra_defines onnxruntime_ENABLE_DAWN_BACKEND_D3D12=ON `
                        onnxruntime_ENABLE_DAWN_BACKEND_VULKAN=OFF

python tools\python\webgpu_small_m_int4\build.py `
  --build-dir b\RelWithDebInfo --config RelWithDebInfo

Push-Location b\RelWithDebInfo\RelWithDebInfo
.\onnxruntime_provider_test.exe `
  '--gtest_filter=*MatMulNBitsSmallMSimt*'
Pop-Location

python -m pytest -q tools\python\webgpu_small_m_int4\test_reference.py
```

The wrapper only invokes `cmake --build` for
`onnxruntime_webgpu_matmul_nbits_benchmark` in an already configured tree.
That target is excluded from the default build and is not a CI performance test.
It links the normal ORT library and, for shared-plugin builds, depends on and
registers the normal WebGPU plugin. Run from the binary output directory so the
newly built runtime and plugin are found together. The plugin runner rejects
ambiguous or non-GPU device enumeration and prints the selected vendor/device
IDs; verify the expected hardware in those IDs and Dawn's adapter log.
The native backend is D3D12 on Windows and Vulkan elsewhere.

The standard provider tests cover all M=1–8, both strides, K/N tails, block
layouts, higher-rank inputs, optional zeros/bias, and explicit failures. They
check independent packed-byte FP64 reference outputs and actual program-dispatch
logs, with CPU fallback disabled and node-provider profiling checks. A missing
EP is a skip, not a CPU run.
The shader-cache test also checks that subgroup opt-out changes cache identity.

## Separate correctness and benchmark tools

`experiment.py` retains the 77 synthetic fixtures and independent exact/FP16-weight
FP64 references. Supply an explicit normal benchmark executable; no historical
build directory or installed ORT package is used:

```powershell
Push-Location C:\ort\b\RelWithDebInfo\RelWithDebInfo
python C:\ort\tools\python\webgpu_small_m_int4\experiment.py `
  --executable C:\ort\b\RelWithDebInfo\RelWithDebInfo\onnxruntime_webgpu_matmul_nbits_benchmark.exe `
  --bn 8 --a-block-padding 1 correctness `
  --output C:\ort\b\simt-correctness-new
Pop-Location
```

Run the Python command with the native binary directory as the working directory
when using a shared plugin. Resolve the script and output paths accordingly.
For real Qwen benchmarks, use the `benchmark` subcommand and supply
`--weights-model` and `--captures`; those model/capture inputs are not part of the
repository. `--matrices 0` uses all available matching projections. These runs can
generate tens of GiB of private external-data weights; never commit them.
M=8 repeats captured M=4 activations, not measured target M=8 activations.

GPU-resident inputs/outputs and weight initialization precede measured execution.
Readbacks fence calls outside shader timestamp intervals. Every GPU program and
node provider is checked; automatic prepack dispatches are included. JSON retains
commands, per-rotation GPU timings, profile paths, and numerical comparisons.
Clock monitoring is explicitly unavailable unless `--monitor-nvidia` enables
the NVIDIA-only monitor. `--icd` optionally overrides Vulkan's ICD; neither
option is required for Intel/D3D12. Unmonitored timings do not establish stable
clocks or a portable speedup.

The current EP API adapter does not forward its native GPU profiler into the
ORT session profile. Shared-plugin correctness runs therefore verify actual
dispatch logs and report `gpu_timestamps_available: false`. Performance commands
fail if native GPU timestamp events are missing; CPU/node durations are never
substituted. For GPU timing, configure the normal in-tree provider with
`--use_webgpu static_lib` instead of `shared_lib` and rebuild the runner.

On Linux/glibc Vulkan builds, explicitly build
`onnxruntime_webgpu_pipeline_stats` and pass its full path using
`resources --stats-library`. It is never loaded for clean timing. Instrumented
compiler-statistics measurements must not be used as performance results.

`validate_metal.py --tint <compiler> --dump <shader-log> --output <new-dir>`
accepts both current and historical shader names and records compiler/pin
provenance. MSL generation is neither Metal SDK compilation nor Apple GPU
execution. A100 or Intel performance must not be extrapolated to Apple hardware.

[RESULTS.md](RESULTS.md) preserves the original standalone A100 investigation,
including historical paths and program names. Those measurements are not new
evidence about the reorganized build, other devices, or automatic selection.
