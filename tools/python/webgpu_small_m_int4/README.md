# Diagnostic small-M INT4 SIMT prototype

This is an explicit, standalone experiment, **not a production kernel or selector**.
`build.py` copies and recompiles the current `MatMulNBits` translation unit and
links it against an existing static ORT WebGPU test build. `auto` retains the
current checkout's selection; `regular` forces the existing regular shader;
`simt` explicitly selects the new program. Production sources, objects, and
libraries are not overwritten.

An isolated copy of `ShaderHelper` omits unused subgroup feature declarations and
subgroup builtin parameters **only for `DiagnosticSmallMSimt`**. The actual
executed shader requires `f16`, not subgroups, subgroup-size control, or subgroup
matrices. Existing programs retain their original generated code.

## Architecture and supported inputs

The default is 128 threads, BN16, BK512, and eight logical K lanes per column.
BM equals M for M=1/2/4/8; odd M uses a guarded BM4/BM8 tile. Each thread has
BM distinct FP32 scalar accumulators. It loads a packed 32-weight block and
scale, dequantizes to FP16, promotes to FP32, and reuses those weights across
all BM rows. A is cooperatively staged as `vec4<f16>`; K tails are zeroed.
The last workgroup barrier precedes a shared-memory K reduction and a single
FP16 output conversion. There is no prepack dispatch or global scratch.

The initial, unpadded control uses 3/6/12 KiB at M=2/4/8: `BM*BK*2` bytes of A
and `BM*128*4` bytes of partials. `--a-block-padding 1` inserts one unused
`vec4<f16>` after each staged block32, changing the logical K-lane address
stride from 64 to 72 bytes. This source-motivated bank-conflict ablation
preserves thread ownership, arithmetic, B loads, and barriers. Its storage
is 3.25/6.5/13 KiB. The unpadded control remains the default; select padding
explicitly when reproducing the improved variant.

Flat local indices define ownership, so
correctness does not depend on the hardware subgroup width. Padded dispatch
workgroups, N tails, and partial-M tiles are guarded.

Supported: FP16 A/scales/bias/output, unsigned packed INT4 with implicit zero 8
or byte-padded explicit uint8 zero points, block32/64/128 and other block sizes
divisible by 32, the existing single-scale convention, non-multiple K/N,
higher-rank A with shared 2D weights. Constant weight-matrix indexing is
implemented but has not been separately exercised with a nonzero index.
Per-row indirect weights and activation-row replay are explicitly rejected.
INT2/INT8 and FP32 inputs are not prototype configurations. The diagnostic
always accumulates in FP32 regardless of `enableMatmulFp32Accumulation`.

BN8/16/32 and BK256/512/1024 are diagnostic controls, not automatic tuning or
production selection. The device workgroup-storage limit is checked.

## Reproduction

Run from the repository root with the existing `.venv` and A100/Vulkan build.
Each experiment output directory must be new. `--build-dir` and `--executable`
can select another compatible static WebGPU build.

```bash
source .venv/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

python tools/python/webgpu_small_m_int4/build.py \
  --output build/webgpu-small-m-simt-pad
python -m pytest -q tools/python/webgpu_small_m_int4/test_reference.py

python tools/python/webgpu_small_m_int4/experiment.py \
  --executable build/webgpu-small-m-simt-pad/diagnostic \
  --bn 8 --a-block-padding 1 correctness \
  --output build/webgpu-small-m-simt/correctness-new

python tools/python/webgpu_small_m_int4/experiment.py \
  --executable build/webgpu-small-m-simt-pad/diagnostic \
  --bn 8 --a-block-padding 1 benchmark \
  --modes regular regular-f32 simt auto --warmup 32 --repeat 15 --reference \
  --output build/webgpu-small-m-simt/full-new

c++ -std=c++17 -O2 -shared -fPIC \
  -I build/webgpu-pipeline-lifetime/Release/_deps/dawn-src/third_party/vulkan-headers/src/include \
  tools/python/webgpu_small_m_int4/vulkan_pipeline_stats.cc \
  -o build/webgpu-small-m-simt-pad/pipeline_stats.so -ldl -pthread
python tools/python/webgpu_small_m_int4/experiment.py \
  --executable build/webgpu-small-m-simt-pad/diagnostic \
  --bn 8 --a-block-padding 1 resources \
  --stats-library build/webgpu-small-m-simt-pad/pipeline_stats.so \
  --output build/webgpu-small-m-simt/resources-new
```

`benchmark` defaults to the existing Qwen export under
`/workspace/models/qwen3.8-27b-int4-webgpu-paged/model.onnx` and MLP captures
under `build/mlp-up-epilogue-20261021/current-wall`.
Override these with `--weights-model` and `--captures`.
All 272 available matching projections are used unless `--matrices` limits
each group. Private external-data files contain the selected real weights;
source model files are never edited. Allow roughly 32 GiB of generated weight
data for three M values. Do not add generated weights or profiles to Git.

M=2/4 MLP inputs come from the captured target. M=8 repeats the M=4 capture;
it is not claimed to be an actual M=8 target activation. Attention inputs
are deterministic synthetic FP16. Weights, activations, and outputs are
GPU-resident during repeated execution. Real readbacks fence completed calls
outside shader timestamp intervals. CPU fallback is disabled. Every timed
program is checked, and automatic-path prepack dispatches are included.

JSON results retain commands, per-rotation samples, paths, clocks, and
numerical comparisons. FP64 references independently decode bytes, multiply
scales, and accumulate; a second reference first rounds weights to FP16.
Cancellation fixtures deliberately expose nonfinite FP16 regular results.
Instrumented `resources` timings must not be used as clean performance data.
The optional compiler-statistics interposer requires Linux/glibc, Vulkan headers,
and `VK_KHR_pipeline_executable_properties`. Its source is preserved as
`vulkan_pipeline_stats.cc`; the historical default library remains
`build/int4-native-investigation/pipeline_stats.so`. It is never loaded for clean
timing or full-model headline runs.

## Metal validation

Use a Linux Tint built from the Dawn dependency in `cmake/deps.txt`.
The existing verified build for this investigation is:

```bash
python tools/python/webgpu_small_m_int4/validate_metal.py \
  --tint /root/.copilot/session-state/f35f1f98-881e-41bd-b257-9c4c9ca8c6fa/files/apple-tint-build/tint \
  --dump build/webgpu-small-m-simt/correctness-new/simt.wgsl \
  --output build/webgpu-small-m-simt/metal-new
```

The tool checks the actual dumped shader, re-resolves WGSL, emits MSL, records
the compiler hash and current dependency pin, and checks the storage footprint.
**MSL generation is neither Metal SDK compilation nor Apple hardware execution.**
No A100 speedup should be extrapolated to M4/M5.
The diagnostic runner and clock monitor are Linux/Vulkan/NVIDIA-specific;
portable WGSL does not make this executable an Apple benchmark harness.

Detailed measurements, go/no-go reasoning, and target-verification estimates
are in [RESULTS.md](RESULTS.md). Raw generated artifacts are under the ignored
`build/webgpu-small-m-simt/` directory.
