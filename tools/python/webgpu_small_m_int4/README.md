# Diagnostic small-M INT4 SIMT prototype

This is an experimental kernel with **production defaults unchanged**. The
standalone runner supports explicit `auto`, `regular`, and `simt` selection.
On `aciddelgado/benchmark_qwen`, the same implementation is also available
through the disabled-by-default full-model opt-in described below.

The shared shader generator is
`onnxruntime/contrib_ops/webgpu/quantization/matmul_nbits_small_m_simt.h`.
`diagnostic.h` supplies only standalone tuning controls. `build.py` copies and
recompiles the current `MatMulNBits` translation unit and links it against an
existing static ORT WebGPU test build, without overwriting production sources,
objects, or libraries. Explicit standalone modes take precedence over the
full-model routing opt-in.

`ShaderHelper` omits unused subgroup feature declarations and
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

## Opt-in full-model routing

`ORT_WEBGPU_EXPERIMENTAL_SMALL_M_SIMT=1` selects the fixed **BN8/BK512,
A-block-padding=1** kernel only when all these conditions hold:

- M is 2, 4, or 8, and `(K,N)` is `(5120,17408)` or `(17408,5120)`.
- Activations, scales, and output are FP16, with shader-f16 available.
- Weights are INT4 block32, accuracy level 4, with one weight batch.
- There is no bias, explicit zero point, weight-index selection, indirect
  weight selection, or activation-row replay.

Unset or `0` retains the existing selector. Other values are errors. N=1024,
N=10240, odd M, and prefill remain on their existing paths. The standalone
BN/BK/padding environment controls do not alter integrated routing.

This is a shape/type/attribute allowlist, not node-name matching. In the genuine
exports used here it routes all 192 target MLP projections per eligible
verification forward. The five-layer drafter has the same MLP dimensions but
omits accuracy level 4, so **its MLPs are not routed**. GPU convolution/GDN
state replay and the existing INT4 selector are retained.

`full_model.py` provides paired frozen-state target replay and genuine
baseline/width-1/width-3 Engine generation. The harness toggles routing within
the same loaded sessions and uses identical binaries, prompt IDs, weights,
and settings. It rejects speculative fallback and incorrect output lengths.
It requires the existing genuine export manifests, workloads, baseline token
sequence, and frozen state snapshots; it does not download or regenerate them.

First build the current plugin in its existing configured tree, then relink
an isolated timestamp-capable copy. `build_full_model.py` instruments a copied
provider translation unit; it does not edit the production execution provider.
Use this identical plugin for every off/on condition. Native GPU profiling
is inactive unless the harness enables it for a diagnostic call.

```bash
source .venv/bin/activate
RUNTIME=/workspace/onnxruntime-genai/build/qwen38-narrow-drafters-20261020
cmake --build "$RUNTIME/ort-plugin/Release" \
  --target onnxruntime_providers_webgpu --parallel 32
python tools/python/webgpu_small_m_int4/build_full_model.py \
  --build-dir "$RUNTIME/ort-plugin/Release" \
  --output build/qwen-small-m-simt-profile-new

export VK_ICD_FILENAMES=/tmp/nvidia_icd.json XDG_RUNTIME_DIR=/tmp
export LD_LIBRARY_PATH="$RUNTIME/ort-home/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
export ORT_LIB_PATH="$RUNTIME/ort-home/lib/libonnxruntime.so"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
PLUGIN=build/qwen-small-m-simt-profile-new/libonnxruntime_providers_webgpu.so
PYTHON=/workspace/venv/bin/python

$PYTHON tools/python/webgpu_small_m_int4/full_model.py --plugin "$PLUGIN" \
  controlled --kind gpu --warmup 8 --repetitions 5 \
  --output build/qwen-small-m-simt-gpu-new
$PYTHON tools/python/webgpu_small_m_int4/full_model.py --plugin "$PLUGIN" \
  controlled --kind wall --warmup 8 --repetitions 10 \
  --output build/qwen-small-m-simt-wall-new
$PYTHON tools/python/webgpu_small_m_int4/full_model.py --plugin "$PLUGIN" \
  controlled --kind assignment --warmup 1 --repetitions 1 \
  --output build/qwen-small-m-simt-assignment-new
$PYTHON tools/python/webgpu_small_m_int4/full_model.py --plugin "$PLUGIN" \
  generation --kind routes --repetitions 1 \
  --output build/qwen-small-m-simt-routes-new
$PYTHON tools/python/webgpu_small_m_int4/full_model.py --plugin "$PLUGIN" \
  generation --kind timing --repetitions 5 \
  --output build/qwen-small-m-simt-generation-new
```

Run GPU campaigns serially. Output directories must be new. `--runtime-root`
and controlled `--state-dir` override archived input locations. Wall/headline
runs have no native GPU or ORT node profiling. GPU runs measure all dispatches;
assignment runs separately reject heavy CPU fallback. The routes smoke generates
16 tokens, not the headline 128/512 tokens. Headline runs have one full-length
warmup per condition and five measured repetitions, reversing order each
repetition. Minimum generated length stays zero: a positive floor disables
the Engine's draft verification.

Frozen-state replay restores all 128 state tensors before every call and
readbacks logits to fence completion. Auxiliary hidden-state readback and state
reset are outside its completed-host interval. GPU timestamp sums come from
separate instrumented runs and must not be substituted for clean host latency.
Raw results retain full token sequences, acceptance/verification counts,
numerical comparisons, actual routed shader/cache keys, binary hashes, and clocks.
Do not commit raw profiles, model weights, or generated binaries.

## Standalone reproduction

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

Historical isolated measurements, go/no-go reasoning, and target-verification
estimates are in [RESULTS.md](RESULTS.md). Raw generated artifacts are under the ignored
`build/webgpu-small-m-simt/` directory.

The preserved branches, actual full-target routing, controlled verification,
and genuine baseline/width-1/width-3 generation results are in
[FULL_MODEL_RESULTS.md](FULL_MODEL_RESULTS.md), with compact samples and counters
in [FULL_MODEL_RESULTS.json](FULL_MODEL_RESULTS.json).
