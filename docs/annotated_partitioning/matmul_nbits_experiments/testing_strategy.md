# MatMulNBits estimation testing strategy

## Purpose and rules for future agents

Qualify workload counts without changing dispatch, numerical outputs or placement policy.
Keep three claims separate:

1. **Logical correctness:** exact `2MNK` dense-equivalent operations and optional `MN` bias additions.
2. **Capture correctness:** actual provider, route, compute mode, shapes and invocation multiplicity.
3. **Executed-work accuracy:** independently justified FP work compared against matching modeled components.

An estimator report or a passing formula test does not establish claim 3. Timing, utilization and weight bit width
do not establish FP counts. Integer dot products are not floating-point operations. Do not fit a universal multiplier
to whole-process counts. Unsupported work remains unavailable, never zero.

Use the [completion plan](../matmul_nbits_counter_completion_plan.md) and
[conventions](../contrib_op_flop_estimation_rollout.md) for the current model contract. Before changing tooling or
models, read repository/path instructions and the applicable build/test skills. Keep raw counters, models and wheels
outside Git; publish small reports and CSV summaries. Do not overwrite historical evidence.

## Evidence already collected

The [November 9 CPU report](cpu_epyc_7v12_2026_11_09/report.md) records 84 configurations and three R/2R trial pairs
per configuration (504 measurements) on an AMD EPYC 7V12 VM. The requests resolved to 60 `sqnbit_fp32` cases and 24
`sqnbit_int8` cases, all `packed_qnbit`. These are implementation families, not verified ISA microkernel symbols.
Derived counts repeated exactly across trials. This establishes repeatability, not counter accuracy.

FP32 measured process counts were about 22-25% below estimates for M=1 and much closer at M=128. INT8 process FP
counts exceeded the modeled subtotal by about 145-292%. These are signed differences between different scopes,
not isolated operator errors. Activation quantization, vector scaling, reductions, reconstruction reuse and
kernel specialization require independent attribution. Do not carry the initial single-case 0.39% result forward
as a universal accuracy claim.

## Reproducible CPU sequence

See [tool commands](../../../tools/python/PythonTools.md#repeatable-matmulnbits-matrix-and-hardware-profiling).
The checked-in tools generate the matrix, validate reporting and collect/report raw counters. Initial coverage:

- FP32 inputs with accuracy requests 0-4; FP16 inputs with requests 0 and 4.
- M=1/8/32/128 and (K,N)=(64,32)/(128,64)/(1024,1024).
- INT4 weights, block size 32, no bias, implicit midpoint zero points, one thread.

Accuracy requests permit choices; they do not force FP16/BF16/INT8 kernels. Group coverage by observed route/mode.
Repeat with bias/explicit zero points, block sizes 64/128, valid tail dimensions, vector/batched inputs and 2/8-bit
weights after the primary discrepancies are understood. Reserve new shapes as held-out validation before fitting.

CPU hardware gates:

- Check perf event availability with a known-arithmetic control; scalar access alone does not qualify SIMD/FMA.
- Require nonzero supported counters, full running time, no missing/duplicate measurements and complete R/2R pairs.
- Record event definitions, hardware/VM, perf version, wheel/native binary hashes and exact driver commands.
- Whole-process subtraction includes recurring runtime work and does not guarantee cancellation of independent
  initialization/warmup work. Phase-isolated benchmarks/instrumentation are needed for defensible component totals.
- Use three pairs for repeatability; publish mean/min/max/sample standard deviation. Zero standard deviation means
  identical derived counts. Do not use profiled process elapsed time as operator latency.

## Immediate audits

1. FP32 M=1 and M=128, K=N=1024: identify selected symbols and reconstruction/scaling arithmetic.
2. INT8 M=1 and M=32, K=128, N=64: isolate activation quantization, vector scaled accumulation, horizontal reductions
   and corrections. Confirm ISA specialization before assigning lane-count coefficients.
3. Add independently derived component references, fix demonstrated gaps and validate held-out shapes.
4. Preserve logical baselines and explicit whole-op exclusions while expanding optional route models.

## CUDA preparation

Stay on the experiment branch, or bring the tools onto another branch only if it includes CPU/CUDA route-aware
estimator commits. An older `matmul_flops` wheel may export logical counts without `expected_flops`. No branch switch
is needed for this workflow. Use a separate CUDA build directory and preferably a separate environment.

On the current machine, eight A100 SM80 GPUs were visible at preparation time, but the inspected `ort-flops`
environment was CPU-only and `nsys`/`ncu` were not on PATH. Do not claim CUDA validation from CPU results.
Coordinate GPU use on shared hosts; choose an approved free device. Never stop other users' work.

Build from the current source checkout (supply real installed CUDA/cuDNN roots):

```bash
conda create -n ort-flops-cuda python=3.12  # only if the environment does not exist
conda activate ort-flops-cuda
cd /absolute/path/to/onnxruntime
export REPO="$PWD"
python -m pip install -r requirements.txt -r tools/ci_build/requirements/pybind/requirements.txt
python -m pip install onnx nvtx
export CUDA_HOME=/usr/local/cuda
export CUDNN_HOME=/absolute/path/to/cudnn
test -d "$CUDA_HOME" && test -d "$CUDNN_HOME"

python tools/ci_build/build.py \
  --build_dir build/estimator-cuda --config Release --update --build \
  --parallel 4 --build_wheel --skip_tests --use_cuda \
  --cuda_home "$CUDA_HOME" --cudnn_home "$CUDNN_HOME" \
  > /tmp/ort-estimator-cuda-build.log 2>&1
```

Only after a successful build, install its exact wheel path:

```bash
ls "$REPO/build/estimator-cuda/Release/dist/"
# Replace the filename with the exact wheel just built:
python -m pip install --force-reinstall --no-deps \
  "$REPO/build/estimator-cuda/Release/dist/REPLACE_WITH_BUILT_WHEEL.whl"
```

The host also needs a supported C++ compiler, CMake >=3.28 and matching CUDA/cuDNN development dependencies.
Avoid coexistence of CPU and GPU ORT distributions in one environment; both install the `onnxruntime` module.
Keep normal kernel coverage enabled, not reduced/quick builds. For profiling install NVIDIA Nsight Systems and
Nsight Compute through the host's supported setup and put `nsys`/`ncu` on PATH. The collector's `--nvtx` requires
the Python `nvtx` package. Verify prerequisites before running:

```bash
cd /tmp
python -c 'import onnxruntime as o; print(o.__file__); print(o.get_build_info()); print(o.get_available_providers())'
nvidia-smi
nsys --version
ncu --version
python -c 'import nvtx'
```

Install missing Python/build dependencies in the activated isolated environment as needed. A listed CUDA provider
is insufficient: actual session creation and captured CUDA invocation must succeed.

## CUDA capture and profiling commands

The [CUDA collector](../../../tools/python/profile_matmul_nbits_cuda.py) takes one generated case.
It uses GPU-resident I/O, disables CPU EP/node fallback and Python run fallback, disables graph optimizations and
CUDA Graph replay, and explicitly synchronizes outputs. Capture is a separate process from hardware profiling.
CUDA report records still represent successful host submission, not completion certification by the collector.
The driver checks completion/output transfer independently. No production collector behavior is changed.

```bash
export REPO=/absolute/path/to/onnxruntime
export EXP=$(mktemp -d /tmp/matmul-cuda-XXXXXX)
# Use an approved physical GPU UUID from nvidia-smi; device 0 below is the remapped ordinal.
export CUDA_VISIBLE_DEVICES=GPU-REPLACE-WITH-APPROVED-UUID
cd /tmp
python "$REPO/tools/python/generate_matmul_nbits_cases.py" --output "$EXP/cases"
export CASE_ID=$(jq -er '
  [.cases[] | select(.settings.a_shape == [128,64]
    and .settings.dtype == "float32"
    and .settings.bias == false and .settings.zero_points == false)][0].case_id
' "$EXP/cases/manifest.json")

python "$REPO/tools/python/profile_matmul_nbits_cuda.py" \
  --manifest "$EXP/cases/manifest.json" --case-id "$CASE_ID" \
  --mode capture --output "$EXP/capture" --device 0

jq '.invocations[] | {provider,route,compute_mode,reconstruction_k,expected_flops}' \
  "$EXP"/capture/estimate_*.json
```

Require one CUDA invocation, no collection drops, matching logical counts and reporting-on/off outputs, and an
available route-aware subtotal. Failures are retained as incomplete metadata/raw reports, not passed cases.
The CPU perf runner must not be used for GPU FP counts.

Timeline pass (includes initialization/warmup outside the named measured NVTX range):

```bash
nsys profile --trace=cuda,nvtx --output="$EXP/timeline" \
  python "$REPO/tools/python/profile_matmul_nbits_cuda.py" \
  --manifest "$EXP/cases/manifest.json" --case-id "$CASE_ID" \
  --mode profile --nvtx --warmup 10 --repeats 10 --device 0 --output "$EXP/timeline-run"
```

Inspect the `matmul_nbits_measure` range in Nsight Systems. Identify every child compute kernel and stream, including
separate dequantization, GEMM and bias when present. Input upload/output download and preparation are outside the
named range; host execution/synchronization remain visible. Distinguish tactic-search trial launches from actual
steady-state work. If those still occur inside the range, increase warmup/investigate before accepting the scope.

Nsight Compute pass (one measured invocation; all kernels inside the NVTX push/pop range):

```bash
ncu --query-metrics > "$EXP/available-ncu-metrics.txt"
ncu --nvtx --nvtx-include 'matmul_nbits_measure/' --set full --export "$EXP/kernel-counters" \
  python "$REPO/tools/python/profile_matmul_nbits_cuda.py" \
  --manifest "$EXP/cases/manifest.json" --case-id "$CASE_ID" \
  --mode profile --nvtx --warmup 10 --repeats 1 --device 0 --output "$EXP/counters-run"
```

Check installed Nsight help if range syntax/options differ. A permission error for GPU counters requires an
administrator-approved solution; do not modify security settings automatically. Full metric sets can be expensive
and replay kernels; they are evidence collection, not normal latency measurements. Do not add a launch-count limit
that silently captures only the first child kernel. Require a nonempty profile containing all intended kernels.

## CUDA qualification matrix and interpretation

Start with FP32/FP16, M=1/8/32/128, aligned and valid irregular K/N, bias/zero-point variants, then supported bits
2/4/8 and block sizes. CUDA paths are not selected by CPU accuracy-level semantics; group by actual routes:
`cuda_fused`, `cuda_fpa_intb_gemv`, `cuda_fpa_intb_gemm`, `cuda_mixed_fpa_intb`, `cuda_dequantize_gemm`.
Do not assume every family is reachable with standard-layout generated models. Mixed-weight routes may need
EP-specific packing/tactics; missing routes are coverage gaps. Keep prepacked-layout cohorts separate.

The currently modeled CUDA core families perform floating-point arithmetic after quantized-weight decoding;
INT4 weight storage does not make their dot products integer. There is no universal NVIDIA FLOP counter:

- Establish architecture-specific instruction/metric meanings, precision, tensor tile operations, FMA convention,
  active/predicated lanes and replay semantics before mapping counters to FLOPs.
- Do not add overlapping scalar/tensor metrics or multiply an already FLOP-valued FMA count again.
- Sum matched constituent kernels, not just GEMM; attribute reconstruction/reductions without overlap.
- Separate logical core/bias, modeled core/dequantization, measured instruction-derived FP work and latency.
- Device duration must cover every participating stream/kernel, not host launch time or only the fastest child.
- Record metric names/definitions, GPU UUID/architecture, driver/toolkit/library versions, packing layout and
  observed routes/tactics. Current collector metadata lacks exact tactic/ISA details; use profiler/source evidence.

After capture gates pass, collect three independent profiling trials for representative routes and retain native
Nsight reports and command logs externally. Validate count interpretation before an automated GPU accuracy table.
The CUDA collector deliberately does not convert arbitrary Nsight metrics into a misleading universal FP count.

## Reporting and acceptance

Publish case settings, requested versus observed mode, hardware/build provenance, component estimates, matched
references, variability and explicit statuses/exclusions. Exact logical agreement and complete capture are mandatory.
Executed-work differences need defensible matching scopes; otherwise label accuracy unvalidated.
Whole-op FP estimates require all recurring FP components or justified absence, not merely an available subtotal.
Do not tune on held-out cases without replacing that cohort. GPU architectures/builds not exercised remain unqualified.
