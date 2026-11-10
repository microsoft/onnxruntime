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

The [November 10 follow-up](cpu_epyc_7v12_2026_11_09/report.md#follow-up-audit-fp32-single-row-avx2-kernel-november-10)
identified the AVX2 single-row specialization for FP32 M=1, K=N=1024. Its source-derived `3NK + 7N` count is
3,152,896, versus measured recurring process work of 3,152,898 and the existing estimate of 4,194,304.
Integer midpoint subtraction explains why this case needs one FP scaling operation per weight, rather than the
nominal two-FP-operation reconstruction scenario. This result is bounded; compiled branch accounting, the residual,
other specializations and held-out validation remain open. No production formula has been changed.

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

The [aligned INT8 follow-up](cpu_epyc_7v12_2026_11_09/report.md#follow-up-audit-aligned-int8-avx2-block-sum-path-november-10)
now records the AVX2 quantizer/block-sum kernel and FMA3 correction GEMM, with candidate total
`MK + 3MG + 14MNG + 9MN`. Define B=`block_size` and G=ceil(K/B); audited B=32/full-block cases simplify to G=K/32.
The original case and three new aligned/even-row/even-block held-out cases
agree with recurring process counts to an approximately +2-operation residual. This corroborates a bounded
total, not independently measured phase counts. Keep estimator formulas unchanged pending a deliberate refinement.
Next audit M=1 INT8, which follows a different branch; quantizer/event semantics and tails remain unqualified.

### General block geometry versus arithmetic coefficients

Always express block count as `G = K // B + (1 if K % B else 0)` with B=`block_size`, and padded width as `Kp=B*G`.
This definition is general; the audited coefficients are not. See the
[report's applicability envelope](cpu_epyc_7v12_2026_11_09/report.md#general-block-notation-and-bounded-formula-applicability).
For block-32 INT4/FP32-input effective INT8 non-VNNI AVX2 branches, the candidate totals are:

- Multi-row: `MK + 3MG + 14MNG + 9MN`, for positive even M>=2.
- Single-row: `K + 3G + 12NG + 7N`, for M=1; the separate correction GEMM is bypassed.

Both assume full blocks, even G, N divisible by four, no bias/explicit zero points and the audited quantizer path.
Substituting G for B=64/128, odd block counts, tails or all-zero activation blocks does not qualify those cases.
Keep candidate references unvalidated/unavailable outside the envelope; preserve independently available logical
counts. Do not change production estimators under this documentation generalization.

1. FP32 M=1 and M=128, K=N=1024: identify selected symbols and reconstruction/scaling arithmetic.
2. INT8 M=1 and M=32, K=128, N=64: isolate activation quantization, vector scaled accumulation, horizontal reductions
   and corrections. Confirm ISA specialization before assigning lane-count coefficients.
3. Add independently derived component references, fix demonstrated gaps and validate held-out shapes.
4. Preserve logical baselines and explicit whole-op exclusions while expanding optional route models.

## Repeatable FP32 M=1 kernel audit

This is the sequence used to select and trace the November 10 case. It requires the saved baseline bundle,
the same installed native runtime, Linux perf sampling access, `jq`, `rg`, and disassembly support. Sampling
permissions may differ from counting permissions. Do not change host security settings automatically.
Keep the baseline and installed wheel untouched; debug symbols must match that binary. Source line numbers may
move, so search symbols rather than relying solely on fixed line ranges.

### Select and preserve the case

Run the following blocks in the same Bash terminal. Replace the repository/baseline roots if moved.

```bash
conda activate ort-flops
cd /tmp
export REPO=/datadisks/disk10/tommasoadani/repos/onnxruntime
export BASE=/datadisks/disk10/tommasoadani/tmp/matmul-baseline-2026-11-09
export AUDIT=$(mktemp -d /datadisks/disk10/tommasoadani/tmp/matmul-m1-audit-XXXXXX)
echo "Audit directory: $AUDIT"

# Use a standalone assignment so a failed jq selection is visible.
CASE_ID=$(jq -er '
  [.cases[] | select(
    .settings.a_shape == [1,1024] and .settings.n == 1024 and
    .settings.dtype == "float32" and .settings.accuracy_level == 0 and
    .settings.bias == false and .settings.zero_points == false and
    .compute_mode == "sqnbit_fp32"
  )] |
  if length == 1 then .[0].case_id else error("Expected exactly one case") end
' "$BASE/estimates-v2/results.json")
```

Stop if selection fails; do not continue with an empty case ID.

```bash
export CASE_ID
export CASE_DIR="$BASE/cases/$CASE_ID"
test -f "$CASE_DIR/model.onnx" && test -f "$CASE_DIR/inputs.npz"
jq --arg id "$CASE_ID" '.cases[] | select(.case_id == $id)' \
  "$BASE/estimates-v2/results.json" > "$AUDIT/case-estimates.json"
cat "$AUDIT/case-estimates.json"
```

### Verify binary provenance

This checks native library hashes, not just version/build strings. Stop on any mismatch.
The saved baseline must contain `baseline.json` with the native library hashes recorded at save time.

```bash
python - <<'PY'
import hashlib
import json
import os
from pathlib import Path
import onnxruntime as ort

baseline = json.loads((Path(os.environ["BASE"]) / "baseline.json").read_text())
libraries = baseline["provenance_captured_at_save"][
    "installed_native_libraries_match_saved_wheel_at_save_time"
]
if not libraries:
    raise RuntimeError("Baseline has no native library hashes")
root = Path(ort.__file__).parent.parent
for library in libraries:
    path = root / library["path"]
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != library["sha256"]:
        raise RuntimeError(f"Native library differs from baseline: {path}")
info = {
    "runtime_file": ort.__file__,
    "build": ort.get_build_info(),
    "native_libraries_match_baseline": True,
}
(Path(os.environ["AUDIT"]) / "runtime.json").write_text(json.dumps(info, indent=2))
print(json.dumps(info, indent=2))
PY
```

### Sample and identify symbols

```bash
perf --version > "$AUDIT/perf-version.txt"
lscpu > "$AUDIT/cpu-info.txt"
perf record --output="$AUDIT/cycles.data" --event=cycles:u --freq=499 \
  --call-graph=dwarf,8192 -- \
  python "$REPO/tools/python/profile_matmul_nbits.py" workload \
    --model "$CASE_DIR/model.onnx" --inputs "$CASE_DIR/inputs.npz" \
    --repeats 10000 --warmup 100 --threads 1
```

Require a successful command and nonzero samples before proceeding. The workload disables reporting, uses CPU,
sequential execution, disabled graph optimizations and one intra-op thread. Sampling includes startup/warmup;
cycle percentages identify hot code, not FLOP counts or isolated inference duration.

```bash
perf --no-pager report --input="$AUDIT/cycles.data" --stdio --no-children \
  --sort=dso,symbol --percent-limit=0.5 > "$AUDIT/hot-symbols.txt"
perf --no-pager report --input="$AUDIT/cycles.data" --stdio --children \
  --percent-limit=1 > "$AUDIT/call-tree.txt"
grep -Ei 'mlas|sq4|sqnbit|qnbit|gemm|avx' "$AUDIT/hot-symbols.txt"
```

The historical hot symbol was `SQ4BitGemmM1Kernel_BlkLen32Plus_CompFp32_avx2<false>` with compiler suffix
`[clone .isra.0]`. Confirm the symbol actually observed on each run; do not infer AVX2 from the CPU alone.
If samples show unknown addresses, preserve the capture and obtain matching symbols before assigning a kernel.

### Trace dispatch and annotate the sampled binary

```bash
rg -n -C 12 'RangeCountM == 1|SQ4BitGemmM1Kernel_CompFp32' \
  "$REPO/onnxruntime/core/mlas/lib/qnbitgemm.cpp"
rg -n 'SQ4BitGemmM1Kernel_CompFp32' "$REPO/onnxruntime/core/mlas/lib" \
  > "$AUDIT/m1-dispatch-references.txt"
cat "$AUDIT/m1-dispatch-references.txt"

rg -n -A 175 '^ComputeDotProducts_BlkLen32Plus_CompFp32_avx2\(' \
  "$REPO/onnxruntime/core/mlas/lib/sqnbitgemm_kernel_avx2.cpp" \
  > "$AUDIT/source-dot-products.txt"
rg -n -A 36 '^FoldAccumulators\(const __m256&' \
  "$REPO/onnxruntime/core/mlas/lib/sqnbitgemm_kernel_avx_common.h" \
  > "$AUDIT/source-reduction.txt"
cat "$AUDIT/source-dot-products.txt" "$AUDIT/source-reduction.txt"

read -r -p "Paste the exact demangled hot symbol from the report: " KERNEL_SYMBOL
perf --no-pager annotate --input="$AUDIT/cycles.data" --stdio \
  --symbol="$KERNEL_SYMBOL" > "$AUDIT/kernel-assembly.txt"
head -160 "$AUDIT/kernel-assembly.txt"
```

Paste only the symbol, not its percentage/DSO columns. Require nonempty disassembly. The first 160 lines are only a
preview: this function contains block-size and tail branches. Follow the actual block-32/no-explicit-zero-point,
four-column branch before counting instructions. Debug source lines are optional; matching disassembly is required
for a compiled instruction claim.

Check integer midpoint subtraction, FP conversion/scaling, FP multiply-adds and final reduction. In AVX2 assembly,
the integer adjustment may be an integer add of -8 rather than a subtraction mnemonic. Do not count conversions,
shuffles or integer instructions as FP arithmetic; count eight-lane FP multiply-add as 16 FLOPs. Sample percentages
are not execution multiplicities. Derive loop counts from the selected geometry and control flow.

For M=1, aligned K, N divisible by four, block size 32 and no bias, the observed source specialization has
`NK` FP scaling, `2NK` accumulation and `7N` reduction. Cross-check independently:

```bash
python - <<'PY'
import json
import os
from pathlib import Path

case = json.loads((Path(os.environ["AUDIT"]) / "case-estimates.json").read_text())
k = case["settings"]["a_shape"][-1]
n = case["settings"]["n"]
print("FP weight scaling:", n * k)
print("FP dot accumulation:", 2 * n * k)
print("FP final reductions:", 7 * n)
print("Source-derived bounded kernel total:", 3 * n * k + 7 * n)
print("Existing modeled subtotal:", case["expected_flops"]["expected_flops"])
PY
```

For the audited case this gives 3,152,896 versus measured recurring process work of 3,152,898. Leave the residual
unattributed. Save the symbol, source/assembly excerpts, runtime hashes and assumptions with the audit.
Repeat for the other aligned M=1 geometries and new held-out shapes before refining a route model. The helper
count is not a general formula for all CPU packed-QNBit implementations.

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
