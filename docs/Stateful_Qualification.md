# Delivered stateful foundation qualification

**Status: completed and published for review in
[#30](https://github.com/onnxruntime/ort-state/pull/30), stacked on M15/#29.**
The specification and F0-F8 execution plan were separately approved. The results
below qualify only the explicitly scoped delivered foundation.

M24 qualifies the delivered M01-M15 C/C++/Python foundation. M16-M23 stateful
bindings are deferred, not implemented or passed. Existing stateless Session
bindings are unchanged. See [SPEC Appendix P](../SPEC.md#appendix-p-m24-delivered-foundation-qualification-approved)
and [M24 tasks](../TASKS.md#m24---integrated-qualification-measurements-and-final-documentation).

## Scope and capabilities

| Surface | Qualified scope / limitation |
|---|---|
| C/C++ and Python | Native loops/chains/state/cache/greedy/DDIM/package contracts |
| CPU EP | Deterministic fixtures and real Phi-3 INT4 |
| CUDA plugin | Native CUDA cache/greedy and model execution; actual A100 device 0 |
| Legacy CUDA | Model execution/DDIM and declared CPU policies; native plugin cache/greedy requirements are rejected |
| SD1.5 optimized pretrained exports | Actual CUDA only; CPU fixtures do not establish pretrained CPU support |
| Other platforms/GPU architectures | Not qualified by this Linux/A100 run |
| C#, Java/JNI, JS/Node, Web/WASM, Go, Rust, Objective-C, React Native | New stateful bindings deferred to M16-M23 |

Definition, execution and value lifetimes remain distinct. Use explicit exits,
native stepping or run-until-exit, explicit Reset after cancellation/failure,
and independently retained immutable results. GPU `StatefulValue.numpy()`
raises: use `copy_to_cpu()` for intentional readback. A CUDA model does not
itself prove device-native cache/sampler dispatch; select a matching initialized
plugin Session and explicit `REQUIRE` policy.

Ordinary Sessions and existing single-component packages remain supported.
Package schema 1.1 adds declarative chains, wiring, exits and native policies,
not scripts or an executable state machine. See
[Python ownership/policy/package guidance](Stateful_Python.md),
[package authoring](../model_package/README.md),
[Phi-3](Stateful_Phi3_Demo.md) and [SD1.5](Stateful_SD15_Demo.md).

## Reproducible native-only measurements

The [measurement helper](../samples/python/stateful_qualification.py) consumes
an explicitly prepared local package. Run outside the repository import root
against a matching locally built wheel; normal commands do not acquire models.
Phi-3 budget variants must have coherent native-validated metadata/exit limits.
The SD1.5 package retains HEURISTIC convolution provenance and mandatory safety.

```bash
ROOT=/path/to/onnxruntime
source "$ROOT/.venv/bin/activate"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd /path/to/qualification-output

# Select the matching local CPU installation for a CPU Phi-3 package,
# or the matching CUDA-plugin installation for a CUDA package.
export PYTHONPATH=/path/to/local-wheel-install
TOOL="$ROOT/samples/python/stateful_qualification.py"
python "$TOOL" --model phi3 --package /path/to/phi3-budget16 \
  --phase latency --output phi3-cpu-16-latency.json
python "$TOOL" --model phi3 --package /path/to/phi3-budget16 \
  --phase resources --output phi3-cpu-16-resources.json
python "$TOOL" --model phi3 --package /path/to/phi3-budget16 \
  --phase steps --output phi3-cpu-16-steps.json
```

For SD1.5, use `--model sd15` and its batched or sequential CUDA package.
Outputs must not already exist. Image outputs are checked by the existing safety
writer and published exclusively; failed runs do not publish a success report.

Each latency worker records one fresh native setup/first completed run, two
Reset/reuse warmups and five measured completed native runs. All raw samples,
median and nearest-rank p95 are retained; p95 is the maximum of five samples,
not a high-confidence tail estimate. Cold does not mean shared OS caches were
flushed. CUDA completion is explicitly synchronized, not timed as enqueue-only
work. Native timing excludes output readback/rendering/PNG publication, which
is separately reported. Setup includes validation/tokenizer/native construction;
recorded encoding time is a portion of that setup, not an additional disjoint
setup interval.

Throughput uses actual delivered tokens/images divided by total measured native
time. A zero-token row reports throughput as inapplicable, not a fabricated
number. Zero decode budget still permits the separately configured prefill.
The independent `steps` pass reports Python-boundary/completion-synchronized
phase observations and time to first token; it is not a decomposition of a
different Run-path timing.

Resource runs are separate from latency samples. RSS/high-water observations
describe the worker process. Glibc allocation statistics describe glibc-managed
memory, not all mappings or ORT-only malloc call counts. NVML observations cover
the current process, including driver/context reservation; absence of a process
row before GPU initialization is explicit, not silently reported as zero.
GPU visibility is required after completed warmup. Phase-boundary observations
are not exact transient GPU peaks; dedicated allocation traces supply separate
traced peak/count evidence. Post-reset/post-close retention alone is not proof
of a leak.

The eight fixed workload rows are Phi-3 CPU INT4/CUDA FP16 at budgets 0/1/16,
and SD1.5 CUDA batched/sequential at 512x512, seed42, 20 DDIM updates,
guidance7.5 and empty negative prompt. Prompts are fixed in the tool. There is
no ordinary-Session performance baseline, cross-precision speedup claim or
numerical performance threshold. Correctness still uses independent references.

### Measured native baselines

Linux, AMD EPYC 7V12, A100-SXM4-80GB ordinal0, driver580.159.04,
CUDA13.0.48/cuDNN9.23.1. Existing RelWithDebInfo LLVM-coverage-instrumented
CPU/plugin builds were measured with profile-file output disabled, not
instrumentation removed. Sessions retain the demos' intra/inter-op thread
settings1/1; no clocks/affinity/global driver settings were changed.
Python3.12.16, NumPy2.5.4, ONNX1.23.2, Transformers4.57.6, Pillow13.0.0,
Diffusers0.35.1 and CPU-only Torch2.8.0 were recorded.

All rows ran serially without concurrent agent-owned tests/profilers.
Setup and cold native execution are distinct; throughput is based on all five
steady samples, not median reciprocal or configured maximum token count.

| Workload | Setup(s) | Cold native(s) | Steady median(s) | p95(s) | Actual throughput |
|---|---:|---:|---:|---:|---:|
| Phi-3 CPU INT4,0tokens | 30.467634 | 4.127243 | 4.015727 | 4.155346 | N/A |
| Phi-3 CPU INT4,1token | 30.593458 | 4.458479 | 4.047851 | 4.182928 | 0.248120tokens/s |
| Phi-3 CPU INT4,16tokens | 30.454489 | 10.661899 | 10.677958 | 10.847453 | 1.497301tokens/s |
| Phi-3 CUDA FP16,0tokens | 14.139142 | 0.489360 | 0.008754 | 0.008786 | N/A |
| Phi-3 CUDA FP16,1token | 14.208991 | 0.490470 | 0.008794 | 0.008806 | 113.792169tokens/s |
| Phi-3 CUDA FP16,16tokens | 14.395683 | 0.669753 | 0.132899 | 0.133245 | 120.330205tokens/s |
| SD1.5 CUDA,batched | 8.982784 | 1.431545 | 0.685177 | 0.689988 | 1.456087images/s |
| SD1.5 CUDA,sequential | 8.938907 | 1.531622 | 0.803387 | 0.809567 | 1.248061images/s |

These are independent configuration-specific baselines, not cross-precision
or ordinary-Session speedup comparisons. Zero-token LLM rows still run their
declared prefill and report one invocation, not zero work.

Raw steady samples, seconds in collection order:

```text
PhiCPU0  [4.003400849,4.013123447,4.155346217,4.091763413,4.015726861]
PhiCPU1  [4.182928301,4.099062798,4.047850962,3.911162689,3.910555334]
PhiCPU16 [10.561810331,10.679640931,10.847452899,10.677957772,10.662612310]
PhiCUDA0 [0.008755997,0.008746309,0.008754384,0.008785712,0.008754073]
PhiCUDA1 [0.008806231,0.008793688,0.008771367,0.008797544,0.008770925]
PhiCUDA16[0.133245466,0.132899189,0.132712803,0.133135259,0.132844509]
SDBatched[0.689987660,0.685048587,0.685176847,0.688616496,0.685031692]
SDSeq    [0.803913432,0.794914198,0.803387282,0.794431700,0.809567386]
```

### Separate Step observations

These are separate cold Step executions, not a decomposition of the steady
Run samples above. Each executed decode step produces one additional token;
the first bootstrap sample invokes no decode model.

| Workload | Prefill(s) | Step-observed first token(s) | Mean executed decode/token(s) |
|---|---:|---:|---:|
| Phi CPU,0tokens | 4.526184 | N/A | N/A |
| Phi CPU,1token | 4.117321 | 4.118006 | N/A:no decode model call |
| Phi CPU,16tokens | 4.143985 | 4.144635 | 0.423662,15calls |
| Phi CUDA,0tokens | 0.497678 | N/A | N/A |
| Phi CUDA,1token | 0.492432 | 0.493156 | N/A:no decode model call |
| Phi CUDA,16tokens | 0.496491 | 0.497209 | 0.011389,15calls |

| SD Step pass | Positive/negative encoding(s) | 20denoising updates(s) | VAE(s) | Preprocess(s) | Safety(s) |
|---|---:|---:|---:|---:|---:|
| Batched | 0.203651/0.005504 | 1.020292 | 0.115357 | 0.041311 | 0.079554 |
| Sequential | 0.216012/0.005736 | 1.167257 | 0.116467 | 0.038089 | 0.033011 |

### Memory and allocation observations

Resource workers are separate fresh processes with two warmups and two
observed reuse/reset cycles. The GPU peak/count columns come from full-worker
CUDA allocation/free event lifetimes, including native setup, not NVML sampling
or CPU payload estimates. Dynamic device, pinned-host and static module storage
are distinguished using the exported trace enums.

| Workload | Process RSS high-water(MiB) | Approx. RSS after close(MiB) | Traced peak dynamic device(GiB) | Dynamic alloc/free |
|---|---:|---:|---:|---:|
| Phi CPU0 | 6633.81 | 596.62 | N/A | N/A |
| Phi CPU1 | 6634.18 | 596.82 | N/A | N/A |
| Phi CPU16 | 6675.25 | 614.75 | N/A | N/A |
| Phi CUDA0 | 2395.59 | 1892.62 | 16.008931 | 30/30 |
| Phi CUDA1 | 2394.61 | 1893.10 | 16.008931 | 30/30 |
| Phi CUDA16 | 2476.80 | 1974.53 | 16.047160 | 36/36 |
| SD batched | 2930.34 | 2873.57 | 7.059305 | 63/63 |
| SD sequential | 2858.88 | 2866.38 | 6.056376 | 72/72 |

RSS uses different Linux observations: `getrusage` high-water and approximate
`/proc/self/statm` current RSS at separate instants. Do not infer a leak or
require an ordering between those counters. Glibc live allocated/mmapped bytes
and region counts at every boundary are retained in the structured results;
they are not an ORT-only total malloc-call count.

Every traced dynamic GPU allocation has a matching free; end-of-process dynamic
live bytes are zero. Static device module storage is separately recorded
(832/860/892B for Phi CUDA0/1/16,219632/219560B for SD batched/sequential).
Post-execution-close NVML observations retain454/454/458MiB for Phi and468MiB
for either SD mode, including driver/context reservation. That is not equivalent
to live traced `cudaMalloc` bytes or a sanitizer leak. No DeviceReset, global
cache flush or leak suppression was used.

## Final validation and coverage

- Fresh CPU native/public 85/153; CUDA-plugin native/public 157/216; legacy
  native/selected-public 145/76; standalone shared/static package 5/5 each pass.
- Final Python: CPU73tests/217subtests, plugin77/235, legacy73/217 pass; the
  inherited legacy metadata warning remains. Direct offline qualification entry
  passes13tests. Scoped pyright reports0errors/0warnings; repository lint and
  whitespace checks pass. Eight focused
  ordinary-Session ownership/provider/inference tests pass per CPU/plugin.
- New reusable helper 255/255 and full-runner delta 2/2 production lines are
  unit-covered, without exclusions; the earlier tiny sample is covered 63/63.
- Actual Phi-3 CPU and CUDA each pass all three budget subtests, independent
  checkpoints, 64 retained caches, reset/replay and typed/package parity.
- Actual SD1.5 batched/sequential tests pass, including original/wrapper
  identity, placement, independent DDIM, exact preprocessing/safety and replay.
  Maximum latent/image RMSE is 0.003967436/0.001455491 batched and
  0.004360999/0.001573004 sequential, within unchanged approved bounds.
- Seven focused cache/sampler/chain lifetime tests pass each applicable memory/
  leak, race and synchronization tool. Actual Phi-3 memory/full leaks and
  full20 SD1.5 memory/full leaks report zero errors/leaks. Actual Phi-3 full-worker
  race/sync and full-size two-update SD all-kernel sync report zero errors/hazards.
- Approved bounded SD GroupNorm racecheck observes2sum160 launches/
  10240explicit barrier events and50sum256/466944events, matching freshly
  captured grid/block geometry. All302scale launches have zero shared memory;
  their global accesses remain covered by full memcheck.
- Actual native-only CUPTI windows assert Phi-3's three16B compact DtoH
  records=48B, without full cache/logit readback. Separate explicit snapshot
  cloning/readback records98304BDtoD and98304BDtoH. Both20-update SD guidance
  modes record2621440Bprediction DtoH; all control/conditioning/internal copy
  records are independently accounted. Copy-only windows need no compute
  kernel table; compute windows require nonzero actual dispatch.

### Exact current-source coverage

The [committed per-file ledger](Stateful_Qualification_Coverage.json) covers
70production source files, including declaration-only entries with no executable
delta. Exact deduplicated coverage is **7369/7369, no exclusions**:

| Scope | Covered/new executable locations |
|---|---:|
| Native host/API/glue/package | 5413/5413 |
| Python API/demo/tooling/runner | 1931/1931 |
| Actual CUDA source locations | 25/25 |
| New M24 production delta | 257/257(subset of the total) |

The denominator is the current delta from pre-M01 commit
`1988019b6e23ac4def4628cddd603c1bfb504fe8`, not all ORT or a sum of milestone
totals. Python includes fresh main and actual CUDA unit subprocess data, with
default exclusions cleared and installed/source bytes checked. Native maps use
verified unchanged source/object/profile cohorts plus the fresh matching
operator-name accessor export. M13's22source/40artifact cohort and preserved
original GPU ELFs were verified. Current Nsight Compute source counters cover
all25emitted greedy CUDA/SASS locations across all three actual kernels.
No dispatch/sanitizer result is substituted for device-line coverage.

### Requirement traceability

These ranges enumerate all93original requirement IDs. Verification is within
Appendix P's delivered scope, not an all-language completion claim.

| Requirement IDs | Status and acceptance evidence |
|---|---|
| SCOPE-1,SCOPE-3..6 | Verified foundation/demos/approved stacked workflow; published M24/#30 |
| SCOPE-2 | C/C++/Python verified; remaining stateful bindings deferred M16-M23 |
| STATE-1..6 | Native/Python context isolation, retained snapshots, reset/reuse, ownership, cancellation/failure tests; resource tools |
| LOOP-1..5 | Native loop/chain and public/Python step/run/termination/accounting suites |
| EXIT-1..7 | Explicit conditions, combinations, zero limits, model-bool/EOS/input exhaustion and invalid declarations |
| PREFILL-1..3 | Full-prompt finite input and empty-prompt rejection; real Phi-3 checkpoints |
| CHAIN-1..6 | Programmatic/package wiring, state/cache handoff, invalid destinations/types/order, downstream suppression |
| CACHE-1..7 | CPU/CUDA growth/share/copy/append/reset and exact ownership/device/stream/resource tests |
| SAMPLE-1..5 | Deterministic greedy/ties/nonfinite/layout contracts; native CUDA dispatch,25/25actual device locations |
| EP-1..5 | Built-in/plugin capability/subset/ABI/lifetime/error tests and actual routing/copy assertions |
| API-1..2 | Delivered C/C++/Python parity verified; deferred languages explicitly outstanding |
| API-3..6 | Existing Session/public ABI, RAII/ownership/error reporting and honest incomplete-parity boundaries |
| PKG-1..7 | Standalone shared/static authoring/parsing/confinement/compatibility and runtime/programmatic parity |
| LLM-1..6 | Pinned local CPU INT4/CUDA FP16 Phi-3,64caches, budgets0/1/16, reference/replay/typed parity |
| SD-1..5 | Pinned local CUDA SD1.5 plus deterministic CPU fixtures; native DDIM/guidance/state, mandatory safety/image handling |
| TEST-1..4 | Linux CPU/A100 CUDA execution; explicit unsupported/deferred hardware/platforms and known failures |
| COVER-1..4 | Exact7369/7369current-source ledger,257/257delta, matched host/device evidence, zero exclusions |
| METRIC-1..4 | Exact fixtures, fixed real-model tolerances/references/replay and eight reproducible native-only baseline rows |
| DELIVERY-1..7 | Separate interview/spec/plan approvals, current task/scratchpad/evidence and independently reviewable stack; M24/#30 OPEN/ready |

### Executed commands and proof locations

The machine-local raw evidence root is
`/datadisks/disk7/dtmp/ort-state-m24-819a29ec`. It shares the root filesystem;
capacity was checked rather than assuming a separate mounted disk. Variants/
images use same-filesystem immutable asset links under
`/home/devpatel/onnxruntime-stateful/m24-qualification-819a29ec`.
Earlier parent manifests/assets/ELFs remain unchanged.

From the existing virtualenv and outside source-package import shadowing:

```bash
ROOT=/datadisks/disk9/devpatel/ngenai/onnxruntime
E=/home/devpatel/.copilot/session-state/819a29ec-8ea8-4b2f-9938-e40236ffd239/files
Q=/datadisks/disk7/dtmp/ort-state-m24-819a29ec
source "$ROOT/.venv/bin/activate"
cd "$Q"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export LD_LIBRARY_PATH=/home/tianlei/cudnn_9.23_cuda13/lib:/home/tianlei/cuda13.0/lib64
export ORT_STATEFUL_TEST_BARRIER="$ROOT/build/state-cpu/RelWithDebInfo/libstateful_test_barrier.so"
SUITES=(
  "$ROOT/onnxruntime/test/python/onnxruntime_test_python_stateful.py"
  "$ROOT/onnxruntime/test/python/onnxruntime_test_python_stateful_package.py"
  "$ROOT/onnxruntime/test/python/onnxruntime_test_python_stateful_threads.py"
  "$ROOT/onnxruntime/test/python/onnxruntime_test_python_phi3_demo.py"
  "$ROOT/onnxruntime/test/python/onnxruntime_test_python_sd15_demo.py"
  "$ROOT/onnxruntime/test/python/onnxruntime_test_python_stateful_qualification.py"
)
PYTHONPATH="$E/m13-installed-cpu:$ROOT/onnxruntime/test/python" \
  python -m pytest -q "${SUITES[@]}"                         # PASS73/217
PYTHONPATH="$E/m15-installed-gpu:$ROOT/onnxruntime/test/python" \
  ORT_STATEFUL_REQUIRE_CUDA_PLUGIN=1 python -m pytest -q "${SUITES[@]}" \
  "$ROOT/onnxruntime/test/python/onnxruntime_test_python_stateful_cuda.py" \
  "$ROOT/onnxruntime/test/python/onnxruntime_test_python_cuda_adapter.py"
# PASS77/235 with matching CUDA/cuDNN LD_LIBRARY_PATH.
PYTHONPATH="$ROOT/build/state-cuda-legacy-m11-13.0/RelWithDebInfo:$ROOT/onnxruntime/test/python" \
  python -m pytest -q "${SUITES[@]}"                         # PASS73/217,1known warning
PYTHONPATH="$E/m13-installed-cpu:$ROOT/onnxruntime/test/python" \
  python "$ROOT/onnxruntime/test/python/onnxruntime_test_python_stateful_qualification.py"
# PASS13tests; direct full-runner entry.
```

`./onnxruntime_session_loop_test` in CPU/plugin/legacy build-output directories
passes85/157/145. `./onnxruntime_autoep_test` passes153CPU/216plugin.
Legacy's explicit
`--gtest_filter='AutoEpSelection.CudaEP:*Stateful*:*Ddim*:ModelPackageApiTest.*:*ModelPackageChain*'`
passes76. Both
`ctest --test-dir "$ROOT/build/state-package-m10" --output-on-failure` and
`ctest --test-dir "$ROOT/build/state-package-m10-static" --output-on-failure`
pass5/5.

Actual asset selectors
`TestPhi3RealAssets::test_cpu_real_reference_boundaries_replay_and_construction_parity`
and `::test_cuda_real_reference_boundaries_replay_and_construction_parity` each
pass1test/3subtests with explicit `ORT_STATEFUL_PHI3_ASSETS`; the complete
`onnxruntime_test_python_sd15_demo_assets.py` passes2tests with explicit
`ORT_STATEFUL_SD15_ASSETS`. All use local-only assets and isolated M24 output.

Actual device workers use
`compute-sanitizer --tool memcheck --leak-check full --show-backtrace no --error-exitcode 99`,
`--tool racecheck --show-backtrace no --error-exitcode 99`, and
`--tool synccheck --show-backtrace no --error-exitcode 99` directly, not on a
pytest parent. Each applicable scope passes zero errors/leaks/hazards. The exact
seven-test selector and bounded GroupNorm commands/geometry are recorded in
[TASKS](../TASKS.md) and raw logs/selection/proof JSON.
Privileged Nsight Compute is limited to the previously authorized profiler
process, with no global settings changed.

Coverage collection uses `python -m coverage run --rcfile=/dev/null`, including
GPU unit subprocess files, fresh LLVM signature-separated profiles and
`llvm-cov export` against the byte-matching provider. The cumulative ledger
records every current per-file SHA/count. `f6-cumulative-coverage.json`,
`f4-transfers-proof.json`, `f4-groupnorm-proof.json`, actual source-counter CSV,
`fixed-workloads.json`, `f5-collected-results.json` and
`f5-baseline-summary.json` retain the detailed machine-local proof.

## Explicit limitations and incomplete evidence

The inherited stateless plugin
`TestInferenceSession.test_set_providers_with_options` negative-device case
still reports `RuntimeError not raised`. It is one separately reproduced known
failure, not part of the passing-test count or a blanket regression waiver.
Legacy package-metadata warnings remain visible.

Real FP16 CUDA arithmetic need not be bit-identical. Approved fixed local/
component/RMSE/safety checks remain in force; accumulated maxima are reported,
not silently retuned. SD1.5 race qualification retains its explicitly approved
bounded GroupNorm scope, not unfiltered full-model race freedom. Zero-match or
zero-instrumentation probes and timed-out/stopped attempts are never passes.
The first current Phi-3 race attempt with eight analysis workers timed out at
600seconds and remains incomplete. The subsequent original automatic-
parallelism command completed without removing checks or narrowing scope.

Large raw logs, profiles, model/image artifacts and identity manifests are kept
outside Git in named qualification storage. The final per-file ledger is
committed; synchronized M24/#30 head/base and final immutable artifact manifest
are recorded in `m24-final-published-pr.json` and
`f7-source-artifact-proof.json` under that evidence root. Feature commit:
`3bcf4d69190fbe347cde1ea9e42d85c8c5642a0d`; its follow-up tracking-only commit
records publication. No PR was merged.
