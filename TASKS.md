# Stateful Session Execution: Execution Plan

**Status:** Approved by the user on 2026-10-15; M01 complete in [PR #11](https://github.com/onnxruntime/ort-state/pull/11).
**Source:** [Approved specification](SPEC.md), approved 2026-10-15.
**Working memory:** [Shared scratchpad](SCRATCHPAD.md).

## 1. Approval and tracking

- [x] Complete the requirements interview.
- [x] Ask whether the user has additional requirements before drafting.
- [x] Draft and review `SPEC.md` with the user.
- [x] Obtain specification approval.
- [x] Prepare this execution-plan draft.
- [x] Validate draft structure, dependency ordering, and all 93 requirement mappings.
- [x] Review this plan with the user and obtain approval before implementation.
- [ ] Complete the milestone stack and its validation gates.

M01 is complete; all subsequent milestones are pending. Check tasks when their
outcome is verified, not when code has merely been written. Record blockers, exact validation commands,
results, evidence locations, and PR links in the affected milestone as work
proceeds. Update this checklist and `SCRATCHPAD.md` at each meaningful checkpoint.

Planning validation: document-only Python 3 checks passed for all three planning
documents, 24 milestone sections/checklist rows, intended outcomes, CPU/CUDA test
plans, completion gates, dependency ordering, and specification-family mappings.
No runtime build, test, or coverage result is implied.

## 2. Execution and completion rules

### 2.1 Rules shared by every milestone

Each milestone is one intended, independently reviewable PR. It must build and
work on its declared prerequisites without relying on a later milestone.

1. Read applicable instructions and domain skills before implementation.
2. Confirm its concrete contracts and tests before adding production code.
3. Keep changes focused, following existing ORT style, helpers, error handling,
   and ownership conventions. Aim for at most ten changed source files per PR.
   If a milestone cannot remain reviewable, split it into explicitly documented
   sub-PRs here before proceeding, preserving outcomes and test gates.
4. Run the smallest relevant build/test selection first. Regenerate CMake when
   adding files or changing configuration. Broaden tests for coupled behavior,
   failures, and final qualification, not by default on every edit.
5. Run CPU EP validation for the milestone's outcome. Also run CUDA EP validation
   where meaningful, explicitly proving native dispatch where required.
6. Achieve measured 100% new executable production-line unit-test coverage.
   The percentage is not branch coverage and not whole-repository coverage.
7. Update directly related API, package, binding, or sample documentation.
8. Record exact commands/results, coverage counts, provider/device/configuration,
   and dependencies in its PR. Inspect and preserve the repository PR template.
9. Mark complete and open ready-for-review only after its gates pass.

Documentation-only sub-PRs need document checks rather than artificial EP tests;
their functional milestone still requires the CPU/CUDA evidence below.

### 2.2 Coverage gate

- Establish instrumentation in M01; do not assume `build.py --code_coverage`
  supports Linux. Its current implementation restricts that option to Android.
- Use suitable existing host tooling, such as compiler coverage plus gcov/LLVM
  reports, selected and verified in M01. Instrument runtime libraries, public
  inline wrappers, and native binding glue, not just the test executable.
- For each binding, combine its language-specific coverage with native-glue
  coverage; a managed-language report alone is insufficient.
- Unit-test new package/demo/tooling production lines, including error paths.
  Keep integration/demo evidence separate from the unit-coverage denominator.
- Measure CUDA device lines separately from CPU/CUDA host lines. Dispatch traces,
  profiler samples, sanitizer success, and test counts do not prove line coverage.
- Map measured executable lines to each PR's new production code and also report
  cumulative new-line coverage against the stack base. Exercise modified earlier
  code again when later milestones change its behavior.
- Record covered/total counts and uncovered locations; do not round 99.x% to 100%
  or quietly exclude hard-to-test code.
- Missing instrumentation or platform access blocks the affected completion gate.
  Request an explicit specification change if it cannot be satisfied. Never
  substitute an unsupported coverage claim.

### 2.3 Honest test results

- Confirm selected tests actually ran: nonzero discovered/executed test counts,
  expected cases present, and no unintended skips.
- Use explicit CPU or CUDA configuration, fresh artifacts, and provider dispatch
  assertions. CUDA-enabled builds that execute CPU fallback are not native CUDA
  evidence.
- Verify the artifact containing the change, including shared provider libraries.
- Missing local models, GPU access, toolchains, or platform-specific test runners
  mean blocked/not run, not passed.
- Reuse small deterministic ONNX fixtures. Never commit real model weights,
  credentials, downloaded third-party source, or environment-specific paths.

## 3. Milestone stack and dependencies

Stack order is the row order. The dependency column records actual semantic
prerequisites; the previous branch is also the mechanical Git base.
M01 is **complete**; M02-M24 are **pending**.

| Done | ID | Intended PR / branch suffix | Semantic prerequisites |
| --- | --- | --- | --- |
| [x] | M01 | Contracts, deterministic fixtures, coverage readiness / `contracts-fixtures` | Approved plan |
| [ ] | M02 | Stateful context and explicit SessionLoop / `context-loop` | M01 |
| [ ] | M03 | SessionChain and tensor/state bindings / `chain-bindings` | M02 |
| [ ] | M04 | CPU KV cache management / `cpu-kv-cache` | M03 |
| [ ] | M05 | CPU greedy sampling and decode iteration / `cpu-greedy` | M04 |
| [ ] | M06 | Public C API and C++ wrappers / `public-api` | M02, M03, M04, M05 |
| [ ] | M07 | Built-in/plugin EP capability contracts / `ep-stateful-hooks` | M04, M05, M06 |
| [ ] | M08 | Native CUDA KV management / `cuda-kv-cache` | M07 |
| [ ] | M09 | Native CUDA greedy sampling / `cuda-greedy` | M07 |
| [ ] | M10 | Standalone package schema and authoring / `package-schema` | M01, M03 |
| [ ] | M11 | ORT package-to-chain loading / `package-runtime` | M06, M07, M08, M09, M10 |
| [ ] | M12 | DDIM scheduler and denoising state / `ddim-policy` | M03, M06, M08 |
| [ ] | M13 | Python API and package parity / `python-api` | M06, M11, M12 |
| [ ] | M14 | Local Phi-3 package and text demo / `phi3-demo` | M05, M08, M09, M11, M13 |
| [ ] | M15 | Local Stable Diffusion package and image demo / `sd15-demo` | M11, M12, M13 |
| [ ] | M16 | C# binding / `csharp-api` | M06, M11, M12 |
| [ ] | M17 | Java/JNI binding / `java-api` | M06, M11, M12 |
| [ ] | M18 | Common JavaScript API and Node binding / `js-node-api` | M06, M11, M12 |
| [ ] | M19 | Browser/WASM binding / `js-web-api` | M18 |
| [ ] | M20 | Go binding / `go-api` | M06, M11, M12 |
| [ ] | M21 | Rust bindings / `rust-api` | M06, M11, M12 |
| [ ] | M22 | Objective-C binding / `objc-api` | M06, M11, M12 |
| [ ] | M23 | React Native binding / `react-native-api` | M18, M22 |
| [ ] | M24 | Integrated qualification and measurements / `qualification` | M01 through M23 |

Do not mark the whole feature complete after the CPU foundation or either demo;
all binding, provider, coverage, and qualification gates remain part of delivery.

## 4. Milestone details

### M01 - Contracts, fixtures, and validation readiness

**Outcome:** Concrete interface decisions and a reproducible validation baseline,
without prematurely publishing an ABI or starting a different product scope.
**Status:** Complete; [PR #11](https://github.com/onnxruntime/ort-state/pull/11)
is ready for review, targeting `ort-state`.

- [x] Inventory existing Session/I/O binding, allocator, cancellation, provider,
  sampling, scheduler, package, and binding helpers before adding alternatives.
- [x] Document concrete contracts in the specification's design appendix:
  signatures/lifetimes, token/EOS accounting, simultaneous exit reasons,
  supported dtypes/layouts, non-finite greedy handling, chain stepping, reset,
  scheduler policy integration, package fields/versioning, and EP/plugin ABI.
  Review material ambiguities with the user before implementing them.
- [x] Add reusable tiny deterministic fixtures and test helpers for finite
  inputs, feedback, renamed outputs, logits/KV, and denoising.
- [x] Establish host/new-line coverage collection and inventory language tools.
- [x] Probe CUDA source-line instrumentation feasibility; retain the requirement
  to measure actual production device lines in the CUDA milestones.
- [x] Identify local model/tokenizer requirements and required supplemental
  platform access. Ask for actual paths/access when needed, not assumed paths.

**CPU test plan:** Run fixture reference Sessions with CPU EP; verify known values
and repeatable baseline execution. Run existing CPU Model Package API tests.
Unit-test any new executable fixture/coverage tooling.
**CUDA test plan:** Check available toolkit/device and record configuration.
Run the same small fixtures with CUDA EP when available and relevant; confirm
which nodes execute there. No claim of native new features at this stage.
**Completion gate:** Baseline commands/counts and host coverage are verified.
Unresolved CUDA/mobile coverage/access risks are assigned to their affected
milestones; they need not prevent unrelated CPU development.

**Execution evidence:**

- First milestone branch created: `state/m01-contracts-fixtures`.
- Isolated `.venv` created with Python 3.12.9.
- Clang/LLVM coverage tools 18.1.8, GCC 13.2.0, and CMake 4.1.2 are present.
- Eight A100 GPUs and CUDA 13.0 are visible. Nsight Compute and Compute Sanitizer
  are installed; this is not evidence of device-line coverage.
- User confirmed all four M01 contracts: explicit native policies, initially
  single-sequence token generation, stable retained outputs, and the proposed
  float32/float16 greedy semantics. Recorded in SPEC Appendix A.
- Added four reusable in-memory model fixtures and five provider-parameterized
  Session reference cases. CPU instantiation is unconditional; CUDA instantiation
  requires a CUDA build and disables CPU EP fallback.
- CPU build passed. All 21 selected tests passed: five new reference cases and
  16 existing Model Package API cases, with zero skipped tests.
- Initial existing-package test failures were caused by missing staged
  `mul_1.onnx`/`mul_16.onnx` assets in the targeted build, not source defects.
  Staging those existing fixtures and rerunning resolved the failures.
- Host profile merge/export succeeded: all seven reusable fixture-helper
  functions and 103/103 executable helper lines executed. CPU test-driver
  coverage is 138/141 lines; three CUDA-setup lines are not executed on CPU.
  There are zero new production lines in M01: its production denominator is
  zero/not applicable, not a claimed production coverage percentage.
- LLVM aggregate reports warn about 60 profile-map entries in other, unchanged
  functions. Detailed diagnostics show none of the seven fixture helpers are
  mismatched. Do not ignore any future warning affecting newly added production
  code, and do not claim whole-repository coverage from this readiness check.
- A tiny debug CUDA kernel ran successfully under CUDA-GDB, with verified hits
  on both executable kernel-body lines (2/2) and normal process exit. This proves
  a source-line measurement mechanism is available; it is not CUDA EP validation
  or coverage of future cache/sampler kernels.
- No pre-existing local build tree was found. The first CPU compilation was
  interrupted at approximately 65%; the resumed build subsequently passed.
  The initial and resumed builds used:

```bash
source .venv/bin/activate
CC=clang CXX=clang++ python tools/ci_build/build.py \
  --build_dir build/state-cpu --config RelWithDebInfo --update --build \
  --parallel 8 --build_shared_lib --skip_submodule_sync \
  --cmake_generator 'Unix Makefiles' --target onnxruntime_autoep_test \
  --cmake_extra_defines \
  'CMAKE_C_FLAGS=-fprofile-instr-generate -fcoverage-mapping' \
  'CMAKE_CXX_FLAGS=-fprofile-instr-generate -fcoverage-mapping'
```

The build uses archive-based dependencies and avoids initializing unrelated
submodules. CMake regeneration and the resumed compilation passed.

Passing CPU validation commands:

```bash
cmake -E make_directory build/state-cpu/RelWithDebInfo/testdata
cmake -E copy_if_different onnxruntime/test/testdata/mul_1.onnx \
  onnxruntime/test/testdata/mul_16.onnx build/state-cpu/RelWithDebInfo/testdata
cd build/state-cpu/RelWithDebInfo
mkdir -p coverage/m01/passed
LLVM_PROFILE_FILE="$PWD/coverage/m01/passed/%m-%p.profraw" \
  ./onnxruntime_autoep_test \
  --gtest_filter='*StatefulExecutionBaselineTest.*:ModelPackageApiTest.*' \
  --gtest_output=xml:coverage/m01/passed/tests.xml
llvm-profdata merge -sparse coverage/m01/passed/*.profraw \
  -o coverage/m01/passed/m01.profdata
llvm-cov report ./onnxruntime_autoep_test -object ./libonnxruntime.so \
  -instr-profile=coverage/m01/passed/m01.profdata \
  ../../../onnxruntime/test/util/include/stateful_execution_test_utils.h \
  ../../../onnxruntime/test/autoep/test_stateful_execution_baseline.cc
```

- [x] Complete CUDA EP build and run the same reference cases with proved node
  assignment. An initial build failed due to mixed CUDA 13.0/13.1 toolkit headers;
  a separate consistent CUDA 13.1/A100 build passed.
  Unrelated contrib/Flash/MEA implementations are disabled for these tiny
  non-attention fixtures; this configuration does not qualify real LLMs.
- CUDA build test result: **26/26 passed**, including five CPU cases, five
  native-CUDA cases, and 16 existing Model Package API regressions. No tests
  were skipped. Each CUDA fixture asserts CUDA node assignment and disables
  CPU fallback.
- Formatting (`clang-format --dry-run --Werror`), 120-column/source whitespace
  checks, planning-document checks, and 93-requirement traceability checks passed.
- [x] Record final validation evidence, complete document/style checks, commit,
  and open the ready-for-review M01 PR.

Passing CUDA build/test commands, with `CUDA_HOME` and `CUDNN_HOME` set to the
matching local CUDA 13.1 installation and its cuDNN root:

```bash
source .venv/bin/activate
PATH="$CUDA_HOME/bin:$PATH" python tools/ci_build/build.py \
  --build_dir build/state-cuda-baseline-13.1 --config RelWithDebInfo \
  --update --build --parallel 8 --build_shared_lib --skip_submodule_sync \
  --cmake_generator 'Unix Makefiles' --target onnxruntime_autoep_test \
  --use_cuda --cuda_home "$CUDA_HOME" --cudnn_home "$CUDNN_HOME" \
  --disable_contrib_ops --nvcc_threads 1 --cmake_extra_defines \
  "CMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc" 'CMAKE_CUDA_ARCHITECTURES=80' \
  'onnxruntime_USE_FLASH_ATTENTION=OFF' \
  'onnxruntime_USE_MEMORY_EFFICIENT_ATTENTION=OFF'
cmake -E make_directory build/state-cuda-baseline-13.1/RelWithDebInfo/testdata
cmake -E copy_if_different onnxruntime/test/testdata/mul_1.onnx \
  onnxruntime/test/testdata/mul_16.onnx \
  build/state-cuda-baseline-13.1/RelWithDebInfo/testdata
cd build/state-cuda-baseline-13.1/RelWithDebInfo
CUDA_VISIBLE_DEVICES=0 \
  LD_LIBRARY_PATH="$PWD:$CUDNN_HOME/lib:$CUDA_HOME/lib64" \
  ./onnxruntime_autoep_test \
  --gtest_filter='*StatefulExecutionBaselineTest.*:ModelPackageApiTest.*' \
  --gtest_output=xml:m01-tests.xml
```

Published the unchanged `ort-state` baseline branch and
`state/m01-contracts-fixtures`; the first PR targets the agreed `ort-state` base,
not `main`. Implementation commit: `64aa026b94`. The PR contains no stateful
runtime production implementation; M02 is the next pending milestone.

### M02 - Execution context and SessionLoop

**Outcome:** Internal sequential stateful execution with explicit control.

- [ ] Implement reusable loop definition plus isolated retained execution state.
- [ ] Implement Step/Next and Run-until-exit on the same stepping path.
- [ ] Implement explicit EndOfInputs, EOS, iteration/token limits, and scalar
  boolean-output exits with OR semantics and documented simultaneous reasons.
- [ ] Implement output access/lifetimes, counters, reset, cancellation, and
  explicit requires-Reset failure state. No custom predicates.

**CPU test plan:** Exact iteration counts; input exhaustion; empty prefill;
EOS/limit/boolean exits alone and combined; zero limits; invalid conditions;
no extra execution after completion; Step/Run parity; caller accumulation;
reset/reuse; distinct-context isolation; cancellation before/during a step;
failure injection and refusal to reuse potentially corrupted state.
**CUDA test plan:** Repeat representative control/output-lifetime and cancellation
tests around a CUDA-backed Session. Cache and sampler native overrides are not
prerequisites for these generic loop tests.
**Completion gate:** Shared rules and 100% new context/loop-line coverage pass.

### M03 - SessionChain and tensor/state bindings

**Outcome:** Internal linear composition without a manifest state machine.

- [ ] Implement ordered Session/SessionLoop composition and external/result ports.
- [ ] Implement renamed tensor mapping, iteration feedback, retained-state slots,
  output selection, and staged handoff.
- [ ] Validate static/dynamic binding constraints and propagate component errors.
  Expose loop progress through the agreed chain interface without hidden resets.

**CPU test plan:** Session->Session, Session->Loop, Loop->Session, and Loop->Loop;
renamed/multiple outputs; feedback; retained outputs after later steps;
missing/duplicate destinations; invalid ordering/references; dtype/shape mismatch;
failure prevents downstream execution; chain reset and Step/Run equivalence.
**CUDA test plan:** Representative CUDA chain with device-value lifetime checks;
test declared transfers and explicit rejection of unsupported combinations.
**Completion gate:** Shared rules and 100% new chain/binding-line coverage pass.

### M04 - CPU KVCacheManager

**Outcome:** Safe contiguous caches shared across stages of one execution.

- [ ] Implement CPU cache initialization, model-declared layout/type/shape
  validation, logical length/capacity, append/update, retention, and reset.
- [ ] Integrate prefill->decode handoff with context/binding ownership.
- [ ] Share compatible allocations and explicitly copy/grow when required;
  preserve model/kernel aliasing rules and checked size arithmetic.

**CPU test plan:** Empty/initialized caches, growth boundaries, multiple layers,
sequential append, renamed handoff, permitted buffer sharing versus copy,
output/input non-alias contracts, reset without stale data, released owners,
invalid layout/length/dtype/capacity, size overflow, and allocation failure.
**CUDA test plan:** No native implementation yet. Check device-input rejection
and that the CPU cache contract does not masquerade as CUDA cache support.
**Completion gate:** CPU chain fixture matches reference; sharing/copy behavior
is observable; shared rules and 100% new cache-line coverage pass.

### M05 - CPU greedy sampling and decode integration

**Outcome:** CPU token selection and a complete tiny prefill/decode chain.

- [ ] Implement documented greedy selection using existing suitable helpers.
- [ ] Validate supported logits shapes/dtypes/position, ties, non-finite behavior,
  token IDs, and integration with explicit EOS/token-limit conditions.
- [ ] Wire prefill-logit first-token handling and iterative decode without
  duplicates or omissions. Keep caller accumulation explicit.

**CPU test plan:** Known argmaxes, ties, negative values, supported precisions,
invalid/empty logits, non-finite policy, vocabulary boundaries, first-token EOS,
token budget across stage handoff, and cache/logit consistency after reset.
Compare the tiny chain token-by-token against direct Session calls.
**CUDA test plan:** No device-native sampler yet. Verify explicit behavior when
device-resident logits reach the CPU-only implementation; no hidden fallback.
**Completion gate:** CPU end-to-end fixture, shared rules, and 100% new sampler/
integration-line coverage pass.

### M06 - Public C API and C++ wrappers

**Outcome:** Complete programmatic access to the implemented core primitives.

- [ ] Add append-only public API entries and opaque resources consistent with
  existing tables. Do not bump `ORT_API_VERSION`.
- [ ] Expose configuration, bindings, exit conditions, contexts, loop/chain
  execution, cache/sampler integration, outputs, reset, cancellation, and release.
- [ ] Add RAII C++ wrappers and Doxygen ownership/lifetime/`\since` documentation.
  Ensure failed calls preserve out-parameters and contain exceptions.

**CPU test plan:** Reproduce M02-M05 behavior through C and C++, not only internals;
null/invalid handles, ownership/release ordering, output lifetime, errors,
resource cleanup, and compatibility with ordinary Session/API use.
**CUDA test plan:** Public-API generic CUDA SessionLoop/Chain smoke plus explicit
unsupported native-feature reporting until the corresponding milestone.
**Completion gate:** C/C++ consumers compile; ABI table checks and shared rules
pass; 100% new API/wrapper-line coverage, including inline code, is measured.

### M07 - Built-in and plugin EP customization

**Outcome:** Providers can implement cache features and/or sampling independently.

- [ ] Add capability-based in-tree hooks and boundary-safe plugin C contracts.
- [ ] Route CPU implementations through the selection contract; specify allowed
  transfers/fallback and make implementation selection observable.
- [ ] Extend a test plugin to demonstrate partial/full support, lifetimes,
  unsupported features, errors, and compatibility with older plugin contracts.

**CPU test plan:** Execute real CPU/cache/sampler overrides and test-plugin calls;
no-override and subset cases; version/size/capability checks; provider failure;
release ordering and rejection of unsafe or undisclosed fallbacks.
**CUDA test plan:** Load/test the extension mechanism in a CUDA-enabled build;
verify unavailable CUDA stateful capabilities are still reported accurately.
**Completion gate:** Built-in and plugin routes both execute; public boundary
documentation and 100% new host/glue-line coverage satisfy shared rules.

### M08 - Native CUDA KV management

**Outcome:** Device-resident cache lifecycle and prefill/decode handoff.

- [ ] Implement CUDA allocation/retention/update/growth/reset using ORT allocator,
  device, and stream abstractions; advertise only supported features.
- [ ] Wire stage sharing and required movement without illegal aliasing.
- [ ] Expose dispatch/copy/allocation evidence for tests and later measurement.

**CPU test plan:** Rerun CPU cache and EP-contract regressions.
**CUDA test plan:** CPU/CUDA cache contents and lengths agree on fixtures; verify
device placement, compatible-buffer sharing, required copies, nondefault stream
ordering, completion before observation/release, growth/reset, invalid devices,
allocation/provider failures, cancellation, and resource cleanup.
**Completion gate:** Actual CUDA dispatch plus measured 100% new host/device-line
coverage. Device coverage unavailability blocks completion, not just a report.

### M09 - Native CUDA greedy sampling

**Outcome:** Greedy token selection without an undisclosed host sampler.

- [ ] Implement CUDA greedy reduction/selection with CPU-equivalent documented
  tie/non-finite semantics and supported dtype/layout handling.
- [ ] Integrate provider discovery, stream ordering, token observation, and exits.
- [ ] Keep cache and sampler capability selection independent.

**CPU test plan:** Rerun greedy/EP baseline and unsupported-capability tests.
**CUDA test plan:** Exact token agreement on deterministic fixtures, vocab sizes
around launch/reduction boundaries, supported dtypes, ties/non-finite cases,
stream ordering, error cases, native-dispatch assertion, and no host sampling.
Exercise the complete tiny prefill/decode chain using M08 caches.
**Completion gate:** Shared rules, CPU parity, and 100% new host/device-line
coverage pass. Latency/copy measurements have no performance pass threshold.

### M10 - Standalone Model Package schema and authoring

**Outcome:** Portable declarative chain metadata without an ORT dependency.

- [ ] Implement the agreed additive schema/version contract, chain ordering,
  component execution kinds, bindings, and explicit policy/condition metadata.
- [ ] Extend C inspection/authoring/commit/round-trip surfaces and fixtures.
- [ ] Preserve old packages, strict parsing, paths, shared assets, and opaque
  consumer payloads; keep execution interpretation outside this library.

**CPU test plan:** Standalone inspection/authoring/commit/asset tests; old/new
schema fixtures; absent optional fields; unknown fields; malformed chains,
versions, and references; portable confinement; installed/shared-asset paths;
round-trip preservation. Rerun existing CPU ORT Model Package API tests.
**CUDA test plan:** No device logic in this standalone library; schema tests are
provider-independent. CUDA chain execution is validated in M11.
**Completion gate:** Standalone build proves no new ORT linkage; package docs and
100% new parser/authoring/API-line coverage satisfy shared rules.

### M11 - ORT package-to-chain loading

**Outcome:** Equivalent stateful execution from a package or programmatic setup.

- [ ] Parse execution-specific policy metadata in ORT's consumer layer; reuse
  existing variant selection, path resolution, options, and Session construction.
- [ ] Construct public loop/chain resources and expose package loading via C/C++.
- [ ] Preserve legacy package Session creation and reject invalid executable
  scripts/transitions, incompatible variants, options, and configurations.

**CPU test plan:** Package/programmatic parity for outputs, exit reasons, reset,
cache ownership, cancellation/errors; malformed/missing conditions and wiring;
external model data/shared assets; existing options and confinement regressions.
**CUDA test plan:** Select actual CUDA variants; prove M08/M09 dispatch; exercise
CPU/CUDA packages and explicit failure when required capabilities/variants are
missing. A successful CPU fallback is not successful CUDA package validation.
**Completion gate:** Both construction paths and legacy APIs pass shared rules
with 100% new loader/API-line coverage.

### M12 - DDIM scheduler and denoising execution policy

**Outcome:** Stateful diffusion iteration, not LLM-specific cache emulation.

- [ ] Implement one documented SD 1.5 DDIM convention: timestep schedule, noise/
  latent update, model prediction interpretation, and guidance combination.
- [ ] Bind timesteps/latents/model outputs through SessionLoop; expose supported
  scheduler configuration programmatically and through native package policies.
- [ ] Make seeded initialization/reset reproducible and validate configuration.
  Document any CPU coefficient/latent work and required CUDA transfers.

**CPU test plan:** Independent analytical/reference DDIM steps and multi-step
trajectory; guidance scale boundaries; timestep/count/shape validation; seed
replay after reset; retained latent ownership; scheduler error/cancellation.
**CUDA test plan:** Tiny CUDA denoiser loop matches reference within specified
tolerances; verify device placement, transfers, step count, and resource lifetime.
A device-native scheduler is not required, but host work must be explicit.
**Completion gate:** New scheduler/policy production lines meet coverage and
shared rules; coefficients/trajectory references and tolerances are documented.

### M13 - Python bindings

**Outcome:** Python construction and package execution with full core parity.

- [ ] Bind contexts, loops/chains, bindings/conditions, stateful policies,
  sampler/cache selection, output access, reset, and cancellation.
- [ ] Follow existing Python/pybind ownership, errors, GIL, and array/OrtValue
  conventions without reimplementing orchestration in Python.
- [ ] Document programmatic/package examples and tokenizer adapter boundaries.

**CPU test plan:** Unit tests for each new wrapper and malformed-input/error path;
construct/step/run/reset/release, caller output accumulation, package parity,
lifetime/GC, and tiny LLM/diffusion fixtures using CPU EP.
**CUDA test plan:** Equivalent fixtures with explicit CUDA/native cache/sampler
dispatch; verify device values and lifetime across Python/native boundaries.
**Completion gate:** Tests use the built library, not a stale installed wheel;
100% new Python/native binding lines and shared rules pass.

### M14 - Phi-3 local package and text-to-text demo

**Outcome:** User-local Phi-3 models execute through the new runtime abstractions.

- [ ] Inspect supplied model signatures, external weights, precision, cache
  layout, and CPU/CUDA variant compatibility before authoring a package.
- [ ] Add tested package-building and demo tooling with caller-provided tokenizer/
  detokenizer adapters, explicit exits, full-prompt prefill, and greedy decode.
- [ ] Document model files/variants, execution commands, expected behavior,
  reference implementation, EOS handling, and stage-boundary token accounting.

**CPU test plan:** Unit-test new builder/demo lines with deterministic models and
fake text adapters, including missing assets and adapter errors. Then run real
local Phi-3 text-in/text-out, reset/replay, and compare against the reference.
**CUDA test plan:** Repeat the real demo using compatible local CUDA models and
native KV/sampling; compare appropriate logits/cache checkpoints with documented
tolerances, not mandatory cross-provider identical generated prose.
**Completion gate:** Both real runs, fixture coverage, construction parity, and
shared rules pass; no weights/downloads are committed or silently substituted.

### M15 - Stable Diffusion local package and image demo

**Outcome:** Text-to-image generation via the new chain and denoising loop.

- [ ] Build a package from local SD 1.5-compatible encoder/denoiser/VAE files.
- [ ] Add tested text/tokenizer integration, guidance inputs, seeded noise,
  DDIM configuration, VAE output conversion, and explicit image saving.
- [ ] Document exact export/model assumptions, reference/tolerances, seed,
  dimensions, step count, guidance, preprocessing, and output handling.

**CPU test plan:** Unit-test all new builder/demo/postprocessing lines, including
file/adapter errors. Real local text-to-image run; verify expected image shape/
encoding, scheduler trajectory, reference checkpoints, and same-config replay.
**CUDA test plan:** Real local CUDA run; verify actual denoiser execution, state/
transfer behavior, documented checkpoint tolerances, seeded replay, and output.
**Completion gate:** CPU/CUDA image artifacts and reference evidence exist;
100% new production-line coverage and shared rules pass.

### M16 - C# binding

**Outcome:** Managed programmatic/package parity with safe native ownership.

- [ ] Implement native interop, typed configuration/results, disposal semantics,
  all core operations, and diffusion policy access.
- [ ] Add API docs/examples and focused tests without requiring unrelated
  pretrained-model downloads.

**CPU test plan:** Direct Linux `dotnet test` with the locally built native
library; fixtures cover lifecycle, state/conditions, invalid input, package
parity, disposal/finalization, output retention, and cancellation.
**CUDA test plan:** Same relevant fixtures using explicit CUDA and native
cache/sampler capability checks.
**Completion gate:** Managed plus native-glue new-line coverage meets 100%.
Do not rely on `build.py` C# test success: its current runner skips non-Windows.

### M17 - Java/JNI binding

**Outcome:** Java Session users can use the complete new execution contract.

- [ ] Implement Java/JNI resources, configuration/results, policy access, error
  mapping, and existing close/ownership conventions.
- [ ] Regenerate JNI declarations through the existing build and document usage.

**CPU test plan:** Existing CMake/Gradle runner with targeted new JUnit tests:
programmatic/package parity, close ordering, retained values, invalid arguments,
exceptions, stepping/reset, cancellation, and tiny pipelines.
**CUDA test plan:** Equivalent relevant tests with native CUDA dispatch.
**Completion gate:** Java/native artifacts come from this checkout; 100% new
Java/JNI-line coverage, formatting, and shared rules pass.

### M18 - Common JavaScript API and Node binding

**Outcome:** Shared typed interfaces plus a working native Node implementation.

- [ ] Add common execution/configuration/result contracts without breaking
  existing InferenceSession or requiring deprecated backends.
- [ ] Implement Node/native wrappers, resource ownership, package access, policies,
  cancellation, and iteration output semantics.
- [ ] Document JavaScript event-loop behavior of the synchronous native contract
  without adding asynchronous streaming.

**CPU test plan:** Common API/TypeScript and Node tests for full programmatic/
package parity, all exit conditions, retained tensors, disposal, errors,
cancellation, and tiny pipelines. Assert the local addon/library is loaded.
**CUDA test plan:** Native Node CUDA fixture runs, cache/sampler dispatch, device
value lifetime, and explicit capability errors.
**Completion gate:** Shared/Node/native new-line coverage is 100%; tests actually
execute the addon, rather than only mocked TypeScript interfaces.

### M19 - Browser/WASM binding

**Outcome:** Stateful execution through the supported Web/WASM Session path.

- [ ] Expose the common API through WASM/native exports and worker/proxy paths.
- [ ] Define package loading from browser-accessible resources/virtual files
  without pretending the browser has an unrestricted host filesystem.
- [ ] Preserve declared synchronous stepping semantics at the execution boundary;
  review any unavoidable transport-level adaptation before changing public APIs.

**CPU test plan:** Linux headless-browser tests executing real WASM CPU models;
programmatic/package parity, multiple inputs/outputs, transfers and retained
buffers, reset/cancellation, worker paths, invalid/missing files and conditions.
Include underlying CPU core regressions.
**CUDA test plan:** Not applicable inside browser/WASM. Explicitly report CUDA
as unavailable there; native CUDA parity remains covered by Node and core tests.
**Completion gate:** Real WASM execution plus 100% new TS/WASM-glue lines;
no new JSEP/WebGL feature work or false CPU substitute for native WebGPU.

### M20 - Go binding

**Outcome:** Idiomatic Go access through the existing dynamically loaded C API.

- [ ] Extend shims and wrappers, configuration/results, policies, handles, errors,
  and close semantics; synchronize required local API header copies.
- [ ] Add examples and unit tests with explicit native-library configuration.

**CPU test plan:** Targeted Go tests with `ORT_LIB_PATH` pointing to the local
library; all operations, context isolation, retained tensors, errors/cleanup,
package parity, tiny pipelines, and applicable race checks.
**CUDA test plan:** Equivalent relevant fixtures with explicit CUDA selection
and native cache/sampler evidence.
**Completion gate:** Go/shim new-line coverage is 100%; nonzero tests ran.
The existing TestMain can exit successfully when no library is found: that is
not acceptable evidence.

### M21 - Rust bindings

**Outcome:** Low-level and safe Rust wrappers with the same execution semantics.

- [ ] Extend low-level C API access and safe owned/borrowed resources without
  rewriting unrelated experimental binding code.
- [ ] Add typed configuration/results, policy support, examples, and tests;
  use the current local ORT build, not downloaded historical binaries.

**CPU test plan:** Focused crate unit/integration/doc tests for all operations,
lifetimes/drop ordering, errors, retained outputs, package parity, and fixtures.
**CUDA test plan:** Relevant tests linked to the local CUDA build with dispatch
and device-resource assertions.
**Completion gate:** Low-level/authored safe-wrapper and native-glue line
coverage is 100%; binding limitations are explicit, not silent parity omissions.

### M22 - Objective-C binding

**Outcome:** Objective-C Session users receive programmatic/package parity.

- [ ] Implement resources/configuration/results, policy access, NSError mapping,
  memory ownership, and documentation following existing Objective-C++ patterns.
- [ ] Add native binding tests for all new paths and link against the new API.

**CPU test plan:** Linux CPU core/API regressions plus actual Objective-C binding
tests on an approved Apple runner using CPU EP, including lifecycle, reset,
outputs, cancellation, package parity, and errors.
**CUDA test plan:** CUDA is not a supported Apple Objective-C runtime target;
core CUDA regressions remain covered elsewhere.
**Completion gate:** Linux core tests alone do not validate this binding.
Apple compilation/tests and 100% new Objective-C++/glue-line coverage are
required. If that access is unavailable, leave this milestone blocked.

### M23 - React Native binding

**Outcome:** Shared JavaScript semantics work through Android and iOS bridges.

- [ ] Extend common/native/mobile bridges and package-resource loading.
- [ ] Follow existing Java/Kotlin/Objective-C++ ownership and exception patterns.
- [ ] Add documentation plus executable host/mobile tests, not only bridge mocks.

**CPU test plan:** Linux host C++/JS tests; Android emulator CPU EP end-to-end
fixtures from Linux; iOS CPU EP fixtures on an approved Apple runner.
Exercise state, retained outputs, errors, reset/cancellation, and package parity.
**CUDA test plan:** Not applicable to Android/iOS runtime targets. Retain native
Linux CUDA compatibility checks for shared code changes.
**Completion gate:** Both mobile paths and all new JS/native bridge lines meet
coverage. Missing iOS access blocks full completion; Android success is partial.

### M24 - Integrated qualification, measurements, and final documentation

**Outcome:** A verified, documented foundation with honest end-to-end evidence.

- [ ] Close every specification requirement against tests, coverage and PR links.
  No binding, native provider, or real-model run remains implicitly unchecked.
- [ ] Run coupled regressions across the stack, resource-lifetime/failure tests,
  applicable sanitizers, and repeated reset/reuse scenarios.
- [ ] Record reproducible CPU/CUDA latency, throughput, peak/retained memory,
  relevant allocations/copies, warmup, model precision, and environment.
- [ ] Finalize examples, API/package migration guidance, capability/platform
  matrices, cumulative coverage reports, and milestone-stack documentation.

**CPU test plan:** Integrated small-fixture matrix plus both real local demos;
ordinary Session/package regressions; all applicable language-binding suites.
**CUDA test plan:** Both demos and native cache/sampler/chain tests with proved
dispatch; exact deterministic results and justified real-model tolerances.
**Completion gate:** All milestones complete, cumulative new-line unit coverage
is 100%, required tests are executed, and no blocker is represented as passed.
Performance numbers establish a baseline, not a numerical pass/fail threshold.

## 5. Planned validation recipes

These are **planned commands, not executed results**. They must be finalized
against available local toolchains and new test names in M01 and updated in the
relevant milestone. Build outputs below are deliberately distinct CPU/CUDA trees;
reuse an existing compatible build only after recording its configuration.

Activate an existing isolated Python environment, or create one when needed,
before builds/tests. Do not install/restore dependencies speculatively.

### 5.1 Core builds and test selection

From the repository root, initial CPU setup/build:

```bash
python tools/ci_build/build.py --build_dir build/state-cpu \
  --config RelWithDebInfo --update --build --parallel --build_shared_lib
```

For a CUDA build, with actual validated `CUDA_HOME` and `CUDNN_HOME`:

```bash
python tools/ci_build/build.py --build_dir build/state-cuda \
  --config RelWithDebInfo --update --build --parallel --build_shared_lib \
  --use_cuda --cuda_home "$CUDA_HOME" --cudnn_home "$CUDNN_HOME"
```

Incremental core-only example; regenerate first when files/configuration change:

```bash
python tools/ci_build/build.py --build_dir build/state-cpu \
  --config RelWithDebInfo --build --parallel --target onnxruntime_test_all
```

Run tests from the relevant build output directory:

```bash
cd build/state-cpu/RelWithDebInfo
./onnxruntime_test_all --gtest_list_tests
./onnxruntime_shared_lib_test --gtest_list_tests
./onnxruntime_autoep_test --gtest_filter='ModelPackageApiTest.*'
```

M01 assigns/discovers targeted new loop/chain/cache/sampler suite names. Replace
list-only commands with those verified nonempty filters for each milestone;
build the matching shared-lib/autoep/provider target as needed. Repeat applicable
filters in the CUDA tree, with tests explicitly selecting the intended EP.
Do not assume provider/kernel tests reside in `onnxruntime_test_all`.

Coverage requires separate verified instrumentation/configuration; the build
examples above are not claimed to produce coverage by themselves.

### 5.2 Standalone package tests

```bash
cmake -S model_package -B build/state-package \
  -DMODEL_PACKAGE_BUILD_TESTS=ON
cmake --build build/state-package --parallel
ctest --test-dir build/state-package --output-on-failure
```

Use an existing local nlohmann/json dependency where available. Confirm expected
inspection, authoring, commit, and hashing tests run, plus the new cases.

### 5.3 Binding validation entry points

| Binding | Existing entry point / planned adaptation |
| --- | --- |
| Python | Build with `--enable_pybind`; run `python -m pytest <new targeted test file>` against that build, outside source-package import shadowing as needed. Add wheel build only when packaging validation requires it. |
| C# | Direct `dotnet test` of `csharp/test/Microsoft.ML.OnnxRuntime.Tests.NetCoreApp/Microsoft.ML.OnnxRuntime.Tests.NetCoreApp.csproj`, focused `--filter`, Linux properties and explicit native output path. `build.py` currently skips C# test execution on Linux. |
| Java | Build with `--build_java`; use the generated `onnxruntime4j_test` CTest/Gradle path with focused tests and the correct `cmakeBuildDir`. |
| Node | Build the local addon using existing scripts/`--build_nodejs`; `npm test --prefix js/node` with the verified focused selector. |
| Web | Build local WASM artifacts and run `npm test --prefix js/web` with validated headless-browser/WASM arguments from its test runner. No downloaded prebuilt artifact may stand in for new native exports. |
| Go | From `go/`: `ORT_LIB_PATH=<absolute native-library directory> go test -v ./onnxruntime -run <verified selector> -coverprofile=<report>`; separately measure shim/native coverage. |
| Rust | From `rust/`: `ORT_RUST_STRATEGY=system ORT_RUST_LIB_LOCATION=<local ORT location> cargo test --workspace <verified selector>` with native library loading configured. Establish Rust coverage tooling before claiming line coverage. |
| Objective-C | Existing Objective-C++ test target on an approved Apple runner; Linux CPU core regressions are supplemental, not a substitute. |
| React Native | `npm run test:cpp --prefix js/react_native`, focused JS tests, and actual Android/iOS CPU EP test applications using rebuilt native libraries. |

Angle-bracket values above are intentional placeholders, not runnable commands.
Record the exact resolved commands and results when executing the milestone.

## 6. Specification traceability

All detailed requirements within each listed family remain applicable, including
negative/error tests. Supporting milestones do not replace final qualification.

| Specification requirements | Primary milestones |
| --- | --- |
| SCOPE-1 through SCOPE-6 | M02-M24, with workflow M01 and Section 7 |
| STATE-1 through STATE-6 | M02-M04, M06, M12, each binding, M24 |
| LOOP-1 through LOOP-5 | M02, M03, M06, M11, each binding |
| EXIT-1 through EXIT-7 | M02, M05, M06, M11, each binding |
| PREFILL-1 through PREFILL-3 | M02, M05, M14 |
| CHAIN-1 through CHAIN-6 | M03, M06, M11, each binding |
| CACHE-1 through CACHE-7 | M04, M07, M08, M14 |
| SAMPLE-1 through SAMPLE-5 | M05, M07, M09, M14 |
| EP-1 through EP-5 | M07-M09, M11 |
| API-1 through API-6 | M06, M11-M13, M16-M23 |
| PKG-1 through PKG-7 | M10, M11, M14, M15, binding package tests |
| LLM-1 through LLM-6 | M14, M24 |
| SD-1 through SD-5 | M12, M15, M24 |
| TEST-1 through TEST-4 | Every milestone, Sections 2 and 5 |
| COVER-1 through COVER-4 | M01, every production-code milestone, M24 |
| METRIC-1 through METRIC-4 | M01, M08, M09, M12, M14, M15, M24 |
| DELIVERY-1 through DELIVERY-7 | Sections 1, 2, and 7; maintained throughout |

## 7. Stacked PR and checkpoint workflow

- First branch: `state/m01-contracts-fixtures`, based on the current `ort-state`
  branch at implementation start. Record the actual base commit then.
- Subsequent branches: `state/m02-context-loop`, and so on using the table
  suffixes, each based on the preceding milestone branch.
- Target repository: `onnxruntime/ort-state`. Each PR's base is the previous
  milestone branch; the first PR targets `ort-state`.
- Each PR documents incremental outcome, prerequisites/stack position, directly
  related usage, exact build/test results, measured coverage, and known limits.
  Preserve the repository template's headings/comments/checklists/order.
- Do not open ready-for-review while that milestone's completion gate is blocked.
  A later PR may depend on an unmerged but verified predecessor.
- Keep later branches rebased/restacked only with care for user changes and
  remote branch safety; do not rewrite shared history destructively.
- Include the required Copilot co-author trailer on commits unless the user
  asks otherwise. Never commit real model files or environment-specific assets.
- Do not auto-merge. Human review and repository policy determine merge timing.
- Keep ordinary source/build outputs out of the scratchpad; record concise
  decisions, discoveries, blockers, and evidence pointers, including agent
  learnings if agents are used later.

## 8. Known blockers and decisions to verify

| Item | Current status | Affected gates / resolution |
| --- | --- | --- |
| Plan approval | Approved 2026-10-15 | M01 authorized; later work follows the approved dependencies and gates. |
| Concrete API/schema contracts | Semantic requirements approved; exact contracts not yet implemented | M01 documents within approved scope; material choices return to user. |
| CPU/compiler/language coverage readiness | Not probed by running builds/tools during planning | Establish in M01; verify 100% per affected PR. |
| CUDA device-line coverage | Tooling feasibility unverified | M01 investigates; M08/M09 cannot complete with host-only measurements. |
| Local Phi-3/SD files and adapters | Paths/integrations not supplied | Request concrete paths before M14/M15; synthetic fixtures do not replace real demos. |
| CPU/CUDA model variants | Actual signatures/precision not inspected | Verify compatible local variants; no silent model substitution/download. |
| Objective-C/iOS validation | Linux is the initial platform; Apple access not provided | M22/M23 need supplemental runner/coverage or remain blocked; no implicit scope reduction. |
| Browser/mobile package resources | Runtime file APIs differ from native filesystem APIs | M01/M19/M23 define adapters preserving package confinement and execution parity. |

Approval record: on 2026-10-15, the user approved this plan without changes and
explicitly authorized beginning M01. Changes to scope or mandatory gates still
require user review.
