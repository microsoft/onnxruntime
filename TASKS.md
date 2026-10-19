# Stateful Session Execution: Execution Plan

**Status:** Approved; M01-M08 and M09a complete and published for review;
M09b requires separate authorization; M10-M24 pending.
M05: [PR #15](https://github.com/onnxruntime/ort-state/pull/15).
M05's requested generation-folder layout follow-up is complete in the same PR.
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

M01-M08 and M09a are complete; M09b/M10-M24 are pending. Check tasks when their
outcome is verified, not when code has merely been written. Record blockers, exact validation commands,
results, evidence locations, and PR links in the affected milestone as work
proceeds. Update this checklist and `SCRATCHPAD.md` at each meaningful checkpoint.

Planning validation: document-only Python 3 checks passed for all three planning
documents, 24 milestone sections/checklist rows, intended outcomes, CPU/CUDA test
plans, completion gates, dependency ordering, and specification-family mappings.
No runtime build, test, or coverage result is implied.

Coverage-command source paths track the current tree layout. Historical milestone
counts remain the evidence recorded when those milestones were completed.

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
M01-M08 and M09a are **complete**; M09b/M10-M24 are **pending**.

| Done | ID | Intended PR / branch suffix | Semantic prerequisites |
| --- | --- | --- | --- |
| [x] | M01 | Contracts, deterministic fixtures, coverage readiness / `contracts-fixtures` | Approved plan |
| [x] | M02 | Stateful context and explicit SessionLoop / `context-loop` | M01 |
| [x] | M03 | SessionChain and tensor/state bindings / `chain-bindings` | M02 |
| [x] | M04 | CPU KV cache management / `cpu-kv-cache` | M03 |
| [x] | M05 | CPU greedy sampling and decode iteration / `cpu-greedy` | M04 |
| [x] | M06 | Public C API and C++ wrappers / `public-api` | M02, M03, M04, M05 |
| [x] | M07 | Built-in/plugin EP capability contracts / `ep-stateful-hooks` | M04, M05, M06 |
| [x] | M08 | Native CUDA KV management / `cuda-kv-cache` | M07 |
| [ ] | M09 | Native CUDA greedy sampling / `cuda-greedy` | M07, M08 |
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
**Status:** Complete; [PR #12](https://github.com/onnxruntime/ort-state/pull/12)
is ready for review. Branch `state/m02-context-loop` stacks on
`state/m01-contracts-fixtures`. Implementation commit: `a21ae23201`.

- [x] Implement reusable loop definition plus isolated retained execution state.
- [x] Implement Step/Next and Run-until-exit on the same stepping path.
- [x] Implement explicit EndOfInputs, EOS, iteration/token limits, and scalar
  boolean-output exits with OR semantics and documented simultaneous reasons.
- [x] Implement output access/lifetimes, counters, reset, cancellation, and
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

**Execution evidence:**

- Internal native implementation and focused test target
  `onnxruntime_session_loop_test` added. No public ABI changes.
- CPU focused suite: **15/15 tests passed**, no failures/skips.
- CPU stateless/package regression suite: **21/21 passed**, no failures/skips.
- Final LLVM coverage: **315/315 new executable production lines** (312 in
  `session_loop.cc`, three in the header), measured after the final rebuild.
  Final focused CPU tests remain **15/15 passed**.
- Detailed LLVM diagnostics list 14 other-function profile-map warnings and no
  mismatches affecting SessionLoop functions.
- CUDA compilation initially encountered use of a protected initialization query.
  The implementation now uses the existing `NewIOBinding` status contract to
  validate initialized Sessions; CPU compilation/testing passed with this fix.
  The corrected CUDA build and test suites passed.
- Cancellation posts an atomic request before/after Session invocation, rather
  than cross-thread-writing the existing non-atomic ORT termination flag.
  A deterministic in-flight cancellation test exercises this contract.
- Stable result snapshots use existing CPUDataTransfer, including handling of
  supported strided tensors; M02 deliberately fetches/snapshots on CPU.
- Concrete limits, ownership, supported dense-tensor inputs, reset and copy
  semantics are documented in SPEC Appendix B.
- CUDA focused suite: **29/29 passed**, including 14 native-CUDA cases and the
  CPU cases; CUDA stateless/package regressions: **26/26 passed**. No skips.
  Tests assert all compute nodes are assigned to the requested CUDA EP.
- Compute Sanitizer memcheck on the CUDA retention/reset and in-flight
  cancellation cases: **2/2 passed, zero memory errors**.
- Final clang-format, 120-column, whitespace, and document checks passed.

- [x] Collect final CPU test and 100% new-line coverage evidence.
- [x] Collect CUDA test evidence and stateless/package regression results.
- [x] Complete style/document checks, commit, and publish the stacked M02 PR.

Reproducible final validation commands (reuse the M01 CPU/CUDA configurations;
the initial M02 regeneration changed the target to `onnxruntime_session_loop_test`):

```bash
LLVM_PROFILE_FILE=/dev/null cmake --build build/state-cpu/RelWithDebInfo \
  --parallel 8 --target onnxruntime_session_loop_test onnxruntime_autoep_test
cd build/state-cpu/RelWithDebInfo
mkdir -p coverage/m02/final
LLVM_PROFILE_FILE="$PWD/coverage/m02/final/%m-%p.profraw" \
  ./onnxruntime_session_loop_test \
  --gtest_output=xml:coverage/m02/final/tests.xml
LLVM_PROFILE_FILE=/dev/null ./onnxruntime_autoep_test \
  --gtest_filter='*StatefulExecutionBaselineTest.*:ModelPackageApiTest.*' \
  --gtest_output=xml:coverage/m02/regressions.xml
llvm-profdata merge -sparse coverage/m02/final/*.profraw \
  -o coverage/m02/final/m02.profdata
llvm-cov report ./onnxruntime_session_loop_test \
  -instr-profile=coverage/m02/final/m02.profdata \
  ../../../onnxruntime/core/session/session_loop.cc \
  ../../../onnxruntime/core/session/session_loop.h
```

From the repository root, with the same CUDA 13.1 environment as M01:

```bash
cmake --build build/state-cuda-baseline-13.1/RelWithDebInfo --parallel 8 \
  --target onnxruntime_session_loop_test onnxruntime_autoep_test
cd build/state-cuda-baseline-13.1/RelWithDebInfo
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH="$PWD:$CUDNN_HOME/lib:$CUDA_HOME/lib64"
./onnxruntime_session_loop_test --gtest_output=xml:m02-tests.xml
./onnxruntime_autoep_test \
  --gtest_filter='*StatefulExecutionBaselineTest.*:ModelPackageApiTest.*' \
  --gtest_output=xml:m02-regressions.xml
"$CUDA_HOME/bin/compute-sanitizer" --tool memcheck --error-exitcode 99 \
  ./onnxruntime_session_loop_test \
  --gtest_filter='Cuda/SessionLoopTest.RetentionResetCompletionAndRunParity/*:Cuda/SessionLoopTest.CancellationBeforeAndDuringRun/*'
```

All commands above passed. This milestone adds no CUDA device production code;
the 100% new-production-line gate is the measured host implementation/header,
not a claim of device-line coverage for unchanged kernels.

### M03 - SessionChain and tensor/state bindings

**Outcome:** Internal linear composition without a manifest state machine.
**Status:** Complete; [PR #13](https://github.com/onnxruntime/ort-state/pull/13)
is ready for review. Branch `state/m03-chain-bindings` stacks on
`state/m02-context-loop`. Implementation commit: `c4ad4d5389`.

- [x] Implement ordered Session/SessionLoop composition and external/result ports.
- [x] Implement renamed tensor mapping, iteration feedback, retained-state slots,
  output selection, and staged handoff.
- [x] Validate static/dynamic binding constraints and propagate component errors.
  Expose loop progress through the agreed chain interface without hidden resets.

**CPU test plan:** Session->Session, Session->Loop, Loop->Session, and Loop->Loop;
renamed/multiple outputs; feedback; retained outputs after later steps;
missing/duplicate destinations; invalid ordering/references; dtype/shape mismatch;
failure prevents downstream execution; chain reset and Step/Run equivalence.
**CUDA test plan:** Representative CUDA chain with device-value lifetime checks;
test declared transfers and explicit rejection of unsupported combinations.
**Completion gate:** Shared rules and 100% new chain/binding-line coverage pass.

**Execution evidence:**

- Added immutable chain definitions and independent execution contexts. Plain
  Sessions reuse a one-invocation SessionLoop; configured loops retain their
  iteration/exit policies. No public API changes.
- Explicit external tensor/finite sequence, previous component output, and state
  namespaces; typed/known-shape validation and dynamic handoff validation.
- All component outputs remain available for downstream binding and observation,
  including outputs not selected by the original loop definition.
- State is committed per successful invocation, with Step/Run parity, isolated
  contexts, reset, stable snapshots, cancellation and downstream failure stopping.
- M02 tensor validation was extracted into a shared internal header rather than
  duplicated. Existing loop behavior is covered in the same focused suite.
- CPU focused suite: **30/30 passed**, including 15 chain and 15 loop/type cases;
  stateless/package regressions: **21/21 passed**, no skipped tests.
- LLVM reports **100% affected runtime line coverage**: 621/621 aggregate line
  entries across chain, shared helpers and retained loop code. LCOV's unique
  physical executable-source-line accounting is **354/354 incremental new
  locations** and **615/615 cumulative locations**; all are covered. The different
  LLVM aggregate/LCOV denominators reflect coverage mapping, not missed lines.
- Eighteen profile-map warnings concern other functions. No chain, loop, or
  shared-helper function is mismatched.
- Initial strict builds rejected a partially initialized exit-condition
  aggregate. Explicitly initializing all fields fixed that diagnostic.
- Final CPU rebuild/testing passed after the test-source style fixes.
- CUDA build: **58/58 focused tests passed**, including 14 native-CUDA chain
  cases and 14 native-CUDA loop cases. Stateless/package regressions:
  **26/26 passed**. No skipped tests. CUDA node assignment is asserted.
- Compute Sanitizer memcheck: **4/4 passed, zero errors**, covering chain state/
  reset/isolation, cancellation, mixed CPU/CUDA staging and dynamic cache tensors.
- Final source formatting, 120-column, document and whitespace checks passed.

- [x] Finalize CPU/CUDA evidence and style/document checks.
- [x] Commit and publish the ready-for-review PR stacked on M02.

Validation commands, using the established M01/M02 configurations:

```bash
source .venv/bin/activate
LLVM_PROFILE_FILE=/dev/null cmake -S cmake -B build/state-cpu/RelWithDebInfo
LLVM_PROFILE_FILE=/dev/null cmake --build build/state-cpu/RelWithDebInfo \
  --parallel 8 --target onnxruntime_session_loop_test onnxruntime_autoep_test
cd build/state-cpu/RelWithDebInfo
mkdir -p coverage/m03/final
LLVM_PROFILE_FILE="$PWD/coverage/m03/final/%m-%p.profraw" \
  ./onnxruntime_session_loop_test --gtest_output=xml:coverage/m03/final/tests.xml
LLVM_PROFILE_FILE=/dev/null ./onnxruntime_autoep_test \
  --gtest_filter='*StatefulExecutionBaselineTest.*:ModelPackageApiTest.*' \
  --gtest_output=xml:coverage/m03/regressions.xml
llvm-profdata merge -sparse coverage/m03/final/*.profraw \
  -o coverage/m03/final/m03.profdata
llvm-cov report ./onnxruntime_session_loop_test \
  -instr-profile=coverage/m03/final/m03.profdata \
  ../../../onnxruntime/core/session/session_chain.cc \
  ../../../onnxruntime/core/session/session_chain.h \
  ../../../onnxruntime/core/framework/stateful_execution_utils.h \
  ../../../onnxruntime/core/session/session_loop.cc \
  ../../../onnxruntime/core/session/session_loop.h
```

From the repository root, with the same matching CUDA 13.1/cuDNN environment:

```bash
PATH="$CUDA_HOME/bin:$PATH" cmake -S cmake \
  -B build/state-cuda-baseline-13.1/RelWithDebInfo
cmake --build build/state-cuda-baseline-13.1/RelWithDebInfo --parallel 8 \
  --target onnxruntime_session_loop_test onnxruntime_autoep_test
cd build/state-cuda-baseline-13.1/RelWithDebInfo
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH="$PWD:$CUDNN_HOME/lib:$CUDA_HOME/lib64"
./onnxruntime_session_loop_test --gtest_output=xml:m03-tests.xml
./onnxruntime_autoep_test \
  --gtest_filter='*StatefulExecutionBaselineTest.*:ModelPackageApiTest.*' \
  --gtest_output=xml:m03-regressions.xml
"$CUDA_HOME/bin/compute-sanitizer" --tool memcheck --error-exitcode 99 \
  ./onnxruntime_session_loop_test \
  --gtest_filter='Cuda/SessionChainTest.RetainedStateStepRunIsolationAndReset/*:Cuda/SessionChainTest.CancellationBeforeAndDuringComponent/*:Cuda/SessionChainTest.CpuCudaHostStagingAndRetainedTensorLifetimes/*:Cuda/SessionChainTest.DynamicCacheTensorsAndLogitsPassBetweenStages/*'
```

All final commands passed. New device production code: none; host instrumentation
is the applicable new-line coverage gate. No claim of device-native cache
management, real-model qualification, or branch-complete coverage is made.

### M04 - CPU KVCacheManager

**Outcome:** Safe contiguous caches shared across stages of one execution.
**Status:** Complete; [PR #14](https://github.com/onnxruntime/ort-state/pull/14)
is ready for review. Branch `state/m04-cpu-kv-cache` stacks on
`state/m03-chain-bindings`. Implementation commit: `fffa292875`.

- [x] Implement CPU cache initialization, model-declared layout/type/shape
  validation, logical length/capacity, append/update, retention, and reset.
- [x] Integrate prefill->decode handoff with context/binding ownership.
- [x] Share compatible allocations and explicitly copy/grow when required;
  preserve model/kernel aliasing rules and checked size arithmetic.

**CPU test plan:** Empty/initialized caches, growth boundaries, multiple layers,
sequential append, renamed handoff, permitted buffer sharing versus copy,
output/input non-alias contracts, reset without stale data, released owners,
invalid layout/length/dtype/capacity, size overflow, and allocation failure.
**CUDA test plan:** No native implementation yet. Check device-input rejection
and that the CPU cache contract does not masquerade as CUDA cache support.
**Completion gate:** CPU chain fixture matches reference; sharing/copy behavior
is observable; shared rules and 100% new cache-line coverage pass.

**Execution evidence:**

- CPU manager and explicit loop/chain managed-cache bindings implemented.
  Layouts/dtypes/sequence axes, compact logical views, capacity, full/delta
  updates, reserve, retention and reset are documented in SPEC Appendix D.
- Shared owned outputs are retained without cache-manager copies. Copy/append/
  reserve use fresh storage to preserve snapshots and never force model aliasing.
- Independent contexts own different managers; chain component transitions retain
  one shared manager. Plain Session wrappers and generic named-state handoff
  interoperate with managed caches.
- Final CPU validation: **42/42 focused tests passed**, including six standalone
  CPU cache cases and six new provider-parameterized integration cases.
  Stateless/package regressions: **21/21 passed**, no skips.
- Final new-line coverage: **335/335 unique new executable source locations**.
  Cumulative affected stateful runtime coverage is **943/943 unique locations**;
  LLVM aggregate line mapping reports **957/957**, all 100%. Eighteen unrelated
  profile-map warnings do not affect cache/loop/chain/shared-helper functions.
- Initial compile issues were resolved by guarding the training-only strided
  tensor API, using checked integer offsets before pointer arithmetic, explicit
  one-element inline capacities for larger option structures, and a typed test
  initializer. No unrelated Session or provider behavior was changed.
- Final lifetime tests release the producing Sessions and exercise leading,
  middle and last sequence axes.
- CUDA build validation: **76/76 focused tests passed**, including six new
  CPU-cache/native-CUDA-Session integration cases; existing stateless/package
  regressions: **26/26 passed**. No skips and CUDA node assignment is asserted.
- Compute Sanitizer memcheck: **5/5 passed, zero errors** on cache retention/reset,
  shared-context handoff, plain-state interoperability, copy/delta policies and
  invalid-output recovery.
- Source formatting, 120-column, document and whitespace checks passed.
- Native CUDA management and training-build qualification are not claimed.
  The strided rejection check is compile-guarded for training builds; the
  executable-line gate is measured for the Linux inference configuration.

- [x] Finalize CPU/CUDA tests, coverage counts and style/document checks.
- [x] Commit and publish the ready-for-review PR stacked on M03.

Final build/test commands, reusing the established inference configurations:

```bash
source .venv/bin/activate
LLVM_PROFILE_FILE=/dev/null cmake -S cmake -B build/state-cpu/RelWithDebInfo
LLVM_PROFILE_FILE=/dev/null cmake --build build/state-cpu/RelWithDebInfo \
  --parallel 8 --target onnxruntime_session_loop_test onnxruntime_autoep_test
cd build/state-cpu/RelWithDebInfo
mkdir -p coverage/m04/final
LLVM_PROFILE_FILE="$PWD/coverage/m04/final/%m-%p.profraw" \
  ./onnxruntime_session_loop_test --gtest_output=xml:coverage/m04/final/tests.xml
LLVM_PROFILE_FILE=/dev/null ./onnxruntime_autoep_test \
  --gtest_filter='*StatefulExecutionBaselineTest.*:ModelPackageApiTest.*' \
  --gtest_output=xml:coverage/m04/regressions.xml
llvm-profdata merge -sparse coverage/m04/final/*.profraw \
  -o coverage/m04/final/m04.profdata
llvm-cov report ./onnxruntime_session_loop_test \
  -instr-profile=coverage/m04/final/m04.profdata \
  ../../../onnxruntime/core/session/kv_cache_manager.cc \
  ../../../onnxruntime/core/session/kv_cache_manager.h \
  ../../../onnxruntime/core/session/session_loop.cc \
  ../../../onnxruntime/core/session/session_loop.h \
  ../../../onnxruntime/core/session/session_chain.cc \
  ../../../onnxruntime/core/session/session_chain.h \
  ../../../onnxruntime/core/framework/stateful_execution_utils.h
```

From the repository root with the matching CUDA 13.1/cuDNN environment:

```bash
PATH="$CUDA_HOME/bin:$PATH" cmake -S cmake \
  -B build/state-cuda-baseline-13.1/RelWithDebInfo
cmake --build build/state-cuda-baseline-13.1/RelWithDebInfo --parallel 8 \
  --target onnxruntime_session_loop_test onnxruntime_autoep_test
cd build/state-cuda-baseline-13.1/RelWithDebInfo
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH="$PWD:$CUDNN_HOME/lib:$CUDA_HOME/lib64"
./onnxruntime_session_loop_test --gtest_output=xml:m04-tests.xml
./onnxruntime_autoep_test \
  --gtest_filter='*StatefulExecutionBaselineTest.*:ModelPackageApiTest.*' \
  --gtest_output=xml:m04-regressions.xml
"$CUDA_HOME/bin/compute-sanitizer" --tool memcheck --error-exitcode 99 \
  ./onnxruntime_session_loop_test \
  --gtest_filter='Cuda/SessionChainTest.ManagedCpuCacheInLoopRetainsAndResets/*:Cuda/SessionChainTest.ManagedCacheSharesAcrossPrefillDecodeAndIndependentContexts/*:Cuda/SessionChainTest.OrdinaryStateBindingsCanReadAndCommitManagedCache/*:Cuda/SessionChainTest.ManagedCacheCopyAndDeltaPolicies/*:Cuda/SessionChainTest.InvalidManagedOutputRequiresResetWithoutCommittingCache/*'
```

All final commands passed. No device production code or native CUDA cache is
introduced; the measured new-line gate is for Linux CPU inference code.

### M05 - CPU greedy sampling and decode integration

**Outcome:** CPU token selection and a complete tiny prefill/decode chain.
**Status:** Complete; [PR #15](https://github.com/onnxruntime/ort-state/pull/15)
is ready for review. Branch `state/m05-cpu-greedy` stacks on
`state/m04-cpu-kv-cache`. Implementation commit: `ddb8260691`.

- [x] Implement documented greedy selection using existing suitable helpers.
- [x] Validate supported logits shapes/dtypes/position, ties, non-finite behavior,
  token IDs, and integration with explicit EOS/token-limit conditions.
- [x] Wire prefill-logit first-token handling and iterative decode without
  duplicates or omissions. Keep caller accumulation explicit.

**CPU test plan:** Known argmaxes, ties, negative values, supported precisions,
invalid/empty logits, non-finite policy, vocabulary boundaries, first-token EOS,
token budget across stage handoff, and cache/logit consistency after reset.
Compare the tiny chain token-by-token against direct Session calls.
**CUDA test plan:** No device-native sampler yet. Verify explicit behavior when
device-resident logits reach the CPU-only implementation; no hidden fallback.
**Completion gate:** CPU end-to-end fixture, shared rules, and 100% new sampler/
integration-line coverage pass.

**Execution evidence:**

- User confirmed the concrete M05 contracts before implementation: identical
  explicit phase budgets, single unpadded int64 prompt, generated mask/position
  feeds, strict any-NaN rejection, CPU/device distinction and context-local
  prefill-before-decode ordering. Documented in SPEC Appendix E.
- Added independently testable CPU greedy sampling and native generation policy.
  Standard argmax tie semantics are reused; no contrib kernel/subgraph dependency.
- Shared context counts generated IDs exactly once across phases, keeps prompt
  tokens separate, checks pending EOS/budget at decode entry, and validates KV
  lengths against consumed inputs. Typed final token-ID ports support empty
  results without inventing unproduced model outputs.
- Generated CPU feeds are retained through device synchronization/Run.
- CPU focused suite: **55/55 passed**, including independent sampler/metadata/
  allocation checks and full prefill/decode fixtures; stateless/package
  regressions: **21/21 passed**, no skips.
- Final new-line coverage: **369/369 unique new executable locations**; cumulative
  affected runtime **1295/1295**, LLVM aggregate **1310/1310**, all 100%.
  Eighteen profile-map warnings concern unrelated functions, not affected code.
- Tests compare tokens/cache/mask/positions against direct Session invocations
  and cover float16, ties/infinities/NaNs, invalid vocab/shape/position/device,
  zero/one budgets, prefill EOS, reset/isolation and failure recovery.
- Initial compile scope error and empty-prompt validation placement were fixed;
  the empty-prompt regression now rejects even a zero-budget prefill.
- A shared explicit state-reset helper avoids a GCC optimized-build
  maybe-uninitialized diagnostic without suppressing warnings. Unspecified
  generation phases reject explicitly rather than reading an unset enum.
- CUDA configuration: **97/97 focused tests passed**, including eight new
  provider-parameterized generation cases; stateless/package regressions:
  **26/26 passed**, no skips. Actual CUDA model-node assignment is asserted.
- Compute Sanitizer memcheck: **5/5 passed, zero errors**, covering reference
  comparison, zero/one budgets, prefill/decode EOS, float16 and failure recovery.
- Final formatting, 120-column, whitespace and document checks passed. No native
  CUDA sampler or training-build coverage is claimed.

- [x] Complete CUDA tests, memory checks and final source/document validation.
- [x] Commit and publish the ready-for-review PR stacked on M04.
- [x] Move token generation to `core/generate`, keep shared validation in
  `core/framework`, and revalidate CPU/CUDA builds, behavior and coverage.
  Requested 2026-10-16; behavior/library targets remain unchanged.

Layout follow-up evidence: source/header bodies are unchanged except include
paths; no old paths or generation-to-session includes remain. CPU **55/55**
focused tests and **21/21** regressions passed, no skips. New source mappings
preserve **1295/1295** unique runtime locations (**1310/1310** LLVM aggregate).
CUDA **97/97** focused tests and **26/26** regressions passed, no skips;
all five memory-check scenarios passed with zero errors. M05's original
**369/369** new-line gate is unchanged. Layout commit `417832d919` is published;
the existing PR's description and source/validation paths are refreshed.

Validation commands (reuse the established instrumented CPU and CUDA 13.1 builds;
regenerate CMake when first adding the native generation source files):

```bash
source .venv/bin/activate
LLVM_PROFILE_FILE=/dev/null cmake -S cmake -B build/state-cpu/RelWithDebInfo
LLVM_PROFILE_FILE=/dev/null cmake --build build/state-cpu/RelWithDebInfo \
  --parallel 8 --target onnxruntime_session_loop_test onnxruntime_autoep_test
cd build/state-cpu/RelWithDebInfo
mkdir -p coverage/m05/layout
LLVM_PROFILE_FILE="$PWD/coverage/m05/layout/%m-%p.profraw" \
  ./onnxruntime_session_loop_test --gtest_output=xml:coverage/m05/layout/tests.xml
LLVM_PROFILE_FILE=/dev/null ./onnxruntime_autoep_test \
  --gtest_filter='*StatefulExecutionBaselineTest.*:ModelPackageApiTest.*' \
  --gtest_output=xml:coverage/m05/layout/regressions.xml
llvm-profdata merge -sparse coverage/m05/layout/*.profraw \
  -o coverage/m05/layout/m05.profdata
llvm-cov report ./onnxruntime_session_loop_test \
  -instr-profile=coverage/m05/layout/m05.profdata \
  ../../../onnxruntime/core/generate/token_generation.cc \
  ../../../onnxruntime/core/generate/token_generation.h \
  ../../../onnxruntime/core/session/session_loop.cc \
  ../../../onnxruntime/core/session/session_loop.h \
  ../../../onnxruntime/core/session/session_chain.cc \
  ../../../onnxruntime/core/session/session_chain.h \
  ../../../onnxruntime/core/session/kv_cache_manager.cc \
  ../../../onnxruntime/core/session/kv_cache_manager.h \
  ../../../onnxruntime/core/framework/stateful_execution_utils.h
```

From the repository root with the matching CUDA/cuDNN environment:

```bash
PATH="$CUDA_HOME/bin:$PATH" cmake -S cmake \
  -B build/state-cuda-baseline-13.1/RelWithDebInfo
cmake --build build/state-cuda-baseline-13.1/RelWithDebInfo --parallel 8 \
  --target onnxruntime_session_loop_test onnxruntime_autoep_test
cd build/state-cuda-baseline-13.1/RelWithDebInfo
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH="$PWD:$CUDNN_HOME/lib:$CUDA_HOME/lib64"
./onnxruntime_session_loop_test --gtest_output=xml:m05-layout-tests.xml
./onnxruntime_autoep_test \
  --gtest_filter='*StatefulExecutionBaselineTest.*:ModelPackageApiTest.*' \
  --gtest_output=xml:m05-layout-regressions.xml
"$CUDA_HOME/bin/compute-sanitizer" --tool memcheck --error-exitcode 99 \
  ./onnxruntime_session_loop_test \
  --gtest_filter='Cuda/SessionChainTest.GreedyGenerationMatchesDirectSessionsAndFeedPolicies/*:Cuda/SessionChainTest.ZeroAndSingleTokenBudgetsAvoidExtraDecodeInvocations/*:Cuda/SessionChainTest.PrefillAndDecodeEosUseOneSharedCounterAndEmitOnce/*:Cuda/SessionChainTest.StandalonePrefillFloat16AndExplicitLogitsPosition/*:Cuda/SessionChainTest.GenerationFailuresRequireResetAndKeepCommittedState/*'
```

All final commands passed. These tests use native CUDA model execution with
explicit CPU logits sampling, not a native CUDA greedy implementation.

### M06 - Public C API and C++ wrappers

**Outcome:** Complete programmatic access to the implemented core primitives.
**Status:** Appendix F and the M06a/M06b split approved 2026-10-16.
M06a is complete: [PR #16](https://github.com/onnxruntime/ort-state/pull/16),
`state/m06a-stateful-c-api`, based on M05 `17e432b025`.
M06b is complete: [PR #17](https://github.com/onnxruntime/ort-state/pull/17),
`state/m06b-stateful-cxx-api`, based on M06a `b5f1431837`.
Both sub-PRs are verified and published ready for review; M06 is complete.

- [x] Add append-only public API entries and opaque resources consistent with
  existing tables. Do not bump `ORT_API_VERSION`.
- [x] Expose configuration, bindings, exit conditions, contexts, loop/chain
  execution, cache/sampler integration, outputs, reset, cancellation, and release.
- [x] Add RAII C++ wrappers and Doxygen ownership/lifetime/`\since` documentation.
  Ensure failed calls preserve out-parameters and contain exceptions.

**CPU test plan:** Reproduce M02-M05 behavior through C and C++, not only internals;
null/invalid handles, ownership/release ordering, output lifetime, errors,
resource cleanup, and compatibility with ordinary Session/API use.
**CUDA test plan:** Public-API generic CUDA SessionLoop/Chain smoke plus explicit
unsupported native-feature reporting until the corresponding milestone.
**Completion gate:** C/C++ consumers compile; ABI table checks and shared rules
pass; 100% new API/wrapper-line coverage, including inline code, is measured.

**Approved reviewable sub-PRs:**

1. **M06a - C ABI and typed configuration.** Add the append-only accessor/table,
   typed configuration/options, loop/chain execution, result/state inspection,
   standalone cache and sampler APIs, release functions, Doxygen and C tests.
   CPU tests cover M02-M05 via C, invalid arguments, exceptions, lifetimes,
   out-parameters and ABI layout. CUDA tests cover actual Session execution and
   explicit unsupported device-native cache/sampler behavior. Compile a pure-C
   consumer and measure 100% new executable C-API/native-glue lines.
2. **M06b - C++ RAII and lifetime-safe views.** Stack on M06a; add conventional
   C++ wrappers, owning immutable-result/view lifetimes, examples and tests.
   CPU/CUDA behavior must match C; test move/release ordering, error mapping and
   retained values after result/execution release. Measure 100% new C++ wrapper
   lines in addition to cumulative C-API coverage.

M06 is complete only after both sub-PRs satisfy their documented gates.
SPEC Appendix F contains the approved concrete public-interface contracts.
The user approved the split and contracts without changes before implementation.

**M06a validation checklist:**

- [x] Initial instrumented CPU build and nine public C-API tests passed.
- [x] Validate the final CPU build, existing regressions and pure-C consumer.
- [x] Measure 100% new executable API/native-glue lines.
- [x] Validate CUDA execution/provider assignment and memory safety.
- [x] Record final evidence, complete style checks and publish the stacked PR.
  Ready-for-review PR #16 is stacked on M05; implementation commit `f0aa2513d7`.

**M06b validation checklist:**

- [x] Add move-only mutable C++ resources and complete public API wrappers.
- [x] Retain result/cache owners in copyable const values and copy metadata.
- [x] Test move/release ordering, errors and full CPU/CUDA behavior parity.
- [x] Measure 100% new wrapper lines and cumulative C/native coverage.
- [x] Validate standalone C++ consumption, CUDA memcheck, examples and style.
- [x] Record final evidence and publish the ready-for-review PR stacked on M06a.
  PR #17 is stacked on PR #16; implementation commit `cd63f2f8ba`.

Initial focused C++ tests passed **5/5 on CPU** and **8/8 in the CUDA build**,
zero skips. Initial measured new-wrapper coverage is **187/187 executable
locations**; the final combined validation below also passed.

**M06b execution evidence:**

- Wrapped every M06a operation/resource in the header-only public C++ SDK.
  Mutable resources use the existing move-only Base; immutable results/cache
  views share C-handle ownership and do not expose ownership-releasing release().
- Added parent-retaining const value storage, distinct from unowned ConstValue.
  Copies retain the parent and moves empty the source; static assertions prevent
  implicit slicing/mutable pointer conversion. A weak-owner test proves actual
  parent lifetime, not merely unchanged data at a potentially freed address.
- Metadata copies names/reasons and uses optional emitted tokens. Result owners
  are allocated before Step/Run. Empty resources release safely when the API
  table is absent; operations still throw explicitly. C ABI/version unchanged.
- Final CPU build: **36/36 API/baseline tests** (10 C + 5 C++ + 21 regressions)
  and **55/55 native tests**, zero skips.
- Final CUDA 13.1/A100 build: **49/49 API/baseline tests** (15 C + 8 C++ + 26
  regressions) and **97/97 native tests**, zero skips. Model-node CUDA assignment
  and disabled CPU fallback are asserted in every CUDA parameterization.
- Coverage: **187/187 new unique executable wrapper locations** (37 SDK
  storage/helper-header lines and 150 inline implementation lines), plus **1857/1857**
  cumulative native/C-API locations: **2044/2044**, all 100%.
  Fresh profiles match final artifacts. SDK export uses the API executable;
  runtime export adds the shared library/native executable. Inspected dumps have
  42/215 zero-hash unused-map entries respectively, none affecting measured new
  wrappers or changed runtime glue. This is line coverage, not branch coverage.
- All four Session/Loop chain combinations, finite feeds/sequences/state,
  explicit exits, zero/one/multiple generation budgets, float16/float32 greedy,
  metadata/const-view lifetimes, move/copy/release ordering, native status-to-
  exception mapping, atomic options and unsupported device rejection pass.
- CUDA Compute Sanitizer: **8/8 C/C++ API cases passed, zero errors**.
- SPEC Appendix F.6's exact standalone example compiled/linked/ran with GCC
  C++17 `-Wall -Wextra -Werror`, both normally and with `ORT_NO_EXCEPTIONS`/
  `-fno-exceptions` client flags. This does not claim a no-exceptions runtime,
  minimal/training qualification, native CUDA cache/sampling or real-model
  attention coverage. Those existing scope boundaries remain unchanged.
- Changed-range formatting, 120 columns, whitespace, all 93 requirement IDs and
  all 24 macro milestones were checked. The attached specification snapshot was
  not modified; only repository documents were updated.

Passing final CPU commands:

```bash
source .venv/bin/activate
LLVM_PROFILE_FILE=/dev/null cmake --build build/state-cpu/RelWithDebInfo \
  --parallel 8 --target onnxruntime_autoep_test onnxruntime_session_loop_test
cd build/state-cpu/RelWithDebInfo
mkdir -p coverage/m06b/complete
LLVM_PROFILE_FILE="$PWD/coverage/m06b/complete/%m-%p.profraw" \
  ./onnxruntime_autoep_test \
  --gtest_filter='*StatefulApi*:*StatefulCxx*:*StatefulExecutionBaselineTest.*:ModelPackageApiTest.*' \
  --gtest_output=xml:coverage/m06b/complete/api-tests.xml
LLVM_PROFILE_FILE="$PWD/coverage/m06b/complete/%m-%p.profraw" \
  ./onnxruntime_session_loop_test \
  --gtest_output=xml:coverage/m06b/complete/native-tests.xml
llvm-profdata merge -sparse coverage/m06b/complete/*.profraw \
  -o coverage/m06b/complete/m06b.profdata
llvm-cov export ./onnxruntime_autoep_test \
  -instr-profile=coverage/m06b/complete/m06b.profdata -format=lcov \
  ../../../include/onnxruntime/core/session/onnxruntime_cxx_api.h \
  ../../../include/onnxruntime/core/session/onnxruntime_cxx_inline.h \
  > coverage/m06b/complete/wrappers.lcov
llvm-cov export ./onnxruntime_autoep_test -object ./libonnxruntime.so \
  -object ./onnxruntime_session_loop_test \
  -instr-profile=coverage/m06b/complete/m06b.profdata -format=lcov \
  ../../../onnxruntime/core/session/stateful_api.cc \
  ../../../onnxruntime/core/session/{session_loop,session_chain,kv_cache_manager}.{cc,h} \
  ../../../onnxruntime/core/generate/token_generation.{cc,h} \
  ../../../onnxruntime/core/framework/stateful_execution_utils.h \
  > coverage/m06b/complete/runtime.lcov
```

Intersect deduplicated SDK DA locations with added lines against M06a
`b5f1431837`; combine with the measured runtime DA locations for the exact
187/2044 denominators. Inspect mismatch dumps rather than suppressing warnings.

Passing CUDA commands (from repository root; matching CUDA/cuDNN paths as M06a):

```bash
source .venv/bin/activate
export CUDA_HOME=/datadisks/disk4/bmeswani/cuda/cuda13.1
export CUDNN_HOME="$CUDA_HOME/targets/x86_64-linux"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDNN_HOME/lib:${LD_LIBRARY_PATH:-}"
cmake --build build/state-cuda-baseline-13.1/RelWithDebInfo \
  --parallel 8 --target onnxruntime_autoep_test onnxruntime_session_loop_test
cd build/state-cuda-baseline-13.1/RelWithDebInfo
mkdir -p coverage/m06b/complete
./onnxruntime_autoep_test \
  --gtest_filter='*StatefulApi*:*StatefulCxx*:*StatefulExecutionBaselineTest.*:ModelPackageApiTest.*' \
  --gtest_output=xml:coverage/m06b/complete/api-tests.xml
./onnxruntime_session_loop_test \
  --gtest_output=xml:coverage/m06b/complete/native-tests.xml
"$CUDA_HOME/bin/compute-sanitizer" --tool memcheck --error-exitcode 99 \
  ./onnxruntime_autoep_test \
  --gtest_filter='Cuda/StatefulApiProviderTest.*:Cuda/StatefulCxxApiProviderTest.*'
```

Passing standalone example command (repository root):

```bash
python - <<'PY' | g++ -std=c++17 -Wall -Wextra -Werror \
  -I include/onnxruntime/core/session -x c++ - \
  -L build/state-cpu/RelWithDebInfo \
  -Wl,-rpath,/datadisks/disk9/devpatel/ngenai/onnxruntime/build/state-cpu/RelWithDebInfo \
  -lonnxruntime -o build/state-cpu/RelWithDebInfo/coverage/m06b/sdk-example
from pathlib import Path
text = Path("SPEC.md").read_text().split("### F.6 C++ ownership and executable example", 1)[1]
print(text.split("```cpp", 1)[1].split("```", 1)[0])
PY
LLVM_PROFILE_FILE=/dev/null build/state-cpu/RelWithDebInfo/coverage/m06b/sdk-example
```

The same compile/link/run also passed with `-DORT_NO_EXCEPTIONS -fno-exceptions`
and output name `sdk-example-noexceptions`. No package installations were needed.

**M06a execution evidence:**

- Appended one runtime-owned `GetStatefulApi` accessor and a dedicated 38-entry
  table; existing entries/signatures/version boundaries are unchanged.
  `ORT_API_VERSION` remains 32; public members document Version 1.32.
- Added opaque owning resources, copied typed configurations, transactional
  SetConfig, immutable definitions, retained execution inputs/results/cache
  views, state inspection, cancellation/reset, cache operations and CPU greedy.
- CPU shared-library build passed: **31/31 API/baseline tests** (10 C-API cases
  plus 21 stateless/package regressions) and **55/55 native cases**, zero skips.
- CUDA 13.1/A100 build passed: **41/41 API/baseline tests** (15 C-API cases,
  including five CUDA parameterizations, plus 26 regressions) and **97/97 native
  cases**, zero skips. CUDA fixture Sessions assert CUDA node assignment and
  disable CPU fallback.
- A checked-in `.c` consumer compiles as C in both normal test targets and
  exercises sampling, cache ownership and retained-view data after manager and
  original tensor release. Independent GCC C11 compilation/link/run also passed
  with `-Wall -Wextra -Werror` against the current shared library.
- New production/glue coverage: **562/562 unique executable lines**, comprising
  546 API lines and 16 native state-inspection lines. Cumulative affected
  production/header coverage: **1857/1857 unique executable locations**, 100%.
  Fresh profiles combine the API/shared-library and native test runs. Detailed
  diagnostics identify 131 zero-hash unused-map entries; none names a new API
  function or the changed state-inspection helper. This is executable-line
  coverage, not 100% branch coverage or qualification of unbuilt configurations.
- Tests cover copied configuration/input-handle lifetimes, all four Session/
  Loop chain combinations, finite sequences/state, explicit exits/native
  generation, zero budgets, stable results/views across reset/release, NULL/
  size/count/enum errors, atomic replacement, exception containment, preserved
  out-parameters and append-only accessor layout.
- Compute Sanitizer memcheck: **5/5 CUDA C-API cases passed, zero errors**.
  Caches and sampling remain CPU implementations with explicit device rejection;
  native CUDA implementations remain M08/M09. Minimal/training builds are not
  qualified here. C++ RAII wrappers remain the separately approved M06b.
- C/C++ formatting, 120-column checks and source whitespace checks passed.

Validation commands (from the repository root unless a `cd` is shown):

```bash
source .venv/bin/activate
LLVM_PROFILE_FILE=/dev/null cmake -S cmake -B build/state-cpu/RelWithDebInfo
LLVM_PROFILE_FILE=/dev/null cmake --build build/state-cpu/RelWithDebInfo \
  --parallel 8 --target onnxruntime_autoep_test onnxruntime_session_loop_test
cd build/state-cpu/RelWithDebInfo
mkdir -p coverage/m06a/complete
LLVM_PROFILE_FILE="$PWD/coverage/m06a/complete/%m-%p.profraw" \
  ./onnxruntime_autoep_test \
  --gtest_filter='*StatefulApi*:*StatefulExecutionBaselineTest.*:ModelPackageApiTest.*' \
  --gtest_output=xml:coverage/m06a/complete/api-tests.xml
LLVM_PROFILE_FILE="$PWD/coverage/m06a/complete/%m-%p.profraw" \
  ./onnxruntime_session_loop_test \
  --gtest_output=xml:coverage/m06a/complete/native-tests.xml
llvm-profdata merge -sparse coverage/m06a/complete/*.profraw \
  -o coverage/m06a/complete/m06a.profdata
llvm-cov export ./onnxruntime_autoep_test -object ./libonnxruntime.so \
  -object ./onnxruntime_session_loop_test \
  -instr-profile=coverage/m06a/complete/m06a.profdata -format=lcov \
  ../../../onnxruntime/core/session/stateful_api.cc \
  ../../../onnxruntime/core/session/onnxruntime_c_api.cc \
  ../../../onnxruntime/core/session/session_loop.cc \
  ../../../onnxruntime/core/session/session_loop.h \
  ../../../onnxruntime/core/session/session_chain.cc \
  ../../../onnxruntime/core/session/session_chain.h \
  ../../../onnxruntime/core/session/kv_cache_manager.cc \
  ../../../onnxruntime/core/session/kv_cache_manager.h \
  ../../../onnxruntime/core/generate/token_generation.cc \
  ../../../onnxruntime/core/generate/token_generation.h \
  ../../../onnxruntime/core/framework/stateful_execution_utils.h \
  > coverage/m06a/complete/cumulative.lcov
printf 'int StatefulApiCConsumerTest(void);\nint main(void) { return StatefulApiCConsumerTest(); }\n' |
  gcc -std=c11 -Wall -Wextra -Werror -I../../../include/onnxruntime -x c - \
  ../../../onnxruntime/test/autoep/stateful_api_c_test.c -L. -Wl,-rpath,"$PWD" \
  -lonnxruntime -o coverage/m06a/complete/c-smoke
LLVM_PROFILE_FILE=/dev/null ./coverage/m06a/complete/c-smoke
```

All commands above passed. Coverage export emitted the inspected warnings noted
above; deduplicated LCOV locations establish the exact new-line denominator.

```bash
source .venv/bin/activate
export CUDA_HOME=/datadisks/disk4/bmeswani/cuda/cuda13.1
export CUDNN_HOME="$CUDA_HOME/targets/x86_64-linux"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDNN_HOME/lib:${LD_LIBRARY_PATH:-}"
cmake -S cmake -B build/state-cuda-baseline-13.1/RelWithDebInfo
cmake --build build/state-cuda-baseline-13.1/RelWithDebInfo \
  --parallel 8 --target onnxruntime_autoep_test onnxruntime_session_loop_test
cd build/state-cuda-baseline-13.1/RelWithDebInfo
mkdir -p coverage/m06a/complete
./onnxruntime_autoep_test \
  --gtest_filter='*StatefulApi*:*StatefulExecutionBaselineTest.*:ModelPackageApiTest.*' \
  --gtest_output=xml:coverage/m06a/complete/api-tests.xml
./onnxruntime_session_loop_test \
  --gtest_output=xml:coverage/m06a/complete/native-tests.xml
"$CUDA_HOME/bin/compute-sanitizer" --tool memcheck --error-exitcode 99 \
  ./onnxruntime_autoep_test --gtest_filter='Cuda/StatefulApiProviderTest.*'
```

All CUDA commands above passed, with the nonzero counts/zero errors stated above.

### M07 - Built-in and plugin EP customization

**Outcome:** Providers can implement cache features and/or sampling independently.
**Status:** SPEC Appendix G and the detailed M07a/M07b plan were approved
2026-10-17 without changes. Both sub-PRs are authorized sequentially.
M07a is complete: [PR #18](https://github.com/onnxruntime/ort-state/pull/18),
`state/m07a-ep-stateful-hooks`, based on M06b `5bde969b66`.
M07b is complete: [PR #19](https://github.com/onnxruntime/ort-state/pull/19),
`state/m07b-plugin-stateful-hooks`, based on M07a `46715fb08a`.
Both independently verified sub-PRs are published ready for review; macro M07 is complete.

- [x] Add capability-based in-tree hooks and boundary-safe plugin C contracts.
- [x] Route CPU implementations through the selection contract; specify allowed
  transfers/fallback and make implementation selection observable.
- [x] Extend a test plugin to demonstrate partial/full support, lifetimes,
  unsupported features, errors, and compatibility with older plugin contracts.

**CPU test plan:** Execute real CPU/cache/sampler overrides and test-plugin calls;
no-override and subset cases; version/size/capability checks; provider failure;
release ordering and rejection of unsafe or undisclosed fallbacks.
**CUDA test plan:** Load/test the extension mechanism in a CUDA-enabled build;
verify unavailable CUDA stateful capabilities are still reported accurately.
**Completion gate:** Built-in and plugin routes both execute; public boundary
documentation and 100% new host/glue-line coverage satisfy shared rules.

#### M07a - Native contracts, CPU routing and public controls

**Intended outcome:** Real in-tree capability-based cache/sampler dispatch,
preserving CPU defaults and all M02-M06 behavior. Native and C/C++ callers can
select/query implementations and observe supported fallback without relying on
plugin registration or new CUDA kernels.
**Dependencies:** M04, M05, M06. Stack this PR on M06b PR #17.

- [x] A1: Define boundary-safe capability, selection, device/layout and provider-
  resource contracts. Support individual KV operations and greedy independently;
  validate advertised callbacks and version/size fields. Shared declarations must
  not leak graph types or C++ containers/ownership across a C/DLL boundary.
- [x] A2: Add optional IExecutionProvider discovery/hooks and wire the shared-
  provider bridge. Unchanged EPs report no overrides. Supply the CPU reference
  implementation through this contract, reusing existing algorithms.
- [x] A3: Resolve independent CPU/PreferProvider/RequireProvider selections for
  caches and sampling. Resolve a single compatible cache backend/device/storage
  contract per chain; reject conflicts and invalid/ambiguous provider requests.
  Allow only compatible missing-capability CPU fallback, never retry after an
  invoked callback fails. Preserve intentional CPU-mode Session I/O staging.
- [x] A4: Integrate dispatch into standalone caches/sampling, loops, generation
  and shared-chain caches. Bind provider-backed standalone use explicitly to an
  initialized Session; retain required callback/resources. Keep token generation
  in core/generate and CPU behavior Session-free where already supported.
- [x] A5: Propagate fallible provider/reset operations through native and public
  error paths. Keep invalidation sticky after possibly mutating failures and
  failed reset. Preserve immutable views and failure out-parameters; add any
  required standalone-cache health query rather than silently reusing bad state.
- [x] A6: Append public policy/binding/capability/dispatch-observation operations
  and C++ wrappers. Preserve M06 V1 minimum sizes and fixed nested array-record
  strides; do not reorder entries, change shipped signatures, bump API version
  or add release-boundary markers. Metadata returned by C++ must retain/copy its
  owners as in M06b.
- [x] A7: Add real synthetic in-tree providers and conformance tests, execute
  all relevant CPU/CUDA regressions, measure every new host/bridge line, document
  API/lifetimes/examples, and publish a ready-for-review PR.

**CPU test plan:**

- Verify unchanged default CPU outputs and real CPU reference dispatch for
  Session-free standalone objects and model-loop/chain execution.
- Exercise independent sampler-only/cache-only and partial-operation overrides;
  compare results with the reference CPU implementation and assert actual
  callback counts/selected operation, not only a registered provider name.
- Cover absent/unknown providers, ambiguous/incompatible devices, missing
  capabilities in PreferProvider/RequireProvider, permitted CPU-storage fallback,
  rejected device fallback and conflicting shared-cache stage requirements.
- Inject callback errors, including NOT_IMPLEMENTED after invocation, reset
  failures and malformed results. Assert no CPU retry, preserved out-parameters,
  invalidation/recovery and stable pre-existing snapshots.
- Reproduce generic loops, all four chain combinations, finite inputs/state,
  prefill/decode, EOS/limits and zero/one token budgets through native and C/C++.
  Zero-budget execution must not introduce model/sampler invocations.
- Verify configuration snapshots, retained contexts/allocators and callback
  resources after mutable-object release. Use observable lifetime/allocator
  counters, not merely reads from potentially freed but unchanged memory.
- Test C/C++ boundary compilation and examples. Keep observability bounded:
  per-operation selection/counters/current metadata, not unbounded token history.
- Rerun M06's 36 API/baseline and 55 native CPU cases, plus the new actual cases;
  counts are baselines, not predictions or a substitute for inspecting new XML.

**CUDA test plan:**

- Build the affected native/shared-provider paths with the existing consistent
  CUDA 13.1/A100 configuration; preserve real CUDA model-node assignment.
- Verify default CPU stateful behavior remains explicit/observable with CUDA
  Sessions, and native CUDA cache/sampler requirements fail until M08/M09.
- Run applicable host-override/error/lifetime and CPU-compatible fallback cases
  in the CUDA build. Reject device-storage CPU fallback rather than copying it
  behind the caller's policy. Do not label CPU callback dispatch native CUDA.
- Rerun the 49 API/baseline and 97 native CUDA-build cases plus new tests; run
  relevant generic API cases under Compute Sanitizer with zero errors.

**M07a completion gate:** Every introduced native/public path, including shared-
provider bridge glue, has measured 100% new executable-line coverage. If a
conditional bridge path is absent from the instrumented CPU build, exercise it
with an appropriate host/shared-provider fixture or collect its matching
instrumented host profile; do not omit it from the denominator. CPU/CUDA tests,
examples, ABI/style/document checks and publication must all pass. M07 itself
remains incomplete while M07b is outstanding.

**Current validation:** Initial focused CPU provider tests passed **6/6**.
Final regressions, CUDA verification, measured coverage and publication passed.
Implementation commit `8fbd403493` is published in ready-for-review PR #18.
The native optional hook is appended after existing virtual
members, preserving old slot positions, and matching full builds were verified.

**M07a verified evidence:**

- Added per-operation, versioned C descriptors; optional native/shared-provider
  discovery; real CPU reference dispatch; independent copied policies; retained
  provider resources; and appended C/C++ configuration/discovery/diagnostics.
  No shipped C entries, version boundaries or M06 nested record strides changed.
- KV callbacks stage results and validate storage/extents before commit.
  Provider failures invalidate mutable state; NOT_IMPLEMENTED after invocation
  is propagated without CPU retry. Failed reset stays invalid. Allocator adapters
  supplied to callbacks are retained with returned tensor storage, not left on
  a dead stack frame. Native reset Status is checked through C/C++ and tests.
- Sampling dispatch remains in core/generate, with framework-level declarations
  instead of reverse native-session includes. A shared/DLL header uses only C
  handles/metadata; no graph types leak across the provider interface.
- Final CPU build/tests passed: **61/61 native** (six new provider cases plus
  55 regressions) and **36/36 public API/baseline cases**, zero skips.
- Final CUDA 13.1/A100 build/tests passed: **107/107 native** (ten provider cases
  plus 97 regressions) and **49/49 public API/baseline cases**, zero skips.
  New CUDA conformance fixtures assert actual model-node assignment and disable CPU fallback.
- Exact new-line coverage: **782/782 unique executable source locations**,
  including native/default/shared-provider headers, callback/resource/error
  glue and all new C++ wrappers. The initial 763/770 result was not rounded up:
  targeted tests closed all gaps, followed by style-final rebuilt profiles.
- SDK coverage is exported from its exercising native test executable; runtime
  coverage additionally includes the public API executable/shared library.
  Detailed dumps identify 47/193 zero-hash unused-map entries respectively,
  none affecting the measured new functions in the selected exports. Mixing
  all SDK maps would include dormant copies from non-exercising images.
- Compute Sanitizer: **4/4 native-provider CUDA cases** and **8/8 public API CUDA
  cases**, zero errors. CPU callbacks/delegations are not native GPU work.
- Extended the checked-in C consumer to exercise Session-free CPU policies,
  diagnostics and cache health. GCC C11 compile/link/run passed with warnings
  as errors. SPEC G.9's exact C++17 provider-selection example compiled/ran both
  normally and with no-exceptions client flags; no runtime-mode qualification
  is implied.
- Tests prove partial/missing/required capabilities, actual callback counts,
  per-stage sampling/shared-cache policy conflicts, reference delegation,
  context/allocator retention, reset failures and CPU staging observability.
  Staging counters are explicit CPU payloads, not inferred DMA measurements.
- Changed-range formatting, 120 columns, whitespace, 93 requirement IDs and
  24 macro milestones passed. No new dependency was installed.

Passing final commands (reuse the established build configurations):

```bash
source .venv/bin/activate
LLVM_PROFILE_FILE=/dev/null cmake --build build/state-cpu/RelWithDebInfo \
  --parallel 8 --target onnxruntime_autoep_test onnxruntime_session_loop_test
cd build/state-cpu/RelWithDebInfo
mkdir -p coverage/m07a/verified
LLVM_PROFILE_FILE="$PWD/coverage/m07a/verified/%m-%p.profraw" \
  ./onnxruntime_session_loop_test \
  --gtest_output=xml:coverage/m07a/verified/native-tests.xml
LLVM_PROFILE_FILE="$PWD/coverage/m07a/verified/%m-%p.profraw" \
  ./onnxruntime_autoep_test \
  --gtest_filter='*StatefulApi*:*StatefulCxx*:*StatefulExecutionBaselineTest.*:ModelPackageApiTest.*' \
  --gtest_output=xml:coverage/m07a/verified/api-tests.xml
llvm-profdata merge -sparse coverage/m07a/verified/*.profraw \
  -o coverage/m07a/verified/m07a.profdata
```

Export SDK headers from onnxruntime_session_loop_test and native changed sources
with onnxruntime_autoep_test/libonnxruntime.so added as objects. Deduplicate DA
locations and intersect added source lines against M06b `5bde969b66`; all 782
executed. New untracked source files are entirely new for the initial export.

From the CUDA build output directory after the matching successful build:

```bash
./onnxruntime_session_loop_test \
  --gtest_output=xml:coverage/m07a/verified/native-tests.xml
./onnxruntime_autoep_test \
  --gtest_filter='*StatefulApi*:*StatefulCxx*:*StatefulExecutionBaselineTest.*:ModelPackageApiTest.*' \
  --gtest_output=xml:coverage/m07a/verified/api-tests.xml
"$CUDA_HOME/bin/compute-sanitizer" --tool memcheck --error-exitcode 99 \
  ./onnxruntime_session_loop_test --gtest_filter='Cuda/StatefulProviderTest.*'
"$CUDA_HOME/bin/compute-sanitizer" --tool memcheck --error-exitcode 99 \
  ./onnxruntime_autoep_test \
  --gtest_filter='Cuda/StatefulApiProviderTest.*:Cuda/StatefulCxxApiProviderTest.*'
```

All commands above passed with the counts stated above. M07b remains required
before macro M07 can be marked complete.

#### M07b - Plugin registration, adapters and compatibility

**Intended outcome:** Dynamically loaded plugins can advertise and execute
independent/partial stateful capabilities safely through the public C boundary,
without breaking older OrtEp allocations or unloading live callback resources.
**Dependency:** M07a. Use a separate branch/PR stacked on the verified M07a head.

- [x] B1: Append optional OrtEpApi registration/discovery entry points for copied,
  versioned, size-tagged descriptors. Do not add mandatory fields to OrtEp or
  read beyond an old allocation. Validate registration atomically; clean up
  retained resources if registration or subsequent EP creation fails.
- [x] B2: Adapt registered callbacks to the native dispatcher. Retain callback
  contexts, factory/library, allocator and any callback-used resources for the
  needed lifetime, including values surviving Session/context release. Make
  unregistration safe while live objects still depend on callback code.
- [x] B3: Extend the real example/test plugin for no overrides, sampler-only,
  partial KV, full supported CPU capabilities and injected failures. Preserve
  numerical/retained-view contracts; assert actual callback results and counts.
- [x] B4: Add a genuine frozen legacy/shorter OrtEp fixture and version/size/ABI
  checks. A current-size instance with registration disabled is insufficient.
  Verify existing no-registration plugins continue to load/run.
- [x] B5: Exercise provider/context/deleter retention, failed creation/registration
  cleanup, exactly-once release and safe lifetime across unregistration. Cover
  reset errors/no-retry fallback with plugin callbacks, not only native mocks.
- [x] B6: Run C/C++ plugin conformance and CPU/CUDA builds/regressions, include
  plugin adapter/registration host glue in fresh coverage, document implementer
  usage and limitations, and publish a ready-for-review stacked PR.

**CPU test plan:**

- Load the real plugin libraries and execute all supported override modes.
  Compare reference outputs and query implementation/fallback evidence.
- Test minimum/undersized/extended descriptors, unknown versions/capabilities,
  missing callbacks, bad pointer/count pairs and owned callback-status handling.
- Use the actual older/shorter ABI fixture; verify no new OrtEp-field read and
  no descriptor is required from legacy plugins.
- Fail registration and EP creation after resources were acquired; prove that
  contexts/library resources are neither leaked nor released twice.
- Release options, definitions, executions and Sessions while retained values
  remain. Exercise their eventual allocator/deleter callbacks safely; verify
  live callback code/resources cannot be unloaded by unregistration.
- Check public C/C++ selection, metadata ownership and error behavior with
  plugins, plus the complete M07a and M02-M06 regressions.

**CUDA test plan:**

- Load/run the same plugin extension in the CUDA-enabled build; exercise
  relevant CUDA Session integration and preserve explicit capability reporting.
- Repeat incompatible-device/missing-required-native-capability cases and
  ensure plugin CPU overrides are not reported as native CUDA implementations.
- Run relevant plugin/API lifetime and model-integration cases under Compute
  Sanitizer. Native CUDA KV/sampling/stream kernels remain M08/M09.

**M07b completion gate:** Real dynamic-plugin dispatch, partial support, genuine
legacy ABI compatibility, error propagation and retained-resource lifetime are
proved. All new registration/adapter/host glue has measured 100% executable-line
coverage with matching fresh executable/library profiles. CPU/CUDA, pure-C ABI,
C++ wrapper/example, style/document and ready-for-review PR gates pass.
Only then mark macro M07 complete.

Current validation: real CPU plugin cases **4/4**, CUDA-build plugin cases
**8/8**, native CPU **62/62**, native CUDA **108/108**, all passed without skips.
Broader CPU AutoEP suite **139/139** passed after staging nine existing fixture
files that the targeted build had not copied.
The CUDA build's supported AutoEP selection passed **137/137**, with 20 existing
EPContext/schema cases explicitly excluded because
`onnxruntime_DISABLE_CONTRIB_OPS=ON`; those cases passed in the full CPU suite.
Final new production/glue line coverage is **80/80**. Style-final rebuilt
profiles, sanitizer confirmation and publication passed. Implementation commit
`fd2331ff8a` is published in ready-for-review PR #19, stacked on PR #18.

**M07b verified evidence:**

- Appended OrtEpApi::RegisterStatefulProvider and its C++ wrapper. The copied,
  version-1 descriptor is registered transactionally on the synchronous CreateEp
  thread, never by reading added OrtEp fields. Nested scopes, duplicate/wrong
  registration, malformed/extended descriptors and failed creation are tested.
- Shared environment library records plus weak private device ownership retain
  factories/DLLs through plugin factory/adapter and backend leases without a
  cycle. Unregistration removes discovery immediately but cannot unload live
  callback-bearing owners. Native logger/custom logging-manager ownership also
  survives Session release for delayed context/provider teardown.
- A real example-plugin context logs through the retained custom Session logger
  after Session release. Host lifetime observers prove that context/factory
  destruction occurs exactly once and only after the last retained cache value.
  Reads from unchanged freed memory are not used as the only lifetime evidence.
- The legacy fixture is a frozen 1.23 prefix, not current OrtEp inheritance:
  exactly offsetof(OrtEp, GetKernelRegistry) bytes against an inaccessible guard
  page, with only original callbacks. CPU/CUDA tests load it and run Sessions
  without descriptor registration or later-field access.
- Full CPU AutoEP **139/139** and native **62/62** passed; supported CUDA AutoEP
  **137/137** and native **108/108** passed, zero skips. All eight CPU/CUDA plugin
  cases are included. New CUDA fixtures assert model-node assignment and disable
  CPU fallback; callbacks remain explicitly CPU implementations.
- The exploratory full CUDA AutoEP run contained 20 existing EPContext-dependent
  cases that cannot pass with this build's existing
  `onnxruntime_DISABLE_CONTRIB_OPS=ON` schema configuration. They are explicitly
  excluded only from the supported CUDA broad selector below, not presented as
  passing/skipped native feature tests. All passed in the full CPU suite.
- Initial broad CPU failures were missing pre-existing fixture copies, not
  source defects. Staged add_mul_add, squeeze_mul_relu, if_mul, loop_sub_one,
  scan_mul, custom_mul, conv_qdq_external_ini.onnx/.bin and sub_mul_sub in both
  build testdata directories; no source fixtures/dependencies changed.
- Final measured new production/glue coverage: **80/80 unique executable
  locations**, including registration, native factory/adapter/library ownership,
  logging-resource retention and the SDK wrapper. Fresh profiles combine all
  CPU AutoEP/native tests; SDK mapping uses the exercising plugin DLL.
  Inspected 132/351 zero-hash unused-map entries in SDK/native exports respectively,
  none affects the measured new functions. Earlier M07a's **782/782** gate remains
  independently documented/verified. These are executable-line, not branch or
  whole-example-fixture coverage percentages.
- Final CUDA Compute Sanitizer: **12/12 plugin/public API cases** plus **4/4
  native provider cases**, zero errors. GCC C11 consumer and C++17 registration
  wrapper compile/link/run passed against the final shared library.
- Formatting, 120 columns, whitespace, all 93 requirement IDs and all 24 macro
  milestones passed. CPU/GPU kernels, EPContext schemas, Model Package support,
  minimal/training and platform qualification were not expanded by this PR.

Passing final CPU commands:

```bash
source .venv/bin/activate
LLVM_PROFILE_FILE=/dev/null cmake --build build/state-cpu/RelWithDebInfo \
  --parallel 8 --target onnxruntime_autoep_test onnxruntime_session_loop_test
cd build/state-cpu/RelWithDebInfo
mkdir -p coverage/m07b/final
LLVM_PROFILE_FILE="$PWD/coverage/m07b/final/%m-%p.profraw" \
  ./onnxruntime_autoep_test --gtest_output=xml:coverage/m07b/final/all-autoep-tests.xml
LLVM_PROFILE_FILE="$PWD/coverage/m07b/final/%m-%p.profraw" \
  ./onnxruntime_session_loop_test --gtest_output=xml:coverage/m07b/final/native-tests.xml
llvm-profdata merge -sparse coverage/m07b/final/*.profraw \
  -o coverage/m07b/final/m07b.profdata
```

Export the new SDK wrapper from libexample_plugin_ep.so; export changed native
sources from onnxruntime_autoep_test with libonnxruntime.so and
onnxruntime_session_loop_test added as objects. Deduplicate DA locations and
intersect added lines against M07a `46715fb08a`; every one of 80 executed.
New untracked registration sources are wholly new for the initial export.

Passing final CUDA build/native/sanitizer commands use the established matching
CUDA 13.1 environment, from the CUDA output directory after building both targets:

```bash
./onnxruntime_session_loop_test --gtest_output=xml:coverage/m07b/final/native-tests.xml
"$CUDA_HOME/bin/compute-sanitizer" --tool memcheck --error-exitcode 99 \
  ./onnxruntime_autoep_test \
  --gtest_filter='Cuda/StatefulPluginTest.*:Cuda/StatefulApiProviderTest.*:Cuda/StatefulCxxApiProviderTest.*'
"$CUDA_HOME/bin/compute-sanitizer" --tool memcheck --error-exitcode 99 \
  ./onnxruntime_session_loop_test --gtest_filter='Cuda/StatefulProviderTest.*'
```

The supported CUDA broad run excludes exactly these existing schema-dependent
cases (139 CPU cases all passed, including these where present):

```bash
unsupported=(
  OrtEpLibrary.PluginEp_GenEpContextModel
  OrtEpLibrary.PluginEp_ExternalInitializerBufferIsPublishedOnlyAfterCompileSucceeds
  OrtEpLibrary.PluginEp_GenEpContextModel_EmbedModeDoesNotUseCallbacks
  OrtEpLibrary.PluginEp_GenAndLoadEpContextModel_ExternalDataUsesFileFallback
  OrtEpLibrary.PluginEp_GenEpContextModel_ExternalDataUsesWriteCallback
  OrtEpLibrary.PluginEp_LoadEpContextModel_ExternalDataUsesReadCallback
  OrtEpLibrary.PluginEp_CompileAndLoadWithoutFilesystem_ExternalInitializersAndEpContextData
  OrtEpLibrary.PluginEp_ExternalEpContextCallbacksRequireAdvertisedSupport
  OrtEpLibrary.PluginEp_GenWeightlessEpContextModel
  OrtEpLibrary.PluginEp_WeightlessAllInitializers_CompileApi
  OrtEpLibrary.PluginEp_ErrorWhenLoadEPContextModel_WithoutRequiredEp
  OrtEpLibrary.PluginEp_VirtGpu_GenEpContextModel
  OrtEpLibrary.PluginEp_CompatibilityInfo_WrittenToMetadata
  OrtEpLibrary.PluginEp_CompatibilityInfo_ValidatedOnLoad
  OrtEpLibrary.PluginEp_Error_LoadCompiledModelWithEpContextEnabled
  OrtEpLibrary.PluginEp_EpContextInference_NotImplemented
  OrtEpLibrary.PluginEp_GenEpContextModel_ErrorOutputModelExists_AutoGenOutputModelName
  OrtEpLibrary.PluginEp_GenEpContextModel_UnicodePath
  OrtEpLibrary.PluginEp_GenEpContextModel_ZeroInputModel
  ModelPackageTest.CheckCompiledModelCompatibilityInfo
)
filter="-$(IFS=:; echo "${unsupported[*]}")"
./onnxruntime_autoep_test --gtest_filter="$filter" \
  --gtest_output=xml:coverage/m07b/final/supported-autoep-tests.xml
```

Result: **137/137 passed**, zero skips; this is not a claim that the original
unfiltered CUDA suite passed. The equivalent final selector was constructed
from this exact stored exclusion list, never from arbitrary unrelated failures.

**Execution/validation procedure for both sub-PRs:**

- Keep this checklist and SCRATCHPAD current; stop for user review if a material
  contract change to approved Appendix G is necessary.
- Use the existing instrumented CPU and consistent CUDA 13.1 build trees.
  Regenerate CMake only when source discovery/targets/configuration changes.
- Start with the smallest new test selectors, then combine related new cases
  and the established API/native regressions. Verify nonzero XML counts, no
  unexpected skips, real callback invocation and required CUDA node assignment.
- Collect fresh `%m-%p.profraw` profiles for CPU executables and relevant shared
  modules; export/deduplicate exact new source locations against the sub-PR base.
  Inspect mismatch dumps and resolve warnings affecting measured changed code.
- Use existing build/test/lint tooling, no dependency additions unless required.
  No device-line coverage claim applies to CPU callbacks; any new device
  implementation would require the separate M08/M09 scope and coverage gates.
- Record exact passing commands/counts/coverage and limitations in this section
  and the repository PR template. Commit/push each independently verified PR;
  do not merge automatically or mark full M07 complete after only M07a.

### M08 - Native CUDA KV management

**Outcome:** Device-resident cache lifecycle and prefill/decode handoff.
**Status:** SPEC Appendix H approved 2026-10-18 without changes. User scoped M08
to CUDA plugin EP only, copy primitives/GPU cache views and an M08a/M08b split.
The user approved the detailed plan without changes, initially authorized M08a,
and separately authorized M08b on 2026-10-19. Both are complete and published
as independently verified ready-for-review PRs #20 and #21. No PR has been merged.

- [x] Implement CUDA allocation/retention/update/growth/reset using ORT allocator,
  device, and stream abstractions; advertise only supported features.
- [x] Wire stage sharing and required movement without illegal aliasing.
- [x] Expose dispatch/copy/allocation evidence for tests and later measurement.

**CPU test plan:** Rerun CPU cache and EP-contract regressions.
**CUDA test plan:** CPU/CUDA cache contents and lengths agree on fixtures; verify
device placement, compatible-buffer sharing, required copies, nondefault stream
ordering, completion before observation/release, growth/reset, invalid devices,
allocation/provider failures, cancellation, and resource cleanup.
**Completion gate:** Actual CUDA plugin dispatch plus measured 100% new
executable host/glue lines and any new device lines. If no device functions are
introduced, their new-line denominator is zero/N/A, not 100%; missing measurement
of device code actually added still blocks completion.

#### M08a - CUDA plugin cache backend and standalone device proof

**Intended outcome:** The CUDA plugin advertises and executes the six supported
KV operations on real GPU storage, using ORT allocators and correct synchronous
stream/resource boundaries. CPU remains the default; CUDA sampling stays absent.
**Dependencies:** M04, M06, M07. Stack this PR on M07b PR #19.

- [x] A1: Establish a separate CUDA-plugin-enabled build/test configuration using
  matching CUDA 13.1/cuDNN and the available A100 hardware. Wire the focused test
  targets' existing dynamic-plugin initialization/definitions/dependencies; do
  not accept a null DefaultCudaExecutionProvider or a legacy build as proof.
- [x] A2: Implement a boundary-safe CUDA-plugin cache callback/context module.
  Register it transactionally during CreateEp, retain code/factory/logger/
  allocator/stream resources, and advertise only implemented KV capabilities,
  supported dtypes/devices and contiguous compact layout. No legacy/shared
  CUDA hook or greedy capability is added.
- [x] A3: Implement GPU initialization, owned-output Share, full Copy, arbitrary-
  sequence-axis delta Append, Reserve/growth and logical Reset. Reuse CUDA copy
  primitives, checked geometry/size calculations and M07 result validation;
  preserve old views and do not assume graph input/output aliasing.
- [x] A4: Use compatible ORT/plugin GPU allocators and appropriate stream/device
  ownership. Establish producer/consumer ordering and complete operation work at
  the synchronous boundary; avoid unrequested global device-wide waits.
  Standalone caller-produced tensors follow the ordinary explicit input-readiness
  contract; unknown external producer streams cannot be inferred from a pointer.
- [x] A5: Add real standalone C/C++ backend tests for device placement, data,
  alias/retention semantics, reference results, errors and operation counters.
  Preserve CPU defaults and explicit no-CPU-fallback/no-sampler behavior.
- [x] A6: Measure 100% every new CUDA host/callback/resource line using the actual
  instrumented plugin artifact. Prefer a compatible Clang/LLVM host build with
  NVCC's supported host compiler, or use matching GCC host coverage where needed;
  do not count a non-executing CPU stub. If custom device functions become
  necessary, return to the strict measured-device-line gate rather than waive it.
- [x] A7: Complete CPU/plugin-GPU regressions, sanitizer, ABI/style/examples,
  exact evidence and independently reviewable publication.

**CPU test plan:**

- Rerun M07's CPU cache, policy, lifetime, public API and plugin-registration
  baselines. A CPU-only build must not acquire a CUDA runtime dependency.
- Verify unchanged default CPU results and explicit unsupported/required errors
  when the selected build/provider lacks the CUDA-plugin cache capability.
- Compile public C/C++ examples and descriptor boundaries; preserve copied
  configuration/metadata and retained const-view behavior.

**CUDA plugin test plan:**

- Load the real CUDA plugin, query its six KV bits/dtype/device contract and
  confirm the greedy bit is absent. Assert callback invocation, physical GPU
  allocation/device id and zero CPU-reference delegation/fallback.
- Compare copied-back contents, logical length and capacity against CPU reference
  for float32/float16/bfloat16, multiple outer dimensions, sequence axes at the
  beginning/middle/end and lengths around zero/growth boundaries.
- Check Share pointer identity and rejection of borrowed/incompatible storage;
  verify Copy/Append/Reserve allocate safely and retained views do not change.
- Test zero capacity, spare capacity, empty appends, no-op reserve, Reset,
  repeated reuse and final destruction after mutable object/Session release.
- Test default/nondefault and caller-configured streams, scoped device-state
  restoration and correct completion. Check stream/event readiness or controlled
  ordering before a readback that could otherwise mask missing synchronization.
- Use controlled allocator/provider failures and overflow/device errors; never
  exhaust a shared GPU to simulate OOM. Verify sticky invalidation/no CPU retry,
  successful reset recovery and safe resource cleanup.
- Run relevant cases under Compute Sanitizer; record actual nonzero counts,
  supported architecture and no unexpected skips.

**M08a completion gate:** Real CUDA-plugin allocations/operations and lifetime
proof pass, including the new callback module's measured 100% host coverage.
Record device coverage as zero/N/A only if no device functions were added.
Publish ready for review; loop/chain GPU-cache integration is not yet claimed.

**M08a complete:** Implementation `e6ce58508e`, published ready for human review
as [PR #20](https://github.com/onnxruntime/ort-state/pull/20), stacked on M07b
PR #19. No PR has been merged. The user authorized M08b on 2026-10-19.

- Six cache operations execute in the real CUDA plugin with float32/float16/
  bfloat16 support and no greedy capability or CPU-reference delegation.
  Compact copies handle every explicit sequence axis and preserve snapshots.
- The size-tagged request has an appended `retain_value` callback for owned
  handle references across the C boundary; no API-table/version change or native
  Session dependency is introduced into the plugin module.
- Equal Tensor pointers can have different OrtValue owners. Repeated Share now
  preserves the existing leased owner; an exact context-destruction regression
  covers repeated Share/Reset and final releases.
- Capacity bytes are checked before tensor creation. Custom allocators can
  return NULL without tensor creation failing, so nonempty storage is explicitly
  checked before any copy. Tests require precise allocation/overflow errors,
  unchanged allocator-call count on overflow, sticky invalidation, reset recovery
  and no leaked CUDA last-error state. The initial sanitizer-detected invalid
  copy is fixed; no broad-suite exclusion or sanitizer suppression is accepted.
- CPU: **63/63 native +139/139 unfiltered AutoEP** passed.
- CUDA plugin: **111/111 native +174/174 unfiltered AutoEP** passed, zero skips,
  with `CUDA_VISIBLE_DEVICES=0,1`. The 174 include **17 new device cases**.
  XML records prove restored caller device **1** and **2** selected cache devices.
  Default/nondefault stream completion is checked before readback; the blocked
  stream uses a prepared allocation so cudaMalloc cannot mask missing completion.
- Legacy/shared CUDA: **43/43** selected regression cases passed, including actual
  GPU model selection, zero native cache bits and exact `ORT_NOT_IMPLEMENTED`
  during required-cache construction with a compatible GPU allocator.
- Final actual-plugin new production/glue coverage: **137/137** executable source
  locations: callback/resource module **123/123**, factory registration **4/4**
  and core retaining/owner glue **10/10**. Fresh profiles use
  `coverage/m08a/complete`. New device functions/lines are **0/0, N/A**, not a
  claimed 100% device result: only existing CUDA runtime copy APIs are used.
- Compute Sanitizer memcheck/leak-check: **17/17 device +6/6 native-provider**
  cases and the documented C++17 CUDA example passed, **zero errors and zero
  leaked bytes/allocations**. A separate two-case allocation-error-then-inference
  regression also passed with zero errors/leaks.
- Genuine C11 consumer compilation and execution passed. The documented C++17
  CPU example passed normally and with no-exceptions **client** flags. The CUDA
  example passed normal execution and memcheck through reset, backend-Session
  release and library unregistration. Its separate caller-owned allocator Session
  remains alive as required by the existing Session-allocator API; final DLL
  unload while that caller resource is alive is not claimed.
- New code passes strict warning-as-error Clang type checking; the callback also
  passes GCC13 host checking. Formatting/120 columns, whitespace, 93 requirement
  IDs and 24 macro milestones passed. The CPU library has no CUDA dependency.

The separate plugin tree uses Clang18 host LLVM coverage, NVCC13.1/GCC13, SM80,
matching CUDA13.1/cuDNN, contrib schemas enabled, and NHWC/Flash/MEA/TRT-fused
attention disabled. Its final regeneration used the supported
`--compile-no-warning-as-error` option for pre-existing Clang pool/Cast and mixed
compiler diagnostics; no unrelated CUDA source was changed to silence them.
The final cache also records CXX flags
`-fprofile-instr-generate -fcoverage-mapping -Wno-error=unused-private-field -Wno-error=instantiation-after-specialization`
and CUDA flag `-Xcompiler=-Wno-error=pragmas`.

Focused native tests use the existing dynamic-provider test infrastructure.
The DYN AutoEP executable instead registers CUDA through the public Env API,
avoiding private native test-library dependencies. All three stateful public
fixtures use the common V2/legacy routing helper. Generic CUDA selection uses
the public delegate for plugins rather than the old internal-only prototype,
recognizes preloaded libraries, and requests a supported option when NHWC is
disabled; CPU fallback is disabled in that CUDA selection proof.

Passing final build/test commands (run tests from each output directory):

```bash
source .venv/bin/activate
cmake --build build/state-cpu/RelWithDebInfo --parallel 8 \
  --target onnxruntime_session_loop_test onnxruntime_autoep_test
cd build/state-cpu/RelWithDebInfo
# Passed: native 63/63; AutoEP 139/139.
./onnxruntime_session_loop_test
./onnxruntime_autoep_test
cd ../../..

cmake --compile-no-warning-as-error -S cmake \
  -B build/state-cuda-plugin-13.1/RelWithDebInfo
cmake --build build/state-cuda-plugin-13.1/RelWithDebInfo --parallel 16 \
  --target onnxruntime_session_loop_test onnxruntime_autoep_test
cd build/state-cuda-plugin-13.1/RelWithDebInfo
export CUDA_VISIBLE_DEVICES=0,1
export CUDA_HOME=/datadisks/disk4/bmeswani/cuda/cuda13.1
export LD_LIBRARY_PATH="$PWD:$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib"
mkdir -p coverage/m08a/complete
LLVM_PROFILE_FILE="$PWD/coverage/m08a/complete/%m-%p.profraw" \
  ./onnxruntime_session_loop_test --gtest_output=xml:coverage/m08a/complete/native-tests.xml
# Passed: 111/111, zero skips.
LLVM_PROFILE_FILE="$PWD/coverage/m08a/complete/%m-%p.profraw" \
  ./onnxruntime_autoep_test --gtest_output=xml:coverage/m08a/complete/autoep-tests.xml
# Passed: 174/174, zero skips; no exclusions.
llvm-profdata merge -sparse coverage/m08a/complete/*.profraw \
  -o coverage/m08a/complete/m08a.profdata
# Passed: module-specific llvm-cov line exports intersected against added lines
# from M07b 03d7476859, including wholly new callback source: 137/137.
export LLVM_PROFILE_FILE=/dev/null
"$CUDA_HOME/bin/compute-sanitizer" --tool memcheck --leak-check full --error-exitcode 99 \
  ./onnxruntime_autoep_test --gtest_filter='*CudaStateful*'
# Passed: 17/17, zero errors/leaks.
"$CUDA_HOME/bin/compute-sanitizer" --tool memcheck --leak-check full --error-exitcode 99 \
  ./onnxruntime_session_loop_test \
  --gtest_filter='StatefulCudaProviderBoundaryTest.*:Cuda/StatefulProviderTest.*'
# Passed: 6/6, zero errors/leaks.

cd ../../state-cuda-baseline-13.1/RelWithDebInfo
export LD_LIBRARY_PATH="$PWD:$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib"
./onnxruntime_autoep_test --gtest_filter='AutoEpSelection.CudaEP:*Stateful*'
# Passed: 43/43, zero skips.
```

Export the callback/factory source from `libonnxruntime_providers_cuda.so` and
the core source from `libonnxruntime.so`. Inspect LLVM mismatch dumps before
accepting coverage: the initial 588 plugin/174 runtime warnings were exclusively
zero-hash dormant maps, with no new-function mismatch; final dumps are checked
again against the passing profiles. Exact logs/report are retained as
`m08a-*-complete-*`, `m08a-plugin-sanitizer-*`, `m08a-cpu-verified-*`,
`m08a-legacy-stateful-tests.log` and `m08a-host-coverage-final.txt` in session files.
M08b was subsequently authorized and completed in PR #21, as recorded below.

#### M08b - Device-resident loop/chain integration

**Status:** Authorized 2026-10-19. Implementation branch
`state/m08b-cuda-plugin-cache-integration`, based on M08a `6959554e55`.

**Intended outcome:** Programmatic loops/chains use the selected GPU cache
without cache host round-trips, while stable ordinary outputs and CPU greedy
sampling retain their existing behavior. Prefill/decode share one GPU backend
across compatible CUDA-plugin Sessions, including distinct streams.
**Dependency:** M08a. Separate branch/PR stacked on its verified/published head.

- [x] B1: Resolve the selected GPU allocator/device from initialized Session
  resources without hard-coded allocator names or hidden CPU substitution.
  Remove only the implemented CUDA-plugin cache integration rejection; keep
  device sampling and legacy/shared native-cache requests unsupported.
- [x] B2: Bind managed cache outputs to GPU memory and feed device-resident
  retained cache values back to Sessions. Process cache callback results without
  converting their state through CPU snapshots.
- [x] B3: Wire selected/component/state/downstream cache tensor surfaces
  consistently: managed cache values remain GPU-resident under explicit GPU
  selection, ordinary non-cache outputs stay stable CPU values, and CPU logits/
  token sampling remain explicitly observable until M09.
- [x] B4: Preserve compatible prefill/decode sharing across Sessions and
  nondefault streams on the same GPU. Validate shared backend/layout/device and
  cache-port compatibility; reject unsupported heterogeneous/cross-device paths.
- [x] B5: Preserve retained GPU views/results across update/reset/release and
  unregistration; propagate callback, model, cancellation and reset failures
  through the existing invalidation/C/C++ error rules. Respect caller-owned
  streams/allocators/OrtEnv obligations.
- [x] B6: Add end-to-end native/public C/C++ parity, device/copy/dispatch/staging
  evidence and stream/lifetime tests. Exercise relevant named-state/component
  handoff paths, not just one preferred cache binding.
- [x] B7: Measure every introduced core/plugin host path and any device code,
  rerun baselines/sanitizer/examples, document exact limits/evidence, and publish
  the ready-for-review stacked PR. Only then complete macro M08.

**CPU test plan:**

- Rerun CPU loops/chains/generation/cache and all M07 contracts/registration/
  lifetime/public API regressions. Verify no CUDA dependency or behavior change
  for CPU-default execution and no accidental device-greedy advertisement.
- Verify explicit policy controls still reject unavailable providers/devices.
  Legacy/shared CUDA remains unsupported for native KV by approved scope.

**CUDA plugin test plan:**

- Run tiny real GPU cache-producing/consuming graphs and prefill/decode fixtures;
  disable CPU model fallback and assert actual CUDA plugin node assignment.
- Compare tokens, GPU cache contents and processed lengths with CPU/reference
  Sessions for zero/one/multiple-token budgets, EOS/iteration exits and reset.
  CPU sampling must be reported as CPU, not native CUDA.
- Assert managed cache ports/results/views stay GPU-resident and do not acquire
  CPU cache copies through internal snapshots or chain retention. Inspect GPU
  contents only with explicit test copies; ordinary outputs/logits remain CPU.
- Exercise different compatible plugin Sessions/streams, all relevant loop/chain
  forms and downstream cache/state handoff. Reject mismatched GPU ids/layouts
  and unsupported cache-port/heterogeneous configurations.
- Verify cache-ready completion before next-stage use/observation and safe
  release after cancellation/model/provider errors. Readback synchronization
  alone is not proof that the API's completion boundary was correct.
- Test isolated contexts, old snapshots after capacity growth/reset, retained
  values after Session release/library unregistration and exactly-once resource
  destruction using counters/observers where relevant.
- Run C/C++ consumers/examples, actual default/nondefault-stream cases and
  relevant integration cases under Compute Sanitizer with zero errors.

**M08b completion gate:** Real GPU-resident cache state and cross-Session
handoff are proved through native and C/C++, not just standalone callbacks.
All new core/plugin host lines and any new device lines meet their measured
coverage gates. CPU/CUDA-plugin correctness, memory/order/lifetime, documentation
and publication pass; mark M08 complete only after both sub-PRs.

**M08b complete:** Implementation `249f60acf8`, published ready for human review
as [PR #21](https://github.com/onnxruntime/ort-state/pull/21), stacked on M08a
PR #20. This completes macro M08. No PR has been merged; M09 is not started.

- Resolve actual Session GPU allocators and validate the declared cache ports'
  planned device. Managed and declared cache-alias outputs are bound directly to
  GPU and retain local allocator/provider leases rather than CPU snapshots.
  Chains use the first cache-managing Session's allocator, not an assumed first
  component allocator. A leading CPU-only component is explicitly covered.
- Track declared internal component/named-state cache edges, including a plain
  Session state writer with no managed cache pair, finite inputs and feedback.
  Arbitrary external/initial-state chain feeds remain host-only. Reject
  cross-device/incompatible providers and host cache consumers before invocation.
- Preserve CPU logits/greedy, ordinary output snapshots and separate staging
  counters. Shared GPU updates have zero CPU reference delegation. Native tests
  inspect actual bound GPU inputs/outputs and Share pointer identity versus
  Copy/Append fresh storage; public tests cover all four Session/Loop chain forms,
  prefill/decode budgets 0/1/3, EOS, reset, independent contexts and retained results.
- Synchronous Run finishes the actual execution streams before result
  publication. Device-cache execution avoids IOBinding's additional provider-wide
  Sync. The normal unrelated-stream test uses warmed same-thread execution,
  ready device inputs and prepared distinct outputs to isolate cache work from
  synchronous host transfers/allocator effects; **10 consecutive repetitions
  passed** before releasing the unrelated blocked stream.
- Final CPU: **63/63 native +139/139 unfiltered AutoEP** passed.
- Final CUDA plugin: **116/116 native +179/179 unfiltered AutoEP** passed,
  **zero skips/exclusions**, with CUDA devices 0 and 1 visible. The new cases
  are five native and five public GPU integration tests; parameter loops cover
  phase/budget/EOS/chain/alias variations. XML confirms the cross-device
  rejection test actually ran on two GPUs.
- Final legacy/shared CUDA: **110/110 native +43/43 selected public/stateful**
  cases passed. Native cache capabilities remain absent; no legacy hook is added.
- Final new runtime executable-line coverage: **130/130**, comprising loop
  integration **88/88** and chain integration **42/42**. The shared dynamic
  decoder fixture helper is separately **12/12**. Fresh matching GPU profiles
  use `coverage/m08b/complete`; the 398 LLVM map warnings are exclusively
  dormant zero-hash maps, with no new-function mismatch.
- New device functions/lines: **0/0, N/A**, not a 100% device-coverage claim.
  M08b adds no CUDA kernel or backend device code.
- Compute Sanitizer memcheck/leak-check: **4/4 native memory cases +22/22 public
  integration/standalone cache cases** passed with **zero errors and zero leaked
  bytes/allocations**. The genuine C11 loop consumer retains a GPU result through
  reset, execution/definition/Session release and library unregistration.
- **Sanitizer timing limitation:** The fifth native case's wall-clock assertion
  times out inside instrumented model Run while the unrelated host callback is
  blocked. It remains in every full normal suite and the coverage denominator,
  and passed ten repeated normal runs. It is excluded **only** from the explicit
  native memcheck memory selector below; it is not claimed to pass memcheck.
  Same-thread warmup/device-only feeds/prepared outputs did not change that
  instrumented result. A separate CUDA-only memset calibration passed normally
  and under memcheck, so no universal sanitizer-serialization claim is made.
  No memory error is suppressed, and no runtime correctness test is excluded.
- Public C11 syntax checks, documented C++17 CPU example with normal and
  no-exceptions client flags, strict changed-source type checks, formatting/
  120-column limits, whitespace and 93-requirement/24-milestone traceability
  passed. The CPU shared library has no CUDA runtime dependency.

Passing commands (test commands run from their corresponding output directories):

```bash
source .venv/bin/activate
cmake --compile-no-warning-as-error -S cmake \
  -B build/state-cuda-plugin-13.1/RelWithDebInfo
cmake --build build/state-cuda-plugin-13.1/RelWithDebInfo --parallel 16 \
  --target onnxruntime_session_loop_test onnxruntime_autoep_test
cd build/state-cuda-plugin-13.1/RelWithDebInfo
export CUDA_VISIBLE_DEVICES=0,1
export CUDA_HOME=/datadisks/disk4/bmeswani/cuda/cuda13.1
export LD_LIBRARY_PATH="$PWD:$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib"
mkdir -p coverage/m08b/complete
LLVM_PROFILE_FILE="$PWD/coverage/m08b/complete/%m-%p.profraw" \
  ./onnxruntime_session_loop_test --gtest_output=xml:coverage/m08b/complete/native-tests.xml
# Passed: 116/116, zero skips.
LLVM_PROFILE_FILE="$PWD/coverage/m08b/complete/%m-%p.profraw" \
  ./onnxruntime_autoep_test --gtest_output=xml:coverage/m08b/complete/autoep-tests.xml
# Passed: 179/179, zero skips.
llvm-profdata merge -sparse coverage/m08b/complete/*.profraw \
  -o coverage/m08b/complete/m08b.profdata
# Passed: LLVM changed-source line exports against M08a6959554e55:
# runtime130/130; helper12/12. No device functions added.
export LLVM_PROFILE_FILE=/dev/null
./onnxruntime_session_loop_test \
  --gtest_filter='CudaCacheIntegrationTest.SynchronousTaskStreamWithoutGlobalDeviceWait' \
  --gtest_repeat=10 --gtest_break_on_failure
# Passed: all ten normal repetitions.
"$CUDA_HOME/bin/compute-sanitizer" --tool memcheck --leak-check full --error-exitcode 99 \
  ./onnxruntime_session_loop_test \
  --gtest_filter='CudaCacheIntegrationTest.*-CudaCacheIntegrationTest.SynchronousTaskStreamWithoutGlobalDeviceWait'
# Passed: 4/4, zero errors/leaks. Timing-only instrumentation limitation above.
"$CUDA_HOME/bin/compute-sanitizer" --tool memcheck --leak-check full --error-exitcode 99 \
  ./onnxruntime_autoep_test --gtest_filter='CudaCacheApiIntegrationTest.*:*CudaStateful*'
# Passed: 22/22, zero errors/leaks.
cd ../../state-cpu/RelWithDebInfo
./onnxruntime_session_loop_test --gtest_output=xml:coverage/m08b/final/native-tests.xml
./onnxruntime_autoep_test --gtest_output=xml:coverage/m08b/final/autoep-tests.xml
# Passed: 63/63 native and 139/139 unfiltered AutoEP.
cd ../../state-cuda-baseline-13.1/RelWithDebInfo
export LD_LIBRARY_PATH="$PWD:$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib"
./onnxruntime_session_loop_test --gtest_output=xml:coverage/m08b/final/native-tests.xml
./onnxruntime_autoep_test --gtest_filter='AutoEpSelection.CudaEP:*Stateful*' \
  --gtest_output=xml:coverage/m08b/final/stateful-tests.xml
# Passed: 110/110 native and 43/43 selected public/stateful cases.
```

Compiler/CUDA/build configuration matches M08a: Clang18 host coverage,
NVCC13.1/GCC13, SM80, matching CUDA13.1/cuDNN. Regeneration retains the supported
nonfatal-warning build configuration for pre-existing mixed-compiler diagnostics.
No dependency or unrelated backend source changes were needed. Exact evidence:
`m08b-gpu-complete-*`, `m08b-cpu-*-tests.log`, `m08b-legacy-*-tests.log`,
`m08b-sanitizer-native-memory.log`, `m08b-sanitizer-public.log`,
`m08b-stream-repeated.log` and `m08b-host-coverage-final.txt` in session files.
M09 GPU sampling is not started or claimed.

**Validation and execution procedure:**

- Keep TASKS/SCRATCHPAD current; stop for review of material changes to approved
  Appendix H. No implementation for legacy/shared CUDA is included.
- Use the existing instrumented CPU tree plus a distinct plugin-enabled CUDA
  tree. Record compiler, CUDA/NVCC host compiler, CMake flags, GPU architecture
  and coverage toolchain. Never mix profiles from older artifacts/build variants.
- Instrument and exercise CUDA host code in the real plugin DLL. Host coverage
  from a CPU mock cannot establish it. Measure any conditional core integration
  paths using an appropriate matching build/fixture.
- Start with smallest relevant selectors; then combine new cases with native/
  public API/registration/allocator baselines. Confirm nonzero XML counts,
  actual callbacks/devices/model assignment and no unexpected skips.
- Preserve the known distinction between the legacy contrib-disabled CUDA
  baseline and the new plugin configuration. Document unsupported schema/test
  exclusions explicitly rather than reporting an unfiltered suite as passing.
- Prefer standard CUDA copy/allocation calls; no new device function means a
  zero/N/A device-line denominator, not missing coverage. New device code, if
  any is introduced, must be measured at 100% or blocks completion.
- Publish independently documented/tested stacked PRs with exact passing
  commands/counts/coverage and the approved plugin-only limitation; do not merge
  automatically or mark full M08 complete after only M08a.

### M09 - Native CUDA greedy sampling

**Outcome:** Greedy token selection without an undisclosed host sampler.
**Status:** Interview complete and SPEC Appendix I approved without changes
2026-10-19. The user approved the detailed plan without changes and authorized
M09a only. Pause before M09b.
The user chose CUDA plugin only, GPU-resident exposed logits, token/status-only
readback and an M09a/M09b split, with no additional requirements. Branch
`state/m09a-cuda-plugin-greedy` is based on M08b `2b6388fe5b`. The standalone
backend is verified and published ready for review in PR #22.
M09b remains unauthorized and no loop/chain GPU sampling is claimed.

- [x] Implement CUDA greedy reduction/selection with CPU-equivalent documented
  tie/non-finite semantics and supported dtype/layout handling.
- [ ] Integrate provider discovery, stream ordering, token observation, and exits.
- [x] Keep cache and sampler capability selection independent.

**CPU test plan:** Rerun greedy/EP baseline and unsupported-capability tests.
**CUDA test plan:** Exact token agreement on deterministic fixtures, vocab sizes
around launch/reduction boundaries, supported dtypes, ties/non-finite cases,
stream ordering, error cases, native-dispatch assertion, and no host sampling.
Exercise the complete tiny prefill/decode chain using M08 caches.
**Completion gate:** Shared rules, CPU parity, and 100% new host/device-line
coverage pass. Latency/copy measurements have no performance pass threshold.

#### M09a - CUDA plugin greedy backend and standalone proof

**Intended outcome:** The real CUDA plugin advertises and executes synchronous
GPU greedy for supported logits, including whole-tensor NaN rejection. Only
token/status information is read back. CPU defaults and cache capabilities stay
unchanged; loop/chain GPU sampling remains explicitly unsupported until M09b.
**Dependencies:** M05 CPU semantics, M06 C/C++, M07 dispatch/plugin ABI and the
verified M08 resource/device baseline. Stack on M08b PR #21 (`2b6388fe5b`).

- [x] A1: Establish device-line measurement before implementing the production
  sampler. Use CUDA13.1/NVCC, source line information and available Nsight Compute
  source/SASS execution counters or another demonstrated device execution-to-
  source method. A small session-local probe must distinguish executed lines
  from a deliberately unexecuted compiled path; line tables/host launch counts
  alone are insufficient. Verify permissions and reproducible export/denominator.
  If unavailable, stop with the evidence rather than waive the device gate.
- [x] A2: Define the smallest boundary-safe callback/resource wiring using the
  existing descriptor. Reuse compatible retained ORT/plugin scratch allocators,
  selected device/stream and M07/M08 leases, including external allocator/stream
  obligations. If a request resource extension is actually necessary, use
  size-checked appended C fields with frozen prior prefixes, not native Session
  types or an unrelated API/version redesign.
- [x] A3: Implement selection plus complete-logits NaN validation on GPU for
  float32/float16. Audit existing last-axis argmax, IsNaN and reduction utilities;
  reuse only when their semantics and index limits fit. Use checked sizes and
  deterministic lowest-ID ties across every reduction boundary. Prefer one
  simple proven path over speculative launch tiers or duplicate algorithms.
- [x] A4: Integrate the greedy callback and advertise its exact capability bit/
  dtypes/devices without changing six-cache behavior or legacy/shared CUDA.
  Transfer only the token and compact validation/status result, complete work
  on its stream before return, restore device state and surface allocation/
  launch/transfer failures without CPU retry or changed out-parameters.
- [x] A5: Add genuine native/C/C++ standalone conformance tests and controlled
  resource/error cases. Assert real callback/device work, zero CPU delegation,
  supported copy sizes and immutable input logits. Validate metadata/index/
  overflow rejection before launch; do not exhaust shared GPUs to provoke OOM.
- [x] A6: Measure every new executable host/device line from exercised matching
  artifacts, including new device helpers, functors and template logic.
  Intersect against the M09a base, report both denominators and close all gaps.
  Identify reused unchanged kernels separately; do not treat M08's copy-only
  result as M09 coverage evidence.
- [x] A7: Finish CPU/CUDA/legacy regressions, sanitizer, ABI/style/public SDK
  examples and representative latency/copy measurement. Record exact commands/
  counts/coverage/limits, then publish an independently reviewable ready PR.
  Pause before M09b unless its execution was separately authorized.

**CPU test plan:**

- Rerun greedy semantics, loop/chain generation, caches, public API/provider/
  registration/lifetime baselines in the CPU tree. No CUDA dependency is added.
- Preserve CPU default selection, explicit Prefer/Require behavior and
  transactional public outputs. Missing GPU capability is not a failed callback;
  advertised callback failures, including NOT_IMPLEMENTED, never retry CPU.
- Compile/run C11 and C++17 consumers; preserve no-exceptions client compilation,
  ABI prefix/version rules and immutable returned-value ownership.

**CUDA plugin test plan:**

- Load the actual plugin and assert the greedy bit, float32/float16 sampler mask,
  selected physical GPU and zero CPU-reference delegation. Assert unchanged KV
  bits and absence of native sampling in the legacy/shared CUDA build.
- Compare exact IDs with CPU for rank2/rank3, default/first/interior/last rows,
  V=1 and sizes below/at/above all chosen launch/reduction boundaries, including
  multiple blocks/tiles where used. Cover realistic larger vocabulary examples.
- Cover negative/positive/mixed infinities, all-negative infinity, equal maxima
  spanning lanes/blocks, signed zeros, float16 limits, and NaNs at the beginning/
  middle/end of selected and unselected rows. Verify rejection leaves the public
  token output unchanged; no full-logits CPU sampling/readback is allowed.
- Cover wrong batch/rank/dtype/position/device/contiguity, empty dimensions,
  checked arithmetic and controlled allocation/provider failures before unsafe
  access. Confirm explicit errors and no CPU retry.
- Test default/nondefault streams, caller device restoration, input readiness
  duties, completion before token observation and cleanup after failure.
  Test standalone use after releasing its bound Session/options where permitted,
  respecting caller-owned allocator/stream/OrtEnv obligations.
- Run applicable cases under Compute Sanitizer with exact nonzero counts and
  zero unexpected memory errors/leaks. Measure token/status movement and latency
  separately from logical CPU staging counters.

**M09a completion gate:** Actual standalone CUDA sampling, exact CPU semantics,
bounded observations, safe resource/error behavior and measured 100% new host/
device executable lines all pass. An unavailable device-coverage measurement
blocks this gate. Loop/chain GPU logits integration is not yet claimed.

**A1 profiling access resolved; source-line proof passed (2026-10-19):**

- Built a session-local CUDA13.1/NVCC SM80 source-line probe, separate from
  production runtime code. It has distinct noinline executed/unexecuted device
  functions selected by a runtime mode, providing a negative control.
- Both ordinary executions passed: mode1 produced32 and mode2 produced-32,
  exactly as expected. This proves only that the test kernel executes, not
  source-line coverage.
- Nsight Compute2025.4.1 launched the probe but source-counter collection failed
  with **`ERR_NVGPUCTRPERM`**: this user cannot access NVIDIA GPU performance
  counters on device0. No valid device-line coverage report was collected.
- At that point no driver settings/permissions were modified, no privileged
  command or permission workaround was attempted, and production implementation
  had not begun. A2-A7 were blocked pending A1.
- The initial required input was profiling permission arranged by the administrator,
  or access to another approved environment/tool that can demonstrate the same
  executed/unexecuted source-line measurement. Do not substitute host coverage,
  successful kernel output or memcheck for the required device proof.
- The user then explicitly authorized admin privileges for profiling. Retrying
  the profiler with `sudo -n -- .../ncu` succeeded without changing any global
  driver setting. Both mode reports were exported as CUDA/SASS source counters.
- Reciprocal negative-control assertions passed: line6's atomicAdd executes32
  predicated-on thread instructions in mode1 and0 in mode2; line10's atomicSub
  executes0 in mode1 and32 in mode2. The taken/untaken call sites also reverse.
  `m09a-device-coverage-proof.txt` records the assertions and source counters.
  A1 is now complete; A2-A7 may proceed under the approved M09a authorization.
- Production device coverage will count authored executable source locations
  mapped to device instructions. Compiler-only padding/epilogue mappings must be
  identified separately, never used to conceal an unexecuted authored statement.
- Evidence/source: `m09a-source-coverage-probe.log` and
  `m09a-device-coverage-probe.cu` in session files. Rebuild its temporary executable
  before rerunning after final cleanup.

```bash
CUDA_HOME=/datadisks/disk4/bmeswani/cuda/cuda13.1
probe=/home/devpatel/.copilot/session-state/819a29ec-8ea8-4b2f-9938-e40236ffd239/files/m09a-device-coverage-probe
"$CUDA_HOME/bin/nvcc" -O2 -lineinfo -arch=sm_80 -ccbin /usr/bin/g++ \
  "$probe.cu" -o "$probe"
CUDA_VISIBLE_DEVICES=0 "$probe" 1
CUDA_VISIBLE_DEVICES=0 "$probe" 2
# Passed: expected32 and-32, respectively. Not coverage evidence.
CUDA_VISIBLE_DEVICES=0 "$CUDA_HOME/bin/ncu" --section SourceCounters \
  --import-source yes --force-overwrite \
  --export /home/devpatel/.copilot/session-state/819a29ec-8ea8-4b2f-9938-e40236ffd239/files/m09a-source-coverage-mode1 \
  "$probe" 1
# Initial unprivileged attempt failed: ERR_NVGPUCTRPERM.
# User-authorized sudo -n -- ncu retries for modes1/2 passed and exported
# source counters that distinguish both executed/unexecuted controls.
```

**M09a complete:** Implementation `73066ddcbe`, published ready for human review
as [PR #22](https://github.com/onnxruntime/ort-state/pull/22), stacked on M08b
PR #21. Macro M09 remains incomplete until M09b is separately authorized and
verified. No PR has been merged.

- Implemented a two-stage CUDA reduction with CUB BlockReduce: a grid-stride
  whole-logits NaN scan and selected-row argmax produce bounded block partials,
  followed by a compact token/status result. Float32/float16, lowest-ID ties,
  infinities, signed zero and subnormal values agree exactly with CPU.
  Existing last-axis argmax's NaN behavior does not satisfy whole-tensor
  rejection, so only suitable lower-level reduction machinery is reused.
- The callback uses the initialized Session's compatible retained scratch
  allocator through an appended size-checked borrowed request field. The native
  binding supplies its C adapter only during the synchronous callback; scratch
  is released before that adapter expires. No new public API-table entry,
  version bump or native Session header dependency in generate/ is introduced.
- Host launch code uses standard cudaLaunchKernel in a separate C++ translation
  unit; registered entry signatures match the NVCC device definitions. This
  makes every authored host launch line LLVM-measurable rather than omitting
  uninstrumented NVCC/GCC host statements. Only the new kernel source adds
  source line information; no device debug/coverage proxy substitutes for
  measured GPU execution.
- CUDA capability bits are now127 (six KV operations plus greedy); cache masks/
  behavior remain unchanged and sampler types are float32/float16 on GPU.
  Legacy/shared CUDA remains unextended. GPU loop/chain sampling still returns
  its explicit integration-unavailable error until separately authorized M09b.
- New conformance covers reduction widths1/31/32/33/255/256/257/1023/1024/1025/
  65535/65536/65537/131073, differing selected rows, cross-block ties, all/mixed
  infinities, signed zeros, subnormals/extreme finite values and NaNs anywhere.
  Tests compare exact CPU results, unchanged input bytes and failed-call token
  sentinels, actual provider/no-delegation counters, wrong metadata/devices,
  checked overflow, default/user streams, device restoration and bound-Session
  release. Controlled failed allocations reject before kernel/data access.
- Final CPU: **63/63 native +139/139 unfiltered AutoEP** passed.
- Final CUDA plugin: **117/117 native +184/184 unfiltered AutoEP** passed on
  visible GPUs0/1, **zero skips/exclusions**. These include five new standalone
  GPU test cases and a new native greedy request/allocator boundary case.
  Existing cache/generation tests retain CPU sampling and remain regressions,
  not an assertion that M09b has already been implemented.
- Legacy/shared CUDA: **110/110 native +43/43 selected public/stateful** cases
  passed; native sampling/cache capability remains absent.
- Measured new executable **host/glue coverage83/83**: CUDA callback52/52,
  C++ kernel launcher15/15, native resource binding14/14 and sampler dispatch2/2.
  Fresh actual-plugin profiles use `coverage/m09a/final`. The589 plugin/174
  runtime LLVM map warnings are exclusively dormant zero-hash entries, with
  no new-function mismatch.
- Measured new executable **device coverage25/25** from real Nsight Compute
  SourceCounters CUDA/SASS exports of44 launched kernels. Predicated-on thread
  instruction counts prove both float/half entries, final reduction and all
  emitted own-source locations executed. Mapped entry/epilogue braces are
  included; **no device source location is excluded**. The declarations-only
  helper header has no additional device instruction mappings. The report is
  matched against the source SHA256 and exercised plugin artifact; kernel
  success/host coverage/memcheck are not used as device coverage proxies.
- Actual movement proof from a separate Nsight Systems CUDA trace:
  **33 Device-to-Host copies, each16 bytes, total528 bytes**, matching33 sampler
  calls. No full-logits readback occurs in the isolated sampler trace. The
  trace also contains201036 Host-to-Device bytes, including the explicit201028-
  byte logits upload. Those uploads are not sampling readback or a claim of a
  fully device-only input pipeline.
- Representative instrumented-build latency for float32 vocabulary50257,
  32 measured calls after GPU warmup: **67.33us GPU** and **98.54us CPU** mean
  synchronous sample time. These are observations, not release performance
  claims or pass thresholds; CPU/GPU profiling/configuration is documented.
- Compute Sanitizer: **5/5 greedy +2/2 native boundary** memcheck/leak-check cases
  and the documented C++17 GPU example passed with **zero errors/leaked bytes/
  allocations**. The shared-memory reduction coverage case also passed
  **racecheck with zero hazards/warnings** and **synccheck with zero errors**.
  No new sampler case is excluded from its applicable sanitizer selector.
  M08's separately documented unrelated-stream timing limitation is unchanged,
  not silently extended or represented as an instrumented timing pass.
- Genuine C11 sampler consumer compilation/execution and the C++17 GPU SDK
  example passed, including sampling after bound-Session release and library
  unregistration while the caller's allocator Session remains alive. The CPU
  C++17 example also passed normal/no-exceptions **client** builds. Runtime
  no-exceptions behavior and final DLL unload while caller resources remain
  alive are not newly claimed.
- All changed-source strict type checks, source/kernel compilation, formatting/
  120-column limits, whitespace and93-requirement/24-milestone traceability pass.
  The CPU shared library has no CUDA dependency. Final public sampler comments
  describe CPU-default/explicit device behavior; these final comment-only edits
  do not change the validated runtime code or ABI.

Passing build/test/measurement commands (tests run from their output directories):

```bash
source .venv/bin/activate
cmake --compile-no-warning-as-error -S cmake \
  -B build/state-cuda-plugin-13.1/RelWithDebInfo
cmake --build build/state-cuda-plugin-13.1/RelWithDebInfo --parallel 24 \
  --target onnxruntime_session_loop_test onnxruntime_autoep_test
cd build/state-cuda-plugin-13.1/RelWithDebInfo
export CUDA_VISIBLE_DEVICES=0,1
export CUDA_HOME=/datadisks/disk4/bmeswani/cuda/cuda13.1
export LD_LIBRARY_PATH="$PWD:$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib"
mkdir -p coverage/m09a/final
LLVM_PROFILE_FILE="$PWD/coverage/m09a/final/%m-%p.profraw" \
  ./onnxruntime_session_loop_test --gtest_output=xml:coverage/m09a/final/native-tests.xml
# Passed:117/117, zero skips.
LLVM_PROFILE_FILE="$PWD/coverage/m09a/final/%m-%p.profraw" \
  ./onnxruntime_autoep_test --gtest_output=xml:coverage/m09a/final/autoep-tests.xml
# Passed:184/184, zero skips.
llvm-profdata merge -sparse coverage/m09a/final/*.profraw \
  -o coverage/m09a/final/m09a.profdata
# Passed: module-specific LLVM new-line measurement83/83 against M08b2b6388fe5b.
export LLVM_PROFILE_FILE=/dev/null
"$CUDA_HOME/bin/compute-sanitizer" --tool memcheck --leak-check full --error-exitcode 99 \
  ./onnxruntime_autoep_test --gtest_filter='CudaStatefulTest.Greedy*'
# Passed:5/5, zero errors/leaks.
"$CUDA_HOME/bin/compute-sanitizer" --tool memcheck --leak-check full --error-exitcode 99 \
  ./onnxruntime_session_loop_test --gtest_filter='StatefulCudaProviderBoundaryTest.*'
# Passed:2/2, zero errors/leaks.
"$CUDA_HOME/bin/compute-sanitizer" --tool racecheck --error-exitcode 99 \
  ./onnxruntime_autoep_test --gtest_filter='CudaStatefulTest.GreedyDeviceCoverage'
# Passed:1/1, zero hazards/errors/warnings.
"$CUDA_HOME/bin/compute-sanitizer" --tool synccheck --error-exitcode 99 \
  ./onnxruntime_autoep_test --gtest_filter='CudaStatefulTest.GreedyDeviceCoverage'
# Passed:1/1, zero errors.
```

The actual measurement used the user's explicitly authorized privileged profiler
process only, with no global driver settings changed:

```bash
# From the CUDA plugin output directory:
sudo -n -- env CUDA_VISIBLE_DEVICES=0 LLVM_PROFILE_FILE=/dev/null \
  LD_LIBRARY_PATH="$PWD:$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib" \
  "$CUDA_HOME/bin/ncu" --devices 0 --section SourceCounters --import-source yes \
  --kernel-name-base function --kernel-name 'regex:.*Greedy.*' --force-overwrite \
  --export /home/devpatel/.copilot/session-state/819a29ec-8ea8-4b2f-9938-e40236ffd239/files/m09a-greedy-device-coverage \
  ./onnxruntime_autoep_test --gtest_filter='CudaStatefulTest.GreedyDeviceCoverage'
# Passed:1/1 and44 profiled launches; CUDA/SASS source exports give25/25.
"$CUDA_HOME/bin/nsys" profile --sample=none --cpuctxsw=none --trace=cuda \
  --force-overwrite=true \
  --output=/home/devpatel/.copilot/session-state/819a29ec-8ea8-4b2f-9938-e40236ffd239/files/m09a-greedy-transfer \
  ./onnxruntime_autoep_test --gtest_filter='CudaStatefulTest.GreedyTransferAndLatency'
# Passed:1/1; exported CUPTI memcpy SQLite records prove33 DtoH transfers16 bytes each.
```

CPU output-directory commands `./onnxruntime_session_loop_test` and
`./onnxruntime_autoep_test` passed63/139. The corresponding legacy CUDA native
command and `./onnxruntime_autoep_test --gtest_filter='AutoEpSelection.CudaEP:*Stateful*'`
passed110/43. Compiler/configuration matches M08: Clang18 host LLVM,
NVCC13.1/GCC13, SM80, matching CUDA13.1/cuDNN, existing nonfatal-warning mode
for baseline mixed-compiler diagnostics. No dependency install was needed.
Exact evidence/source/counter exports remain in session files as
`m09a-host-coverage-final.txt`, `m09a-device-coverage-final.txt`,
`m09a-greedy-device-coverage.ncu-rep`, `m09a-greedy-device-source.csv`,
`m09a-transfer-proof.txt`, `m09a-greedy-transfer.sqlite`, `m09a-*-tests.log`
and sanitizer logs. M09b remains unimplemented and requires user authorization.

#### M09b - GPU logits and loop/chain sampling integration

**Intended outcome:** Explicit CUDA-plugin sampling runs end-to-end with
GPU-resident logits and compatible M08 caches. Exposed logits stay GPU; ordinary
outputs, host token history and CPU-default sampling preserve their contracts.
**Dependency:** Verified/published M09a. Separate branch/PR stacked on its head.

- [ ] B1: Resolve sampler device/allocator resources from initialized Sessions
  and validate actual logits producer placement. Remove only the implemented
  CUDA-plugin sampler rejection; leave legacy/shared and unsupported devices
  explicit. Reuse/refine M08 routing without conflating cache and sampler policy.
- [ ] B2: Bind explicitly GPU-sampled logits on GPU and retain stable device
  values in selected/component/named-state/downstream surfaces, including
  internal chain retention of unselected model outputs. CPU sampler logits and
  other ordinary outputs remain stable CPU snapshots. Conflicting roles/device
  requirements must reject rather than force a hidden transfer.
- [ ] B3: Invoke GPU sampling through existing dispatch, observe the completed
  token/status before updating history/exits, and preserve prefill pending-token
  handoff, matching budgets, EOS, iteration exits and zero-budget no-invocation.
  Keep generated CPU input-feed behavior visible; do not claim fully device-only
  token generation or reintroduce a CPU argmax path.
- [ ] B4: Preserve independent cache/sampler choices and per-stage selection:
  GPU cache/CPU sampler, CPU cache/GPU sampler, and compatible GPU cache/GPU
  sampler. Prove cross-Session/nondefault-stream handoff on the selected GPU;
  reject unsupported mismatched producers/devices and arbitrary migration.
- [ ] B5: Exercise cancellation, NaN/model/provider/transfer failure, reset,
  isolated contexts and retained GPU logits through update/release/Session
  teardown/library unregistration. Preserve public C/C++ transactional errors,
  sticky execution invalidation and caller-owned resource lifetime duties.
- [ ] B6: Add native and genuine C/C++ end-to-end parity, explicit device/
  dispatch/transfer/staging evidence and reference comparisons. Measure all new
  core/plugin host code and any new device code; preserve M09a's independent
  device coverage proof rather than replacing it with host-only results.
- [ ] B7: Rerun CPU/plugin/legacy baselines and applicable sanitizer checks,
  verify retained-output/public SDK/style contracts, record latency/copy results
  and exact coverage evidence, and publish the second ready-for-review PR.
  Complete macro M09 only after both independently verified sub-PRs are published.

**CPU test plan:**

- Run CPU loops/chains/generation/sampler/cache and provider/public ABI/lifetime
  baselines. CPU remains default and retains CPU logits snapshots.
- Confirm unchanged zero-budget validation, first-token/EOS semantics,
  metadata/error behavior, staging counters and no CUDA runtime dependency.
- Preserve unsupported-capability errors and explicitly compatible CPU fallback;
  no failed GPU callback may silently become CPU sampling.

**CUDA plugin test plan:**

- Execute the complete tiny prefill/decode chain using real GPU model assignment,
  GPU logits, native GPU greedy and compatible M08 GPU caches. Disable model CPU
  fallback and assert actual callbacks; token equality alone is insufficient.
- Compare tokens, processed lengths, cache data and exit reasons against CPU for
  budgets0/1/multiple, EOS/iteration exits and reset. Cover float32/float16 logits
  and explicit row selection with NaN rejection across the complete tensor.
- Assert selected/component/state-retained logits remain GPU and survive later
  executions/reset/release. Read GPU data only via explicit test copies. Verify
  ordinary outputs remain CPU and no chain bookkeeping introduces full-logits
  readback under GPU sampling.
- Test each cache/sampler policy combination and differing stage sampler choices,
  default/user streams and compatible cross-Session ordering. Count actual
  token/status transfer payloads or trace them; do not infer DMA from dispatch
  labels or mislabel ordinary host token feeds as GPU-native work.
- Exercise errors/NaNs/cancellation/reset/teardown and library unregistration,
  including no downstream invocation after failure. Check completion before an
  observation/readback can conceal a synchronization error.
- Run new memory/order/lifetime cases under Compute Sanitizer where applicable.
  Preserve M08's exact already-documented timing-only instrumentation limitation
  separately; do not silently extend that exclusion to new sampler cases.

**M09b completion gate:** GPU-resident logits and actual GPU sampling are proved
through standalone-to-generation native/C/C++ integration. All new executable
host/device lines are measured at100%, prior M09a gates remain valid, CPU/GPU/
legacy regressions and applicable sanitizer checks pass, and exact limitations/
measurements are documented in the published stacked PR.

**Execution and evidence procedure for both sub-PRs:**

- Keep SPEC/TASKS/SCRATCHPAD current; the attached editor snapshots are read-only.
  Seek review before any material departure from approved Appendix I.
- Use the existing instrumented CPU tree and separate CUDA13.1 plugin tree;
  legacy CUDA only establishes its own regression/unsupported-capability results.
  Preserve the supported NVCC host compiler, source information and matching
  artifact/profile identity when combining host and device measurements.
- Do not claim source coverage from host wrapper counters, successful launches
  or sanitizer output. Save device source mapping/executed-line exports, test
  selectors and denominator calculations; retain negative-control evidence.
- Start with smallest conformance selectors, then full relevant baselines.
  Verify nonzero counts, intended backend execution and no unexpected skips.
  Explicitly disclose any tool/configuration-specific exclusion and its scope.
- Stage only necessary fixtures and clean exact temporary consumers/probes.
  Publish each independently verified PR with the repository template and exact
  validation results; do not automatically merge or begin M10.

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
| CUDA device-line coverage | M08 copy-primitive approach approved; M09 still requires device instrumentation | Actual GPU dispatch remains mandatory. M08 may report zero/N/A only with no new device functions; any device code added in M08/M09 needs measured 100% coverage. |
| Local Phi-3/SD files and adapters | Paths/integrations not supplied | Request concrete paths before M14/M15; synthetic fixtures do not replace real demos. |
| CPU/CUDA model variants | Actual signatures/precision not inspected | Verify compatible local variants; no silent model substitution/download. |
| Objective-C/iOS validation | Linux is the initial platform; Apple access not provided | M22/M23 need supplemental runner/coverage or remain blocked; no implicit scope reduction. |
| Browser/mobile package resources | Runtime file APIs differ from native filesystem APIs | M01/M19/M23 define adapters preserving package confinement and execution parity. |

Approval record: on 2026-10-15, the user approved this plan without changes and
explicitly authorized beginning M01. Changes to scope or mandatory gates still
require user review.
