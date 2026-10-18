# Stateful Session Execution: Execution Plan

**Status:** Approved; M01-M06 and M07a complete; M07b in progress.
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

M01-M06 are complete; M07-M24 are pending. Check tasks when their
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
M01-M06 are **complete**; M07-M24 are **pending**.

| Done | ID | Intended PR / branch suffix | Semantic prerequisites |
| --- | --- | --- | --- |
| [x] | M01 | Contracts, deterministic fixtures, coverage readiness / `contracts-fixtures` | Approved plan |
| [x] | M02 | Stateful context and explicit SessionLoop / `context-loop` | M01 |
| [x] | M03 | SessionChain and tensor/state bindings / `chain-bindings` | M02 |
| [x] | M04 | CPU KV cache management / `cpu-kv-cache` | M03 |
| [x] | M05 | CPU greedy sampling and decode iteration / `cpu-greedy` | M04 |
| [x] | M06 | Public C API and C++ wrappers / `public-api` | M02, M03, M04, M05 |
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
M07b follows the verified/published M07a head; macro M07 remains incomplete.
The authorized plugin implementation is in progress on
`state/m07b-plugin-stateful-hooks`, based on M07a `46715fb08a`.

- [ ] Add capability-based in-tree hooks and boundary-safe plugin C contracts.
- [x] Route CPU implementations through the selection contract; specify allowed
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
- [ ] B6: Run C/C++ plugin conformance and CPU/CUDA builds/regressions, include
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
profiles and sanitizer confirmation passed; publication remains.

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
