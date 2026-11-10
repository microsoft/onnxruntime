# Stateful Session Execution: Execution Plan

**Status:** Approved; M01-M12 complete and published for review; M13-M24 pending.
M05: [PR #34044](https://github.com/microsoft/onnxruntime/pull/34044).
M05's requested generation-folder layout follow-up is complete in the same PR.
**Source:** [Approved specification](SPEC.md), approved 2026-10-15.
**Working memory:** [Shared scratchpad](SCRATCHPAD.md).
**M10 refinement:** SPEC Appendix J and the detailed M10 implementation plan
were separately approved on 2026-10-19.
**M11 refinement:** SPEC Appendix K and the detailed M11 plan were separately
approved 2026-10-20. CPU-first implementation is verified; GPU qualification
resumed on 2026-10-21 after a compatible toolchain was supplied.
**M12 refinement:** SPEC Appendix L was separately approved on 2026-10-30.
The detailed M12 execution plan was separately approved on 2026-10-30;
CPU/CUDA qualification and publication completed on 2026-10-31.

## 1. Approval and tracking

- [x] Complete the requirements interview.
- [x] Ask whether the user has additional requirements before drafting.
- [x] Draft and review `SPEC.md` with the user.
- [x] Obtain specification approval.
- [x] Prepare this execution-plan draft.
- [x] Validate draft structure, dependency ordering, and all 93 requirement mappings.
- [x] Review this plan with the user and obtain approval before implementation.
- [ ] Complete the milestone stack and its validation gates.

M01-M12 are complete and M13-M24 are pending. Check tasks when their
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
M01-M12 are **complete**; M13-M24 are **pending**.

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
| [x] | M09 | Native CUDA greedy sampling / `cuda-greedy` | M07, M08 |
| [x] | M10 | Standalone package schema and authoring / `package-schema` | M01, M03 |
| [x] | M11 | ORT package-to-chain loading / `package-runtime` | M06, M07, M08, M09, M10 |
| [x] | M12 | DDIM scheduler and denoising state / `ddim-policy` | M03, M06, M08 |
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
**Status:** Complete; [PR #34040](https://github.com/microsoft/onnxruntime/pull/34040)
is ready for review, targeting `state/stateful-base-1988019b`.

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

- First milestone branch created: `stateful-M01-contracts-fixtures`.
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

Published the unchanged `state/stateful-base-1988019b` baseline branch and
`stateful-M01-contracts-fixtures`; the first PR targets the agreed `state/stateful-base-1988019b` base,
not `main`. Implementation commit: `64aa026b94`. The PR contains no stateful
runtime production implementation; M02 is the next pending milestone.

### M02 - Execution context and SessionLoop

**Outcome:** Internal sequential stateful execution with explicit control.
**Status:** Complete; [PR #34041](https://github.com/microsoft/onnxruntime/pull/34041)
is ready for review. Branch `stateful-M02-context-loop` stacks on
`stateful-M01-contracts-fixtures`. Implementation commit: `a21ae23201`.

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
**Status:** Complete; [PR #34042](https://github.com/microsoft/onnxruntime/pull/34042)
is ready for review. Branch `stateful-M03-chain-bindings` stacks on
`stateful-M02-context-loop`. Implementation commit: `c4ad4d5389`.

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
**Status:** Complete; [PR #34043](https://github.com/microsoft/onnxruntime/pull/34043)
is ready for review. Branch `stateful-M04-cpu-kv-cache` stacks on
`stateful-M03-chain-bindings`. Implementation commit: `fffa292875`.

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
**Status:** Complete; [PR #34044](https://github.com/microsoft/onnxruntime/pull/34044)
is ready for review. Branch `stateful-M05-cpu-greedy` stacks on
`stateful-M04-cpu-kv-cache`. Implementation commit: `ddb8260691`.

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
M06a is complete: [PR #34045](https://github.com/microsoft/onnxruntime/pull/34045),
`stateful-M06-stateful-c-api`, based on M05 `17e432b025`.
M06b is complete: [PR #34046](https://github.com/microsoft/onnxruntime/pull/34046),
`stateful-M07-stateful-cxx-api`, based on M06a `b5f1431837`.
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
  Ready-for-review PR #34045 is stacked on M05; implementation commit `f0aa2513d7`.

**M06b validation checklist:**

- [x] Add move-only mutable C++ resources and complete public API wrappers.
- [x] Retain result/cache owners in copyable const values and copy metadata.
- [x] Test move/release ordering, errors and full CPU/CUDA behavior parity.
- [x] Measure 100% new wrapper lines and cumulative C/native coverage.
- [x] Validate standalone C++ consumption, CUDA memcheck, examples and style.
- [x] Record final evidence and publish the ready-for-review PR stacked on M06a.
  PR #34046 is stacked on PR #34045; implementation commit `cd63f2f8ba`.

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
M07a is complete: [PR #34047](https://github.com/microsoft/onnxruntime/pull/34047),
`stateful-M08-ep-stateful-hooks`, based on M06b `5bde969b66`.
M07b is complete: [PR #34048](https://github.com/microsoft/onnxruntime/pull/34048),
`stateful-M09-plugin-stateful-hooks`, based on M07a `46715fb08a`.
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
**Dependencies:** M04, M05, M06. Stack this PR on M06b PR #34046.

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
Implementation commit `8fbd403493` is published in ready-for-review PR #34047.
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
`fd2331ff8a` is published in ready-for-review PR #34048, stacked on PR #34047.

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
as independently verified ready-for-review PRs #34049 and #34050. No PR has been merged.

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
**Dependencies:** M04, M06, M07. Stack this PR on M07b PR #34048.

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
as [PR #34049](https://github.com/microsoft/onnxruntime/pull/34049), stacked on M07b
PR #34048. No PR has been merged. The user authorized M08b on 2026-10-19.

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
M08b was subsequently authorized and completed in PR #34050, as recorded below.

#### M08b - Device-resident loop/chain integration

**Status:** Authorized 2026-10-19. Implementation branch
`stateful-M11-cuda-plugin-cache-integration`, based on M08a `6959554e55`.

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
as [PR #34050](https://github.com/microsoft/onnxruntime/pull/34050), stacked on M08a
PR #34049. This completes macro M08. No PR has been merged; M09 is not started.

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
`stateful-M12-cuda-plugin-greedy` is based on M08b `2b6388fe5b`. The standalone
backend is verified and published ready for review in PR #34051.
M09b was separately authorized on 2026-10-19 and is complete in PR #34052.

- [x] Implement CUDA greedy reduction/selection with CPU-equivalent documented
  tie/non-finite semantics and supported dtype/layout handling.
- [x] Integrate provider discovery, stream ordering, token observation, and exits.
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
verified M08 resource/device baseline. Stack on M08b PR #34050 (`2b6388fe5b`).

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
as [PR #34051](https://github.com/microsoft/onnxruntime/pull/34051), stacked on M08b
PR #34050. Macro M09 remains incomplete until M09b is separately authorized and
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
and sanitizer logs. M09b was subsequently authorized and completed in PR #34052.

#### M09b - GPU logits and loop/chain sampling integration

**Status:** Authorized 2026-10-19. Branch
`stateful-M13-cuda-plugin-sampling-integration` is based on M09a `ff2363e3b8`,
stacked on PR #34051. GPU kernels are unchanged; this milestone wires host execution.

**Intended outcome:** Explicit CUDA-plugin sampling runs end-to-end with
GPU-resident logits and compatible M08 caches. Exposed logits stay GPU; ordinary
outputs, host token history and CPU-default sampling preserve their contracts.
**Dependency:** Verified/published M09a. Separate branch/PR stacked on its head.

- [x] B1: Resolve sampler device/allocator resources from initialized Sessions
  and validate actual logits producer placement. Remove only the implemented
  CUDA-plugin sampler rejection; leave legacy/shared and unsupported devices
  explicit. Reuse/refine M08 routing without conflating cache and sampler policy.
- [x] B2: Bind explicitly GPU-sampled logits on GPU and retain stable device
  values in selected/component/named-state/downstream surfaces, including
  internal chain retention of unselected model outputs. CPU sampler logits and
  other ordinary outputs remain stable CPU snapshots. Conflicting roles/device
  requirements must reject rather than force a hidden transfer.
- [x] B3: Invoke GPU sampling through existing dispatch, observe the completed
  token/status before updating history/exits, and preserve prefill pending-token
  handoff, matching budgets, EOS, iteration exits and zero-budget no-invocation.
  Keep generated CPU input-feed behavior visible; do not claim fully device-only
  token generation or reintroduce a CPU argmax path.
- [x] B4: Preserve independent cache/sampler choices and per-stage selection:
  GPU cache/CPU sampler, CPU cache/GPU sampler, and compatible GPU cache/GPU
  sampler. Prove cross-Session/nondefault-stream handoff on the selected GPU;
  reject unsupported mismatched producers/devices and arbitrary migration.
- [x] B5: Exercise cancellation, NaN/model/provider/transfer failure, reset,
  isolated contexts and retained GPU logits through update/release/Session
  teardown/library unregistration. Preserve public C/C++ transactional errors,
  sticky execution invalidation and caller-owned resource lifetime duties.
- [x] B6: Add native and genuine C/C++ end-to-end parity, explicit device/
  dispatch/transfer/staging evidence and reference comparisons. Measure all new
  core/plugin host code and any new device code; preserve M09a's independent
  device coverage proof rather than replacing it with host-only results.
- [x] B7: Rerun CPU/plugin/legacy baselines and applicable sanitizer checks,
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

**M09b complete:** Implementation `2a684c8ba1`, published ready for human review
as [PR #34052](https://github.com/microsoft/onnxruntime/pull/34052), stacked on M09a
PR #34051. This completes macro M09. No PR has been merged; M10 is not started.

- Generalized private GPU cache port routing to declared device values, using
  the retained initialized-Session allocator already owned by the native provider
  binding. Cache allocation still follows cache policy, not whichever GPU value
  happens to configure the Session: CPU cache/GPU sampling remains CPU storage.
- GPU-sampled logits are bound and retained on GPU in selected/component/state
  results and declared downstream/finite inputs. Chain routes carry their
  originating provider/device; GPU cache routes, sampler-only routes and
  per-stage CPU/GPU sampling remain independent. Compatible Require/Prefer
  policies are not confused with physical device compatibility.
- Reject CPU/GPU role conflicts, mismatched routed devices, host producers/
  consumers and GPU-feedback inputs with CPU-feedback outputs before invocation.
  Snapshot publication validates the declared storage role, rather than accepting
  unexpected CPU data for a device output. Arbitrary chain external/initial-state
  inputs retain their host-only contract.
- Preserve actual task-stream Run completion without extra provider-wide Sync,
  scalar token/status observation before history/exits, zero-budget validation,
  prefill pending-token handoff, reset/cancellation boundaries and stable old
  logits/cache results. Model/NaN failures suppress downstream invocations and
  require reset; public failed-call output pointers remain unchanged.
- Final CPU: **63/63 native +139/139 unfiltered AutoEP** passed.
- Final CUDA plugin: **121/121 native +190/190 unfiltered AutoEP** passed,
  **zero skips/exclusions**, with visible GPUs0/1.
- New integration tests: **4 native +6 public**, including96 combinations of
  cache policy, phase sampler policy, float32/float16, budgets0/1/3 and EOS.
  They also cover explicit row selection, sampler-only/no-cache models,
  unselected internal logits, named-state/component/finite routes, actual bound
  placements, corrupt output storage, resource teardown and cross-device
  rejection. XML confirms `cross_device_logits_tested=true`.
- Legacy/shared CUDA: **110/110 native +43/43 selected public/stateful**
  regression cases passed; no native GPU sampler/cache hook is added.
- Measured new executable **host coverage116/116**: loop routing59/59,
  chain routing54/54 and retained-allocator accessor3/3. Fresh actual-plugin
  profiles use `coverage/m09b/final`. All403 LLVM mismatch-map entries are
  dormant zero-hash maps, with no new-function mismatch.
- New device functions/lines: **0/0, N/A**, not an invented100% device result.
  The M09a greedy device source **and compiled object** hashes are unchanged,
  preserving its independently measured **25/25** device-source proof. M09b
  changes only host execution/resource routing and tests.
- End-to-end Nsight Systems trace with vocabulary257 and GPU caches/logits:
  **3 Device-to-Host copies, each16 bytes, total48 bytes**, matching3 GPU greedy
  callbacks. There is no2056-byte prefill/1028-byte decode logits readback or
  cache readback. The trace separately records5 Host-to-Device copies totaling
  8256 bytes (weights/prompt/generated-token setup); host feeds are not called
  device-only work or confused with logical CPU staging counters.
- A normal instrumented-suite synchronous3-token Run observation was
  **28533.93us**. This includes execution setup/host orchestration and is not a
  release throughput benchmark or numerical performance gate.
- Compute Sanitizer memcheck/leak-check: **4/4 native +6/6 public integration
  cases and the executable C++17 generation example** passed with **zero errors
  and zero leaked bytes/allocations**. No new sampling case is excluded.
  The previously documented M08 unrelated-stream timing-only limitation remains
  separate; the full normal suite includes that test and passes it.
- Genuine C11 generation consumer and public C++17 end-to-end example pass
  through reset, definition/execution/Session release and library unregistration,
  with explicit inspection of retained GPU logits afterward. CPU C++17 normal
  and no-exceptions **client** examples, strict changed-unit type checks,
  formatting/120-column limits, whitespace and93-requirement/24-milestone
  traceability pass. The CPU shared library has no CUDA dependency.
- The IDE test runner reported no connected client; actual repository test
  binaries were run from their output directories. SDK model generation first
  failed with missing Python `onnx`; only then was ONNX1.23.2 and its dependencies
  installed into the existing isolated `.venv`. No runtime dependency/manifest
  was added or shared driver setting changed.

Passing final commands (test commands run from the respective output directory):

```bash
source .venv/bin/activate
cmake --build build/state-cuda-plugin-13.1/RelWithDebInfo --parallel 24 \
  --target onnxruntime_session_loop_test onnxruntime_autoep_test
cd build/state-cuda-plugin-13.1/RelWithDebInfo
export CUDA_VISIBLE_DEVICES=0,1
export CUDA_HOME=/datadisks/disk4/bmeswani/cuda/cuda13.1
export LD_LIBRARY_PATH="$PWD:$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib"
mkdir -p coverage/m09b/final
LLVM_PROFILE_FILE="$PWD/coverage/m09b/final/%m-%p.profraw" \
  ./onnxruntime_session_loop_test --gtest_output=xml:coverage/m09b/final/native-tests.xml
# Passed:121/121, zero skips.
LLVM_PROFILE_FILE="$PWD/coverage/m09b/final/%m-%p.profraw" \
  ./onnxruntime_autoep_test --gtest_output=xml:coverage/m09b/final/autoep-tests.xml
# Passed:190/190, zero skips.
llvm-profdata merge -sparse coverage/m09b/final/*.profraw \
  -o coverage/m09b/final/m09b.profdata
# Passed: changed-source LLVM measurement116/116 against M09aff2363e3b8.
export LLVM_PROFILE_FILE=/dev/null
"$CUDA_HOME/bin/compute-sanitizer" --tool memcheck --leak-check full --error-exitcode 99 \
  ./onnxruntime_session_loop_test --gtest_filter='CudaSamplingIntegrationTest.*'
# Passed:4/4, zero errors/leaks.
"$CUDA_HOME/bin/compute-sanitizer" --tool memcheck --leak-check full --error-exitcode 99 \
  ./onnxruntime_autoep_test --gtest_filter='CudaSamplingApiIntegrationTest.*'
# Passed:6/6, zero errors/leaks.
"$CUDA_HOME/bin/nsys" profile --sample=none --cpuctxsw=none --trace=cuda \
  --force-overwrite=true \
  --output=/home/devpatel/.copilot/session-state/819a29ec-8ea8-4b2f-9938-e40236ffd239/files/m09b-generation-transfer \
  ./onnxruntime_autoep_test --gtest_filter='CudaSamplingApiIntegrationTest.GpuSamplingTransferAndLatency'
# Passed:1/1; exported CUPTI memcpy SQLite records prove3 DtoH copies16 bytes each.
cd ../../state-cpu/RelWithDebInfo
./onnxruntime_session_loop_test --gtest_output=xml:coverage/m09b/final/native-tests.xml
./onnxruntime_autoep_test --gtest_output=xml:coverage/m09b/final/autoep-tests.xml
# Passed:63/63 native and139/139 unfiltered AutoEP.
cd ../../state-cuda-baseline-13.1/RelWithDebInfo
export LD_LIBRARY_PATH="$PWD:$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib"
./onnxruntime_session_loop_test --gtest_output=xml:coverage/m09b/final/native-tests.xml
./onnxruntime_autoep_test --gtest_filter='AutoEpSelection.CudaEP:*Stateful*' \
  --gtest_output=xml:coverage/m09b/final/stateful-tests.xml
# Passed:110/110 native and43/43 selected public/stateful cases.
```

Compiler/build settings match the independently verified M09a artifacts:
Clang18 host LLVM coverage, NVCC13.1/GCC13, SM80, matching CUDA13.1/cuDNN and
the existing nonfatal-warning configuration for baseline mixed-compiler warnings.
Exact logs and reports remain in session files as `m09b-*-tests.log`,
`m09b-native-sanitizer.log`, `m09b-public-sanitizer.log`,
`m09b-sdk-sanitizer.log`, `m09b-host-coverage-final.txt`,
`m09b-device-baseline-sha256.txt`, `m09b-transfer-proof.txt` and
`m09b-generation-transfer.sqlite`. SPEC I.7 contains the executable SDK example;
`m09b_make_sdk_model.py` regenerates its checked validation fixture. M10 is not
started and no PR has been merged.

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

**Status:** Complete; [PR #34053](https://github.com/microsoft/onnxruntime/pull/34053)
is open/ready for human review, stacked on #34052. The user approved SPEC J.9's
expanded save-as requirements and separately approved the detailed plan
amendment on 2026-10-19.
**Branch/base:** `stateful-M14-package-chain-schema`, stacked on verified M09b
`a7a1173cd1c64d318857fa2987caf26658048509` / PR #34052.
**Outcome:** Read, inspect, author and commit schema-1.1 named linear chains
through the independent Model Package library, with JSON-only APIs and immediate
common-reference integrity. No executable ORT chain construction in this PR.
**Contract:** [SPEC Appendix J](SPEC.md#appendix-j-m10-standalone-package-chains-approved).

#### Approval gates

- [x] Review the decoder-only candidate and its model references with the user.
- [x] Confirm sampling-free prefill and decode's initial-logits handoff.
- [x] Confirm no additional requirements before drafting the specification.
- [x] Draft and obtain approval of SPEC Appendix J.
- [x] Prepare the detailed M10 execution plan.
- [x] Review this plan with the user and obtain implementation authorization.
- [x] Complete all implementation, test, coverage and publication gates below.

#### Implementation tasks and dependencies

These are internal tasks for one independently reviewable M10 PR, not separately
authorized milestones. Stop and review a split before expanding beyond a
coherent standalone package change.

| Task | Actual prerequisites | Intended result |
| --- | --- | --- |
| C1 | Plan approval | Contract fixtures and baseline evidence for the existing library. |
| C2 | C1 | Versioned common-schema validation and JSON inspection. |
| C3 | C2 | Transactional chain authoring and reference-safe component removal. |
| C4 | C3, amendment approval | Self-contained save-as repair and persistence/compatibility qualification. |
| C5 | C3 | Real C11 consumption and standalone shared/static independence. |
| C6 | C4, C5 | Fresh-artifact coverage and existing CPU ORT package regressions. |
| C7 | C6 | Reviewed documentation, evidence and a ready stacked PR. |

- [x] **C1 - Establish fixtures and baseline.** Reuse the standalone tests'
  deterministic temporary-package helpers and existing assertions. Add compact
  fixtures for 1.0, 1.1 and a future minor; one/multiple chains; ordinary Session
  and SessionLoop consumer payloads; inline/external components; opaque foreign
  namespaces; and the approved decoder-only metadata. No real model download or
  execution is needed for standalone fixtures. Record existing inspection,
  authoring, commit and hashing test results before production changes.
- [x] **C2 - Parse and inspect common chain structure.** Add the optional map,
  nonempty ordered steps, unique IDs, nonempty/NUL-free identifiers, existing
  component references and inline namespace-object validation. Keep consumer
  payload internals opaque. Coordinate parser known-minor and New-package
  version at 1.1. Add `GetChainsJson` and `GetChainJson`, stable package-owned
  JSON caches, unchanged-output-on-error behavior and correct absent/not-found
  results. Preserve old/future-minor strictness and unknown-field behavior.
- [x] **C3 - Author chains without dangling drafts.** Add `SetChainsJson`,
  `SetChainJson` and `RemoveChain`. Validate and prepare replacement data before
  publishing changes; do not introduce post-mutation filesystem/cache-refresh
  failures that would violate the new atomicity contract. Upgrade 1.0 only on
  successful chain-setting mutations and never downgrade a future minor.
  Protect `RemoveComponent` before orphan bookkeeping or record mutation.
  Preserve idempotent removal and same-key component replacement. Reuse common
  validation across parsing and mutation instead of duplicating schema logic.
- [x] **C4 - Repair and qualify persistence and compatibility.** Apply approved
  SPEC J.9 through D1-D4 below, then verify all existing commit
  modes serialize/reopen the same map, order, opaque namespaces and tolerated
  unknown fields. Exercise single-chain replacement versus whole-map replacement,
  last-chain removal and successful recovery after rejected mutations. Preserve
  external variant executor-info, shared assets, installed paths, portable
  confinement and prune behavior. Verify a frozen M09b/1.0 reader can preserve
  additive 1.1 chain data through its existing future-minor behavior.
- [x] **C5 - Prove the standalone C surface.** Add a small reusable C11 test
  consumer using all five APIs and explicit pass/fail checks that remain active
  under `NDEBUG`. Enable C only in the standalone test configuration as needed;
  do not introduce a runtime C dependency merely for a test. Run shared and
  static standalone builds using cached nlohmann/json headers, with no ORT
  targets available. Inspect shared-library dependencies and static consumer
  linkage to prove no ORT/ONNX/CUDA runtime dependency was introduced.
- [x] **C6 - Measure and regress.** Run the smallest affected CTest selection
  during development, then the complete standalone suite with nonzero internal
  case counts. Rebuild the existing CPU ORT package-test target and run its
  verified package API selector from the output directory. Measure every new
  executable production line against the M09b base with fresh LLVM profiles,
  including validation/error paths, getters, authoring and the component-removal
  guard. Add targeted unit cases until covered/total is exactly 100%; missing
  instrumentation or unexecuted new code blocks completion.
- [x] **C7 - Document and publish.** Update standalone schema/API/authoring
  documentation and only the necessary ORT consumer-boundary guidance. Keep
  SPEC/TASKS/SCRATCHPAD current. Record commands/results, case counts, source/
  object identities, coverage counts, CPU configuration and CUDA N/A rationale.
  Inspect the repository PR template, create the documented/tested M10 PR on
  #34052 and mark ready for human review only after every gate passes. Include the
  required commit co-author trailer. Do not merge or begin M11 automatically.

#### Test matrix

| Area | Required cases and observable result |
| --- | --- |
| Structure | Missing/empty map; one/multiple chains; ordered steps; reused components; duplicate IDs; empty steps; wrong types; missing fields; invalid/NUL identifiers; unknown components. |
| Versioning | Unchanged 1.0 reads/unrelated commits; chains under 1.0 rejected; new authoring at 1.1; atomic automatic upgrades; no future-minor downgrade; unsupported majors. |
| Opaque boundary | Session/loop kinds and bindings preserved without interpretation; foreign namespaces and arbitrary payload fields preserved; inline-only new namespace envelopes; variant external-file support unchanged. |
| Inspection | Whole-map and named JSON getters; absent map gives `{}`; unknown chain errors; null/empty arguments; unchanged out-parameters; borrowed strings survive read-only calls. |
| Authoring | Whole-map/single-chain replacement; version upgrade; no-op/last-chain removal; rejected mutations leave JSON/version/views intact; referenced-component removal rejects before mutation; removal succeeds after references are removed. |
| Persistence | In-place/portable/installed PRESERVE/DENSE commit and source-independent reopen; exact model/external-data bytes and empty directories; unknown-field preservation; old-reader round trip; external file/directory components and namespace JSON; installed external variants; symlink/overlap/collision/failure state; shared-asset integrity and portable confinement. |
| C consumer/linkage | Genuine C11 compilation, every new API called, explicit nonzero check counts, shared/static linkage without ORT/ONNX/CUDA. |
| CPU EP regression | Existing `ModelPackageApiTest.*` plus a tiny-model export inference test exercise source-independent copied-model execution, ordinary selection, repeated Session creation, values/lifetimes, paths and errors using the rebuilt CPU artifact. |
| Coverage | All new executable production lines execute under unit tests; exact covered/total counts and zero uncovered new lines, not merely a green CTest summary. |

**CPU test plan:** Standalone metadata tests run on the host without any EP;
actual CPU EP compatibility is separately established by the existing ORT package
API tests. The distinction must remain explicit in evidence.
**CUDA test plan:** Device execution is N/A: no kernels, provider hooks or runtime
generation code change in M10. CPU/CUDA payload round trips are metadata tests,
not proof of GPU execution. M11 owns native execution and parity.

#### Proposed validation commands (not yet run for M10)

Use the repository's documented standalone CMake/CTest entry points. Existing
Clang/LLVM 18.1.8 and cached JSON headers are available. No dependency install is
authorized merely to prepare this plan. After approval, create isolated build/
profile directories; retain exact resolved commands and actual results here.

```bash
# From the repository root; shared standalone build with unit coverage.
ROOT="$PWD"
BUILD="$ROOT/build/state-package-m10"
JSON_DIR="$ROOT/build/state-cpu/RelWithDebInfo/_deps/nlohmann_json-src"
cmake -S model_package -B "$BUILD" \
  -DMODEL_PACKAGE_BUILD_TESTS=ON -DMODEL_PACKAGE_BUILD_SHARED=ON \
  -DNLOHMANN_JSON_DIR="$JSON_DIR" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_C_COMPILER=clang-18 -DCMAKE_CXX_COMPILER=clang++-18 \
  -DCMAKE_C_FLAGS="-fprofile-instr-generate -fcoverage-mapping" \
  -DCMAKE_CXX_FLAGS="-fprofile-instr-generate -fcoverage-mapping" \
  -DCMAKE_EXE_LINKER_FLAGS="-fprofile-instr-generate" \
  -DCMAKE_SHARED_LINKER_FLAGS="-fprofile-instr-generate"
cmake --build "$BUILD" --parallel 8
mkdir -p "$BUILD/coverage/targeted" "$BUILD/coverage/final"
LLVM_PROFILE_FILE="$BUILD/coverage/targeted/%m-%p.profraw" \
  ctest --test-dir "$BUILD" --output-on-failure \
    -R '^(inspection|authoring|chains_c)$'
LLVM_PROFILE_FILE="$BUILD/coverage/final/%m-%p.profraw" \
  ctest --test-dir "$BUILD" --output-on-failure

# After the final rebuild, use only profiles from those matching artifacts.
llvm-profdata merge -sparse "$BUILD"/coverage/final/*.profraw \
  -o "$BUILD/coverage/final/merged.profdata"
llvm-cov export "$BUILD/test_inspection" \
  -object "$BUILD/test_authoring" -object "$BUILD/test_commit" \
  -object "$BUILD/test_asset_hashing" -object "$BUILD/test_chains_c" \
  -object "$BUILD/libmodel_package.so" \
  -instr-profile="$BUILD/coverage/final/merged.profdata" \
  > "$BUILD/coverage/final/coverage.json"
ldd "$BUILD/libmodel_package.so"

# Independent static build and the same standalone suite.
cmake -S model_package -B "$ROOT/build/state-package-m10-static" \
  -DMODEL_PACKAGE_BUILD_TESTS=ON -DMODEL_PACKAGE_BUILD_SHARED=OFF \
  -DNLOHMANN_JSON_DIR="$JSON_DIR" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_C_COMPILER=clang-18 -DCMAKE_CXX_COMPILER=clang++-18
cmake --build "$ROOT/build/state-package-m10-static" --parallel 8
ctest --test-dir "$ROOT/build/state-package-m10-static" --output-on-failure

# Rebuild existing CPU ORT package regression artifact; run from its directory.
cmake --build "$ROOT/build/state-cpu/RelWithDebInfo" \
  --target onnxruntime_autoep_test --parallel 8
(
  cd "$ROOT/build/state-cpu/RelWithDebInfo"
  ./onnxruntime_autoep_test --gtest_filter='ModelPackageApiTest.*'
)
```

`chains_c` / `test_chains_c` are planned new test/target names, not existing
results. Resolve and verify every selector after implementation. Baseline CTest
currently has inspection, authoring, commit and asset_hashing; retain all four.
The coverage export must be intersected with added/changed executable production
lines against `a7a1173cd1c64d318857fa2987caf26658048509`, reporting per-file and total
covered/total counts plus any uncovered locations. Add all compiled production
objects if the implementation changes the linkage; no silent denominator
exclusions or stale-profile merges.

Build/run the frozen-parent compatibility probe in an explicitly scoped session
subdirectory using the same cached JSON dependency. Record its resolved command
and outputs when executed; do not introduce a second runtime implementation or
commit copied parent sources.

#### Completion and authorization boundary

- [x] Shared/static standalone suites and real C11 consumer pass, with nonzero
  expected case counts and no unintended exclusions.
- [x] Existing CPU ORT package API tests pass on the rebuilt artifact.
- [x] Exact new-production executable-line coverage is 100%, including helpers
  and tightly coupled changes to existing code.
- [x] Compatibility, reference-safe mutation, round-trip/path/asset and no-ORT
  dependency evidence are recorded; temporary fixtures/probes are cleaned.
- [x] Directly related docs are current and the stacked PR is ready for review.

M10 approval authorizes standalone schema/inspection/authoring and, after the
amendment approval below, the tightly bounded save-as materialization repair.
It does not authorize package execution or native generation changes. In
particular, sampling-free prefill and decode's first sample from `initial_logits`
remain M11 work. Before implementing M11, refine budget/EOS/Step accounting and
the native/public API behavior, preserving existing prefill-sampling contracts.

**Implementation checkpoint (not final qualification):** The unchanged
instrumented standalone baseline passed 4/4 CTest entries. New shared-build
inspection/authoring/C11 tests pass, but the full shared/static suites fail the
new save-as round-trip case: variant/model directories are not copied. The same
defect was reproduced with the frozen M09b parent. The genuine older reader
separately preserved schema-1.1 chains in a metadata-only commit. Existing rebuilt
CPU `ModelPackageApiTest.*` passed 16/16. No final coverage claim is made.
The user chose to expand scope; J.9 and a revised detailed plan must be approved
before implementing the repair. Do not weaken the failed persistence test.

#### Save-as repair plan amendment (approved 2026-10-19)

**Outcome:** A destination package contains its model/variant files and can
reopen/infer independently of the source package, satisfying SPEC J.9.
**Scope:** Existing standalone commit/copy/path helpers, directly coupled unit
tests, one CPU package integration test and related documentation. Remain in the
M10 PR because working chain persistence is its completion requirement. No new
schema field, public API, dependency, model parser or GPU code.

- [x] Reproduce missing-directory save-as failure with the frozen parent.
- [x] Review and obtain approval of SPEC J.9.
- [x] Review this detailed amendment and obtain implementation authorization.
- [x] **D1 - Prepare a confined export snapshot.** Before live-state changes,
  resolve the destination and preflight empty-destination, source/destination
  overlap and structural output conflicts. Build a deterministic package-local
  export layout from component/variant indices, not user identifiers as paths.
  Rebase known structural component/variant paths in the snapshot so installed
  external paths become local without rewriting opaque consumer payloads.
  PRESERVE retains inline/external shapes, not necessarily original filenames.
- [x] **D2 - Materialize and reanchor.** Copy every resolved variant tree using
  no-follow regular-file/directory semantics, including nested data and empty
  directories. Reuse the existing copy helper, correcting tightly coupled
  filesystem error reporting where necessary. Materialize external namespace
  JSON at confined collision-safe paths within each variant; preserve its
  parsed payload. Emit external component `component.json` records correctly.
  In DENSE mode, rebase variant anchors when flattening external components;
  retain the existing external-namespace rejection rather than silently inlining.
  Reparse the exported snapshot and swap live state only after success.
- [x] **D3 - Qualify success and failure paths.** Keep the original failing
  real-file chain save-as test. Add directory-form external components, installed
  external directories/JSON, nested model data, empty directories, missing/
  unresolved draft variants and multiple components/variants. Make the source
  unavailable before reopening destination and compare bytes and payloads.
  Test symlinks/unsupported objects, overlapping copies, metadata/model path
  collisions and injected/controlled filesystem failures; assert unchanged live
  root, version, JSON/views and source bytes on rejected/failed exports.
- [x] **D4 - Prove actual consumption and coverage.** Add a CPU ORT package API
  test exporting an existing tiny local ONNX fixture, make the source package
  unavailable, create a Session from the export and check its inference result.
  Repeat shared/static standalone suites and existing CPU package regressions.
  Include commit/copy/rebase helper/error lines in fresh LLVM new-line coverage,
  adding focused unit tests until measured 100%. Record the parent negative
  control separately from repaired behavior and document structural-path rebasing
  plus the opaque-payload file-discovery boundary.

**Dependencies:** D1 requires amendment approval and C3; D2 requires D1; D3
requires D2; D4 requires D3. C4 is complete only after D4 and its compatibility
checks. C6/C7 remain gated on completed C4/C5. Existing proposed shared/static/
CPU commands apply unchanged; extend the coverage source list to include
`model_package/src/commit_prune.cc` and any other changed production helper.
Keep per-file source/object/profile identity and zero uncovered new executable
lines. GPU execution is N/A for these file operations.

#### Qualification checkpoint

Final shared and static standalone suites both pass **5/5** CTest entries:
inspection **25**, asset hashing **15**, authoring **26**, commit **20** internal
cases, plus a genuine C11 consumer with **15** explicit checks across all five
new APIs. No test is skipped. Variant bytes, nested external data and empty
directories survive source-independent PRESERVE/DENSE exports in both layouts.
Installed external component/metadata copying, re-export after namespace removal,
in-place dense anchors and failed-copy state preservation pass.

Fresh LLVM unit profiles in
`build/state-package-m10/coverage/final-verified-2/` measure **282/282** new
executable production lines against M09b `a7a1173cd1`:

| Production source | Covered / new executable lines |
| --- | --- |
| `manifest_parser.cc` | 70/70 |
| `model_package_impl.cc` | 29/29 |
| `authoring.cc` | 60/60 |
| `commit_prune.cc` | 123/123 |

The diagnostic dump identifies exactly three dormant hash-zero nlohmann
input-stream adapter maps. No authored function has mismatched profile data;
no new production line is excluded. This is executable-line coverage, not a
branch-coverage claim. `ldd` on the standalone shared library and static C11
consumer confirms no ORT/ONNX/CUDA dependency.

The frozen original M09b reader also opens/in-place-commits the repaired
schema-1.1 export, preserving chains and copied bytes. Its earlier negative
control still reproduces the original missing-model-directory save-as defect.
The rebuilt CPU package API suite passed **17/17**, including
`ExportedModelsRunWithoutSourcePackage`, which runs actual square-model inference
from PRESERVE and DENSE exports after the original source path becomes
unavailable. No reported test skips. A final incremental rebuild/rerun confirmed
the tested artifact includes the last source changes.

Executed qualification commands (all **passed**, except the deliberately failing
frozen-parent negative control described above):

```bash
cmake --build build/state-package-m10 --parallel 8
LLVM_PROFILE_FILE="$PWD/build/state-package-m10/coverage/final-verified-2/%m-%p.profraw" \
  ctest --test-dir build/state-package-m10 --output-on-failure -V
llvm-profdata merge -sparse \
  build/state-package-m10/coverage/final-verified-2/*.profraw \
  -o build/state-package-m10/coverage/final-verified-2/merged.profdata
llvm-cov export build/state-package-m10/libmodel_package.so \
  -instr-profile=build/state-package-m10/coverage/final-verified-2/merged.profdata \
  > build/state-package-m10/coverage/final-verified-2/coverage.json
cmake --build build/state-package-m10-static --parallel 8
ctest --test-dir build/state-package-m10-static --output-on-failure -V
ldd build/state-package-m10/libmodel_package.so
ldd build/state-package-m10-static/test_chains_c
cmake --build build/state-cpu/RelWithDebInfo \
  --target onnxruntime_autoep_test --parallel 8
# From build/state-cpu/RelWithDebInfo:
./onnxruntime_autoep_test --gtest_filter='ModelPackageApiTest.*'
```

Configure options are recorded in the command section above; actual builds used
cached nlohmann/json headers and Clang/LLVM18. No dependency install was needed.
The source-distributed library's schema is 1.1; no ORT API version was bumped.
Changed-range clang-format and `git diff --check` passed. The embedded SPEC JSON
still exactly matches the reviewed candidate and all 24 macro milestones remain.
CUDA execution/device coverage are N/A; no new device code or runtime routing.

Exact logs, parent-probe/coverage script sources, profile diagnostics and artifact
hashes are retained in session files as `m10-final-*.log`,
`m10-cpu-final-*.log`, `m10-parent-*.log`, `m10-host-coverage-final.txt`,
`m10-verified-artifact-hashes.txt` and `measure_m10_coverage.py`. Final LLVM
exports/profiles remain in the standalone build tree. No production line was
waived for tool availability. Implementation `7f48009fb30054f931624a2068ca7d2bf0518088`
is published as ready-for-review PR #34053, stacked on verified M09b PR #34052.
All M10 gates pass. Temporary compatibility build/source copies and probe
binaries were cleaned; source probes, reports, profiles and hashes remain.
No PR was merged and M11 has not begun.

### M11 - ORT package-to-chain loading

**Status:** Complete; CPU/GPU qualified and published for review on 2026-10-21.
**Specification:** [Approved Appendix K](SPEC.md#appendix-k-m11-executable-package-chains-approved).
**Branch/base:** `stateful-M15-package-chain-loading` on verified M10
`b03b4c2f72955e869401be4d8eaa277728dc6fa8` / PR #34053.
**Delivery:** ONE independently documented/tested PR, explicitly requested by the
user. Internal tasks are not an M11a/M11b PR split.
**PR:** [#34054](https://github.com/microsoft/onnxruntime/pull/34054), OPEN and ready for
review, stacked on #34053. GPU qualification commit `3e1124c6eb91e0325d58084d742b482d37275248`.
**Outcome:** Package-built and equivalent programmatic chains share the native
runtime and C/C++ execution/result surface, including sampling-free prefill and
an observable sample-only first decode Step.

#### Approval and environment gates

- [x] Interview the user and clarify decode loop versus ONNX-model inputs.
- [x] Confirm no additional requirements before specification drafting.
- [x] Draft/review SPEC Appendix K and obtain specification approval.
- [x] Prepare this detailed CPU-first/GPU-blocked plan.
- [x] Review this plan and obtain separate implementation authorization.
- [x] Complete native/API/loader/CPU validation tasks.
- [x] Restore the CUDA prerequisite and pass real GPU qualification.
- [x] Pass full new-line coverage/regression/publication gates.

**Historical GPU blocker:** On 2026-10-20, all eight A100 GPUs were visible, but
the CUDA13.1/cuDNN installation referenced by both existing GPU build caches was
absent. `/usr/local/cuda` was also a broken link; `nvcc`, profilers and Compute
Sanitizer were not available on PATH or at the previously used locations.
The user chose to finish the plan with **CPU work first and GPU qualification
blocked until tooling is restored**. No toolchain installation or global
CUDA/driver change is authorized by that choice.

Old GPU binaries do not validate new headers/API tables/routing. Do not run a
new consumer against those stale tables or claim a cached artifact passed M11.
CPU work can proceed after plan approval, but M11 remains incomplete/not ready
until rebuilt, genuine CUDA tests and GPU-dependent coverage pass. If a review
draft is requested before then, clearly label it blocked; do not mark the
milestone complete or publish it ready by waiving GPU tests.

**Restoration checkpoint (2026-10-21):** The user supplied CUDA13.0.48 at
`/home/tianlei/cuda13.0`, matching cuDNN9.23.1 at
`/home/tianlei/cudnn_9.23_cuda13`, Nsight Compute2025.3, Nsight Systems2025.3.2
and Compute Sanitizer2025.3. Old13.1 caches remain untouched. Fresh plugin
`build/state-cuda-plugin-m11-13.0/RelWithDebInfo` and legacy
`build/state-cuda-legacy-m11-13.0/RelWithDebInfo` trees use matching SDK paths
and cached dependency **source** directories, not old dependency build products.
Clang host LLVM instrumentation stays in C/C++ flags; global profile linker flags
were removed after nvcc's GCC try-link correctly rejected them.
Both fresh configurations and builds passed. Final GPU results and coverage
closure are recorded below.

#### Tasks, dependencies and intended outcomes

| Task | Actual prerequisites | Intended result |
| --- | --- | --- |
| H0 | Restored compatible toolchain supplied by the user | Rebuildable plugin/legacy CUDA environments with matching libraries and proof tools. |
| H1 | Plan approval | Fresh M10 CPU baseline and deterministic reusable handoff/package fixtures. |
| H2 | H1 | Native sample-only bootstrap and correct Step/Run/exit/state accounting. |
| H3 | H2 | Additive programmatic C/C++ bootstrap surface with ABI/error guarantees. |
| H4 | H3 | Strict selected-chain ORT parser and owned native Session-chain construction. |
| H5 | H4 | Public package C/C++ factory and retained Session ownership. |
| H6 | H3, H5 | Complete CPU/native/public/package/lifetime/minimal-guard qualification. |
| H7 | H0, H6 | Rebuilt real CUDA-plugin parity, routing, transfer and lifetime proof. |
| H8 | H6, H7 | Measured 100% new executable lines and final compatibility regressions. |
| H9 | H8 | Updated documentation and one ready stacked M11 PR. |

- [x] **H0 - Restore and rebaseline GPU prerequisites.** Once the
  user supplies/restores tooling, resolve actual nvcc, supported host compiler,
  CUDA runtime, cuDNN, Nsight and Compute Sanitizer locations. Reuse existing
  build trees only if their compiler/library configuration is valid; otherwise
  configure fresh plugin/legacy trees with the established flags and reproduce
  relevant M08/M09 baselines first. Do not mix13.0/13.1 components, force a
  compiler into an incompatible cache, change global driver settings or install
  dependencies without the required authorization/failure basis.
- [x] **H1 - Establish CPU baseline/fixtures.** Rebuild existing focused native
  and AutoEP targets against M10 and record nonzero relevant baseline counts.
  Reuse `stateful_execution_test_utils.h`, existing session/chain and package
  fixture helpers. Add tiny local prefill/decode fixtures supporting known
  float32/float16 logits, prompt lengths, cache growth, masks/positions and a
  cache-free path. Build manifests from these same models for parity tests;
  avoid downloads, large committed weights or a second reference orchestrator.
- [x] **H2 - Implement native bootstrap and progress.** Add explicit copied
  logits/prompt source metadata to the decode component. Validate dependency,
  full-prompt shape/source, known logits metadata and cache/provider invariants.
  Derive processed-token count from the real prompt. Preserve generic
  sampler-free prefill; initialize generation once at the handoff.
  Expose a sample-only token event with no model invocation/cache mutation.
  Make chain Step return on that event rather than looping until `executed`.
  Preserve complete history, first-token EOS/limits, zero-budget prefill work,
  reset/cancellation/failure invalidation and legacy-mode behavior. Keep
  generation logic under `core/generate` and wiring under the session layer.
- [x] **H3 - Wire native public C/C++.** Add the size-tagged bootstrap record
  and append `SessionChainOptionsSetGenerationBootstrap` to the stateful table/
  initializer. Add C++ Set/Clear conveniences; keep old prefixes, record
  strides, table signatures and ORT API version unchanged. Copy setters
  transactionally; NULL clears, SetConfig clears overrides. Reuse existing
  status/size/source helpers and test true old-prefix consumers, invalid sizes/
  enums/pointers, options immutability and sample-only result flags.
- [x] **H4 - Parse and construct selected chains.** Add focused ORT consumer
  implementation in `core/session/model_package`, reusing standalone chain
  JSON getters and existing component selection/Session creation.
  Centralize strict typed field/source/provider/exit/cache parsing rather than
  scattering JSON access through execution code. Resolve IDs to prior steps,
  prompt bindings from explicit `nonempty_input`/`prompt_input`, and all existing
  model/state/finite/feedback/cache/native-generation bindings. Reject missing
  exits, unknown fields/policies, scripts and incompatible roles before
  publishing an executable definition. Do not parse unrelated chains for legacy
  Session creation or turn future-minor tolerance into policy execution.
- [x] **H5 - Add package factory and ownership.** Add the frozen per-step
  SessionOptions array record, size-tagged outer config and appended
  `OrtModelPackageApi::CreateSessionChain` member/initializer, with C++ context
  wrapper returning the existing SessionChain. Default/per-step options are
  complete snapshots; reuse default/advanced package precedence exactly.
  Construct separately initialized Sessions per step. Own them with the
  retained native definition so package/options/chain-handle release cannot
  invalidate active executions. Roll back partial construction and preserve
  out-parameters on every failure. Document Env/external resource obligations.
- [x] **H6 - Qualify CPU and ABI/lifetimes.** Test native versus public C/C++
  versus package-created chains against identical fixtures/configurations.
  Include ordinary Sessions, non-generation loops and both generation modes;
  all binding/exit kinds; budgets0/1/N; first/later EOS; simultaneous exits;
  first-sample Step counts and no duplicate events; cache-free masks/positions;
  reset, cancellation, model/NaN errors and immutable result leases.
  Cover malformed payload permutations, output/port/type failures, per-step
  overrides on repeated components, option precedence and construction rollback.
  Release caller package/options/chain owners before continued execution and
  retain outputs after teardown. Exercise C11, C++17 and no-exceptions clients.
  Execute any new minimal-build unsupported stub using a genuine minimal
  configuration or isolated compilation of that actual production source, not
  a replacement implementation; preserve outputs and include its lines in
  coverage. Rerun single-component package APIs and standalone shared/static
  suites, including source-independent M10 exports.
- [x] **H7 - Qualify CUDA after H0.** Route initial prefill logits on the
  compatible GPU using the actual producer allocator/provider origin and
  initialized Sessions. Prove real plugin EP model/greedy/cache dispatch,
  independent cache/sampler policies, nondefault/distinct compatible streams,
  mismatched-device/CPU-producer rejection and no hidden full-logits migration.
  Include cached/cache-free paths, mixed ordinary/loop steps, retained GPU values,
  library unregistration and all-owner release. Verify budget0 performs prefill
  work but zero sampler calls; budget1/first EOS performs no decode-model call.
  Trace a dedicated end-to-end package Run, separating compact token/status
  readback from model feeds/weights and explicit caller-copy diagnostics.
  Run applicable Compute Sanitizer memory/lifetime/ordering tests. Preserve
  existing M08 timing-only sanitizer exclusion narrowly; do not suppress a new
  memory error, broaden the exclusion or infer universal stream serialization.
- [x] **H8 - Close coverage and regressions.** Instrument every new production
  native/loader/parser/API/C++ inline/glue line with matching artifacts. Use
  fresh per-module/per-configuration profiles and verify source/function maps.
  Combine CPU/plugin/minimal evidence at source-line level only after mapping
  verification; never blindly merge incompatible hashes. Add focused unit
  cases until covered/new executable totals are exactly100%.
  No new device algorithm is intended: verify reused greedy device-code identity
  and conformance; if implementation changes, measure every new device line.
  Rerun complete focused CPU/plugin native and AutoEP suites; use the documented
  legacy selector rather than claiming an inherited unfiltered legacy pass.
- [x] **H9 - Document and publish one PR.** Update public API ownership/Step
  documentation, ORT package grammar/loading examples, SPEC/TASKS/SCRATCHPAD and
  the exact local build/test/coverage/dispatch/transfer evidence. Preserve the
  repository PR template and required commit co-author trailer. Publish one
  independently verified M11 PR stacked on #34053 only after H8 passes. Do not
  merge automatically or begin M12.

#### Required test observations

| Area | Exact behavior to verify |
| --- | --- |
| New-mode Steps | Prefill invokes its model; next decode Step emits token1 with executed=false, has_generated_token=true, iterations0 and unchanged invocation count; later Steps invoke at most one model. |
| Budgets/EOS | N=0: prefill1/decode0/samples0/tokens0. N=1: prefill1/decode0/samples1/tokens1. N>1 without other exits: decode N-1, samples N, tokens N. First EOS stops before any decode Run. |
| Prompt/context | Validate nonempty CPU int64[1,S], full-prompt sequence count1, retained prompt source, mask/position offsets and shared-cache logical length; no inference of S from logits. |
| Outputs/errors | No duplicate completion/token events, no substituted/unproduced model ports, stable retained views; explicit native-equivalent errors and reset requirement after advancement. |
| Loader/options | Explicit named selection, no default; selected-chain-only ORT validation; all supported source/exit/cache policies; full per-step options replacement; existing package option/path restrictions. |
| Ownership | Executions survive package/options/chain-handle release; package-created Sessions retained correctly; caller Sessions not adopted; Env/external-buffer obligations unchanged. |
| Compatibility | Frozen prefixes/array strides, old generation zero-budget behavior, single-component APIs, minimal unsupported path and standalone shared/static independence. |
| CUDA | Actual selected CUDA variant and callback/device placement, sample-only first call, no full-logits readback, correct streams/device restoration and post-unregistration resource leases. |
| Coverage | 100% of all new executable production host/glue/wrapper/minimal lines; separately measured device additions if any. Missing tooling leaves gates blocked, never green. |

#### Proposed commands (not M11 validation results)

Run from the repository root unless a subshell changes directories. The tests
named Bootstrap/ModelPackageChain below are planned selectors; discover and
verify actual nonzero counts before treating any command as evidence.

```bash
source .venv/bin/activate
ROOT="$PWD"
CPU="$ROOT/build/state-cpu/RelWithDebInfo"
cmake --build "$CPU" \
  --target onnxruntime_session_loop_test onnxruntime_autoep_test --parallel 8
mkdir -p "$CPU/coverage/m11/baseline" "$CPU/coverage/m11/targeted"
(
  cd "$CPU"
  LLVM_PROFILE_FILE="$CPU/coverage/m11/baseline/%m-%p.profraw" \
    ./onnxruntime_session_loop_test
  LLVM_PROFILE_FILE="$CPU/coverage/m11/baseline/%m-%p.profraw" \
    ./onnxruntime_autoep_test --gtest_filter='*Stateful*:*ModelPackage*'
)

# After coherent source/header/API edits; regenerate for new test sources.
cmake -S cmake -B "$CPU"
cmake --build "$CPU" \
  --target onnxruntime_session_loop_test onnxruntime_autoep_test --parallel 8
(
  cd "$CPU"
  LLVM_PROFILE_FILE="$CPU/coverage/m11/targeted/%m-%p.profraw" \
    ./onnxruntime_session_loop_test --gtest_filter='*Bootstrap*'
  LLVM_PROFILE_FILE="$CPU/coverage/m11/targeted/%m-%p.profraw" \
    ./onnxruntime_autoep_test --gtest_filter='*Bootstrap*:*ModelPackageChain*'
)

# Use a fresh final directory after the final rebuild.
mkdir -p "$CPU/coverage/m11/final"
(
  cd "$CPU"
  LLVM_PROFILE_FILE="$CPU/coverage/m11/final/%m-%p.profraw" \
    ./onnxruntime_session_loop_test
  LLVM_PROFILE_FILE="$CPU/coverage/m11/final/%m-%p.profraw" \
    ./onnxruntime_autoep_test
)
llvm-profdata merge -sparse "$CPU"/coverage/m11/final/*.profraw \
  -o "$CPU/coverage/m11/final/merged.profdata"
llvm-cov export "$CPU/onnxruntime_session_loop_test" \
  -object "$CPU/onnxruntime_autoep_test" -object "$CPU/libonnxruntime.so" \
  -instr-profile="$CPU/coverage/m11/final/merged.profdata" \
  > "$CPU/coverage/m11/final/coverage.json"

cmake --build build/state-package-m10 --parallel 8
ctest --test-dir build/state-package-m10 --output-on-failure
cmake --build build/state-package-m10-static --parallel 8
ctest --test-dir build/state-package-m10-static --output-on-failure
```

Batch public-header/API/formatting edits before expensive rebuilds. CMake's
session-source glob uses CONFIGURE_DEPENDS, but explicitly regenerate when adding
AutoEP test files/build configuration. Preserve existing compiler/coverage flags.
Resolve the exact C11/C++17/no-exceptions and minimal-source consumer commands
from the final build targets; record them and their results, not placeholders.

The following is **blocked/not run**, and only becomes applicable after H0
resolves compatible `GPU_BUILD`, `LEGACY_BUILD`, loader-library and proof-tool
paths. Do not reuse obsolete compiler/runtime cache entries:

```bash
cmake --build "$GPU_BUILD" \
  --target onnxruntime_session_loop_test onnxruntime_autoep_test --parallel 8
(
  cd "$GPU_BUILD"
  ./onnxruntime_session_loop_test
  ./onnxruntime_autoep_test
)
cmake --build "$LEGACY_BUILD" \
  --target onnxruntime_session_loop_test onnxruntime_autoep_test --parallel 8
(
  cd "$LEGACY_BUILD"
  ./onnxruntime_session_loop_test
  ./onnxruntime_autoep_test \
    --gtest_filter='AutoEpSelection.CudaEP:*Stateful*:ModelPackageApiTest.*:*ModelPackageChain*'
)
```

Use fresh plugin LLVM profiles and the actual plugin/host objects for coverage.
Run all available-device conformance before restricting a dedicated profiler
case to one selected GPU; do not hide multidevice coverage behind visibility
settings. Respect the shared machine's active workloads. Resolve explicit
Nsight/Compute Sanitizer commands and verified selectors after tooling returns.
The existing administrator authorization is for profiling only, not driver
settings, permission changes or environment-wide installation.

Retain the inherited legacy-contrib/EPContext and sanitizer timing limitations
with their exact scopes; neither is a waiver for M11 regressions. A new hardware/
tool limitation blocks its gate and must be disclosed with the failed command.

#### Completion gates

- [x] Native and public bootstrap semantics/old-mode regressions pass on CPU.
- [x] Package/programmatic C/C++ parity, ownership, option precedence and error
  validation pass with actual local models and source-independent exports.
- [x] Genuine C11/C++17/no-exceptions clients, frozen prefixes and new minimal
  unsupported path are verified; standalone library remains ORT-independent.
- [x] Compatible toolchain is restored; rebuilt CUDA-plugin and selected legacy
  suites pass with real dispatch/placement/transfer/lifetime evidence.
- [x] Every new executable production line reaches measured100% across verified
  CPU/plugin/minimal artifacts; device additions independently measured if any.
- [x] Documentation/evidence are current and one stacked PR is ready for review.

Until every gate passes, M11 is not complete. Approving this plan authorizes
CPU-first implementation and the declared scope, but does not authorize removing
the GPU gate, an unrequested toolchain install, a second milestone PR, merging,
or beginning M12.

**CPU implementation checkpoint:** Fresh M10 native baseline63/63 and focused
public baseline52/52 passed. Native bootstrap is implemented and verified by
three new tests (including budget/dtype/cache/source matrices); the full native
suite passes66/66. Sample-only events stop chain Step without publishing model
outputs or updating invocation/cache bookkeeping. New C/C++ setters, strict ORT
parser, owned-session package factory and positive public/package tests are
being compiled/qualified; they are not yet a passed API/loader gate.
GPU tooling, complete coverage and milestone publication remain blocked/pending.

#### Verified CPU checkpoint (GPU completion still blocked)

The CPU implementation is functional and H1-H6 are verified. Full native
`onnxruntime_session_loop_test` passes **69/69**; unfiltered
`onnxruntime_autoep_test` passes **148/148**, with no reported skips.
The new tests include:

- Sample-only initial decode Steps, budgets0/1/3, first-token EOS, masks/positions,
  float32/float16, cache-free/cached models, all prompt source kinds, reset and
  explicit malformed/configuration/runtime errors.
- Actual package/programmatic parity, ordinary Sessions and feedback/Boolean
  loops, named state, explicit plain-Session prompt handoff, cache Share/Copy/
  Append mappings and actual CPU-provider dispatch.
- Package/context/options/chain-handle release before continued execution,
  source-independent model1/model2 exports, partial-construction rollback and
  unchanged out-parameters. Per-step snapshots and fixed array strides are tested.
- A genuine C11 package consumer, independent C++17/no-exceptions consumers, and
  a client compiled against genuine frozen M10 headers using the new runtime's
  old table prefixes. Standalone shared/static package suites both pass5/5.
- An isolated compile of the **actual** package API production source with
  ORT_MINIMAL_BUILD executes the new factory's NOT_IMPLEMENTED path and preserves
  a non-NULL out sentinel. The harness forwards only generic status creation to
  the ordinary runtime; it does not replace the factory or its guard logic.

Fresh CPU and separately mapped minimal-source profiles cover **737/743** new
executable production lines against M10 `b03b4c2f72`:

| Source | Covered / new executable lines |
| --- | --- |
| `core/generate/token_generation.cc` | 9/9 |
| `core/session/session_loop.cc` | 37/37 |
| `core/session/session_chain.cc` | 129/135 |
| `core/session/stateful_api.cc` | 45/45 |
| `core/session/model_package_api.cc` | 90/90 |
| `core/session/model_package/model_package_context.cc` | 14/14 |
| `core/session/model_package/model_package_chain.cc` | 400/400 |
| Public C++ inline wrappers | 13/13 |

All six uncovered locations are the actual GPU-origin bootstrap routing prepass
in `session_chain.cc` (currently lines180-185). They remain in the denominator,
including mapped braces; no fake provider, exclusion, CPU fallback or rounded
100% claim substitutes for real GPU execution. H0/H7/H8/H9 remain blocked.
The profile audit found358 dormant hash-zero maps across loaded inline/helper
instantiations and **no nonzero function-hash mismatch**. Active production maps
match their artifacts; minimal/nonminimal mappings are evaluated separately.

Executed CPU commands/results:

```bash
cmake --build build/state-cpu/RelWithDebInfo \
  --target onnxruntime_session_loop_test onnxruntime_autoep_test --parallel 8
# From build/state-cpu/RelWithDebInfo; each command passed:
LLVM_PROFILE_FILE="$PWD/coverage/m11/cpu-final-3/%m-%p.profraw" \
  ./onnxruntime_session_loop_test
LLVM_PROFILE_FILE="$PWD/coverage/m11/cpu-final-3/%m-%p.profraw" \
  ./onnxruntime_autoep_test
llvm-profdata merge -sparse coverage/m11/cpu-final-3/*.profraw \
  -o coverage/m11/cpu-final-3/merged.profdata
# From repository root; both shared/static builds and CTest suites passed:
cmake --build build/state-package-m10 --parallel 8
ctest --test-dir build/state-package-m10 --output-on-failure
cmake --build build/state-package-m10-static --parallel 8
ctest --test-dir build/state-package-m10-static --output-on-failure
```

Independent SDK builds used `clang++-18 -std=c++17` with the public session
include directory and the rebuilt CPU library; the no-exceptions variant added
`-fno-exceptions -DORT_NO_EXCEPTIONS`. The frozen-header client used
`clang-18 -std=c11`. All consumers passed against the local generated tiny
package, including after owner release/reset. The minimal compile uses the
actual Session target's flags.make/compiler because the existing compilation
database omitted that target; this fallback is explicitly logged.

Exact consumer/generator/minimal/measurement sources, commands, logs, profile
diagnostics and source/object/report hashes are retained in session files with
the `m11-*` prefix. CPU profiles are in `coverage/m11/cpu-final-3`; the measured
minimal object/probe/profiles are retained separately. The final public-header
follow-up is Doxygen-only and does not change code/record layout.

No toolchain dependency was installed and no GPU test was claimed. Preserve
this tested CPU checkpoint on the M11 branch, but do not mark M11 complete or
publish a ready PR until compatible CUDA tooling and all remaining gates pass.
CPU checkpoint `2f3e6c0237` is committed and pushed on
`stateful-M15-package-chain-loading`, stacked on #34053. At that checkpoint no M11 PR existed;
the requested single-PR delivery remained gated on GPU/tooling and737/743 coverage
closure. Non-measured SDK binaries, generated fixture and frozen header copies
were cleaned; reproducible sources/logs and measured CPU/minimal artifacts remain.

#### Final GPU qualification and coverage closure

The user restored compatible tools and authorized completion on 2026-10-21.
Fresh CUDA13.0.48/cuDNN9.23.1 builds, not stale13.1 artifacts, are verified:

| Configuration | Native tests | Public/AutoEP tests |
| --- | --- | --- |
| CPU, Clang18 | 69/69 | 148/148 unfiltered |
| CUDA plugin, Clang18 host/NVCC13.0/GCC13 | 132/132 | 204/204 unfiltered |
| Legacy CUDA, GCC13 | 120/120 | 69/69 selected |

No reported skips. The legacy selector retains its documented scope rather than
claiming an unfiltered legacy/contrib/EPContext pass. The first fresh legacy
public run lacked required repository testdata; staging those assets and
rerunning all69 resolved the setup failure without exclusions.

Four new `CudaBootstrapPackageTest` cases pass, including **72** cache/sampler/
dtype/budget/EOS combinations against both programmatic and CPU references,
distinct nonblocking Session streams, cached/cache-free generation, plain Session
prefill, first-token EOS, cancellation, NaN failure, CPU-producer/mismatched-GPU
rejection, and retained values after all owned Sessions/library registration.
CPU variant files intentionally use different token scores, so accidental
selection of a CPU variant cannot satisfy the CUDA expected-token reference.

The initial unfiltered native run exposed eight **test harness** device
mismatches: the generic dynamic-EP infrastructure's first enumerated CUDA device
was not ordinal0, while native CUDA fixtures explicitly request0. The native
helper now honors its requested ordinal from configured devices and forwards
its supplied compute stream. Runtime device validation was not relaxed.
A new device1/stream/invalid-ordinal regression and all previously failing
fixtures pass; final unfiltered native result is132/132.

Full source-mapped CPU/plugin/minimal unit evidence closes **743/743 (100%)**
new executable production lines. Chain routing is now135/135, including the
previously uncovered six GPU prepass locations; other per-file totals above are
unchanged. CPU profiles use `coverage/m11/final-after-harness`, plugin profiles
use `coverage/m11/final-2`, and minimal-source evidence is mapped separately.
There are no nonzero function-hash mismatches;403 plugin,358 CPU and17 minimal
dormant hash-zero inline/helper maps are explicitly audited, not hidden.

No new CUDA function/source was authored. Greedy source SHA remains
`92837bc204a43c0e2ca0f7da8b5f0a037d5a6200d572a235412d9530e587cbcd`.
Because the compiler changed, actual rebuilt-device Nsight SourceCounters were
collected again: **25/25 emitted source locations executed**, including mapped
entry/epilogue braces, with no location excluded. Unprivileged collection failed
ERR_NVGPUCTRPERM; the existing administrator authorization was used only for the
profiler process, with no global driver/permission change.

The dedicated real package Run at vocabulary257 has **3 DtoH copies of16 bytes,
48 bytes total**, matching its three native greedy callbacks. No2056-byte
prefill or1028-byte decoder logits readback. HtoD is separately **5 copies,
8256 bytes** for weights/CPU token feeds. This is not a device-only generation
or release-performance claim.

Compute Sanitizer qualification:

- New public GPU4 tests, new native/explicit-device9 tests and standalone public
  C++17 GPU SDK: memcheck **zero errors and zero leaked bytes**.
- Dedicated package-generation racecheck: **zero hazards/errors/warnings**.
- Dedicated package-generation synccheck: **zero errors**.
- The inherited M08 timing-only memcheck limitation remains narrow/unchanged;
  no new test or memory error was excluded.
- Genuine C11 GPU package construction/execution, public C++17 and no-exceptions
  GPU clients pass, including reset, owner release, library unregistration and
  explicit access to retained GPU logits afterwards.

Exact final commands/results (run tests from output directories with matching
CUDA/cuDNN loader paths):

```bash
# Fresh plugin output directory; both passed132/204:
LLVM_PROFILE_FILE="$PWD/coverage/m11/final-2/%m-%p.profraw" \
  ./onnxruntime_session_loop_test
LLVM_PROFILE_FILE="$PWD/coverage/m11/final-2/%m-%p.profraw" \
  ./onnxruntime_autoep_test
# Fresh legacy output directory; passed120/69:
./onnxruntime_session_loop_test
./onnxruntime_autoep_test \
  --gtest_filter='AutoEpSelection.CudaEP:*Stateful*:ModelPackageApiTest.*:*ModelPackageChain*'
# CUDA13.0 tools, from plugin output directory; all passed:
compute-sanitizer --tool memcheck --leak-check full --error-exitcode 99 \
  ./onnxruntime_autoep_test --gtest_filter='CudaBootstrapPackageTest.*'
compute-sanitizer --tool memcheck --leak-check full --error-exitcode 99 \
  ./onnxruntime_session_loop_test \
  --gtest_filter='CudaCacheIntegrationTest.NativeFactoryHonorsExplicitOrdinalAndStream:*Bootstrap*'
compute-sanitizer --tool racecheck --error-exitcode 99 \
  ./onnxruntime_autoep_test --gtest_filter='CudaBootstrapPackageTest.BootstrapPackageTransfer'
compute-sanitizer --tool synccheck --error-exitcode 99 \
  ./onnxruntime_autoep_test --gtest_filter='CudaBootstrapPackageTest.BootstrapPackageTransfer'
```

Actual tool paths, matching library env, complete SDK/profiler arguments and
configure options are retained in `m11_configure_gpu.py`, `m11-gpu-*`,
`m11-helper-*`, `m11-legacy-final-*`, `m11-bootstrap-transfer*`,
`m11-greedy-device*`, `m11-host-coverage-final.txt`,
`m11-device-coverage-final.txt`, profile diagnostics and artifact hashes in
session files. The actual CUDA plugin filename is
`libonnxruntime_providers_cuda.so`; the first SDK attempt used an incorrect
`_plugin` suffix and was corrected before successful runs.

All validation gates pass. H9 is complete:
[#34054](https://github.com/microsoft/onnxruntime/pull/34054) is OPEN, not draft,
and uses the repository template with exact validation commands/results.
The base is `stateful-M14-package-chain-schema` / #34053; no extra milestone PR was
created. No M12 work or automatic merge is authorized.
Pinned lintrunner/clang-format20.1.8 checks and `git diff --check` pass.
The initially missing lint adapters were restored from the repository's pinned
requirements in the existing virtual environment; no GPU tooling was installed
by the assistant. Non-measured SDK executables/fixture and the compiler-probe
profile were cleaned after hashing; measured builds/profiles and proof reports
remain.

### M12 - DDIM scheduler and denoising execution policy

**Status:** Complete; CPU/CUDA qualified and published for review on 2026-10-31.
**Specification:** [Approved Appendix L](SPEC.md#appendix-l-m12-ddim-denoising-policy-approved).
**Proposed branch/base:** `stateful-M16-ddim-policy` on verified M11
`0586ee3acfc1456d1d48bdd060e4789eb0b99c98` /
[#34054](https://github.com/microsoft/onnxruntime/pull/34054).
**Delivery:** One intended, independently documented/tested stacked PR; no merge.
**PR:** [#34055](https://github.com/microsoft/onnxruntime/pull/34055), OPEN and ready for
review on the M11 base. Qualified implementation commit
`ae416c8708513316be6e59ae74faade73a105fd4`.
**Outcome:** Stateful deterministic DDIM iteration through the existing native
loop/chain and public C/C++/package surfaces, not LLM-specific cache emulation.
The full Stable Diffusion package/image demonstration remains M15.

#### Approval and current prerequisites

- [x] Interview the user about reference, guidance, initialization, CPU/CUDA scope,
  precision, batch, Step/reset/cancellation, exits and per-execution controls.
- [x] Explicitly confirm no additional requirements before specification drafting.
- [x] Draft/review Appendix L and obtain specification approval (2026-10-30).
- [x] Prepare this detailed plan from the approved specification.
- [x] Review the detailed plan and obtain separate implementation approval.
- [x] Implement, qualify and publish only after that approval.

Read-only inspection on 2026-10-30 confirms clean M11 baseline/remote head,
CUDA13.0.48, the matching cuDNN installation, profiler/sanitizer tools and eight
idle A100 GPUs. Existing CPU/plugin builds use Clang/LLVM instrumentation;
legacy CUDA uses GCC. These are prerequisites, **not M12 build/test results**.
Recheck availability and shared-machine workloads when implementation starts.

Root filesystem free space is about21GB; `/home` has about122GB free. Preserve
matching M11 measured binaries/libraries/profiles/device objects before reusing
their build trees. Prefer a bounded artifact archive plus incremental builds
over copying complete build/dependency trees. Leave old13.1 trees untouched.
Do not install GPU tooling, alter global driver settings or delete other work.

#### Tasks, dependencies, outcomes and tests

| Task | Actual prerequisites | Intended outcome |
| --- | --- | --- |
| D0 | Plan approval | Verified branch/base, preserved M11 proof, usable CPU/CUDA/reference tools and fresh baseline results. |
| D1 | D0 | Independent numerical fixtures and reusable CPU DDIM/guidance/initialization core. |
| D2 | D1 | Native loop policy, complete-update Step/reset semantics, logical ports and accurate invocation counts. |
| D3 | D2 | Chain validation/wiring/state/output publication and downstream latent handoff. |
| D4 | D2, D3 | Append-only public configuration and equivalent C/C++ execution. |
| D5 | D3, D4 | Strict declarative package policy using the same native implementation/owned Sessions. |
| D6 | D4, D5 | Complete CPU native/public/package/SDK/error/lifetime qualification. |
| D7 | D0, D6 | Real CUDA denoiser parity, both guidance paths, stream/transfer/lifetime evidence. |
| D8 | D6, D7 | Measured100% new executable production lines and compatibility regression closure. |
| D9 | D8 | Current documentation/evidence and one ready stacked PR on M11. |

- [x] **D0 - Establish and preserve the baseline.** Create the proposed branch
  from the verified M11 head after approval; inspect applicable instructions and
  current tools/configuration. Archive only required matching M11 host binaries,
  shared libraries, profiles, device proof/object and hashes before rebuilding;
  retain source identity via the commit. Reuse valid build caches/dependency
  sources, not obsolete13.1 artifacts. Stage required testdata in fresh output
  trees if needed.
  **Tests:** Record fresh CPU native/public, plugin native/unfiltered-public and
  legacy native/selected-public baseline counts, nonzero execution and skips.
  Verify loader paths, actual CUDA ordinal/stream selection and tool versions.
  Establish the pinned numerical-reference command in the isolated environment;
  restore missing packages only after a genuine missing-dependency failure.

- [x] **D1 - Implement and independently verify the numerical core.** Add
  internal `core/diffusion/` options/state and CPU math using existing ORT tensor,
  allocation, status and checked-arithmetic patterns. Implement the fixed
  `sd15_ddim_v1` profile: scaled-linear coefficients, leading-offset1 schedule,
  epsilon/eta0 update, private MT19937_64/Box-Muller initialization and both
  guidance input/combination helpers. Keep reusable math separate from Session
  orchestration; do not introduce a general policy/plugin framework.
  **Tests:** Independent analytical one-step cases and pinned Diffusers
  v0.35.1 trajectories from supplied latents; coefficient/intermediate checkpoints;
  schedule1/2/50/999 and invalid0/negative/1000; g=0/1/>1; batch>1/nontrivial
  dimensions; seed0/largeuint64, odd-size normal generation, independent RNG
  contexts and exact same-build replay. Cover NaN/Inf, enum/rank/dimension errors
  and arithmetic overflow without attempting real OOM allocations.

- [x] **D2 - Integrate native SessionLoop execution.** Add optional DDIM options
  and per-execution state. Validate explicitly named model ports, four timestep
  formats, managed inputs and logical initialization/conditioning feeds.
  Accept only CPUuint64scalar seed or matching CPUFP32NCHW initialization;
  normalize neither invalid dtypes nor GPU logical feeds silently.
  Support exactly the explicit matching MaxIterations exit and reject conflicting
  token/cache/feedback/finite-input features. Keep the existing generic/token
  paths unchanged.
  **Tests:** One Step performs one complete update and1/2 actual calls, preserving
  x across sequential guidance calls. Publish only validated CPU latent snapshots;
  distinguish them from raw prediction outputs. Cover pre-Step initialization,
  all timestep encodings, both guidance modes, state inspection, completion-only
  Step, reset, boundary cancellation and controlled failures in either call.
  Prior results must remain unchanged after advancement/failure/Reset/release.

- [x] **D3 - Wire every chain surface.** Extend loop/chain port validation only
  where DDIM needs typed logical inputs and policy-produced latent outputs.
  Handle external inputs, earlier component outputs and retained-state sources,
  explicit state writers, final outputs and downstream consumers. Do not weaken
  ordinary model metadata checks or mutate the Session graph to fake a port.
  Replace the existing chain `+1 per executed Step` assumption with actual
  invocation deltas, preserving ordinary/token bootstrap counting.
  **Tests:** Tiny ordinary-conditioning producer -> DDIM loop -> ordinary latent
  consumer, both direct-output and retained-state handoff; invalid static and
  dynamic types/shapes, missing/colliding names, source order and independent
  executions. Assert k/2k DDIM invocation counts and correct whole-chain totals.
  Re-run M11 sample-only/zero-budget and ordinary mixed-component regressions.

- [x] **D4 - Add public C/C++ configuration and tests.** Add the size-tagged DDIM
  record/enums and append the setter to `OrtStatefulApi` plus its initializer.
  Add C++ Set/Clear wrappers. Copy names/configuration transactionally, clear
  DDIM on complete base-config replacement, and leave existing definitions
  unchanged. Reuse execution inputs/results; fix the standalone public
  `invocations = iterations` assumption to report actual model calls.
  **Tests:** Genuine C11 and C++17 consumers, public native-loop/chain parity,
  Set/Clear/replacement/rollback, NULL/size/enum/range/name validation, old-prefix
  ABI clients, unchanged failed out-parameters and no-exceptions-client success.
  Confirm no ORT_API_VERSION bump or old public layout/array-stride change.

- [x] **D5 - Load DDIM from Model Packages.** Parse strict `loop.ddim` fields
  into the same typed native options; retain explicit chain choice and ordinary
  package SessionOptions/variant/path precedence. Update factory and chain
  output selection so they select the logical latent output instead of
  blindly exposing all raw ONNX outputs. No second scheduler implementation.
  **Tests:** Equivalent programmatic/package trajectories, both guidance modes,
  seed/supplied initialization, logical feeds from external/earlier/state
  sources, per-step SessionOptions, repeated components and owned Session
  lifetime. Reject unknown profiles/fields, incompatible models, conflicting
  exits/policies and invalid references. Exercise save/export/reopen with actual
  tiny local ONNX files, opaque other namespaces and legacy package behavior.

- [x] **D6 - Close CPU/API/lifetime qualification.** Complete all native/public
  reference matrices and genuine standalone SDK clients. Use controlled local
  allocators/failure fixtures where necessary, not production-only test hooks,
  huge allocation attempts or global allocator mutation.
  **Tests:** CPU loop/chain/package parity, S=1/2 and50-step reference trajectories,
  cheap999-step core trajectory, both guidance branches, g0/1/>1, N1/N>1,
  four timestep formats, reset/cancel/failure and retained outputs after owner
  teardown. Verify existing minimal/unsupported factory behavior using actual
  production-source probes where applicable; do not label a source probe a
  complete minimal build. Run independent public C++17/no-exceptions and frozen
  M11-header clients, plus standalone Model Package shared/static tests.

- [x] **D7 - Qualify real CUDA denoising and transfers.** Use tiny deterministic
  FP32 denoisers with observable latent/timestep/conditioning dependence.
  Prove actual CUDA assignment/dispatch and fail the targeted CUDA case rather
  than accepting CPU fallback. Keep all scheduler/guidance/result work on CPU.
  Exercise plugin CUDA and applicable legacy CUDA Session behavior without
  advertising new cache/sampler hooks.
  **Tests:** Native/programmatic/public/package parity for batched and sequential
  guidance, N>1, seeded/supplied initialization, reset and nondefault Session
  stream. Check retained CPU latents after Session/package/registration release.
  Run focused memcheck/leak checks and applicable racecheck/synccheck.
  Dedicated trace cases must avoid caller diagnostic GPU reads and classify
  setup/weights separately from Run feeds/predictions.

- [x] **D8 - Close coverage, formatting and regressions.** Instrument all new
  production core/native/chain/parser/API/inline/glue paths. Map independently
  verified CPU/plugin/minimal evidence to the M12 diff against M11; inspect hash
  mismatches rather than merge incompatible profile layouts. Explicitly report
  executable covered/total counts and any uncovered locations.
  **Tests:** Reach100% without line exclusions, stale objects or rounded partial
  results. Report cumulative stack new-line coverage with source-safe mappings;
  re-execute modified earlier code. Final CPU/plugin native and unfiltered public,
  selected legacy compatibility, SDK/standalone tests and pinned style/whitespace
  checks must pass. No device-line gate is waived: no new device code is planned;
  a device implementation change requires revised scope/approval and coverage.

- [x] **D9 - Document and publish.** Update public Doxygen and the ORT Model
  Package integration README with profile/RNG/port names, logical initialization
  union, both guidance conventions, explicit matching exit, progress/output/state
  semantics, CPU-only policy work, CUDA transfer evidence and limitations.
  Update SPEC/TASKS/SCRATCHPAD status and exact commands/results.
  **Tests:** Review documentation examples against executed tiny fixtures,
  inspect/preserve the repository PR template and verify the one ready/open PR
  uses the M11 base and synchronized tested head. No merge or M13 work.

#### Source scope and implementation constraints

Expected cohesive surfaces are new internal diffusion header/source and native
tests; existing SessionLoop/SessionChain, stateful C implementation, three public
C/C++ headers, ORT chain parser/factory, relevant CPU/CUDA/public-C tests, CMake
source/test registration and related documentation. This crosses the preferred
ten-file target because the same behavior must be wired through every existing
surface; do not hide missing integration to shrink the diff or add cosmetic work.
One intended PR remains the proposal for this review; request a plan amendment
before any scope/PR split or GPU-native implementation.

Known coupled assumptions requiring explicit changes/tests:

1. SessionLoop currently accepts only graph-input/output names.
2. SessionChain rebuilds loop output selection from every model graph output;
   the package factory does the same.
3. Chain input/state/source compatibility uses model NodeArgs, not logical policy
   feeds/results. Preserve genuine metadata validation with small policy-aware
   helpers and stable ownership, not unchecked casts or broad acceptance.
4. Chain invocation accounting adds1 per executed Step; standalone C API reports
   iterations as invocations. Sequential guidance requires real counts at both
   surfaces without altering M11 sample-only events.
5. Session I/O CPU staging counters are payload observations, not DMA evidence.

#### Numerical and transfer acceptance

- Pinned reference:
  Diffusersv0.35.1/`0f252be0ed42006c125ef4429156cb13ae6c1d60`, with the exact
  Appendix L profile, not the public PNDM scheduler implementation.
  Generate/check tiny golden values without downloading model weights.
- Per-element check is `abs(actual-expected) <= atol + rtol*abs(expected)`.
  Coefficients/single steps: atol=rtol=1e-5. Trajectories/CPU-CUDA parity:
  atol=rtol=1e-4. Use supplied common initial latents when comparing schedulers;
  verify the ORT RNG separately. No automatic tolerance widening.
- Final trace fixture proposal: S=3, `[N,C,H,W]=[2,2,3,5]`, L=60 FP32 elements.
  Batched guidance reads back one2L prediction per update:3 copies of480 bytes.
  Sequential guidance reads back twoL predictions per update:6 copies of240 bytes.
  Required prediction payload is therefore1440 bytes in either mode.
  Verify these counts/bytes from actual dedicated traces; any additional DtoH
  must be identified and explained, not silently counted as expected prediction.
- Report HtoD latents, conditioning, timestep and model/weight copies separately.
  Include setup copies honestly if the profiler capture contains setup. No
  device-only/compact-readback/performance claim is appropriate for this policy.

#### Proposed commands (not M12 validation results)

Resolve existing valid build/cache/reference paths at D0; the following is the
intended command structure. No build/test/reference installation has run for M12.

```bash
source .venv/bin/activate
ROOT="$(git rev-parse --show-toplevel)"
CPU_BUILD="$ROOT/build/state-cpu/RelWithDebInfo"
GPU_BUILD="$ROOT/build/state-cuda-plugin-m11-13.0/RelWithDebInfo"
LEGACY_BUILD="$ROOT/build/state-cuda-legacy-m11-13.0/RelWithDebInfo"

# After preserving M11 artifacts and registering the new sources:
cmake -S "$ROOT/cmake" -B "$CPU_BUILD"
cmake --build "$CPU_BUILD" \
  --target onnxruntime_session_loop_test onnxruntime_autoep_test --parallel 8
(
  cd "$CPU_BUILD"
  mkdir -p coverage/m12/focused
  export LLVM_PROFILE_FILE="$PWD/coverage/m12/focused/%m-%p.profraw"
  ./onnxruntime_session_loop_test --gtest_filter='*Ddim*'
  ./onnxruntime_autoep_test --gtest_filter='*Ddim*'
)

# Reconfigure valid CUDA caches after new source/CMake changes:
cmake --compile-no-warning-as-error -S "$ROOT/cmake" -B "$GPU_BUILD"
cmake --build "$GPU_BUILD" \
  --target onnxruntime_session_loop_test onnxruntime_autoep_test --parallel 8
(
  cd "$GPU_BUILD"
  export LD_LIBRARY_PATH="$PWD:/home/tianlei/cudnn_9.23_cuda13/lib:/home/tianlei/cuda13.0/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  mkdir -p coverage/m12/focused
  export LLVM_PROFILE_FILE="$PWD/coverage/m12/focused/%m-%p.profraw"
  ./onnxruntime_session_loop_test --gtest_filter='*Ddim*'
  ./onnxruntime_autoep_test --gtest_filter='*Ddim*'
  /usr/local/cuda/bin/compute-sanitizer --tool memcheck --leak-check full \
    --error-exitcode 99 ./onnxruntime_autoep_test --gtest_filter='*Ddim*'
  /usr/local/cuda/bin/compute-sanitizer --tool memcheck --leak-check full \
    --error-exitcode 99 ./onnxruntime_session_loop_test --gtest_filter='*Ddim*'
)
```

Final CPU and plugin commands run both executables **unfiltered**, with fresh
`coverage/m12/final` profiles. Build the legacy targets with their valid cache,
then run native unfiltered and public with the existing selector extended to
include the new DDIM suites:

```bash
./onnxruntime_autoep_test \
  --gtest_filter='AutoEpSelection.CudaEP:*Stateful*:*Ddim*:ModelPackageApiTest.*:*ModelPackageChain*'
```

All commands run from the corresponding output directory, with matching loader
paths. Record expected/new nonzero counts and no unintended skips; adjust planned
suite names to actual registrations before claiming a selector passed.
Standalone package tests use their existing shared/static build directories and
`ctest --test-dir "$PACKAGE_BUILD" --output-on-failure`.

Dedicated profiling uses separate proposed
`CudaDdimPackageTest.TransferBatched` and `.TransferSequential` cases:

```bash
/usr/local/cuda/bin/nsys profile --trace=cuda --sample=none --cpuctxsw=none \
  --force-overwrite=true -o "$EVIDENCE/m12-ddim-batched" \
  ./onnxruntime_autoep_test --gtest_filter='CudaDdimPackageTest.TransferBatched'
/usr/local/cuda/bin/compute-sanitizer --tool racecheck --error-exitcode 99 \
  ./onnxruntime_autoep_test --gtest_filter='CudaDdimPackageTest.Transfer*'
/usr/local/cuda/bin/compute-sanitizer --tool synccheck --error-exitcode 99 \
  ./onnxruntime_autoep_test --gtest_filter='CudaDdimPackageTest.Transfer*'
```

Repeat the Nsight command for sequential guidance with a distinct output;
export/inspect actual CUPTI copy tables. These are intended selectors, not
currently existing tests. No Nsight Compute source proof is substituted for host
coverage, and no new CUDA source is expected.

Merge profiles only with their matching objects per build configuration. Use
`llvm-cov` on native test, public test and shared-library mappings, plus genuine
SDK/minimal object mappings where new lines require them. Save per-file
covered/total counts, source/object/profile hashes and mismatch diagnostics in
session evidence. Keep reports/scripts rather than committing generated ONNX
fixtures, build products or real model weights.

#### Completion gates

- [x] Specification and detailed execution plan are separately approved.
- [x] Fixed-profile scheduler/guidance/RNG match independent CPU references.
- [x] Native loop/chain and public C/C++ semantics/counts/lifetimes pass.
- [x] Programmatic/package parity, explicit logical feeds and downstream latent
  handoff pass with actual tiny local models.
- [x] Genuine C11/C++17/no-exceptions/frozen-prefix and applicable minimal/
  standalone compatibility checks pass.
- [x] Real CUDA denoiser tests prove both guidance conventions, nondefault-stream
  behavior, expected prediction readbacks and zero applicable sanitizer errors.
- [x] Every new executable production line reaches measured100%, with modified
  earlier surfaces re-exercised and cumulative coverage reported safely.
- [x] Final CPU/plugin/selected-legacy regressions and style checks pass.
- [x] Documentation/evidence are current and one correct-base stacked PR is
  ready for human review.

Until every gate passes, M12 is not complete. Specification approval authorizes
this plan preparation only; wait for its separate review/approval before coding,
building or publishing M12. No M13 work or automatic merge is authorized.

**Implementation authorization:** The user approved this detailed plan on
2026-10-30. Branch `stateful-M16-ddim-policy` was created on the verified M11 head.
Fresh CPU69/148, CUDA-plugin132/204 and legacy120/69 baseline suites pass with
no reported skips. Matching M11 CPU/plugin binaries/libraries/profiles/device
object were archived with hashes before rebuilding. Genuine reference imports
failed first for missing Diffusers, then Torch; the existing isolated environment
now has Diffusers0.35.1 and CPU-only Torch2.8.0+cpu. The actual pinned reference
generated coefficient/1/2/50/999-step trajectory values without model downloads.
At that initial checkpoint only the numerical core was being built/tested.

**CPU qualification checkpoint (2026-10-31):** Native85/85 and unfiltered public
153/153 pass, with no reported skips. DDIM-focused selectors execute16 native
tests and5 public tests; matrices inside those cases cover both guidance modes,
all four model timestep formats, seed/supplied initialization, independent
concurrent executions, 1/2/50-step native trajectories and999-step core reference
trajectory, explicit invariant forwarding, typed seed sources through ordinary
producer output/retained state, downstream latent handoff, cancellation at the
complete-update boundary, first/second invocation failures and fail-fast
nonfinite predictions.

Independent public-only C++17 and no-exceptions SDK consumers pass both guidance
modes (4 executions), including programmatic/package parity, real invocation
counts, Reset and retained CPU values after owner teardown. Genuine frozen
M11-commit C11/C++17 header clients pass against the new library. Standalone
Model Package shared/static tests pass5/5 each. A separately compiled actual
`ORT_MINIMAL_BUILD` package-factory source probe returns NOT_IMPLEMENTED and
preserves its output; this is not claimed as a complete minimal build. M11's
measured minimal object/profiles remain untouched.

Final matching-source CPU unit evidence measures **537/537 new executable
production lines** against M11, including math, loop/chain, parser/factory, C API
and C++ inline wrappers. The initial13 uncovered locations were closed with
actual immutable-options, invariant/managed-input and typed producer/state
cases, not exclusions. The final CPU map audit classifies366 dormant hash-zero
maps and no nonzero function-hash mismatch. GPU/regression closure remains
required before completion/publication.

CUDA-plugin and legacy13.0 builds are in progress using their valid M11 caches
and matching libraries. No GPU DDIM result, sanitizer success, transfer count or
M12 completion is claimed yet. Dedicated GPU cases are authored for12
guidance/scale/initialization combinations, nondefault stream, retained CPU
values after library unregistration, device-logical-feed rejection and both
prediction-readback traces. No new CUDA kernel or device source was authored.

The first plugin regeneration omitted M11's command-line
`--compile-no-warning-as-error` option (it is not preserved merely by reusing the
cache), enabling `-Werror`/NVCC all-warnings-as-errors. The rebuild then stopped on
pre-existing braced-scalar-initializer warnings in unchanged GroupQueryAttention
registration. Restored the exact M11 warning policy and restarted the build;
no unrelated source or dispatch flags were changed. This setup failure is not a
GPU test failure or a passed qualification result.

The legacy regeneration had the same omitted M11 command-line warning policy.
GCC additionally identified two dangling-else warnings around new GoogleTest
macros; both new test sites now use explicit braces. Left the pre-existing M11
test warning untouched and restored legacy's exact baseline policy as well.
Both correct-policy CUDA builds remain active.

One counter-ordering correction computes each component's invocation delta
before adding it to the checked chain total, avoiding an unnecessary
intermediate overflow at very large cumulative counts. Fresh CPU final-2
native85/85/public153/153 and **537/537** new lines pass afterwards; the matching
map audit again finds366 dormant hash-zero maps and no nonzero mismatch.
No GPU result or ready PR is claimed at this checkpoint.

#### Final CPU/CUDA qualification (2026-10-31)

All implementation/qualification/publication gates pass.
Final source-safe results:

| Configuration | Native | Public/AutoEP |
| --- | --- | --- |
| CPU, Clang18 | 85/85 | 153/153 unfiltered |
| CUDA plugin, Clang18 host/NVCC13.0/GCC13 | 157/157 | 216/216 unfiltered |
| Legacy CUDA, GCC13 | 145/145 | 76/76 selected |

No reported skips. Final focused plugin DDIM suites execute25 native and12
public tests; legacy focused DDIM suites execute25 native and7 public tests.
The plugin matrix contains12 guidance/scale/initialization configurations with
nontrivial batch/dimensions and a nondefault stream. Deliberately different CPU
variant weights prevent wrong variant selection from satisfying the reference.
Native/public provider-assignment checks and disabled CPU fallback prove actual
CUDA denoiser work, not value-only parity.

- New production executable host lines: **537/537 (100%)**.
- Current-tree cumulative mapped host lines against stack base
  `1988019b6e23ac4def4628cddd603c1bfb504fe8`: **4752/4752 (100%)**.
  CPU `coverage/m12/final-3`, plugin `coverage/m12/final`, independently built
  standalone package tests and actual minimal-source probe are mapped separately;
  no incompatible profile merge, line exclusion or percentage rounding.
- Profile audits: CPU366/plugin1008/standalone9/minimal17 dormant hash-zero maps,
  **no nonzero function-hash mismatch**. Declaration/table-initializer-only
  changes have no executable-line denominator.
- No CUDA source/kernel was authored. Unchanged greedy source SHA remains
  `92837bc204a43c0e2ca0f7da8b5f0a037d5a6200d572a235412d9530e587cbcd`;
  parent device evidence is retained and current device conformance re-executed.
- Seed0 and full-uint64-maximum initialization additionally match independently
  generated MT19937_64/Box-Muller goldens, rather than only testing replay.
- Actual Nsight/CUPTI copies for S=3, `[2,2,3,5]`:

  | Guidance | All prediction DtoH | All HtoD, including setup |
  | --- | --- | --- |
  | Batched | 3x480 =1440 bytes | 11 copies,1508 bytes |
  | Sequential | 6x240 =1440 bytes | 20 copies,1520 bytes |

  No additional DtoH in either dedicated trace. HtoD is classified as two
  four-byte weight setup copies, latent/conditioning feeds and one four-byte
  timestep per invocation. Both traces include setup. Full prediction readback
  is intentional CPU policy work; no device-only/compact-readback/performance
  claim is made.
- Compute Sanitizer memcheck: plugin all25 native/all12 public DDIM tests,
  legacy GPU9 native/2 public tests and standalone GPU SDK all report
  **zero errors and zero leaked bytes**. Racecheck and synccheck each execute
  both dedicated guidance cases: zero hazards/errors/warnings.
- Independent public-only C++17 and no-exceptions GPU clients pass both guidance
  modes (4 executions), including library unregistration and retained CPU values.
  CPU SDK, genuine C11, frozen M11 C/C++ clients and standalone shared/static/
  instrumented tests remain verified. The inherited narrow M08 sanitizer timing
  limitation is unchanged; the normal unfiltered native suite includes/passes
  that test. No new test or memory error was excluded.

Exact final functional and sanitizer commands (from each output directory with
matching CUDA/cuDNN loader paths):

```bash
# CPU: passed85/153; final3 profiles match current test bodies and runtime:
LLVM_PROFILE_FILE="$PWD/coverage/m12/final-3/%m-%p.profraw" ./onnxruntime_session_loop_test
LLVM_PROFILE_FILE="$PWD/coverage/m12/final-3/%m-%p.profraw" ./onnxruntime_autoep_test
# CUDA plugin: passed157/216, both unfiltered:
LLVM_PROFILE_FILE="$PWD/coverage/m12/final/%m-%p.profraw" ./onnxruntime_session_loop_test
LLVM_PROFILE_FILE="$PWD/coverage/m12/final/%m-%p.profraw" ./onnxruntime_autoep_test
# Legacy: passed145/76; the public selector is intentionally not unfiltered:
./onnxruntime_session_loop_test
./onnxruntime_autoep_test \
  --gtest_filter='AutoEpSelection.CudaEP:*Stateful*:*Ddim*:ModelPackageApiTest.*:*ModelPackageChain*'
# Plugin memcheck: passed25/12 respectively, zero errors/leaks:
compute-sanitizer --tool memcheck --leak-check full --error-exitcode 99 \
  ./onnxruntime_session_loop_test --gtest_filter='*Ddim*'
compute-sanitizer --tool memcheck --leak-check full --error-exitcode 99 \
  ./onnxruntime_autoep_test --gtest_filter='*Ddim*'
# Plugin racecheck/synccheck: each passed2/2, zero hazards/errors:
compute-sanitizer --tool racecheck --error-exitcode 99 \
  ./onnxruntime_autoep_test --gtest_filter='CudaDdimPackageTest.Transfer*'
compute-sanitizer --tool synccheck --error-exitcode 99 \
  ./onnxruntime_autoep_test --gtest_filter='CudaDdimPackageTest.Transfer*'
```

Tool full paths are `/usr/local/cuda/bin/{compute-sanitizer,nsys}`. Matching loader
paths, exact SDK/frozen/minimal/standalone commands, source/object/profile hashes,
trace databases/reports, numerical/RNG references and independent verification
scripts are retained in session files under `m12-*`. The actual CUDA plugin is
`libonnxruntime_providers_cuda.so`. Pinned lintrunner, strict C11 syntax,
clang-format and whitespace checks pass. No dependency/toolchain installation
occurred beyond the genuinely missing pinned CPU reference packages.

[#34055](https://github.com/microsoft/onnxruntime/pull/34055) is OPEN, not draft, on
`stateful-M15-package-chain-loading` / #34054. Its body preserves the repository
template and records exact validation commands/results and limitations. The
qualified source, matching measured binaries/profiles, traces, verification
scripts and hashes are retained. Only inspected generated fixtures and hashed
unmeasured client binaries were cleaned. No merge or M13 work.

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

- First branch: `stateful-M01-contracts-fixtures`, based on the current `state/stateful-base-1988019b`
  branch at implementation start. Record the actual base commit then.
- Subsequent branches: `stateful-M02-context-loop`, and so on using the table
  suffixes, each based on the preceding milestone branch.
- Target repository: `microsoft/onnxruntime`. Each PR's base is the previous
  milestone branch; the first PR targets `state/stateful-base-1988019b`.
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
