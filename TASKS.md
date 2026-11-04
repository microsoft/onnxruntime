# Stateful Session Execution: Execution Plan

**Status:** Approved; M01-M15 complete and published for review. M15 is
[#29](https://github.com/onnxruntime/ort-state/pull/29), stacked on M14/#28.
M16-M23 are deferred at the user's direction. M24's requirements interview is
complete; [Appendix P](SPEC.md#appendix-p-m24-delivered-foundation-qualification-approved)
and the detailed M24 F0-F8 execution plan were separately approved. M24 is
complete and published OPEN/ready in
[#30](https://github.com/onnxruntime/ort-state/pull/30) on M15/#29.
M05: [PR #15](https://github.com/onnxruntime/ort-state/pull/15).
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
**M13 refinement:** SPEC Appendix M was separately approved on 2026-11-02.
The detailed M13 plan was separately approved on 2026-11-02; implementation,
qualification and publication are complete in [#27](https://github.com/onnxruntime/ort-state/pull/27).
**M14 refinement:** Separately approved and completed in
[#28](https://github.com/onnxruntime/ort-state/pull/28).
**M15 refinement:** Appendix O and separate P/R/Q/N plans are approved and
implemented. CPU/CUDA qualification, exact new-line coverage and the approved
O.10 bounded-race refinement pass; ready stacked publication is complete in #29.

## 1. Approval and tracking

- [x] Complete the requirements interview.
- [x] Ask whether the user has additional requirements before drafting.
- [x] Draft and review `SPEC.md` with the user.
- [x] Obtain specification approval.
- [x] Prepare this execution-plan draft.
- [x] Validate draft structure, dependency ordering, and all 93 requirement mappings.
- [x] Review this plan with the user and obtain approval before implementation.
- [ ] Complete the milestone stack and its validation gates.

M01-M15 and scoped M24 are complete; M16-M23 remain deferred.
Check tasks when their
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

Active stack order follows the completed milestones. Deferred rows are not
mechanical Git bases: M24 stacks directly on M15. The dependency column records
actual semantic prerequisites.
M01-M15 and scoped M24 are **complete**. M16-M23 are **deferred, not completed**.
M24's specification and detailed execution plan were separately approved.

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
| [x] | M13 | Python API and package parity / `python-api` | M06, M11, M12 |
| [x] | M14 | Local Phi-3 package and text demo / `phi3-demo` | M05, M08, M09, M11, M13 |
| [x] | M15 | Local Stable Diffusion package and image demo / `sd15-demo` | M11, M12, M13 |
| [ ] | M16 | **Deferred:** C# binding / `csharp-api` | M06, M11, M12 |
| [ ] | M17 | **Deferred:** Java/JNI binding / `java-api` | M06, M11, M12 |
| [ ] | M18 | **Deferred:** Common JavaScript API and Node binding / `js-node-api` | M06, M11, M12 |
| [ ] | M19 | **Deferred:** Browser/WASM binding / `js-web-api` | M18 |
| [ ] | M20 | **Deferred:** Go binding / `go-api` | M06, M11, M12 |
| [ ] | M21 | **Deferred:** Rust bindings / `rust-api` | M06, M11, M12 |
| [ ] | M22 | **Deferred:** Objective-C binding / `objc-api` | M06, M11, M12 |
| [ ] | M23 | **Deferred:** React Native binding / `react-native-api` | M18, M22 |
| [x] | M24 | Integrated qualification and measurements / `qualification` | M01 through M15; approved plan |

Do not mark the whole feature complete after the CPU foundation or either demo;
all binding, provider, coverage, and qualification gates remain part of delivery.
The user's2026-11-04 approved refinement defers M16-M23 and scopes M24 qualification to
the delivered C/C++/Python foundation only. It does not close the deferred
all-language requirements or authorize a claim of complete binding parity.

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
M09b was separately authorized on 2026-10-19 and is complete in PR #23.

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
and sanitizer logs. M09b was subsequently authorized and completed in PR #23.

#### M09b - GPU logits and loop/chain sampling integration

**Status:** Authorized 2026-10-19. Branch
`state/m09b-cuda-plugin-sampling-integration` is based on M09a `ff2363e3b8`,
stacked on PR #22. GPU kernels are unchanged; this milestone wires host execution.

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
as [PR #23](https://github.com/onnxruntime/ort-state/pull/23), stacked on M09a
PR #22. This completes macro M09. No PR has been merged; M10 is not started.

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

**Status:** Complete; [PR #24](https://github.com/onnxruntime/ort-state/pull/24)
is open/ready for human review, stacked on #23. The user approved SPEC J.9's
expanded save-as requirements and separately approved the detailed plan
amendment on 2026-10-19.
**Branch/base:** `state/m10-package-chain-schema`, stacked on verified M09b
`a7a1173cd1c64d318857fa2987caf26658048509` / PR #23.
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
  #23 and mark ready for human review only after every gate passes. Include the
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
is published as ready-for-review PR #24, stacked on verified M09b PR #23.
All M10 gates pass. Temporary compatibility build/source copies and probe
binaries were cleaned; source probes, reports, profiles and hashes remain.
No PR was merged and M11 has not begun.

### M11 - ORT package-to-chain loading

**Status:** Complete; CPU/GPU qualified and published for review on 2026-10-21.
**Specification:** [Approved Appendix K](SPEC.md#appendix-k-m11-executable-package-chains-approved).
**Branch/base:** `state/m11-package-chain-loading` on verified M10
`b03b4c2f72955e869401be4d8eaa277728dc6fa8` / PR #24.
**Delivery:** ONE independently documented/tested PR, explicitly requested by the
user. Internal tasks are not an M11a/M11b PR split.
**PR:** [#25](https://github.com/onnxruntime/ort-state/pull/25), OPEN and ready for
review, stacked on #24. GPU qualification commit `3e1124c6eb91e0325d58084d742b482d37275248`.
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
  independently verified M11 PR stacked on #24 only after H8 passes. Do not
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
`state/m11-package-chain-loading`, stacked on #24. At that checkpoint no M11 PR existed;
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
[#25](https://github.com/onnxruntime/ort-state/pull/25) is OPEN, not draft,
and uses the repository template with exact validation commands/results.
The base is `state/m10-package-chain-schema` / #24; no extra milestone PR was
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
**Proposed branch/base:** `state/m12-ddim-policy` on verified M11
`0586ee3acfc1456d1d48bdd060e4789eb0b99c98` /
[#25](https://github.com/onnxruntime/ort-state/pull/25).
**Delivery:** One intended, independently documented/tested stacked PR; no merge.
**PR:** [#26](https://github.com/onnxruntime/ort-state/pull/26), OPEN and ready for
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
2026-10-30. Branch `state/m12-ddim-policy` was created on the verified M11 head.
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

[#26](https://github.com/onnxruntime/ort-state/pull/26) is OPEN, not draft, on
`state/m11-package-chain-loading` / #25. Its body preserves the repository
template and records exact validation commands/results and limitations. The
qualified source, matching measured binaries/profiles, traces, verification
scripts and hashes are retained. Only inspected generated fixtures and hashed
unmeasured client binaries were cleaned. No merge or M13 work.

### M13 - Python bindings

**Status:** Completed. Implementation, CPU/plugin/legacy qualification, installed
wheels, coverage, sanitizers and transfer evidence verified.
**PR:** [#27](https://github.com/onnxruntime/ort-state/pull/27), OPEN and ready for review.
**Specification:** [Approved Appendix M](SPEC.md#appendix-m-m13-python-stateful-bindings-approved).
**Branch/base:** `state/m13-python-api` on verified M12
`09d169b3de6001f9ac5a2a5e52747bf1cadfa003` /
[#26](https://github.com/onnxruntime/ort-state/pull/26).
**Delivery:** One independently documented/tested stacked PR on M12#26; no merge.
**Outcome:** Typed `onnxruntime.stateful` Python construction/execution/package
parity with the existing native engine, including retained immutable values,
explicit device copies, GIL-safe cancellation and safe owner teardown.
Real-model demonstrations remain M14/M15; package authoring/export is deferred.

#### Approval and current prerequisites

- [x] Interview the user about namespace/configuration, tensor/owner contracts,
  package scope, GIL/cancellation and execution conveniences.
- [x] Explicitly confirm no additional requirements before specification drafting.
- [x] Draft/review Appendix M and obtain specification approval (2026-11-02).
- [x] Prepare this detailed plan from the approved specification.
- [x] Review the detailed plan and obtain separate implementation approval.
- [x] Implement, qualify and publish only after that approval.

At approval, inspection verified the clean M12 branch/remote PR head above.
Existing CPU/plugin/legacy caches had Python disabled; all three were
subsequently enabled and their actual extension/source identities verified.
Qualification used the existing `.venv`, CUDA13.0.48/matching cuDNN and A100 GPUs.

Disk is constrained. Matching M12 measured artifacts were preserved and their
hashes verified before changing caches. Valid incremental CPU/plugin/legacy
trees and cached dependencies were reused; M13 profiles were moved to session
storage with symlinks preserving their original build paths.
Do not delete M11/M12 proof, alter old13.1 trees or change global driver state.
Preserve the exact noncached `--compile-no-warning-as-error` configure option
used by the verified parent; omitting it caused unnecessary M12 rebuild failures.

#### Tasks, dependencies, outcomes and tests

| Task | Actual prerequisites | Intended outcome |
| --- | --- | --- |
| P0 | Plan approval | M13 branch, preserved M12 evidence, Python-enabled baseline builds and verified import paths. |
| P1 | P0 | Typed immutable configuration/enums and packaged public namespace. |
| P2 | P1 | Native resource ownership, operation guards and immutable value/copy interop. |
| P3 | P2 | Python loop/chain/execution/result/diagnostic parity, including bootstrap and DDIM. |
| P4 | P2 | Standalone cache/sampler/capability selection and retained-view parity. |
| P5 | P3 | Read-only package inspection/selection/ordinary Sessions and owned chain construction. |
| P6 | P3, P4, P5 | CPU, GIL/cancel/GC/lifetime, negative-path and package/wheel qualification. |
| P7 | P0, P6 | Real CUDA cache/greedy/DDIM parity, explicit copies, stream/transfer/lifetime evidence. |
| P8 | P6, P7 | 100% new Python/native binding lines, compatible cumulative evidence and regressions. |
| P9 | P8 | Current API/examples/evidence and one ready stacked PR on M12. |

- [x] **P0 - Establish the Python-enabled baseline.** Create the branch only
  after approval and verify M12's exact head/proof hashes. Enable pybind in the
  existing CPU/plugin/legacy configurations with the isolated interpreter and
  matching NumPy headers. Retain all parent provider/coverage/dispatch flags and
  configure warning policy. Build the Python extension plus focused native/public
  targets, stage required testdata and record extension/library identities.
  **Tests:** Import the built package outside repository-root shadowing; verify
  `onnxruntime.__file__`, wrapper/extension paths, build info, device enumeration
  and real CPU/CUDA Sessions. Establish selected existing Session/OrtValue/
  provider-copy Python baselines, plus native85/153 CPU,157/216 plugin and
 145/76 selected legacy expected parent counts. Record actual counts/skips,
  not these expectations as results. Restore a missing dependency only after
  the chosen command actually fails for it.

- [x] **P1 - Add typed configuration and namespace staging.** Provide frozen
  typed records/enums from Appendix M, normalizing nested lists/maps to owned
  immutable values. Keep Python orchestration thin: validate argument kinds/
  storage/ownership and marshal to native options; do not duplicate native
  scheduler, exit, cache or sampling logic. Stage `onnxruntime.stateful` in the
  ordinary package with explicit public exports, annotations and docstrings.
  **Tests:** Constructor/type/enum/count/name errors, duplicate/conflicting
  bindings, deep immutability and definition configuration capture. Verify
  namespace import and availability errors on unsupported builds without
  breaking ordinary package import. Reject reflection/unknown kwargs, unsupported
  array protocols and silent dtype/shape/contiguity conversions.

- [x] **P2 - Implement the ownership/value bridge.** Use the existing native
  stateful/package APIs and exception translator through focused pybind code.
  Capture exact low-level PyInferenceSession owners, NumPy/public OrtValue
  storage and SessionOptions callback snapshots. Add per-resource nonblocking
  busy/closed guards, deterministic close and GIL-safe native ownership scopes.
  Implement read-only stateful result/cache values, CPU-only copying `numpy()`,
  same-device copying `to_ortvalue()` and explicit `copy_to_cpu()`, reusing ORT
  allocators/transfers with retained Session/provider resources.
  **Tests:** Temporary arrays/OrtValues/stateful feeds survive GC; existing
  definitions survive high-level `set_providers()` replacement. Parent close
  cannot invalidate children, multiple value handles are independent, and
  writable copying conversions cannot mutate native snapshots/cache state.
  CPU/GPU metadata access is copy-free; GPU `numpy()` rejects without readback.
  Cover scalar/empty/FP16/bool and supported native element types, unsupported
  dtypes/copies/devices, endian/stride errors and allocation/transfer failures
  with checked sizes. Native owners never dereference Python references or
  destroy callback state without the GIL.

- [x] **P3 - Bind loop/chain/execution parity.** Expose definition construction,
  explicit input/batch/sequence/state collection conversion, Step/Run/Reset/
  Cancel, immutable result metadata/groups, state snapshots and dispatch/
  staging/failure diagnostics. Preserve native bootstrap and DDIM contracts.
  Acquire ownership before releasing the GIL and invalidate after a possible
  advancement if result delivery fails; never retry implicitly.
  **Tests:** Generic feedback, ordinary Session+loop chains, differently named
  bindings, every supported source/exit/output family, simultaneous exits and
  finite exhaustion. Cover old generation, new sampler-free prefill budgets0/1/N,
  first EOS and sample-only accounting. DDIM covers both guidance modes, native
  reference tolerances, seed/supplied latents, arbitrary batch, four timestep
  encodings and one/two-call counts. Test bad state, failed Run/Reset, closed/
  busy access and post-advancement wrapping failure without losing prior results.

- [x] **P4 - Bind standalone cache/sampler parity.** Expose copied cache layouts/
  provider selection and safe allocator-owning managers, Share/Copy/Append/
  Reserve/Reset, logical views/length/capacity/stats, reset-invalid checks,
  greedy sampling and capability/dispatch diagnostics. Reuse existing callback
  dispatch; do not enable Python-authored C EP callbacks.
  **Tests:** Immutable views after update/reset/manager close, retained borrowed
  source owners for Share, copies versus shared storage, boundary shapes/
  capacities, failed Reserve/Update/Reset and no silent CPU fallback. Greedy
  tests cover FP32/FP16, row selection, ties, NaN and explicit unsupported modes.
  Provider Require/Prefer selection and cache/sampler independence must match
  native routes with safe Session/library retention.

- [x] **P5 - Add read-only package construction.** Reuse native ModelPackage
  context/selection/factories and typed Python option snapshots for default/
  per-step SessionOptions. Copy inspection metadata and expose explicit named
  chain selection. Adopt an already initialized package-created ordinary Session
  into the public InferenceSession wrapper without loading a second model or
  losing callback/option metadata. Share a small existing metadata-initialization
  helper if necessary; keep ordinary Session construction/fallback unchanged.
  **Tests:** Component/variant inspection/selection, ordinary Session creation,
  programmatic/package loop/chain parity, bootstrap/DDIM, per-step complete
  replacements, invalid names/paths/policies/variants and construction rollback.
  Package/options/selected component/chain close and GC leave captured Sessions/
  callbacks/results valid. Source-independent parent exports and installed/
  portable packages still execute; no new authoring/export behavior is added.

- [x] **P6 - Qualify CPU ownership, concurrency and packaging.** Complete all
  public Python CPU/config/error/GC/copy/lifecycle matrices. Use `unittest`
  tests under pytest with tiny generated ONNX fixtures. Prove native-entry/
  release/cancel ordering using a bounded existing test custom-op/EP handshake;
  add a test-only primitive only if existing prior art cannot express it.
  Do not add a production callback/test hook or rely on sleep-only races.
  **Tests:** GIL-free Run/Step permits a second thread to act; overlapping Step/
  Run/Reset/state/close rejects; Cancel remains safe; independent executions
  run concurrently. Verify exception propagation from existing option callbacks
  and wrapping failures without deadlock/dangling references. Build and install
  the local CPU wheel into an isolated target with `--no-deps`; exercise public
  imports/examples there and assert its actual extension identity. Verify
  annotations with scoped existing pyright settings and relevant Session/
  OrtValue/provider-copy regressions; no unrelated baseline fixes.

- [x] **P7 - Qualify real CUDA and explicit copy behavior.** Use the same tiny
  native-equivalent LLM/DDIM fixtures against plugin CUDA, verifying actual
  model placement and native cache/greedy dispatch, not a CPU-enabled fallback.
  Verify nondefault stream, requested ordinal, retained GPU values after
  Session/library teardown, same-device clones and explicit CPU copies.
  Preserve legacy's unsupported native cache/sampler boundary; qualify legacy
  DDIM/model execution and explicit copy paths where supported.
  **Tests:** Programmatic/package/NumPy/OrtValue input parity; generation budgets/
  EOS/policies, DDIM both conventions and native tolerances. Dedicated traces
  separate normal result access from explicit conversion: normal GPU logits/
  caches remain resident with compact greedy token/status readback; clone is
  same-device; `copy_to_cpu()` is the only requested full readback. DDIM retains
  the parent's intentional CPU policy/full predictions. Run applicable focused
  memcheck/leak/race/sync checks against the actual Python extension. Install/
  smoke the local CUDA-plugin wheel and report only verified configurations.

- [x] **P8 - Close Python/native coverage and regressions.** Measure every new
  executable production Python line with coverage.py, and native binding/glue/
  conversion/ownership lines using instrumented pybind/runtime/provider objects.
  Include any shared helper edits and both Python/native error paths. Map staged/
  installed wrapper sources to the exact source SHA and native profiles to
  their actual extension; never omit the pybind `.so` from native coverage.
  **Tests:** Reach100% independently for new Python and native binding code,
  then report safe cumulative stack mappings. Keep CPU/plugin/legacy/
  standalone/minimal evidence separate unless mappings are compatible. Inspect
  nonzero profile-hash mismatches and dormant zero maps explicitly. Final
  focused Python suites, complete selected existing Python regressions, CPU/
  plugin native+public and selected legacy suites, standalone package tests,
  scoped type/style/whitespace and installed examples must pass. Missing tools/
  instrumentation blocks a gate; no line exclusions/rounding/new kernel claims.

- [x] **P9 - Document and publish.** Document the exact typed API, copied versus
  borrowed storage, readonly value/conversion/device behavior, captured Session
  identity, package options/callback ownership, close/context managers, GIL/
  busy/Cancel/Reset rules and errors. Add executable tiny-model examples for
  generic orchestration, bootstrap and DDIM without tokenizer dependencies.
  **Tests:** Run documented examples on CPU and meaningful CUDA configurations,
  validate public signatures/imports from installed local wheels, and record
  exact test selectors/counts/skips, coverage and trace/hash evidence. Update
  SPEC/TASKS/SCRATCHPAD, inspect/preserve the PR template and verify one ready
  synchronized PR on M12. No merge or M14 execution.

#### Source scope and known integration constraints

Prefer one public module, one focused pybind implementation with small ownership
helpers only as needed, and targeted tests/fixtures/docs. Expected integration
surfaces are Python module registration, PyInferenceSession adoption/keepalive,
the existing high-level Session metadata initialization, CMake package staging
and local wheel packaging. A root `onnxruntime/stateful.py` can be staged like
the existing root initializer; avoid duplicate publicly importable type copies
in both root and `capi`. Verify actual setup.py behavior before adding packaging
changes merely for completeness.

This full-parity milestone may exceed the preferred ten-file target; focused
binding/tests/build/docs changes are necessary, not a reason to omit native
features. Request a reviewed amendment before a PR split or scope change.

Concrete coupled constraints discovered during plan preparation:

1. PyInferenceSession owns a unique native Session; high-level `set_providers`
   replaces its `_sess`. A high-level keepalive alone is insufficient.
2. PySessionOptions snapshots own EP-selection/context-data registrations. Raw
   native SessionOptions pointers cannot outlive those Python registrations.
3. Ordinary OrtValue exposes mutation/raw pointers/DLPack. A separate read-only
   bridge is required; returning its underlying mutable OrtValue is not safe.
4. Existing `copy_tensors` uses the Python Env transfer path. Verify selected
   device allocator/transfer ownership after library unregistration rather than
   assuming Env registration or source Session still exists.
5. Existing package factories return initialized native Sessions. Python
   adoption must not reload the model or discard metadata/custom-op ownership.
6. GIL release does not serialize mutable resources or make Python destructor/
   callback reference cleanup safe. Operation guards and destruction ordering
   must be native-safe, including builds that advertise free-threaded support.
   Do not claim a free-threaded interpreter was tested if none is available.
7. Unsupported builds may not provide stateful API tables; import/config-value
   construction must remain usable while resource creation reports unsupported.

#### Reproducible configuration and qualification commands

These configurations and qualification commands were executed; final counts
and known baseline limitations are recorded below. Enabling Python requires
regenerating valid caches with the parent warning policy:

```bash
source .venv/bin/activate
ROOT="$(git rev-parse --show-toplevel)"
PYTHON="$ROOT/.venv/bin/python"
CPU_BUILD="$ROOT/build/state-cpu/RelWithDebInfo"
GPU_BUILD="$ROOT/build/state-cuda-plugin-m11-13.0/RelWithDebInfo"
LEGACY_BUILD="$ROOT/build/state-cuda-legacy-m11-13.0/RelWithDebInfo"

cmake --compile-no-warning-as-error -S "$ROOT/cmake" -B "$CPU_BUILD" \
  -Donnxruntime_ENABLE_PYTHON=ON -DPython_EXECUTABLE="$PYTHON"
cmake --build "$CPU_BUILD" --parallel 8 \
  --target onnxruntime_pybind11_state onnxruntime_session_loop_test onnxruntime_autoep_test
# Repeat for GPU_BUILD and LEGACY_BUILD with their unchanged provider flags.
```

Run Python from the matching output directory, **not the repository root**.
All test modules below are implemented:

```bash
cd "$CPU_BUILD"
mkdir -p coverage/m13/python coverage/m13/native
export LLVM_PROFILE_FILE="$PWD/coverage/m13/native/%m-%p.profraw"
export COVERAGE_FILE="$PWD/coverage/m13/python/.coverage"
"$PYTHON" -c 'import onnxruntime as ort; from onnxruntime.capi import onnxruntime_pybind11_state as c; print(ort.__file__, c.__file__, ort.get_build_info())'
"$PYTHON" -m coverage run \
  --include="$PWD/onnxruntime/stateful.py,$PWD/onnxruntime/capi/onnxruntime_inference_collection.py" \
  -m pytest -q "$ROOT/onnxruntime/test/python/onnxruntime_test_python_stateful.py" \
  "$ROOT/onnxruntime/test/python/onnxruntime_test_python_stateful_package.py" \
  "$ROOT/onnxruntime/test/python/onnxruntime_test_python_stateful_threads.py"
"$PYTHON" -m coverage report -m
./onnxruntime_session_loop_test
./onnxruntime_autoep_test
```

Collect explicit changed-line evidence too; do not substitute a module percentage
for native/glue coverage. Use coverage source identities for copied/staged files.
If source normalization is needed, retain exact original paths and SHA evidence.

```bash
cd "$GPU_BUILD"
export LD_LIBRARY_PATH="$PWD:/home/tianlei/cudnn_9.23_cuda13/lib:/home/tianlei/cuda13.0/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LLVM_PROFILE_FILE="$PWD/coverage/m13/native/%m-%p.profraw"
"$PYTHON" -m pytest -q \
  "$ROOT/onnxruntime/test/python/onnxruntime_test_python_stateful.py" \
  "$ROOT/onnxruntime/test/python/onnxruntime_test_python_stateful_package.py" \
  "$ROOT/onnxruntime/test/python/onnxruntime_test_python_stateful_threads.py" \
  "$ROOT/onnxruntime/test/python/onnxruntime_test_python_stateful_cuda.py"
for worker in lifetime pipeline; do
  /usr/local/cuda/bin/compute-sanitizer --tool memcheck --leak-check full \
    --show-backtrace no --error-exitcode 99 "$PYTHON" \
    "$ROOT/onnxruntime/test/python/onnxruntime_test_python_stateful_cuda.py" "--$worker-worker"
done
./onnxruntime_session_loop_test
./onnxruntime_autoep_test
```

Run race/sync and transfer profilers on verified dedicated Python selectors
without unrelated diagnostics or caller readbacks. Establish explicit expected
sizes from fixture dimensions, not a proxy payload counter. Record any
interpreter baseline sanitizer behavior separately; do not suppress new errors
or omit new lifecycle tests to obtain a clean report.

Legacy runs use the same focused Python modules with explicit unavailable/
unsupported feature tests, then native unfiltered and the parent public selector:

```bash
./onnxruntime_autoep_test \
  --gtest_filter='AutoEpSelection.CudaEP:*Stateful*:*Ddim*:ModelPackageApiTest.*:*ModelPackageChain*'
```

Use the existing build.py/setup.py local-wheel mechanism after verifying staging;
install each local wheel into a separate isolated smoke target with `--no-deps`.
Verify imports from those targets outside repository/build shadowing and run
the tiny public examples. Do not fetch a published ORT wheel as evidence for
the new code. Package build/installation is required to validate the new public
namespace, not a separate release/distribution.

For native coverage, merge only matching profiles per configuration and include
the actual `onnxruntime_pybind11_state` extension alongside runtime/test/provider
objects. Save per-file Python/native counts, loaded module and source hashes,
profile diagnostics and explicit transfer proof. Existing native cumulative
maps may be reused only when source/object identity is verified; changed earlier
paths must be rerun. No new CUDA source is planned.

#### Completion gates

- [x] Specification and detailed execution plan are separately approved.
- [x] Typed namespace/configuration and CPU native/public Python parity pass.
- [x] Read-only results/copies, captured Sessions, input/callback/GC ownership
  and explicit close/context-manager contracts pass.
- [x] Deterministic GIL/cancel/busy/independent-execution tests pass without
  Python callback/destructor deadlock.
- [x] Read-only package inspection/component/chain loading and both native
  generation/bootstrap/DDIM policies pass from Python.
- [x] Real CUDA/model/cache/sampler dispatch, same-device/explicit CPU copies,
  stream/transfer/lifetime and applicable legacy/sanitizer checks pass.
- [x] Installed local CPU/plugin package imports/examples use verified new
  extensions, not stale wheels or source shadowing.
- [x] Every new executable Python and native binding/glue line reaches
  measured100%; safe cumulative mappings and modified earlier tests are current.
- [x] Existing relevant Python/native/package regressions, type/style and
  documented examples pass with exact commands/results recorded; the unchanged
  stateless plugin negative-device baseline failure is separately reproduced below.
- [x] One correct-base stacked PR is OPEN/ready for review and synchronized.

The detailed plan was separately approved before implementation. All gates are
complete; no M14 work or automatic merge is authorized. The following
dated checkpoints preserve intermediate results; final qualification below
supersedes their historical pending-work statements.

**Implementation authorization/checkpoint (2026-11-02):** The user separately
approved this detailed plan. Branch `state/m13-python-api` was created on the
verified M12 head. All preserved M12 measured-artifact hashes passed verification
before enabling Python. CPU pybind import resolves to the actual build package/
extension, and native85/85/public153/153 parent baselines pass. Six focused
existing Python provider/SessionOptions/Session-scoped OrtValue tests also pass.
Plugin and legacy Python-enabled baseline builds remain active; no new CUDA
Python result is claimed.

The first typed configuration/enums/public-namespace slice is staged into the
actual CPU package. Four configuration tests plus28 subtests pass, with
**297/297 executable Python statements** covered, source/staged SHA identity
verified, scoped pyright zero errors/warnings and pinned lint clean. Native
resource/execution/package bindings and full M13 coverage/qualification are not
implemented/complete yet. Configuration records can describe definitions; this
checkpoint is not a claim that the Python runtime surface is complete.

`onnxruntime.stateful` staging follows the existing root initializer's copy path;
configure-time copy also tracks Python-only edits, avoiding stale staged source.
Native memory device identifiers are verified against the existing Python enum,
including FPGA2 and NPU3. No new provider capability is implied.

The first selected Python test/coverage/type commands genuinely failed for
missing pytest, coverage and pyright. Those tools were restored only inside the
existing isolated environment. No GPU tools, model assets or global settings
were installed/changed. Baseline direct-CMake imports report the existing missing
`build_and_package_info` warning; package/wheel metadata remains P6 validation.

Coverage's dotted `--source=onnxruntime.stateful` probe pre-imported the package
and caused NumPy's extension to reject repeat initialization during collection.
The same actual built tests pass normally and under a source-file
`--include=<build>/onnxruntime/stateful.py` filter, with exact source SHA and
coverage JSON retained. This changes discovery only, not the measured statement
denominator or exclusions. Future commands must avoid the side-effecting dotted
probe and still account for every new production wrapper file.

**Binding implementation checkpoint (2026-11-02):**

- Python-enabled parent regressions pass: CPU85 native/153 public, plugin157
  native/216 public, legacy145 native/76 selected public. Actual extension/build
  identities are recorded; legacy public remains deliberately selected.
- Implemented thin native/Python loop/chain/execution/result/diagnostic bindings,
  immutable value handles, explicit same-device/CPU copies, retained inputs and
  captured low-level Session identity. CPU tests cover generic/state/ordinary
  components, sample-only bootstrap budgets0/1/N/first EOS, DDIM both guidance
  modes/all four timestep encodings, result-delivery failure/Reset and GC.
- Explicit dependency leases destroy native children before parent Sessions/
  allocators/input owners, including transitive value copies and cancellation.
  Cleanup does not rely on tuple/dict attribute destruction order. Public close
  releases one handle; retained children remain independent.
- Standalone cache/view/greedy/capability bindings now pass CPU tests. Share
  requires allocator-owned values; borrowed NumPy storage is rejected by the
  native policy, never silently copied. Copy/Append/reserve/reset preserve old
  views. Input owners are conservatively retained until manager close, including
  reused Reset storage. CPU suite: **17 tests plus64 subtests pass**; legacy
  extension runs the same suite successfully.
- Real plugin GPU value test passes for FP32 and FP16, with actual model
  assignment/native greedy dispatch, rejected implicit GPU numpy(), same-device
  writable clones and explicit CPU copies after parent/library release. This
  lifetime test runs in a bounded isolated subprocess and propagates child
  failures; coverage collects the actual worker. Full cache/sampler/package GPU
  qualification, GIL/cancellation concurrency, traces and sanitizers remain P6/P7.
- Current Python CPU slice has **688/689 statements** covered; the remaining
  statement is the GPU numpy() rejection and is covered in the isolated GPU case.
  Do not combine old source versions blindly or call this full M13 coverage.
  Fresh native binding coverage and remaining feature implementation are still
  required. P4 plugin rebuild/combined tests are active; package loading is not
  implemented and no M13 PR exists.

Two baseline/setup limitations are recorded without unrelated source changes:

1. The selected existing plugin Python provider-options test expects a
   RuntimeError for negative device ID, but the unchanged stateless plugin
   selection code chooses a device instead. That selection produced5 passes/
   1 failure, not a clean6-test result. New stateful provider validation must
   remain explicit; do not fix or hide the unrelated stateless behavior here.
2. The first GPU lifetime fixture unnecessarily unregistered/re-registered the
   CUDA factory before any Session. It failed Gather; a bounded ordinary-Session
   diagnostic also stalled after that setup cycle. Normal bundled-factory
   ordinary Run, IOBinding and CPU/GPU-sampled loops all pass. The isolated test
   now uses a fresh normal factory and unregisters only after producing values,
   still proving the required post-unregistration lifetime. The diagnostic
   process was specifically stopped; no global device/driver setting changed.

An in-flight plugin build initially used the old source list and omitted the
new bridge implementation, causing an undefined registration symbol on import.
Explicit source-list regeneration fixed it. Later Python-only edits also
required source refresh before tests; exact source/staged hashes are checked.
Only M13-generated profiles were moved to session storage to relieve root disk
pressure; symlinks preserve recorded build paths. M11/M12 evidence is untouched.

**Latest checkpoint:** P2-P5 implementations now include read-only package
inspection, component/variant selection, adoption of initialized ordinary
Sessions, and named owned-chain loading. The adopting wrapper preserves callback
snapshots and reloads the actual selected file on explicit `set_providers()`.
SessionOptions snapshots use supported copy construction, not unimplemented
copy assignment. Final coverage/qualification gates remain open.

- Current CPU Python suites: **24 tests/77 subtests pass**, including deterministic
  native-barrier GIL release, concurrent cancellation, busy-operation rejection,
  and independent executions on one Session.
- Legacy core/package suites: **22 tests/72 subtests pass**. The isolated installed
  CPU wheel passes the same22 tests/72 subtests outside source/build shadowing.
- Both actual CUDA worker tests pass: FP32/FP16, all four independent CPU/GPU
  cache/sampler policy combinations and budgets0/1/3 (24 package configurations),
  standalone device cache/sampling, both DDIM guidance modes, nondefault stream,
  and retained values after Session/library release. The GPU wheel is built;
  isolated installation/smoke remains open.
- Current Python CPU coverage is **755/756 statements**; the GPU-only `numpy()`
  rejection runs in the CUDA worker. Source-identical union and native changed-line
  closure are still required; this is not a100% M13 coverage claim.
- Compute Sanitizer2025.3.0 with default host backtraces retained ordinary Python
  `InferenceSession.__init__` frames, preventing Session destruction. A plain
  Python CUDA baseline without any stateful imports reproduced19,180,048bytes/
  14allocations; weak-reference diagnostics confirmed the retention. The same
  baseline releases both Sessions normally without instrumentation.
- `--show-backtrace no` removes that diagnostic-induced retention without
  disabling memory, leak, API-error, or application-exit checks. Full memcheck
  of **each actual worker** now passes: **zero errors and zero leaked bytes**.
  Original failing reports and ordinary baseline diagnostics remain preserved.
  Race/sync, dedicated transfer traces, final regressions and coverage remain open.

The successful memcheck commands, from the plugin output directory with the
documented matching loader/PYTHONPATH, were:

```bash
/usr/local/cuda/bin/compute-sanitizer --tool memcheck --leak-check full \
  --show-backtrace no --error-exitcode 99 "$ROOT/.venv/bin/python" \
  "$ROOT/onnxruntime/test/python/onnxruntime_test_python_stateful_cuda.py" --lifetime-worker
/usr/local/cuda/bin/compute-sanitizer --tool memcheck --leak-check full \
  --show-backtrace no --error-exitcode 99 "$ROOT/.venv/bin/python" \
  "$ROOT/onnxruntime/test/python/onnxruntime_test_python_stateful_cuda.py" --pipeline-worker
```

Both exited0. Evidence: `m13-gpu-{lifetime,pipeline}-memcheck-final.log` in the
session evidence directory.

#### Final qualification (2026-11-02)

All implementation/qualification/publication gates P0-P9 are complete. PR
[#27](https://github.com/onnxruntime/ort-state/pull/27) is OPEN/ready with base
`state/m12-ddim-policy` (#26). Implementation commit
`dc054a42b20fade26e6a65f5a4b587fefdb6b278` contains the qualified code; the
publication bookkeeping commit changes tracking only.
User-facing contracts and runnable examples are in
[Stateful Python execution](docs/Stateful_Python.md) and
[stateful_execution.py](samples/python/stateful_execution.py).

| Surface | Verified result |
| --- | --- |
| CPU native/public | 85/85 and153/153, unfiltered |
| CUDA-plugin native/public | 157/157 and216/216, unfiltered |
| Legacy CUDA native/public | 145/145 and76/76 with the explicitly documented parent selector |
| CPU and legacy new Python suites | Each27 tests/84 subtests, no skips |
| Installed local CPU wheel | 27 tests/84 subtests, verified isolated package/extension |
| Installed local GPU wheel | 29 tests/84 subtests including both actual CUDA workers, no skips |
| Existing CPU Python Session/OrtValue | 8 passed/87 deselected |
| Existing CPU callback regressions | 2 passed/14 deselected, including actual EPContext data callbacks |
| Existing plugin Session/OrtValue | 7 passed/88 deselected; known failure separately rerun |
| Standalone shared/static package | Each5/5 CTests |
| Examples | Feedback, bootstrap, both DDIM modes pass on installed CPU/plugin wheels and legacy CUDA |
| Scoped type/style/whitespace | pyright0 errors/warnings; lintrunner and `git diff --check` pass |

Final measured new production coverage:

- Native binding/glue **660/660 executable mapped lines**, including bridge
  **652/652**, registration1/1, adopting constructor5/5 and internal accessors2/2.
  The actual minimal-build registration source was independently compiled and
  executed, reporting `available=False`; it is not a mocked native-coverage claim.
- Public Python module **756/756 statements** and changed existing Session
  helper **13/13**, totaling **769/769 new binding/helper statements**.
- Executable example **63/63 statements** across its CPU/CUDA runs.
- Safe cumulative native stack mapping **5412/5412**, including the actual Python
  extension, current CPU/plugin tests/runtime and unchanged, identity-verified
  standalone/minimal parent maps. No new CUDA source or device coverage claim.
- Every one of882 audited profile diagnostic entries has hash0 (dormant maps);
  **zero nonzero source/profile hash mismatches**. Reports retain the warnings.
  Old untouched `TYPE_CHECKING` lines20-22 in the existing Python module are not
  new executable code; no new line was excluded.

Coverage and source-safe normalization scripts/reports are preserved under
`$EVIDENCE=/home/devpatel/.copilot/session-state/819a29ec-8ea8-4b2f-9938-e40236ffd239/files`:

```bash
# Passed660/660 native and769/769 Python:
"$PYTHON" "$EVIDENCE/measure_m13_coverage.py"
# Passed5412/5412 cumulative native and769/769 Python:
"$PYTHON" "$EVIDENCE/measure_m13_coverage.py" --cumulative
# Passed exact CUPTI byte/count assertions for every captured transfer:
"$PYTHON" "$EVIDENCE/m13_verify_transfers.py"
```

Final GPU worker also rejects CPU logits and mismatched ordinal1 GPU logits
under required ordinal0 sampling, with zero CPU/provider invocations for those
precondition failures. Both FP32/FP16 paths run on the available A100s.
All three sanitizer tools pass both actual worker processes:
memcheck/full-leak **0 errors/0 bytes**, racecheck **0 hazards/errors/warnings**,
synccheck **0 errors**. `--show-backtrace no` is the documented workaround for
diagnostic-induced Python frame retention; memory/leak/API/exit checks remain on.

Five dedicated `nsys profile --trace=cuda --sample=none --cpuctxsw=none
--capture-range=cudaProfilerApi --capture-range-end=stop` ranges were exported
with `nsys export --type=sqlite` and asserted against CUPTI records:

| Captured operation | Exact transfer result |
| --- | --- |
| Native greedy Run plus ordinary metadata/result access | 16B HtoD input and16B DtoH compact result; no32,768B logits readback |
| `to_ortvalue()` | One32,768B DtoD; zero DtoH |
| `copy_to_cpu()` | One32,768B DtoH |
| DDIM batched, S3/N2C2H3W5 | Three480B prediction DtoH =1440B |
| DDIM sequential, same fixture | Six240B prediction DtoH =1440B |

Intentional DDIM input transfers are also asserted: batched3each of4B timestep,
16B conditioning and480B latents; sequential6each of4B,8B and240B respectively.
Warmup/setup and explicit caller readbacks are outside unrelated capture ranges.

Final directly related existing-Python commands (matching output directory,
loaded build and documented loader) were:

```bash
# CPU: passed8; plugin: passed7 when the explicitly known test is separated:
"$PYTHON" -m pytest -q -rs \
  "$ROOT/onnxruntime/test/python/onnxruntime_test_python.py" \
  -k 'test_get_providers or test_set_providers or test_session_scoped_cpu_ortvalue or test_run_session_owned_output_is_zero_copy or test_device_ortvalue_provenance_rules or test_get_graph_provider_assignment_info'
# Plugin passing selector adds: and not test_set_providers_with_options
# CPU: passed2, including actual native read/write callback execution:
"$PYTHON" -m pytest -q -rs \
  "$ROOT/onnxruntime/test/python/onnxruntime_test_python_compile_api.py" \
  -k 'test_callback_finalizers_can_reenter_configuration or test_external_ep_context_data_callbacks'
# Plugin: FAILED1 exactly as the pre-change baseline; not a new-test failure:
"$PYTHON" -m pytest -q \
  "$ROOT/onnxruntime/test/python/onnxruntime_test_python.py" \
  -k test_set_providers_with_options
```

The known stateless test still expects RuntimeError for a negative CUDA device
option while unchanged plugin selection ignores it. No unrelated provider
selection fix or hidden test suppression was made. Direct-CMake legacy imports
still report missing wheel metadata; installed wheels have correct metadata.
Free-threaded Python, other GPU architectures and real-model M14/M15 demos were
not claimed. Package Python authoring/export remains deferred.

### M14 - Phi-3 local package and text-to-text demo

**Status:** [Appendix N](SPEC.md#appendix-n-m14-phi-3-package-and-text-demo-approved)
and the detailed plan below were separately approved on 2026-11-02.
Implementation, CPU/CUDA qualification and publication are complete.
**PR:** [#28](https://github.com/onnxruntime/ort-state/pull/28), OPEN/ready on M13#27.
**Planning branch/base:** `state/m14-phi3-demo` on M13
`42ab052f08c8dc44a76a8012e3e70a053fe64b1a` /
[#27](https://github.com/onnxruntime/ort-state/pull/27).

**Outcome:** Explicitly acquired, revision-pinned Phi-3 CPU INT4 and CUDA FP16
packages execute text-in/text-out through native sampler-free prefill and decode
SessionLoops, shared KV management and greedy sampling. Python supplies the CLI,
local tokenizer adapter, package construction and reports, not a second engine.

#### Approval and confirmed inputs

- [x] Interview and explicitly confirm no additional requirements.
- [x] Review and approve Appendix N before preparing this plan.
- [x] Prepare the detailed outcomes, prerequisites and CPU/CUDA test gates below.
- [x] Review/approve this plan before acquisition, dependency changes or implementation.

Asset root: `/home/devpatel/onnxruntime-stateful/phi3-mini-4k-instruct`.
HF repository: `microsoft/Phi-3-mini-4k-instruct-onnx`, revision
`5f5f794c1c23c9d5ee142af85df02a6cc52d6945`.
Selected variants: `cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4` and
`cuda/cuda-fp16`; approximately10.4GB graphs/weights/tokenizer metadata.
Normal runs are offline. No downloaded Python source, model binaries or external
weights enter Git. The local tokenizer adapter is optional demo tooling with
remote code execution disabled; chat-template default and explicit raw mode.

#### Scope and implementation surfaces

- Prefer one focused `samples/python/phi3_demo.py` CLI/helper module, targeted
  unittest-based Python suites, optional demo-only dependency declaration if
  needed, and `docs/Stateful_Phi3_Demo.md`. Keep the core ORT dependency manifests
  and deferred public Python authoring/export surface unchanged.
- Existing M13 package fixtures and CUDA policy/bootstrap manifests demonstrate
  the required schema. The standalone package library has no authoring CLI or
  public Python authoring wrapper to reuse; a small declarative manifest builder
  may emit existing-schema JSON and must validate/load through native factories.
  Do not add a generic parser, authoring API or C++ CLI solely for this demo.
- Inspect tokenization prior art without copying its engine or unsafe
  `trust_remote_code=True` settings. Use a local-only, remote-code-disabled adapter.
- Reuse existing Python test staging/runner conventions. Keep offline fixture
  tests discoverable/runnable without model downloads. Real asset/GPU qualification
  is explicit opt-in with a supplied asset/package path and checked nonzero counts.
- No runtime/native/kernel changes are planned. An actual unsupported signature,
  unsafe aliasing or missing native feature is a stop-and-amend gate, not permission
  to silently rewrite models, relax ownership or implement a Python fallback.

#### Tasks, dependencies, outcomes and tests

| Task | Actual prerequisites | Intended outcome |
| --- | --- | --- |
| P0 | Plan approval | Verified M13 base/proof, isolated environment, storage/device preflight and focused baselines. |
| P1 | P0 | Tested explicit acquisition, verified local assets and actual graph compatibility report. |
| P2 | P1 | Portable EP-specific packages with native-validated cache/bootstrap/exit declarations. |
| P3 | P1 | Local tokenizer/chat-template/raw adapter and caller-adapter contract. |
| P4 | P2, P3 | Python CLI/helpers driving native execution and correct text/token/result lifecycle. |
| P5 | P4 | Ordinary-Session reference and real CPU INT4 numerical/boundary/replay qualification. |
| P6 | P5 | Real CUDA FP16 native cache/greedy qualification, resident-data and sanitizer evidence. |
| P7 | P5, P6 | 100% new executable production lines, regressions and isolated-wheel/offline checks. |
| P8 | P7 | Reproducible user documentation/evidence and one ready stacked PR on M13. |

- [x] **P0 - Preserve the parent and establish bounded baselines.**
  **Outcome:** Reverify local/remote M13 head/base, planning-only modifications,
  actual CPU/plugin/legacy extension identities and the M13 proof manifest.
  Verify parent source hashes against the parent commit where planning documents
  now differ intentionally. Preserve matching parent binaries/profiles before
  any rebuild; reuse valid build caches and existing isolated `.venv`.
  Recheck space under the selected asset root and available GPU memory before
  reserving approximately10.4GB plus bounded working space. Do not clean shared
  directories, alter GPU state or install global tooling.
  **Tests:** Existing M13 CPU27/GPU29 Python-suite counts, actual module paths,
  plugin-device enumeration and native capability queries. Use known CPU85/153,
  plugin157/216 and selected legacy145/76 native/public parent results as expected
  counts, never as new results without executing the commands. Record the known
  stateless negative-device failure separately rather than hiding or fixing it.

- [x] **P1 - Implement and test explicit acquisition and graph inspection.**
  **Outcome:** Add allowlisted, revision-pinned setup acquisition into the approved
  asset root, with bounded streaming, explicit errors, expected size/LFS SHA-256
  checks, local provenance and license retention. Reuse verified existing assets;
  never treat a partial/corrupt file as success. Keep normal commands offline.
  Download only the two approved variants and required data/tokenizer/config/license
  assets; exclude remote Python implementations. Inspect ONNX without loading
  all external weights into an inspection process.
  **CPU tests:** Tiny/fake assets and mocked bounded HTTP cover successful and
  already-verified downloads, network/HTTP/partial/write/hash/size errors,
  malformed metadata, unknown assets and cleanup. Inspect tiny external-data
  graphs and reject missing/escaping references, unsupported ports/ranks/types,
  inconsistent layer pairs and ambiguous cache layouts.
  **Real CPU/CUDA gate:** Verify every selected graph/weight hash and inspect all
  actual ports,64 expected key/value bindings, element types, sequence axes,
  masks, opsets and past/present aliasing assumptions. Compare with upstream
  metadata without assuming it is authoritative. Record compatibility separately
  for CPU INT4 and CUDA FP16. Stop for an amendment if native support is missing.

- [x] **P2 - Construct native-validated CPU and CUDA packages.**
  **Outcome:** Build portable EP-specific manifests under the asset root without
  duplicating each variant's weights for its two stages. Reuse one decoder
  component/file for both stages if P1 confirms compatibility. Generate explicit
  input/finite-sequence bindings, cache specifications and sampler-free prefill
  EndOfInputs, then decode initial-logits bootstrap, attention-mask conventions,
  EOS IDs32000/32001/32007 and declared generated-token budget.
  CPU selects native CPU cache/greedy; CUDA requires native CUDA-plugin cache/
  greedy on an explicit ordinal. The standard demo budget is16 and belongs to
  the constructed definition, not an execution override.
  **CPU tests:** Tiny model packages validate through `sf.ModelPackage`, select
  components, create named chains and match equivalent typed programmatic
  definitions. Cover wrong paths, incompatible/missing layers/types, invalid
  budgets/context bounds, missing assets and native-parser rejection. Check
  generated portable references, provenance and lack of unintended weight copies.
  **CUDA tests:** Tiny matching device fixtures prove required provider policy,
  correct FP16 cache declarations and no CPU sampler/cache substitution.
  Construct real packages only after each graph passes P1; do not infer cache
  dtype from quantized weight precision.

- [x] **P3 - Add the local tokenizer boundary.**
  **Outcome:** Preserve callable encode/decode adapters and provide an optional
  local Hugging Face adapter, with no network or remote-code execution during
  loading. Apply the pinned chat template plus assistant prefix by default;
  raw mode bypasses formatting. Avoid duplicate special-token insertion.
  Reconcile the generation metadata's EOS set with tokenizer EOS32000.
  **CPU tests:** Fake adapters cover template/raw behavior, exact input tokens,
  empty/non-integer/invalid token output, missing local files, incompatible
  metadata, tokenizer exceptions and special-token text rendering. Verify loading
  arguments prohibit network/remote code and preserve caller adapter errors.
  **Real CPU/CUDA tests:** Use downloaded local tokenizer files; verify exact
  prompt-token identity for both EP packages and the reference. Tokenization is
  CPU application work, not a CUDA kernel requirement.

- [x] **P4 - Wire the CLI/helpers to native execution.**
  **Outcome:** Provide explicit setup, offline package-building and run/validation
  entry points with clear help/errors. Exact flag spelling may follow existing
  conventions; separate declared-budget construction from normal offline Run.
  Native ModelPackage -> SessionChain -> StatefulExecution handles all iteration,
  cache and sampling. Helpers expose token IDs, continuation text, termination/
  invocation/dispatch information and deterministic resource ownership.
  **CPU tests:** Tiny fixtures/fake adapters cover CLI and helper success/errors,
  prompt/context bounds, budget0/1/N, first-EOS, full-prompt EndOfInputs,
  sample-only first decode step, no missing/duplicate tokens, close/GC/input
  retention, cancellation and reset-required failures/replay. Verify ordinary
  result text does not duplicate the original prompt or expose special tokens
  while token-level outputs retain observed EOS.
  **CUDA tests:** Tiny device fixtures check identical API behavior with native
  required cache/greedy, ordinary access without `.numpy()` GPU readback, and
  explicit failure for unsupported legacy native-feature requests.

- [x] **P5 - Establish the per-variant reference and qualify real CPU INT4.**
  **Outcome:** Add an ordinary InferenceSession/IOBinding reference using explicit
  test-side state, matching tokenized prompts, weights, masks, exits and greedy
  semantics. It must not call the stateful engine or another GenAI engine.
  Compare the real CPU package to that reference at prefill/decode logit/cache
  checkpoints, final tokens and termination/invocation counts.
  **Tests:** Short prompt with16-token limit, zero/one-token cases, EOS behavior,
  context-bound validation and reset/replay. Exercise first-EOS and other hard-to-
  force boundaries with deterministic fixtures rather than claiming real-model
  coverage that did not occur. Budget variations use explicitly constructed
  definitions/test package metadata, never mutate an active context's policy.
  Reuse verified weight files safely for tests without extra full copies.
  Record precision-aware tolerances before final qualification; investigate
  nonfinite values/outliers and near ties instead of widening bounds to pass.
  Compare subsequent checkpoints only on matching token histories.
  **CUDA preparation:** Reuse the reference design, not CPU model outputs as
  the CUDA FP16 oracle. Different CPU INT4/CUDA FP16 prose is not a failure.

- [x] **P6 - Qualify real CUDA FP16, residency and lifetime.**
  **Outcome:** Repeat the real text/reference/boundary/replay checks on CUDA
  with actual requested ordinal, attention/large-compute placement and native
  cache/greedy dispatch assertions. Document intentional CPU shape/control work.
  Verify cache/logit dtype/device/length and retained snapshots; explicit reference
  readbacks are validation work and excluded from normal-decode transfer claims.
  **Tests:** Native bootstrap first-token accounting and budgets0/1/16; a short
  sanitizer case must include prefill, sample-only bootstrap and at least one
  actual decode-model invocation (budget1 alone is insufficient). Use bounded
  direct workers, not an uninstrumented child hidden behind pytest.
  Run full memcheck/leak, racecheck and synccheck with M13's documented
  `--show-backtrace no`, keeping memory/leak/API/exit checks enabled.
  Dedicated CUPTI traces separate setup/warmup, normal prefill/decode, explicit
  copies and reference checkpoints. Assert compact greedy token/status readbacks
  and no full GPU logits/cache payload readback in normal decoding; classify
  allowed small control transfers from actual records.
  **CPU checks:** Retain per-variant differential/parity tests and report CPU
  preprocessing/timing separately. No speedup or absolute performance gate.

- [x] **P7 - Close coverage, offline/wheel validation and regressions.**
  **Outcome:** Measure100% of all new executable production Python lines,
  including acquisition, inspection, CLI/error/adapter/builder/helper code.
  Use fixture/mocked-error tests for deterministic coverage, actual model runs
  for real qualification, and exact file identities when combining CPU/GPU data.
  No new exclusions or rounded-to100 claims.
  **Tests:** Scoped coverage.py file filters, pyright, lintrunner and whitespace.
  Re-run new fixtures plus M13 CPU27/GPU29 Python suites and directly related
  native/public/package tests; qualify legacy's unsupported-native-feature boundary
  without pretending it supplies native GPU cache/greedy. Run offline demos from
  the existing installed local CPU/plugin wheels outside source/build shadowing.
  This does not claim that repository demo tools ship inside the ORT wheel.
  Preserve unchanged M13 native/cumulative maps only with identity checks.
  If a separately approved amendment adds native production lines, rebuild and
  measure those lines100% too; a0/0 native delta is not a new native coverage claim.

- [x] **P8 - Document and publish one stacked M14 PR.**
  **Outcome:** Explain pinned assets/licenses, explicit download/setup, package
  paths, tokenizer/raw mode, example prompts, declared exits/budgets, masks/cache
  layout, reference method/tolerances, token accounting and offline/error behavior.
  Publish actual timing, CPU/CUDA/dispatch/sanitizer/transfer/coverage evidence,
  exact commands/results and source/object/profile/model hashes.
  **Tests:** Execute documented commands using isolated installed CPU/plugin
  wheels and approved local assets. Confirm no weights/downloaded Python code/
  credentials enter Git. Preserve the PR template; create one OPEN/non-draft PR
  with base `state/m13-python-api`, synchronize its head and keep this checklist
  and SCRATCHPAD current. No automatic merge or M15 execution.

#### Command and evidence conventions

The following are planned validation shapes, not results or implemented CLI
promises. Resolve exact names/paths after approval and record every actual
command, nonzero count and skip.

```bash
source .venv/bin/activate
ROOT="$(git rev-parse --show-toplevel)"
PYTHON="$ROOT/.venv/bin/python"
ASSETS="/home/devpatel/onnxruntime-stateful/phi3-mini-4k-instruct"
# Work from the matching built/installed package environment outside ROOT.
"$PYTHON" -m coverage run --include="$ROOT/samples/python/phi3_demo.py" \
  -m pytest -q "$ROOT/onnxruntime/test/python/onnxruntime_test_python_phi3_demo.py"
# Run actual asset/CUDA workers explicitly; their final selectors will be recorded.
# Full leak checking remains enabled:
compute-sanitizer --tool memcheck --leak-check full --show-backtrace no \
  --error-exitcode 99 "$PYTHON" <actual-direct-CUDA-worker> <approved-local-package>
compute-sanitizer --tool racecheck --show-backtrace no --error-exitcode 99 \
  "$PYTHON" <actual-direct-CUDA-worker> <approved-local-package>
compute-sanitizer --tool synccheck --show-backtrace no --error-exitcode 99 \
  "$PYTHON" <actual-direct-CUDA-worker> <approved-local-package>
pyright samples/python/phi3_demo.py
git diff --check
```

Use the existing CUDA13.0/cuDNN loader paths and actual bundled plugin filename
`libonnxruntime_providers_cuda.so`. Use absolute coverage `--include` filters, not
the side-effecting dotted M13 `--source` probe. Install optional demo dependencies
only in the existing isolated environment after a declared dependency change or
a genuinely missing-dependency validation failure. Rebuild ORT only when necessary,
following the build skill and inherited noncached warning policy.

#### Completion gates

- [x] Requirements interview and Appendix N specification approval.
- [x] Separate execution-plan approval.
- [x] Approved assets downloaded/verified locally; actual graph/aliasing compatibility established.
- [x] CPU/CUDA packages load through native factories and match programmatic construction.
- [x] Local-only tokenizer/chat/raw behavior and caller-adapter/error contracts pass.
- [x] Real CPU INT4 and CUDA FP16 text/reference/checkpoint/boundary/replay runs pass.
- [x] Actual CUDA cache/greedy/compute, resident-data and focused sanitizer checks pass.
- [x] New executable production lines reach measured100%; regressions/type/style/offline examples pass.
- [x] Exact reproducible documentation/evidence and one correct-base OPEN/ready PR are synchronized.

**Implementation authorization (2026-11-02):** The user explicitly approved this
plan and authorized starting M14. P0 passes: all62 parent source/artifact hashes
verify, focused CPU27/84 and isolated installed-plugin29/84 tests pass, actual
CPU/plugin/legacy extension identities are recorded and eight A10080GB devices
have sufficient free memory. No parent rebuild occurred.

P1 is in progress. Both small ONNX graphs and external weight files are now
downloaded outside Git at the approved paths; sizes and published LFS SHA-256
values verify. Actual graphs are IR7/opset14, have INT64 token/mask inputs and
64 rank4 cache tensors, FP32 CPU versus FP16 CUDA, and32 GroupQueryAttention
nodes each. This is structural inspection, not yet a real-runtime compatibility
or generation claim. Metadata/tokenizer acquisition and setup/inspection/builder
tooling/tests are being implemented; no M14 PR or complete coverage is claimed.

**Latest verified checkpoint:** P1-P3 and the real CPU reference gate P5 pass.
All approved graph/data/tokenizer/config/license assets are verified outside Git.
Both portable packages construct two native owned Sessions before atomic manifest
publication. Local-only tokenizer loading initially failed for missing
`transformers`; the optional demo requirements were declared and installed into
the existing `.venv` only (Transformers4.57.6, tokenizers0.22.2, Hub0.36.2).
No core dependency manifest, native runtime or CUDA source was changed.

Real CPU INT4 and CUDA FP16 text runs both produce16 tokens with16 model
invocations and the declared token-limit exit. CPU uses native CPU greedy;
CUDA records16 provider calls and zero CPU sampling. An initial eight-thread
CPU configuration exceeded its300-second diagnostic bound; a timed single-thread
probe completed, showing the sample-only first decode step and15 actual decode
calls. The demo now uses one intra-op thread for reproducible low-contention runs.
The failed diagnostic is retained, not counted as passing qualification.

Actual per-variant stateless-reference reports show **zero maximum absolute
logit/cache checkpoint error**, all64 caches, final cache length29 for the14-token
prompt plus15 executed decode calls, unchanged retained prefill snapshots and
successful reset/replay. CPU placement is all CPU; CUDA has all32 GQA and161
MatMul nodes on CUDA, with only one mask-shape Gather/Cast pair on CPU.

Explicit real-asset tests also pass budgets0/1/16 and typed programmatic/package
construction parity: CPU1test/3subtests in281.43s and CUDA1test/3subtests in88.66s.
No tests in those selectors were skipped. The first setup/execution slice reached
278/278 statements under unit coverage; the newly added reference/placement/error
checks still require fresh coverage closure. Current full offline fixtures pass
10tests/49subtests. CUDA real memcheck has already returned zero errors/leaks;
remaining sanitizer tools, transfer traces, final regressions/docs and publication
are still in progress. Do not treat this checkpoint as complete M14 qualification.

#### Final M14 qualification (2026-11-03)

All implementation/qualification/publication gates P0-P8 pass. PR
[#28](https://github.com/onnxruntime/ort-state/pull/28) is OPEN/ready with base
`state/m13-python-api` (#27). Implementation commit
`111ae8c28fb0014316206a000631a51fe7e6c372` contains the qualified code; the
publication bookkeeping commit changes tracking only.
The [user guide](docs/Stateful_Phi3_Demo.md) documents the actual
[CLI/helpers](samples/python/phi3_demo.py) and optional demo dependencies.
No native runtime or CUDA kernel source was changed.

| Surface | Actual final result |
|---|---|
| Offline M14 fixture/verifier/CLI/runner suite | 15 tests/63 subtests passed |
| Combined installed CPU-wheel M13/M14 suites | 42 tests/147 subtests passed |
| Combined installed GPU-wheel M13/M14 suites | 44 tests/147 subtests passed, including actual M13 CUDA workers |
| Combined legacy M13/M14 offline suites | 42 tests/147 subtests passed; native GPU-feature request explicitly rejected |
| Real CPU INT4 | 1 test/3 subtests for budgets0/1/16, stateless checkpoints, replay and typed/package parity |
| Real CUDA FP16 | 1 test/3 subtests with the same gates; actual raw-prompt bootstrap also passed |
| CPU native/public, unfiltered | 85/85 and153/153 passed |
| Plugin native/public, unfiltered | 157/157 and216/216 passed |
| Legacy native/selected public | 145/145 and76/76 passed with the inherited explicit selector |
| Standalone shared/static package | Each5/5 CTests passed |
| Existing CPU Session/OrtValue and callbacks | 8 and2 passed respectively |
| Existing plugin Session/OrtValue | 7 passed; known stateless negative-device failure separately reproduced |
| Type/style/whitespace | pyright0 errors/warnings; lintrunner and whitespace checks passed |

New executable production Python coverage is **377/377**:
`samples/python/phi3_demo.py` **376/376** and the existing full Python test-runner
integration **1/1 changed statement**. The offline unit suite now runs from the
standard full-build Python runner without model assets or downloads; reduced/
minimal builds retain their existing skip guard. Its orchestration call is
itself exercised with a mocked subprocess runner.

The report explicitly clears coverage.py's default exclusions so the Protocol
declarations are not silently omitted: **zero excluded lines**. Current source
SHA and exact changed-line analysis are saved in
`m14-production-coverage-final.{txt,json}` using `measure_m14_coverage.py`.
There is no new native delta or new device-coverage claim; M13's exact-source/
artifact proof was preserved and all62 parent hashes verified.

Direct actual-model sanitizer workers execute prefill, sample-only bootstrap
and two real decode calls, then access a retained GPU cache after context close:
full memcheck/leak **0 errors/0 bytes**, racecheck **0 hazards/errors/warnings**,
synccheck **0 errors**. The inherited `--show-backtrace no` diagnostic workaround
keeps full memory/leak/API/exit checks enabled.

Dedicated normal-run CUPTI records are exactly:

- HtoD: two8B token feeds; two112B prompt token/mask feeds; one120B and one128B
  decode attention mask.
- DtoH: **three16B compact token/status records =48B**. No full cache/logit
  readbacks occur in normal decoding.
- DtoD:64each of86,016B and92,160B cache-prefix copies. These are allowed device-
  resident copies, not a zero-copy claim.
- Separate explicit-copy capture: one98,304B DtoD clone and one98,304B DtoH copy.
- The normal capture contains1,722 GPU kernels on ordinal0, including three
  `GreedyPartialsHalf` and three `GreedyFinish` launches.

The exact DMA record lists and kernel counts were asserted against exported
SQLite, not inferred from host payload counters. Warmup/setup, reference
checkpoint readbacks and post-capture lifecycle copies are separate.

Final command/results, using the existing isolated Python and matching actual
build/installed wheel with source/test paths and loader configured:

```bash
# CPU42/147; plugin44/147 adds the stateful_cuda.py module; legacy42/147:
"$PYTHON" -m pytest -q \
  "$ROOT/onnxruntime/test/python/onnxruntime_test_python_stateful.py" \
  "$ROOT/onnxruntime/test/python/onnxruntime_test_python_stateful_package.py" \
  "$ROOT/onnxruntime/test/python/onnxruntime_test_python_stateful_threads.py" \
  "$ROOT/onnxruntime/test/python/onnxruntime_test_python_phi3_demo.py"
# Each actual asset selector passed1test/3subtests, with no skips:
ORT_STATEFUL_PHI3_ASSETS="$ASSETS" "$PYTHON" -m pytest -q \
  "$ROOT/onnxruntime/test/python/onnxruntime_test_python_phi3_demo_assets.py" -k test_cpu_real
ORT_STATEFUL_PHI3_ASSETS="$ASSETS" "$PYTHON" -m pytest -q \
  "$ROOT/onnxruntime/test/python/onnxruntime_test_python_phi3_demo_assets.py" -k test_cuda_real
# Each passed with the actual direct worker, not a parent-only pytest process:
compute-sanitizer --tool memcheck --leak-check full --show-backtrace no \
  --error-exitcode 99 "$PYTHON" \
  "$ROOT/onnxruntime/test/python/onnxruntime_test_python_phi3_demo_assets.py" --cuda-sanitizer-worker
compute-sanitizer --tool racecheck --show-backtrace no --error-exitcode 99 "$PYTHON" \
  "$ROOT/onnxruntime/test/python/onnxruntime_test_python_phi3_demo_assets.py" --cuda-sanitizer-worker
compute-sanitizer --tool synccheck --show-backtrace no --error-exitcode 99 "$PYTHON" \
  "$ROOT/onnxruntime/test/python/onnxruntime_test_python_phi3_demo_assets.py" --cuda-sanitizer-worker
# Passed377/377, zero exclusions:
"$PYTHON" "$EVIDENCE/measure_m14_coverage.py"
# Normal/copy worker captures and SQLite exports passed; exact records asserted:
nsys profile --trace=cuda --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true \
  -o "$EVIDENCE/m14-real-trace" "$PYTHON" \
  "$ROOT/onnxruntime/test/python/onnxruntime_test_python_phi3_demo_assets.py" --cuda-trace-worker
nsys profile --trace=cuda --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true \
  -o "$EVIDENCE/m14-real-copy" "$PYTHON" \
  "$ROOT/onnxruntime/test/python/onnxruntime_test_python_phi3_demo_assets.py" --cuda-copy-worker
```

`$ASSETS` is the approved external root; `$EVIDENCE` is the session's persistent
files directory. Exact source/object/profile/model hashes, numerical/placement
reports, raw failure diagnostics and logs are retained under `m14-*`.
Documented acquire/inspect/build/run/validate commands were executed offline
where specified against isolated local CPU/plugin wheels; final manifests
contain per-asset provenance and the original graphs/weights remain hash-identical.

Known limitations remain explicit: the unchanged stateless plugin
`test_set_providers_with_options` still fails its negative-device RuntimeError
expectation and was separately re-run (1failed), not hidden or fixed here.
Direct-CMake legacy imports retain the pre-existing missing wheel metadata
warning. Other GPU architectures, free-threaded Python, additional models,
stochastic sampling and M15 are not claimed. No weights or downloaded Python
code enter the PR, and no milestone is automatically merged.

### M15 - Stable Diffusion local package and image demo

**Status:** [Appendix O](SPEC.md#appendix-o-m15-cuda-sd15-package-and-image-demo-approved)
and the detailed plan below were separately approved on 2026-11-03.
Completed and published OPEN/ready in
[#29](https://github.com/onnxruntime/ort-state/pull/29) on M14/#28.
The [O.7 repair](SPEC.md#o7-cuda-plugin-compatibility-repair-specification-approved)
and R0-R4 gates pass using a fresh byte-verified local wheel. O.8/O.9 and their
separate Q/N execution plans are approved and implemented. Both real guidance
modes pass repeatedly; CPU/plugin/legacy regressions and 504/504 new production
lines pass with zero exclusions. Final qualification scope and publication
are recorded below. The user approved O.10's bounded real GroupNorm racecheck
after an unfiltered two-update attempt timed out; no full-model race pass is
claimed.
**Planning branch/base:** `state/m15-sd15-demo` on M14
`44e512c7719c1e7ebc2e231c47281535c2c52a90` /
[#28](https://github.com/onnxruntime/ort-state/pull/28).
**Approved interview refinement:** The selected `tlwu` exports are CUDA-only.
Retain CPU EP fixture/unit coverage; do not require or claim real CPU inference.
Use existing native DDIM instead of upstream PNDM, retain the required safety
checker, and permit explicit cast/scaling boundary wrappers without changing
original weights. Defaults:512x512 PNG, seed42,20steps, guidance7.5,
empty negative prompt and batched guidance.

**Outcome:** A revision-pinned CUDA-only SD1.5 package produces a safety-checked
512x512 PNG through one native chain, including text encoding, deterministic
CPU-FP32 DDIM/guidance, CUDA denoising, VAE decoding and declared preprocessing/
safety stages. Python supplies the application boundary and output writer,
not the denoising scheduler. CPU EP fixtures remain mandatory.

#### Approval and confirmed inputs

- [x] Interview and explicitly confirm no additional requirements.
- [x] Review and approve Appendix O before detailed planning.
- [x] Prepare the outcomes, actual prerequisites and test gates below.
- [x] Review/approve this plan before model acquisition, dependencies or implementation.

Asset root: `/home/devpatel/onnxruntime-stateful/sd15`.
HF repository: `tlwu/stable-diffusion-v1-5-onnxruntime`, revision
`f22c0e90205a7a2c08bf3e8927b39dc84e4064c2`.
Download only text encoder, UNet, VAE decoder, required safety checker, associated
tokenizer/configuration and authoritative CreativeML OpenRAIL-M license/
attribution. Approximately2.7GB original assets; derived wrappers need explicitly
checked working space. Exclude the VAE encoder and remote Python source.

Default: batch1,512x512 RGB PNG, seed42,20 DDIM updates, guidance7.5, empty
negative prompt and batched guidance. The source PNDM scheduler is provenance
only; the selected native policy remains `sd15_ddim_v1`. Model interfaces/dtypes
are not known until approved download/inspection.

#### Scope and implementation surfaces

- Prefer focused `samples/python/sd15_demo.py` application helpers/CLI, a small
  adapter/model-builder module only if complexity requires it, targeted offline
  and explicit-asset tests, optional demo requirements and
  `docs/Stateful_SD15_Demo.md`. Do not add core tokenizer/image/Diffusers
  dependencies, public package authoring bindings or a second execution engine.
- Reuse M14's verified asset/atomic-publication patterns and M12/M13's existing
  DDIM/package interfaces. Extract only genuinely reusable small helpers if that
  avoids duplication; preserve both consumers' behavior/coverage. Do not make
  the diffusion runtime depend on Phi-3-specific module policy.
- Preserve downloaded originals and weight values. Derived wrappers may add only
  the approved cast/scaling/layout/preprocessing boundaries. Native factories
  remain the package/schema authority.
- A standalone native DDIM execution already exposes initial latents through
  `get_state()` before its first Step. Use that existing initialization path
  when sharing native-seeded initial latents with the independent scheduler
  reference; do not assume Torch RNG matches M12 or add an unnecessary RNG API.
- Exact stage topology follows actual signatures: two text-encoder calls or one
  batched call plus explicit split, native DDIM loop, VAE boundary/decoder,
  safety preprocessing and checker. CPU preprocessing adapters are permitted;
  supplied pretrained-model inference remains CUDA-only.
- If an interface/preprocessing mismatch cannot be satisfied with approved
  adapters and existing native APIs, stop with concrete evidence and request a
  specification/plan amendment. Do not bypass safety, silently choose another
  model or introduce a native feature under this plan.

#### Tasks, dependencies, intended outcomes and tests

| Task | Actual prerequisites | Intended outcome |
|---|---|---|
| P0 | Plan approval | Verified M14 base/proof/environment, storage/GPU preflight and focused baselines. |
| P1 | P0 | Explicit verified local assets/license and actual CUDA interface/compatibility report. |
| P2 | P1 | Approved boundary wrappers, unchanged original weights and original/wrapper equivalence. |
| P3 | P1 | Local CLIP tokenization, image/safety preprocessing and checked PNG-output contracts. |
| P4 | P2, P3 | Native-validated end-to-end package, typed construction and Python CLI/helpers. |
| P5 | P4 | Complete deterministic CPU EP fixture, seed/state/reference and error/lifecycle evidence. |
| P6 | P5 | Real default CUDA image, independent DDIM trajectory/checkpoints, safety and replay/parity proof. |
| P7 | P6 | Actual CUDA placement/transfers and focused real memory/lifetime/race/sync qualification. |
| P8 | P5, P7 | 100% new executable production lines and current regression/type/style/offline evidence. |
| P9 | P8 | Reproducible user guide/image/provenance and one ready stacked PR on M14. |

- [x] **P0 - Preserve the parent and establish bounded baselines.**
  **Outcome:** Reverify local/remote M14 head/base, planning-only modifications,
  installed CPU/plugin/legacy extension identities and all11source/53artifact
  M14 proof entries. Compare intentionally edited planning documents against the
  parent commit. Preserve matching parent proof before any rebuild; do not
  alter old profiles, weights or global GPU settings.
  Recheck space under the approved asset root for originals plus derived graphs/
  metadata and GPU memory for temporary original/wrapper/reference Sessions.
  **Tests:** Focused parent CPU42/147 and plugin44/147 combined Python suites,
  module paths and actual device enumeration. Record actual counts/results.
  Existing native/public85/153 CPU,157/216 plugin and145/76 selected legacy are
  expected baselines, not fresh M15 results until executed. Retain the known
  stateless plugin negative-device and legacy metadata limitations explicitly.

- [x] **P1 - Acquire, verify and inspect the actual CUDA assets.**
  **Outcome:** Implement/test explicit allowlisted revision-pinned setup with
  size/LFS SHA-256 checks, supporting-file provenance, bounded streaming,
  atomic publication and no invalid-cache overwrite. Pin/retain the model card's
  authoritative CreativeML license/attribution. Keep normal commands offline.
  Inspect all model ports, dtypes, dimensions, dynamic/static batch assumptions,
  opsets/custom operators and external/embedded initializer layout.
  **CPU tests:** Tiny/fake assets cover verified reuse, HTTP/network/partial/
  oversized/checksum/write errors, malformed metadata, forbidden files and
  cleanup. Test validators with synthetic graphs, not real CUDA-only models.
  **Real CUDA gate:** Load supplied components only on the requested CUDA device;
  identify text-encoder/UNet/VAE/checker and feature-input contracts. Confirm
  batch2 batched guidance and whether batch1 sequential guidance is supported.
  Record actual VAE scale/layout, safety interfaces, model memory and operator
  placement constraints. Stop for unsupported interfaces rather than substituting.

- [x] **P2 - Build approved cast/scaling/layout boundary wrappers.**
  **Outcome:** Generate derived models only where inspection requires them:
  FP32 DDIM input/output boundaries around optimized denoising, appropriate
  conditioning casts/splits and exactly-once VAE scaling/layout handling.
  Keep original models immutable and preserve every original initializer's
  name/type/shape/value bytes. Record wrapper/model/transform hashes.
  **CPU tests:** Small fixtures verify port renaming, dtype/shape preservation,
  explicit casts, no cast of token IDs to the wrong integer type, one scale
  application, no duplicated/suppressed outputs, original-weight identity and
  safe temporary/atomic publication. Native-load the generated fixture models.
  **CUDA tests:** Compare each real wrapper to its original component with
  independently prepared equivalent feeds; distinguish expected FP16 boundary
  rounding from semantic changes. Validate all required outputs before accepting
  the wrapper into the pipeline. No scheduler-precision change or re-export.

- [x] **P3 - Implement local CLIP, safety/image transforms and PNG handling.**
  **Outcome:** Add caller tokenizer support and a demo-only local HF CLIP adapter,
  with remote Python/network loading disabled. Respect77-token special/padding
  rules, valid empty negative prompt and explicit overlength/empty-positive errors.
  Define VAE-image range/layout conversion and pinned safety preprocessing.
  **CPU tests:** Fake adapters and known images cover token shapes/dtypes,
  padding/bounds, caller exceptions, RGB/channel/range conversion, clipping/
  rounding, invalid/nonfinite tensors, output permissions/encoding/overwrite,
  temporary cleanup and blocked-checker suppression.
  Validate resize/crop/RGB/rescale/normalization against the pinned feature
  configuration:224crop/shortest-edge resize, resample3,1/255 rescale and declared
  CLIP means/stds. Account for any uint8 round-trip used by the source reference.
  **CUDA tests:** Express needed safety preprocessing as declared model adapters
  within the native chain and compare feature tensors/verdicts against the
  independent local image-processor reference on synthetic and real decoded images.
  If adapters cannot satisfy that contract, request an amendment; do not skip
  the checker or insert an undeclared Python callback.

- [x] **P4 - Construct the native package, CLI and reusable helpers.**
  **Outcome:** Build a declarative package with explicit model/adapter variants,
  logical inputs/bindings, native DDIM configuration and matching MaxIterations.
  Use the existing native parser/factories and per-step complete SessionOptions
  replacements for intentional CPU adapters versus CUDA model stages.
  Provide explicit setup/build, offline Run and validation commands. Shape/
  guidance/steps belong to constructed policy; prompt/negative/uint64seed or
  valid supplied latents are execution data.
  **CPU tests:** Tiny end-to-end chains match typed programmatic construction,
  feed conditioning correctly, use latents/state rather than KV cache, maintain
  stage order/counters, suppress downstream work/image output after errors and
  preserve old valid packages on construction failure.
  **CUDA tests:** Initialize the real package through native owned Sessions,
  validate explicit ordinal selection and no pretrained-model CPU substitution.
  Check safety outputs are selected and application output cannot bypass them.
  Do not claim a complete image merely because package initialization passed.

- [x] **P5 - Qualify CPU fixture behavior, native state and reference semantics.**
  **Outcome:** Complete offline unittest fixtures for the entire application/
  builder/adapter/reference/image/error surface. Register them in the existing
  full Python test runner without downloads, keeping reduced/minimal guards.
  **Tests:** Both guidance modes, supported timestep formats, steps1/2/20,
  rejected zero/out-of-profile settings, configured shape/context bounds, seed
  uint64 range/type, supplied-initial latents, state snapshots, retained outputs,
  cancellation at complete-update boundaries, failed execution requiring Reset,
  reset/replay, isolation and explicit close/GC/input ownership.
  Use existing pre-Step latent-state inspection to feed identical initialized
  noise to the independent reference. Seeded/supplied runs must agree.
  Validate exact safety-block/no-save behavior with deterministic fixtures.
  **Scope:** These are CPU EP fixture tests, not real CPU inference claims for
  the CUDA-only pretrained graphs.

- [x] **P6 - Run the real CUDA image and independent differential reference.**
  **Outcome:** Execute default512x512/seed42/20steps/guidance7.5/empty-negative
  native package to a safety-approved RGB PNG. Retain effective policy/model
  hashes, safety verdict, image identity and decoded dimensions/encoding.
  **Tests:** Ordinary CUDA Sessions execute the same original/validated wrapper
  components; Diffusers0.35.1 DDIMScheduler supplies independent CPU scheduler
  arithmetic using the fixed M12 profile and exactly matched initial latents.
  Compare conditioning, each relevant prediction/update, VAE output, safety
  preprocessing/verdict and final conversion with documented precision-aware
  tolerances fixed before final acceptance. Do not compare to upstream PNDM
  or equate Torch random seeds with native initialization.
  Verify20 complete DDIM updates, per-loop invocation counts20batched or40
  sequential when supported, plus separately counted ordinary chain stages.
  Check typed/package parity, seed/supplied-init parity, same-config replay and
  unchanged retained snapshots. Validate sequential real guidance only when
  actual signatures support it; otherwise document the constraint and retain
  CPU sequential fixtures without claiming a real run.

- [x] **P7 - Prove CUDA placement, transfers and real lifecycle safety.**
  **Outcome:** Verify CUDA denoiser/large-compute placement separately from values
  and account for intentional CPU scheduler/control/preprocessing work.
  Dedicated actual-worker traces separate setup/warmup, native denoising, explicit
  reference/checkpoint copies and final image/safety handling.
  **Tests:** For S20/N1/C4/H64/W64 FP32 boundary predictions, expected DDIM DtoH
  is20x131,072B batched or40x65,536B sequential =2,621,440B, subject to confirmed
  interfaces. Establish actual counts from fixture geometry and assert CUPTI
  records; classify conditioning, timestep, VAE/safety and final-image transfers
  separately. Do not claim GPU-resident or zero-copy scheduling.
  Focused real memcheck/full-leak, racecheck and synccheck use direct workers
  that execute at least two complete DDIM updates and required downstream safety/
  image handling, not an uninstrumented child. Use the inherited
  `--show-backtrace no` diagnostic workaround with checks still enabled.
  Check post-parent-close value lifetime and fail explicitly on any new hazard.
  **Approved O.10 refinement:** Full20 memcheck and all-kernel two-update
  synccheck remain mandatory. Real racecheck uses bounded sampling of the
  actual GroupNorm scale/sum160/sum256 variants, with launch-name corroboration
  and explicit limits; no unfiltered/full-model race claim.

- [x] **P8 - Close coverage, regressions, local-wheel/offline and style gates.**
  **Outcome:** Reach measured100% of all new/extracted/modified executable
  production lines, including any shared asset helper and existing-runner call.
  Measure CPU fixtures plus explicit applicable GPU cases with exact current
  source identities, no new/default exclusions hiding declarations or errors.
  **Tests:** Fresh coverage/changed-line reports, scoped pyright/lintrunner/
  whitespace, native/public CPU/plugin and relevant package regressions.
  Re-run M14 offline/parent suites if shared tooling changes; preserve proven
  source/object/profile maps where runtime source is unchanged. No new native
  or device delta is planned; any approved amendment adding code needs fresh
  build/instrumentation and full new-line evidence.
  Execute documented offline commands with the installed local CUDA-plugin
  wheel; CPU installed-wheel tests use fixtures only. Validate unsupported EP/
  device errors and inherited legacy capability boundaries explicitly, without
  claiming real CPU/legacy-native-feature support that was not exercised.

- [x] **P9 - Document, preserve evidence and publish one stacked M15 PR.**
  **Outcome:** Document original/derived-model assumptions and hashes, CreativeML
  license/attribution, CUDA-only scope, explicit setup, local tokenizer, DDIM
  substitution, actual shape/guidance constraints, VAE scale, safety preprocessing/
  blocking, image saving, timing/tolerances and all test/coverage/transfer results.
  **Tests:** Execute the exact documented commands, verify a real non-blocked
  512x512 RGB PNG and its safety/provenance report, preserve parent proofs and
  final source/object/profile/model/image identities, and exclude all downloaded
  weights/code/credentials from Git.
  Inspect/preserve the PR template; publish one OPEN/ready PR with base
  `state/m14-phi3-demo`, synchronize head/base, complete checklists and leave the
  worktree clean. No automatic merge or M16.

#### Command and evidence conventions

The following are planned validation shapes, not results or final CLI syntax.
Resolve actual filenames/selectors after approval and record exact nonzero
counts, skips, image path and every command/result.

```bash
source .venv/bin/activate
ROOT="$(git rev-parse --show-toplevel)"
PYTHON="$ROOT/.venv/bin/python"
ASSETS="/home/devpatel/onnxruntime-stateful/sd15"
# Use a matching installed local wheel outside repository-root shadowing.
"$PYTHON" -m coverage run \
  --include="$ROOT/samples/python/sd15_demo.py,<actual-shared-adapter/helper-files>" \
  -m pytest -q "$ROOT/onnxruntime/test/python/onnxruntime_test_python_sd15_demo.py"
# Execute actual CUDA workers directly, not just a pytest parent process:
compute-sanitizer --tool memcheck --leak-check full --show-backtrace no \
  --error-exitcode 99 "$PYTHON" <actual-direct-SD-worker> <approved-local-package>
compute-sanitizer --tool racecheck --show-backtrace no --error-exitcode 99 \
  "$PYTHON" <actual-direct-SD-worker> <approved-local-package>
compute-sanitizer --tool synccheck --show-backtrace no --error-exitcode 99 \
  "$PYTHON" <actual-direct-SD-worker> <approved-local-package>
pyright samples/python/sd15_demo.py
git diff --check
```

Keep all models/derived wrappers/PNG artifacts under the approved external asset/
output directories and logs/profiles under session evidence; root space remains
constrained. Use existing CUDA13.0/cuDNN loader paths and actual
`libonnxruntime_providers_cuda.so`. Install only declared/missing demo dependencies
in the existing isolated environment, never a published ORT wheel as evidence.
Rebuild native code only if necessary after an explicitly approved amendment;
follow inherited build/test/lint skills and warning policy.

#### Completion gates

- [x] Interview and separate Appendix O specification approval.
- [x] Separate execution-plan approval.
- [x] Pinned assets/license/actual CUDA signatures verified; original weights unchanged.
- [x] Approved wrappers and safety preprocessing independently match their references.
- [x] One native package/typed chain covers text encoding, DDIM, VAE and mandatory safety.
- [x] CPU EP fixtures and all error/lifecycle/no-save contracts pass.
- [x] Real CUDA default image, matching DDIM checkpoints, seeded replay/parity and safety pass.
- [x] Actual CUDA placement, measured intentional transfers and focused sanitizers pass.
- [x] New executable production lines reach100%; regressions/type/style/offline commands pass.
- [x] Exact user documentation/image/provenance and one synchronized OPEN/ready PR exist.

**Implementation authorization (2026-11-03):** The user explicitly approved this
plan and authorized starting M15. P0 passes: all64 parent source/artifact/asset
identities verify, CPU42/147 and plugin44/147 baselines pass, and the actual
eight A10080GB devices have sufficient memory.

All four original models and nine supporting/tokenizer/config/license files
were acquired outside Git and verified against their pinned LFS/Git identities.
Authoritative CreativeML license provenance is pinned to
`CompVis/stable-diffusion-license` revision
`14d42d09bffd871b1666a084fc954a50cff72ac0`.

Actual CUDA load succeeds for all originals. Text-encoder IDs are INT32; floating
model interfaces are FP16. FP32 boundary wrappers preserve all original
initializer fingerprints:148 text encoder,672 UNet,136 VAE and400 checker.
The original VAE input feeds a transpose/decoder without latent scaling; the
derived boundary applies the declared scale once.

A model-only CPU safety preprocessor reproduces Pillow's pinned two-pass
fixed-point bicubic convention and local CLIP image processing exactly on tested
random/gradient/checker images; actual ONNX output images and every feature value
match the independent reference with zero error for two512x512 random cases.
Extracted shared asset primitives retain M14's15tests/63subtests passing.
The real package initializes all native stages successfully.

**Initial compatibility blocker (resolved by R0-R4):** Both the first complete pipeline and the independent
original/wrapper comparison segfault. The text encoder compares exactly; the
unmodified original UNet alone reproduces the failure outside stateful code.
GDB shows CUDA `Shape::ComputeInternal` dereferencing `input == nullptr`.
Disabling memory patterns still segfaults. Disabling graph optimizations instead
fails at CUDA Transpose with `input count mismatch`, so neither is a working
configuration. Full image generation and UNet/VAE/checker equivalence are
unverified; no image or success is claimed.

The separately approved O.7 repair traced the missing Shape input to a missing
upstream SkipGroupNorm residual. Adapter `KernelDef::OpName()` returns the node
display name instead of its operator type; normally named `SkipGroupNorm_1`
therefore disables the kernel's skip behavior. Identical ordinary CUDA graphs
pass with node name `SkipGroupNorm` and fail with an arbitrary node name.
Standalone CUDA Shape/Transpose probes pass. The sole native edit corrects that
existing accessor to `GetOperatorType()`, with no Shape guard, input-count
workaround, scheduler change, model substitution or safety omission.

Evidence is retained in `m15-original-unet-gdb*.log`,
`m15-original-unet-no-*.log`, `m15-first-real-image.log`,
`m15-real-wrapper-equivalence.log`, model/weight-preservation inspection reports
and safety-preprocessing reference reports. Subsequent diagnostics disable
new process core dumps; the first task-owned systemd core is identified in
metadata and has not been deleted or altered. Root space is constrained;
an approved rebuild must preserve parent proof and plan storage carefully.

#### Targeted O.7 repair execution-plan amendment (approved)

**Specification and targeted plan amendments:** Separately approved2026-11-03.
The focused repair/rebuild is authorized. The approved model, FP32 scheduler,
safety policy and final M15 completion gates remain unchanged.

- [x] Review/approve O.7's focused existing-runtime repair scope.
- [x] Separately review/approve this R0-R4 repair plan.

| Repair task | Actual prerequisites | Intended outcome |
|---|---|---|
| R0 | Repair-plan approval, completed P0 and captured failure | Minimal reproduced defect, exact input/constant/metadata diagnosis and safe build storage/proof. |
| R1 | R0 | Small root-cause adapter/Shape repair with preserved existing contracts. |
| R2 | R1 | Focused CPU/CUDA constant/runtime/empty/Shape/Transpose regressions and explicit placement checks. |
| R3 | R2 | Correctly rebuilt/staged local artifacts,100% new native/glue lines and parent regressions. |
| R4 | R3 | Original UNet and wrapper equivalence pass; release P1 compatibility gate and resume P2-P9. |

- [x] **R0 - Reproduce and locate the input-handling defect before editing.**
  **Outcome:** Reduce the original UNet failure to small graphs and inspect the
  actual failing node, logical/runtime input counts, constant-input snapshots,
  optional/missing inputs and tensor/shape validity. GDB currently proves only
  a null Shape input; constant/runtime adapter handling is a hypothesis until
  confirmed. Keep the unmodified model as the integration reproducer.
  **Tests:** Record failing plugin cases independently of stateful execution
  and wrappers, with graph optimization enabled/disabled as useful. Compare
  CPU and applicable legacy CUDA behavior without claiming unsupported CPU
  execution of the real model. Verify the reproducer executes the intended
  CUDA-plugin Shape/Transpose path, not a numerically correct CPU fallback.
  **Storage/proof:** Preserve exact current provider/runtime/test objects and
  profiles before any rebuild. Measured trees are22GB plugin,23GB CPU and12GB
  legacy; root has about1-2GB free and home40GB. If required, relocate only the
  explicitly identified task-owned
  `build/state-cuda-plugin-m11-13.0/RelWithDebInfo` tree into named session storage
  and preserve its original logical path with a symlink after size/hash checks.
  Recheck remaining space before committing to the move; do not recursively
  clean shared/root directories, touch old13.1 trees or modify/delete root-owned
  systemd core files. Stop if storage cannot safely preserve parent proof.

- [x] **R1 - Repair the actual constant/runtime/tensor-metadata contract.**
  **Outcome:** Use existing adapter/C API mechanisms to supply the correct
  logical inputs and tensor shape information. Preserve optional absent-input,
  zero-element tensor, scalar and initializer semantics; do not use DataRaw
  pointer presence as validity unless that is proven to be the correct contract.
  Keep the fix narrow, type-safe and consistent across affected adapters.
  **Tests:** The new minimal reproducer must fail before and pass after the
  change. A required missing input must fail explicitly through standard status
  handling, not dereference null. A null guard that merely changes the real
  UNet crash to an unusable-model error is not completion.
  No new public API, model rewrite/substitution, scheduler type/precision change,
  CPU-model fallback or safety omission.

- [x] **R2 - Add meaningful CPU/CUDA and unaffected-path regressions.**
  **Outcome:** Cover the precise failure plus relevant constant-only,
  constant+runtime, optional-absent, empty-data and normal tensor paths.
  Exercise Shape vanilla/sliced semantics and relevant Transpose input counts
  where implicated, reusing existing EP/provider test infrastructure.
  **Tests:** CPU expected results and actual CUDA-plugin placement/results
  must agree for valid cases. Test error conditions with checked statuses.
  Keep legacy and other existing adapters' semantics unchanged. Confirm nonzero
  selectors and actual plugin library identities; do not broaden unrelated
  operator/kernel behavior to accommodate the reproducer.

- [x] **R3 - Rebuild accurately and close native coverage/regressions.**
  **Outcome:** Reuse the valid RelWithDebInfo CPU/plugin configurations, matching
  CUDA13.0/cuDNN, architectures, warning and coverage flags. Build only required
  runtime/provider/test/pybind targets; regenerate CMake only for source-list/
  configuration changes, preserving noncached `--compile-no-warning-as-error`.
  Do not enable QUICK_BUILD or change kernel routing to make the crash disappear.
  **Tests:** Verify actual changed provider/extension mtimes/hashes, stage the
  rebuilt library and create/install a new isolated local wheel as needed.
  Never test the old M13 installed wheel while claiming the repaired provider.
  Run focused regressions then applicable parent CPU85/153,plugin157/216 and
  selected legacy145/76 expectations, recording actual new counts.
  Measure100% of all new/modified executable native/adapter/glue lines using
  fresh matching objects/profiles; retain untouched parent archives separately
  and audit nonzero profile mismatches. Re-run affected Python suites and
  focused memory/leak/race/sync checks against the actual repaired artifacts.
  Any additional implicated runtime code outside this focused repair needs
  explicit evidence and scope review, not speculative changes.

- [x] **R4 - Prove real compatibility and resume the original M15 plan.**
  **Outcome:** The unchanged original UNet runs successfully on repaired plugin
  CUDA, then all original/wrapper comparisons pass with unchanged weights and
  original SHA. Confirm normal configuration and document any intentional,
  supported settings rather than hiding a workaround.
  **Tests:** Repeat real UNet/VAE/checker equivalence, the full safety-checked
  native pipeline and all original P5-P9 fixture/reference/image/transfer/
  sanitizer/coverage gates. Preserve the failed diagnostics and repair evidence
  in the milestone documentation. Do not mark the image milestone complete
  merely because the null dereference is removed.

**Completed repair diagnosis:** The user approved this plan and authorized continuing
M15. R0 identified the root defect: adapter `KernelDef::OpName()` returns the
node's name rather than its operator type. `GroupNorm` consequently does not
recognize a normally named `SkipGroupNorm_1` node as SkipGroupNorm, omits its
residual output, and a downstream Shape dereferences the unallocated value.
Tiny ordinary-Session tests prove identical SkipGroupNorm graphs pass when the
node name is exactly `SkipGroupNorm` and fail when named `SkipGroupNorm_1`.
Tiny standalone CUDA Shape/Transpose cases pass, disproving the earlier generic
input-count hypothesis.

The narrow repair changes that existing accessor to `GetOperatorType()`.
No Shape null guard, GroupNorm arithmetic, new API or scheduler/model/safety
change was added. Named-node/residual/downstream Shape/Transpose regressions
were added. Parent plugin ELFs are separately preserved with hashes; the exact
22GB task-owned build tree was relocated under session storage with its original
logical path preserved by symlink, freeing root space without touching system
core files. Configuration succeeds at the preserved path.

The first build invocation used the wrong CMake target name and failed before
compilation. The actual target is `onnxruntime_providers_cuda_plugin`; the corrected
incremental native build subsequently completed, as recorded below. The old
M13 installed wheel remains intentionally unchanged.

**Independent M15 CPU/legacy progress while R3 builds:**

- Complete offline application, wrapper, asset, CLI, tokenizer, reference,
  safety/output and typed/package fixtures pass: **29 tests/98 subtests**,
  including the existing M14 fixtures. The direct full-runner SD command also
  passes **14 tests**.
- Combined current CPU Python suites pass **56 tests/182 subtests**. An initial
  invocation omitted the thread-test barrier path and failed those three tests;
  the corrected explicit `ORT_STATEFUL_TEST_BARRIER` invocation passes all tests.
- Current CPU native/public binaries pass **85/153**, and unchanged legacy CUDA
  native/selected-public binaries pass **145/76**. Legacy Python passes
  **56/182**, retaining its existing package-name/version metadata warning.
- Python production coverage is exactly **463/463 new executable lines**, with
  **zero exclusions**: SD helpers315, boundary/preprocessing100, shared assets41,
  modified Phi-3 imports6, new full-runner statement1. Native coverage is still
  pending; this is not a complete M15 coverage claim.
- CPU fixtures verify both guidance modes, steps1/2/20, independent scheduler
  and exact CLIP preprocessing, supplied-latent/typed/package equality, input
  isolation, complete-update cancellation/required Reset, retained snapshots,
  rollback/temporary cleanup, blocked/no-save and exclusive PNG publication.
- All current implementation/regression files pass lintrunner; production
  Python type checks report0errors/0warnings. Supporting metadata publication
  is allowlisted/verified, including upstream model card/config/license; derived
  model and preprocessing hashes are recorded in the manifest.
- The new user guide and optional dependency file are written. Real original/
  wrapper, reference/image, CUDA placement and direct sanitizer/trace workers
  are authored, but have not yet passed against the repaired provider.

**Subsequent fresh-build and image results (2026-11-03):**

- The corrected native plugin/pybind/native/public build completes. A fresh
  local GPU wheel is built and installed in `m15-installed-gpu`; installed
  provider and extension compare byte-for-byte with the rebuilt ELFs. The
  provider SHA-256 is
  `78167b02af406e61aeb2745080fec1f85abf453e30ce236bed5692d6026b727b`.
  An initial extension check guessed a CPython-suffixed name; the actual wheel
  correctly contains `onnxruntime_pybind11_state.so`, and its comparison passes.
- The new operator-identity regression passes **2tests/18subtests** with actual
  CUDA placement in an isolated repaired-provider probe. All names, both dtypes,
  activations, residual values and downstream Shape/Transpose cases pass.
- Original/wrapper CUDA checks now pass for all four components, including
  UNet batch1 and2 and actual large-op placement. Observed maximum errors:
  text0, UNet0.001954/0.003907, VAE0.003907, checker0. The unmodified original
  VAE itself repeats with0.005433 maximum variation; the wrapper's input cast
  matches the independent scaled FP16 feed exactly. Deterministic-compute
  diagnostics do not remove that variation.
- The first complete repaired run reached safety but failed on the source
  checker's incorrectly shared224px/512px symbolic height/width constraints.
  A model-only boundary metadata repair distinguishes the image axes. A tiny
  CPU regression and an actual CUDA checker probe verify it; no computation,
  original weight or additional native runtime code changes.
- The fresh-wheel default pipeline then saves an actual **512x512 RGB PNG**,
  with **25invocations, seed42, unsafe=false**. Image SHA-256:
  `d5e768e59a54e38838eb265ff266c5ae873ae1d5a2f403a014d0fccc866dfa9a`.
  This is a working-image result, not a completed M15 qualification claim.
- Current complete CPU Python suites pass **58tests/182subtests**. New Python
  production coverage is **473/473 with zero exclusions** after the axis fix.
  The changed native getter executes180times and maps to its fresh matching
  function/profile; six unrelated mapping warnings still need an explicit audit.
- Both real guidance-mode qualification tests fail the unchanged closed-loop
  pointwise latent gate. Independent ordinary/ordinary baselines exhibit larger
  discrepancies than native/reference, while same-input local-update errors
  remain below0.0086. The evidence and proposed non-adaptive RMSE/max policy are
  in SPEC O.8. **O.8 is not approved; no gate is waived or silently relaxed.**
  Obtain specification approval and a separately reviewed implementation/test
  plan before changing that validation policy or publishing M15.

**Completed focused repair qualification:**

- Fresh isolated-wheel combined Python tests pass **62tests/200subtests**,
  including the required actual-plugin adapter cases and CUDA stateful tests.
  Fresh native/public executables pass **157/216**, both unfiltered.
- Direct repaired-adapter memcheck passes with **zero errors and zero leaked
  bytes/allocations**; racecheck reports **zero errors/warnings/hazards** and
  synccheck reports **zero errors**. Each actual direct worker executes2tests;
  the established host-backtrace workaround is retained without disabling
  memory, leak or API checks.
- Native coverage audit selects the provider-only raw profile, identifies the
  unique changed adapter getter, and verifies its hash/counter mapping and
  source region23..25 with count180. Changed line24 is covered: **1/1**.
  The six unrelated mapping diagnostics remain visible in retained logs; none
  is the changed function, and no new line is excluded. Current changed
  production coverage is therefore **473Python +1native =474/474**.
- R0-R4 are complete. Remaining P3/P6-P9 gates, including the precision-policy
  decision, real lifecycle/transfer qualification and publication, are not
  waived by this focused repair result.

#### O.8 precision-verification execution plan (approved)

**Specification:** O.8 was explicitly approved2026-11-03. The user supplied no
additional requirements and requested this plan for separate review.
**Implementation authorization:** Explicitly approved2026-11-03. Q1 is in
progress; no additional requirements or plan changes were requested.

**Intended milestone outcome:** Complete M15 with the existing native/runtime/
model scope, a fully independent closed-loop reference plus strict same-input
local-update checks, transparent FP16 uncertainty metrics, complete CPU/CUDA
qualification and one ready PR stacked on M14. No deterministic-kernel project,
automatic tolerance calibration or precision/safety/coverage waiver.

| Task | Actual prerequisites | Intended outcome |
|---|---|---|
| Q0 | Approved O.8, completed R0-R4 | Preserve verified current wheel/image/components and failed/passing numerical baselines. |
| Q1 | Q0, plan approval | Implement the exact approved fixed metrics and local/closed-loop verifier separation. |
| Q2 | Q1 | CPU fixtures verify every new line, exact results and precise metric boundaries/errors. |
| Q3 | Q2 | Both real CUDA guidance modes pass independent differential, safety, replay and construction checks. |
| Q4 | Q3 | Full real CUDA lifecycle/sanitizer and dedicated actual-transfer qualification pass. |
| Q5 | Q2, Q3, Q4 | Current regression/type/style/coverage/documentation evidence and one independently reviewable M15 PR. |

- [x] **Q0 - Preserve the current qualification baseline.**
  **Outcome:** Retain the original-model/initializer identities, fresh provider/
  extension/wheel identities, safety-approved default PNG, exact CPU results,
  current failed closed-loop checks and both ordinary/ordinary uncertainty
  baselines. Preserve M14 artifacts and the completed R0-R4 repair proof.
  **Tests/evidence:** Already recorded CPU58/182, plugin62/200, native/public
  CPU85/153 and plugin157/216, legacy145/76, focused adapter sanitizers,
 473Python/1native covered lines, component/placement reports and numerical JSON.
  These are current baselines, not permission to skip final reruns after Q1.

- [x] **Q1 - Implement the approved verifier without changing generation.**
  **Outcome:** Keep `generate`, native DDIM state, models and CUDA scheduling
  unchanged. Add a small shared validation metric helper using finite,
  shape-compatible arrays and FP64 error accumulation. Fix latent RMSE<=0.01/
  max<=0.2 and normalized-image RMSE<=0.005/max<=0.2 in code; do not expose
  loosening flags or auto-calibrate from a run.
  For each native update, evaluate the ordinary model and independent Diffusers
  step from the preceding native latent tensor, preserving local
  `rtol=0.002, atol=0.02`. Separately advance the fully independent closed-loop
  reference from identical initial noise and check both approved global bounds
  at every update. Compare VAE execution on matching native latents separately
  from closed-reference image drift; keep the documented FP16 component bound.
  Preserve exact same-decoded-image CLIP preprocessing and safety verdict checks.
  Apply the fixed global bounds to real typed/package, seed/supplied and replay
  comparisons, retaining exact immutable-snapshot checks and CPU expected values.
  Report per-update/local/full errors, maxima/RMSE and individual image hashes.
  **CPU tests:** Existing exact fixtures plus metric/error/shape/nonfinite paths.
  **CUDA tests:** Reserved for Q3; a failed check remains explicit and fails
  rather than widening or suppressing a threshold.

- [x] **Q2 - Close CPU tests and exact new-line coverage.**
  **Outcome:** Add focused unittest cases for all Q1 logic, error messages and
  report fields. Keep the deterministic tiny-model results exact and preserve
  existing CLI/build/asset/output safety/cancellation/reset tests.
  **Tests:** Both guidance modes, successful boundary values equal to each
  approved limit, values just over either limit, differently shaped arrays,
  nonfinite data, local-check violations and report/accounting failures.
  Exercise complete native-owned CPU chains, not only mocked success paths.
  Run the existing combined CPU suite and exact direct build-runner command.
  Measure100% of every new/modified executable production line with zero
  exclusions, including shared validation/runner glue. Recheck pyright/lint.
  **CUDA scope:** CPU fixtures do not imply CPU support for the actual models.

- [x] **Q3 - Requalify real CUDA differential behavior.**
  **Outcome:** Both20-step batched and sequential native chains pass the fixed
  approved criteria using the verified installed wheel and original models.
  **Tests:** Repeat both guidance modes with identical initial noise; preserve
  original/wrapper weight/feed equivalence and actual large-op CUDA placement.
  Check conditioning, every same-input local DDIM update, the entire independent
  latent trajectory, matching-latent VAE output, exact real decoded-image CLIP
  features, exact safety verdicts and normalized final-image metrics.
  Verify20 complete updates,25/45 total invocations, typed/package parity,
  supplied-initialization parity, same-config reset/replay, input isolation and
  unchanged retained snapshots. Retain actual RGB PNG metadata/hash/policy and
  observed metric reports from nonzero selectors with no skips.
  **Failure rule:** If a fixed bound fails, investigate and report it; do not
  silently relax the approved policy or replace a model/provider/checker.

- [x] **Q4 - Complete full real lifecycle and transfer proofs.**
  **Outcome:** Real native execution, not just the tiny repair regression,
  passes memory/leak/race/synchronization and actual transfer accounting.
  **Tests:** Update only test workers to use the approved real image/replay
  checks, then run direct Compute Sanitizer memcheck with full leak/API checks,
  racecheck and synccheck. Retain the established `--show-backtrace no`
  host-backtrace workaround; no other checks/suppressions are removed.
  Verify cancellation/required Reset, explicit close/GC and retained values on
  actual models. Capture dedicated native-only denoising after setup/warmup/
  text encoding, separately from explicit reference and final-image reads.
  Assert actual CUPTI prediction records:20x131,072B batched or40x65,536B
  sequential =2,621,440B. Account separately for conditioning/control and other
  stages; do not claim a GPU-resident FP32 scheduler or unexplained zero-copy.
  Reuse the existing profiler/toolchain without global driver/permission edits.
  **Qualification scope:** Apply the separately approved O.10 bounded-race
  refinement; retain full20 memcheck and all-kernel two-update synccheck.

- [x] **Q5 - Finalize and publish only after all gates pass.**
  **Outcome:** Current tested source/artifact/model/image identities, complete
  guide and task/scratchpad records, and one ready PR stacked on M14/#28.
  **Tests:** Rerun related CPU/plugin/legacy Python and native/public suites,
  installed-wheel offline CLI, current metric/helper100% coverage, pyright,
  repository lint/whitespace and relevant package regressions. Preserve known
  parent limitations and every failed diagnostic honestly.
  Refresh docs with exact commands/results, approved tolerances and observed
  values. Inspect/preserve the repository PR template, commit with the required
  coauthor trailer, push the M15 branch, verify remote head/base and clean
  worktree. Do not merge or begin M16. Clean only named task-owned temporary
  intermediates after final proof is retained; keep user assets/parent archives.

**Historical Q1-Q4 results before O.9 approval (not final qualification):**

- O.8/Q0-Q5 were separately approved. Q1's verifier is implemented exactly:
  strict same-input local updates, independent closed-loop RMSE/max checks,
  matching-latent VAE comparison and transparent replay/supplied/image metrics.
  Q2 passes **60CPUtests/193subtests** and **501/501 new Python production
  lines**, zero exclusions; native coverage remains1/1. Types/style are clean.
- Under the unchanged approved bounds, real sequential qualification passes.
  The batched test reaches the typed/package check but fails maximum0.202148
  despite image RMSE0.001992. No threshold was increased or failure relabeled.
- Existing CUDA GroupNorm source confirms nondeterministic inclusive scans and
  atomics; its deterministic-compute option explicitly warns that no
  deterministic CUDA kernel exists. This explains why that prior diagnostic
  did not remove the observed variation.
- Real memcheck fails20,744B/3allocations. An ordinary original-VAE Session
  reproduces it, outside stateful execution and all wrappers. Native allocation
  traces identify cuDNN library frames; DEFAULT/HEURISTIC search diagnostics
  each pass zero-leak memcheck. DEFAULT logs slow GPU convolution fallback,
  so the proposed application option is HEURISTIC, not a silent CPU fallback.
- An exploratory HEURISTIC-only test still fails the accumulated max gate.
  An explicitly labeled HEURISTIC + RMSE-only diagnostic then passes2real
  tests and full real memcheck with zero errors/leaks. These are diagnostic
  evidence, **not approved production qualification**.
- SPEC O.9 proposes explicit demo HEURISTIC configuration and keeping the
  approved hard RMSE/local/exact checks while reporting accumulated maxima.
  Production defaults/max-gate remain unchanged pending specification and
  separate plan approval. Q3/Q4/publication remain blocked; no M15 PR exists.

#### O.9 final configuration/requalification plan (approved)

**Specification:** O.9 explicitly approved2026-11-03; no additional requirements.
**Implementation authorization:** Explicitly approved2026-11-03; no plan changes
or additional requirements. N0-N4 pass; final evidence/publication is complete.

**Intended outcome:** Finish the original M15 milestone using existing
HEURISTIC cuDNN selection and the approved fixed RMSE/local/exact criteria,
with all actual CPU/CUDA/coverage/lifecycle/transfer evidence and one ready PR
on M14. Do not add deterministic kernels, suppress leaks or retry-until-green.

| Task | Actual prerequisites | Intended outcome |
|---|---|---|
| N0 | O.9 approval; preserved Q1/Q2/diagnostics | Freeze current wheel/model/source/failed baseline identities. |
| N1 | N0, plan approval | Apply only the approved demo option/provenance and maximum-reporting changes. |
| N2 | N1 | Exact CPU/error/boundary tests and100% new-line coverage pass. |
| N3 | N2 | Both real CUDA modes and complete lifecycle/actual-transfer proofs pass. |
| N4 | N2, N3 | Current regression/docs/artifact evidence and one independently reviewable M15 PR. |

- [x] **N0 - Preserve the current evidence and decision boundary.**
  **Outcome:** Retain current failed O.8 max comparisons and real/ordinary-VAE
  leak evidence, HEURISTIC exploratory results, source/kernel nondeterminism
  evidence, original-model identities, fresh-wheel identity and working PNG.
  **Tests/evidence:** CPU60/193,501Python/1native covered lines, existing
  CPU/plugin/legacy native/public results and the strict/local baseline remain
  available. Exploratory RMSE-only/HEURISTIC results are not final acceptance.

- [x] **N1 - Apply the approved application-level configuration.**
  **Outcome:** Set `cudnn_conv_algo_search=HEURISTIC` in the demo's actual
  CUDA SessionOptions. Record/check this qualified option in package metadata;
  require explicit package rebuild rather than silently claim an old package
  has the new provenance. CPU options, native API/dispatch build flags,
  original weights, boundary models, DDIM math and safety remain unchanged.
  Keep hard RMSE limits0.01latents/0.005images. Continue reporting absolute
  maxima, but remove only their accumulated trajectory/parity/replay hard gate.
  Strict local-update pointwise, matching-input component and exact CPU/
  preprocessing/safety/ownership/accounting checks are unchanged.
  **CPU tests:** Verify provider option shape/value, old/mismatched metadata
  errors, exact RMSE boundary/just-over failures and reported large maxima
  when RMSE is within budget. Never return success on nonfinite/invalid shapes.
  **CUDA tests:** N3; no global environment/driver or vendor-library edits.

- [x] **N2 - Close focused CPU validation and exact coverage.**
  **Outcome:** Update only directly related tests and guide/provenance text.
  Keep deterministic fixture expectations exact and retain all failure/cleanup,
  cancellation/reset, caller-input, immutable-value and safety/no-save cases.
  **Tests:** Current combined CPU suites and direct full-runner command;
  measure100% of every new/modified production line with zero exclusions,
  including option/metadata/reporting glue. Pyright/lint/whitespace pass.
  Reuse the unchanged fresh native provider's matched changed-line proof.

- [x] **N3 - Complete real CUDA and lifetime/transfer qualification.**
  **Outcome:** Rebuild default and sequential packages with explicit qualified
  HEURISTIC provenance, then pass the approved criteria on both actual modes.
  **Tests:** Original/wrapper/weight equivalence, actual CUDA large-op placement,
  both20-step local/independent trajectories, real decoded-image CLIP features,
  exact safety verdicts, RGB PNG metadata/hashes,25/45invocations,
  typed/package, seed/supplied, reset/replay and unchanged retained snapshots.
  Repeat both modes; retain each result, not only the best trial.
  Run direct real memcheck/full leak/API, racecheck and synccheck workers.
  Keep the established host-backtrace workaround only; zero errors/leaks/
  hazards are mandatory, with no device-reset cleanup masking.
  Capture native-only denoising in both modes and assert actual CUPTI prediction
  copies total2,621,440B, with conditioning/control/stage/explicit reads
  independently accounted. Stop/report any failed fixed criterion; do not
  relax RMSE/local checks, change models/providers or bypass the checker.
  **Approved O.10 refinement:** Racecheck is bounded sampling of each actual
  GroupNorm variant, not unfiltered/full-model race freedom. All-kernel
  two-update synccheck and full20 memcheck remain unchanged.

- [x] **N4 - Final regressions, documentation and publication.**
  **Outcome:** Complete original P3/P6-P9 and Q3-Q5 gates, current exact
  source/artifact/model/image/metric identities, synchronized task/scratchpad/
  guide records and one ready M15 PR stacked on M14/#28.
  **Tests:** Rerun directly related CPU/plugin/legacy Python and native/public
  regressions, installed-wheel offline commands and existing package tests;
  preserve known parent limitations honestly. Final100% new-line counts,
  type/style/whitespace, sanitizer/transfer and numerical results are recorded
  with exact commands. Preserve the repository PR template, include the
  coauthor trailer, verify pushed head/base and clean worktree.
  No automatic merge or M16. Clean only named task-owned intermediates after
  retaining final evidence; preserve user assets, original models, parent
  archives and the task-owned rebuild tree.

#### Finalization evidence and approved bounded qualification

The user authorized finishing M15 and offered `/datadisks/disk7/dtmp` storage.
Fresh final logs/profiles are retained under
`/datadisks/disk7/dtmp/ort-state-m15-819a29ec-final`; older evidence remains in
the session `files` directory. This path currently shares the root filesystem,
so available space was checked rather than assuming a separate mounted disk.
The prior plugin regression passed but profile writes failed with ENOSPC.
Those truncated profiles are not used. A fresh rerun passes64/213, all new
profiles write successfully, and LLVM reports a unique matching getter with
count18 and **no mapping diagnostics**.

| Surface | Executed result |
|---|---|
| Current combined CPU Python | 60tests/195subtests, no skips |
| Current installed plugin Python | 64tests/213subtests, no skips |
| Current legacy Python | 60tests/195subtests; inherited package-metadata warning |
| Direct offline SD unittest entry | 18tests OK |
| CPU native/public | 85/153 pass |
| Repaired plugin native/unfiltered public | 157/216 pass |
| Legacy native/selected public | 145/76 pass |
| Standalone shared/static package | 5/5 each pass |
| Python new production lines | 503/503, no exclusions |
| Native changed getter line | 1/1, matched fresh provider/profile |
| Scoped types/lint/whitespace | 0errors/0warnings; repository lint passes |
| Both real20-step guidance modes | 2tests pass in each of three recorded runs |
| Full20 real memcheck/full leak/API | 0errors; 0bytes/0allocations leaked |
| Full-size two-update all-kernel synccheck | 0errors; 7invocations; replay/cancel/reset/retention pass |

Final installed-wheel CLI generation is offline, unsafe=false,512x512RGB,
seed42/20updates/25invocations. Its PNG is
`/home/devpatel/onnxruntime-stateful/sd15/qualification/sd15-m15-final-cli.png`,
SHA-256 `600eee83de0b9f00ea595c3d93e0376a0c5f753d098731d4982b89a5165312d7`.
Observed one-run setup/execution/output times are12.383/2.491/0.164seconds,
not benchmark/speedup claims.

Final independent CUDA reports:

| Mode | Maximum closed-loop latent RMSE | Image RMSE | Invocations |
|---|---:|---:|---:|
| Batched | 0.005744151 | 0.002029416 | 25 |
| Sequential | 0.007715663 | 0.002610830 | 45 |

Both meet fixed0.01latent/0.005image budgets, strict local-update/component
checks and exact safety/preprocessing/ownership/accounting checks. Final
batched typed/package RMSE is0.006237432latents/0.001958254image. Retain all
three successful runs, earlier failures and exploratory reports, not only
the most favorable numbers.

Actual20-update denoising CUPTI copies are asserted, not estimated:

| Mode | HtoD records (bytes x count) | Prediction DtoH |
|---|---|---|
| Batched | 4x20,131072x20,473088x20 | 131072x20=2621440B |
| Sequential | 4x40,65536x40,236544x40 | 65536x40=2621440B |

Sequential additionally records internal DtoD2Bx40,640Bx200,1280Bx200,
2560Bx560. Setup/warmup/text encoding/reference reads/VAE/safety/final-image
work are outside this native-denoising capture. No GPU-resident scheduler or
zero-copy claim. Independent two-update launch records confirm actual FP16
GroupNorm scale122, sum160114 and sum2568 launches.

**O.10 qualification approval and execution plan:** After the unfiltered
two-update race worker timed out at600seconds, the user explicitly selected
"Approve bounded GroupNorm-variant sampling". Preserve the timed-out/
interrupted attempts as incomplete. Withdraw the old lowercase-only
`groupNorm` filter result: actual symbols contain `GroupNorm`.
Run a complete full-size two-update lifecycle for each bounded, case-correct
variant selection; correlate mangled names with actual CUPTI records, retain
the explicit launch bound and verify zero hazards/errors/warnings. Keep
full20 memcheck and all-kernel two-update synccheck. This does not certify
unfiltered/full-model race freedom. One-update diagnostic workers never
saved images, including the unsafe=true one-update result.

Reproducible command definitions (run in the existing virtualenv, outside
source-package shadowing):

```bash
ROOT=/datadisks/disk9/devpatel/ngenai/onnxruntime
E=/home/devpatel/.copilot/session-state/819a29ec-8ea8-4b2f-9938-e40236ffd239/files
F=/datadisks/disk7/dtmp/ort-state-m15-819a29ec-final
ASSETS=/home/devpatel/onnxruntime-stateful/sd15
source "$ROOT/.venv/bin/activate"
cd "$E"
export LD_LIBRARY_PATH=/home/tianlei/cudnn_9.23_cuda13/lib:/home/tianlei/cuda13.0/lib64
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export ORT_STATEFUL_TEST_BARRIER="$ROOT/build/state-cpu/RelWithDebInfo/libstateful_test_barrier.so"
CPU_SUITES=(
  "$ROOT/onnxruntime/test/python/onnxruntime_test_python_stateful.py"
  "$ROOT/onnxruntime/test/python/onnxruntime_test_python_stateful_package.py"
  "$ROOT/onnxruntime/test/python/onnxruntime_test_python_stateful_threads.py"
  "$ROOT/onnxruntime/test/python/onnxruntime_test_python_phi3_demo.py"
  "$ROOT/onnxruntime/test/python/onnxruntime_test_python_sd15_demo.py"
)
PYTHONPATH="$E/m13-installed-cpu:$ROOT/onnxruntime/test/python" \
  LLVM_PROFILE_FILE="$F/cpu-%m-%p.profraw" \
  python -m coverage run --rcfile=/dev/null --data-file="$E/m15-offline.coverage" \
  --include="$ROOT/samples/python/sd15_demo.py,$ROOT/samples/python/sd15_models.py,$ROOT/samples/python/stateful_demo_assets.py,$ROOT/samples/python/phi3_demo.py,$ROOT/tools/ci_build/build.py" \
  -m pytest -q "${CPU_SUITES[@]}"
# PASS60/195; measure_m15_coverage.py:503/503, exclusions empty.
PYTHONPATH="$E/m15-installed-gpu:$ROOT/onnxruntime/test/python" \
  ORT_STATEFUL_REQUIRE_CUDA_PLUGIN=1 LLVM_PROFILE_FILE="$F/plugin-%m-%p.profraw" \
  python -m pytest -q "${CPU_SUITES[@]}" \
  "$ROOT/onnxruntime/test/python/onnxruntime_test_python_stateful_cuda.py" \
  "$ROOT/onnxruntime/test/python/onnxruntime_test_python_cuda_adapter.py"
# PASS64/213; fresh profiles complete.
PYTHONPATH="$E/m15-installed-gpu:$ROOT/onnxruntime/test/python" \
  ORT_STATEFUL_SD15_ASSETS="$ASSETS" LLVM_PROFILE_FILE="$F/real-%m-%p.profraw" \
  python -m pytest -q "$ROOT/onnxruntime/test/python/onnxruntime_test_python_sd15_demo_assets.py"
# PASS2; both actual guidance modes, no skips.
```

In each actual CPU/plugin/legacy build-output directory,
`./onnxruntime_session_loop_test` passes85/157/145 respectively. Unfiltered
`./onnxruntime_autoep_test` passes153CPU/216plugin; legacy
`./onnxruntime_autoep_test --gtest_filter='AutoEpSelection.CudaEP:*Stateful*:*Ddim*:ModelPackageApiTest.*:*ModelPackageChain*'`
passes76. Existing shared/static package `ctest --output-on-failure` passes5/5.
The inherited stateless plugin negative-device RuntimeError mismatch remains
outside the stateful suites, and is not fixed or hidden by this milestone.

Native coverage merges only valid fresh provider-signature profiles with
`llvm-profdata merge -sparse`; `llvm-cov export` against the byte-matching
installed provider and getter-name filter confirms line24, count18. The
Python coverage script uses the fixed M14 base even after staging/committing.
Final source/object/profile/model/image/report identities and parent proof
verification are retained externally; no weights, local installs or profiles
are committed.

**Completed O.10 race proof (2026-11-04 UTC):** All three complete full-size
two-update/replay/cancel/reset/retention/safety workers pass zero hazards,
errors and warnings. Actual full-worker CUPTI records (including VAE/replay)
show302 scale,228 sum160 and50 sum256 launches. Scale has zero static/dynamic
shared memory; global accesses remain covered by full memcheck. Exact-mangled
sum256 selection checks all50 occurrences. Sum160 uses a measured full-worker
launch window ending at360, selecting actual launches337/359. Nonzero
instrumentation is independently checked: exactly10,240 explicit barrier warp
events for those two sum160 launches and466,944 for all50 sum256 launches,
matching actual grid/block geometry and the barrier at
`group_norm_impl_kernel.cuh:278`. No sampler/threshold/production change.

Low `--launch-count 2` probes reached no selected kernels and are explicitly
discarded, despite their zero-hazard summaries. Do not copy that cap or claim
those probes as qualification. The measured360 window and exact event counts
belong to this toolchain/package; regenerate launch geometry and verify events
when changing the configuration or artifacts.

Exact accepted race commands, using the environment above and
`ORT_STATEFUL_SD15_PACKAGE="$(cat "$F/two-update-package.txt")"`:

```bash
WORKER="$ROOT/onnxruntime/test/python/onnxruntime_test_python_sd15_demo_assets.py"
CS=/usr/local/cuda/bin/compute-sanitizer
SUM160=_ZN11onnxruntime7contrib4cuda22GroupNormNHWCSumKernelI6__halfLi160ELi2EEEvPT_PfPKS4_S8_S8_iiiiiiiib
SUM256=_ZN11onnxruntime7contrib4cuda22GroupNormNHWCSumKernelI6__halfLi256ELi2EEEvPT_PfPKS4_S8_S8_iiiiiiiib
SCALE=_ZN11onnxruntime7contrib4cuda24GroupNormNHWCScaleKernelI6__halfLi2EEEvPT_PKS4_S7_PKfS9_S7_S9_fiiiiifiib
"$CS" --tool racecheck --racecheck-num-workers 8 --racecheck-trace-sync yes \
  --launch-count 360 --kernel-name "kne=$SUM160" --show-backtrace no \
  --error-exitcode 99 python "$WORKER" --cuda-sanitizer-worker
# PASS0hazards/errors/warnings;10240measured explicit-barrier events.
"$CS" --tool racecheck --racecheck-num-workers 8 --racecheck-trace-sync yes \
  --kernel-name "kne=$SUM256" --show-backtrace no --error-exitcode 99 \
  python "$WORKER" --cuda-sanitizer-worker
# PASS0hazards/errors/warnings;466944measured explicit-barrier events.
"$CS" --tool racecheck --racecheck-num-workers 8 --kernel-name "kne=$SCALE" \
  --show-backtrace no --error-exitcode 99 python "$WORKER" --cuda-sanitizer-worker
# PASS0hazards/errors/warnings;302actual launches, no shared memory.
"$CS" --tool synccheck --show-backtrace no --error-exitcode 99 \
  python "$WORKER" --cuda-sanitizer-worker
# PASSall-kernelS2,0errors.
```

Full20 memcheck uses the default20-update package, not the two-update override:
`compute-sanitizer --tool memcheck --leak-check full --show-backtrace no --error-exitcode 99 python "$WORKER" --cuda-sanitizer-worker`
passes0errors/0leaks with `ORT_STATEFUL_SD15_PACKAGE` unset.
The final proof manifest verifies18 current source files, exact new-line
coverage, successful logs/profiles, models/images/metrics/traces, all11source/
53artifact M14 entries and five separately preserved parent native artifacts.
Publication P9/Q5/N4 passes: **[#29](https://github.com/onnxruntime/ort-state/pull/29)
is OPEN/ready**, head `state/m15-sd15-demo`, base `state/m14-phi3-demo`.
Feature commit: `6f0a3e09f84621381d5c9174e663e4ba090a58c9`.
This subsequent documentation-only commit records completed publication.
Final synchronized head/base and clean-worktree verification are retained in
`m15-final-published-pr.json` and the refreshed external
`m15-final-proof.json`. No PR was merged and M16 is not started.

### M16 - C# binding

**Status:** Deferred by the user on2026-11-04; not completed.

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

**Status:** Deferred by the user on2026-11-04; not completed.

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

**Status:** Deferred by the user on2026-11-04; not completed.

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

**Status:** Deferred by the user on2026-11-04; not completed.

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

**Status:** Deferred by the user on2026-11-04; not completed.

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

**Status:** Deferred by the user on2026-11-04; not completed.

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

**Status:** Deferred by the user on2026-11-04; not completed.

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

**Status:** Deferred by the user on2026-11-04; not completed.

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

**Specification:** [Appendix P](SPEC.md#appendix-p-m24-delivered-foundation-qualification-approved),
approved2026-11-04 without changes after explicit confirmation of no additional
requirements. **Execution plan separately approved2026-11-04 without changes.**
F0-F8 pass; published OPEN/ready in [#30](https://github.com/onnxruntime/ort-state/pull/30).

**Outcome:** Qualify the delivered M01-M15 C/C++/Python foundation, establish
reproducible native-only performance/resource baselines, and publish one honest
integrated report and ready stacked PR. M16-M23 remain deferred/unchecked;
this is not completion of the original all-language binding-parity goal.

**Branch/base:** `state/m24-qualification` on M15/#29,
`4fcf16f7d25d6ce8c693839fdf5686c7d3c4c1d8`.
**Cumulative source baseline:** pre-M01 commit
`1988019b6e23ac4def4628cddd603c1bfb504fe8`, parent of the first M01 commit
`64aa026b94bc41476763e23b013a1905296d59cc`.
These commit/build-path lookups are planning inspections, not qualification
results.

#### Approval and scope gates

- [x] Interview M24 requirements and explicitly confirm no additional requirements.
- [x] Review and approve Appendix P before preparing this detailed plan.
- [x] Prepare task outcomes, actual prerequisites, workloads and CPU/CUDA tests.
- [x] Review/approve this execution plan before implementation or measurement.
- [x] Complete F0-F8 and verify a synchronized OPEN/ready PR on M15/#29.

**Implementation boundaries:** Reuse existing native APIs, demos, fixtures,
verified local assets and installed local wheels. Qualification/report tooling,
missing tests and tightly coupled defects within existing approved contracts
are in scope. New public APIs/features/kernels, behavior/ownership changes,
numerical relaxation or model substitution require a separate amendment.
Do not implement deferred bindings, add a Python generation/scheduling engine,
change global GPU settings, discard slow samples, suppress leaks or use
DeviceReset. Reference implementations remain correctness oracles, not
performance comparison targets.

#### Tasks and actual dependencies

| Task | Actual prerequisites | Intended outcome |
|---|---|---|
| F0 | Plan approval, completed M01-M15 | Frozen parent/environment/storage proof and complete requirement/source/evidence inventory. |
| F1 | F0 | Supported native/Python behavior, compatibility and missing regression cases are qualified. |
| F2 | F0 | Minimal reusable native-measurement/report tooling with complete CPU unit/error coverage. |
| F3 | F1 | Fresh CPU/CUDA real-model correctness, package/typed parity and safety pass. |
| F4 | F2, F3 | Applicable lifecycle/device checks and actual transfer/allocation evidence match final artifacts. |
| F5 | F2, F3, F4 | Fixed native-only latency/throughput and separate memory/resource baselines are collected. |
| F6 | F1, F2, F3, F4, F5 | Exact deduplicated cumulative/current-delta coverage and final coupled regressions pass. |
| F7 | F4, F5, F6 | Consolidated report, capability/migration guidance and immutable final proof are complete. |
| F8 | F7 | One independently reviewable ready PR is published on M15 with synchronized tracking. |

F1/F2 may proceed independently after F0. Run latency workloads serially,
without concurrent agent-owned tests/profilers/benchmarks. Any repair after
evidence collection invalidates affected results: rerun the dependent tests,
coverage, device proofs and measurements before completion.

- [x] **F0 - Freeze the delivered baseline and audit evidence readiness.**
  **Outcome:** Verify local/remote M15 base, clean intended scope, existing
  CPU/plugin/legacy builds, installed-extension/provider identities, actual A100
  dispatch, tool versions, optional dependencies and available storage.
  Preserve M15's18source/112artifact manifest, M14's11source/53artifact proof
  and separately preserved parent native artifacts without overwriting them.
  For intentionally edited planning files, verify parent contents with
  `git show` at the pinned parent; do not regenerate an old milestone proof
  against M24's working tree or relabel historical results as fresh.
  Inventory every specification requirement as verified/deferred/blocked,
  every current production delta since the pre-M01 baseline, renames/extractions,
  applicable host/device coverage and matching archive/object/profile paths.
  **CPU tests:** Confirm nonempty test discovery and actual installed module
  paths; identify focused stateless/package compatibility selectors and known
  baseline failures. Audit executable-line classification and archived source
  identity without silently excluding wrappers, errors or tool entry points.
  **CUDA tests:** Enumerate the requested actual plugin device and verify
  loader/provider/build identities, not merely provider registration.
  **Storage:** Use a named M24 evidence directory under the user-offered
  `/datadisks/disk7/dtmp`, checking its actual mount/free space. Keep large
  originals and existing proof/build trees untouched. Recheck capacity before
  rebuilds/traces; never delete shared/root trees, old toolchains or system cores.

- [x] **F1 - Close integrated native/Python behavior and compatibility gaps.**
  **Outcome:** Execute the supported C/C++ and Python programmatic/package
  matrix and add only missing focused regression cases. Cover supported loop
  exits/combinations, step/run and chain/state handoff, cache growth/sharing,
  greedy sampling, DDIM/configuration, capability/subset/provider failures,
  retained snapshots, reset/reuse/isolation, cancellation/required Reset,
  release ordering and downstream suppression after errors.
  **CPU tests:** Existing native/public, standalone shared/static package and
  combined Python suites, plus focused ordinary-Session compatibility cases.
  Preserve established exact deterministic fixtures and public ABI/ownership.
  **CUDA tests:** Actual plugin cache/sampler/chain/DDIM paths with dispatch/
  resource assertions; supported legacy compatibility subset and explicit
  unsupported behavior. A CPU fallback is not a GPU pass.
  **Failure policy:** Reproduce and classify failures before editing. Keep
  the inherited stateless plugin negative-device RuntimeError mismatch and
  legacy metadata warning explicit; do not count them as passed or use them
  to excuse a new regression. Fix only tightly coupled defects; contract changes
  require review. Rebuild only affected targets and preserve old proof first.

- [x] **F2 - Implement the smallest checked measurement/report boundary.**
  **Outcome:** Prefer one focused sample/tool such as
  `samples/python/stateful_qualification.py` and its offline unittest.
  Reuse existing Phi-3/SD helpers and native execution; do not add a generic
  benchmark framework, new core instrumentation API or Python scheduler.
  Use explicit verified local packages, no downloads/remote code/fallback.
  Build isolated M24 budget/guidance package variants using validated existing
  native factories. Where needed, hardlink immutable assets only on the same
  filesystem and keep each manifest separate; never modify linked files in
  place or overwrite M14/M15 manifests/reports.
  Report raw samples, fixed statistics/counts, actual dispatch, phase/timing
  definitions, resource observations/limitations and artifact identities.
  Guard CUDA completion at timing boundaries with existing checked mechanisms;
  never report enqueue-only time as completed GPU execution.
  **CPU tests:** Real tiny native chains plus unit tests for statistics/nearest-
  rank p95, empty/invalid/nonfinite samples, zero-token applicability, counter/
  shape mismatches, tokenizer/input isolation, output/permission/overwrite/
  cleanup errors, failed execution, Reset/reuse and retained values.
  Test CUDA/error plumbing with controlled mocks, not success-shaped defaults.
  New reusable tooling and full-runner glue must have100% new executable-line
  unit coverage, no exclusions. Wire offline tests into the existing full
  Python runner with its reduced/minimal guards and execute that actual entry.
  **CUDA tests:** F3/F4/F5 validate the real completed-work and resource paths;
  mocked CUDA unit paths alone are not integration evidence.

- [x] **F3 - Requalify the actual local demonstrations.**
  **Outcome:** Fresh nonzero CPU/CUDA real-asset checks pass against the frozen
  current artifacts and original-model identities, with isolated M24 outputs.
  **CPU tests:** Real Phi-3 INT4 at budgets0/1/16 using the established prompt,
  ordinary-Session correctness references, cache/state checkpoints, seed/input
  ownership and package/typed parity. SD1.5 uses deterministic CPU fixtures
  only, covering both guidance modes and mandatory blocked/no-save behavior.
  **CUDA tests:** Real Phi-3 FP16 budgets0/1/16 with native KV/greedy dispatch,
  state/retention/reset and reference checks. Real SD1.5 default512x512,
  seed42/20updates/guidance7.5/empty negative, both batched and sequential,
  independent DDIM/local/component checks, exact preprocessing/safety,
  typed/package and seeded/supplied/reset parity,25/45invocations and safe PNGs.
  Preserve HEURISTIC provenance and all approved M14/M15 numerical criteria.
  Retain every outcome; no retry-until-green, threshold changes or unsafe save.

- [x] **F4 - Close applicable device/lifetime and actual resource proofs.**
  **Outcome:** Current source/object/worker mappings prove applicable CPU/
  CUDA ownership, allocator/provider lifetime and actual device work.
  **CPU tests:** Resource/failure/reset/release-order/retained-value tests and
  existing applicable host diagnostics. Distinguish intentional allocator/
  context retention from a leak; do not infer leak freedom from RSS alone.
  **CUDA tests:** Established cache/sampler/chain lifetime workers and actual
  dispatch/copy accounting. Retain SD1.5 full20 memcheck/full-leak/API scope,
  all-kernel full-size two-update synccheck and approved bounded GroupNorm race
  scope. Prove nonzero instrumentation; rederive launch windows/event counts
  if source/artifacts/worker/configuration change. No unfiltered full-model
  race-freedom claim or acceptance of zero-match/zero-instrumentation probes.
  Assert native-denoising prediction DtoH exactly2621440B in each20-update
  SD guidance mode, and classify conditioning/control/internal copies and
  explicit reference/output reads separately. Derive Phi-3 expected counts
  from the actual configured workload, not an unrelated historical trace.
  Keep sanitizer/profiler observations separate from latency samples and
  preserve stopped/failed attempts honestly.

- [x] **F5 - Collect fixed native-only performance/resource baselines.**
  **Outcome:** Execute the eight workload rows below with no ordinary-Session
  performance baseline, cross-precision speedup claim or performance gate.
  Use the existing package-backed native execution through the demo's Python
  boundary. C/C++ and typed construction remain correctness-qualified paths,
  not additional performance-comparison targets.
  For each row, use a fresh worker for setup/first completed native execution,
  then the same execution with explicit Reset for2warmups and5measured
  completed runs. Record the cold result separately, all5raw steady samples,
  median, nearest-rank p95(maximum for5samples), actual generated/output counts
  and throughput. Do not divide by configured token budgets when actual counts
  differ, or fabricate throughput for a zero-token run.
  Report tokenization/rendering/PNG publication outside native timing. Any
  dedicated step-observed phase pass is separately labeled; its synchronization/
  boundary overhead must not be presented as a decomposition of a different
  Run-path measurement.
  Collect memory/allocation/copy observations separately using the same fixed
  configuration, explicit baseline/peak/post-reset/post-close boundaries,
  existing telemetry/profiling and at least two observed reuse cycles after
  warmup. Disclose sampling intervals, process versus allocator/driver scope,
  instrumentation overhead and unavailable/inapplicable counters. Required
  measurement gaps remain open, not fabricated zeros.
  **CPU tests:** Phi-3 CPU rows and measurement/report invariants; no pretrained
  SD CPU baseline. **CUDA tests:** Phi-3/SD rows on actual A100 ordinal0 with
  checked completion and safety. Preserve existing explicit thread/provider
  options, log observed contention, and do not alter global clocks/affinity,
  flush shared OS caches or interfere with other users' processes.

- [x] **F6 - Close exact cumulative coverage and final regressions.**
  **Outcome:** A deduplicated current-source manifest since the pinned pre-M01
  baseline shows100% covered/new executable production lines across delivered
  runtime/APIs/Python/glue/package/demo/tooling and applicable CUDA device code,
  including all M24 deltas. Separate exact host/device/per-file counts.
  Verify archived proof only for unchanged code with valid source/object/
  profile mappings; regenerate missing/invalid evidence. Do not sum historical
  milestone totals, infer device lines from host traces, accept truncated/
  mismatched profiles, round percentages or exclude reachable error/entry paths.
  **CPU tests:** Current unit/coverage suites including the new tooling,
  native/public and shared/static package regressions, direct offline runner,
  scoped types/repository lint/whitespace and documented local-wheel commands.
  **CUDA tests:** Current plugin native/public/Python and relevant legacy
  regressions; matched host/device collection and affected F3/F4 rechecks.
  Verify nonzero selectors, actual artifacts and unsupported cases explicitly.
  Any unresolved executable production line or required validation blocks F7/F8.

- [x] **F7 - Publish the qualification report and finalize user guidance.**
  **Outcome:** A compact committed `docs/Stateful_Qualification.md` (or an
  existing directly related report if it fits better) provides the requirement/
  PR/test traceability, exact executed commands/results, cumulative coverage,
  eight native baseline rows, environment/model/source/ELF identities, resource/
  transfer data and all limitations. Update existing examples, ownership and
  API/package migration guidance and capability/provider/platform/language
  matrices. Deferred bindings remain deferred, not passed.
  **CPU/CUDA validation:** Check report values against raw structured results,
  counts/units/shapes, actual safety-approved outputs, profiler records and
  source/object/profile hashes. Preserve parent proof and immutable final
  manifests outside Git. Keep weights, large traces/profiles, local installs
  and credentials out of commits. Missing hardware/platform/counters must
  remain explicit; required in-scope gaps block completion.

- [x] **F8 - Publish one independently reviewable M24 PR.**
  **Outcome:** Commit only the approved incremental scope with the required
  coauthor trailer; inspect/preserve the repository PR template; include exact
  validation results, native-only baseline interpretation and deferred scopes.
  Push `state/m24-qualification`, open ready-for-review on
  `state/m15-sd15-demo`/#29, and verify remote head/base, OPEN/non-draft state,
  updated task/scratchpad records and clean worktree.
  **CPU/CUDA tests:** F1-F6 must have passed against final relevant source/
  artifacts. Documentation-only publication tracking needs no artificial EP
  rerun; refresh proof identities after final commits.
  No automatic merge, binding implementation or completion claim for M16-M23.

#### Fixed workload and measurement policy

| ID | Actual model/provider | Configuration |
|---|---|---|
| L-CPU-0/1/16 | Approved Phi-3 CPU INT4 / CPU EP | Prompt `Answer briefly: what is 2 + 2?`; respective generation budgets0/1/16 |
| L-CUDA-0/1/16 | Approved Phi-3 FP16 / CUDA plugin | Same prompt and budgets, device ordinal0 |
| D-CUDA-B | Approved SD1.5 / CUDA plugin plus declared CPU preprocessing/DDIM | 512x512,seed42,20updates,guidance7.5,empty negative,batched |
| D-CUDA-S | Same SD1.5 configuration | Sequential guidance; no other policy change |

SD prompt: `A peaceful mountain lake at sunrise, soft golden light, landscape painting`.
Each of these eight rows has1cold completed run,2warmups and5measured completed
steady runs. All measured outcomes are recorded, not only the best; shorter
valid LLM output uses its actual token count. The cold label means fresh native
process/execution state, not forced eviction of shared filesystem caches.
Preserve the current demo thread settings and exact variant/provider options.
No new model acquisition, benchmark against stateless reference loops, numerical
performance threshold or architecture-wide performance claim.

#### Validation entry points and inherited reference counts

These are **planned M24 commands, not executed M24 results**. Run in the existing
virtualenv and a dedicated directory outside source-package import shadowing.
Resolve new tooling/fixture selectors against their implemented entry points
and record exact actual results. Do not install dependencies speculatively.

```bash
ROOT=/datadisks/disk9/devpatel/ngenai/onnxruntime
E=/home/devpatel/.copilot/session-state/819a29ec-8ea8-4b2f-9938-e40236ffd239/files
Q=/datadisks/disk7/dtmp/ort-state-m24-819a29ec
source "$ROOT/.venv/bin/activate"
# Create/check the dedicated Q directory only after plan approval.
cd "$Q"
export LD_LIBRARY_PATH=/home/tianlei/cudnn_9.23_cuda13/lib:/home/tianlei/cuda13.0/lib64
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export ORT_STATEFUL_TEST_BARRIER="$ROOT/build/state-cpu/RelWithDebInfo/libstateful_test_barrier.so"
export LLVM_PROFILE_FILE="$Q/cpu-%m-%p.profraw"
CPU_SUITES=(
  "$ROOT/onnxruntime/test/python/onnxruntime_test_python_stateful.py"
  "$ROOT/onnxruntime/test/python/onnxruntime_test_python_stateful_package.py"
  "$ROOT/onnxruntime/test/python/onnxruntime_test_python_stateful_threads.py"
  "$ROOT/onnxruntime/test/python/onnxruntime_test_python_phi3_demo.py"
  "$ROOT/onnxruntime/test/python/onnxruntime_test_python_sd15_demo.py"
)
PYTHONPATH="$E/m13-installed-cpu:$ROOT/onnxruntime/test/python" \
  python -m pytest -q "${CPU_SUITES[@]}"
# Add the new offline qualification unittest in F2/F6.
PYTHONPATH="$E/m15-installed-gpu:$ROOT/onnxruntime/test/python" \
  ORT_STATEFUL_REQUIRE_CUDA_PLUGIN=1 LLVM_PROFILE_FILE="$Q/plugin-%m-%p.profraw" \
  python -m pytest -q "${CPU_SUITES[@]}" \
  "$ROOT/onnxruntime/test/python/onnxruntime_test_python_stateful_cuda.py" \
  "$ROOT/onnxruntime/test/python/onnxruntime_test_python_cuda_adapter.py"
# Real suites run in isolated M24 output/package locations, never overwriting
# old qualification reports; retain their ordinary-Session correctness checks.
# Explicit local assets:
# ORT_STATEFUL_PHI3_ASSETS=/home/devpatel/onnxruntime-stateful/phi3-mini-4k-instruct
# ORT_STATEFUL_SD15_ASSETS=/home/devpatel/onnxruntime-stateful/sd15
```

In the corresponding existing CPU/plugin/legacy build-output directory, use
`./onnxruntime_session_loop_test` and the applicable
`./onnxruntime_autoep_test` selectors. Legacy's supported public selector is
`--gtest_filter='AutoEpSelection.CudaEP:*Stateful*:*Ddim*:ModelPackageApiTest.*:*ModelPackageChain*'`.
Run shared/static standalone package tests with
`ctest --test-dir "$ROOT/build/state-package-m10" --output-on-failure` and
`ctest --test-dir "$ROOT/build/state-package-m10-static" --output-on-failure`.
Use per-build fresh `%m-%p` profile paths; do not overwrite default/parent
profiles or merge module signatures without verified mappings.

The M15 reference counts are CPU Python60/195, plugin64/213, legacy60/195;
CPU native/public85/153, plugin157/216, legacy145/76; package5/5each and direct
SD unittest18. They are **inherited expectations, not new M24 passes**.
Record actual final counts, added tests, skips and all diagnostics.

#### M24 completion checklist

- [x] F0 parent/environment/source/requirement/proof inventory passes.
- [x] F1 supported integrated behavior/compatibility and error/lifetime matrix passes.
- [x] F2 new reusable measurement/tool/runner code has100% unit coverage.
- [x] F3 real Phi-3 CPU/CUDA and SD1.5 CUDA/CPU-fixture correctness passes.
- [x] F4 applicable lifetime/sanitizer/actual dispatch/transfer evidence passes.
- [x] F5 all eight native-only baseline rows and required resource metrics persist.
- [x] F6 exact deduplicated cumulative production-line coverage is100%; final regressions/type/style pass.
- [x] F7 complete report, migration/capability guidance and final immutable proof exist.
- [x] F8 one synchronized OPEN/ready M24 PR exists on M15/#29; worktree is clean.

**Completion interpretation:** Qualify the delivered M01-M15 foundation under
Appendix P. M16-M23 remain deferred and their binding-parity requirements remain
outstanding. Do not mark the entire original all-language feature complete.

**F0 executed readiness (2026-11-04):** M15's18pinned source blobs and112
artifacts/assets verify without changing the old proof. M14's11source/53artifact
entries and five separately preserved parent native files also verify. Current
CPU/plugin imports resolve to the intended isolated local installations; the
plugin enumerates all eight actual A10080GB devices including ordinal0.
Storage preflight records19GB root/51GB home free; offered disk7/dtmp shares
the root mount, not a separate volume. New evidence is isolated under
`/datadisks/disk7/dtmp/ort-state-m24-819a29ec`.

The pre-M01 delta has126paths:60native production files,1CUDA device source,
8Python production files,41test/test-library fixtures and16documentation/
build/dependency metadata files. All93original requirement IDs are inventoried
as pending actual M24 qualification, with the language portions of SCOPE-2/
API-1/API-2 explicitly deferred. The old native cumulative5412/5412 report,
M15getter1/1 and25/25 actually emitted CUDA source locations are inherited
evidence candidates, not a new final cumulative denominator. Current code/
object/profile mapping and Python/sample/tool coverage still require F1/F6.

Actual environment: Python3.12.16, NumPy2.5.4, ONNX1.23.2, pytest9.2.0,
coverage7.17.0, Transformers4.57.6, Pillow13.0.0, Diffusers0.35.1,
Torch2.8.0+cpu, LLVM18.1.8, Compute Sanitizer/Nsight Compute2025.3 and
Nsight Systems2025.3.2. An initial metadata check incorrectly requested the
separate pytest-subtests distribution; pytest9.2 supplies built-in subtests.
The corrected check confirms that implementation; no dependency was installed
or restored. No new runtime or real-model qualification pass is implied by F0.

**F1/F2 executed results:** Fresh CPU Python60/195, native/public85/153;
plugin Python64/213, native/public157/216; legacy Python60/195(warning retained),
native/public145/76; shared/static package5/5each all pass. Eight focused
ordinary-Session provider/ownership/inference tests pass on each CPU/plugin
runtime. Tiny native feedback/bootstrap/both-DDIM examples execute correctly on
CPU and legacy CUDA. The separate inherited
`TestInferenceSession.test_set_providers_with_options` still fails with
`RuntimeError not raised` for the plugin negative-device case, matching the
preserved baseline: explicitly **one known failure, not a passed test**.

New measurement tooling uses checked completed-work synchronization,
1cold/2warmup/5steady samples, separate step/resource passes, process RSS/glibc
allocation observations and explicit NVML visibility/scope. It publishes only
safe images and exclusively linked JSON, rolling back partial new links.
Unit tests caught an in-place rewrite of a previously linked staging report;
unique temporary report files fix that root cause and preserve existing output.
GPU visibility is checked after completed warmup, not presumed during lazy
Session construction. No native runtime/API/kernel code changed.

Current offline tooling/Phi-3 tests pass28tests/85subtests; direct full-runner
qualification entry passes13tests. New production helper255/255 and new
runner2/2 executable lines are covered, zero exclusions; the older tiny sample
is now unit-covered63/63. Scoped types report0errors/0warnings and repository
lint passes. These are unit/delta counts, not the final cumulative F6 result.
Test-only `ORT_STATEFUL_QUALIFICATION_OUTPUT` isolates linked variant packages/
real images from earlier milestones; empty values fail explicitly and unit tests
prove original manifests are unchanged. Real numerical/lifecycle/performance
gates remain pending F3-F6.

**F3 executed real-model correctness:** CPU Phi-3 passes1test/3budget subtests
in278.61seconds; CUDA Phi-3 passes1/3 in89.87seconds. Both retain64cache
checkpoints, independent references, replay and typed/package parity. Budgets
0/1/16 produce1/1/16actual model invocations; zero generation still includes
the separately configured prefill, not a fabricated zero-work chain.
Both CUDA SD1.5 tests pass in57.42seconds, with unsafe=false and25/45invocations.
Batched maximum latent RMSE0.003967436/image RMSE0.001455491; sequential
0.004360999/0.001573004. Local/component/exact preprocessing/safety and
seed/supplied/replay/retained-value criteria remain unchanged and pass.
Reports/images/variant links are isolated under
`/home/devpatel/onnxruntime-stateful/m24-qualification-819a29ec/f3`.
These test durations are qualification elapsed times, not F5 native baselines.
All61native/device source files match M13 byte-for-byte except the already
qualified M15 operator-name accessor; final mapped/cumulative proof is still F6.

**F4 executed device/resource proof:** Seven selected actual CUDA cache/sampler/
chain lifetime tests pass each memcheck/full leaks, racecheck and synccheck,
all zero errors/leaks/hazards. Full real Phi-3 memcheck/full leaks passes0/0;
its original automatic-parallelism full-worker racecheck and synccheck also
pass0errors/hazards. An earlier eight-worker race attempt hit its600second
budget and remains **incomplete**, not passed; the completed original-command
run changes analysis parallelism only, not scope, checks or numerical criteria.
Full20 real SD1.5 memcheck/full leaks passes0/0; full-size two-update all-kernel
synccheck passes0errors. Current bounded GroupNorm racecheck passes all three
variants with actual geometry independently rederived:2sum160 launches/
10240explicit barrier events,50sum256/466944events,302scale launches with
zero shared memory. The environment-specific360window was verified anew,
not blindly copied.

Fresh actual Nsight Compute source counters cover25/25emitted CUDA greedy
source locations across all three kernels using the previously authorized
profiler-only `sudo -n -- ncu` procedure; no global setting was changed.
Actual CUPTI asserts normal Phi-3 DtoH48B(three16Brecords), separate explicit
98304Bclone/readback, and SD prediction DtoH2621440B per guidance mode.
All HtoD/internal DtoD records also match. Explicit-copy capture has no compute
kernel table by design; it is validated as a copy-only window, not a missing
dispatch silently ignored. Full-worker allocation telemetry distinguishes
pinned-host, dynamic-device and static-device storage via exported enums.

Read-only current-source coverage candidates deduplicate5413host native,
1931Python and25device locations=7369/7369, with original M13 object/profile
cohort verified(22source/40artifact entries; five preserved original GPU ELFs).
Real CUDA subprocess coverage supplies the device-value NumPy rejection path;
parent-only Python coverage is not treated as sufficient. Final F6 collection/
identity checks and F5 measurements remain pending.

**F5 executed fixed native-only measurements:** All eight rows complete
serially, each1cold/2warmup/5steady native runs plus separate resource and Step
workers. Source and manifest identities remain frozen. No ordinary-Session
performance comparison, concurrent owned profiler/test, global tuning or best-
sample filtering was used. Existing RelWithDebInfo coverage-instrumented builds
were measured; profile-file output was disabled for timing, not instrumentation
or unit coverage.

| Workload | Steady median(s) | p95(s),5samples | Actual throughput |
|---|---:|---:|---:|
| Phi-3 CPU INT4,budget0 | 4.015727 | 4.155346 | Not applicable:no generated tokens |
| Phi-3 CPU INT4,budget1 | 4.047851 | 4.182928 | 0.248120tokens/s |
| Phi-3 CPU INT4,budget16 | 10.677958 | 10.847453 | 1.497301tokens/s |
| Phi-3 CUDA FP16,budget0 | 0.008754 | 0.008786 | Not applicable:no generated tokens |
| Phi-3 CUDA FP16,budget1 | 0.008794 | 0.008806 | 113.792169tokens/s |
| Phi-3 CUDA FP16,budget16 | 0.132899 | 0.133245 | 120.330205tokens/s |
| SD1.5 CUDA,batched | 0.685177 | 0.689988 | 1.456087images/s |
| SD1.5 CUDA,sequential | 0.803387 | 0.809567 | 1.248061images/s |

Five-sample nearest-rank p95 is the maximum sample, not a statistically strong
tail estimate. Different quantization/provider rows are independent baselines,
not speedup comparisons. Raw cold/setup/tokenization/output/reset observations,
all samples, step-observed first-token/phase times, RSS/glibc and NVML snapshots
are retained in `f5-collected-results.json` and per-worker JSON/logs.

Whole resource-worker GPU traces independently derive dynamic allocation peaks
from exported allocation/free enums and address lifetimes: Phi-3 CUDA0/1
17189459204B(30alloc/free each), CUDA16 17230506504B(36each); SD batched
7579871508B(63each), sequential6502983956B(72each). All dynamic allocations
have matching frees and zero traced dynamic bytes remain at process end.
Static device module storage is reported separately, not misclassified as a
leak. Pinned-host events are not counted as device allocations. CPU process RSS
peaks are6633.81/6634.18/6675.25MiB for the three CPU rows; these include
application/setup state, not just live runtime cache. Exact byte/count/scope
data is in `f5-baseline-summary.json`; sanitizer evidence remains independent.

**F6 final executed closure:** CPU73tests/217subtests, plugin77/235,
legacy73/217(with inherited metadata warning) pass; native/public CPU85/153,
plugin157/216, legacy145/76 pass; standalone shared/static5/5each and direct
offline qualification13tests pass. Scoped types0errors/0warnings, repository
lint and whitespace pass. Final fresh Python coverage includes main processes
and both real CUDA unit subprocesses.

Exact deduplicated cumulative production coverage is **7369/7369**, no
exclusions: native host5413/5413, Python1931/1931, actual fresh CUDA source
counters25/25. The M24delta is257/257(helper255 plus full-runner2). Native
coverage uses verified unchanged source/object/profile cohorts plus the fresh
current accessor export; it is not a sum of milestone claims. All60native host
files and all9current production Python files are enumerated, including
declaration-only files with no executable delta and the older tiny sample.
Final proof is `f6-cumulative-coverage.json`; no required line remains uncovered.

**F7 report/proof completed:** `docs/Stateful_Qualification.md` records all eight
raw/summary native baseline rows, separate Step/resource observations, exact
validation results, all93original requirement IDs, capability/migration/ownership
guidance and explicit deferred/known-failure scopes.
`docs/Stateful_Qualification_Coverage.json` commits the70file exact per-file
SHA/count ledger. Structured raw results, maps, profiles and model/image
identities remain outside Git. `f7-source-artifact-proof.json` validates81source
identities and598artifact/asset entries, with no changed source/package identity
during measurements. Linux RSS/current high-water observations and static/
dynamic/pinned GPU memory are not conflated with leak detection. Publication
F8 remains pending; no milestone/binding or PR merge beyond scope is implied.

**Exact F4 resource selector and collection forms (executed, passed):**

```bash
# In the CUDA-plugin native build directory, matching CUDA/cuDNN loader set:
FILTER='CudaStatefulTest.NondefaultStreamCompletionBeforeReadback:CudaStatefulTest.CConsumerAndRetainedValuesAfterSessionRelease:CudaStatefulTest.CallerAllocatorAndManagerOutliveBackendSession:CudaStatefulTest.GreedyMatchesCpuAcrossShapesDtypesAndReductionBoundaries:CudaSamplingApiIntegrationTest.C11GpuSamplingResultsSurviveSessionAndLibraryRelease:CudaBootstrapPackageTest.C11AndGpuValuesSurviveOwnedSessionsAndLibraryUnregistration:CudaDdimPackageTest.CpuLatentsSurviveOwnedSessionsAndLibraryUnregistration'
CS=/usr/local/cuda/bin/compute-sanitizer
"$CS" --tool memcheck --leak-check full --show-backtrace no --error-exitcode 99 \
  ./onnxruntime_autoep_test --gtest_filter="$FILTER"  # PASS7,0errors/0leaks
"$CS" --tool racecheck --show-backtrace no --error-exitcode 99 \
  ./onnxruntime_autoep_test --gtest_filter="$FILTER"  # PASS7,0hazards
"$CS" --tool synccheck --show-backtrace no --error-exitcode 99 \
  ./onnxruntime_autoep_test --gtest_filter="$FILTER"  # PASS7,0errors
# Actual direct model workers, not subprocess-owning pytest wrappers:
PHI_WORKER="$ROOT/onnxruntime/test/python/onnxruntime_test_python_phi3_demo_assets.py"
SD_WORKER="$ROOT/onnxruntime/test/python/onnxruntime_test_python_sd15_demo_assets.py"
"$CS" --tool memcheck --leak-check full --show-backtrace no --error-exitcode 99 \
  "$ROOT/.venv/bin/python" "$PHI_WORKER" --cuda-sanitizer-worker  # PASS0/0
"$CS" --tool racecheck --show-backtrace no --error-exitcode 99 \
  "$ROOT/.venv/bin/python" "$PHI_WORKER" --cuda-sanitizer-worker  # PASS0hazards
"$CS" --tool synccheck --show-backtrace no --error-exitcode 99 \
  "$ROOT/.venv/bin/python" "$PHI_WORKER" --cuda-sanitizer-worker  # PASS0errors
# SD full20: ORT_STATEFUL_SD15_PACKAGE unset; full-size S2 sync/race:
# ORT_STATEFUL_SD15_PACKAGE="$(cat "$Q/f4-two-update-package.txt")".
"$CS" --tool memcheck --leak-check full --show-backtrace no --error-exitcode 99 \
  "$ROOT/.venv/bin/python" "$SD_WORKER" --cuda-sanitizer-worker  # PASSfull20,0/0
"$CS" --tool synccheck --show-backtrace no --error-exitcode 99 \
  "$ROOT/.venv/bin/python" "$SD_WORKER" --cuda-sanitizer-worker  # PASSS2,0errors
```

The bounded real SD race commands use the exact symbols recorded in
`f4-groupnorm-selection.json`: `--tool racecheck --racecheck-num-workers 8
--show-backtrace no --error-exitcode 99 --kernel-name "kne=$SYMBOL"`.
Sum160 additionally uses measured `--launch-count 360 --racecheck-trace-sync yes`;
sum256 uses `--racecheck-trace-sync yes` without a cap; scale has no cap.
All pass and `f4-groupnorm-proof.json` independently checks the10240/466944
explicit-event counts against current geometry.

Actual device counters use the prior authorized profiler-only form:
`sudo -n -- env CUDA_VISIBLE_DEVICES=0 LLVM_PROFILE_FILE=/dev/null LD_LIBRARY_PATH="$BUILD:/home/tianlei/cudnn_9.23_cuda13/lib:/home/tianlei/cuda13.0/lib64" /usr/local/cuda/bin/ncu --devices 0 --section SourceCounters --import-source yes --kernel-name-base function --kernel-name 'regex:.*Greedy.*' --export "$Q/f4-greedy-device" ./onnxruntime_autoep_test --gtest_filter='CudaStatefulTest.GreedyDeviceCoverage'`
passes1test; `ncu --import "$Q/f4-greedy-device.ncu-rep" --page source
--print-source cuda,sass --csv` exports the actual25/25source evidence.

Each timing/resource/Step row executes
`"$ROOT/.venv/bin/python" "$ROOT/samples/python/stateful_qualification.py"
--model "$MODEL" --package "$PACKAGE" --phase "$PHASE"
--output "$Q/f5/$ID/$PHASE.json"` with the concrete eight rows/source/manifest
hashes in `fixed-workloads.json`; all24workers pass. CUDA resource workers are
wrapped with `nsys profile --trace=cuda --cuda-memory-usage=true --sample=none
--cpuctxsw=none --output "$Q/f5/$ID/resources-trace"`, followed by
`nsys export --type sqlite --output "$Q/f5/$ID/resources-trace.sqlite"
"$Q/f5/$ID/resources-trace.nsys-rep"`. Exact dynamic allocation/free enums and
per-address lifetimes, rather than total event bytes, determine the peaks.

**F8 publication completed:** [#30](https://github.com/onnxruntime/ort-state/pull/30)
is OPEN/non-draft, head `state/m24-qualification`, exact base
`state/m15-sd15-demo` at `4fcf16f7d25d6ce8c693839fdf5686c7d3c4c1d8`.
Feature commit: `3bcf4d69190fbe347cde1ea9e42d85c8c5642a0d`.
This subsequent documentation-only commit records completed publication.
Final synchronized head/base, preserved template/body, source/artifact proof
and clean-tree verification are retained in `m24-final-published-pr.json` and
the refreshed `f7-source-artifact-proof.json`. No PR was merged.
M16-M23 remain explicitly deferred/unchecked; the overall all-language feature
completion checkbox remains open.

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
| Local Phi-3/SD files and adapters | M14 assets qualified. M15 CUDA-only `tlwu` source/revision, tokenizer and asset directory confirmed; downloads await approvals. | Inspect actual SD interfaces after approved acquisition. CPU fixtures do not substitute for the real CUDA image demo, and CUDA-only exports must not be claimed as real CPU models. |
| CPU/CUDA model variants | Actual signatures/precision not inspected | Verify compatible local variants; no silent model substitution/download. |
| Objective-C/iOS validation | Linux is the initial platform; Apple access not provided | M22/M23 need supplemental runner/coverage or remain blocked; no implicit scope reduction. |
| Browser/mobile package resources | Runtime file APIs differ from native filesystem APIs | M01/M19/M23 define adapters preserving package confinement and execution parity. |

Approval record: on 2026-10-15, the user approved this plan without changes and
explicitly authorized beginning M01. Changes to scope or mandatory gates still
require user review.

## 9. Contributor-facing overview (documentation-only)

**Requested:** 2026-11-04, following the20published implementation/qualification
PRs(#11-#30). This is documentation of delivered work, not a new runtime milestone
or authorization to implement deferred bindings.
**Branch/base:** `state/contributor-overview` on M24/#30 at
`84237bbf6b9d059e4f96904b9c181402d6e11da6`.

- [x] Verify all20PR titles/statuses/stack bases and the delivered source contracts.
- [x] Write `docs/Stateful_Execution_Overview.md` covering constructs/idioms,
  built-in/plugin EP hooks, Model Package changes, capabilities/limits,
  reproducible Phi-3/SD examples and contributor impact.
- [x] Link the overview from the existing Python guide.
- [x] Validate relative links,20PR references, example CLI shapes and documentation whitespace.
- [x] Publish the21st documentation-only PR on M24/#30 and verify remote head/base/clean tree.

**Validation:** Documentation-only: no new production lines or runtime behavior,
so no CPU/CUDA rebuild, model run or unit-coverage claim is required. Existing
M24 measurements/coverage are explicitly attributed to that qualified baseline,
not recollected or relabeled as new evidence.
Executed documentation checks pass:25relative targets exist; the roadmap
contains exactly20unique PRs(#11-#30); all3Bash examples pass `bash -n`;
documented CLI commands/options match the existing argparse definitions;
the public source-build link resolves; `git diff --check` passes.
No runtime tests, downloads, model runs or dependency installation were performed.

**Published:** [#31](https://github.com/onnxruntime/ort-state/pull/31), OPEN/ready,
head `state/contributor-overview`, exact base `state/m24-qualification` at
`84237bbf6b9d059e4f96904b9c181402d6e11da6`. Overview commit:
`8965109858c9087d428e86919ec3716c91413b43`.
This tracking-only follow-up records completed publication. Final synchronized
head/base, preserved PR template/body and clean-worktree verification are retained
in session `contributor-overview-final-pr.json`. No PR was merged and no deferred
binding milestone was changed.
