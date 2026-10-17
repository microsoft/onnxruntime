# Shared Scratchpad

Compressed project memory. This is not the execution plan.

## Workflow checkpoint

- 2026-10-15: Requirements interview completed. User explicitly said there was
  no additional information and authorized drafting `SPEC.md`, then stopping
  for review.
- 2026-10-15: User answered "1" to the specification review: approved `SPEC.md`
  and authorized preparing `TASKS.md` for separate review.
- 2026-10-15: User approved `TASKS.md` without changes and authorized beginning
  M01. `SPEC.md` and the execution plan are approved.
- M01 is complete and published as ready-for-review PR #11:
  https://github.com/onnxruntime/ort-state/pull/11
- No agents launched. M01-M06 are complete; M07-M24 remain pending.
  Runtime production code starts with M02.
- Future tracking: keep `TASKS.md` checklist current and record compressed
  discoveries/agent learnings here.

## Confirmed product decisions

- Production-oriented foundation, leaving serving optimizations for later.
- Preserve ordinary Session behavior; add SessionLoop, SessionChain,
  KVCacheManager, and Sampler with per-execution state.
- Programmatic and package construction through every Session language binding.
- State retained until explicit Reset; sequential execution initially.
- KV sharing within one execution, especially prefill/decode; cross-request
  prefix reuse deferred.
- Contiguous tensor-backed CPU cache baseline accepted. Prefer sharing;
  copies/reallocation permitted. No hard zero-copy or memory-budget requirement.
- Explicit loop exit conditions through both programmatic and package interfaces.
  EndOfInputs for prefill, EOS for decode, limits, boolean model output; OR
  composition. Reject loops with no conditions. No custom predicates.
- Full-prompt prefill initially; chunking deferred; reject empty prompt.
- Synchronous Step/Next; caller accumulates ordinary per-iteration outputs;
  run-until-exit convenience.
- Cancellation/error handling accepted; potentially corrupted state requires Reset.
- Greedy sampling only initially. CPU first; real CUDA KV and greedy may be
  separate milestones. Plugin EP customization required.
- Package declares chain/order, execution kind, bindings, and policies, not a
  state-machine program.
- LLM: local Phi-3-mini-4k-instruct ONNX, package to be built, two SessionLoops,
  text in/text out, caller tokenizer/detokenizer.
- Diffusion: local SD 1.5-compatible text encoder/denoising loop/VAE, DDIM,
  classifier-free guidance, seeded noise, caller tokenizer, text in/image file out.
- Linux validation initially; CPU every milestone and CUDA where applicable.
- Measured 100% new executable-line unit-test coverage; not branch coverage.
- Record performance measurements, no numerical performance gates.
- Exact deterministic fixtures; documented numerical tolerances for real models.
- Tested/documented stacked ready-for-review PRs in onnxruntime/ort-state.

## Repository facts

- Initial branch: `ort-state`; initial worktree clean.
- Origin: `https://github.com/onnxruntime/ort-state.git`.
- Root planning documents did not exist before this draft.
- Read `AGENTS.md` and all four `.github/instructions/*.instructions.md` scopes.
  C API instructions apply to anticipated public-header work. Do not expand
  deprecated ACL/JSEP/WebGL functionality.
- Standalone `model_package/` owns schema, authoring and path resolution, and
  has no ORT dependency. Preserve this boundary.
- ORT package integration owns `executor_info["ort"]`, EP variant selection,
  Session creation and `OrtModelPackageApi`. Reuse those mechanisms.
- Package parser defaults to strict unknown-field handling; schema evolution
  must account for it.
- JavaScript includes shared interfaces plus Node, Web and React Native.
- Go and Rust Session bindings exist; Rust README calls its binding experimental.
  Do not quietly exclude either from the requested parity.

## Review/verification risks

- Exact API signatures and manifest spelling are not frozen by the requirements
  draft; document concrete contracts before implementation.
- Approved edge cases include zero-limit immediate termination and observable EOS.
- Linux cannot validate every platform-specific binding. Keep unavailable parity
  tests/coverage unverified; seek supplemental access rather than waive the gate.
- CUDA device-line coverage needs distinct instrumentation; host coverage is
  insufficient. No measured evidence has been collected.
- Need explicit local model paths, compatible CPU/CUDA variants, external
  weights and tokenizer adapters before real-model validation.

## Execution-plan draft

- 24 intended PR-sized milestones: contracts/fixtures; context/loop; chain;
  CPU cache; CPU greedy; public C/C++; EP/plugin hooks; CUDA cache; CUDA greedy;
  standalone package; ORT loading; DDIM; Python; Phi-3 demo; diffusion demo;
  C#; Java; common JS/Node; Web/WASM; Go; Rust; Objective-C; React Native;
  integrated qualification/measurements.
- M01-M05 complete after explicit approval; M06-M24 remain pending.
- Each milestone includes intended outcome, tasks, CPU/CUDA test plan and gate.
  Actual prerequisites are listed separately from mechanical stack ordering.
- M01 verifies concrete contracts, small fixtures and coverage readiness.
  Unavailable CUDA/Apple validation blocks affected milestones, not unrelated
  CPU development, and never counts as passed.
- Plan recipes are unexecuted templates; resolve exact selectors/paths during
  execution and verify nonzero actual tests plus intended provider dispatch.
- Document-only checks passed with `python3`: ASCII/whitespace/final newlines,
  24 milestone sections/rows with outcomes and CPU/CUDA/gate text, dependency
  ordering, and all 93 specification requirement mappings. No runtime tests ran.
- The unactivated shell has `python3` but no `python` command; activate the
  intended virtual environment before using the plan's Python build recipes.

## Additional repository findings for execution

- Loaded `ort-build` and `ort-test` skills during plan preparation.
- Existing Model Package gtests: `ModelPackageApiTest.*` in
  `onnxruntime_autoep_test`, not the core/provider test executables.
- `tools/ci_build/build.py --code_coverage` currently rejects non-Android builds.
  Linux host instrumentation requires a separately verified configuration.
- `build.py`'s `run_csharp_tests` returns immediately off Windows. Run managed
  Linux tests directly; successful native build is not C# test evidence.
- Go `TestMain` in `go/onnxruntime/ort_test.go` can exit 0 without tests when it
  cannot find the native library. Set `ORT_LIB_PATH` explicitly and verify counts.
- Java tests run via CMake/Gradle; local independent Gradle needs `cmakeBuildDir`.
- Standalone package CTest names: inspection, asset_hashing, authoring, commit.
- React Native has real host C++ test script plus JS/mobile bridges; mocked JS
  tests alone cannot establish mobile native coverage.

## M01 active work

- Branch `state/m01-contracts-fixtures` created; isolated `.venv` uses Python 3.12.9.
- No existing workspace builds were found. CPU baseline build launched with
  Clang 18, LLVM instrumentation (`-fprofile-instr-generate -fcoverage-mapping`),
  RelWithDebInfo, shared library, eight jobs, target `onnxruntime_autoep_test`,
  Unix Makefiles, and `--skip_submodule_sync` (archive dependencies).
- Build shell `m01-cpu-build` was stopped during the interrupted interaction.
  Log `m01-cpu-build.log` in session artifact storage shows configuration/generation
  succeeded, compilation reached approximately 65%, then make was terminated.
  No compiler error found and no target executable yet. Resume incrementally;
  no process remains active under that shell ID.
- CUDA 13.0 at `/usr/local/cuda`, eight visible A100s, Nsight Compute 2025.3,
  Compute Sanitizer 2025.3. Availability is not device-line coverage evidence.
- Reuse `IOBinding` and DataTransferManager for device feeds and output ownership.
- Existing contrib generation code prepares/updates attention masks, position IDs,
  cache feeds, and next tokens. Plain tensor renaming alone cannot perform this.
  Need an explicit supported native iteration-policy contract, not manifest code.
- Existing Python DDIM helper can inform reference testing but depends on
  torch/numpy; native runtime must not acquire those dependencies.
- User explicitly approved all four recommendations after clarification:
  native policies; initially single-sequence token generation (generic tensor
  batching unaffected); stable retained outputs with necessary copies;
  float32/float16 logits, int64 token IDs, lowest-ID ties, NaNs error, infinities
  numeric ordering. Recorded with concrete design in SPEC Appendix A.
- Added reusable in-memory scalar feedback/boolean, renamed-stage, decoder
  logits/growing-cache, and denoiser fixtures plus five CPU Session reference
  tests in `test/autoep/test_stateful_execution_baseline.cc`.
- New fixture files pass `clang-format --dry-run --Werror` after formatting.
  No stateful production code yet; new production-line denominator is zero/N/A.
- CPU build resumed after regeneration under shell `m01-cpu-resume`, same log.
- CPU build completed successfully. First targeted run found missing staged
  `mul_1.onnx`/`mul_16.onnx`; copied those existing fixtures with CMake, reran:
  all 21 tests passed (five new CPU cases, 16 existing Model Package API cases),
  no failures/errors/skips. XML/profiles: `coverage/m01/passed` in CPU build.
- Host LLVM source mapping works: reusable helper header 103/103 executable lines,
  seven functions. Test driver 138/141 CPU-only lines (three CUDA setup lines).
  M01 has zero new production lines, so production-line coverage is N/A.
- Aggregate LLVM maps warn about 60 other-function records; `--dump` shows no
  mismatch for new helper functions. Preserve warning evidence, and fail future
  new-production coverage gates if their mappings are incomplete/mismatched.
- Tiny CUDA debug probe succeeded: CUDA-GDB breakpoint hits on both kernel-body
  statements, normal exit. Possible device-line measurement route established;
  not actual CUDA EP or future production-kernel coverage.
- Provider-parameterized baseline cases assert all recorded nodes use the
  requested EP; CUDA setup disables CPU fallback. CPU instantiation always runs.
- Initial CUDA baseline build failed from mixed 13.0/13.1 CUDA headers. Retried
  under shell `m01-cuda-13-1-build` with a consistent system-registered CUDA 13.1/
  cuDNN root in a separate `build/state-cuda-baseline-13.1` tree, A100-only,
  contrib/Flash/MEA disabled for non-attention fixture validation.
- First CUDA log: `m01-cuda-build.log`; retry: `m01-cuda-13.1-build.log`.
  Retry completed successfully. `m01-cuda-tests.log` and CUDA build XML show
  26/26 tests passed: five CPU, five native-CUDA, 16 existing package API cases.
  Zero failures/errors/skips; all CUDA nodes assigned as expected, fallback off.
- Final C++ format/120-column/whitespace and document/93-requirement mapping
  checks passed. M01 commit `64aa026b94` and ready-for-review PR #11 published.
- Available language tools: .NET SDK 8.0.422, Java 17, Node 24, Go 1.26.4,
  Cargo 1.96.1. Apple/mobile runners and real-model assets still unprovided.
- Remote default is `main`; published the unchanged `ort-state` baseline at
  `1988019b6e` and `state/m01-contracts-fixtures`. PR #11 targets `ort-state`.
- Throwaway CUDA probe source/binary were removed; feasibility log retained.
- Next: M02 (context/SessionLoop); no M02 implementation has started.

## M02

- 2026-10-16: User approved moving forward. Created
  `state/m02-context-loop` from M01; M02 in progress.
- Keep implementation internal; no public ABI or sampler/cache-manager work.
- Use native InferenceSession/IOBinding; generic feedback and finite inputs,
  explicit typed exits, CPU-owned stable result snapshots.
- Existing ORT termination is a non-atomic bool. Do not introduce a cross-thread
  write/read race through it: use an atomic loop cancellation request, checked
  before/after Session Run. A running kernel/Session need not be preempted.
- Added internal `session_loop.h/.cc`, focused `framework/session_loop_test.cc`
  and `onnxruntime_session_loop_test` CMake target. SPEC Appendix B records
  supported inputs, outputs, counters, exit ordering and atomic cancellation.
- CPU tests: 15/15 focused cases passed; 21/21 M01/package regressions passed.
  Initial production coverage 313/313 lines (310 .cc + 3 header), no mismatches
  for new functions. Fourteen unchanged-function LLVM warnings documented.
- After final wrapping to satisfy 120-column style, rebuild/rerun and collect
  final coverage. Done: **315/315 production lines**, 312 .cc plus three header,
  final focused tests 15/15 passed, zero skips. Final profiles/LCOV/XML are under
  `coverage/m02/final` in CPU output. Final detailed diagnostics still list
  14 unchanged-function warnings, no SessionLoop mismatch.
- Initial builds failed because `InferenceSession::IsInitialized` is protected;
  now validate via existing NewIOBinding status API instead of changing Session.
- Current CUDA rebuild: shell `m02-cuda-rebuild`; log `m02-cuda-build.log` in
  session files. Uses verified CUDA 13.1/A100 configuration from M01.
- Runtime inputs use dense non-string tensors; token exits observe explicitly
  configured scalar int64 output (sampler/native generation remains M05).
- No agents launched. M02 implementation committed as `a21ae23201`.
- Source encoding/whitespace/120-column and clang-format checks passed.
- Current CUDA rebuild continues compiling existing provider kernels; no CUDA
  test results were available at that checkpoint.
- CUDA rebuild completed. All 29 focused tests passed (14 native-CUDA cases,
  14 CPU cases plus one CPU-only type/initialization case), zero skips.
- CUDA stateless/package regressions: 26/26 passed, zero skips.
- Compute Sanitizer memcheck: two representative native-CUDA lifetime/reset and
  in-flight cancellation tests passed, zero errors.
- M02 validated and published in ready-for-review PR #12:
  https://github.com/onnxruntime/ort-state/pull/12
  Base branch `state/m01-contracts-fixtures`, head `state/m02-context-loop`.
  No new device production code, so
  measured host 315/315 coverage is the applicable new-code gate.
- Next pending milestone is M03 (SessionChain and bindings); not started.

## M03

- 2026-10-16: User authorized M03. Created `state/m03-chain-bindings` stacked
  on M02. No work on the read-only attached planning snapshot.
- Reuse SessionLoop execution for both loops and plain one-invocation Sessions,
  including snapshots, input validation, errors and cancellation boundaries.
- Explicit source namespaces: external tensor, external finite tensor sequence,
  earlier component output, retained named state. No positional feed forwarding.
- Retain all component model outputs internally for downstream ports/state.
  Chain Step must execute no more than one model invocation; zero-iteration
  component bookkeeping may advance without extra model execution.
- Keep M02's explicit CPU snapshot boundary. Validate host inputs; supported
  CPU/CUDA Sessions perform their declared IOBinding transfers. Device-native
  cache sharing remains M04/M08, not an implicit M03 claim.
- Added `session_chain.h/.cc`, shared `stateful_execution_utils.h` extracted
  from loop validation, and `framework/session_chain_test.cc`. Reuse existing
  `onnxruntime_session_loop_test` binary for chain + loop tests.
- SPEC Appendix C documents source namespaces, finite sequences, zero-iteration
  stages, state snapshots, host staging, lifetime/reset and failure semantics.
- CPU tests passed: 30/30 focused (15 chain + 15 loop/type); 21/21 stateless/
  package regressions. No skips.
- Coverage all affected runtime files is 100%: LLVM aggregate 621/621; LCOV
  unique physical executable locations 615/615 cumulative, 354/354 new M03.
  All 18 profile-map warnings are unrelated to chain/loop/shared-helper functions.
- Initial builds failed strict aggregate warnings for omitted LoopExitCondition
  fields; fixed using full initializer. Current CUDA rebuild `m03-cuda-rebuild`
  continues using established CUDA 13.1/A100 configuration.
- Final test-source formatting fixes (trailing commas to prevent >120-column
  initializer collapse) require one final focused rebuild/test. Production
  source already passed CPU tests and coverage; no production changes after that.
- M03 implementation commit `c4ad4d5389` pushed; no agents used.
- Final CPU suite 30/30 and final coverage rechecked: 354/354 new unique
  executable source locations, 615/615 cumulative; LLVM aggregate 621/621.
- CUDA build/tests passed: 58/58 focused, 26/26 regressions, no skips.
  Fourteen native-CUDA chain cases + 14 native-CUDA loop cases actually executed
  with graph node assignment checks.
- Compute Sanitizer memcheck: four chain state/reset/cancellation/host-staging/
  dynamic-cache tests passed, zero errors.
- Final clang-format and 120-column checks passed. M03 complete in ready-for-review
  PR #13: https://github.com/onnxruntime/ort-state/pull/13
  Base `state/m02-context-loop`, head `state/m03-chain-bindings`.
  No new device code or public API. M04 (CPU KVCacheManager) is next, not started.

## M04

- 2026-10-16: User authorized M04; branch `state/m04-cpu-kv-cache` stacks on M03.
- Internal CPU manager: explicit layouts/dtypes/sequence axis, logical length
  versus allocation capacity, validated full-output adoption/copy, append and
  reserve. Arbitrary sequence axis must use compact logical packing, not
  incorrectly reinterpret per-head capacity-strided buffers as dense tensors.
- Preserve immutable snapshots by allocating fresh buffers on mutation; retain
  compatible allocator-owned model outputs without extra copies. No model
  input/output aliasing is inferred or enabled.
- Managed loop cache bindings feed the manager and commit model outputs; chains
  share one manager across their components, while independent executions do not.
- Keep native CUDA management deferred. CPU/device rejection and host-staged
  CUDA Sessions will be tested without claiming CUDA-backed cache support.
- Implemented `kv_cache_manager.h/.cc`; wired explicit managed input/output
  bindings into loops and chains. Added standalone/integration tests to existing
  `session_chain_test.cc`, keeping milestone to ten files.
- CPU first validation: 42/42 focused tests + 21/21 regressions passed, no skips.
  LLVM reports 100% affected inference-runtime coverage; final denominator
  verification after the last test/lifetime-only edits is pending.
- Compile fixes: Tensor::IsContiguous exists only under ENABLE_STRIDED_TENSORS
  (training); guard it. Convert SafeInt row offsets to size_t before pointer
  arithmetic. ORT InlinedVector does not support N=0; explicitly use N=1 for
  enlarged cache-binding/component structures. Typed array fixes enum/int test.
- CPU inference configuration does not compile the training-only strided guard.
  Do not claim training coverage. No new device code.
- Manager share adopts allocator-owned outputs at their actual capacity;
  copied/appended views retain compact logical packing and use fresh buffers.
  Lifetime counters count manager data-buffer operations, not Session snapshots.
- Additional final tests release producing Sessions and exercise last-axis
  packing. Need final CPU rebuild/profiles and CUDA rebuild/test pass.
- M04 implementation committed as `fffa292875`; no agents used.
- Final validation passed: CPU 42/42 focused + 21/21 regressions; CUDA build
  76/76 focused + 26/26 regressions, no skips. Six new native-CUDA Session
  integration cases still use the explicitly CPU-resident cache manager.
- Final line gate: 335/335 new unique executable locations; 943/943 cumulative
  unique locations. LLVM aggregate mapping 957/957. Eighteen unrelated
  profile-map warnings, none affecting new or coupled runtime functions.
- Compute Sanitizer: five CPU-cache/CUDA-Session integration cases passed,
  zero errors. Producer Session release and all three axis positions tested.
- All ten files pass formatting/whitespace/120-column/document checks.
  M04 complete in ready-for-review PR #14:
  https://github.com/onnxruntime/ort-state/pull/14
  Base `state/m03-chain-bindings`, head `state/m04-cpu-kv-cache`.
- M05 (CPU greedy sampling and decode integration) is next, not started.

## M05

- 2026-10-16: User authorized M05; branch `state/m05-cpu-greedy` from M04.
- User explicitly approved four concrete contracts:
  identical explicit MaxGeneratedTokens budgets on prefill/decode if present,
  including zero-budget no-generation-invocation semantics; unpadded [1,S]
  int64 IDs with optional int64 all-ones mask/sequential position ports;
  CPU float32/float16 greedy, reject any NaN in the entire supplied tensor and
  reject device logits; decode follows prefill in the same context initially.
- Reuse M04 cache handoff and M02 stepping. Shared generation state tracks
  sampled IDs and processed input length; a prefill sample is counted/emitted
  once and consumed by the first decode call. Decode checks EOS/budget on entry.
- Generic loops and model-scalar token exits retain their current behavior.
- Prior art reviewed: contrib generation feeds/TopK helpers are coupled to
  subgraph/kernel contexts; standalone sampler should use standard deterministic
  argmax logic rather than pulling contrib generation dependencies into Session.
- Added `token_generation.h/.cc` for CPU Sampler and native feed/policy helpers;
  wired typed options, shared token history, first-token handoff and EOS/budgets
  into loop/chain. Typed GeneratedTokenIds final ports yield owned [count] tensors,
  including [0], while ordinary unproduced model outputs still error.
- Explicit shared budgets were added to the illustrative manifest's prefill
  phase as well as decode. SPEC Appendix E records all approved semantics.
- Generated host feeds stay owned until synchronization/Session completion;
  no temporary CPU feed may be freed before a potential async device copy.
- CPU 55/55 focused tests and 21/21 regressions passed, no skips. Scope issue
  in sampled-token declaration and empty-batch validation placement were fixed.
- Coverage: new 362/362 unique executable locations, cumulative 1288/1288;
  LLVM aggregate 1303/1303. All 18 mapping warnings unrelated to affected code.
- CUDA rebuild running as `m05-cuda-rebuild` (log `m05-cuda-build.log`).
  Final CUDA run, memcheck, style/document checks and PR remain pending.
- M05 remains internal: CPU sampler with explicit CUDA Session host staging;
  native GPU sampling M09, public API M06. No agents used.
- Final verified CPU run: 55/55 focused + 21/21 regressions; CUDA build:
  97/97 focused + 26/26 regressions. No skips. Five generation CUDA memcheck
  cases passed, zero errors.
- Final gate after reset/default-phase refinements: 369/369 new unique executable
  source locations; 1295/1295 cumulative unique; LLVM aggregate 1310/1310.
  Eighteen unrelated map warnings, none affecting the stateful runtime.
- Explicit TokenGenerationState::Reset avoids GCC false-positive warnings from
  aggregate assignment of an Abseil inline vector. Default phase is Unspecified,
  rejected explicitly. No warning suppression or success-shaped fallbacks.
- Final XML/profiles/LCOV: CPU `coverage/m05/verified`, CUDA
  `m05-verified-tests.xml` and `m05-verified-regressions.xml` in build output.
- M05 implementation commit `ddb8260691` published in ready-for-review PR #15:
  https://github.com/onnxruntime/ort-state/pull/15
  Base `state/m04-cpu-kv-cache`, head `state/m05-cpu-greedy`.
- All ten files pass style/whitespace/120-column checks. No new device production
  code or public API. M06 (public C API/C++ wrappers) is next, not started.

### M05 layout follow-up

- 2026-10-16: User requested token generation outside `core/session`, suggesting
  `core/generate`. Moved `token_generation.h/.cc` there.
- Moved shared `stateful_execution_utils.h` into `core/framework` to avoid a
  generation-to-session include dependency; no helper duplication or logic change.
- Session/loop/chain interfaces stay in `core/session`. Source discovery includes
  the new generation directory in the existing session library target.
- Updated includes and documented source/coverage paths; no M06 work.
- CPU rebuilt and passed 55/55 focused + 21/21 regressions, zero skips.
  `coverage/m05/layout` confirms generation/framework paths in the actual
  instrumented artifacts and no old source mappings. Coverage stays 100%:
  1295/1295 unique runtime locations, 1310/1310 LLVM aggregate.
- CUDA validation passed: 97/97 focused + 26/26 regressions, no skips. Five
  generation memcheck scenarios passed with zero errors.
- M05's new-line gate remains 369/369 (same functions, new source paths).
  Verified that PR #15 has no concurrent body/branch edits before refreshing it.
- Layout refactor complete in commit `417832d919`, pushed to the existing M05
  branch. PR #15 retains its template and now documents the new layout and
  post-refactor validation. No behavior, namespace or library-target changes.

## M06

- 2026-10-16: User authorized M06. Created `state/m06-public-api` from
  updated M05 head `17e432b025`.
- Read public C API instructions and companion C/C++ patterns. Current
  ORT_API_VERSION=32: append accessor/table entries, no version bump; new
  public members need Doxygen `\since Version 1.32.`.
- Proposed contracts in SPEC Appendix F await user approval: size-tagged typed
  configuration copied into opaque options, typed inputs at execution creation,
  caller-owned Sessions, retained const result/cache views and C++ parent-owner
  lifetimes, standalone cache/sampler operations.
- Proposed M06a C ABI + M06b C++ RAII sub-PRs in TASKS for reviewability.
  Macro M06 remains incomplete until both pass CPU/CUDA and 100% new-line gates.
- No runtime/public-header edits, builds, tests, commits or PRs for M06 yet.
- User approved Appendix F and M06a/M06b split without changes; authorized M06a.
  Renamed the work branch `state/m06a-stateful-c-api`; M06b remains pending.
- M06a now implements the append-only accessor and 38-entry C table, owning
  options/definitions/executions/results/cache/sampler handles and typed records.
  All nine initial public C-API CPU tests passed. Final CPU/CUDA rebuilds,
  pure-C runtime smoke, measured line coverage and publication remain pending.
- Size checks freeze the original last-field endpoints; metadata copies preserve
  oversized caller-buffer tails. Future appended fields cannot raise V1 minimums.
- Standalone loop state snapshots expose feedback and managed caches; retained
  views survive reset. Failed execution snapshots are rejected until reset.
- Final M06a CPU: 31/31 API/baseline + 55/55 native tests; CUDA: 41/41 API/baseline
  + 97/97 native tests. Zero skips. CUDA API memcheck 5/5, zero errors; actual
  CUDA node assignments asserted. Caches/sampling remain CPU, not M08/M09.
- Pure-C consumer is now checked in and compiled as a C object in the autoep
  target; standalone GCC C11 `-Wall -Wextra -Werror` link/runtime smoke passed.
- Final fresh combined coverage: 562/562 new unique executable lines (546 API,
  16 native GetState); cumulative production/inline headers 1857/1857. The 131
  zero-hash profile-map warnings do not name any new/changed function.
- Full CPU/CUDA reconfiguration was necessary for the new test files; source
  globs do not use CONFIGURE_DEPENDS. Final logs use `m06a-*-complete-*` names.
  C++ RAII is deliberately still pending M06b; macro M06 is not complete.
- Published ready-for-review M06a PR #16:
  https://github.com/onnxruntime/ort-state/pull/16, stacked on
  `state/m05-cpu-greedy`. Implementation commit `f0aa2513d7`.
  M06a is complete; pause for review/authorization before M06b.

## M06b

- User authorized finishing M06b on 2026-10-17. Branch
  `state/m06b-stateful-cxx-api` starts at published M06a `b5f1431837`.
- Use existing SDK Base for move-only resources. Immutable results/cache views
  retain shared C-handle holders; distinct ConstValueImpl storage retains the
  parent without implicit slicing to unowned ConstValue.
- Allocate result control blocks before native Step/Run; allocate no owner
  after advancing. Copy strings/reasons into C++ metadata. Empty resources must
  release safely even when the stateful table is unavailable.
- No agents used. CPU/CUDA tests, measured coverage and publication pending.
- Initial focused tests passed: CPU 5/5; CUDA build 8/8, no skips. Initial
  wrapper coverage 187/187 added executable locations. Executable-only SDK
  coverage has 12 zero-hash unused-map warnings, none for new stateful wrappers.
- clang-format deliberately uses ColumnLimit 0: manually keep new lines <=120.
  Final combined tests/profiles, standalone SDK example and memcheck pending.
- Final fresh M06b CPU: 36/36 C/C++ API/baseline + 55/55 native; CUDA:
  49/49 API/baseline + 97/97 native, zero skips. API memcheck 8/8, zero errors.
- New C++ wrappers 187/187 unique executable locations; cumulative stateful
  native/C API 1857/1857, combined 2044/2044. Export SDK from API executable only;
  runtime export adds .so/native objects. Inspected 42/215 zero-hash unused maps,
  no measured new wrapper/changed runtime function affected.
- SPEC F.6 contains a standalone public-SDK example. Exact documented source
  compiled/linked/ran with GCC C++17, both normal and no-exceptions client flags.
  Weak-owner test proves lifetime retention; static asserts forbid unowned
  slicing/mutable pointers. All changed-range style/columns/doc checks passed.
- M06 implementation/testing gates are now satisfied; publication remains.
- Published M06b ready-for-review PR #17:
  https://github.com/onnxruntime/ort-state/pull/17, stacked on M06a PR #16.
  Implementation commit `cd63f2f8ba`; all source changes tested before committing.
  M06 is complete; M07-M24 remain pending. No PRs were merged.
