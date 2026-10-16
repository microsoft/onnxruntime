# Shared Scratchpad

Compressed project memory. This is not the execution plan.

## Workflow checkpoint

- 2026-10-15: Requirements interview completed. User explicitly said there was
  no additional information and authorized drafting `SPEC.md`, then stopping
  for review.
- 2026-10-15: User answered "1" to the specification review: approved `SPEC.md`
  and authorized preparing `TASKS.md` for separate review.
- 2026-10-15: User approved `TASKS.md` without changes and authorized beginning
  M01. `SPEC.md` and the execution plan are approved; M01 is in progress.
- No agents launched; no runtime production changes, commits, pushes, or PRs yet.
  CPU and CUDA baseline builds/tests passed; M01 PR publication is next.
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
- M01 in progress after explicit plan approval; M02-M24 remain pending.
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
  checks passed. M01 validated; commit/PR tracking still pending.
- Available language tools: .NET SDK 8.0.422, Java 17, Node 24, Go 1.26.4,
  Cargo 1.96.1. Apple/mobile runners and real-model assets still unprovided.
- GitHub access verified. Remote default is `main`; no remote `ort-state` yet.
  Local `main` and `ort-state` both start at `1988019b6e`. Publish that unchanged
  baseline branch for the agreed first-PR target after M01 gates pass.
