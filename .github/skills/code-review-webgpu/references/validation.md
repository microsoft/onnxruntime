# Validation

**Owns:** Evidence establishing WebGPU behavior, shader-key consistency, target
coverage, and regression-test routing. Performance methods live in
[profiling and performance](profiling-and-performance.md). General build/test mechanics stay in the
repository build/test skills.
**Update when:** Validation tools, test routing, CI runtime coverage, or the
evidence required for a changed variant changes. Recheck live workflows rather
than treating a past CI configuration as a permanent coverage guarantee.

## Select tests that execute the change

Every behavior change needs an `OpTester` or WebGPU-specific test that runs on
WebGPU. Cover each new shader variant: f16/f32, vectorized/scalar, supported
ranks, broadcast/non-broadcast, empty tensors, and large tensors for indexing
changes. Shared `OpTester` tests use WebGPU when built unless excluded; verify
the intended path actually executes, with nonzero test counts and skips
reported separately.

An exclusion of `kWebGpuExecutionProvider` or a looser tolerance needs a stated
reason. f16 precision can justify tolerance; relaxed f32 tolerance usually
hides a defect and needs evidence.

- Acquire the provider and check required capabilities before constructing
  `OpTester`. `GTEST_SKIP()` returns only from its containing function; a helper
  skipping after construction can leave a tester that never ran and traps in
  Debug. Disable CPU fallback when a regression claims WebGPU execution.
- Provider availability does not prove optional f16 or subgroup-matrix support.
  Guard native-context probes/includes for plugin adapter builds; use supported
  capability APIs. Choose nonzero, distinguishable inputs that expose row/head,
  component, stride, and candidate-order errors.
- For cache variants, execute both orders in one context and control cold/warm
  state and earlier session lifetimes. For tiled paths, verify checked output
  from every new wave, BK choice, and tail, not only padded/inactive lanes.
- For test moves/extraction, preserve provider setup, fallbacks, options, dtype
  tolerances, conditional compilation, fixtures, and build-source registration.
  Remove unused generators/binaries only after checking consumers. Shared tests
  also need other-provider review; new test location alone adds no coverage.

## Coverage and claims

- Inspect the current jobs and their actual hardware, backend, and workload.
  Distinguish build-only, software-adapter, and hardware execution coverage.
  Vendor-specific changes need results from that hardware in the PR description,
  or an explicit gap.
- Follow `ort-build` and `ort-test`, plus the active checkout's local build
  instructions. Do not substitute CPU fallback or another branch's binary for
  WebGPU runtime evidence. Build-only is valid evidence only for compilation.

In a hosted PR review, inspect the changed tests and available CI evidence.
Run checks when the review environment supports them; if the required GPU,
backend, or dependency is unavailable, state the gap and the check needed.
Distinguish a missing regression test from an existing test that could not be
executed in the review environment. Do not claim a local or hardware run from
source inspection alone.

Without hardware on Linux, load `webgpu-local-testing` for lavapipe's scope and
limitations; it does not establish hardware/vendor coverage.

## Shader generation and cache keys

Run the shader-key validation used by nightly CI when shader generation or
identity changes. Find the current job by searching for the validator or
`ORT_WEBGPU_EP_SHADER_DUMP_FILE`; workflow filenames can move. Set that environment
variable to a fresh task-specific dump path and run the active build's provider
tests for the changed operators. From the repository root, pass the dump on stdin
to `node .github/actions/webgpu-validate-shader-key/validate-shader-key.js`.
Check both commands' exit codes. A passing
validator on an empty dump or an unexercised specialization proves nothing.
The dump includes WGSL for each program and key.

For WebGPU validation errors, set EP option `validationMode` (config key
`ep.webgpuexecutionprovider.validationMode`) to `full`. Check the effective
defaults/options in `WebGpuContextConfig` and the public parsers in the reviewed
revision. For template-generator changes, run
`python tools/python/wgsl_template/test/run_tests.py`.
