# Validation

**Owns:** Evidence establishing WebGPU behavior, shader-key consistency, target
coverage, and regression-test routing. Performance methods live in
[profiling and performance](profiling-and-performance.md). General build/test mechanics stay in the
repository build/test skills.
**Update when:** Validation tools, test routing, CI runtime coverage, or the
evidence required for a changed variant changes. Recheck live workflows rather
than treating the hardware examples below as permanent coverage guarantees.

## Select tests that execute the change

Every behavior change needs an `OpTester` or WebGPU-specific test that runs on
WebGPU. Cover each new shader variant: f16/f32, vectorized/scalar, supported
ranks, broadcast/non-broadcast, empty tensors, and large tensors for indexing
changes. Shared `OpTester` tests use WebGPU when built unless excluded; verify
the intended path actually executes, with nonzero test counts and skips
reported separately.

`onnxruntime/test/providers/webgpu/` includes `conv_large_index_test.cc`,
`matmul_large_test.cc`, and `activation_cache_key_test.cc`.

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

Review evidence: [#33242 skip placement](https://github.com/microsoft/onnxruntime/pull/33242#discussion_r4236769458),
[#33057 fallback](https://github.com/microsoft/onnxruntime/pull/33057#discussion_r4161278467),
[#32796 coverage](https://github.com/microsoft/onnxruntime/pull/32796#discussion_r4101460222),
and [#33269 test migration](https://github.com/microsoft/onnxruntime/pull/33269).

## Coverage and claims

- Inspect the current jobs. The imported guidance's CI baseline is Linux
  build-only, Windows tests on NVIDIA, macOS on Apple GPUs. This does not
  establish Intel, AMD, or Qualcomm execution: vendor-specific changes need
  results from that hardware in the PR description, or an explicit gap.
- Follow `ort-build` and `ort-test`, plus the active checkout's local build
  instructions. Do not substitute CPU fallback or another branch's binary for
  WebGPU runtime evidence. Build-only is valid evidence only for compilation.

In a hosted PR review, inspect the changed tests and available CI evidence.
Run checks when the review environment supports them; if the required GPU,
backend, or dependency is unavailable, state the gap and the check needed.
Distinguish a missing regression test from an existing test that could not be
executed in the review environment. Do not claim a local or hardware run from
source inspection alone.

Portable Linux example (adapt to local build requirements):

```bash
./build.sh --config Release --parallel --use_webgpu
./build/Linux/Release/onnxruntime_provider_test --gtest_filter="*<OpName>*"
```

Without hardware on Linux, load `webgpu-local-testing` for lavapipe's scope and
limitations; it does not establish hardware/vendor coverage.

## Shader generation and cache keys

Run the shader-key validation used by nightly CI when shader generation or
identity changes. Find the current job by searching for the validator or
`ORT_WEBGPU_EP_SHADER_DUMP_FILE`; workflow filenames can move. Run from the
repository root, using the active build's provider test binary:

```bash
export ORT_WEBGPU_EP_SHADER_DUMP_FILE=/tmp/webgpu_shaders.log
./build/Linux/Release/onnxruntime_provider_test --gtest_filter="*<OpName>*"
node .github/actions/webgpu-validate-shader-key/validate-shader-key.js < /tmp/webgpu_shaders.log
```

In PowerShell set `$env:ORT_WEBGPU_EP_SHADER_DUMP_FILE` and pipe
`Get-Content <dump>` into `node .github/actions/webgpu-validate-shader-key/validate-shader-key.js`.
Use a fresh task-specific dump and check both commands' exit codes. A passing
validator on an empty dump or an unexercised specialization proves nothing.
The dump includes WGSL for each program and key.

For WebGPU validation errors, set EP option `validationMode` (config key
`ep.webgpuexecutionprovider.validationMode`) to `full`. Release defaults to
`disabled`. For template-generator changes, run:

```bash
python tools/python/wgsl_template/test/run_tests.py
```
