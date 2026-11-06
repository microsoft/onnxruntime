# Validation

**Owns:** WebGPU execution evidence, shader-key consistency, target coverage, and
regression routing. [Performance methods](profiling-and-performance.md) and
repository build/test mechanics have separate owners. Review inspects source
and existing CI results; **do not build, configure, install dependencies, trigger
CI, or run the validation commands below**. CI bots execute build/test validation.
**Update when:** Tools, test routing, CI coverage, or variant-evidence needs change.
Recheck live workflows rather than relying on past coverage.

## Select tests that execute the change

Every behavior change needs an `OpTester`/WebGPU test executing it on WebGPU.
Cover each new variant across f16/f32, vector/scalar, every supported rank,
broadcast/non-broadcast, empty tensors, and large tensors for indexing changes.
Shared `OpTester` tests include WebGPU unless excluded: verify the path, nonzero
counts, and separate skips. Explain exclusions of `kWebGpuExecutionProvider` and
tolerance changes; f16 may justify tolerance, while relaxed f32 can hide defects
and needs error analysis.

- Acquire provider/capabilities **before** constructing `OpTester`. `GTEST_SKIP()`
  returns only from its function; helper skips after construction can leave an
  unrun tester that traps in Debug. Disable CPU fallback for WebGPU regressions.
- Provider availability does not prove f16/subgroup-matrix support. Guard native
  probes/includes in plugin adapter builds and use supported capability APIs.
  Choose nonzero, distinguishable data exposing row/head/component/stride/order errors.
- Cache tests need both orders in one context, controlled cold/warm state and
  earlier-session lifetimes. Tiled tests must check output from every new wave,
  BK choice, and tail, including active lanes.
- Prefer test-only storage limits to huge allocations for segmentation coverage.
  Count every segmented binding; gate optional-output cases on device limits
  before constructing `OpTester`, retaining cases supported by minimum limits.
- Test moves/extraction must preserve provider setup, fallbacks/options,
  tolerances, guards, fixtures, and source registration. Check consumers before
  deleting generators/binaries; review other providers for shared tests.
  Relocation alone adds no coverage.

## Coverage and claims

- Assess source, tests, and existing CI evidence; record unverified requirements
  for CI/author follow-up. Intentionally manual tests do not require new
  CI wiring unless the repository contract or changed behavior requires it.
- Inspect current jobs' hardware/backend/workload; separate compilation,
  software adapters, and hardware runs. Vendor changes need that hardware's
  results in the PR description or an explicit gap.
- Check CI configuration against `ort-build`/`ort-test` and checkout instructions.
  CPU fallback or another branch's binary is not WebGPU evidence; build-only
  proves compilation.
- Local and hosted reviews inspect tests/CI without launching validation. Name unavailable
  GPU/backend/dependencies and the needed check. Distinguish missing tests from
  tests present but unexecuted; source inspection proves no runtime result.
- For Linux software-adapter evidence, consult `webgpu-local-testing` only for
  lavapipe's scope/limits; its build/run workflow is outside review, and its
  results do not establish hardware/vendor coverage.

## Shader generation and cache keys

For codegen/identity changes, assess nightly CI or author-provided evidence.
Find the job via the validator or `ORT_WEBGPU_EP_SHADER_DUMP_FILE`: it should
use a fresh dump from the reviewed build's affected provider tests and pass it
on stdin, from the repository root, to
`node .github/actions/webgpu-validate-shader-key/validate-shader-key.js`.
Verify recorded exit codes and a nonempty dump exercising differing specializations;
the dump contains each program's WGSL/key. Empty/unexercised validation proves nothing.

For WebGPU errors, evidence should use `validationMode`
(`ep.webgpuexecutionprovider.validationMode`) set to `full`; inspect effective
defaults/options in `WebGpuContextConfig` and public parsers. Template-generator
changes need results from `python tools/python/wgsl_template/test/run_tests.py`.
