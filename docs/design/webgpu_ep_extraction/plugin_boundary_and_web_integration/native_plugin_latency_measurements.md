# Native plugin EP latency measurements

Tracks the native (shared-library) half of this workstream's latency criterion:

> Inference latency stays within an accepted tolerance of the same baselines, for both static plugin registration and
> the native shared-library plugin.

The static-plugin (wasm) half is already measured, in
[the work log](static_plugin_ep_work_log.md#measured-cost-of-the-plugin-path-on-the-web-build). This document covers
the native `--use_webgpu shared_lib` plugin against a built-in `--use_webgpu` baseline.

Commit under test: `873f221786`.

## Measurement environment

| adapter | PnP status | role |
| --- | --- | --- |
| NVIDIA GeForce RTX 5060 Ti | `OK`, driver `32.0.16.1656` (616.56), 16 GB | **the measurement device** |
| NVIDIA Quadro P620 | `Error` | Pascal, stranded on the old driver — see below |
| Microsoft Remote Display Adapter | `OK` | remote-session indirect display, not a compute device |

`nvidia-smi -L` reports the RTX 5060 Ti as the only NVIDIA GPU. This is a convenient property for an A/B: with a
single real GPU adapter present there is no ambiguity about which device either arm ran on.

The P620 is unusable because NVIDIA ended Maxwell/Pascal/Volta support after the R570 driver branch, so the installed
616.56 package contains no kernel driver that will bind to it. Only one NVIDIA kernel driver loads at a time, so
installing a driver new enough for a Blackwell part necessarily drops the Pascal part. This is expected, not a
misconfiguration.

### Earlier episode: measurements taken on the wrong GPU

Recorded because it nearly corrupted this document's results, and because the same trap applies to anyone reusing
this machine.

For part of 2026-09-04 the situation was **inverted**: the RTX 5060 Ti was in a driver-failure state
(`CM_PROB_FAILED_ADD`, Code 31) while the P620 was healthy, so every WebGPU run silently landed on the P620. Pascal
does not advertise `shader-f16`, so f16 models failed at the first node with

```
Program GatherBlockQuantized requires f16 but the device does not support it.
```

from `ORT_RETURN_IF_NOT(webgpu_context_.DeviceHasFeature(wgpu::FeatureName::ShaderF16), ...)` in
`onnxruntime/core/providers/webgpu/shader_helper.cc:421`. `WebGpuContext` only requests `ShaderF16` when the adapter
advertises it (`webgpu_context.cc:793-816`), so this was an adapter capability limit rather than a configuration
mistake. Reinstalling the driver resolved it, and both Qwen scenarios then ran clean.

Two lessons worth keeping:

- **A failure seen first on the plugin path is not necessarily caused by the plugin path.** The f16 failure was
  reproduced identically on the built-in arm (`build/int`, `-e webgpu`), which is what identified it as
  environmental. Both arms reach the same `GetAvailableRequiredFeatures()` call (`webgpu_context.cc:157`), so they
  agree by construction. Always run the built-in control before attributing anything to the plugin boundary.
- **Timing runs do not announce which GPU they used.** Nothing in ORT's output names the adapter, so a machine with
  more than one GPU can quietly produce numbers from the wrong one. Confirm the device externally — `nvidia-smi`
  during a long run shows the process against a specific GPU — and record it alongside the numbers.

### There is no adapter-selection knob

`kDeviceId` (`ep.webgpuexecutionprovider.deviceId`) looks like an adapter index but is not one. It is parsed into
`WebGpuContextConfig::context_id` (`webgpu_provider_factory.cc:155-162`) and used only for context caching and
cross-device copy checks. ORT exposes no way to pick a physical adapter, so on a multi-GPU box the adapter is
whatever Dawn selects.

**This matters for the A/B itself, not just for f16.** If the two arms could land on different adapters the
comparison is meaningless, and neither ORT nor perf_test will tell you — `--list_ep_devices` reports vendor
`Microsoft` with no GPU identity. On this machine the point is currently moot, since only one real GPU adapter is
present, but record the adapter alongside any numbers regardless.

## Measurement setup

### The matched build pair

Two Ninja build directories from the same commit, differing only in the EP flag:

```
python tools\ci_build\build.py --config Release --build_dir D:\source\onnxruntime_4\build\nat_builtin ^
  --cmake_generator Ninja --build_shared_lib --skip_submodule_sync --parallel --update --build --skip_tests ^
  --target onnxruntime_perf_test --use_webgpu
```

and the same command with `--build_dir ...\build\nat_plugin --use_webgpu shared_lib`.

Constraints that are easy to get wrong:

- **`--build_shared_lib` is required on both arms.** The plugin DLL resolves `OrtGetApiBase` from the host, so a
  statically linked host executable cannot load it.
- **Do not disable unit tests.** `onnxruntime_perf_test` is defined in `onnxruntime_unittests.cmake`; `--skip_tests`
  only skips *running* them, which is what is wanted.
- **The plugin arm needs a second build step**, `ninja onnxruntime_providers_webgpu`, because perf_test `dlopen`s the
  plugin and `--target onnxruntime_perf_test` alone does not produce the DLL.
- `build/int` is **not** usable as the baseline: it differs in four ways at once (RelWithDebInfo vs Release, Visual
  Studio vs Ninja generator, `BUILD_SHARED_LIB=OFF` vs `ON`, and `ENABLE_DAWN_BACKEND_VULKAN=ON` vs `OFF`). It is
  fine as a qualitative control, as used for the f16 result above, but not for timing.

**Fairness control:** diff the two `CMakeCache.txt` files and confirm that every delta is an EP-path option. This is
the check that makes the comparison trustworthy; do not skip it on the assumption that the build commands matched.

### Invocations

Built-in arm:

```
onnxruntime_perf_test.exe -e webgpu -m times -r <N> <model_dir>\model.onnx
```

Plugin arm:

```
onnxruntime_perf_test.exe --plugin_ep_libs "WebGPU|onnxruntime_providers_webgpu.dll" ^
  --plugin_eps WebGpuExecutionProvider -m times -r <N> <model_dir>\model.onnx
```

- The EP name is **`WebGpuExecutionProvider`**, not `WebGPU`. The registration name before the `|` is arbitrary.
- perf_test **rejects a model directory** — pass the `.onnx` file. It picks up `test_data_set_0` from that directory.
- `-i` is ignored for WebGPU: `-e webgpu` hardcodes `AppendExecutionProvider("WebGPU", {})`
  (`ort_test_session.cc:608-613`). Use **`-C "ep.webgpuexecutionprovider.<key>|<value>"`**, which routes through
  `ConfigOptions` and therefore applies **identically on both arms** — exactly what an A/B needs. The available keys
  are in `webgpu_provider_options.h`.

Methodology: interleaved rounds, taking the median of each round's minimum — the same procedure used for the wasm
numbers, so the two sets are directly relatable.

Note that perf_test **cannot drive a *statically* linked plugin EP**: both `common_utils.cc:93-99` and
`ort_test_session.cc:111-112` gate on `registered_plugin_eps`, which is populated only by `--plugin_ep_libs`. That is
moot for this dynamic-library A/B, but it is a real gap if static-plugin native perf is ever needed.

### Models

`bench_compute` (16 nodes) and `bench_dispatch` (300 nodes) are fp32. `bench_dispatch` is the
dispatch-overhead-sensitive case, and is where the wasm A/B found the per-kernel cost.

A Qwen3.5-0.8B int4 case is also included, to check whether the per-kernel overhead is visible on a real LLM. Both
its scenarios have been confirmed to run end to end on the RTX 5060 Ti.

#### Why the Qwen inputs had to be generated by hand

perf_test's `-I` flag cannot produce valid inputs for this model. `InitializeTensorWithSeed`
(`ort_test_session.cc:1020-1040`) randomizes only `float` and `int8_t`/`uint8_t`; every other type falls through to
`random_init = false` and prints "this type of data won't be random initialized". `input_ids` and `position_ids` are
`int64` and would be left **uninitialized**, indexing far outside the 248320-entry embedding table. Unspecified free
dimensions also default to 1 (`:1135-1137`), which would give a nonsense shape regardless.

So a real `test_data_set_0` was generated instead. Tensor **names** are set on every `TensorProto`, which makes the
lexicographic `input_10.pb` < `input_2.pb` file ordering harmless — `TestCase.cc:621-624` prefers the name over the
positional index whenever the name is non-empty.

The model is hybrid-attention (`qwen3_5_text`: 24 layers, vocab 248320, hidden 1024, 8 heads, 2 KV heads,
head size 256), so its 51 inputs are not uniform:

| input | index | type | shape |
| --- | --- | --- | --- |
| `input_ids` | 0 | int64 | `[B, S]` |
| `attention_mask` | 1 | int64 | `[B, T]` |
| `past_key_values.{L}.key`, L in {3,7,11,15,19,23} | 2-7 | float16 | `[B, 2, P, 256]` |
| `past_key_values.{L}.value`, same L | 8-13 | float16 | `[B, 2, P, 256]` |
| `position_ids` | 14 | int64 | `[B, S]` |
| `past_key_values.{L}.conv_state`, the other 18 layers | 15-50 | float16 | `[B, 6144, 3]` |
| `past_key_values.{L}.recurrent_state`, same 18 layers | 15-50 | float16 | `[B, 16, 128, 128]` |

Only 6 of the 24 layers carry a KV cache; the remaining 18 are linear-attention layers with convolution and
recurrent state. Two scenarios were materialized under `D:\test\qwen_perf`: `decode` (B=1, S=1, P=128, T=129, the
dispatch-sensitive case) and `prefill` (B=1, S=128, P=0, T=128, a compute-bound control).

`genai_config.json` sets `past_present_share_buffer: true` and WebGPU options `enableGraphCapture: "0"` /
`validationMode: "basic"`. perf_test does not apply these, so pass them with `-C` if the comparison should match how
GenAI runs the model.

## Results

Pending — the matched build pair is still building. Record with the results: the adapter used, the commit SHA, both
build command lines, the `CMakeCache.txt` delta, and the raw per-round minimums rather than only the medians.
