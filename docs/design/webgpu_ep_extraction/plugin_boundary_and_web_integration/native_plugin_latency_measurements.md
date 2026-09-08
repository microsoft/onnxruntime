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

### Fairness control: verify the two builds differ only in the EP

Identical `build.py` command lines do **not** guarantee identical builds, so verify the generated build rather than
trusting the invocation.

1. Diff the two `CMakeCache.txt` files and confirm every delta is an EP-path option. Normalize the build-directory
   name first, or the path embedded in most entries will swamp the diff.
2. Confirm the optimization flags actually reached the compiler, by counting them in the generated `build.ninja`:

```powershell
$bn = 'D:\source\onnxruntime_4\build\<dir>\Release\build.ninja'
foreach ($f in '/O2','/Ob2','/DNDEBUG') {
  "$f = " + (Select-String -Path $bn -Pattern $f -SimpleMatch -AllMatches | Measure-Object).Count
}
```

Both arms should report several thousand of each. **Zero means an unoptimized build**: MSVC defaults to `/Od` when
no `/O` flag is given, and a missing `/DNDEBUG` also leaves asserts enabled.

This is not hypothetical — step 2 caught exactly that failure here. A first `nat_builtin` configure was interrupted
part-way through and left `CMAKE_CXX_FLAGS_RELEASE` **empty** in its cache. A later `build.py --update` reused the
damaged cache rather than regenerating it, and the build then completed with no error, producing a working but
**unoptimized** `onnxruntime_perf_test.exe`. The two arms' compile flags were otherwise character-identical:

```
nat_builtin: ... -DEIGEN_HAS_C99_MATH                   -std:c++20 -MD -Zi /GR /W4 ...
nat_plugin:  ... -DEIGEN_HAS_C99_MATH /O2 /Ob2 /DNDEBUG -std:c++20 -MD -Zi /GR /W4 ...
```

Had this gone unnoticed it would have produced a large, entirely bogus result in the plugin's favour. The fix is to
**delete the build directory and configure from scratch** — repairing a cache in an unknown state is not worth the
risk. A corrupted cache also shows a secondary tell: stray `CMAKE_ADDR2LINE`, `CMAKE_NM`, `CMAKE_OBJCOPY`,
`CMAKE_OBJDUMP`, `CMAKE_READELF`, `CMAKE_STRIP` and `CMAKE_TAPI` entries that the healthy cache does not have.

**Generalization: if a build directory was ever interrupted mid-configure, do not measure with it.** Interrupted
CMake configures fail silently in a way that survives `--update` and still produces a runnable binary.

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
- **Do not gate on the process exit code.** perf_test returns non-zero even on a fully successful run, because ORT's
  memory-leak checker reports a handful of 16-byte CRT static-initializer allocations (`initterm`) at shutdown. The
  report is emitted after timing and affects both arms identically, so it does not bias the comparison — but scripts
  should parse `Min Latency` from stdout rather than trusting the exit status.

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

Measured on the RTX 5060 Ti, on an otherwise idle machine. The two `build.py` configures differ by exactly one
generator flag, `-Donnxruntime_USE_EP_API_ADAPTERS=ON` on the plugin arm.

### Checks performed before measuring

| check | result |
| --- | --- |
| Normalized `CMakeCache.txt` diff | **2 lines, one key**: `onnxruntime_USE_EP_API_ADAPTERS` `OFF` vs `ON` |
| `/O2` / `/Ob2` / `/DNDEBUG` in `build.ninja` | 4184 built-in, 4183 plugin — off-by-one is WebGPU built in-tree |
| Binaries current with the tree | every commit since `873f221786` is doc-only, so the code under test is HEAD's |

### The plugin arm really is going through the plugin

A latency number is meaningless if the "plugin" arm quietly fell back to something else, so this was established by
making the alternatives fail rather than by reading the code:

| invocation on the plugin build | outcome |
| --- | --- |
| `-e webgpu` (built-in path) | fails: "WebGPU execution provider is not supported in this build" |
| `--plugin_eps BogusEpName` | fails: "[Plugin EP]: No matching EP devices found." (`common_utils.cc:171`) |
| `--plugin_eps WebGpuExecutionProvider` | runs |

The first line is the important one: WebGPU is genuinely absent from this build's `onnxruntime.dll`, so a run that
succeeds can only have come from `onnxruntime_providers_webgpu.dll`. The second confirms the EP-name lookup is real
and not silently permissive.

### Control: the two binaries are equivalent when no plugin is involved

Matching build flags still do not prove the two executables perform alike. Running both on the **CPU EP**, where
neither arm loads a plugin, isolates the plugin boundary from any residual build difference.

Run with default threading this control was actively misleading — `bench_compute` came out **24% faster** on the
plugin executable (4.61 ms vs 3.50 ms), which is not a real effect in any direction and simply shows how noisy the
multi-threaded CPU EP is here. Repeating it single-threaded (`-x 1 -y 1`, 10 rounds) removes the thread-pool
variance:

| `bench_compute`, CPU EP, single-threaded | min | median | max |
| --- | --- | --- | --- |
| built-in build | 20.011 ms | 20.285 ms | 20.734 ms |
| plugin build | 20.023 ms | 20.499 ms | 20.848 ms |

Ratio of minima **1.001**, medians within 1%. The two executables are performance-equivalent absent a plugin, so the
WebGPU differences below are attributable to the plugin boundary rather than to the builds.

### WebGPU A/B

8 interleaved rounds; `-r 30` for the bench models, `-r 20` for Qwen. "median" is the median of the per-round
minimums, matching the wasm procedure; "best" is the single fastest round, which is the most noise-resistant
statistic available.

| model | nodes | built-in median | plugin median | delta | ratio | best-of ratio | rounds plugin slower |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `bench_compute` | 16 | 1.662 ms | 1.917 ms | +0.255 ms | 1.153 | 1.054 | 8/8 |
| `bench_dispatch` | 300 | 5.118 ms | 5.562 ms | +0.444 ms | 1.087 | 1.084 | 8/8 |
| Qwen decode | 1043 | 34.95 ms | 36.95 ms | +1.99 ms | 1.057 | 1.092 | 6/8 |
| Qwen prefill | 1043 | 47.66 ms | 49.57 ms | +1.92 ms | 1.040 | 1.064 | 6/8 |

**The plugin EP is consistently slower.** Raw per-round minimums, in milliseconds, in round order:

| model | arm | per-round minimums |
| --- | --- | --- |
| `bench_compute` | built-in | 1.653, 1.626, 1.686, 1.661, 1.895, 1.877, 1.662, 1.614 |
| `bench_compute` | plugin | 1.700, 1.862, 2.055, 1.702, 1.971, 2.189, 1.701, 2.345 |
| `bench_dispatch` | built-in | 5.170, 5.131, 5.105, 5.158, 5.112, 5.124, 5.111, 5.104 |
| `bench_dispatch` | plugin | 5.577, 5.544, 5.623, 5.584, 5.543, 5.546, 6.091, 5.533 |
| Qwen decode | built-in | 31.34, 34.90, 40.56, 36.13, 35.01, 38.00, 30.57, 34.63 |
| Qwen decode | plugin | 35.32, 40.86, 43.65, 34.24, 36.33, 33.40, 39.88, 37.56 |
| Qwen prefill | built-in | 43.05, 49.42, 44.24, 50.92, 47.75, 47.56, 46.17, 48.23 |
| Qwen prefill | plugin | 49.99, 49.56, 50.59, 52.32, 48.80, 45.80, 49.58, 45.89 |

### How much of this is signal

The four rows are not equally trustworthy, and the medians alone hide that:

- **`bench_dispatch` is the reliable result.** The built-in arm spans 5.104–5.170 ms — a 66 µs range — and the
  per-round delta is +0.41 to +0.52 ms in seven of eight rounds (the eighth, +0.98 ms, is a visible outlier). An
  effect of +0.43 ms against that spread is unambiguous.
- **`bench_compute` is noisier than its median suggests.** Per-round deltas range from +0.039 to +0.731 ms. The
  plugin arm's slow rounds inflate the median to 1.153; the best-of ratio of 1.054 is the more defensible figure.
- **Both Qwen rows are noise-dominated.** Per-round deltas swing from −4.6 to +9.3 ms, and the sign flips in 2 of 8
  rounds. The direction agrees with the microbenchmarks on both the median and the best-of statistic, but these
  numbers should be read as "consistent with a few percent" and not as a measurement of it.

### Mechanism

Fitting the two microbenchmarks' best-of deltas as a fixed per-`Run` cost plus a per-node cost gives roughly
**67 µs per `Run` + 1.2 µs per node**. That is a two-point fit and should be treated as an order-of-magnitude model
only, but it does predict outside its fitting range: for Qwen's 1043 nodes it gives ≈1.3 ms, against an observed
median delta of ≈1.9–2.0 ms — the right magnitude, and under-predicting by about 1.5x. Per-node dispatch cost is
therefore a plausible dominant term, which is what the shape of the boundary would predict: each kernel dispatch
crosses the `OrtEp` C API adapter into a separate DLL instead of making an in-module C++ virtual call.

Compared with the wasm static-plugin A/B, which found ≈4.8 µs/node
([work log](static_plugin_ep_work_log.md#measured-cost-of-the-plugin-path-on-the-web-build)), the native
shared-library boundary costs roughly **1.2–1.5 µs/node**, some 3–4x less per node.

### Assessment against the criterion

A per-node cost means the overhead is a function of graph size, not a fixed tax, so no single percentage
characterizes it. On the real-model cases it lands at **4–6%**, and it grows as models get more numerous, smaller
kernels. Whether that is "within an accepted tolerance" is a judgement call that has not yet been made — the
tolerance has never been quantified, and it should be, since these results show the answer is not "zero overhead".

Two caveats on scope:

- Only steady-state `Min Latency` was compared. Session-creation cost, where the plugin must additionally discover
  and load the DLL, was not measured and is likely worse in relative terms.
- Both arms ran with default WebGPU options. `genai_config.json` uses `enableGraphCapture: "0"` and
  `validationMode: "basic"`; graph capture in particular could change the per-dispatch picture substantially, and is
  worth a follow-up since it targets exactly the cost identified here.

Reproduce with `run_native_ab.ps1` (parses `Min Latency`, ignores the exit code); raw rows land in
`native_ab_raw.csv`.

## Where the per-node cost goes (CPU profile)

The 1.2 µs/node figure above is far too large to be call-transport cost — a cross-DLL indirect call through the C API
is tens of cycles, while 1.2 µs at 3.696 GHz is ≈4,400 cycles. That gap motivated profiling the two arms to find out
what the cycles are actually spent on. The answer is that they are spent on **per-kernel marshalling work that the
built-in path does not do at all**, not on crossing the boundary.

### Method

Both arms were relinked with `/DEBUG` to produce PDBs. This was a link-only change: `LINK_FLAGS` already carried an
explicit `/OPT:REF,ICF,LBR`, so adding `/DEBUG` does not fall back to the `/OPT:NOREF,NOICF` debug defaults, and no
source file was recompiled. Binary sizes were unchanged and the A/B ratio was re-measured at 1.087 afterwards, so the
profiled binaries are the same ones the latency numbers came from.

Two profilers were tried and rejected before settling on a third:

- **ORT's own `-p` profiler cannot be used for this comparison.** Instrumentation is asymmetric — the built-in arm
  emits 15,300 `Api` category events and the plugin arm emits **zero**, so under `-p` the built-in arm measures
  *slower* (0.772 s vs 0.644 s over 50 runs), reversing the true result. It also inflates per-node time from ~17 µs
  to ~46 µs by forcing per-node synchronisation. One result did survive: **per-op median kernel time is identical on
  both arms (29.0 µs for Add/Mul/Sub)**, establishing early that GPU-side work is unchanged and the cost is host-side.
  That the plugin path silently loses the WebGPU EP's `Api` profiling events is itself an observability regression.
- **ETW/xperf resolved no symbols.** Traces collected fine (355k+ stack samples), but every frame rendered as
  `***unknown***` — zero `Module!Func` tokens in the entire report — despite symcache files being generated for all
  three PDBs, and with or without the Microsoft symbol server on the path.
- **A purpose-built sampling profiler** (`sampler.cpp`) was used instead: it launches the target, samples every thread's
  instruction pointer at ~1 kHz, and symbolizes leaf addresses with dbghelp against the local PDBs. Needs no
  elevation, and symbolizes ORT frames reliably.

Runs were 6000 reps of `bench_dispatch` (300 nodes), discarding the first 3 s, with the two arms pooled over two
repetitions. Sampling did not destroy the effect being studied: the latency ratio under the profiler was 1.083,
1.081 and 1.083 across three runs, against 1.087 unsampled.

### Result

Samples are counted only when a thread is actually running (leaf address not in a wait stub). One sample corresponds
to roughly one sampler loop period, ~1.0–1.2 ms of thread time, so **absolute µs below are approximate to about
±20%; the relative comparisons are not affected.**

| Scope | built-in | plugin | delta | ratio |
| --- | ---: | ---: | ---: | ---: |
| All modules, running samples/rep | 3.388 | 3.922 | +0.534 | 1.158 |
| `onnxruntime.dll` + `onnxruntime_providers_webgpu.dll` only | 1.175 | 1.441 | +0.267 | 1.227 |

Per-module numbers are meaningless on their own here, because the WebGPU code simply *moves* out of `onnxruntime.dll`
into `onnxruntime_providers_webgpu.dll`; only the combined ORT-side total is comparable.

The two totals cross-check the latency result. The extra CPU of +0.534 samples/rep is ≈0.53–0.64 ms/rep against a
measured extra wall latency of 0.42 ms/rep, and the ORT-side share alone (+0.267 samples/rep ≈ 0.9–1.1 µs/node) is
close to the 1.2 µs/node from the latency fit. **The overhead is host-side CPU work, and most of it is ORT-side.**

Attributing the +0.534 samples/rep by what the code is doing:

| Category | built-in | plugin | delta | share of delta |
| --- | ---: | ---: | ---: | ---: |
| other (mostly GPU driver, see below) | 1.752 | 1.952 | +0.199 | 37.3% |
| alloc/free | 0.573 | 0.669 | +0.096 | 17.9% |
| `Tensor`/`TensorShape` construction | 0.083 | 0.165 | +0.082 | 15.3% |
| EP API adapter | 0.002 | 0.074 | +0.072 | 13.4% |
| dawn (refcounting, error checks) | 0.495 | 0.549 | +0.054 | 10.1% |
| webgpu kernel bodies | 0.173 | 0.208 | +0.034 | 6.4% |
| ort framework | 0.128 | 0.118 | −0.011 | −2.0% |

The `ep-adapter` and `Tensor`/`TensorShape` rows are the reproducible core: they were +0.060/+0.063 and +0.030/+0.032
in the two individual runs, i.e. stable, and the adapter work is *new* — it is ~0.002 in the built-in arm, meaning it
does not exist there. The `dawn`, `webgpu-kernel` and `alloc/free` rows moved noticeably between the two runs and
should be read as directional only.

These symbols appear in the plugin arm and are absent from the built-in arm:

| Symbol | plugin samples/rep |
| --- | ---: |
| `onnxruntime::TensorShape::TensorShape` | 0.020 |
| `onnxruntime::webgpu::ComputeContext::Input<onnxruntime::Tensor>` | 0.016 |
| `onnxruntime::ep::adapter::CreateTensorFromApiValue` | 0.013 |
| `Ort::GetApi` | 0.013 |
| `onnxruntime::TensorShape::Allocate` | 0.013 |
| `onnxruntime::PluginEpOpKernel::Compute` | 0.008 |
| `onnxruntime::Tensor::operator=` | 0.007 |
| `onnxruntime::ep::adapter::KernelImpl::ComputeImpl` | 0.006 |
| `OrtApis::IsTensor` | 0.006 |
| `OrtApis::GetTensorElementTypeAndShapeDataReference` | 0.005 |
| `onnxruntime::DataTypeImpl::TensorTypeFromONNXEnum` | 0.004 |

### Mechanism

The profile points at `ep::adapter::CreateTensorFromApiValue`
([`include/onnxruntime/ep/adapter/tensor_helper.h`](../../../../include/onnxruntime/ep/adapter/tensor_helper.h)),
which rebuilds an `onnxruntime::Tensor` from an `OrtValue` on every access. Each call makes roughly nine C API round
trips (`IsTensor`, `GetTensorElementTypeAndShapeDataReference`, `GetTensorMemoryInfo`, then `GetAllocatorName`,
`GetAllocatorType`, `GetDeviceType`, `GetMemoryType`, `GetVendorId`, `GetDeviceId`), allocates a `std::string` for
the allocator name, constructs a `TensorShape` (a heap allocation once the rank exceeds the inline capacity), looks
up `DataTypeImpl::TensorTypeFromONNXEnum`, and constructs a `Tensor`.

The adapter `OpKernelContext` ([`op_kernel.h`](../../../../include/onnxruntime/ep/adapter/op_kernel.h)) is built
fresh per `Compute()` call and `resize()`s an input and an output `std::vector<Tensor>`. Its tensor cache lives only
for the duration of that one call, so the work above repeats for every input and every output of every node on every
`Run`. The built-in path does none of this: it receives `const Tensor*` directly from the executor.

That is the answer to the original question. The overhead is **not** the cost of crossing the DLL boundary; it is
redundant per-kernel wrapper construction and its allocation traffic, which is why it is ~4,400 cycles rather than
the tens of cycles an indirect cross-module call would cost.

### Caveats

- Part of the delta is plausibly *consequence* rather than cause. The `other` bucket is dominated by GPU driver
  modules (`nvwgf2umx.dll` +0.085, `win32u.dll` +0.017, `D3D12Core.dll` +0.014), and driver threads that poll will
  accumulate samples in proportion to wall time — which the plugin arm has more of. The unambiguous causal signal is
  the adapter and tensor-construction work, which is absent from the built-in arm entirely.
- Driver and `ntdll` frames symbolize to the nearest export (no PDBs available), so names such as
  `RtlCreateUnicodeString+0x118f` identify a region, not the actual function. Module attribution is still sound.
- Suspend/resume sampling perturbs both arms, but equally; the preserved latency ratio is the check on this.
- This is a flat (leaf-address) profile, so it attributes self time only and does not give call trees.

### Implication for the tolerance question

Open item 8 in the work log asks what tolerance is acceptable. These results argue against negotiating one yet: the
dominant identified cost is redundant work in the adapter, not an intrinsic property of the plugin boundary, so it
looks reducible — for example by caching the `Tensor` wrapper across `Compute()` calls, or by avoiding the
allocator-name string and shape allocation on the per-node path. The tolerance question is better asked after that
is attempted.
