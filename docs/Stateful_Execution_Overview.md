# From single inference calls to native stateful pipelines

## What this series delivers

Across **20 stacked PRs, #11 through #30**, this work adds a native stateful
execution layer **above ONNX Runtime's unchanged stateless Session API**.
Applications can define repeated inference, connect stages, retain execution
state, manage KV caches and sample tokens without writing their own generation
loop. The same foundation runs an autoregressive language model and an
iterative diffusion pipeline.

The delivered interfaces are **C, lifetime-safe C++ wrappers and typed Python**.
CPU implementations, actual CUDA-plugin cache/sampling implementations,
declarative Model Package chains and two real-model demonstrations are included.
As of 2026-11-04, the 20 PRs are open for review, not merged or a claim that these
APIs are already in a published ONNX Runtime release.

M16-M23, covering additional language bindings, are explicitly **deferred, not
completed**. This document is the contributor overview following M24's
[integrated qualification](Stateful_Qualification.md).

## New constructs and programming idioms

| Construct | Responsibility |
|---|---|
| **SessionLoop** | Repeatedly execute one initialized Session with explicit inputs, feedback, selected outputs and exit conditions. |
| **SessionChain** | Execute an ordered sequence of ordinary Sessions and SessionLoops, with named data/state connections. |
| **Per-execution context** | Own mutable progress, retained state, caches, counters, cancellation and applicable RNG state separately from reusable definitions. |
| **KVCacheManager** | Manage typed, contiguous cache tensors, logical length/capacity, sharing, copying, appending, growth and reset. |
| **Sampler** | Perform independently usable greedy token selection, on CPU or through a selected EP implementation. |
| **Bindings and immutable results** | Connect external inputs/sequences, earlier outputs and named state; expose selected/component/state snapshots with explicit lifetimes. |
| **Native generation and DDIM policies** | Supply token feedback/bootstrap or a fixed diffusion update policy without turning Python or the manifest into an execution engine. |

```text
Reusable initialized Sessions + loop/chain definitions
                         |
                  CreateExecution
                         |
            inputs + retained state + progress
                         |
                 Step / Run until exit
                         |
              immutable results/snapshots
                         |
                 explicit Reset / reuse
```

The important idioms are:

- **A definition is not a conversation.** Independent execution contexts do not
  inherit one another's mutable state. Completion and output access do not
  silently reset a context; Reset is explicit.
- **Termination is part of the interface.** Loops declare EndOfInputs, maximum
  iterations/generated tokens, EOS IDs or a scalar boolean model output.
  Conditions combine with OR semantics, including zero-limit boundaries.
  No custom predicate callbacks, implicit termination or extra iteration.
- **Named wiring replaces positional assumptions.** Tensor names may differ
  between stages; external sequences, state, feedback and upstream outputs
  have distinct meanings. Invalid wiring fails instead of producing an empty
  success, and downstream stages do not run after a component failure.
- **Step describes progress, not necessarily a model call.** A decode bootstrap
  can sample prefill logits with `executed=false` and a generated token present.
  Inspect completion separately. This emits the first token exactly once,
  avoids a redundant decoder invocation, and permits sampler-free prefill.
  Zero decode budget can still allow the separately configured prefill.
- **State is not synonymous with KV cache.** Token generation uses KV caches;
  diffusion retains latents, timestep and scheduler/RNG state directly.
- **Ownership and copies are explicit.** Independently retained result/value
  handles survive later steps/reset and release of definitions/executions.
  Borrowed C output pointers still require their owning result to stay alive.
  Python GPU
  values require explicit `copy_to_cpu()` before NumPy conversion; writable
  OrtValue conversions produce independent copies. Caller-owned feeds/streams
  retain their documented lifetime and non-mutation obligations.
- **Failure has a recovery contract.** Cancellation or failure after possible
  advancement requires Reset. One mutable context is not a concurrent/reentrant
  execution object; this is synchronous sequential orchestration.

The [typed Python guide](Stateful_Python.md) and
[tiny executable examples](../samples/python/stateful_execution.py) show these
contracts without downloading real models. Public entry points are exposed
through [`OrtApi::GetStatefulApi()` and `OrtStatefulApi`](../include/onnxruntime/core/session/onnxruntime_c_api.h),
with [C++ RAII wrappers](../include/onnxruntime/core/session/onnxruntime_cxx_api.h)
and [`onnxruntime.stateful`](../onnxruntime/stateful.py). Python binds native
execution; it does not implement a second scheduler or generation engine.

## Execution Provider interfaces: stateful work becomes extensible

Assigning model nodes to CUDA does not establish that cache management or token
selection runs on CUDA. This series makes those **independent, observable
capabilities**, rather than assuming an EP supplies an entire generation engine.

### Extension points

- In-tree providers may return an optional immutable descriptor from
  [`IExecutionProvider::GetStatefulProvider()`](../include/onnxruntime/core/framework/execution_provider.h).
  The default is no override.
- Plugin providers register through
  [`OrtEpApi::RegisterStatefulProvider()`](../include/onnxruntime/core/session/onnxruntime_ep_c_api.h)
  during synchronous `OrtEpFactory::CreateEp`. Registration copies/validates
  the descriptor and rolls back pending resources on failed creation.
  Existing `OrtEp` fields are not extended or read speculatively; legacy
  plugins do not need to register.
- A versioned, size-tagged
  [`OrtStatefulProviderDescriptor`](../include/onnxruntime/core/session/onnxruntime_c_api.h)
  advertises operation, element-type, memory-type and device capabilities.
  It supplies synchronous `CacheOperation`/`Greedy` callbacks and context
  `Retain`/`Release` hooks. Only C ABI types cross the plugin boundary.
  Existing public API tables keep their prior layouts; additions are append-only.

The discoverable operations are **initialize, share, copy, append, reserve and
reset cache**, plus **greedy sampling**. A provider can implement a subset.
Cache and sampler choices are independent of each other and of model variant/
node selection.

### Selection, resource safety and proof

`CPU`, `PREFER` and `REQUIRE` policies make intent explicit. `REQUIRE` fails
when the requested capability/device is unavailable. `PREFER` is not permission
for an undisclosed device migration: fallback must obey the memory contract.
A callback can explicitly delegate to the CPU reference only for compatible
CPU operands; delegation is reported. An advertised callback failure is
propagated, not silently retried on CPU.

Descriptors/contexts, allocator and stream-dependent resources, and callback-
bearing plugin libraries stay alive while retained stateful objects need them,
subject to the usual Env/caller-resource obligations. Published values remain
immutable, and synchronous callbacks must complete work before returning.

Capability queries and per-operation dispatch counters report requested versus
actual implementation, CPU/provider calls, delegation and failures.
Host staging counters are explicitly **not** measured DMA counters.

The implemented CUDA-plugin backend provides GPU-resident cache operations
and FP32/FP16 device-native greedy reduction, using the selected device,
allocator and stream. NaN/tie and other supported input semantics match the CPU
contract. Normal sampling returns a compact token/status record instead of
copying full logits for host argmax. Copies needed for cache growth or immutable
snapshots are permitted: this is not a universal zero-copy promise.
**Legacy CUDA model execution is not a substitute for these plugin callbacks.**

Real-model qualification also exposed and fixed a narrow pre-existing adapter
defect: [`KernelDef::OpName()`](../include/onnxruntime/ep/adapter/kernel_def.h)
returned a node's display name instead of its operator type. Correcting it lets
arbitrarily named SkipGroupNorm nodes preserve their required residual outputs;
the original UNet can run without changing its weights or kernel arithmetic.

## Model Package: from selecting one component to loading a pipeline

The existing model/variant/shared-asset mechanisms remain. The new separation
is between **portable pipeline description**, **runtime interpretation** and
**execution state**:

| Layer | What changed |
|---|---|
| Standalone package format/library | Schema **1.1** adds an optional named `chains` map. Each chain contains ordered steps with unique IDs and existing component references. Components may be reused across steps. |
| Consumer-specific metadata | Chain/step `executor_info.ort` describes Session versus SessionLoop, bindings, explicit exits, cache/sampler/provider choices, generation/bootstrap and DDIM policy. |
| ORT package consumer | `OrtModelPackageApi::CreateSessionChain` selects variants, validates executable bindings/policies, creates initialized per-step Sessions and returns an owned native chain. |
| Standalone authoring/export | JSON chain get/set/remove APIs, reference validation, version-preserving round trips and self-contained save-as support portable pipeline artifacts. |

The [standalone C library](../model_package/README.md) remains independent of
ORT/ONNX execution. It validates common structure and references while treating
consumer payloads as opaque. The
[ORT consumer](../onnxruntime/core/session/model_package/README.md) validates its
own strict execution namespace. Future common-schema fields do not authorize
unknown ORT policies.

Existing schema-1.0 packages and single-component Session creation remain
supported. Reading/unrelated edits do not force an old package to 1.1; adding
chains upgrades it. Callers explicitly select a chain by name. Package-built
chains own their initialized Sessions; execution/result retention does not
require keeping the public package handle open. Optional per-step
SessionOptions replace complete default snapshots, enabling deliberately mixed
CPU preprocessing and CUDA model stages.

Save-as copies variant contents, model external data and supported metadata,
rebases structural paths and retains shared-asset rules. It rejects symlinks,
overlapping trees and output collisions. Opaque consumer paths are not magically
discovered or rewritten; required files belong in variant/shared-asset locations.
Failure preserves the live package, but directory-level rollback of partial
destination files is not promised.

**The manifest is not a state machine or a script.** It describes components,
order, connections and supported policies, not arbitrary jumps, Python
callbacks or user-defined transition predicates. Native implementations own
iteration and state updates. Python currently exposes package inspection/loading,
not the standalone C library's authoring/export API.

## Two complete examples and how to reproduce them

### Prerequisites shared by both examples

Use the complete stack through #30 (or the overview branch based on it), not an
unmodified upstream checkout or an older published wheel. Build/install the
matching local CPU wheel for CPU qualification or **CUDA-plugin** wheel for
native GPU cache/sampling. See the
[source-build guide](https://onnxruntime.ai/docs/build/inferencing.html) and
[CUDA-plugin build instructions](cuda_plugin_ep/QUICK_START.md).
The plugin build requires
`--cmake_extra_defines "onnxruntime_BUILD_CUDA_EP_AS_PLUGIN=ON"`; the default
legacy CUDA build does not provide the new native cache/sampler hooks.

Activate an isolated environment, point to the matching local wheel installation
if it is not installed directly in that environment, and run outside the source
root to avoid import shadowing:

```bash
ROOT=/path/to/onnxruntime
source "$ROOT/.venv/bin/activate"
export PYTHONPATH=/path/to/matching-local-wheel-install
cd "$HOME"
python -c 'import onnxruntime as ort; print(ort.__file__); print(ort.get_build_info())'
```

For GPU runs, configure loader paths for the CUDA/cuDNN versions used to build
that wheel and verify `cuda-plugin-ep=1` in its build info; the provider name
alone does not distinguish plugin from legacy CUDA.
Qualification used Linux/A100, CUDA13.0 and cuDNN9.23.1; it is not
evidence for every GPU architecture or platform.
Setup is explicitly network-enabled; Run/Validate use verified local assets
only. Models and their licenses live outside Git.

### Phi-3: two loops, one conversation

The package runs:

```text
prompt -> sampler-free prefill SessionLoop -> decode SessionLoop -> token IDs/text
                         \---- 64 shared managed key/value caches ----/
```

Prefill consumes one full prompt and exits on EndOfInputs. Decode first samples
the retained prefill logits, then feeds each pending token back into the model.
EOS/token limits are explicit. Both stages reuse one model/weight file per
variant; no weight duplication or Python autoregressive loop is required.

ONNX files on Hugging Face:

- [Phi-3 CPU INT4](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx/blob/5f5f794c1c23c9d5ee142af85df02a6cc52d6945/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/phi3-mini-4k-instruct-cpu-int4-rtn-block-32-acc-level-4.onnx),
  with FP32 logits/cache.
- [Phi-3 CUDA FP16](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx/blob/5f5f794c1c23c9d5ee142af85df02a6cc52d6945/cuda/cuda-fp16/phi3-mini-4k-instruct-cuda-fp16.onnx).

The acquisition command below also fetches the required external weights and metadata.
Local tokenization/chat formatting and rendering stay application-owned.

The [sample Phi-3-style decoder-chain manifest](phi3-sample-manifest.json) shows
sampler-free prefill and decode with explicit exits. It uses placeholder
`model1.onnx`/`model2.onnx` filenames and one layer's key/value caches, not the
real Phi-3 signatures. Use the build command below to generate the runnable
Phi-3 package with its actual model paths, 64 caches and generation metadata.

Choose `TARGET=cpu` with the local CPU wheel, or `TARGET=cuda` with the local
plugin wheel, then run:

```bash
python -m pip install -r "$ROOT/samples/python/requirements-phi3-demo.txt"
PHI="$HOME/onnxruntime-stateful/phi3-mini-4k-instruct"
TARGET=cpu
DEMO="$ROOT/samples/python/phi3_demo.py"
python "$DEMO" --assets "$PHI" --target "$TARGET" acquire
python "$DEMO" --assets "$PHI" --target "$TARGET" inspect
python "$DEMO" --assets "$PHI" --target "$TARGET" build --max-new-tokens 16
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python "$DEMO" --assets "$PHI" --target "$TARGET" run \
  --prompt 'Answer briefly: what is 2 + 2?'
python "$DEMO" --assets "$PHI" --target "$TARGET" validate \
  --prompt 'Answer briefly: what is 2 + 2?'
```

The package is at `"$PHI/$TARGET"`; output JSON reports text, tokens, exit
reasons, invocation counts and actual sampler routing. Rebuild to change the
budget. Validation compares against ordinary Sessions for the **same** model
variant, not identical CPU-INT4 versus CUDA-FP16 prose.
See the [Phi-3 guide](Stateful_Phi3_Demo.md) for asset size, licensing, loader
configuration, adapters and full reference checks.

### Stable Diffusion 1.5: native iteration beyond token generation

```text
positive/negative text encoders
  -> UNet SessionLoop + native DDIM/guidance
  -> scaled VAE -> declared CPU CLIP preprocessing -> CUDA safety checker
  -> checked PNG writer
```

Defaults are 512x512 RGB, seed 42, 20 DDIM updates, guidance 7.5 and empty negative
prompt. Batched guidance takes 20 UNet calls / 25 total model invocations;
sequential guidance takes 40 / 45. Latents/scheduler state are not KV caches.
Python only provides application boundaries and validation/image tooling.

ONNX files on Hugging Face:

- [Text encoder](https://huggingface.co/tlwu/stable-diffusion-v1-5-onnxruntime/blob/f22c0e90205a7a2c08bf3e8927b39dc84e4064c2/text_encoder/model.onnx)
- [UNet denoiser](https://huggingface.co/tlwu/stable-diffusion-v1-5-onnxruntime/blob/f22c0e90205a7a2c08bf3e8927b39dc84e4064c2/unet/model.onnx)
- [VAE decoder](https://huggingface.co/tlwu/stable-diffusion-v1-5-onnxruntime/blob/f22c0e90205a7a2c08bf3e8927b39dc84e4064c2/vae_decoder/model.onnx)
- [Safety checker](https://huggingface.co/tlwu/stable-diffusion-v1-5-onnxruntime/blob/f22c0e90205a7a2c08bf3e8927b39dc84e4064c2/safety_checker/model.onnx)

These optimized graphs are **CUDA-only**; CPU EP tests use deterministic
fixtures. Explicit FP32/FP16 boundaries preserve original initializers, apply
VAE scaling once and retain mandatory safety. The selected native policy is
`sd15_ddim_v1` DDIM, **not upstream PNDM parity**. Its FP32 scheduler/guidance
remains on CPU, with intentional prediction readbacks.

Use the local plugin wheel and its loader paths:

```bash
# Optional reference dependencies stay outside the core runtime:
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r "$ROOT/samples/python/requirements-sd15-demo.txt"
SD="$HOME/onnxruntime-stateful/sd15"
DEMO="$ROOT/samples/python/sd15_demo.py"
python "$DEMO" --assets "$SD" acquire
python "$DEMO" --assets "$SD" build
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python "$DEMO" --assets "$SD" run \
  --prompt 'A peaceful mountain lake at sunrise, soft golden light, landscape painting' \
  --seed 42 --output "$SD/qualification/overview.png"
python "$DEMO" --assets "$SD" validate \
  --prompt 'A peaceful mountain lake at sunrise, soft golden light, landscape painting' \
  --seed 42 --output "$SD/qualification/overview-validated.png"
```

The package is `"$SD/package"`. Output paths must be new. A flagged safety
verdict is an explicit error and saves no unfiltered image. The package records
the qualified `cudnn_conv_algo_search=HEURISTIC` option; rebuild older packages.
Real CUDA reductions need not yield byte-identical PNGs: validation retains
fixed local/component/exact-safety checks and independent latent/image RMSE
limits 0.01 / 0.005. See the [SD1.5 guide](Stateful_SD15_Demo.md) for provenance,
CreativeML license, exact preprocessing and numerical policy.

## Evidence, capabilities and deliberate boundaries

M24's [report](Stateful_Qualification.md) and
[per-file coverage ledger](Stateful_Qualification_Coverage.json) record:

- **7,369/7,369** cumulative current executable production locations covered,
  without exclusions: 5,413 native host, 1,931 Python and 25 actual CUDA locations.
  This is scoped new-code line coverage, not whole-repository or branch coverage.
- CPU Python 73 tests / 217 subtests and plugin 77 / 235; native/public CPU 85 / 153 and
  plugin 157 / 216; standalone shared/static package 5 / 5 each, with explicit legacy
  compatibility results.
- Independent real-model correctness, reset/replay, ownership/lifetime and
  actual CUDA dispatch/transfer evidence, rather than a CPU fallback masquerading
  as GPU qualification.
- Memory/full-leak and applicable synchronization/race checks. SD race evidence
  is explicitly bounded actual GroupNorm sampling, not full-model race freedom.
- Eight reproducible native-only baseline workloads, with cold/steady/phase/
  memory/copy scopes separated. These are instrumented-build observations,
  **not** ordinary-Session speedup or cross-precision performance claims.

For example, the 16-token Phi-3 CUDA baseline records 120.33 tokens/s; SD batched
records 1.456 images/s on the qualified host. These illustrate measured workloads,
not portability or performance guarantees. The
[qualification helper](../samples/python/stateful_qualification.py) can collect
`latency`, `resources` and `steps` reports from the explicit local packages.

Intentional limits remain: greedy-only sampling; fixed DDIM rather than arbitrary
schedulers; contiguous caches rather than paging/offload; no chunked prefill,
cross-request prefix reuse, concurrent execution, async streaming, speculative
decoding or serving/batching engine. Additional stateful language bindings and
other platform/GPU qualification are outstanding.
The known stateless plugin negative-device RuntimeError test mismatch and
legacy metadata warning remain documented, not counted as passed.

## Why this moves ONNX Runtime forward

1. **A reusable execution contract, not another application loop.** Repeated
   inference, state handoff, exit semantics and failure/lifetime rules become
   native runtime capabilities shared by language bindings and packages.
2. **Hardware customization has a precise boundary.** EP authors can accelerate
   cache and sampling independently, advertise partial support and prove actual
   dispatch without replacing Session or implementing a whole serving stack.
3. **Packages become deployable multi-stage applications.** One artifact can
   select models/variants and declare a native executable pipeline while
   preserving path confinement, consumer separation and backward compatibility.
4. **The abstraction serves more than LLMs.** Phi-3 and diffusion exercise the
   same loop/chain/state foundation with different state and update policies.
5. **Future work has a measured foundation.** Explicit ownership, immutable
   snapshots, ABI-safe hooks, independent references and current-source coverage
   give contributors concrete extension points and regression evidence for
   future bindings, policies and performance work.

This is a foundation for those extensions, not evidence that the deferred
features or all-language parity have already been delivered.

## The 20-PR roadmap

All links below refer to `onnxruntime/ort-state`; each PR is independently
documented/tested on its predecessor. Split milestones account for the difference
between milestone numbers and PR count.

| PR | Milestone | Main contribution |
|---|---|---|
| [#11](https://github.com/onnxruntime/ort-state/pull/11) | M01 | Contracts, deterministic CPU/CUDA references and coverage readiness |
| [#12](https://github.com/onnxruntime/ort-state/pull/12) | M02 | Per-execution state and explicit SessionLoop |
| [#13](https://github.com/onnxruntime/ort-state/pull/13) | M03 | SessionChain and named tensor/state bindings |
| [#14](https://github.com/onnxruntime/ort-state/pull/14) | M04 | CPU KVCacheManager and loop/chain integration |
| [#15](https://github.com/onnxruntime/ort-state/pull/15) | M05 | CPU greedy sampling and prefill/decode generation |
| [#16](https://github.com/onnxruntime/ort-state/pull/16) | M06a | Public C stateful API |
| [#17](https://github.com/onnxruntime/ort-state/pull/17) | M06b | Lifetime-safe C++ wrappers |
| [#18](https://github.com/onnxruntime/ort-state/pull/18) | M07a | Built-in EP capabilities and stateful dispatch |
| [#19](https://github.com/onnxruntime/ort-state/pull/19) | M07b | Plugin registration and retained resource/library lifetimes |
| [#20](https://github.com/onnxruntime/ort-state/pull/20) | M08a | Standalone CUDA-plugin KV cache backend |
| [#21](https://github.com/onnxruntime/ort-state/pull/21) | M08b | GPU-resident managed caches in loops/chains |
| [#22](https://github.com/onnxruntime/ort-state/pull/22) | M09a | Standalone native CUDA greedy sampling |
| [#23](https://github.com/onnxruntime/ort-state/pull/23) | M09b | GPU logits/sampling integrated into loops/chains |
| [#24](https://github.com/onnxruntime/ort-state/pull/24) | M10 | Package chain schema, authoring and self-contained export |
| [#25](https://github.com/onnxruntime/ort-state/pull/25) | M11 | Owned package-chain loading and initial-logits bootstrap |
| [#26](https://github.com/onnxruntime/ort-state/pull/26) | M12 | Native DDIM loops, state and package policy |
| [#27](https://github.com/onnxruntime/ort-state/pull/27) | M13 | Typed Python/native/package API parity |
| [#28](https://github.com/onnxruntime/ort-state/pull/28) | M14 | Real Phi-3 CPU/CUDA package and text demo |
| [#29](https://github.com/onnxruntime/ort-state/pull/29) | M15 | Real safety-checked CUDA SD1.5 package and image demo |
| [#30](https://github.com/onnxruntime/ort-state/pull/30) | M24 | Integrated qualification, native baselines and cumulative evidence |

For contracts and detailed execution history, see [SPEC](../SPEC.md) and
[TASKS](../TASKS.md). For implementation, start with the
[native session layer](../onnxruntime/core/session/),
[generation policies](../onnxruntime/core/generate/),
[DDIM](../onnxruntime/core/diffusion/),
[CUDA-plugin backend](../onnxruntime/core/providers/cuda/plugin/) and
[Python binding](../onnxruntime/stateful.py).
