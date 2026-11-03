# Stateful Python execution

`onnxruntime.stateful` is a typed binding to ONNX Runtime's native SessionLoop,
SessionChain, KVCacheManager, Sampler and Model Package APIs. Ordinary
`onnxruntime.InferenceSession` remains stateless. There is no Python scheduler,
custom predicate, implicit iterator, async API or tokenizer dependency.

Use a locally built wheel with Python enabled. Minimal builds keep configuration
types importable but report an explicit unavailable-feature error when execution
is requested.

## Executable examples

For the real CPU INT4/CUDA FP16 model-package example, see
[Local Phi-3 stateful text generation](Stateful_Phi3_Demo.md).

[stateful_execution.py](../samples/python/stateful_execution.py) generates tiny
ONNX models and checks feedback accumulation, sampler-free prefill followed by
sample-only decode bootstrap, and both deterministic DDIM guidance conventions.
It requires NumPy and ONNX, but downloads no models.

Run outside the repository root so the source package cannot shadow an installed
wheel:

```bash
python /path/to/onnxruntime/samples/python/stateful_execution.py
python /path/to/onnxruntime/samples/python/stateful_execution.py \
  --provider CUDAExecutionProvider
```

The CUDA command requires the matching GPU wheel, driver and CUDA/cuDNN loader
paths. It disables CPU **model-node** fallback. These examples intentionally use
the default CPU cache/sampler policies, including when the model runs on CUDA.
DDIM's scheduler/guidance also remains CPU FP32; full CUDA prediction readbacks
are intentional. This is a fixed `sd15_ddim_v1` policy, not arbitrary Diffusers
scheduler compatibility.

## Definitions, executions and explicit exits

```python
import numpy as np
import onnxruntime as ort
from onnxruntime import stateful as sf

session = ort.InferenceSession("feedback.onnx", providers=["CPUExecutionProvider"])
config = sf.LoopConfig(
    output_names=("present",),
    exit_conditions=(sf.ExitCondition(sf.ExitKind.END_OF_INPUTS),),
    finite_input_names=("x",),
    feedback=(sf.FeedbackBinding("past", "present"),),
)
with sf.SessionLoop(session, config) as loop:
    execution = loop.create_execution(
        {"past": np.array(0, dtype=np.float32)},
        batches=[{"x": np.array(2, dtype=np.float32)}],
    )
with execution, execution.run() as result:
    independent_array = result.get_outputs()["present"].numpy()
```

Frozen configuration records copy nested configuration collections; they do not
copy tensor feeds. Use enum members rather than strings/integers. Unknown keyword
arguments, unsupported types and invalid native configurations raise errors.

Each definition captures the **exact initialized low-level Session**. A later
`session.set_providers()` affects newly constructed definitions, not existing
ones. A definition can create independent execution contexts. Every loop declares
its exits: finite-input exhaustion, maximum iterations/generated tokens, EOS or
an explicit boolean model output. There is no hidden generation-limit override.

`SessionChain(ChainConfig(...))` connects ordered Session/SessionLoop components
using `InputBinding`, `ValueSource`, `StateBinding` and `ChainOutput`. External
inputs, finite sequences, earlier component outputs and named state are distinct
namespaces. A chain execution accepts `inputs`, `sequences` and `initial_state`.

Use `step()` for progress or `run()` for completion, not both expecting a replay.
`reset()` restores the initial execution using its originally captured feeds.
Decode bootstrap's first step samples prefill logits without invoking the decode
model (`executed=False`); generated-token and model-invocation counts differ.
Zero decode budget still permits declared sampler-free prefill.

## Tensor ownership and explicit copies

Inputs accept supported numeric/bool NumPy arrays, public `ort.OrtValue` objects,
or `sf.StatefulValue`. NumPy arrays must be contiguous, native-endian and already
have the required dtype/shape. No implicit cast or contiguity copy is performed.
The binding retains storage owners, but the caller must **not mutate borrowed
storage while native work can use it**, including later steps or Reset replay.
Native model/provider policies may still require staging or copying.

Result/state/cache collections are immutable snapshots. Their `StatefulValue`
handles expose metadata without reading back GPU payloads. They expose no writable
buffer, raw pointer, DLPack or implicit array alias.

| Method | Result | Copy behavior |
|---|---|---|
| `get_info()` | Frozen shape/type/device metadata | No tensor readback |
| `numpy()` | Independent writable NumPy array | CPU only; GPU values raise `ValueError` |
| `to_ortvalue()` | Writable public OrtValue | Independent copy on the original device |
| `copy_to_cpu()` | Writable CPU OrtValue | Explicit full CPU copy/readback |

Children and snapshots retain dependencies and survive parent `close()` or GC,
including provider-backed values after library unregistration. Use context
managers or idempotent `close()` for predictable release. Closing a parent does
not close independently retained children. Caller-created streams must remain
valid until all Sessions, executions and values using them are released.

## Cache, sampling and provider policy

Use `CacheSpec` and `CacheBinding` for managed loop/chain caches, or standalone
`KVCacheManager` with `update()`, `reserve()`, `reset()`, `get_view()` and
diagnostics. Share, Copy and Append have distinct native semantics.
**Share requires allocator-owned tensors**; `OrtValue.ortvalue_from_numpy()` on
CPU borrows NumPy storage and is not an implicit owning-copy shortcut. For an
owning tensor, use an allocator-backed OrtValue or
`session.create_ortvalue_from_shape_and_type()` and explicitly populate it.
Managers conservatively retain input owners until close; retained views have
their own lifetimes.

Standalone `Sampler` performs native greedy sampling. Cache and sampler policies
are independent. For native CUDA cache/greedy, use the CUDA **plugin** build and
an initialized matching Session:

```python
required_cuda = sf.ProviderSelection(
    sf.ProviderMode.REQUIRE, "CUDAExecutionProvider", sf.DeviceType.GPU, 0
)
with sf.Sampler(provider=required_cuda, session=session) as sampler:
    token = sampler.sample(readonly_gpu_logits)
    route = sampler.get_dispatch_info()
    assert route.origin == sf.DispatchOrigin.PROVIDER
```

`REQUIRE` raises rather than silently using CPU. `PREFER` follows native
capability/fallback rules. Legacy CUDA supports CUDA model execution/DDIM but
does not implement the plugin's native cache/greedy callbacks.
`query_provider_capabilities()` reports actual registered capabilities.
Execution `get_dispatch_info(feature, operation, component=0)` reports observed
dispatch; model placement alone does not prove cache/sampler routing.
`get_staging_info()` reports CPU payload counters, **not measured CUDA DMA**.

## Read-only Model Packages

```python
with sf.ModelPackage("/models/package") as package:
    info = package.get_info()
    chain = package.create_chain("generate", session_options)
with chain:
    execution = chain.create_execution(
        sequences={"prompt": [np.array([[1, 2]], dtype=np.int64)]}
    )
with execution, execution.run() as result:
    token_ids = result.get_outputs()["tokens"].numpy()
```

Use the input/output names declared by the selected manifest chain.
`get_info()` copies schema-major/component/variant metadata.
`resolve_string_ref()` uses the native portable/installed path rules.
`select_component(name, session_options)` selects a variant; its
`create_session(session_options)` adopts one initialized ordinary Session without
loading/initializing twice. Selection options and Session creation options keep
the native API's separate precedence; pass execution options explicitly.

Package-created ordinary Sessions disable automatic Python provider fallback.
Explicit `set_providers()` reloads the same selected model file, not a different
package variant. `create_chain(name, defaults, step_session_options={...})`
requires a named chain; per-step options are **complete replacements**, not
dictionary merges. Callback registrations/options are captured before native
work and retained with their actual Sessions, including cloned SessionOptions.
Package authoring/export wrappers are intentionally deferred; the standalone
[Model Package library](../model_package/README.md) remains available.

## Concurrency and errors

Long native work releases the GIL. Mutable operations on one resource are
non-overlapping: busy `step`, `run`, `reset`, state/diagnostic access or `close`
raise `RuntimeError` rather than queueing. Independent execution contexts may run
concurrently. `cancel()` alone may overlap execution, including from another
Python thread. Cancellation is observed at native boundaries, not GPU preemption.

After failed/possibly advanced work, including a Python result-delivery allocation
failure, the context requires `reset()`. Check `requires_reset()` and retain prior
snapshots as needed; do not retry an advanced operation implicitly.
Native status errors use existing ONNX Runtime Python exceptions; Python argument
errors remain explicit `TypeError`/`ValueError`.
