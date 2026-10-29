# Mutable opaque state in ONNX Runtime

ONNX opaque types let a model carry state whose natural implementation is a C++
object without exposing that object's layout as tensors. ONNX Runtime can preserve
the graph's functional meaning while reusing the physical object when a kernel
declares that reuse is supported and the runtime proves it is safe.

This document describes two supported cases:

- planner-controlled reuse of graph-internal opaque values; and
- caller-requested reuse when the same opaque object is bound as a model input and
  its directly related model output.

Both use `KernelDefBuilder::MayInplace`. It is permission to reuse, not a requirement,
and every participating kernel must also implement the distinct-output path.

## Opaque state and functional graphs

Consider a deterministic pseudo-random number generator:

```cpp
class Prng {
 public:
  void Seed(uint64_t seed);
  float Next();
};
```

Its model type can be identified without exposing its representation:

```text
opaque(example.prng,State)
```

The opaque domain and name identify the abstract type. They do not describe object
size, layout, allocation, destruction, or copying. The runtime datatype registration
and kernels provide those details.

A state-changing method is represented as a functional transition:

```text
Seed(seed: scalar tensor(int64)) -> State
Generate(state: State) -> (value: scalar tensor(float), next_state: State)
```

Successive operations explicitly thread logical state:

```text
s0 = Seed(seed)
(v1, s1) = Generate(s0)
(v2, s2) = Generate(s1)
```

`s0`, `s1`, and `s2` remain distinct graph values even when ORT implements a safe
transition by advancing one physical C++ object.

## Safe graph-internal reuse

A kernel that supports both aliased and distinct state objects declares:

```cpp
def.MayInplace(0, 1);
```

For the example `Generate` operator, input 0 is the old state and output 1 is the
next state. This declaration says that output 1 may reuse input 0.

The allocation planner permits reuse only when its existing safety rules allow it.
For opaque values, compatibility additionally requires:

- both values are opaque;
- their opaque domain and name match; and
- their planned memory locations match.

The usual liveness, observability, execution-order, and execution-mode restrictions
still apply. In particular:

- a graph input remains caller-owned and is not selected for planner reuse;
- a state that is still observable as a graph output is preserved;
- a fork preserves the old logical state for other consumers; and
- parallel execution remains conservative.

Opaque compatibility alone never authorizes mutation. The kernel's `MayInplace`
mapping and the planner's safety analysis are both required.

### Forked state

A functional value may have multiple consumers:

```text
(left1, left_state1) = Generate(s0)
(right1, right_state1) = Generate(s0)
```

Both consumers must observe the same original `s0`. ORT may allow one last-use edge
to reuse the original object, but another edge needs a distinct output. The kernel
therefore must produce the same logical result in both cases.

The test kernel follows this pattern:

```cpp
const auto* input = ctx->Input<TestPrng>(0);
auto* output = ctx->Output<TestPrng>(1);
if (input != output) {
  output->CopyFrom(*input);
}
*ctx->Output(0, TensorShape{})->MutableData<float>() = output->Next();
```

ORT does not generically clone opaque objects. The kernel implements the fallback,
whether by copying logical state or by computing the result another way without
modifying the input.

## Caller-bound model input and output

Opaque state often persists across model runs. A caller can request same-object
mutation for the direct state-update case by binding one concrete object as both
the model input and its preallocated output:

```text
state = OrtValue(PRNG object)

BindInput("state", state)
BindOutput("next_state", state)
Run()
```

Copying an `OrtValue` shares ownership of its underlying object, so the input and
output wrappers need not be the same wrapper instance. They must refer to the same
object and shared ownership control block.

Before execution, ORT validates an opaque input/output alias request. The supported
relationship must:

- match exactly one opaque model input and one opaque preallocated output;
- connect that graph input directly to that graph output through the same kernel;
- have exactly one input use, including implicit control-flow captures and repeated
  input positions, rather than a fork or multi-output alias;
- preserve a graph input that is also a graph output, even if that pass-through
  output is unrequested or has no preallocated fetch;
- use the same planned device; and
- correspond to the kernel's `MayInplace(input_index, output_index)` declaration.

Validation happens before kernels execute. A rejected request therefore leaves the
caller object unchanged.

For an accepted request:

- the caller permits the object to be mutated during the run;
- successful completion leaves the same object holding the logical output state;
- the caller must not concurrently read, modify, or run with that object; and
- a failure after execution begins may leave the object partially updated.

Binding distinct input and output objects remains fully supported. In that case the
kernel preserves the input and writes the next state to the output object.

A graph input returned unchanged as an output is a pass-through, not a mutable
alias request. It remains supported even when multiple inputs share one caller
object, with either preallocated or unallocated pass-through fetches.

### Why the direct relationship is required

ORT has no generic operation for copying arbitrary opaque state back into a
caller-provided object. A longer path such as:

```text
input_state -> Generate -> temporary_state -> Generate -> output_state
```

may compute the correct final logical value in a temporary object, but the framework
cannot assign that private state into the bound output object without type-specific
help. The current contract therefore accepts only a direct input-to-output kernel
relationship and rejects broader alias requests before execution.

## Tensor key-value cache precedent

Tensor-backed state already uses the same API convention. The CPU
[`TensorScatter` kernel](../onnxruntime/core/providers/cpu/llm/tensorscatter.cc)
declares:

```cpp
def.MayInplace(0, 0);
```

Its
[`InPlace_IOBinding` test](../onnxruntime/test/providers/cpu/llm/tensorscatter_op_test.cc)
binds the same tensor `OrtValue` as the past-cache input and present-cache output.
The kernel handles both shared and distinct buffers.

CUDA and WebGPU Group Query Attention kernels similarly declare cache input/output
`MayInplace` mappings. Their operator-specific attributes and capacity conventions
are separate from the fundamental mechanism: a kernel advertises an alias-safe pair,
and the caller can bind shared storage for that pair.

Opaque state follows this precedent while adding opaque-specific boundary validation,
because arbitrary C++ objects do not have tensor shape metadata or a generic copy-back
operation.

## Object creation, ownership, and lifetime

[`OrtValue`](../include/onnxruntime/core/framework/ort_value.h) stores a
`std::shared_ptr<void>` plus an `MLDataType`. For an in-tree opaque type, the shared
pointer can point directly to the concrete C++ object.

`OpaqueType<T, ...>` derives from `NonTensorType<T>`. Its type descriptor supplies:

- a factory that creates `T`; and
- a deleter that destroys `T`.

The generic execution framework invokes those callbacks without knowing the class's
representation. The standard template factory requires a default-constructible
class. Types needing constructor arguments or custom allocation require their own
datatype implementation; this support does not add a new public factory API.

When the execution frame sees a safe opaque reuse plan, it assigns the reused
`OrtValue` to the output. That shares the object and its ownership control block,
instead of copying only a raw address. The object is destroyed exactly once after
the last owning `OrtValue` is released.

When reuse is unavailable, the execution frame asks the datatype factory for a new
output object. The kernel must then populate it without changing any still-observable
input state.

The object's class remains responsible for its private resources. ORT does not
automatically pool internal buffers or infer deep-copy behavior.

Owning references in graph-internal opaque reuse slots are released after their
last use, including unused outputs. The original slot remains alive through the
last use of the reuse group, so dead objects can release their private resources
before later kernels execute rather than being retained until the run ends.

## Executable coverage

[`OpaqueReuseTest`](../onnxruntime/test/framework/opaque_reuse_test.cc) defines
test-only opaque PRNG schemas and CPU kernels. It covers:

- linear graph-internal reuse and shared ownership lifetime;
- forked state with correct independent logical sequences;
- preservation of observable graph inputs and outputs;
- rejection when `MayInplace` is absent;
- conservative parallel execution behavior;
- rejection of mismatched opaque domains and names;
- distinct allocation in the optimizer execution frame;
- same-object model input/output binding through preallocated outputs and I/O binding;
- pre-run rejection of non-linear boundary aliases, including executable `If`
  captures and repeated inputs;
- multi-stream rejection of internal reuse when a kernel reads the same opaque
  state in multiple input positions;
- destruction of dead reused objects before later kernels execute;
- pass-through outputs when multiple graph inputs share one caller object;
- preservation of pass-through outputs regardless of fetch allocation or ordering; and
- the distinct-output fallback at the model boundary.

The tests use counters for construction, destruction, copying, and actual address
reuse. Value equality alone would not prove that the intended alias path executed.

## Limitations

This support is intentionally narrow:

- Opaque types, schemas, and kernels still require in-tree registration.
- No plugin registration, unload, or public custom factory API is added.
- ORT does not clone, assign, serialize, or copy back arbitrary opaque state.
- Planner reuse is not guaranteed; kernels must support distinct outputs.
- Caller-bound same-object mutation is limited to a direct linear kernel relation.
- Branching, control-flow, cross-device, and longer-path identity guarantees are not
  provided by this contract.
- Aliasing mutable state introduces the usual synchronization responsibility for the
  caller; the same object must not participate in overlapping runs.

The graph contract remains functional in every case. Physical mutation is an
implementation choice used only when the runtime and caller have established that
the earlier logical state no longer needs to remain independently observable.
