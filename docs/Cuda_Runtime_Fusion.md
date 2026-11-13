# CUDA runtime fusion

These EP-owned optimizations use ORT's existing capability optimization callbacks
and the generic `Pattern` / `PatternRewriter` APIs for matching and graph rewriting.

## Op-owned source declarations v0

The internal companions `contrib_ops/cuda/fusion/scaled_silu_sources.cc`,
`causal_conv_with_state_sources.cc`, and `moe_sources.cc` own the three immutable executable definitions.
`Select` uses the lightweight `CudaSourceReplacementFactories()` collection,
materializing only the selected declaration after configuration, capability-consumer,
optimization-level, and graph-admission gates. `CudaSourceReplacements()` is a
derived eager catalog used for explicit inventory and declaration coverage checks. The source
body's Rule supplies the actual pattern, guard, opsets, and Replacement used
by discovery and acceptance. Offered snapshot predicates overlay the original
guards. Kernel lookup remains CUDA-local, and the common body adds no public
API, bridge, function-expansion behavior, or compiled-EP registration.

The typed profiles distinguish standalone Swish non-NaN bit identity/NaN
classification from complete causal finite-input analytical budgets and exact
state, and bounded FP32 MoE mathematical semantics with tolerance.
The causal companion explicitly records its owned flattened bias copy.
Neither executable metadata nor registered kernels alone prove unconditional
source equality. The single-target CUDA adapter declines multi-target
Replacement DAGs before mutation or lookup; four causal input ports, two
output ports, one target node, and GPU launch count remain separate concepts.

`ContribSourceReplacementPolicyTest.*` checks exact schema inventory coverage,
PendingTodo/Special explanations, conflicts/orphans, malformed executable
metadata, version exemptions, EP/variant-scoped Available, and multi-target
decline. Run it in a full contrib configuration, alongside the host and actual
CUDA numerical/routing tests below. Reduced inventories, skipped device tests,
or stale provider libraries do not establish full qualification. Logs and build/
provider provenance belong in validation artifacts, not inferred from this text.

### Adding a CUDA source declaration

The catalog is a metadata projection of **explicitly collected factories**,
not automatic enumeration of companion files. A new author must:

1. Add the op-associated companion factory and its declaration in
   `cuda_source_replacements.h`. Own the actual Rule consumed by discovery and
   acceptance, its exact target key, named/scoped numerical profile, and any
   typed initializer transformation. In v0 the variant identifier is the rule
   name; CUDA Available requires a parsed Replacement DAG, not a callback-only
   replacement.
2. Add the host companion TU to `onnxruntime_optimizer.cmake`. Keep it excluded
   from both normal and plugin CUDA-provider recursive source lists.
3. Add the factory to `CudaSourceReplacementFactories()`. Do not
   add a parallel support list or infer support from filenames/kernel coverage.
4. Integrate the rule ID with the explicit ordered rule loop in
   `CUDAExecutionProvider::GetCapability`
   (`core/providers/cuda/cuda_execution_provider.cc`), which calls
   `g_host->SelectCudaRuntimeFusion`. The host policy hook is
   `cuda_runtime_fusion::Select` (`core/optimizer/cuda_runtime_fusion.cc`),
   using the factory-owned config entry. A new ID needs explicit opt-in there;
   collecting metadata alone
   does not cause production `GetCapability` to select it. v0 does not change
   that production hook for future rules.
5. Add structural coverage and host discovery/acceptance tests, plus fresh
   CUDA routing and numerical qualification for the declared scope. Run
   `ContribSourceReplacementPolicyTest.FullRegisteredInventory` in a full-contrib
   configuration; reduced/minimal inventory is not complete coverage.

The immutable bodies are parsed lazily on first admitted selection of their
factory. Default-off selection does not materialize declarations or parse their
bodies. Explicit access to `CudaSourceReplacements()` materializes all declarations
from the same factory collection for inventory checks, without a separate registry.

## Unit-Swish kernel replacement

The internal session entry `ep.cuda.runtime_fusion.unit_swish` enables the
EP-owned replacement of standard unit-alpha Swish with an existing CUDA kernel.
It is off by default and accepts only the exact strings `"0"` and `"1"`.
The internal identifier remains `unit_swish_scaled_silu_v1`.

The replacement maps standard ONNX Swish-24 with absent or FLOAT alpha=1 to the
existing `com.microsoft.ScaledSiLU-1`, with alpha=1 and no scale input.
Input and output must be dense, matching FLOAT or BF16 tensors with compatible
shape and initializer metadata. FLOAT16, DOUBLE, mixed types, non-unit alpha,
and nonfinite alpha are not admitted. No casts, schema changes, or target
kernel changes are introduced.

### Discovery and acceptance

`cuda_runtime_fusion::Select` validates configuration and common admission gates,
then delegates to the ScaledSiLU companion's discovery callback. Discovery finds candidates;
the case-local `OfferSwish` constructs each capability and its deferred
optimization callback. Constructing an offer does not modify the source graph.
ORT invokes the callback only after checking that the selected region is
available for assignment.

Discovery requires a capability consumer that executes optimization callbacks,
optimization level 2 or higher, ONNX-24/Microsoft-1 imports, an unfiltered main
graph, normal CUDA layout, and no resource accountant. Minimal CUDA,
contrib-disabled, and plugin CUDA builds do not offer this optimization.
ORT-format loading does not execute these callbacks and opts out.

Discovery reads metadata only, including for initializer inputs: it does not
open external files or inspect tensor payloads. A scratch graph probes the
exact typed target kernel. A missing target leaves ordinary execution eligible.
Symbolic, anonymous, scalar, and empty shapes are eligible. Known element counts
must fit the target's 256-thread grid with a signed-int block count; unknown
extents remain subject to runtime kernel validation.

Acceptance revalidates the live source, assignment, attributes, NodeArg
identities, and owned type/shape snapshot. A precheck preserves the existing
stale-offer diagnostic instead of `RunOnce`'s generic no-match error.
An anchored, exact-region
`PatternRewriter::RunOnce` over the original single node must perform exactly
one rewrite to one target.
The original input and output NodeArgs, external fanout, and graph outputs
are preserved. Stale offers fail rather than silently selecting another source.
Changing an input's producer alone does not invalidate its NodeArg snapshot.

### Numerical scope and tests

At unit alpha, the source CUDA Sigmoid computes in FLOAT before converting to
the element type, and the element product converts back to that type. The
target also evaluates its stable sigmoid in FLOAT, with element conversions.
Non-unit alpha is not covered.

Focused source/fused/direct tests require identical non-NaN bits (including
signed zero) and matching NaN classifications, not matching NaN payloads.
They cover scalar, empty, odd, dynamic, and model-shaped inputs. FLOAT cases
use pinned special values and seeded random raw bits. BF16 coverage includes
all 65536 bit patterns in one rank-1 case; other shapes use pinned special
values and seeded random bits. Graph assertions and initialized CUDA kernel
bindings verify placement and the replacement; values alone are insufficient evidence.
These tests are not a guarantee for every input, shape, dtype, device, model,
or performance regime.

Run freshly built tests from the build directory:

```sh
./onnxruntime_test_all --gtest_filter='UnitSwishHostTest.*'
./onnxruntime_provider_test --gtest_filter='CudaRuntimeFusionTest.UnitSwish*'
```

Confirm nonzero counts and inspect skips. CUDA numerical tests require a
supported device; BF16 tests check device support before constructing test
fixtures. Qualification logs, model counts, and hardware timings belong in
validation artifacts rather than this contract.

## Stateful causal convolution

The internal session entry `ep.cuda.runtime_fusion.causal_conv_with_state`
enables stateful CausalConv fusion (internal identifier
`causal_conv_with_state_mobius_bf16_v1`). Despite its historical identifier,
the fusion supports matching FLOAT, FLOAT16, or BF16
regions. It is off by default and accepts only the exact strings `"0"` and
`"1"`. The discovery-consumer, optimization-level, import, layout, and build
restrictions described above also apply.

### Mathematical contract

The fusion replaces a channels-first causal depthwise convolution region:

- Concatenate history `[B,C,(K-1)*D]` with input `[B,C,S]` along axis 2.
- Apply an unpadded, stride-1 Conv with constant weight `[C,1,K]`,
  group `C`, positive dilation `D`, and a constant per-channel bias.
- Apply unit-alpha Swish and return its output and the final history tail.

Bias may be a Conv input, a broadcast `[1,C,1]` Add, or a `[C]` initializer
reshaped before Add. The history tail must use the recognized Shape/Gather/
Sub/Reshape/Slice construction with the correct axis and window. The
normalized empty tail is supported for `K=1`. Shared internal results that
would be removed cannot escape the matched region.

The existing target is `com.microsoft.CausalConvWithState-1`, with
`ndim=1`, `channels_last=0`, `state_window=0`, the proven dilation, and
`activation="silu"`. Its mathematical Conv/bias/SiLU and history semantics
must match the source. It does not reproduce every source floating-point
materialization boundary: accumulation order, intermediate rounding, and
overflow/NaN/Inf behavior may differ. In particular, a source Conv may
overflow its element type before Swish, whereas the target can retain a
finite FP32 accumulator through SiLU. Opting in accepts this numerical
contract, not changes to tensor layout, bias, axes, or state updates.

### Discovery and acceptance

Discovery checks eligible, unassigned Swish roots before invoking `Pattern::VisitMatches`.
Channels, kernel size, dilation, shapes, control constants, initializer
storage, and native index/byte bounds are checked from the matched region.
Small control constants must be embedded. Learned weight and bias payloads
are not read and external files are not opened during discovery. Unknown
runtime dimensions are not replaced with guessed workload sizes.

A scratch graph probes the exact typed native target. Missing kernels or
unsupported contracts leave ordinary execution eligible. Discovery does not
mutate the source graph. Acceptance revalidates the exact offered region and
contract, rewrites it once, preserves both original output NodeArgs, and
resolves the graph. A folded rank-3 bias receives an owned, flattened bitwise
copy without reshaping shared source storage. Stale acceptance or target
construction errors fail initialization rather than silently selecting another candidate.

Stateful CausalConv fusion is offered before the Swish kernel replacement when both flags are enabled.
Accepted causal regions own their contained Swish; other eligible Swish
nodes remain available to the independent Swish kernel replacement.

### Native resource behavior and validation

Native dimensions, index arithmetic, allocation sizes, and empty batches
are validated independently of either fusion flag. Prefill uses the existing
shared-memory kernels when their index/grid domains and device/function
shared-memory capacity permit. Legal extended shared memory is configured
to a fixed capacity after subtracting static shared memory, never to a
shape-dependent smaller limit. Function attributes are queried in the
current CUDA context; there is no process-global initialization cache.

During stream capture no shared-memory attribute is changed. An unwarmed
extended-memory case uses the existing direct kernel; already configured
cases can use the shared kernel. Channels-last layouts and shapes beyond
the shared kernel's capacity or index/grid domain also use the direct path.
That path can be slower for channels-first long sequences. Unexpected CUDA
query, configuration, and launch errors are propagated, not cleared and
retried as successful fallback.

Focused tests check exact history state, native dispatch boundaries,
source/fused/direct CUDA placement, and finite-input numerical errors against
an independent double-precision mathematical reference with dtype-aware
forward-error budgets. Qualification is scoped to the complete matched
region with identical inputs and history, including both outputs. Such local
bounds are not whole-model error bounds or a guarantee of long-generation
accuracy. Any separate model-level qualification requires independently
selected criteria, not thresholds fitted to observed mismatches. Finite
outputs or matching greedy tokens alone are insufficient.

```sh
./onnxruntime_test_all --gtest_filter='CUDARuntimeFusionHostTest.*:PatternRewriterTests.*:PatternComponentTests.*:PatternCapabilityTests.*'
./onnxruntime_provider_test --gtest_filter='CudaRuntimeFusionTest.*:CudaRuntimeFusionProbe.ZeroBatch:CausalConvWithStateTest.*-*LocalAuthentic*'
```

## Dense FP32 MoE

The internal entry `ep.cuda.runtime_fusion.moe` is default-off and accepts only
the exact strings `"0"` and `"1"`. Its rule is `moe_fp32_dense_swiglu_v1`,
targeting the existing `com.microsoft.MoE-1` CUDA kernel. The ordered capability
loop tries causal convolution, then unit Swish, then MoE, without changing
the build, layout, resource, consumer, level-2, main-graph or unfiltered-view gates.
MoE additionally requires actual device **SM80 or newer**: its FP32 GEMM
dispatcher has no usable pre-SM80 branch. The CUDA EP checks device properties
before asking the host adapter for MoE offers; registered-kernel availability
alone is insufficient. On pre-SM80 devices, a metadata-only host call still
validates the exact configuration strings without materializing sources or
probing target kernels. CPU, WebGPU and QMoE are not included.

The bounded source requires exactly ONNX-18/Microsoft-1 imports; other import
versions decline. It has fully resolved FP32
`x[N,H]`, raw `logits[N,E]`, immutable `W1[E,2I,H]` and `W2[E,H,I]`.
It selects raw-logit TopK, applies Softmax only over the selected logits,
uses chunked gate/up SwiGLU, and accumulates expert contributions in
increasing-expert, left-associated order. That source order is a matching
restriction, **not target bitwise parity**. E is 2 through 64, k is 1 through E,
N is positive and at most `65535 * 1024` (the existing routing fallback's
grid-Y bound), H and I are positive multiples of 16, and native integer
index bounds must hold. The target's float-sized workspace calculations for
`N*k*H*4` and `N*k*I*4` bytes must not round below the required integer sizes;
exact and upward-rounded sizes remain eligible. A single PatternLibrary declaration
has 63 expert-count alternatives and a guarded immutable INT64 `[1]` route-k input; it does not
construct every (E,k) declaration.
Selection uses one canonical collection of lightweight op-owned rule/config/factory
metadata. Executable declarations materialize only after configuration and
admission; explicitly requesting the full inventory materializes all declarations
from that same collection. Default-off or pilot-only selection does not parse MoE.

The replacement retains five input slots, with absent FC1 bias, and exactly
`activation_type="swiglu"`, `swiglu_fusion=2`, `activation_alpha=1`,
`activation_beta=0`, `normalize_routing_weights=1`, `use_sparse_mixer=0` and
the bound integer `k`. `swiglu_limit` is omitted: no explicit clamp is introduced.
The profile `cuda_moe_fp32_mathematical_v1` permits ordinary reassociation,
normalization, division and accumulation rounding differences while preserving
the schema/source mathematical semantics. It claims neither bitwise/stepwise
parity, a universal numerical error bound, nor exceptional-value parity.
Initializer transformation is `None`.

Discovery first requires an unassigned ONNX Add whose right input is produced
by an ONNX Mul, a necessary topology check, not a second matcher.
It batches eligible roots in one read-only existing-matcher scan,
preparing concrete plans and the producer index once; no prepared graph state
survives the call. Matches/offers retain only the selected alternative, not siblings.
Discovery probes the exact typed replacement through CUDA
kernel lookup. Missing kernels decline unchanged. Small controls, including
route k, are decoded with `ReadConstantTensor` (which can materialize external
storage). Malformed captured route-k storage returns the reader's original
Status. Malformed literal controls retain the matcher's existing throwing-decoder
boundary (`ORT_THROW_IF_ERROR`); no broad catch or silent default is introduced.
This is **not payload-free**
discovery. Learned bank payloads are never inspected or copied. Offers own
source identity, metadata, attributes, initializer headers and decoded-control
snapshots; acceptance rematches and rechecks them before one anchored
exact-region rewrite. Shared banks and original x/logits/w1/w2/y identities
are preserved. Stale offers fail before mutation.

Host tests cover catalog/admission, typed probing, discovery, stale offers,
near misses and exact-region acceptance. The CUDA test
`MoE/CudaMoERuntimeFusionTest.SourceFusedDirectTF32OffAndOn/*` compares the original
ONNX graph with fusion enabled under identical inputs and TF32 disabled,
and independently checks fused output against the directly authored MoE target
with TF32 both disabled and enabled,
including ties, signed gates, cancellation, decode/prefill and E2/8/16/64.
Its 96 individually filterable/shardable cases retain all three k choices,
including the duplicate k=2 choices for E=2, with a k-choice index in each name.
They disable ConstantFolding and TransposeOptimizer to retain this precise source topology through
GetCapability; prior optimizations that erase it are not matched.
Applications that need to preserve the immutable-bank Gather/Squeeze/Transpose
topology must likewise disable those optimizers before session initialization.
The MoE flag alone under ordinary Level2 optimizations cannot guarantee that this
topology survives until GetCapability. For example, with the Python session API:

```python
import onnxruntime

options = onnxruntime.SessionOptions()
options.add_session_config_entry("ep.cuda.runtime_fusion.moe", "1")
options.add_session_config_entry(
    "optimization.disable_specified_optimizers", "ConstantFolding;TransposeOptimizer"
)
session = onnxruntime.InferenceSession(
    model_path, sess_options=options, providers=["CUDAExecutionProvider"]
)
```

`optimization.disable_specified_optimizers` is the existing session configuration
key (`kOrtSessionOptionsDisableSpecifiedOptimizers` in C++), with optimizer names
separated by semicolons as parsed by `InferenceSession`; no production
optimizer behavior is changed by the MoE flag.

Source/fused fixture bounds with TF32 disabled are atol=1e-5, rtol=2e-4.
Fused/direct consistency retains atol=1e-5, rtol=2e-4 in both modes.
With TF32 enabled, the original source still executes and retains placement,
shape and finite-output checks, but source/fused numerical differences are not
an acceptance gate: source MatMul may use TF32 while the existing FP32 MoE uses
SIMT. That cross-kernel precision comparison belongs to kernel qualification,
not rewrite integration.
These are local graph-equivalence checks, not universal kernel-accuracy contracts.
An independent high-precision reference is not an acceptance gate for the existing
source or MoE kernel; kernel-accuracy qualification belongs in operator tests.
CUDA execution/qualification requires a fresh consistent CUDA build and supported
device execution with recorded placement, results and resource conditions.
Catalog `Available`, passing host tests, and prior runs without complete
resource/placement evidence do not establish GPU qualification for this integration.
The numerical test skips unsupported direct-target modes below SM80; a separate
actual-pre-SM80 fixture checks decline before target lookup. That architecture
coverage requires real older hardware, not an architecture override.

```sh
./onnxruntime_test_all --gtest_filter='MoE*:*SwishHostTest.*:CUDARuntimeFusionHostTest.*:ContribSourceReplacementPolicyTest.*:PatternRewriterTests.*:PatternComponentTests.*:PatternCapabilityTests.*'
./onnxruntime_provider_test --gtest_filter='CudaRuntimeFusionTest.MoE*'
```
