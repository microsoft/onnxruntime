# Extensible Runtime Fusion for Canonical ONNX Models

Status: Proposal  

## Executive summary

ONNX Runtime's current approach to high-performance generative-model execution often replaces a
standard ONNX subgraph with a provider-influenced contrib operator before runtime. The
`com.microsoft.GroupQueryAttention` path is a representative example. This approach delivers
performance for the CUDA Execution Provider (EP), but it also moves the model's
portability boundary from ONNX semantics to an CUDA EP-specific operator contract.

That tradeoff is becoming increasingly expensive:

- New model architectures are arriving faster than coordinated updates can be made across Mobius,
  Olive, ONNX Runtime, EPs, and customer applications.
- A contrib operator designed around one implementation becomes a contract that every other EP must
  adopt, decompose, or reject.
- Compiler EPs lose visibility into the original graph and therefore lose optimization freedom.
- Quantization choices become constrained by the tensor shapes, layouts, and data types
  accepted by composite operators.
- Customers receive target-specific model variants instead of the canonical ONNX model they value.
- A hardware vendor or customer cannot introduce a new fused implementation without changes to ONNX
  Runtime.

This proposal establishes a different north star:

> Mobius produces a stable, canonical ONNX graph.
> Olive quantizes that graph aggressively without being constrained by fused-operator contracts.
> Compiler EPs compile the graph directly. Separately shipped fused-kernel extensions advertise the
> source patterns they can replace. ONNX Runtime provides generic discovery, partitioning,
> validation, and safe graph-rewrite infrastructure, but contains no per-fused-op pattern knowledge.

The desired outcome is a durable separation of concerns:

- **Model semantics belong in canonical ONNX.**
- **Quantization policy belongs in Olive.**
- **Hardware-specific pattern recognition and performance optimization belong with the
  implementation that benefits from them.**
- **ONNX Runtime provides the stable mechanism, not an ever-growing catalog of model-specific
  fusions.**

This is an additive transition. Existing contrib operators and models remain supported. The change
is to make canonical ONNX the preferred interchange and deployment boundary, and to make
provider-owned runtime fusion the preferred path for new implementations.

This proposal also builds on work already underway in ONNX Runtime. The CUDA team has added support
for standard ONNX `Attention`, `RotaryEmbedding`, and cache-update operations, demonstrating that
canonical ONNX and high-performance CUDA execution are compatible goals. The remaining gap is not
whether CUDA EP can execute standard operators. It is whether new kernel-based fusions, quantization
formats, and vendor-specific optimizations can be introduced independently, without adding each
source pattern to ONNX Runtime or baking a target-specific node into the model.

## 1. Motivation

### 1.1 Model development is outpacing the software release train

The generative-model ecosystem changes relentlessly. New architectures introduce variations in:

- attention and positional encoding;
- mixture-of-experts routing;
- state-space and convolutional sequence layers;
- sparsity and quantization formats.

Today, enabling a new model at high performance can require coordinated work across several
packages:

1. Mobius learns how to export the architecture.
2. Olive learns how to recognize, transform, and quantize it.
3. ONNX Runtime adds or modifies a contrib operator, schema, or graph transformer.
4. One or more EPs add kernels or compiler support.
5. Customers update model artifacts and multiple software packages together.

Each additional coordinated release increases lead time and creates compatibility matrices between
model versions, transformation tools, runtimes, EPs, and fused-op contracts. The ecosystem needs an
architecture in which a new model can run correctly from its canonical ONNX representation first,
and independent parties can add performance without waiting for the entire stack to move in
lockstep.

### 1.2 Hardware vendors need an independent path to performance

Every hardware vendor has different preferred:

- tensor layouts;
- fusion boundaries;
- cache representations;
- memory-planning strategies;
- compiler intermediate representations;
- kernel libraries and launch strategies.

A high-level contrib operator can simplify one implementation while constraining another. Hardware
vendors should be able to recognize and optimize the canonical graph according to their own
hardware without first adopting a contract shaped by another provider.

Vendors also need to ship support on their own schedule. A new fused implementation should not
require an ONNX Runtime source change or a new model export.

### 1.3 Customers value the canonical ONNX model

Customers want a model artifact that:

- preserves the model's standardized semantics;
- can be inspected, validated, and transformed with ONNX tooling;
- can be moved between hardware vendors;
- remains correct when a preferred fusion is unavailable;
- does not encode deployment decisions for one provider;
- does not require maintaining a collection of nearly equivalent provider-specific variants.

Provider-specific optimized artifacts can still be valuable as caches or deployment derivatives,
but they should not replace the canonical model as the source of truth.

## 2. Current approach

A common current flow is:

```text
model architecture
        |
        v
Mobius / exporter
        |
        v
standard or semi-standard attention graph
        |
        v
Olive / offline graph surgery
        |
        v
com.microsoft.GroupQueryAttention and related packed forms
        |
        v
ONNX Runtime provider-aware graph transformations
        |
        v
EP kernel
```

For example, an offline transform may replace attention and cache-management nodes with
`com.microsoft.GroupQueryAttention`. A later ONNX Runtime optimization can merge Q, K, and V
projections and absorb rotary embedding into the GQA contract for CUDA.

This approach has real advantages:

- It provides a high-level semantic node that is easier to optimize than a variable primitive graph.
- It allows rapid delivery of a CUDA kernel without first designing a general extension mechanism.
- It can reduce runtime graph-matching complexity.
- It creates a stable target for in-tree kernel tests.
- It can encode generation-specific behavior that was not yet available in standard ONNX.

Those benefits explain why the approach was reasonable. The issue is not that contrib
operators should never exist. The issue is using a provider-influenced contrib operator as the
default portability boundary for an ecosystem with many EPs, compilers, quantization strategies, and
independently shipped implementations.

### 2.1 Progress already aligned with this direction

ONNX Runtime is not starting from zero. Recent work has established important foundations:

- compiler EPs already use `GetCapability` and `Compile` to own pattern recognition and lowering;
- plugin EP and custom-op APIs already allow implementations to ship outside the core runtime.

This proposal does not replace those investments. It extends their architectural direction to the
remaining kernel-fusion gap: allowing an independently shipped fusion implementation to recognize and
replace a source region.

## 3. Disadvantages of the current approach

### 3.1 The canonical model becomes provider-influenced

Replacing a standard ONNX subgraph with an ONNX Runtime contrib operator makes the resulting model
dependent on an ONNX Runtime-specific contract. Other runtimes and tools must implement that
contract or cannot consume the model.

Even within ONNX Runtime, every EP that wants to execute the node must understand the contrib
operator's:

- input and output ordering;
- optional-input conventions;
- attributes;
- layout requirements;
- type constraints;
- packed and unpacked forms;
- aliasing and buffer-sharing expectations.

The model is no longer merely expressing attention. It is expressing the particular abstraction
chosen by the implementation that introduced the contrib operator.

### 3.2 Provider implementation choices become ecosystem contracts

An implementation detail can become permanent once it is exposed through a serialized operator
schema. Changing them later requires schema growth, compatibility logic, graph adapters, or new operators. 
The contract tends to accumulate optional inputs and attributes because existing serialized models 
must continue to work.

### 3.3 Other EPs must follow rather than innovate independently

Once the contrib operator GQA is used in the graph, another EP cannot inspect the original attention, cache,
projection, or positional-encoding structure. It must:

1. implement the contrib operator;
2. decompose it back into a graph;
3. compile it as an opaque custom operator; or
4. decline it and allow fallback.

This reverses the desirable relationship. Hardware vendors should choose the fusion and layout best
suited to their hardware. They should not have to reconstruct standard semantics from one
provider-influenced composite contract before optimizing them.

### 3.4 Compiler EPs lose optimization visibility

Compiler EPs are designed to consume subgraphs and perform their own:

- pattern recognition;
- operator fusion;
- layout selection;
- constant folding;
- memory planning;
- scheduling.

Replacing a rich standard subgraph with a contrib node hides information the compiler could have
used. The compiler may understand the primitive attention graph better than the contrib contract, or
may prefer a different fusion boundary that spans attention, normalization, residual operations, and
the following projection.

An early composite rewrite can therefore reduce, rather than improve, compiler performance.

### 3.5 Quantization becomes constrained by composite-operator support

Olive should be free to choose quantization technqiues based on accuracy, hardware capability, 
and model characteristics.

A composite operator constrains that freedom when its schema or kernels accept only selected:

- tensor ranks and shapes;
- scale layouts;
- zero-point layouts;
- data-type combinations;
- packed-weight formats;
- intermediate quantization schemes.

This can force one of three undesirable outcomes:

- Olive chooses a less effective quantization because the fused operator requires it.
- Olive performs the desired quantization but loses the fusion.
- The fused-operator contract expands for every new quantization combination.

The number of combinations grows rapidly when Q, K, V, activations, and KV caches can each use
different types, granularities, and layouts. Encoding that matrix into composite schemas and
centrally maintained transformations does not scale.

### 3.6 New model enablement requires coordinated package updates

When model structure, contrib contracts, offline transforms, and runtime fusions are coupled, a new
architecture may require updates to Mobius, Olive, ONNX Runtime, and one or more EPs.

Consequences include:

- longer time to first correct execution;
- longer time to optimized execution;
- synchronized release dependencies;
- difficult version-support matrices;
- customer pressure to update the entire stack;

The slowest package or team becomes the release bottleneck.

### 3.7 Fused implementations cannot be shipped independently

Today, introducing a new kernel that recognizes a new source pattern generally requires one or more
of:

- adding matcher logic to ONNX Runtime;
- adding a graph transformer to ONNX Runtime;
- modifying an EP's `GetCapability`;
- adding a new contrib operator or extending an existing schema;
- modifying Mobius to produce a compatible graph.

This prevents a customer or hardware vendor from shipping a fused implementation as an independent
extension against an existing ONNX Runtime and EP.

### 3.8 Pattern ownership is misplaced

The party implementing a fused kernel knows:

- the exact source forms it can replace;
- its hardware restrictions;
- its supported quantization layouts;
- its alignment and shape constraints;
- the numerical assumptions under which the fusion is valid.

When the pattern is maintained in ONNX Runtime or an offline model tool instead, the matcher and
implementation can drift. Updating the kernel may require synchronized changes in a different
repository owned by a different team.

### 3.9 Provider-specific model variants proliferate

When fusion is baked into the model, customers may need separate artifacts for:

- CUDA;
- WebGPU;
- OpenVINO
- QNN or other NPUs;
- CPU fallback;
- different hardware generations.

Variant proliferation increases storage, validation, deployment, rollback, and support costs. It
also obscures which artifact is the authoritative model.

### 3.10 Fallback becomes coarse or unavailable

With a primitive standard graph, an unsupported fusion can simply remain unfused. Individual
operators can still be partitioned and executed.

With a contrib composite node, failure to support one feature combination can cause:

- fallback of the whole composite to CPU;
- session initialization failure when fallback is disabled;
- a need to regenerate the model without the contrib operator.

The cliff is much larger than missing an optional fusion.

### 3.11 Serialized optimized models become less portable

Saving a graph after provider-specific fusion creates an artifact tied to:

- a provider;
- an operator schema version;
- a kernel feature set;
- layout and quantization assumptions;
- potentially a hardware generation.

Such artifacts are useful as deployment caches, but they are poor replacements for the canonical
model. Treating them as interchangeable with canonical ONNX creates upgrade and rollback risk.

### 3.12 Schema evolution carries long-term compatibility cost

Contrib schemas are easier to introduce than standard ONNX operators, but once customers serialize
them they become compatibility commitments. New model features tend to produce:

- more optional inputs;
- more attributes;
- interactions between old and new modes;
- larger shape-inference surfaces;
- larger test matrices.

A source pattern advertised by a separately shipped implementation can evolve with that
implementation without turning every implementation detail into a shared serialized contract.

### 3.13 Optimization delivery depends on proximity to the core runtime

An implementation maintained alongside ONNX Runtime can coordinate changes across contrib schemas,
graph transformers, and kernels more easily than an external implementation. That difference raises
the cost of first-class support for independently maintained EPs, kernel libraries, and customer
extensions.

## 4. Costs and tradeoffs of the proposed direction

The proposed architecture is not free. Its costs should be accepted explicitly rather than hidden
behind the portability benefits.

### 4.1 A topology contract replaces part of the composite-op contract

A fused-op schema is explicit, versioned, and validated. A source pattern depends on graph topology,
which can be changed by Mobius, Olive, or ONNX Runtime optimizations. The proposal therefore trades
some serialized operator coupling for topology coupling.

That trade is worthwhile only if the topology contract is managed deliberately:

- canonical graph forms used for fusion have published golden models;
- Mobius and Olive CI detect unintended topology drift;
- intentional form changes are versioned or otherwise discoverable;
- the runtime defines which normalization and optimization steps occur before extension matching;
- matching failure always preserves a valid graph;
- reusable matching utilities absorb semantically irrelevant differences where practical.

The topology contract is expected to be cheaper because it is not mandatory for correctness and can
evolve with the consuming implementation. A contrib schema, by contrast, becomes a shared serialized
compatibility commitment for every implementation and customer model.

### 4.2 Unfused execution may be functionally correct but operationally inadequate

For autoregressive decoding, a naive primitive cache graph can copy growing K/V state on every token,
causing unacceptable time and memory growth. Canonical ONNX must therefore express efficient bounded
cache update and aliasing semantics using standard operations where available, rather than treating
an asymptotically poor graph as a sufficient fallback.

The baseline must be measured on real prompt and decode workloads. Correctness alone is not enough to
change the default workflow. The proposal requires a viable unfused or compiler-compiled path, while
recognizing that optional fusion remains essential for target performance.

### 4.3 Runtime matching adds initialization work

Pattern discovery over large graphs can increase session initialization and first-token latency.
Matching must be bounded, indexed, diagnosable, and compatible with caching. The implementation plan
must establish an explicit initialization-overhead budget.

### 4.4 Extension compatibility and testing become ecosystem responsibilities

Independently shipped fusion extensions must be tested against supported ONNX Runtime versions, EP versions,
devices, and model forms. Some work moves out of the ONNX Runtime repository rather than
disappearing. The extension contract needs clear versioning and conformance expectations.

### 4.5 Dynamic extension loading is not universal

Mobile, web, sandboxed, and reduced builds may not permit native dynamic libraries. Those
environments need static registration or build-time inclusion using the same logical contract. The
architecture must not equate extensibility exclusively with dynamic loading.

### 4.6 Migration creates temporary dual maintenance

Existing GQA models, offline transforms, and runtime fusions must remain supported while the new path
is proven. The planning phase must define ownership, expected duration, and sunset criteria for
duplicated paths so an additive transition does not become permanent duplication.

## 5. North star

### 5.1 Canonical ONNX is the source of truth

Mobius produces a deterministic model expressed in standard ONNX operators and functions. Lowering to a 
stable primitive graph is optional.

This creates a deliberate layering rule:

- retain a standard semantic operator;
- lower the operator when important internal tensors require independent quantization or when the
  standard composite operatror cannot represent the model semantics;
- keep the result standard ONNX in either case.

The canonical model:

- contains the complete model semantics;
- is not specialized for an EP;
- remains executable without optional fusions;
- is the artifact customers store, exchange, inspect, and validate;

Provider-specific optimized models remain optional derivatives or caches.

### 5.2 Quantization is independent of fused-kernel contracts

Olive sees the tensors and operations it needs to quantize. It chooses data types, granularities,
block sizes, scale layouts, and cache representations without being limited by the currently
installed fused implementations.

Fusion is opportunistic:

```text
Olive-selected quantized graph
            |
            +--> matching fused implementation available: fuse
            |
            +--> no matching implementation: execute or compile the valid primitive graph
```

The absence of a fusion must not invalidate the model or force Olive to choose a weaker
quantization.

Olive may also produce an explicitly requested, reproducible target-aware derivative that chooses a
quantization favored by an EP. That is a deployment optimization, not a requirement imposed on 
the canonical model.

### 5.3 Performance implementations own their patterns

A fused implementation ships with the knowledge required to use it:

- target operator schema or compiled implementation;
- kernel or backend code;
- supported source patterns;
- applicability constraints;
- rewrite or compilation metadata;
- diagnostics.

The implementation package can be supplied by:

- an ONNX Runtime team;
- a hardware vendor;
- a kernel-library vendor;
- an enterprise customer;
- an independent optimization project.

### 5.4 ONNX Runtime supplies a generic mechanism

ONNX Runtime is updated once to support an extensible fusion mechanism. It is responsible for:

- loading and registering fusion extensions;
- exposing a stable read-only graph-inspection surface;
- collecting candidate fusions;
- coordinating candidates with EP partitioning;
- applying accepted rewrites safely;
- preserving an unfused fallback;
- reporting fusion decisions.

ONNX Runtime does not know that a particular pattern is GQA, rotary embedding, mixture of experts,
or a model-specific block.

### 5.5 Compiler EPs keep pattern handling in their compilers

Compiler-based EPs continue to claim standard subgraphs and perform pattern recognition, fusion,
layout selection, and lowering internally. They are not required to use a runtime fusion-extension
mechanism.

### 5.6 New models require the fewest possible coordinated updates

The ideal enablement path is:

1. Mobius represents the new model in canonical ONNX.
2. Existing ONNX Runtime and EPs execute the graph using existing standard operators where possible.
3. Olive quantizes it using general tensor-level machinery.
4. Hardware vendors independently update their compilers or ship fusion extensions for performance.

Not every new architecture will require zero changes everywhere. The goal is to avoid making
cross-package changes the default. A package should change only when it owns a genuinely new
semantic or performance requirement.

### 5.7 What the CUDA EP gains

The proposed direction is not a request for CUDA to give up performance or delivery velocity. It
offers CUDA:

- fewer contrib schema modes to version and support;
- the ability to accept third-party CUDA kernels without adding each matcher to the CUDA EP;
- clearer separation between stable CUDA execution support and optional model-specific fusions;
- canonical graphs that can be debugged before and after CUDA-specific optimization;
- a path for new quantization formats to prove themselves without first expanding a shared contrib Op
  contract.

CUDA EP's GQA is proposed as the first proof because it is mature, performance-sensitive, and well
understood — not because CUDA should bear permanent special-case complexity.

## 6. Responsibility boundaries

### Mobius

- Export new architectures into canonical ONNX.
- Produce deterministic, documented graph forms where stability benefits downstream tooling.
- Lower composite operators before Olive when requested.
- Avoid provider-specific fused operators in the canonical artifact.
- Version intentional changes to any graph-shape conventions relied on by downstream tools.

### Olive

- Quantize aggressively at tensor and operator granularity.
- Preserve model semantics and canonical ONNX validity.
- Avoid making portable quantization conditional on a particular fused implementation.
- Produce EP specific derivatives only when explicitly requested.

### ONNX Runtime

- Execute and partition the canonical graph correctly.
- Provide the generic fusion-extension mechanism.
- Validate extensions and preserve graph correctness.
- Keep fallback behavior explicit and observable.
- Avoid adding implementation-specific source patterns to core for new fusions.

### Fused operators implementation package

- Register its executable implementation.
- Advertise the source patterns it supports.
- Own hardware, shape, layout, quantization, and numerical constraints.
- Supply actionable match and rejection diagnostics.
- Evolve on its own release cadence within the stable extension contract.

### Execution Provider

- Expose standard kernel or compilation capability.

### Customer application

- Load the canonical ONNX model.
- Select EPs and optional optimization extensions.
- Optionally cache provider-specific compiled or optimized derivatives.
- Retain the canonical model for portability, fallback, and future optimization.

## 7. Design principles

The implementation design should follow these principles:

1. **Pattern knowledge ships with the implementation that consumes the pattern.**
2. **Accepted rewrites are validated and applied safely.**
3. **Diagnostics explain both successful and rejected candidates.**
4. **Extension loading is explicit and governed by the application's trust policy.**

## 8. Scope and non-goals

### Fusion boundary assumption

This proposal begins after a valid canonical ONNX graph exists. The graph already defines the
model's mathematical semantics, externally visible inputs and outputs, and any state passed between
invocations.

The proposal asks only:

> Given that canonical graph, how can an EP compiler or independently shipped kernel implementation
> recognize a supported region and replace it with a faster implementation without requiring the
> model to contain that implementation's fused operator?

For example:

- an MoE implementation may fuse routing, token permutation, grouped expert GEMMs, activation, and
  mixing as one region or as several independently profitable regions;
- a compressed-attention implementation may fuse recurrent, sparse-selection, or attention regions
  whose semantics are already present in the model.

The mechanism does not require every implementation to fuse the same boundary. A compiler EP may
claim a whole layer, while a kernel-based implementation may advertise several smaller,
composable patterns.

### In scope

- Enabling independently shipped, provider-targeted fusion implementations.
- Integrating optional fusion selection with runtime partitioning.
- Maintaining existing model and contrib-op compatibility during transition.

### Non-goals

- Requiring all EPs to use one fusion mechanism.
- Dictating kernel algorithms or fusion boundaries to hardware vendors.
- Selecting a detailed callback ABI or pattern language in this proposal.

## 9. Proposed plan

The plan deliberately defines outcomes and boundaries while leaving detailed API and implementation
choices to the engineering design process.

### Phase 1: align on the portability boundary

- Agree that canonical ONNX, not a provider-influenced contrib graph, is the source of truth for new
  model enablement.
- Document when Mobius should retain a standard composite operator and when it should lower to
  primitives before Olive.
- Define the minimum stability expectations for Mobius graph forms consumed by quantization and
  optional fusion.
- Define canonical versus provider-specific derivative artifacts in Olive workflows.
- Establish baseline metrics for model-enable time, package-update count, portability, fallback, and
  performance.
- Assign an owning team, rough effort range, dependencies, and decision point to each later phase.
- Define a stop criterion if the prototype cannot meet performance, initialization, or maintenance
  goals.

**Exit criterion:** stakeholders agree on responsibilities and on a canonical model artifact that
does not require a target-specific contrib operator.

### Phase 2: prove canonical export and unconstrained quantization

- Select representative models covering GQA, rotary embedding, KV caching, sliding-window
  attention, MoE, paged attention, compressed or sparse attention, and several quantization
  formats.
- Export deterministic canonical ONNX from Mobius.
- Quantize those graphs in Olive without requiring a GQA or other fused composite contract.
- Validate correctness on unfused execution.
- Measure prompt throughput, decode throughput, peak memory, and cache-update complexity against the
  existing GQA path.
- Identify any missing standard semantics or runtime kernels that prevent correct fallback.

**Exit criterion:** canonical quantized models execute correctly without an offline contrib-op
replacement.

### Phase 3: define the extensible fusion contract

- Design a stable registration and discovery contract for independently shipped fusion
  implementations.
- Define how candidates interact with EP ordering and partitioning.
- Define extension-author responsibility for semantic-equivalence testing.
- Define deterministic conflict resolution and diagnostics.
- Define versioning, lifetime, security, and compatibility expectations for extensions.
- Ensure the contract supports both in-tree and customer-provided implementations.
- Prototype the interaction between fusion candidates, EP priority, and partitioning before
  committing to a public extension surface.

**Exit criterion:** an approved architecture allows a fusion package to be introduced without
modifying ONNX Runtime, Mobius, Olive, or the target EP after the generic mechanism exists.

### Phase 4: demonstrate with CUDA GQA

- Adapt the existing CUDA GQA optimization as the first implementation using the generic contract.
- Match the canonical graph rather than requiring an offline GQA node.
- Preserve existing GQA models and kernels for compatibility.
- Compare performance, model coverage, diagnostics, and maintenance cost with the current path.
- Establish explicit acceptance budgets for throughput, peak memory, session initialization, and
  first-token latency on an agreed model and GPU matrix.
- Use the exercise to improve the generic contract rather than adding GQA-specific behavior to it.

**Exit criterion:** CUDA achieves equivalent target performance from the canonical graph without a
GQA-specific matcher in ONNX Runtime core or a required Olive GQA surgery.

### Phase 5: validate fusion breadth and vendor independence

- Partner with at least one external or separately maintained EP.
- Have that vendor compile the canonical graph or ship an independently registered fusion.
- Demonstrate a different fusion boundary, layout, or quantization support from CUDA.
- Demonstrate that the mechanism can represent both a large, variable region such as MoE and a
  state-carrying attention region such as paged or compressed attention, without adding
  workload-specific logic to ONNX Runtime core.
- Demonstrate partial fusion: an implementation may accelerate a supported subregion while the
  remaining canonical graph executes normally.
- Demonstrate that the vendor can update model coverage without coordinated ONNX Runtime, Mobius, or
  Olive releases.
- Build a customer-provided sample fused implementation against a released ONNX Runtime.

**Exit criterion:** a non-CUDA implementation reaches optimized execution on its own release cadence.

### Phase 6: transition the default workflow

- Make canonical export and quantization the recommended path for new models.
- Treat offline contrib-op conversion as an explicit target-specific optimization.
- Keep compatibility support for existing contrib models.
- Publish guidance for canonical artifacts, optional extensions, compiler EPs, and optimized caches.
- Track remaining centrally maintained fusions and migrate them when the benefit justifies the work.

**Exit criterion:** adding a new model or fused implementation no longer assumes coordinated changes
across the full stack.

## 10. Success criteria

The proposal is successful when the following are demonstrably true:

### Ecosystem velocity

- A new model can reach correct ONNX Runtime execution with changes primarily in the model exporter.
- Most model variations do not require simultaneous Mobius, Olive, ONNX Runtime, and EP releases.
- Time from public model introduction to hardware-vendor support decreases.

### Customer experience

- Customers retain one canonical ONNX source model across hardware targets.
- EP specific derivatives are optional and reproducible.
- Missing optional fusions preserve correctness and provide clear diagnostics.
- Updating an optimization package does not require regenerating the canonical model.

### Vendor autonomy

- A hardware vendor can add model-specific performance without changes elsewhere.
- A kernel vendor or customer can ship a new fused implementation against an existing runtime and
  EP.
- Vendors can choose their own fusion boundaries, layouts, and supported quantization formats.

### Runtime maintainability

- ONNX Runtime core contains generic fusion infrastructure rather than per-implementation patterns.
- Fusion decisions are deterministic, observable, and testable.

## 11. Risks and mitigations

### Risk: primitive graphs are more difficult to match

**Mitigation:** Mobius produces deterministic graph forms and versions intentional changes.
Implementations own their matchers and can update them independently. Matching failure preserves
correct execution. Golden canonical models and CI detect unintended topology drift.

### Risk: primitive graphs increase runtime initialization cost

**Mitigation:** cache matching results or provider-specific compiled artifacts without replacing the
canonical model. Measure initialization explicitly and optimize the generic mechanism.

### Risk: an extension ecosystem fragments pattern definitions

**Mitigation:** standardize only the registration, safety, and diagnostic contracts. Allow patterns
to differ because implementation support legitimately differs. Publish reusable matching utilities
without centralizing implementation policy.

### Risk: independently loaded extensions increase the trust surface

**Mitigation:** make loading explicit, apply normal native-code trust and signing policies, expose
extension identity in diagnostics, and ensure graph rewrites are validated by ONNX Runtime.

### Risk: fusion competition makes partitioning less predictable

**Mitigation:** define deterministic precedence, overlap, and cost rules. Provide tooling that
explains which candidates were considered and why one won.

### Risk: unfused fallback is initially slower or incomplete

**Mitigation:** correctness coverage for canonical standard operators is a prerequisite for changing
the default workflow. Performance remains supplied by compilers and optional fusion extensions.

### Risk: short-term duplication during migration

**Mitigation:** retain current GQA paths while proving the new mechanism. Migrate only after
correctness and performance are demonstrated. Set an explicit end state so temporary duplication
does not become permanent.

## 12. Alternatives considered

### Continue adding centrally maintained contrib fusions

This preserves the current delivery model but does not address package coordination, vendor
autonomy, quantization freedom, model portability, or customer-provided implementations.

### Standardize every fused operator in ONNX

Standardization is valuable for stable semantic abstractions, but it cannot keep pace with every
hardware-specific fusion, quantization combination, or model variation. It also risks turning
implementation choices into permanent cross-vendor contracts.

### Require every EP to decompose contrib operators

This restores some visibility but adds work to every EP and may not perfectly recover information
removed by the original rewrite. It is less efficient than preserving canonical semantics in the
first place.