# Stateful Session Execution for ONNX Runtime

**Status:** Specification and separate execution plan approved by the user.
M01-M14 are published for review; M15 implementation/qualification is complete
and stacked publication is in progress. Later milestones require their own
approved refinements.
**Requirements captured:** 2026-10-15.

**M10 refinement:** Appendix J was approved by the user on 2026-10-19.
The detailed `TASKS.md` plan was separately approved on 2026-10-19.
The save-as scope amendment in J.9 and its revised implementation plan were
separately approved on 2026-10-19.
**M11 refinement:** Appendix K was approved on 2026-10-20 for detailed plan
preparation. The CPU-first `TASKS.md` plan was separately approved on 2026-10-20;
GPU qualification was resumed and verified on 2026-10-21 with compatible tooling.
M11 is published for review in [#25](https://github.com/onnxruntime/ort-state/pull/25),
stacked on M10.
**M12 refinement:** The requirements interview completed on 2026-10-30.
Appendix L was approved by the user for detailed plan preparation.
The detailed M12 plan was separately approved on 2026-10-30 and implementation
and CPU/CUDA qualification are verified on 2026-10-31. M12 is published for review
in [#26](https://github.com/onnxruntime/ort-state/pull/26), stacked on M11.
M16 and later implementation milestones are not started.
**M13 refinement:** The requirements interview completed on 2026-11-02.
Appendix M was approved for detailed plan preparation. The requested stack
parent is verified M12. The detailed plan was separately approved on 2026-11-02;
implementation and qualification are complete, published as ready-for-review
[#27](https://github.com/onnxruntime/ort-state/pull/27).
**M14 refinement:** The requirements interview completed on 2026-11-02.
The user explicitly confirmed no additional requirements and approved Appendix N
on 2026-11-02 for detailed execution-plan preparation. The detailed plan was
separately approved on 2026-11-02. Implementation and CPU/CUDA qualification are
verified; M14 is published for review as
[#28](https://github.com/onnxruntime/ort-state/pull/28), stacked on M13.
**M15 refinement:** The requirements interview completed on 2026-11-03.
The user explicitly confirmed no additional requirements and selected CUDA-only
real inference with CPU EP fixture tests retained. Appendix O was approved on
2026-11-03 for detailed execution-plan preparation. The detailed plan was
separately approved on 2026-11-03; M15 implementation is starting.

## 1. Purpose

Introduce a production-oriented foundation for stateful model execution above
ONNX Runtime's existing Session abstraction. Applications must be able to run a
Session repeatedly, connect Sessions and loops into a chain, manage KV caches,
and sample tokens without implementing the orchestration themselves.

Ordinary Session behavior remains unchanged. The new layer owns execution state;
it does not make every Session implicitly stateful.

Extend Model Package to describe component chains and their declarative
connections, then demonstrate package-driven decoder-only language generation
and Stable Diffusion.

This document specifies required behavior and acceptance criteria, not an
execution plan. Prepare `TASKS.md` only after the user approves this specification.

## 2. Scope

### 2.1 Required capabilities

- **SCOPE-1:** Provide SessionLoop, SessionChain, KVCacheManager, and Sampler.
- **SCOPE-2:** Support programmatic construction and Model Package loading through
  every repository language binding that supports Session.
- **SCOPE-3:** Provide CPU implementations first, followed by CUDA KV management
  and device-native greedy sampling. Allow plugin EP implementations.
- **SCOPE-4:** Support sequential, state-retaining execution, explicit reset,
  synchronous stepping, run-until-exit, and synchronous cancellation.
- **SCOPE-5:** Demonstrate both requested pipelines from packages built using
  caller-supplied local models and text-processing integrations.
- **SCOPE-6:** Deliver independently documented and tested milestones as stacked
  PRs, following separate specification and execution-plan approval gates.

### 2.2 Deferred capabilities

The foundation should not preclude these features, but this delivery does not
implement them:

- Concurrent or pipelined chain execution.
- Asynchronous token streaming and serving-oriented request scheduling/batching.
- Speculative decoding, beam search, and sampling beyond greedy.
- Chunked prefill.
- Cross-request shared-prefix reuse or coordinated mutable cache sharing.
- Paged KV caches, cache offload, and cache-quantization optimizations.
- Execution-state persistence across process restarts.
- Arbitrary graph/DAG orchestration, manifest scripts, or user-defined predicates.
- Additional device-native implementations beyond CPU and CUDA.

Deferral does not change existing Session capabilities, including ordinary
batched inference.

## 3. Fundamental concepts

| Concept | Responsibility |
| --- | --- |
| Session | Existing model execution abstraction; existing contracts remain intact. |
| SessionLoop | Repeatedly executes a Session under explicit input, state, output, and exit-condition contracts. |
| SessionChain | Sequentially composes Session and SessionLoop components using declarative bindings. |
| Execution context | Owns mutable state, progress, counters, cancellation state, and applicable RNG state for one execution. |
| KVCacheManager | Owns or retains cache storage and manages cache initialization, handoff, updates, and reset. |
| Sampler | Converts logits into token choices; greedy is the initial algorithm. |
| Tensor/state binding | Connects an input, output, or retained-state slot to its producer or consumer, including differently named tensors. |
| Exit condition | Explicitly controls when a SessionLoop stops. |
| Diffusion scheduler | Implements the selected denoising timestep and latent-update policy. |

These are semantic responsibilities, not a requirement to introduce a separate
public class for every row. Reuse existing ORT abstractions where appropriate.
Exact API names and signatures beyond the requested core concepts remain design
work; they must preserve these contracts.

## 4. State ownership and lifecycle

- **STATE-1:** Separate reusable execution definitions from mutable per-execution
  state. Constructing another execution must not accidentally inherit another
  execution's mutable cache, counters, or outputs.
- **STATE-2:** Retain state across calls until the caller explicitly resets it.
  Completion, output retrieval, and ordinary call boundaries must not silently
  reset state. Reset starts a fresh conversation/execution.
- **STATE-3:** Reset clears execution progress, termination state, retained
  outputs/state, cache validity, cancellation state, and applicable RNG progress.
  Implementations may retain allocations, but stale content must not become
  logically visible. A configured seed must support reproducible restarted runs.
- **STATE-4:** Define and document ownership and lifetime of Sessions, contexts,
  cache handles, provider resources, and returned values. Follow existing ORT
  resource-management conventions in each binding.
- **STATE-5:** Simultaneous or reentrant use of one mutable execution context is
  unsupported initially. Do not introduce implicit concurrent chain execution.
  Sequential use of distinct contexts must remain isolated.
- **STATE-6:** Cancellation and failures must propagate explicit status
  information. If an interrupted or failed step may have partially mutated
  state, mark the affected execution as requiring Reset before further execution.
  Do not silently resume potentially corrupt state.

Successful completed steps need not be rolled back after a later failure.
Exceptions must not cross the public C API boundary.

## 5. SessionLoop

### 5.1 Explicit configuration and execution

- **LOOP-1:** The interface explicitly accepts the Session, input/state bindings,
  output selection, and exit conditions. Package loading supplies the same
  semantic configuration and uses the same validation rules.
- **LOOP-2:** Provide synchronous `Step`/`Next` semantics: advance at most one
  logical iteration, make that iteration's selected outputs available, and
  report progress or termination. Observing termination must not itself run
  another model iteration.
- **LOOP-3:** Provide run-until-exit as a convenience over the same stepping
  semantics, not a separate execution implementation with different behavior.
- **LOOP-4:** For ordinary tensor loops, the caller collects per-iteration
  outputs. Do not require the runtime to accumulate every intermediate tensor.
  Returned-value lifetimes and any requirement to copy borrowed/reused buffers
  must be explicit; retained outputs must not silently change.
- **LOOP-5:** Return an explicit exit reason and iteration/token accounting.
  Completion retains state. Stepping an already completed loop reports
  completion without executing again; Reset permits a fresh execution.

The stepping API is not a commitment to asynchronous streaming.

### 5.2 Exit conditions

- **EXIT-1:** Every loop must explicitly declare at least one exit condition.
  Reject configuration without one. Do not infer termination from names such as
  `prefill` or `decode`.
- **EXIT-2:** Support EndOfInputs, EOS token ID(s), maximum iterations, maximum
  generated tokens, and a named scalar boolean model-output condition.
- **EXIT-3:** Exit conditions are inputs to the SessionLoop interface, whether
  provided programmatically or through a package. They are not implicitly
  additional ONNX model inputs. A model-output condition explicitly binds its
  observed tensor.
- **EXIT-4:** Combine conditions with OR semantics: stop when any is satisfied.
  If multiple conditions become satisfied together, report all satisfied
  conditions or use a documented deterministic precedence.
- **EXIT-5:** Evaluate input exhaustion before attempting to consume another
  input. Evaluate output-dependent conditions after the producing iteration.
  Do not execute an extra iteration after termination.
- **EXIT-6:** Count generated tokens independently of prompt tokens. Validate
  limits, condition bindings, token IDs, and boolean-output types/shapes.
- **EXIT-7:** Do not implement custom predicate callbacks or executable
  manifest predicates in this delivery.

Edge-case contract: a zero iteration/token limit permits
immediate completion without model execution. This does not make an empty
prefill prompt valid. An emitted EOS is observable to the caller even if the
demo's detokenizer omits special tokens.

### 5.3 Prefill

- **PREFILL-1:** Initially accept the entire prompt as one finite input batch.
  EndOfInputs explicitly terminates its SessionLoop after that input is consumed.
- **PREFILL-2:** Reject an empty prompt with an actionable error.
- **PREFILL-3:** Preserve the distinction between logical input exhaustion and an
  empty tensor. Chunked prefill is deferred without changing EndOfInputs semantics.

## 6. SessionChain and bindings

- **CHAIN-1:** Represent a linear ordered chain of Session or SessionLoop
  components. Execute components sequentially.
- **CHAIN-2:** Explicitly map external inputs, upstream outputs, and retained
  state to component inputs. Tensor names need not match across components.
- **CHAIN-3:** Support iteration feedback and state handoff without treating all
  outputs as the next component's positional inputs.
- **CHAIN-4:** Validate missing/duplicate destinations, invalid references,
  unsupported types, incompatible known shapes/devices, and invalid component
  ordering before execution where metadata permits. Validate dynamic constraints
  at the appropriate runtime boundary.
- **CHAIN-5:** Make selected chain outputs and retained state accessible without
  silently discarding outputs needed by downstream components or the caller.
- **CHAIN-6:** Stop on a component failure, propagate the failing component and
  error, and do not execute subsequent components as though the chain succeeded.

Example binding: prefill output `present_key_0` becomes decode input `past_key_0`.
This is declarative data wiring, not a manifest state-machine transition.

## 7. KVCacheManager

- **CACHE-1:** Provide a CPU implementation first. Manage contiguous,
  tensor-backed caches with explicit tensor mappings and layout metadata.
- **CACHE-2:** Support initialization, retention, update/append as required by the
  model signature, logical length/capacity tracking, stage handoff, and reset.
- **CACHE-3:** Share caches among Sessions/loops within the same execution,
  particularly prefill and decode. Preserve ownership and lifetime across stages.
  Cross-request sharing is deferred.
- **CACHE-4:** Share compatible storage without unnecessary copies. Copies or
  reallocations are permitted when required by model I/O, growth, or supported
  device movement. Universal zero-copy handoff is not required.
- **CACHE-5:** Do not assume model outputs can alias model inputs. Respect model
  and kernel aliasing contracts; cache sharing must not require an unsafe graph
  rewrite or unsupported in-place execution.
- **CACHE-6:** Validate layout, dtype, shape, sequence lengths, capacity, and device
  compatibility. Use checked allocation-size arithmetic and propagate allocation
  failures. No new configurable hard memory-budget feature is required.
- **CACHE-7:** Provide real CUDA-backed management, respecting device/stream
  ordering and resource lifetimes, after the CPU baseline.

This is not a commitment to arbitrary heterogeneous multi-device pipelines.
Transfers needed by supported configurations must be explicit and correct;
unsupported configurations must fail clearly.

## 8. Sampler

- **SAMPLE-1:** Implement greedy sampling on CPU and expose it through the new
  execution layer. Keep the sampler independently testable.
- **SAMPLE-2:** Define supported logits shapes/dtypes and selection of the logits
  position. Return token IDs using a documented integer type.
- **SAMPLE-3:** Greedy selection chooses an argmax with deterministic tie-breaking.
  Define and test non-finite-logit handling consistently across implementations;
  malformed inputs must not produce success-shaped defaults.
- **SAMPLE-4:** Implement device-native CUDA greedy sampling. Verify dispatch to
  that implementation, rather than passing a CUDA test through CPU sampling.
- **SAMPLE-5:** Permit EP-provided sampler implementations. Temperature, top-k,
  top-p, beam search, and other algorithms are deferred.

The final specification approval covers the semantic requirement; exact
supported dtype/layout combinations and non-finite handling must be documented
with the API design before implementation.

## 9. Execution Provider customization

- **EP-1:** Extend the built-in and plugin EP interfaces so an EP can implement
  one or more KV-management features and/or sampling.
- **EP-2:** Provide capability discovery and clear ownership/lifetime contracts.
  Providers must be able to support a subset of features without pretending to
  implement the rest.
- **EP-3:** Preserve shipped public ABI contracts. Use version/capability checks
  at plugin boundaries; do not expose in-tree C++ implementation types across
  a boundary that cannot safely consume them.
- **EP-4:** Make implementation selection and required data movement observable.
  A missing required native capability must produce an explicit unsupported
  error, not an undisclosed CPU success. Any allowed fallback must obey the
  declared memory/device contract and be distinguishable in validation evidence.
- **EP-5:** Prove extension behavior with a test provider/plugin, including
  partial support, unsupported operations, and provider errors.

CUDA KV management and CUDA greedy sampling may be separate milestones.
Extending deprecated EPs/backends is not required to provide the new capability.

## 10. Public APIs and language bindings

- **API-1:** Cover C, C++, Python, C#, Java, JavaScript/TypeScript, Objective-C,
  Go, and Rust wherever Session is exposed in this repository. Include the
  applicable JavaScript runtime bindings rather than treating Node alone as
  complete JavaScript coverage.
- **API-2:** Expose programmatic construction, package construction, bindings,
  exit conditions, execution/stepping, output access, reset, cancellation, and
  resource release using each language's existing conventions.
- **API-3:** Preserve existing Session APIs and behavior. Preserve shipped public
  C API table layouts/signatures; additions are append-only in the applicable
  tables. Do not bump `ORT_API_VERSION` solely for these additions.
- **API-4:** Provide appropriate RAII C++ wrappers for new C API resources and
  document all public ownership and lifetime contracts.
- **API-5:** Surface errors through repository-standard mechanisms. Do not hide
  unsupported behavior in empty outputs, silent defaults, or broad catches.
- **API-6:** Binding work may arrive in successive milestone PRs, but incomplete
  bindings must not be represented as completed feature parity.

Follow repository C++ style, established helpers and types, platform/build
guards, and matching path-scoped instructions. Do not expand deprecated JSEP or
WebGL backends to satisfy binding parity; use supported execution paths and
explicit capability reporting.

## 11. Model Package extension

Existing references:

- [Standalone Model Package library](model_package/README.md).
- [ORT Model Package integration](onnxruntime/core/session/model_package/README.md).

- **PKG-1:** Extend `manifest.json` to declare named component chains, component
  order, Session versus SessionLoop execution, tensor/state bindings, explicit
  exit conditions, and references/configuration for supported native policies.
- **PKG-2:** Model files, variants, provider intent, and shared assets continue to
  use the existing package mechanisms. Reuse the existing parser, authoring,
  path-resolution, and ORT variant-selection infrastructure.
- **PKG-3:** Do not describe an executable state machine. No arbitrary jumps,
  transition scripts, embedded user code, or custom predicate expressions.
  Runtime implementations own execution and state-update semantics.
- **PKG-4:** Keep the standalone library independent of ORT, ONNX execution,
  Sessions, and EP implementation details. It may represent/validate declarative
  package structure; ORT owns execution-specific interpretation.
- **PKG-5:** Preserve existing packages and single-component Session creation.
  Specify schema versioning, absence/default behavior for new optional fields,
  strict unknown-field behavior, and round-trip authoring expectations.
- **PKG-6:** Preserve portable/installed path confinement, shared-asset
  resolution, and current restrictions on package-supplied Session options.
  Chain metadata must not bypass them.
- **PKG-7:** Programmatic and package-built executions with equivalent
  configuration must have equivalent results, termination, and error behavior.

The manifest field spelling and exact API signatures are not fixed by this
requirements draft. Document their concrete design before implementing the
schema, without changing these semantic requirements.

## 12. Demonstrations

### 12.1 Decoder-only LLM

- **LLM-1:** Use local files corresponding to
  [Microsoft Phi-3-mini-4k-instruct ONNX](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx)
  as the reference example. Inspect the actual on-disk variant and model
  signatures; do not assume all repository variants support both CPU and CUDA.
- **LLM-2:** Provide package-authoring instructions/tooling to produce
  `manifest.json`, component metadata, and references to model files/external
  weights. Do not commit large model binaries. Downloads require explicit
  user-approved setup; normal package loading and inference remain local-only.
- **LLM-3:** Run prefill as one SessionLoop and decode as another, connected in
  a SessionChain and sharing the execution's KV cache.
- **LLM-4:** Accept a text prompt and return generated text, using
  caller-provided tokenizer and detokenizer integrations. These integrations are
  application inputs, not executable package scripts.
- **LLM-5:** Use greedy sampling and explicit EndOfInputs/EOS/token-limit
  conditions. Account for the first token from prefill logits exactly once and
  avoid missing, duplicated, or extra tokens at the stage boundary.
- **LLM-6:** Run the package on CPU and CUDA using suitable local variants.
  The same ONNX file may serve both stages if its signature supports both;
  separate prefill/decode model files are not inherently required.

### 12.2 Stable Diffusion

- **SD-1:** Use a local Stable Diffusion 1.5-compatible ONNX pipeline:
  text encoder -> denoising SessionLoop -> VAE decoder.
- **SD-2:** Accept a text prompt and save an image file, using a caller-provided
  tokenizer. All required models/weights are supplied locally.
- **SD-3:** Implement one documented DDIM scheduler, classifier-free guidance,
  seeded initial noise, and explicit denoising termination.
- **SD-4:** Manage latents, scheduler progress, and relevant intermediate state
  through the new execution layer. KVCacheManager must not be misused as a
  universal container for diffusion state.
- **SD-5:** Provide package construction and execution instructions, including
  model/export assumptions, seed, step count, guidance settings, preprocessing
  and output postprocessing. For M15's explicitly selected CUDA-only exports,
  real inference is CUDA-only and CPU EP fixture validation remains required,
  as approved during the Appendix O interview.

Both demos must report missing local files, unsupported model signatures,
tokenizer integration failures, and incompatible EP configurations explicitly.
Model licenses and external-weight references must be respected.

## 13. Validation and acceptance criteria

### 13.1 Platform and provider requirements

- **TEST-1:** Linux is the initial build/test platform.
- **TEST-2:** Every milestone has CPU EP validation appropriate to its outcome.
  CUDA-relevant milestones also have CUDA EP validation; a CUDA-only feature
  still needs CPU baseline/compatibility checks where applicable.
- **TEST-3:** Validate native CUDA KV/sampling dispatch and resource behavior.
  Merely registering CUDA EP, or allowing CPU fallback, is not proof.
- **TEST-4:** Report unavailable hardware, dependencies, model files, and
  platform-specific tooling as not run/blocked, never as passed.

### 13.2 Unit-test coverage

- **COVER-1:** Unit tests must achieve measured **100% coverage of new executable
  production lines**. Branch coverage is not the required metric. This is
  coverage of new code, not 100% of the existing repository.
- **COVER-2:** The requirement applies across new runtime code, public APIs,
  bindings, package/tooling/demo production code, and CUDA host/device code.
  Test code, comments, documentation, and unchanged dependency/generated code
  are not new production-line coverage targets.
- **COVER-3:** Report the covered/total new executable lines, per-file evidence,
  instrumentation, and exact collection commands. Do not exclude reachable error
  paths or wrappers merely to obtain 100%.
- **COVER-4:** Passing integration tests or host-only instrumentation is not proof
  of device-code coverage. Unmeasured code remains an unmet gate unless the user
  explicitly approves a specification change.

### 13.3 Required behavioral evidence

| Area | Acceptance evidence |
| --- | --- |
| Compatibility | Existing stateless Session and existing package behavior remain intact. |
| Lifecycle | Retention, reset, sequential-context isolation, repeated completion, cancellation, and failure requiring reset. |
| Loop control | Each supported exit condition, combinations, missing/invalid conditions, limits, no extra iteration, and empty-prefill rejection. |
| Output delivery | Step/run equivalence, per-iteration output access, explicit lifetimes, and caller accumulation. |
| Bindings/chains | Renamed tensors, state feedback, valid handoff, invalid wiring/types/shapes, and downstream suppression after errors. |
| Cache | Initialization/growth/reset, stage sharing, permitted copies, checked sizes, lifetime/resource cleanup, and CPU/CUDA behavior. |
| Sampling | Known greedy results, ties, supported shapes/dtypes, malformed/non-finite inputs, and CPU/CUDA agreement for deterministic fixtures. |
| EP extension | Actual override dispatch, subset capabilities, plugin lifetime, unsupported operations, and provider errors. |
| Package | Author/parse/round-trip behavior, schema compatibility, confinement, invalid declarations, and programmatic/package parity. |
| Language bindings | Construction, execution, outputs, reset/cancellation, ownership, and error propagation in each applicable binding. |
| LLM | Tiny deterministic fixtures plus a real local Phi-3 text-to-text run on CPU and CUDA. |
| Diffusion | CPU EP deterministic fixtures plus a real CUDA text-to-image run for M15's approved CUDA-only exports, exercising DDIM and guidance. |

### 13.4 Numerical correctness and measurements

- **METRIC-1:** Use exact expected results for deterministic synthetic fixtures,
  including greedy token selection.
- **METRIC-2:** Use documented, justified numerical tolerances and an identified
  reference for real-model CPU/CUDA comparisons. Do not require identical real
  generated text/images across providers where numerical differences affect
  generation trajectories.
- **METRIC-3:** Verify seeded reproducibility within a fixed supported
  configuration. Record model variant, precision, provider, settings, and seed.
- **METRIC-4:** Establish reproducible latency, throughput, memory, and relevant
  copy/allocation measurements. There are no numerical performance thresholds
  for this delivery.

## 14. Delivery and approval workflow

- **DELIVERY-1:** Review and obtain approval of this document before preparing
  `TASKS.md`. Confirmed interview completion does not itself approve this draft.
- **DELIVERY-2:** Derive `TASKS.md` from the approved specification. Every
  milestone must state its tasks, intended outcome, dependencies, CPU test plan,
  applicable CUDA test plan, coverage evidence, and completion criteria.
- **DELIVERY-3:** Review and obtain approval of `TASKS.md` before implementation.
- **DELIVERY-4:** Keep its checklist current throughout execution. Use shared
  `SCRATCHPAD.md` for compressed decisions, discoveries, and any agent learnings.
- **DELIVERY-5:** Each milestone is an independently documented, tested,
  human-reviewable PR. Later PRs may depend on preceding PRs, but their incremental
  diff, prerequisites, outcome, and validation must be clear.
- **DELIVERY-6:** Target `onnxruntime/ort-state`. Base the first milestone branch
  on the current `ort-state` branch and stack later branches on their predecessor,
  using descriptive branch names.
- **DELIVERY-7:** Open PRs ready-for-review when their required evidence is ready.
  Honor the repository PR template and include exact validation commands/results.
  Do not claim incomplete or blocked validation as completed.

## 15. Dependencies and verification risks

These do not waive the requirements:

1. **Local model assets:** The caller must identify suitable local Phi-3 and
   Stable Diffusion files, external weights, and tokenizer/detokenizer
   integrations before the real-model demos can be verified.
2. **All bindings versus Linux validation:** Some binding/runtime targets,
   notably Objective-C and mobile variants, cannot be fully built or exercised
   on Linux. Their API parity remains in scope. Any unavailable validation and
   coverage must remain explicitly unverified; completing those gates may need
   user-approved supplemental platform/CI access.
3. **CUDA coverage instrumentation:** Tool availability for executable device-line
   coverage must be established. CPU/host coverage cannot silently replace the
   literal new-line coverage requirement.
4. **Model compatibility:** Actual cache signatures, precision, external-data
   layouts, and available CPU/CUDA variants must be verified before claiming a
   real model works. Unsupported variants must be diagnosed, not substituted
   silently.
5. **Concrete contract design:** Before implementation, document exact public
   signatures, manifest schema, cache-layout support, greedy tie/non-finite
   semantics, and simultaneous exit-reason reporting. Any departure from these
   requirements requires user review.

## 16. Approval record

On 2026-10-15, the user selected the first review option: approve `SPEC.md` and
prepare `TASKS.md` for a separate review. This approves the requirements and
edge-case contracts above without waiving the verification risks in Section 15.

Implementation, builds, tests, commits, pushes, and PR creation remain gated on
the separate execution-plan review. The specification approval alone does not
authorize starting those activities.

The user subsequently approved `TASKS.md` without changes on 2026-10-15 and
authorized M01 to begin. Progress and validation evidence are maintained there.

## Appendix A. M01 contract decisions

The user approved the execution plan and then explicitly confirmed all four
additional contracts below on 2026-10-15. This appendix refines, rather than
replaces, the requirements above.

### A.1 Confirmed behavioral contracts

1. **Native iteration policies.** Support generic tensor feedback, token
   generation, and DDIM denoising as explicit native implementations. Tensor
   wiring alone cannot synthesize attention masks, position IDs, sampled tokens,
   or scheduler inputs. Callers/packages select a supported policy and name its
   tensor ports and typed parameters. No executable manifest scripts, arbitrary
   predicate callbacks, or implicit policy inference from component names.
2. **Initial generation batching.** The token-generation policy initially
   supports one sequence per execution context. Generic tensor loops retain
   ordinary tensor batching. Fixed batched token generation, per-sequence EOS,
   and serving batching are not part of the initial generation policy.
3. **Stable selected outputs.** An iteration result owns its selected output
   lifetimes. Retaining that result across later steps or reset must preserve
   its values. Retain existing owned tensors where safe; copy selected tensors
   when their backing storage would otherwise be overwritten. Account for these
   copies. Mutable cache inspection is a separate explicit state operation.
4. **Greedy semantics.** Initially support float32 and float16 logits with int64
   token IDs. Lowest token ID wins ties. Reject NaN logits with an explicit
   error. Positive/negative infinity follow numeric ordering and the same tie
   rule; all-negative-infinity logits therefore select token zero. Other
   dtypes are explicitly unsupported until implemented. CPU and CUDA agree.

### A.2 Execution and ownership design

The following design implements the approved semantics. Exact declarations must
still be checked against existing public API conventions before publication.

- A loop definition is immutable after construction and references an existing
  Session. A chain definition retains its component definitions.
- A new execution owns counters, retained state values, cache/scheduler progress,
  pending token state, and cancellation/termination state. It retains its
  definition so releasing a definition handle does not invalidate an execution.
- Existing Session handles remain caller-owned: the Session must outlive the
  definitions/executions that reference it. Do not silently transfer ownership
  of an existing Session or alter existing Session release behavior.
- Input OrtValues are retained using existing ORT ownership semantics. If a value
  wraps caller-owned memory, that memory must remain valid for its documented
  use period. Retention is not a promise to deep-copy every input buffer.
- A step result reports whether a model iteration executed, component identity,
  completed iteration/token counts, completion state, and all simultaneously
  satisfied exit reasons in documented enum order.
- Loop `Step` executes at most one model iteration. Chain `Step` executes at most
  one component Session invocation or one component-loop iteration; transitioning
  bookkeeping does not execute the next Session behind the caller's back.
- Calling Step after completion returns completion with `executed == false`.
  No new model invocation or duplicate output emission occurs.
- Selected output getters borrow immutable values from the owning result.
  Language wrappers retain the result owner when exposing a borrowed value.
  Releasing a result invalidates its borrowed views, not other retained results.
- Reset invalidates logical mutable state, not already retained step results.
  An execution requiring reset refuses further execution with an explicit error.
- Cancellation uses supported ORT termination mechanisms and between-step
  checks; it does not promise preemption of a running device kernel or permit
  concurrent/reentrant execution of the context.

### A.3 Inputs, policies, and token accounting

- Distinguish invariant external inputs, finite per-iteration input batches,
  previous-iteration outputs, retained state, and native-policy outputs.
  Source namespaces are explicit so equal tensor names cannot create ambiguity.
- EndOfInputs is valid only for a declared finite input source. It means logical
  exhaustion, not an empty tensor or a missing required feed.
- Generic feedback can consume finite batches and/or retained feedback according
  to its declared bindings. Validation rejects conflicting producers.
- Token-generation configuration explicitly declares prefill or decode phase,
  token/logits/cache ports, relevant mask/position ports, and supported layout
  conventions. Unknown or incompatible signatures fail; they are not guessed.
- Full-prompt prefill has one finite input batch and EndOfInputs. Its configured
  native policy may sample the first output token from the prompt logits.
- The first sampled token counts against the shared generation token budget and
  is emitted exactly once. Decode consumes that pending token as its next model
  input, updates masks/positions/cache, then samples the next token.
- EOS and token-budget conditions declared for decode are checked on stage entry
  as well as after token production. An EOS/budget reached by the prefill sample
  must not cause an unnecessary first decode model invocation.
- Prompt tokens do not increment the generated-token counter. Loop iteration
  counts are local to that loop; generation-token accounting spans the explicitly
  connected prefill/decode policies in the same execution.
- Greedy accepts `[1, vocabulary]` or `[1, sequence, vocabulary]` logits. Vocabulary
  and sequence dimensions must be positive; rank-three selection uses an
  explicitly configured, validated position, with last position as the supported
  token-generation default. It does not silently select another batch row.
- DDIM configuration explicitly identifies latent, timestep, prediction, and
  guidance ports. Scheduler state is ordinary execution state, not a KV cache.

### A.4 Public API organization

Use a dedicated append-only `OrtStatefulApi` table obtained through one new
append-only `OrtApi` accessor. Do not add dozens of unrelated functions directly
to the main table or change `ORT_API_VERSION` during feature development.

Intended opaque resources:

| Resource | Purpose |
| --- | --- |
| `OrtSessionLoopOptions` | Mutable typed loop configuration before construction. |
| `OrtSessionLoop` | Immutable Session/policy/binding/condition definition. |
| `OrtSessionChainOptions` | Mutable ordered components and connection configuration. |
| `OrtSessionChain` | Immutable chain definition. |
| `OrtStatefulExecution` | Per-execution mutable context for a loop or chain. |
| `OrtStatefulStepResult` | Owned iteration/progress metadata and selected outputs. |
| `OrtKVCacheOptions` | Typed layout/device/cache binding configuration. |
| `OrtSamplerOptions` | Typed algorithm/logits/token configuration. |

Core signatures to carry into M06, using existing ORT status/calling conventions:

```c
OrtStatus* CreateSessionLoop(
    OrtSession* session, const OrtSessionLoopOptions* options,
    OrtSessionLoop** out);
OrtStatus* CreateSessionChain(
    const OrtSessionChainOptions* options, OrtSessionChain** out);
OrtStatus* CreateLoopExecution(
    const OrtSessionLoop* loop, OrtStatefulExecution** out);
OrtStatus* CreateChainExecution(
    const OrtSessionChain* chain, OrtStatefulExecution** out);
OrtStatus* StatefulExecutionStep(
    OrtStatefulExecution* execution, OrtStatefulStepResult** out);
OrtStatus* StatefulExecutionRun(
    OrtStatefulExecution* execution, OrtStatefulStepResult** out);
OrtStatus* StatefulExecutionReset(OrtStatefulExecution* execution);
OrtStatus* StatefulExecutionCancel(OrtStatefulExecution* execution);
```

The option/input/result/cache inspection functions must use typed parameters and
documented array counts. They must cover all stateful operations, not require
users to supply opaque JSON to construct programmatic execution. Cache operations
and sampler invocation remain independently testable. Every created resource
has its corresponding release function and appropriate C++ RAII wrapper.

Run repeatedly uses Step, retaining only the final result unless selected
policy-level result collection was explicitly configured. Applications requiring
each ordinary tensor result use Step and accumulate results themselves.

These are design signatures, not shipped declarations; SAL annotations,
per-function API documentation, table initialization, and version negotiation are
part of M06. Material semantic changes return to the user before implementation.

### A.5 KV and provider boundary design

- Describe each contiguous KV tensor using named input/output/state ports,
  element type, dimensions, sequence axis, logical length, capacity, and device.
  Validate all non-sequence dimensions; never infer a compatible layout solely
  from equal allocation sizes.
- Separate model-managed growing outputs from compatible shared-buffer outputs.
  Only use shared input/output buffers when the model/kernel contract permits.
- CPU implementations and CUDA overrides implement the same validated operations:
  initialize, retain/handoff, append/update, grow/copy where supported, and reset.
- In-tree provider hooks discover individual capabilities rather than requiring
  every EP to subclass an all-or-nothing cache/sampler implementation.
- Plugin callbacks use a versioned, size-tagged C descriptor containing only
  boundary-safe handles, primitive metadata, and function pointers. ORT retains
  provider/plugin resources while registered stateful objects depend on them.
- Prefer registering that optional descriptor through an append-only `OrtEpApi`
  function. Do not require ORT to read new fields from a shorter, older plugin's
  `OrtEp` allocation.
- Old plugins continue without registering stateful overrides. New callbacks
  follow negotiated runtime/API capabilities. Unsupported mandatory operations
  fail explicitly; CPU fallback and movement require an allowed, observable path.

### A.6 Declarative manifest design

Use an additive schema-minor extension, initially `1.1`, with optional named
`chains`. Existing `1.0` packages remain accepted and existing component/variant
resolution is unchanged. An older strict parser may reject new chain fields;
forward readability by old strict consumers is not promised.

The standalone library owns generic chain/step identities and component
references. ORT-specific execution kinds, bindings, exits, and policies live in
consumer-owned `executor_info["ort"]` payloads. This preserves the library's
execution-independent boundary while keeping the declarations in `manifest.json`.

Illustrative shape, not yet a complete runnable LLM package:

```json
{
  "schema_version": "1.1",
  "components": {
    "decoder": "decoder"
  },
  "chains": {
    "text_generation": {
      "steps": [
        {
          "id": "prefill",
          "component": "decoder",
          "executor_info": {
            "ort": {
              "execution": "session_loop",
              "iteration_policy": {
                "kind": "token_generation",
                "phase": "prefill"
              },
              "exit_conditions": [
                { "kind": "end_of_inputs" },
                { "kind": "max_generated_tokens", "limit": 128 }
              ]
            }
          }
        },
        {
          "id": "decode",
          "component": "decoder",
          "executor_info": {
            "ort": {
              "execution": "session_loop",
              "iteration_policy": {
                "kind": "token_generation",
                "phase": "decode"
              },
              "exit_conditions": [
                {
                  "kind": "eos",
                  "source": "policy.generated_token",
                  "token_ids": [32000]
                },
                { "kind": "max_generated_tokens", "limit": 128 }
              ]
            }
          }
        }
      ],
      "executor_info": {
        "ort": {
          "bindings": [
            {
              "from": {
                "step": "prefill",
                "source": "state",
                "name": "kv"
              },
              "to": {
                "step": "decode",
                "destination": "state",
                "name": "kv"
              }
            }
          ]
        }
      }
    }
  }
}
```

The illustrated EOS ID is not a substitute for reading a real model's tokenizer
configuration. Complete model tensor/cache ports and text adapters are supplied
when authoring the actual demo package.

Chain/step external consumer-payload references resolve from the package root;
existing variant-payload references remain anchored at the variant directory.
Use the existing resolver and confinement rules for both. Standalone APIs must
inspect, author, commit, and round-trip the new generic declarations without
interpreting the ORT payload. ORT validates policy/binding/condition semantics.

### A.7 Fixture and instrumentation strategy

M01 adds only test fixtures/helpers and documentation, not stateful runtime
production code. Tiny models exercise scalar feedback/boolean output, renamed
stages, known logits with growing contiguous cache, and denoiser timestep input.
Direct stateless Session runs provide expected outputs for later milestones.

The initial host coverage configuration uses Clang 18 with
`-fprofile-instr-generate -fcoverage-mapping`, separate raw profiles per process/
instrumented image, `llvm-profdata`, and `llvm-cov`. Successful collection and
source mapping must be demonstrated before claiming readiness.

There are zero new executable production lines in this test-only milestone;
its production coverage denominator is zero/not applicable, not a fabricated
100%. Instrumented fixture/helper execution is a readiness check. M02 onward
must measure actual new production lines. Nsight Compute or Compute Sanitizer
availability alone does not establish CUDA device-line coverage.

A M01 feasibility probe subsequently ran a tiny debug kernel under CUDA-GDB,
verified breakpoint hits on both executable kernel-body lines, and exited
normally. This establishes a possible source-line measurement workflow, not
coverage of stateful production kernels. Those kernels still require their own
complete line mapping and unit-test hit evidence.

## Appendix B. M02 internal SessionLoop contract

M02 implements native `SessionLoop` definitions and `SessionLoopExecution`
contexts. It does not publish the C API or implement token sampling, KV
management, or SessionChain.

- Definitions reference an initialized caller-owned InferenceSession. Executions
  retain the immutable definition; the Session must outlive both.
- Options explicitly select outputs, finite input names, optional output-to-input
  feedback, and typed exits. At least one selected output and one exit are
  required. The initial implementation accepts dense non-string tensors.
- Initial feeds contain invariant inputs and initial feedback values. Finite
  batches contain exactly the named finite inputs. Model names, types, ranks,
  and known dimensions are validated before execution.
- EndOfInputs treats an empty generic finite sequence as immediate completion.
  Configuring a `nonempty_input` additionally rejects an empty sequence or tensor,
  including when the iteration limit is zero, for prefill-style inputs.
- EOS/token-limit testing uses an explicitly named scalar int64 model output:
  one produced token per successful iteration. This is not a substitute for the
  sampler/native generation policy in M05.
- Conditions combine with OR semantics; simultaneous reasons appear in
  `LoopExitKind` order: EndOfInputs, MaxIterations, MaxGeneratedTokens, Eos,
  BooleanOutput. Model-output checks occur after the producing invocation;
  finite-source exhaustion and zero budgets also allow immediate completion.
- Step performs at most one invocation. Run repeatedly calls Step and returns
  its final result. Completed calls report `executed == false`, with counters/
  reasons preserved and no duplicate output emission.
- M02 deliberately fetches dense outputs to CPU through IOBinding and snapshots
  them into CPU-owned storage. This includes control/feedback outputs needed for
  correctness, even when not selected for caller observation. This copy-based
  baseline is explicit; M04/M08 provide device-native cache management later.
- Returned results retain read-only snapshots across subsequent steps/reset.
  Caller-owned input buffers still require their ordinary ORT lifetime. Reset
  clears progress/feedback/cancellation and reuses the execution's original feeds.
- Cancel is the only operation permitted concurrently with execution. It posts
  an atomic request observed before/after a Session invocation; it does not
  preempt that invocation. Do not cross-thread-write ORT's existing non-atomic
  RunOptions termination flag. Cancelled or potentially mutated failed execution
  requires Reset; completed successful steps are not rolled back.
- Status failures preserve the caller's result. Exceptions propagate under the
  usual internal ORT convention while leaving the context requiring Reset.

## Appendix C. M03 internal SessionChain contract

- A chain is an immutable, nonempty, ordered list of named Session or SessionLoop
  components with explicit input bindings and at least one selected output port.
  Executions retain their definition; Sessions remain caller-owned.
- A plain Session component executes once. Internally this reuses a one-iteration
  SessionLoop rather than adding a second invocation/snapshot implementation.
  An explicitly supplied SessionLoop retains its configured exits and feedback.
- Source namespaces distinguish an external tensor, an external finite sequence
  of tensors, an earlier component output, and a named retained-state slot.
  Every model input has exactly one binding; names need not match across stages.
- An external tensor bound to a finite input supplies one finite batch. External
  sequences may bind only to declared finite inputs, and all finite sources for
  a component must have equal lengths. There is no implicit sequence broadcast.
- A state slot first consumed before any producer requires an initial state
  value. Declared state outputs update slots after each successful invocation;
  later components see the most recently committed snapshot. State types and
  known dimensions must remain compatible.
- Definitions validate names, duplicate destinations, source kinds, output ports,
  forward/self references, and known type/shape incompatibilities. Execution
  creation validates external/initial values and finite sequences. Dynamic
  handoff constraints are validated before invoking the consuming component.
- Step executes at most one Session invocation or one loop iteration. Completed
  zero-iteration components may be skipped as bookkeeping before an invocation.
  Completing one component never invokes the next behind the caller's back.
- Each result identifies the current component, its loop progress/exit reasons,
  total successful model invocations, and the number of completed components.
  Per-invocation component outputs are available immediately. Named chain output
  ports are emitted when the whole chain completes, without duplicate emission
  on subsequent calls. Run is repeated Step, not a separate executor.
- The chain retains all dense component outputs needed for later wiring or
  observation, including outputs not selected by an original loop definition.
  A zero-iteration component does not invent model outputs: attempting to consume
  or return an unproduced value fails explicitly.
- M03 preserves M02's CPU snapshot boundary. External and initial-state tensors
  must be host-accessible; unsupported device inputs fail before invocation.
  CUDA Sessions use their existing IOBinding transfers and compute paths, with
  CPU-owned handoff/output snapshots. No native KV cache/device-buffer sharing
  or arbitrary multi-device orchestration is claimed.
- Retained step results remain stable across later invocations, reset, and
  execution release. State inspection exposes retained values; original
  caller-owned input memory follows ordinary ORT lifetime rules.
- Errors identify the failing component and preserve the underlying Status code/
  message. A failed or cancelled execution conservatively requires Reset, and
  downstream components do not execute. Internal exceptions leave the context
  requiring Reset and propagate normally.
- Cancel is an atomic request observed at component invocation boundaries, not
  preemption. Only Cancel may be concurrent. Reset drops active component
  execution, results/state/progress/cancellation and reinstates the original
  external inputs/initial state. Independent executions never share mutable
  progress or implicitly inherit another execution's updated state.

## Appendix D. M04 CPU KVCacheManager contract

- Each cache has an explicit name, float32/float16/bfloat16 element type, rank,
  positive fixed non-sequence dimensions, sequence axis, and initial allocation
  capacity. The layout's initial logical sequence dimension is zero. Cache
  names distinguish layers/keys/values; no layout is inferred from buffer size.
- Logical length and allocation capacity are separate. The exposed tensor is
  densely packed for its logical shape, even when the sequence axis is not the
  leading axis. Unused capacity is not exposed as tensor elements or accidentally
  treated as per-head padding.
- A newly created execution initializes its managed caches empty. Update/import,
  append, reserve, inspection, and reset are independently testable on the CPU
  manager. Nonempty cache seeding through the public execution APIs remains
  future API work; ordinary user feeds cannot override managed input ports.
- Full updates explicitly choose Share or Copy. Share retains an allocator-owned
  CPU model output without extra cache-manager data copies. Borrowed buffers
  require Copy. A supplied shared tensor must not subsequently be mutated by
  the caller; ordinary ORT ownership/lifetime rules still apply.
- Sharing adopts the supplied tensor's actual allocation extent: capacity becomes
  its logical sequence length, rather than claiming it has the old reservation.
  Copy preserves at least the prior reservation and grows when needed.
- Append accepts only the newly produced sequence portion. It copies old/new
  rows into compact logical packing with checked sizes/offsets. Empty appends
  are no-ops. Full updates may preserve or grow logical length, not shrink it;
  Reset is the explicit operation for clearing logical content.
- Reserve is an at-least-capacity request, a no-op when capacity is sufficient.
  Copy/Append/Reserve mutation uses fresh storage so previously returned views
  remain stable; no unsupported model input/output aliasing is enabled.
- Reset clears logical length while allowing allocation retention. Zero logical
  shape prevents stale data becoming visible, and later writes do not overwrite
  retained pre-reset views. Views retain backing allocations after manager,
  context, Session, or producing result release.
- Manager statistics report lifetime data-buffer allocations, bytes copied, and
  successful shared updates. They exclude metadata allocations, Session output
  snapshots and I/O transfers; they do not imply a performance threshold or
  device-native cache implementation.
- Validate dtype, rank, fixed dimensions, contiguous CPU location, sequence axis,
  capacity and checked allocation/index arithmetic before copying. Unknown names,
  unsupported update kinds, invalid metadata, failed/null allocation, and overflow
  propagate explicit Status errors or normal internal ORT exceptions.
- A failed single-cache mutation preserves that cache's prior value. Across
  multiple cache updates, earlier successful updates need not be rolled back;
  a failed execution is marked requiring Reset before reuse.
- SessionLoop has explicit managed input/output bindings and full/delta update
  policies. Managed inputs do not also accept finite/invariant/feedback bindings.
  Control and selected outputs retain their existing snapshot contracts.
- A SessionChain owns one cache manager per execution, shared by its component
  loops and plain Session wrappers. Same-name caches require identical layout
  declarations. Named state can observe/hand off those caches; compatible full
  state outputs may update them, but duplicate managed/state writers are invalid.
- Standalone loop reset clears its own manager. Component transitions do not
  reset the chain's shared manager; chain reset clears it once. Independent
  executions never share mutable cache-manager state.
- CPU caches reject GPU/device allocations and device-resident updates. Testing
  a CUDA Session with host snapshots verifies integration and rejection behavior,
  not CUDA-backed cache management. Native CUDA storage remains M08.

## Appendix E. M05 CPU greedy sampling and generation contract

The user explicitly approved these refinements on 2026-10-16:

1. If used, a shared generated-token budget is declared explicitly and identically
   on prefill and decode. Do not infer or move exit conditions between stages.
2. Initial native generation uses unpadded int64 IDs shaped `[1, sequence]`,
   optional named int64 all-ones attention-mask inputs, and sequential position IDs.
3. CPU greedy supports float32/float16 and rejects any NaN in the supplied logits
   tensor, including unselected positions. Device logits are explicitly rejected.
4. Decode follows prefill in the same execution context. Importing partial
   generation state/nonempty cache through execution APIs remains future work.

### E.1 Sampler

- Token generation and sampling reside in `onnxruntime/core/generate/`, separate
  from the Session/SessionLoop/SessionChain execution interfaces. Shared tensor/
  argument validation resides in `onnxruntime/core/framework/` so generation
  code does not depend on session implementation headers.
- `Sampler::Greedy` is independently testable and CPU-only. Supported logits are
  `[1, vocabulary]` or `[1, sequence, vocabulary]`; dimensions must be nonempty.
- An optional nonnegative position selects a sequence row; absence selects the
  last row. Rank-two logits have one row, so only position zero is valid.
- Return int64 IDs. Standard argmax selection chooses the lowest ID on ties;
  infinities follow numeric ordering, including all-negative-infinity -> zero.
  Validate the whole supplied tensor for NaN before returning a token.
- Reject unsupported dtypes, batch/rank/position, null buffers and device memory.
  Validate known metadata at definition time and dynamic values at execution.

### E.2 Native iteration policy

- `TokenGenerationOptions` explicitly identifies Prefill/Decode phase, input IDs,
  logits output/position, optional attention-mask and position-ID ports.
  Generation-managed inputs cannot also be finite/invariant/feedback/cache ports.
- Prefill consumes exactly one nonempty finite prompt batch and declares
  EndOfInputs. Empty or negative-ID prompts are invalid even with a zero budget.
  Decode generates its own one-token input from the last sampled pending token.
- Masks contain ones for all processed/current tokens. Position IDs begin at the
  number of previously consumed input tokens. Other layouts/padding/signatures
  are not guessed.
- Generated CPU feed buffers remain owned through IOBinding synchronization and
  Session execution so asynchronous device copies cannot outlive their sources.
- A chain initially supports one prefill/decode pipeline. Phases have matching
  explicit token budgets (or neither has one) and matching managed cache names/
  layouts. A standalone prefill loop is supported; standalone decode is not.
- The execution owns a small history of generated token IDs, processed-input
  count, and prefill-completion state. This does not accumulate ordinary tensor
  intermediates; callers still collect per-iteration tensor results themselves.

### E.3 Stage boundaries, exits and outputs

- The first sampled token comes from prefill logits, increments the shared
  generation count, and is emitted once. Decode consumes that pending token
  before sampling the next one. Prompt tokens do not count as generated tokens.
- EOS and token limits inspect pending state on decode entry, so prefill EOS or
  an already-exhausted budget causes no extra decode invocation or emission.
- A zero budget performs no prefill/decode invocation and yields an empty token
  result. Unrelated generic chain components retain their own execution rules.
- Each successful sampling iteration exposes one optional `generated_token` in
  its step result. Completion-only calls do not emit it again. Counts are shared
  across phases while loop iteration counts remain local.
- A typed `GeneratedTokenIds` chain output port returns an owned CPU int64 tensor
  shaped `[generated_count]`, including `[0]`. This supports Run convenience and
  zero-invocation completion without inventing model logits/cache outputs.
  Selecting an unproduced ordinary model output still fails explicitly.
- Cache lengths are checked against processed input tokens before invocation and
  after updates. The final pending token has not yet entered the KV cache.
- Reset clears generated IDs, consumed-input counters, pending/prefill state and
  managed caches. Retained result tensors remain stable. Failures/cancellation
  retain the existing requires-Reset contract; no silent sampling fallback.

M05 does not add public ABI or EP/plugin overrides. CUDA Session integration uses
explicit host staging and this CPU sampler; native CUDA sampling remains M09.
The measured line-coverage gate targets the Linux inference configuration, not
unbuilt training-only strided-tensor paths.

## Appendix F. M06 public C/C++ interface

**Status:** Approved 2026-10-16; M06b authorized by the user on 2026-10-17.

This contract preserves the approved runtime behavior and ownership model.
It makes concrete the configuration, input and observation surfaces needed to
wrap M02-M05. M06 has two independently tested/reviewable sub-PRs:
the complete C ABI first, then C++ RAII wrappers. Neither alone completes M06.

### F.1 ABI organization and compatibility

- Append one `OrtApi::GetStatefulApi()` accessor and its implementation pointer;
  never reorder/change existing entries or increment `ORT_API_VERSION`.
- Define a dedicated append-only `OrtStatefulApi` table using existing ORT
  calling/status/release conventions. The table is runtime-owned, not released.
- Current `ORT_API_VERSION` is 32, so new API members are documented with
  `\since Version 1.32.` Runtime/header version negotiation remains via
  `OrtApiBase::GetApi(ORT_API_VERSION)` before accessing the appended member.
- Follow existing companion-API behavior for unsupported builds: explicitly
  document a NULL table/unsupported status and a corresponding C++ exception,
  not successful empty results. Linux inference is the initial validation target.

### F.2 Typed configuration

Use C-compatible enums and size-tagged POD descriptions, not JSON:

- Loop configuration: finite input names, feedback bindings, selected outputs,
  explicit exits/limits/EOS IDs, optional model-token output/nonempty input,
  managed cache bindings and optional native generation policy/ports.
- Chain configuration: named ordered Session/SessionLoop components, typed
  external/sequence/component-output/state sources, state writers, cache bindings,
  and selected model/generated-token output ports.
- Cache configuration: names, ONNX element types, dimensions, sequence axes and
  initial capacities; typed Share/Copy/Append update modes.
- Sampler configuration: Greedy algorithm and optional logits position.
- Input configuration: named OrtValues, finite loop batches, named chain
  sequences and initial ordinary state. Reject fields belonging to the wrong
  execution kind rather than ignoring them.

Copy all configuration strings/arrays into opaque mutable options. A typed
SetConfig operation replaces an options configuration atomically; definitions
capture an immutable snapshot. Validate pointer/count pairs, names/enums and
size tags before dereferencing/copying. Size tags allow future append-only
configuration evolution without reading beyond caller-provided structures.
Keep the original last-field endpoints as the minimum accepted V1 sizes.
Metadata getters populate only the known prefix and leave excess bytes untouched.

Intended resources are the approved LoopOptions, Loop, ChainOptions, Chain,
StatefulExecution, StatefulStepResult, KVCacheOptions and SamplerOptions handles,
plus KVCacheManager and retained KVCacheView handles for independent cache use.
Every resource has its matching release function; release accepts NULL.

### F.3 Construction and execution

Retain the Section A.4 names, with the approved signature refinement:

```c
OrtStatus* CreateLoopExecution(
    const OrtSessionLoop* loop, const OrtStatefulInputs* inputs,
    OrtStatefulExecution** out);
OrtStatus* CreateChainExecution(
    const OrtSessionChain* chain, const OrtStatefulInputs* inputs,
    OrtStatefulExecution** out);
```

Providing typed inputs at creation matches the native implementations and
prevents partially configured executions. Reset reuses the original inputs,
as already implemented/approved; constructing with a different prompt requires
a new execution. Configurations and OrtValue handles are retained/copied;
caller-backed tensor memory still follows ordinary ORT lifetime requirements.

Expose Step, Run, Reset, Cancel, requires-Reset inspection, and retained-state
inspection. Step/Run return an owned result. Failed calls preserve every supplied
out-parameter. Use API_IMPL_BEGIN/API_IMPL_END at status-returning boundaries
so internal exceptions become OrtStatus and never cross the C ABI.

### F.4 Results and ownership

- Sessions remain caller-owned and must outlive referencing definitions and
  executions, exactly as in the approved native contract.
- Definitions retain component definitions; executions retain their definition.
  Releasing options/definition handles does not invalidate a captured execution.
- An owned step result exposes execution kind, progress/counters, component
  identity, completion/executed flags, exit reasons and optional emitted token.
- Output collections distinguish selected outputs, current component outputs
  and explicit state snapshots. Names and const OrtValue pointers are borrowed
  from the owning result and remain valid until that result is released.
- State inspection must retain/snapshot its data, not expose a mutable container
  whose values silently change on subsequent steps/reset.
- Standalone loop state uses feedback input names and managed cache names;
  collisions report an error rather than silently omitting a value. Its
  component-output collection mirrors selected outputs. Chain snapshots expose
  ordinary named state and managed caches; component outputs describe the
  current component. Invalidated executions require reset before state inspection.
- Cache views own the retained value plus logical length/capacity metadata.
  Their const value remains valid across manager updates/reset/release.
- C++ read-only value views retain a shared owner of the result/cache-view
  handle. They therefore cannot dangle merely because the original C++ result
  wrapper was moved/destroyed. No deep data copy is required just to retain a
  read-only view.
- Follow existing C++ move-only RAII patterns for mutable resources. Document
  the necessary shared ownership used by immutable result/value views.

### F.5 Independent cache and sampler operations

Expose cache create/get-view/update/reserve/reset/statistics using the validated
M04 CPU implementation and an explicit OrtAllocator. Document that a caller's
custom allocator must outlive any values/views still using its callbacks.
Reject unsupported device configurations; do not imply M08 functionality.

Expose standalone greedy sampling through SamplerOptions and the M05 CPU
sampler. Device logits remain unsupported; no M09 fallback is invented.
Package construction and EP/plugin override registration remain M11/M07 work.

Before publication, fully document every public member's arguments, output
ownership, failure behavior and release requirements. C-only compilation,
public C/C++ parity, ABI table offsets and measured new-line coverage are
mandatory gates for the two approved sub-PRs.

### F.6 C++ ownership and executable example

The header-only C++ API wraps every C operation. Configuration constructors and
SetConfig take the same typed C records, including their explicit struct_size.
Mutable resources are move-only; C status failures become Ort::Exception.
Sessions and caller tensor buffers retain the C API lifetime requirements.

StatefulStepResult and KVCacheView are copyable shared owners, not ordinary
pointer-sized resources. Their const C pointer conversions are borrowed; they
do not provide release(), which would conflict with retained views.
StatefulValue reuses the SDK's const value operations with parent-retaining
storage. Copies retain the owner; moves empty the source. It cannot implicitly
slice to unowned ConstValue or convert to mutable OrtValue pointers.
Raw tensor/metadata pointers remain borrowed from the retaining value view.

StatefulStepInfo copies component names/exit reasons and represents emitted
tokens with std::optional. These metadata copies do not depend on result
lifetime. Step/Run allocate result ownership before native advancement, so an
ownership allocation failure cannot lose an already-executed iteration.
Empty resources can be destroyed even when the stateful API is unavailable;
attempting an operation still throws explicitly.

This standalone C++17 example uses only the public SDK. The cache tensor
remains valid after the original tensor, manager and view are destroyed:

```cpp
#include "onnxruntime_cxx_api.h"

int main() {
  Ort::AllocatorWithDefaultOptions allocator;
  const int64_t empty_shape[]{0}, value_shape[]{1}, logits_shape[]{1, 3};
  const OrtStatefulKVCacheSpec spec{
      "key", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, empty_shape, 1, 0, 0};
  const OrtStatefulKVCacheConfig cache_config{sizeof(cache_config), &spec, 1};
  Ort::KVCacheOptions options(cache_config);
  auto snapshot = [&] {
    Ort::KVCacheManager cache(options, allocator);
    auto key = Ort::Value::CreateTensor<float>(allocator, value_shape, 1);
    key.GetTensorMutableData<float>()[0] = 3.f;
    cache.Update("key", key, ORT_KV_CACHE_SHARE);
    return cache.GetView("key").GetInfo();
  }();

  const OrtStatefulSamplerConfig config{sizeof(config), ORT_STATEFUL_GREEDY, 0, 0};
  Ort::SamplerOptions sampler(config);
  auto logits = Ort::Value::CreateTensor<float>(allocator, logits_shape, 2);
  auto* scores = logits.GetTensorMutableData<float>();
  scores[0] = 1.f;
  scores[1] = 4.f;
  scores[2] = 2.f;
  return sampler.Sample(logits) == 1 && snapshot.length == 1 &&
                 snapshot.value.GetTensorData<float>()[0] == 3.f
             ? 0
             : 1;
}
```

For model execution, construct SessionLoop(session, options) or
SessionChain(options), then CreateExecution(inputs). Step/Run return immutable
results; GetOutputs returns owned names and retaining const values. GetInfo
returns copied metadata, and GetState returns retained state snapshots.
The public-API unit tests exercise all four Session/Loop chain combinations,
explicit exits, reset/cancellation, prefill/decode and zero/one/multiple-token
budgets on CPU and CUDA Sessions.

## Appendix G. M07 provider capability and dispatch contract

**Status:** Approved by the user on 2026-10-17 without changes. The behavior
choices, M07a/M07b split and absence of additional specification input are
confirmed. The detailed execution plan and both sequential sub-PRs were also
approved without changes before implementation.

This appendix refines EP-1 through EP-5 and Appendix A.5. It does not add native
CUDA cache/sampling, new sampling algorithms, opaque/paged cache layouts,
arbitrary heterogeneous pipelines or package configuration. Those existing
scope boundaries and later milestones remain unchanged.

### G.1 Independent selection and backward compatibility

- Cache management and sampling have independent provider selections. Selecting
  one does not implicitly select the other.
- The default remains explicit CPU behavior, preserving M02-M06. Registering a
  CUDA/plugin EP for model execution alone does not change stateful dispatch.
- Expose CPU, PreferProvider and RequireProvider policies through native and
  public C/C++ interfaces. Provider-backed choices identify an EP already
  registered in an initialized Session and a compatible device.
- CPU uses the reference implementation. PreferProvider uses an advertised,
  compatible override; missing capabilities may use CPU only under G.3.
  RequireProvider never substitutes CPU for a required provider operation.
- Unknown provider names, ambiguous devices and incompatible memory/layout
  requests report explicit errors rather than silently choosing CPU.
- Existing Session-free standalone CPU cache/sampler APIs remain valid.
  Provider-backed standalone use explicitly binds an initialized Session;
  objects retain the required provider resources instead of an unsafe borrowed
  callback pointer. Merely choosing a provider does not load an arbitrary DLL.

### G.2 Per-feature capabilities and coherent cache state

- Discovery distinguishes individual supported KV operations and greedy
  sampling; providers need not implement an all-or-nothing manager or sampler.
- KV capabilities cover initialization/allocation, full-output sharing/copy,
  delta append, reserve/growth and reset, with retention/handoff supported by
  the common tensor/resource ownership contract. Ordinary metadata inspection
  and retaining an OrtValue are not falsely reported as native compute.
- Advertise supported element types, layouts, memory/device contracts and the
  callbacks needed for each claimed capability. Reject inconsistent descriptors
  rather than advertising operations that cannot be invoked.
- Partial support remains useful: a sampler-only provider leaves cache behavior
  independent, and a compatible CPU-storage cache override may specialize only
  some operations under PreferProvider.
- Resolve one cache backend/device/storage contract for an entire chain
  execution. Shared prefill/decode caches do not switch backend between stages.
  Conflicting stage requirements fail clearly. Sampling may be selected per
  stage independently of the shared cache.
- Cache storage remains contiguous and tensor-backed with the validated M04
  layout, ownership, length/capacity and retained-view guarantees. A partial
  override cannot hide an incompatible private layout behind CPU fallback.
- Validate required capabilities before execution when they are known, and
  before each later requested operation. Absence of an unused optional
  capability does not itself invalidate otherwise supported operations.

### G.3 Fallback, movement and failure

- CPU fallback is permitted only for a missing capability, when the policy
  permits it and all operands/storage satisfy the CPU implementation's
  declared memory and layout contract.
- Do not transfer device data to CPU merely to make an otherwise incompatible
  fallback appear successful. A device/backend mismatch is unsupported.
- Existing, intentional CPU-mode Session I/O staging and stable CPU output
  snapshots remain supported. Record that movement separately; it is not
  evidence of native cache/sampler dispatch. Backend selection/fallback must
  not introduce undisclosed transfers.
- Once an advertised provider callback is invoked, propagate any failure,
  including NOT_IMPLEMENTED. Never retry it on CPU or report CPU success after
  a provider failure.
- Preserve the existing reset-after-failure contract for execution contexts.
  A failed provider operation that may have mutated standalone cache state
  also prevents further mutable use until successful reset. Validation failures
  before mutation do not pretend to be provider execution.
- Reset/provider errors must be returned or raised explicitly; failed recovery
  must not mark an invalidated object healthy.
- Callbacks preserve retained immutable values/views and cannot mutate their
  previously published storage through an unsafe alias. Cancellation remains
  an atomic boundary request, not kernel preemption.

### G.4 Built-in and shared-provider boundary

- Extend IExecutionProvider with optional discovery/hooks; unchanged providers
  advertise no overrides. No changes are required to deprecated EPs/backends.
- Route CPU behavior through the same validated selection/dispatch contract,
  reusing the existing CPU cache and greedy implementations rather than copying
  their algorithms.
- Wire the shared-provider bridge as necessary. Boundary-visible descriptors
  use opaque C handles, primitive metadata and function pointers, not in-tree
  graph/Tensor/C++ ownership types. Graph-specific validation stays in the
  appropriate host-side wrappers.
- M07 proves real built-in dispatch and callback invocation. CUDA may continue
  to advertise no native stateful capabilities until M08/M09; RequireProvider
  must then fail explicitly for those unavailable operations.

### G.5 Plugin registration and ABI

- Use an optional, versioned, size-tagged C descriptor registered through
  appended OrtEpApi entry points, as approved in Appendix A.5.
- Do not append mandatory callback fields to OrtEp or read beyond an older
  plugin's allocation. Old plugins register no descriptor and continue with
  their existing behavior.
- Copy registration metadata/configuration; define callback/context ownership
  and retain/release responsibilities. Active registrations are immutable while
  dependent stateful objects use them.
- Validate descriptor version/size, pointer/count pairs, capability/callback
  consistency and memory contracts. Failed registration or failed EP creation
  must not leak retained contexts/resources.
- Negotiate through existing API/version mechanisms. Keep shipped entries and
  signatures unchanged, append public functions/initializers only, and do not
  bump ORT_API_VERSION or introduce release-boundary markers.
- C callback errors use owned OrtStatus results with standard release rules;
  C++ exceptions must not cross the callback boundary.

### G.6 Lifetime and synchronous execution

- Retain provider contexts, allocators, factory/library dependencies and any
  callback-bearing resources for as long as dependent stateful objects need them.
- Existing caller-owned model Sessions must still outlive referencing loop/
  chain definitions and executions. Retained result/cache values may outlive
  those contexts/Sessions without dangling provider deleters or unloaded code.
- Unregistration must not unload callback code or release needed resources while
  live objects still depend on them. Release each owned resource exactly once.
- Retain the synchronous Step/Run observation contract. Do not claim native GPU
  stream implementations from CPU-only overrides; M08/M09 must supply their
  actual device and ordering guarantees.

### G.7 Observability and acceptance

- Query capabilities without executing the operation. Report selected policy,
  provider/device, operation and whether CPU was explicitly selected, an
  override ran, or a permitted missing-capability fallback occurred.
- Expose required movement separately from implementation selection. Provider
  callback/copy/allocation evidence must distinguish real dispatch from model
  EP registration, a fallback, an empty test or an uncalled callback.
- Tests cover default CPU, no overrides, independent and partial overrides,
  RequireProvider failures, compatible fallback, rejected device fallback,
  callback errors/reset, malformed registrations, release ordering and resource
  retention after context release.
- Plugin compatibility evidence includes a genuine older/shorter OrtEp contract,
  not only a current-size plugin that happens not to register overrides.
- Execute CPU and relevant CUDA-build tests, actual built-in/plugin callbacks,
  public C/C++ parity and pure-C boundary compilation. Measure 100% new
  executable host/glue lines without claiming unimplemented GPU coverage.

### G.8 Reviewable delivery boundaries

- M07a establishes native capability/selection contracts, real CPU routing,
  C/C++ selection/observation surfaces and built-in test-provider evidence.
- M07b adds plugin registration/adapters, retained plugin resources and legacy
  ABI/partial-capability test-plugin evidence, stacked on M07a.
- Each sub-PR is independently documented/tested and ready for human review.
  Macro M07 is complete only after both satisfy their CPU/CUDA and measured
  coverage gates. The detailed tasks/test plan is prepared after this appendix
  is approved and reviewed before implementation.

### G.9 M07a interface mapping and callback storage

The implemented native hook returns an optional OrtStatefulProviderDescriptor,
using only the public C types even in a shared-provider DLL. Its version-1
capabilities describe six cache operations and greedy sampling independently.
Initialization/growth callbacks return capacity-shaped storage containing
compact logical data, together with explicit logical length/capacity and
operation-buffer counters. Published buffers remain immutable. Share preserves
the allocator-owned source buffer; the runtime validates result layout/extents.

Callbacks may explicitly invoke the provided CPU reference once, only with
CPU-compatible operands. This is a reported provider delegation, not native GPU
work or an automatic retry. Request pointers/reference contexts are valid only
during the synchronous call. ORT retains the supplied allocator adapter with
returned storage, allowing CreateTensorAsOrtValue to retain its allocator safely.
Providers must not otherwise save a borrowed request or CPU-reference context.

Loop and component policy setters copy selections; standalone cache/sampler
setters resolve and retain resources from the supplied initialized Session.
SetConfig replaces the complete configuration and resets provider choices.
Unmodified component choices inherit their immutable loop definition.
The added C table entries and C++ wrappers do not change M06 record strides.

Dispatch metadata distinguishes the actual provider from the requested provider,
uses NOT_INVOKED before an operation runs, and keeps bounded per-operation
CPU/provider/delegation/failure counters. C++ copies both names. Missing required
capabilities and incompatible CPU fallback return explicit unsupported errors.

Separate execution-lifetime staging counters report successful CPU BindInput
request payloads (including overwritten bindings), CPU outputs observed after
Run, and successful CPU stable-snapshot copies. They survive Reset and are
readable after failure. They are not measured DMA bytes, do not imply a new
fallback transfer, and do not prove native CUDA cache/sampling.

This public-SDK example explicitly selects the registered CPU provider rather
than merely relying on the default CPU path. Run from a build output directory
containing the established testdata/mul_1.onnx fixture:

```cpp
#include "onnxruntime_cxx_api.h"

int main() {
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "stateful-provider-example");
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  Ort::Session session(env, ORT_TSTR("testdata/mul_1.onnx"), session_options);
  const auto capabilities = Ort::QueryStatefulProviderCapabilities(session, "CPUExecutionProvider");
  const OrtStatefulProviderSelection selection{
      sizeof(selection), ORT_STATEFUL_PROVIDER_REQUIRE, "CPUExecutionProvider",
      OrtMemoryInfoDeviceType_CPU, 0};
  const OrtStatefulSamplerConfig config{sizeof(config), ORT_STATEFUL_GREEDY, 0, 0};
  Ort::SamplerOptions sampler(config);
  sampler.SetProvider(selection, session);
  Ort::AllocatorWithDefaultOptions allocator;
  const int64_t shape[]{1, 3};
  auto logits = Ort::Value::CreateTensor<float>(allocator, shape, 2);
  auto* scores = logits.GetTensorMutableData<float>();
  scores[0] = 1.f;
  scores[1] = 4.f;
  scores[2] = 2.f;
  const auto token = sampler.Sample(logits);
  const auto dispatch = sampler.GetDispatchInfo();
  return capabilities.operations != 0 && token == 1 &&
                 dispatch.origin == ORT_STATEFUL_DISPATCH_PROVIDER &&
                 dispatch.provider_calls == 1 && dispatch.cpu_delegations == 1
             ? 0
             : 1;
}
```

### G.10 M07b plugin registration and retained resource owners

Plugin authors call OrtEpApi::RegisterStatefulProvider (or the C++ free wrapper)
once for the EP being returned, on the calling thread during synchronous
OrtEpFactory::CreateEp. Registration validates/copies the versioned descriptor
and retains its context. It never reads a new OrtEp field. Nested creation has
an independent transaction; failed, duplicate or mismatched creation cannot
publish stale registrations or leak retained contexts. No registration is
required for a legacy/no-override EP.

The native plugin adapter owns the committed registration. Provider resource
leases keep it and the EP alive, while strong factory/library owners defer
unloading after unregistration. Private device records use weak owners to avoid
cycles. Removing a library from discovery does not destroy callback-bearing
factories while live stateful values still depend on them.

Retained backend leases also keep the original Session logger and any custom
logging manager alive, so delayed provider/context teardown does not call a
dead logger. The ordinary OrtEnv and caller-owned callback-state lifetime
requirements remain in effect. Contexts must own independent callback/deleter
resources rather than borrowing a released model graph or request pointer.

The example plugin demonstrates no overrides, sampler-only, append-only and
full supported CPU capabilities, descriptor compatibility and injected errors.
Its genuine frozen 1.23 OrtEp prefix ends at the original
GetCompiledModelCompatibilityInfo slot. An inaccessible page immediately after
that allocation detects later-field reads; this is not a current-sized struct
with only its version number changed.

## Appendix H. M08 CUDA plugin KV cache contract

**Status:** Approved by the user on 2026-10-18 without changes. Scope is CUDA
plugin EP only, CUDA copy primitives where sufficient, GPU-resident cache
views/results, and an M08a/M08b split, with no additional specification input.
The detailed execution plan is approved without changes. M08a is complete in
PR #20. M08b, separately authorized on 2026-10-19, is complete in PR #21.
Both CUDA-plugin-only sub-milestones are independently verified and ready for review.

### H.1 Scope and preserved behavior

- Implement real device-resident KV management for the CUDA plugin EP only,
  through M07's existing optional registration and per-operation dispatch.
- Legacy/shared CUDA is explicitly outside this milestone's implementation
  scope. It must not advertise native KV capabilities merely because the plugin
  implementation exists. Required unsupported choices continue to fail clearly.
- Preserve the explicit CPU default and independent cache/sampler selection.
  CUDA greedy sampling remains M09; M08 does not advertise that capability.
- Preserve float32, float16 and bfloat16 cache support, contiguous tensor-backed
  storage, explicit sequence axes and compact logical packing. One selected CUDA
  device is used per execution/shared cache; arbitrary multi-device migration,
  cross-request sharing, paged/opaque layouts and dtype conversion remain out
  of scope.

### H.2 Device cache operations

- Advertise only CUDA-plugin operations actually implemented and tested:
  initialize/allocation, allocator-owned full-output sharing, full copy, delta
  append, reserve/growth and logical reset.
- Allocate GPU storage through the compatible ORT/plugin allocator, retaining
  its device metadata/callback resources. Logical sequence length and allocation
  capacity remain distinct; initially empty storage does not expose uninitialized
  capacity bytes as logical data.
- Share retains the owned GPU model output at its actual extent without an
  unnecessary data copy. Do not infer model input/output aliasing or introduce
  in-place execution that its kernel/graph contract does not permit.
- Copy/append/reserve preserve required contents and fixed dimensions. Append
  supports the existing explicit sequence-axis layout rather than assuming the
  sequence dimension is last. Use checked shape, offset, pitch and byte arithmetic.
- Copy/growth/update must not overwrite storage visible through retained views.
  Reset clears logical length and may retain valid capacity; existing snapshots
  remain unchanged across update/reserve/reset/release.
- Preserve explicit errors for invalid shape/dtype/device, shrinkage, overflow,
  allocation failures, CUDA failures and invalidated state. Never retry a failed
  CUDA callback on CPU or return a successful empty cache.

### H.3 CUDA primitives and execution ordering

- Prefer existing CUDA allocation/copy facilities, including async/2D device
  copy primitives where appropriate, over unnecessary custom device kernels.
  Validate supported copy geometry and handle relevant API limits explicitly.
- Respect ORT/plugin allocator and stream abstractions. Device work must be
  ordered after its producer and before subsequent model consumption,
  observation or release, including nondefault streams and distinct Sessions
  sharing a cache on the same GPU.
- Preserve the synchronous callback/Step/Run contract: async copies may be
  enqueued internally, but published results are ready when the corresponding
  synchronous API returns. Do not introduce a serving/async-streaming API.
- Use appropriate stream/event completion rather than masking ordering defects
  with unrequested global device-wide synchronization. Restore caller device
  state when a scoped device change is required.
- Caller-owned external streams and allocator/logging state retain their
  documented lifetime obligations; internally owned resources remain retained
  until dependent work and values no longer need them.

### H.4 Loop/chain and output integration

- Remove M07's device-cache integration rejection only for the implemented,
  explicitly selected CUDA plugin cache path. Device sampling remains explicitly
  unavailable until M09.
- Resolve a compatible GPU allocator and bind managed past/present cache ports
  to the selected GPU. Keep cache state and prefill/decode handoff device-resident,
  rather than staging every cache output through CPU to invoke the manager.
- Preserve one compatible cache backend/device contract across the chain,
  including different initialized CUDA plugin Sessions on that GPU. Reject
  conflicting or unsupported device/cache-port configurations instead of silently
  introducing an arbitrary heterogeneous pipeline.
- Under explicit GPU-cache selection, managed cache values, retained cache
  views and cache tensors exposed in step/component results stay on GPU.
  Their device placement is observable, and CPU inspection requires an explicit
  caller copy; a raw tensor-data getter does not perform a hidden readback.
- Ordinary non-cache selected outputs retain their stable CPU behavior.
  In particular, CPU logits/token sampling remains supported and observable
  while cache management is GPU-native. CPU-default execution is unchanged.
- Unselected cache outputs retained for downstream components must not trigger
  hidden cache readback merely because chain bookkeeping retains model outputs.

### H.5 Lifetime, failure and cancellation

- Retain GPU allocations, callback contexts, allocator adapters, provider/
  factory/library and required stream/logger resources for dependent values.
  M07's ownership and delayed-unregistration guarantees continue to apply.
- The size-tagged cache request gains a trailing `retain_value` callback for
  retaining borrowed input/storage handles without data copies or native core
  types. Plugins check the request size before accessing the extension; existing
  request fields and the public API table/version remain unchanged.
- Retained GPU cache snapshots remain usable after manager/context reset and
  release, and after the originating Session is released, subject to ordinary
  caller-owned OrtEnv/resource lifetime rules.
- Operations complete before resources needed by queued work can be released.
  Failure after possible mutation keeps the relevant manager/execution invalid
  until successful reset; failed reset cannot mark it healthy.
- Cancellation remains a boundary request. Ensure partially completed model/
  device work is safely ordered/cleaned up; no kernel-preemption guarantee is added.

### H.6 Observable proof and coverage

- Query and assert actual CUDA plugin cache capabilities and invocation, GPU
  placement/device id, logical length/capacity, shared-buffer identity where
  applicable, and device-copy/allocation evidence. CUDA model registration alone
  is not proof of native cache dispatch.
- Native GPU cache operations have no undisclosed CPU fallback/reference
  delegation. Existing CPU logits/ordinary-output staging is recorded separately
  and is not reported as native GPU sampling or cache movement.
- Compare GPU contents with the CPU reference across supported dtypes, sequence
  axes, zero/growing lengths and retained snapshots using explicit test readback.
- Test CPU regressions, real CUDA-plugin standalone and loop/chain execution,
  default/nondefault and cross-Session stream ordering, device mismatch,
  controlled allocation/provider errors, cancellation, release ordering and
  sanitizer correctness. Do not exhaust a shared GPU merely to provoke OOM.
- Measure 100% new executable host/glue lines in the actual exercised CUDA
  plugin/core paths using matching instrumented artifacts and fresh profiles.
  The plugin build is separate from the legacy/shared CUDA build; one is not
  validation evidence for the other.
- If standard CUDA APIs suffice and no device functions are added, the new
  device-line denominator is zero/not applicable, not a claimed 100% device
  coverage result. Any new device code actually introduced must satisfy the
  measured 100% device-line gate; lack of that evidence blocks completion.

### H.7 Reviewable delivery boundaries

- M08a adds the CUDA plugin cache backend, registration, standalone conformance,
  resource/stream correctness and measured new-line validation.
- M08b adds GPU cache allocator/port/result wiring through loops/chains,
  prefill/decode sharing, C/C++ parity and end-to-end lifecycle/ordering evidence,
  stacked on independently verified M08a.
- Each sub-PR is independently documented/tested and ready for review. Macro
  M08 remains incomplete until both pass their CPU/CUDA-plugin/coverage gates.
  Prepare the detailed execution plan after this appendix is approved, and
  review that plan before implementation.

### H.8 Standalone CUDA-plugin SDK example (M08a)

M08a enables standalone caches; M08b adds device-cache loop/chain integration.
The plugin uses checked, compact per-row CUDA copies, avoiding 2D-copy pitch
limits without new device functions. Operations finish on the default or
caller-configured stream before returning; external inputs must already be
ready. No CPU reference delegation or GPU sampler is advertised.
Capacity bytes and nonempty allocation pointers are validated before copying.
CUDA errors are reported and consumed at the callback boundary rather than
silently poisoning subsequent inference after an explicit cache failure.

This C++17 example takes the CUDA plugin library and an ONNX model (for example
`onnxruntime/test/testdata/mul_1.onnx`). Only public SDK/CUDA headers are needed.
The allocator and environment outlive every retained value. A separate
allocator Session stays alive because the existing public Session-allocator API
requires it; only the Session used to select the cache backend is released.
The explicit readback follows manager/backend-Session release and library
unregistration. This example does not claim final DLL unload while the
caller-owned allocator Session still needs the library.

```cpp
#include <array>
#include <filesystem>
#include <stdexcept>
#include <string_view>
#include <cuda_runtime_api.h>
#include "onnxruntime_cxx_api.h"

static void CheckCuda(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

int main(int argc, char** argv) {
  if (argc != 3) return 2;
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "cuda-cache-example");
  env.RegisterExecutionProviderLibrary("stateful-cuda", std::filesystem::path(argv[1]).native());
  Ort::ConstEpDevice device(nullptr);
  for (auto candidate : env.GetEpDevices()) {
    if (std::string_view(candidate.EpName()) == "CUDAExecutionProvider" &&
        candidate.GetMemoryInfo(OrtDeviceMemoryType_DEFAULT).GetDeviceId() == 0) {
      device = candidate;
      break;
    }
  }
  if (!device) throw std::runtime_error("CUDA plugin device 0 unavailable");
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  session_options.AppendExecutionProvider_V2(env, {device}, Ort::KeyValuePairs{});
  Ort::Session allocator_session(env, std::filesystem::path(argv[2]).native().c_str(), session_options);
  Ort::Allocator allocator(allocator_session, device.GetMemoryInfo(OrtDeviceMemoryType_DEFAULT));
  Ort::Session session(env, std::filesystem::path(argv[2]).native().c_str(), session_options);
  const int64_t empty_shape[]{2, 0, 2};
  const OrtStatefulKVCacheSpec spec{
      "key", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, empty_shape, 3, 1, 0};
  const OrtStatefulKVCacheConfig config{sizeof(config), &spec, 1};
  const OrtStatefulProviderSelection selection{
      sizeof(selection), ORT_STATEFUL_PROVIDER_REQUIRE, "CUDAExecutionProvider",
      OrtMemoryInfoDeviceType_GPU, 0};
  Ort::KVCacheOptions options(config);
  options.SetProvider(selection, session);
  Ort::KVCacheManager cache(options, allocator);
  const int64_t delta_shape[]{2, 1, 2};
  auto delta = Ort::Value::CreateTensor<float>(allocator, delta_shape, 3);
  const std::array<float, 4> input{1, 2, 3, 4};
  CheckCuda(cudaMemcpy(delta.GetTensorMutableData<float>(), input.data(),
                       sizeof(input), cudaMemcpyHostToDevice));
  cache.Update("key", delta, ORT_KV_CACHE_COPY);
  cache.Update("key", delta, ORT_KV_CACHE_APPEND);
  auto retained = cache.GetView("key").GetInfo();
  const auto dispatch = cache.GetDispatchInfo(ORT_STATEFUL_CACHE_APPEND);
  if (dispatch.provider_calls != 1 || dispatch.cpu_calls || dispatch.cpu_delegations) return 3;
  cache.Reset();
  cache = Ort::KVCacheManager(nullptr);
  options = Ort::KVCacheOptions(nullptr);
  delta = Ort::Value(nullptr);
  session = Ort::Session(nullptr);
  env.UnregisterExecutionProviderLibrary("stateful-cuda");
  std::array<float, 8> output{};
  CheckCuda(cudaMemcpy(output.data(), retained.value.GetTensorRawData(),
                       sizeof(output), cudaMemcpyDeviceToHost));
  const std::array<float, 8> expected{1, 2, 1, 2, 3, 4, 3, 4};
  return retained.length == 2 && output == expected ? 0 : 4;
}
```

### H.9 Device-resident loop/chain integration (M08b)

- Select `CUDAExecutionProvider` with GPU memory and an explicit compatible
  device (or the provider's unambiguous device). CPU remains the default.
- Resolve cache allocation from the initialized Session's actual GPU allocator,
  including chains whose first component is CPU-only. Verify all declared cache
  producers/consumers execute on the selected GPU; reject mismatches instead of
  inserting an implicit cache transfer.
- Bind managed cache outputs directly to that GPU and retain their allocator/
  provider lease. Ordinary outputs and logits keep CPU snapshots; CPU sampling
  is reported separately. No GPU sampling is added.
- Follow declared managed-cache output, component-output and named-state edges.
  A state output writing a shared cache slot is device-bound even when its
  component has no managed past/present pair. Feedback and finite loop inputs
  routed from device caches keep their device values and validate placement.
  Arbitrary external/initial-state chain inputs retain the host-accessible
  contract; this milestone does not infer a cache from a tensor's shape or
  silently treat unrelated device inputs as a declared cache edge.
- Synchronous Run cleans up/synchronizes its actual execution streams before
  publishing values. The explicit device-cache path does not additionally call
  IOBinding's provider-wide Sync, which would wait on unrelated CUDA work.
  Independent caller streams/allocators keep their ordinary readiness/lifetime
  obligations.
- Retained GPU result/state tensors survive execution/reset and originating
  Session release. Definitions/executions still require their caller Sessions
  to remain alive. GPU tensor-data getters never perform implicit readback.

## Appendix I. M09 CUDA plugin greedy sampling contract

**Status:** Approved by the user without changes on 2026-10-19, following the
completed interview.
The user chose CUDA plugin EP only, GPU-resident exposed logits under explicit
GPU sampling, token/status-only readback, and an M09a/M09b delivery split.
Additional specification inputs: None. The user approved the detailed plan
without changes and initially authorized M09a only. M09a is verified and published
ready for review in PR #22. M09b, separately authorized on 2026-10-19, is verified
and published ready for review in PR #23. Both parts of macro M09 are complete.

### I.1 Scope and preserved behavior

- Implement real GPU greedy sampling for the CUDA plugin EP only through the
  existing optional stateful provider descriptor and independent sampler policy.
  Legacy/shared CUDA does not acquire native sampling capabilities.
- CPU remains the default sampler. Existing CPU greedy, GPU-cache/CPU-sampler
  execution, ordinary stateless Session behavior and explicit loop exits remain
  unchanged. Sampling algorithms beyond greedy remain out of scope.
- Preserve independently selected cache/sampler policies. GPU sampling does not
  require GPU cache selection; any movement resulting from an explicitly chosen
  CPU cache remains distinct from sampling and is not called GPU-native caching.
- Advertise the greedy operation and supported dtype/device metadata only when
  implemented and verified. Require/Prefer/CPU retain M07's dispatch rules:
  compatible fallback only for a missing capability, no retry after an advertised
  callback fails, and no undisclosed device transfer.
- Initially support contiguous float32/float16 logits shaped `[1,V]` or
  `[1,S,V]`, with nonempty dimensions and the existing optional row position.
  Bfloat16 sampling, batched requests, strided layouts, arbitrary cross-device
  pipelines, asynchronous/serving APIs and device-resident token-feed generation
  are not introduced by M09.

### I.2 Exact greedy semantics

- Return an int64 token ID in `[0,V)`. Omitted position selects the last row;
  rank-two logits have one row and only position zero is valid.
- Match CPU comparisons exactly for supported inputs. The lowest token ID wins
  ties, including ties spanning reduction/launch boundaries. Positive/negative
  infinity follow numeric ordering; an all-negative-infinity row selects zero.
- Reject any NaN anywhere in the complete supplied logits tensor, including
  unselected rows. Row selection must not narrow the NaN-validation domain.
- Validate rank/batch/dtype/position/contiguity, dimensions, device placement,
  buffers and checked size/index/scratch arithmetic before accessing device
  data. Reject invalid or unsupported inputs with an explicit error.
- GPU code performs both selection and whole-tensor NaN validation. Do not
  silently copy logits to CPU, call the CPU reference, perform host argmax or
  replace a failed GPU result with a default token. Failed calls preserve public
  out-parameters and follow existing invalidation/error contracts.

### I.3 Execution, resources and observable movement

- Use compatible ORT/plugin allocator and device/stream abstractions for
  scratch/results. Reuse existing suitable CUDA reduction utilities where they
  fully satisfy these semantics; do not duplicate kernels or introduce complex
  launch tiers without a correctness or measured need.
- Order work on the appropriate selected device/stream. Complete kernel work
  and required result transfers before the synchronous callback/API returns;
  do not add an unrequested device-wide wait to establish completion.
- Restore caller device state and retain callback/library/logger/allocator/
  stream resources for all work and dependent values. Caller-owned external
  streams, allocators and inputs retain their normal lifetime/readiness duties.
- The synchronous API may read back the resulting token ID and compact
  validation/status information. It must not read back the full logits tensor
  as part of sampling.
- The size-tagged greedy request has an appended borrowed scratch-allocator
  field. Native dispatch resolves and retains a compatible initialized-Session
  allocator, provides its C adapter only for the synchronous callback, and
  destroys scratch before returning. Existing request prefixes and public API
  tables/version remain unchanged; generation code does not import native
  Session allocator-adapter implementation headers.
- Existing host token history, EOS/budget checks and generated input-feed
  construction remain unchanged. Their scalar observations and CPU input
  payloads must not be misrepresented as an entirely device-only generation
  pipeline.
- Expose actual provider dispatch and zero CPU-reference delegation for native
  sampling. Record token/status transfer and existing staging evidence
  separately; distinguish measured transfers from logical payload counters.
  Record representative latency/copy measurements without performance thresholds.

### I.4 Loop/chain integration and output ownership

- M09b removes the device-sampler integration rejection only for the implemented,
  explicitly selected CUDA plugin path. Resolve compatible Session resources and
  validate actual logits producer placement instead of assuming CUDA registration
  alone proves GPU logits or silently staging a CPU-produced logits tensor.
- Under explicit GPU sampling, bind logits to GPU and keep selected/component
  logits results GPU-resident, including outputs retained only for downstream
  chain bookkeeping. Retain immutable logits snapshots with their required
  allocator/provider resources. CPU inspection requires an explicit caller copy;
  raw tensor-data getters never perform an implicit readback.
- Other ordinary non-cache/non-logits outputs retain stable CPU behavior.
  CPU-default sampling keeps its existing CPU logits snapshots. M08's explicitly
  selected GPU cache outputs/state/handoff remain GPU-resident.
- Support the existing standalone prefill and shared prefill/decode chain
  contracts with matching explicit budgets, EndOfInputs, EOS and iteration limits.
  Device sampling, cache sharing and per-stage sampler selection must agree on
  actual compatible producer/device placement; unsupported combinations reject
  rather than introduce arbitrary migration.
- Preserve zero-budget validation with no model or sampler invocation, first-token
  handoff, context isolation, cancellation boundaries, reset and retained values.
  Keep scalar result observation synchronized before history/exits consume it.
- Model/provider/launch/transfer failures remain explicit. Potentially advanced
  mutable execution requires reset; failed reset cannot mark it healthy. Do not
  retry a failed GPU sampler on CPU.

### I.5 Proof, coverage and reviewable delivery

- Prove actual CUDA plugin greedy callbacks and GPU execution, correct physical
  device placement, token/status-only observation and no host argmax/logits
  fallback. CPU-equivalent values alone do not prove the intended path ran.
- Test float32/float16, supported shapes/positions, all-infinity rows, ties,
  NaNs in selected/unselected rows, zero/one/multiple token budgets, and sizes
  around relevant reduction/launch boundaries. Include nondefault and distinct
  compatible Session streams, error/resource lifetime cases and retained logits.
- Compare standalone and complete tiny prefill/decode results against CPU
  references through native and public C/C++ interfaces. Rerun CPU, M08 GPU cache,
  provider-registration/policy and legacy unsupported-capability regressions.
- Measure 100% of every new executable host/glue line and every new executable
  device line using the actual exercised artifacts. Establish a credible device
  execution-to-source measurement method before claiming the device gate.
  Tool installation, line information, kernel launch success, host coverage and
  Compute Sanitizer are not substitutes for device-line coverage.
- Reuse of existing CUDA code must be identified explicitly. Do not automatically
  carry forward M08's copy-only zero-device-code result, exclude new device
  helpers/template logic from the denominator, or waive measurement because the
  tooling is unavailable. Unmeasured new device code blocks completion.
- Run applicable memory/order/lifetime checks under Compute Sanitizer. Disclose
  exact selectors and any instrumentation limitation separately from functional
  correctness and coverage; do not silently turn failures into skips.
- M09a supplies the CUDA plugin sampler backend, standalone C/C++ conformance,
  resource/stream correctness and measured host/device coverage, stacked on
  independently verified M08b PR #21. It does not claim loop/chain GPU sampling.
- M09b supplies GPU logits binding/retention, loop/chain generation integration,
  public/native parity and end-to-end evidence, stacked on verified M09a.
  Each sub-PR is independently documented, tested and ready for human review.
  Macro M09 is complete only after both pass their gates.
- After approval of this appendix, prepare and separately review the detailed
  execution plan and authorization boundary before runtime implementation.

### I.6 Standalone CUDA-plugin SDK example (M09a)

M09a exposes standalone GPU greedy, not GPU-sampled loops/chains (M09b).
This C++17 example takes the CUDA plugin library and an ONNX model such as
`onnxruntime/test/testdata/mul_1.onnx`. The caller explicitly uploads the logits;
sampling reads back only a compact token/status result. A separate caller-owned
allocator Session stays alive through all input/scratch use, as required by
the existing public allocator contract. Final DLL unload while that Session
still needs it is not claimed.

```cpp
#include <array>
#include <filesystem>
#include <stdexcept>
#include <string_view>
#include <cuda_runtime_api.h>
#include "onnxruntime_cxx_api.h"

static void CheckCuda(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

int main(int argc, char** argv) {
  if (argc != 3) return 2;
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "cuda-greedy-example");
  env.RegisterExecutionProviderLibrary("stateful-cuda", std::filesystem::path(argv[1]).native());
  Ort::ConstEpDevice device(nullptr);
  for (auto candidate : env.GetEpDevices()) {
    if (std::string_view(candidate.EpName()) == "CUDAExecutionProvider" &&
        candidate.GetMemoryInfo(OrtDeviceMemoryType_DEFAULT).GetDeviceId() == 0) {
      device = candidate;
      break;
    }
  }
  if (!device) throw std::runtime_error("CUDA plugin device 0 unavailable");
  Ort::SessionOptions options;
  options.SetIntraOpNumThreads(1);
  options.AppendExecutionProvider_V2(env, {device}, Ort::KeyValuePairs{});
  const auto model = std::filesystem::path(argv[2]).native();
  Ort::Session allocator_session(env, model.c_str(), options);
  Ort::Allocator allocator(allocator_session, device.GetMemoryInfo(OrtDeviceMemoryType_DEFAULT));
  Ort::Session session(env, model.c_str(), options);
  const int64_t shape[]{1, 3};
  const std::array<float, 3> values{1.f, 8.f, 8.f};
  auto logits = Ort::Value::CreateTensor<float>(allocator, shape, 2);
  CheckCuda(cudaMemcpy(logits.GetTensorMutableData<float>(), values.data(),
                       sizeof(values), cudaMemcpyHostToDevice));
  const OrtStatefulSamplerConfig config{sizeof(config), ORT_STATEFUL_GREEDY, 0, 0};
  const OrtStatefulProviderSelection selection{
      sizeof(selection), ORT_STATEFUL_PROVIDER_REQUIRE, "CUDAExecutionProvider",
      OrtMemoryInfoDeviceType_GPU, 0};
  Ort::SamplerOptions sampler(config);
  sampler.SetProvider(selection, session);
  if (sampler.Sample(logits) != 1) return 3;
  session = Ort::Session(nullptr);
  env.UnregisterExecutionProviderLibrary("stateful-cuda");
  if (sampler.Sample(logits) != 1) return 4;
  const auto dispatch = sampler.GetDispatchInfo();
  return dispatch.provider_calls == 2 && dispatch.cpu_calls == 0 &&
                 dispatch.cpu_delegations == 0
             ? 0 : 5;
}
```

### I.7 Device-resident generation SDK example (M09b)

GPU cache and sampler choices remain independent. Each explicitly sampled GPU
logits producer must execute on its selected CUDA plugin device; ordinary outputs
and CPU-sampled logits keep CPU snapshots. Declared GPU component/state inputs
retain their origin's device. Conflicting CPU/GPU output roles, mismatched devices
and GPU-feedback inputs with CPU-feedback outputs reject before invocation.
Arbitrary external/initial-state chain inputs retain their host-only contract.

This C++17 public-SDK example accepts the plugin library and one dynamic decoder
model used for both phases. Its ports match the reusable tiny fixture:
`token:int64[1,S]`, `past_key:float[1,1,P,1]`, `logits:float[1,S,4]`,
`present_key:float[1,1,P+S,1]`, with the fixture's deterministic score table.
The expected generated IDs for prompt `[0,1]` are `[2,3,0]`. A retained GPU
logits value survives all definitions/executions/Sessions and library discovery
release; the caller-owned Env remains alive. CPU inspection is explicit.

```cpp
#include <algorithm>
#include <array>
#include <filesystem>
#include <stdexcept>
#include <string_view>
#include <cuda_runtime_api.h>
#include "onnxruntime_cxx_api.h"

static Ort::SessionLoop MakeLoop(const Ort::Session& session, bool decode) {
  const int64_t shape[]{1, 1, 0, 1};
  const OrtStatefulKVCacheBinding cache{
      {"key", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, shape, 4, 2, 0},
      "past_key", "present_key", ORT_KV_CACHE_SHARE};
  const OrtStatefulGenerationConfig generation{
      sizeof(generation), decode ? ORT_STATEFUL_DECODE : ORT_STATEFUL_PREFILL,
      "token", "logits", nullptr, nullptr, 0, 0};
  const char* finite[]{"token"};
  const char* outputs[]{"logits", "present_key"};
  const OrtStatefulExitCondition exits[]{
      {ORT_STATEFUL_MAX_GENERATED_TOKENS, 3, nullptr, nullptr, 0},
      {ORT_STATEFUL_END_OF_INPUTS, 0, nullptr, nullptr, 0}};
  OrtStatefulLoopConfig config{};
  config.struct_size = sizeof(config);
  config.finite_input_names = decode ? nullptr : finite;
  config.num_finite_inputs = decode ? 0 : 1;
  config.output_names = outputs;
  config.num_outputs = 2;
  config.exit_conditions = exits;
  config.num_exit_conditions = decode ? 1 : 2;
  config.kv_caches = &cache;
  config.num_kv_caches = 1;
  config.generation = &generation;
  Ort::SessionLoopOptions options(config);
  const OrtStatefulProviderSelection gpu{
      sizeof(gpu), ORT_STATEFUL_PROVIDER_REQUIRE, "CUDAExecutionProvider",
      OrtMemoryInfoDeviceType_GPU, 0};
  options.SetProvider(ORT_STATEFUL_FEATURE_CACHE, gpu);
  options.SetProvider(ORT_STATEFUL_FEATURE_SAMPLER, gpu);
  return Ort::SessionLoop(session, options);
}

int main(int argc, char** argv) {
  if (argc != 3) return 2;
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "gpu-generation");
  env.RegisterExecutionProviderLibrary("gpu-generation", std::filesystem::path(argv[1]).native());
  Ort::ConstEpDevice device(nullptr);
  for (auto candidate : env.GetEpDevices()) {
    if (std::string_view(candidate.EpName()) == "CUDAExecutionProvider" &&
        candidate.GetMemoryInfo(OrtDeviceMemoryType_DEFAULT).GetDeviceId() == 0) {
      device = candidate;
      break;
    }
  }
  if (!device) throw std::runtime_error("CUDA plugin device 0 unavailable");
  Ort::SessionOptions options;
  options.SetIntraOpNumThreads(1).SetGraphOptimizationLevel(ORT_DISABLE_ALL);
  options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
  options.AppendExecutionProvider_V2(env, {device}, Ort::KeyValuePairs{});
  const auto path = std::filesystem::path(argv[2]).native();
  Ort::Session prefill(env, path.c_str(), options), decode(env, path.c_str(), options);
  auto first = MakeLoop(prefill, false), second = MakeLoop(decode, true);
  const OrtStatefulInputBinding prompt_binding{
      "token", {ORT_STATEFUL_EXTERNAL_INPUT, "prompt", 0}};
  const OrtStatefulChainComponent components[]{
      {"prefill", nullptr, first, &prompt_binding, 1, nullptr, 0, nullptr, 0},
      {"decode", nullptr, second, nullptr, 0, nullptr, 0, nullptr, 0}};
  const OrtStatefulChainOutput ports[]{
      {"tokens", 0, nullptr, ORT_STATEFUL_GENERATED_TOKEN_IDS},
      {"logits", 1, "logits", ORT_STATEFUL_MODEL_OUTPUT}};
  const OrtStatefulChainConfig config{sizeof(config), components, 2, ports, 2};
  Ort::SessionChainOptions chain_options(config);
  Ort::SessionChain chain(chain_options);
  Ort::AllocatorWithDefaultOptions cpu;
  const int64_t input_shape[]{1, 2};
  auto prompt = Ort::Value::CreateTensor<int64_t>(cpu, input_shape, 2);
  prompt.GetTensorMutableData<int64_t>()[0] = 0;
  prompt.GetTensorMutableData<int64_t>()[1] = 1;
  const OrtStatefulNamedValue input{"prompt", prompt};
  OrtStatefulInputs inputs{};
  inputs.struct_size = sizeof(inputs);
  inputs.inputs = &input;
  inputs.num_inputs = 1;
  auto execution = chain.CreateExecution(inputs);
  auto result = execution.Run();
  Ort::StatefulValue retained(nullptr);
  bool saw_tokens = false;
  for (const auto& value : result.GetOutputs()) {
    if (value.name == "tokens") {
      saw_tokens = true;
      const std::array<int64_t, 3> expected{2, 3, 0};
      if (value.value.GetTensorTypeAndShapeInfo().GetElementCount() != 3 ||
          !std::equal(expected.begin(), expected.end(), value.value.GetTensorData<int64_t>())) return 3;
    }
    if (value.name == "logits") retained = value.value;
  }
  if (!saw_tokens || !retained ||
      retained.GetTensorMemoryInfo().GetDeviceType() != OrtMemoryInfoDeviceType_GPU) return 4;
  execution.Reset();
  result = Ort::StatefulStepResult(nullptr);
  execution = Ort::StatefulExecution(nullptr);
  chain = Ort::SessionChain(nullptr);
  chain_options = Ort::SessionChainOptions(nullptr);
  first = Ort::SessionLoop(nullptr);
  second = Ort::SessionLoop(nullptr);
  prefill = Ort::Session(nullptr);
  decode = Ort::Session(nullptr);
  env.UnregisterExecutionProviderLibrary("gpu-generation");
  std::array<float, 4> logits{};
  if (retained.GetTensorTypeAndShapeInfo().GetElementCount() != logits.size()) return 5;
  if (cudaMemcpy(logits.data(), retained.GetTensorRawData(), sizeof(logits),
                 cudaMemcpyDeviceToHost) != cudaSuccess) return 6;
  return logits[0] > logits[1] ? 0 : 7;
}
```

## Appendix J. M10 standalone package chains (approved)

**Interview completed:** 2026-10-19. The user approved the revised candidate
manifest and confirmed there were no additional requirements before this draft.
**Specification approved:** 2026-10-19, for detailed execution-plan preparation.
This appendix refines PKG-1 through PKG-7; it does not authorize implementation.
Earlier approved runtime/API contracts remain unchanged by M10.

### J.1 Scope and consumer boundary

- Add optional named, ordered linear chains to the standalone Model Package
  format. Reuse existing components, variants, model references and shared assets.
- Expose chain inspection and authoring as JSON only. Do not add typed chain/step
  info structs, ORT handles, ONNX dependencies or execution-provider dependencies
  to the standalone library.
- Common structure consists of chain names, ordered steps, step IDs and component
  references. All execution configuration and tensor/state wiring belong inside
  consumer namespaces under `executor_info`, including `executor_info.ort`.
- M10 validates common structure and component references. It preserves consumer
  payloads without interpreting execution kinds, tensor names, upstream-output
  references, exits, sampling, cache policies or provider selection.
- M11 validates and interprets the ORT payload and constructs executable chains.
  Package-driven execution always requires an explicit chain name, even when
  only one chain exists. There is no implicit/default chain.
- Chains are declarative ordered component lists, not executable state machines.
  No common transition language, jumps, embedded scripts or custom predicates
  are introduced. M11 must reject unsupported executable ORT configurations;
  standalone preservation of an opaque payload is not execution approval.

### J.2 Schema version and compatibility

- The new optional `chains` field is introduced in schema `"1.1"`. Update the
  parser's highest known minor and the version stamped by `ModelPackage_New`
  together. The supported major remains 1.
- Existing 1.0 packages remain readable with unchanged component/variant/path
  semantics. A 1.0 package without chains does not gain a chain or get upgraded
  merely by opening it, inspecting it, or committing unrelated existing data.
- A manifest that declares `chains` must declare minor 1 or newer. Reject
  `chains` under 1.0 as a schema error, including in permissive parsing mode;
  this prevents a known feature from contradicting its declared schema version.
- Successful chain-setting mutations on a 1.0 package upgrade it to `"1.1"`.
  Never downgrade a package with a newer minor. Removing chains does not
  automatically downgrade the schema.
- Preserve existing unsupported-major errors and acceptance of future minors.
  A minor newer than this build knows relaxes unknown-field strictness as today;
  known fields must still have valid types and references.
- For known minors, honor the existing strict-unknown-field option in the common
  manifest, chain and step envelopes. With strictness disabled, preserve unknown
  fields through JSON inspection, unrelated mutation and commit.
- Consumer payload contents and `additional_metadata` remain opaque in strict
  and permissive modes. Their private fields are not common-schema unknown keys.
- An older 1.0 reader can already accept a 1.1 package by its future-minor rule
  and preserve unrecognized chain data. Do not claim it understands chains or
  will execute them, or that all old strict readers reject 1.1.

### J.3 Common JSON structure

`manifest.chains`, when present, is an object mapping nonempty UTF-8 names to
inline chain objects. An absent map or an empty map declares no chains.
`null`, arrays, strings and scalar values are not alternate forms.

| Scope | Field | Requirement |
| --- | --- | --- |
| Chain | `steps` | Required nonempty array; its order is execution order. |
| Chain | `executor_info` | Optional object mapping consumer namespaces to inline JSON objects. |
| Chain | `additional_metadata` | Optional opaque JSON value. |
| Step | `id` | Required nonempty UTF-8 string; unique within its chain. |
| Step | `component` | Required nonempty UTF-8 string naming an existing package component. |
| Step | `executor_info` | Optional object mapping consumer namespaces to inline JSON objects. |
| Step | `additional_metadata` | Optional opaque JSON value. |

Chain names, step IDs, component references and consumer namespace keys cannot
contain embedded NUL characters. Namespace keys must be nonempty. Identifiers
are not paths and do not cause filesystem access.

- The same component may appear in multiple steps or chains. Step IDs need only
  be unique within their own chain.
- A component reference targets the component map key, not its descriptive
  `component_name`. Existing placeholder components with empty variants remain
  supported; executable variant availability is checked by the consumer.
- An absent consumer namespace means no configuration for that consumer. It does
  not imply Session execution, a default sampler or default loop exits.
- New chain/step bodies and their namespace payloads are inline only in M10.
  External chain files and string-valued chain/step namespace payloads are not
  introduced. Existing external components and variant executor-info files keep
  their existing support and resolution rules.
- Unknown fields inside an accepted common envelope must survive round trips
  when permissive parsing applies. Array order must survive all round trips.
  Whitespace and object-key order are not execution semantics.

### J.4 Reviewed decoder-only example and ORT semantics

The reviewed example contains `decoder_prefill` and `decoder_decode`, whose CPU
variants reference `model1.onnx` and `model2.onnx`, respectively. The model files
resolve relative to their variant directories using existing ORT package rules.
There is no encoder component or encoder chain.

This self-contained example fixes the candidate field spelling. Its model
signatures, dimensions, token IDs and capacity are illustrative, not a claim
that unspecified model files form a runnable package.

```json
{
  "schema_version": "1.1",
  "package_name": "session-and-loop-example",
  "layout": "portable",
  "components": {
    "decoder_prefill": {
      "variants": {
        "cpu": {
          "variant_directory": "decoder_prefill/cpu",
          "ep": "CPUExecutionProvider",
          "device": "cpu",
          "executor_info": {
            "ort": { "model_file": "model1.onnx" }
          }
        }
      }
    },
    "decoder_decode": {
      "variants": {
        "cpu": {
          "variant_directory": "decoder_decode/cpu",
          "ep": "CPUExecutionProvider",
          "device": "cpu",
          "executor_info": {
            "ort": { "model_file": "model2.onnx" }
          }
        }
      }
    }
  },
  "chains": {
    "generate": {
      "steps": [
        {
          "id": "prefill",
          "component": "decoder_prefill",
          "executor_info": {
            "ort": {
              "kind": "session_loop",
              "inputs": {
                "input_ids": { "kind": "external_sequence", "name": "prompt_batches" }
              },
              "loop": {
                "finite_input_names": ["input_ids"],
                "nonempty_input": "input_ids",
                "exit_conditions": [
                  { "kind": "end_of_inputs" }
                ],
                "cache_provider": { "mode": "cpu" },
                "kv_caches": [
                  { "cache": "layer0.key", "input": "past_key", "output": "present_key", "update": "share" },
                  { "cache": "layer0.value", "input": "past_value", "output": "present_value", "update": "share" }
                ]
              }
            }
          }
        },
        {
          "id": "decode",
          "component": "decoder_decode",
          "executor_info": {
            "ort": {
              "kind": "session_loop",
              "loop": {
                "exit_conditions": [
                  { "kind": "max_generated_tokens", "limit": 128 },
                  { "kind": "eos", "token_ids": [2] }
                ],
                "generation": {
                  "phase": "decode",
                  "input_ids": "input_ids",
                  "logits_output": "logits",
                  "initial_logits": { "kind": "model_output", "step": "prefill", "output": "logits" }
                },
                "sampler": { "algorithm": "greedy", "provider": { "mode": "cpu" } },
                "cache_provider": { "mode": "cpu" },
                "kv_caches": [
                  { "cache": "layer0.key", "input": "past_key", "output": "present_key", "update": "share" },
                  { "cache": "layer0.value", "input": "past_value", "output": "present_value", "update": "share" }
                ]
              }
            }
          }
        }
      ],
      "executor_info": {
        "ort": {
          "kv_caches": {
            "layer0.key": {
              "element_type": "float32",
              "shape": [1, 2, 0, 64],
              "sequence_axis": 2,
              "initial_capacity": 128
            },
            "layer0.value": {
              "element_type": "float32",
              "shape": [1, 2, 0, 64],
              "sequence_axis": 2,
              "initial_capacity": 128
            }
          },
          "outputs": {
            "token_ids": { "kind": "generated_token_ids", "step": "decode" }
          }
        }
      }
    }
  }
}
```

Execution-specific interpretation belongs to M11:

- Prefill consumes the finite prompt input, updates shared KV caches and exits
  on EndOfInputs. It has no sampler, generation policy, EOS condition or
  generated-token budget.
- Decode owns greedy sampling. `generation.initial_logits` explicitly names the
  final prefill logits used to sample the first generated token before the first
  decode-model invocation. That token counts once toward decode's generated
  history and token budget; it is not sampled or emitted again at the boundary.
- Subsequent decode invocations consume the pending generated token and sample
  their own logits. Decode owns EOS and generated-token-limit termination.
- The sampler's provider policy also governs the initial-logits sample. M11 must
  validate producer placement, dtype, shape, compatible providers and retained
  value lifetimes without silently copying full GPU logits to the host.
- `kind: "session"` remains available for ordinary steps without introducing a
  mandatory encoder example.

This sampling-free prefill handoff is a newly requested runtime capability, not
an already implemented native contract. M10 only preserves its JSON. Before M11
implementation, refine and test zero/one-token budgets, EOS on the first sample,
Step/iteration accounting, cancellation, reset and CPU/CUDA parity. Do not
silently reinterpret existing programmatic prefill-sampling APIs.

### J.5 JSON-only standalone C API

Add the following entry points to the existing source-distributed C library,
using its export macro, error/status convention and UTF-8 string rules:

```c
ModelPackageStatus* ModelPackage_GetChainsJson(
    const ModelPackage* pkg, const char** out_json);
ModelPackageStatus* ModelPackage_GetChainJson(
    const ModelPackage* pkg, const char* chain_name, const char** out_json);
ModelPackageStatus* ModelPackage_SetChainsJson(
    ModelPackage* pkg, const char* chains_json);
ModelPackageStatus* ModelPackage_SetChainJson(
    ModelPackage* pkg, const char* chain_name, const char* chain_json);
ModelPackageStatus* ModelPackage_RemoveChain(
    ModelPackage* pkg, const char* chain_name);
```

- `GetChainsJson` returns the complete chain map, or `"{}"` when absent.
  `GetChainJson` returns the selected chain object, or
  `MODEL_PACKAGE_ERR_NOT_FOUND` for an unknown name.
- Returned strings are package-owned NUL-terminated JSON, not caller-freed
  buffers. They remain valid across read-only inspections until a successful
  package mutation, commit that mutates package state, or Close. Document this
  conservative lifetime rather than requiring consumers to infer cache details.
- A failing status call leaves all output parameters unchanged. Null required
  pointers and empty name arguments return `MODEL_PACKAGE_ERR_INVALID_ARG`.
  Invalid JSON, shape, version/feature mismatch or common references return
  `MODEL_PACKAGE_ERR_SCHEMA`; unsupported majors retain their existing version
  error. Diagnostics identify the chain/step and offending field where possible.
- `SetChainsJson` replaces the complete map; it requires an object, including
  `"{}"` to declare an empty map. `SetChainJson` inserts or replaces one chain;
  it requires an object and does not replace other chains.
- `RemoveChain` is idempotent for an unknown name, following component removal's
  existing convention. Removing the final chain removes the optional `chains`
  member. A no-op removal does not upgrade the schema or invalidate views.
- Do not add fields to `ModelPackageInfo` or introduce typed chain records for
  this milestone. Existing component/variant inspection remains unchanged.

### J.6 Immediate reference validation and mutation safety

- Validate each proposed chain/map completely before changing package state.
  Components must exist before authoring a chain that references them. A failed
  chain mutation leaves the manifest, schema version, info/JSON views, component
  records and commit/prune bookkeeping unchanged.
- Removing a component referenced by any chain fails before mutation with a
  schema diagnostic identifying a referring chain/step. No automatic cascade,
  silent chain removal or dangling-reference draft is allowed.
- Authors can first replace/remove referring chains, then remove the component.
  Whole-map replacement provides an atomic way to replace multiple chains.
- Component replacement under the same map key preserves common references.
  Variant changes/removal retain their existing authoring behavior; M10 does not
  infer executable validity from opaque consumer payloads.
- The immediate-reference guarantee covers only common component references.
  It does not validate `initial_logits.step`, tensor/state wiring or other
  consumer-private references. ORT rejects invalid private references in M11
  before constructing a runnable execution.
- Reuse existing status handling and authoring/cache helpers where appropriate.
  Do not broaden unrelated pre-existing mutation semantics to implement chains.

### J.7 Commit, paths and round-trip preservation

- Preserve chains, step order, opaque namespaces, metadata and tolerated unknown
  fields through in-memory inspection, chain mutation, portable/in-place commit,
  installed commit and reopen. Preserve unrelated components and variants when
  updating one chain; whole-map replacement intentionally replaces the map.
- New chain metadata does not introduce model-path overrides, path anchors,
  external execution files or package-supplied Session-option escape hatches.
  Model files and external variant executor-info continue to resolve through
  the selected component/variant's existing machinery.
- Existing portable confinement, installed external paths, shared-asset
  resolution, hashing, staging and orphan/prune behavior remain intact. Do not
  infer that a shared asset is unused by inspecting only known chain payloads;
  consumer-private references remain opaque.
- An ORT-like field inside an opaque payload is not permission to bypass path
  confinement or restricted Session options. These remain consumer validation
  obligations in M11, not standalone execution support.

### J.8 M10 verification and completion requirements

- Independently configure/build/test the standalone library without ORT linkage.
  Exercise inspection, authoring, all existing commit modes and asset tests,
  plus a real C consumer of the new JSON-only functions.
- Cover absent/empty chains, one/multiple chains, reused components, ordered
  steps, duplicate IDs, malformed types, missing/unknown component references,
  UTF-8 names, invalid arguments and unchanged output parameters on errors.
- Cover 1.0 compatibility, automatic authoring upgrades, 1.1 strict/permissive
  fields, future-minor preservation, unsupported majors, wrong-version chains,
  opaque consumer fields and atomic failures without lost views/bookkeeping.
- Cover reference-safe component removal/replacement, selected/whole-map
  authoring, idempotent removals, unknown-field preservation and commit/reopen
  parity with inline/external components and existing variant payload forms.
- Rerun existing CPU ORT Model Package API regressions to ensure ordinary
  single-component Session creation and package confinement remain unchanged.
- Measure 100% of new executable production parser/authoring/API/helper lines
  using the actual standalone test artifacts. Do not replace this with a test
  count, successful build or an unmeasured coverage assertion.
- CUDA device execution is not applicable to this standalone metadata milestone.
  Provider-independent payload preservation is not a GPU execution result.
  M11 owns CPU/CUDA executable package/programmatic parity and the new handoff.
- Update directly related package documentation and deliver a separately
  documented/tested M10 PR stacked on verified M09b. Do not merge automatically.
  Prepare and review the detailed execution plan only after this appendix is
  approved; implementation requires that separate plan approval.

### J.9 Save-as model materialization (approved scope amendment)

**Approved:** 2026-10-19, for revised execution-plan preparation.

During M10 persistence qualification, the unchanged M09b parent reproduced a
pre-existing defect: save-as writes metadata/shared assets but does not copy
variant directories. Reopening the destination fails when an explicit variant
directory is missing. The user requested expanding M10 to repair this defect
rather than documenting it as an exclusion.

- A successful save-as must materialize every resolved variant directory,
  including its model files, external tensor data, nested files and empty
  subdirectories, into the destination package. Reuse existing copy/confinement
  infrastructure; do not parse ONNX models or consumer payloads.
- Rebase structural component/variant paths as needed so the destination opens
  and uses the copied files without depending on the original package root.
  This includes installed-layout variant directories outside the original root.
  Retain inline versus external component and namespace shapes in PRESERVE
  mode. DENSE mode retains its existing external-namespace restriction.
- External components referenced by directory must still produce
  `component.json` at the correct location. DENSE flattening must preserve the
  effective variant-directory anchor, including subsequent in-place commits.
- External variant executor-info JSON must be materialized with its external
  shape and resolve within the copied variant directory in PRESERVE mode,
  including when its original installed path was external. Use package-local,
  collision-safe paths; never overwrite a copied model/data file with metadata.
- Consumer-private JSON remains opaque and byte/value-preserved as appropriate
  for its existing serialization contract. Only known structural paths may be
  rewritten. Files referenced by absolute or out-of-directory strings inside
  opaque payloads are not discoverable by this library; callers must place
  required package-local files within variant directories or shared assets.
- Do not copy the entire source package indiscriminately or modify any source
  file/external directory. Reject symlinks and unsupported filesystem objects
  while copying rather than following them into unrelated locations.
- Validate destination confinement and source/destination overlap before copy;
  reject self-copy/recursive destinations and output-path collisions explicitly.
  Preserve the existing empty/nonexistent-destination requirement.
- A failed save-as must not replace the live package state/root or invalidate
  its borrowed views. Partial destination files may remain on failure, as with
  existing commit behavior; this amendment does not promise atomic directory
  replacement or authorize recursive cleanup of caller-owned directories.
- Apply the repair to existing 1.0 packages as well as 1.1 chains. It requires no
  new schema field or public API. Preserve ordinary in-place commits, shared
  assets, path confinement and existing unsupported-operation errors.
- Qualify source-independent reopen and exact copied-file bytes after the
  original package becomes unavailable, across PRESERVE/DENSE and
  portable/installed layouts. Include external component file/directory forms,
  installed external variants/metadata, nested model data, empty directories,
  path collisions, symlinks, failed-copy state preservation and chain round trips.
  A CPU ORT Session created from an exported tiny real-model package must infer
  correctly without the source package. CUDA remains N/A for this file-copy fix.
- Include every new executable copy/rebase/helper/error line in the measured
  100% M10 unit-test coverage gate. Update commit documentation to state the
  structural rebasing and opaque-path boundary precisely.

Review this amendment before revising the detailed M10 plan. Implementation of
the expanded commit repair requires separate approval of that revised plan.

## Appendix K. M11 executable package chains (approved)

**Interview completed:** 2026-10-20. The user confirmed there were no additional
requirements before drafting. This appendix refines PKG-1 through PKG-7 and the
sampling-free prefill handoff approved in J.4. It does not authorize implementation.
**Specification approved:** 2026-10-20, for detailed execution-plan preparation.
M11 will be **one PR**, stacked on verified M10, as explicitly requested.

### K.1 Outcome and boundaries

- Load an explicitly named chain from a schema-1.1 Model Package into an owned,
  immutable SessionChain through public C/C++ APIs. Execute, Step, inspect,
  cancel, reset and retain results through the existing stateful interfaces.
- Support ordinary Session and SessionLoop steps and the existing declarative
  input/state/output/cache/native-generation features. Reuse existing variant
  selection, package paths, Session options and Session initialization.
- Add an explicit programmatic chain binding for initial logits and their
  originating prompt. Package loading uses the same native implementation;
  it must not implement a separate JSON-specific generation engine.
- Prefill supplies logits to the **decode SessionLoop**, not directly to a
  conventional decoder ONNX model. The loop selects a token and supplies token
  IDs to that model. Prefill itself has no sampler.
- Preserve existing programmatic prefill-sampling behavior, ordinary stateless
  Session APIs, single-component package loading and standalone package-library
  independence. No existing package silently changes execution mode.
- Manifest policies are authoritative: no per-execution token-limit/EOS override
  feature is added. Prompt/feed values still vary between executions.
- No implicit chain choice, graph/DAG scheduler, custom predicate, executable
  manifest code, new sampling algorithm, chunked prefill, DDIM implementation or
  new language binding is introduced. Standalone decode restoration from
  arbitrary external KV state is not part of this chain-handoff extension.

### K.2 Sampling-free prefill and decode bootstrap

- New mode is opt-in through an explicit bootstrap binding on the decode
  component. The logits source must be an earlier component output. The
  producer has no native prefill-sampling policy; legacy and new bootstrap
  modes cannot initialize the same generation history.
- A prompt source explicitly identifies the original nonempty CPU int64
  `[1, sequence]` token tensor. Reuse native prompt validation, including token
  nonnegativity. An external sequence source must contain exactly one complete
  prompt batch; chunking remains deferred.
- Initialize consumed-input position from the actual prompt sequence length,
  not logits shape, a guessed model-port name or a fixed package constant.
  This also supports models without managed caches and optional decode
  attention/position inputs.
- Managed prefill/decode caches must agree on names, specs and compatible actual
  provider/device identity. Their logical lengths at handoff must equal the
  consumed prompt length. Shared-prefix reuse is not implicitly enabled.
- Initial logits follow existing greedy dtype/shape/position/NaN/tie rules.
  Validate compatible model metadata, actual tensor placement and known
  vocabulary dimensions. Validate dynamic values when they become available;
  do not claim that model-dependent output data was validated before running it.
- The decode component's sampler/provider policy owns both the initial sample
  and later samples. Initial-row selection defaults to the last logits row,
  independently of the row selection for subsequent decoder model outputs.
- The first sampled token enters generated history and the token budget once.
  It is the pending input for the next decode-model invocation; it is not
  sampled or emitted again just to cross the component boundary.
- Generic sampler-free prefill uses its explicit model input bindings. If its
  model needs attention masks/positions, bind those feeds explicitly; do not
  silently attach the legacy sampling generation policy to prepare them.

### K.3 Step, exits, zero budgets and lifecycle

- After prefill completes, the first decode Step is **sample-only**. It emits
  the first generated token without calling the decode model.
- Report `executed = false`, `has_generated_token = true`, decode iterations
  zero, unchanged total model-invocation count, the decode step name and updated
  generated-token count. Reuse existing result fields; `executed` continues to
  mean a model invocation, not any observable progress.
- Chain Step must return on this token event. It must not spin into a decode
  model call merely because `executed` is false. Run continues through
  sample-only and model-executing Steps until explicit completion.
- Subsequent decode Steps perform at most one model invocation, consume the
  pending token, update caches and sample the next token as today.
- Evaluate already-satisfied exits before sampling. After the initial sample,
  evaluate token-limit/EOS exits immediately. Budget one or first-token EOS can
  finish decode with zero decode-model invocations. Preserve OR semantics and
  deterministic simultaneous exit reasons.
- In the **new mode**, a zero decode token budget does not override prefill's
  explicit EndOfInputs execution. Prefill runs, populating caches/outputs;
  decode invokes neither the sampler nor its model and returns no generated IDs.
  Existing legacy generation mode keeps its prior zero-budget behavior.
- For one prefill invocation and no earlier EOS/other exit, budget N yields:

  | Budget | Prefill model calls | Initial samples | Decode model calls | Generated tokens |
  | --- | --- | --- | --- | --- |
  | 0 | 1 | 0 | 0 | 0 |
  | 1 | 1 | 1 | 0 | 1 |
  | N > 1 | 1 | 1 | N - 1 | N |

- Sample-only Steps do not republish model outputs or perform cache updates.
  A completion-only call after completion never repeats the token event.
  Do not substitute prefill logits for an unproduced decode-model output;
  retain explicit missing-output errors when a selected port was never produced.
- Preserve atomic boundary cancellation, not GPU preemption. Cancellation and
  failures after possible advancement require Reset. Failed Reset stays invalid.
  Reset clears bootstrap progress/history/cache state and reuses original feeds.
  Immutable prior results/values remain valid under existing lifetime contracts.

### K.4 Programmatic C/C++ extension

Add a size-tagged bootstrap record after the existing source type definitions:

```c
typedef struct OrtStatefulGenerationBootstrapConfig {
  size_t struct_size;
  OrtStatefulValueSource logits;
  OrtStatefulValueSource prompt;
  int has_logits_position;
  int64_t logits_position;
} OrtStatefulGenerationBootstrapConfig;
```

Append this member to `OrtStatefulApi` and its initializer:

```c
OrtStatus* SessionChainOptionsSetGenerationBootstrap(
    OrtSessionChainOptions* options,
    size_t component,
    const OrtStatefulGenerationBootstrapConfig* bootstrap_or_null);
```

- `logits` names an earlier COMPONENT_OUTPUT. `prompt` uses an existing tensor
  source; an EXTERNAL_SEQUENCE source is resolved to its single full-prompt
  element. Validate source kinds, dependencies and types rather than treating
  bootstrap data as an ONNX model input port.
- Copy configuration and strings transactionally. NULL removes this component's
  bootstrap setting. Unknown component indices, invalid enum/position flags,
  undersized records and conflicting policies fail without changing options.
  `SessionChainOptionsSetConfig` clears bootstrap overrides, as with provider
  overrides. Created definitions remain immutable.
- Add C++ `SessionChainOptions::SetGenerationBootstrap(component, config)` and
  `ClearGenerationBootstrap(component)` conveniences. Both use the same C API
  and existing move-only options/result conventions.
- Preserve all old record sizes/array strides, prefix endpoints, table members
  and signatures. Add a frozen minimum-size endpoint for the new record.
  No `ORT_API_VERSION` bump or release markers; document additions as Version1.32.
- Keep generation math/input-policy logic under `core/generate`; session/chain
  code owns execution wiring. Reuse existing validation/dispatch helpers.

### K.5 ORT chain payload

The common M10 structure stays unchanged. Parse only `executor_info.ort` for the
selected chain and its steps; other namespaces remain opaque. New ORT-private
fields do not require a common-schema bump beyond 1.1.

| Scope | Supported fields |
| --- | --- |
| Chain `executor_info.ort` | `kv_caches`, required nonempty `outputs`. |
| Step `executor_info.ort` | Required `kind`, optional `inputs`, optional `state_outputs`; SessionLoop steps use required `loop`; ordinary Session steps may use `kv_caches` and `cache_provider`. |
| `loop` | `finite_input_names`, `nonempty_input`, `feedback`, required nonempty `exit_conditions`, `generated_token_output`, `generation`, `sampler`, `cache_provider`, `kv_caches`. |
| `generation` | Required `phase`, `input_ids`, `logits_output`; optional `attention_mask`, `position_ids`, `logits_position`, `initial_logits`. |

- `kind` is exactly `"session"` or `"session_loop"`. Do not infer it from
  component/model names. Reject irrelevant/conflicting kind-specific fields.
- `inputs` maps model-input names to existing source kinds:
  `{"kind":"external_input","name":"..."}`,
  `{"kind":"external_sequence","name":"..."}`,
  `{"kind":"model_output","step":"...","output":"..."}`, or
  `{"kind":"state","name":"..."}`.
  Step references use unique IDs, not array indices; model-output references
  must name an earlier step. Validate required inputs and native managed-port
  conflicts using model signatures and the existing chain validator.
- `state_outputs` maps retained-state names to model-output names. `feedback`
  is an array of `{"input":"...","output":"..."}` loop bindings.
- Exit entries map to existing native conditions:
  `end_of_inputs`; `max_iterations`/`max_generated_tokens` with `limit`;
  `eos` with `token_ids`; `boolean_output` with `output`.
  Validate names, integer ranges and scalar types through native contracts.
  Never supply an absent SessionLoop exit implicitly.
- Chain `kv_caches` maps names to specs containing `element_type`, `shape`,
  `sequence_axis`, `initial_capacity`. Element types are `float32`, `float16`
  or `bfloat16`. Per-step/loop bindings contain `cache`, `input`, `output`,
  `update`, with supported updates `share`, `copy`, `append`.
  Specs must exist and agree across references; reuse native cache validation.
- Native generation phases remain `prefill` and `decode`. A generation policy
  requires an explicit `sampler` object with `algorithm: "greedy"`; no other
  algorithm is supported. A sampler without a corresponding generation policy
  is invalid. Scalar model-token output loops do not imply a runtime sampler.
- Provider objects use `mode: "cpu"`, `"prefer_provider"` or
  `"require_provider"`. Non-CPU modes explicitly supply `provider_name`,
  `device_type` (`cpu`, `gpu`, `npu`) and nonnegative `device_id`.
  CPU uses no provider name and CPU device0. Omitting a cache provider, or the
  provider within an explicit greedy sampler, means the existing CPU policy.
  Model EP/variant selection is separate; these fields never register an EP.
- `outputs` maps public names to either a model-output source or
  `{"kind":"generated_token_ids","step":"..."}` for a generation step.
  Do not return an empty/default chain when an output or binding is invalid.

For new-mode decode, `generation.initial_logits` is:

```json
{
  "kind": "model_output",
  "step": "prefill",
  "output": "logits"
}
```

The source step's explicitly declared `loop.nonempty_input` identifies its
prompt input binding by default. This keeps the reviewed M10 candidate valid
without guessing the literal name `"input_ids"` or inferring length from logits.
An optional `prompt_input` names that upstream model input explicitly when
needed, including an ordinary Session producer. An optional `position` selects
the initial logits row. Resolve the prompt input's source into the native
bootstrap record and validate that it is a full prompt.

This extension is valid only with decode generation. The sampler-free producer
and explicit prompt source must be consistent; ambiguous/missing prompt sources,
future-step references, incompatible cache sets and duplicate initialization
fail. Existing legacy prefill-sampling chains without `initial_logits` continue
to use their existing matching-budget/handoff contracts.

Reject unknown or malformed fields in the supported ORT execution payload,
unsupported policies, scripts, transitions and custom predicates. Future-minor
common-schema tolerance is not permission to execute unknown ORT policy.
Do not validate unrelated chains' execution payloads when loading one named
chain or creating a legacy single-component Session.

### K.6 Package construction API and Session options

Use an owned chain returned through the existing `OrtSessionChain` handle.
Avoid a second executor/step-result API or an unnecessary opaque builder:

```c
typedef struct OrtModelPackageStepSessionOptions {
  const char* step_id;
  const OrtSessionOptions* session_options;
} OrtModelPackageStepSessionOptions;

typedef struct OrtModelPackageChainConfig {
  size_t struct_size;
  const char* chain_name;
  const OrtSessionOptions* default_session_options;
  const OrtModelPackageStepSessionOptions* step_session_options;
  size_t num_step_session_options;
} OrtModelPackageChainConfig;
```

Append to `OrtModelPackageApi` and its initializer:

```c
OrtStatus* CreateSessionChain(
    const OrtEnv* env,
    const OrtModelPackageContext* context,
    const OrtModelPackageChainConfig* config,
    OrtSessionChain** out);
```

- The chain name is required/nonempty even when the package has one chain.
  Absence of chains or an unknown name fails explicitly.
- Optional per-step SessionOptions replace the shared default **as complete
  snapshots**, not a field-by-field merge of opaque objects. Callers can clone
  defaults and modify the clone. Reject duplicate/unknown/empty step IDs and
  NULL per-step options. Inputs are borrowed only for the synchronous call.
- Reuse per-component selection and creation precedence. With caller options,
  retain advanced-path behavior, including missing package path options.
  With no caller options, use the established default creation path and CPU
  selection defaults. Apply existing package Session/provider options rules;
  never silently ignore them or invent new merge precedence.
- Resolve variants independently per step using the effective options. Repeated
  references to a component create independently initialized per-step Sessions;
  model/Session pooling is not introduced. Require intended variants/providers
  rather than misreport CPU fallback as a successful CUDA chain.
- Reuse existing model-file/external-data/shared-asset resolution and forbidden
  output-option checks. No chain field can override model-file anchors or
  inject package-supplied output file options. M10 exported layouts work without
  the source package.
- Build and validate transactionally. Release partially constructed Sessions
  and definitions on failure; preserve `out` and existing package resources.
  Preserve nested native error categories while adding chain/step/field context.
- The new fixed array record has frozen stride. Version extensions use new
  records or appended top-level fields, not growth of its array element type.
- Add `Ort::ModelPackageContext::CreateSessionChain(env, config)` returning the
  existing move-only `Ort::SessionChain`. Reuse existing `CreateExecution`,
  Step/Run/Reset/Cancel and immutable result/value wrappers.

### K.7 Ownership, validation and regressions

- Package-built definitions/executions retain the owned Sessions and resources
  they actually use. Releasing package context, caller SessionOptions or the
  public chain handle after creating an execution must not invalidate it.
  Ownership must live with the retained native definition, not just its C handle.
- Caller-owned programmatic Sessions retain their existing outlive-definition/
  execution contract. Do not silently transfer ownership of caller Sessions.
- Standard Env, external-buffer/custom-op/allocator/stream obligations remain.
  Document them explicitly rather than claiming arbitrary caller resources are
  made independent. Retained results/GPU values preserve their established
  provider/allocator/library leases after execution/chain release.
- Validate static schema/wiring/model/provider compatibility before publishing
  a chain. Validate execution feeds before advancement where available; dynamic
  output validation occurs before consumption/publication.
- Preserve sequential mutable operations, independent execution contexts, reset
  replay, cancellation invalidation and explicit errors. No retry on callback
  failure or success-shaped fallback. Minimal builds report unsupported APIs
  without touching output parameters.

### K.8 CUDA behavior and proof

- GPU cache/greedy remain **CUDA plugin EP only**. Legacy/shared CUDA must retain
  explicit unsupported-capability behavior; no implicit new backend.
- With GPU sampling, route the initial producer logits directly to the decode
  sampler on the compatible GPU. Retain the actual producer's allocator/provider
  origin; do not assume the cache provider or first Session owns all device data.
- Preserve independent CPU/GPU cache and sampler policies. Compare actual device
  compatibility, not Require/Prefer enum equality. Reject CPU-only producers,
  mismatched GPUs, conflicting data roles and undisclosed migrations.
- Reuse M09 greedy callbacks/kernels and compact token/status observation.
  No full-logits host readback/host argmax for the initial sample. CPU token feeds
  and any actual host/device staging remain explicitly measured, not described
  as device-only generation.
- Respect actual Session task streams/nondefault-stream ordering without adding
  a provider-wide/device-wide synchronization as a shortcut.
- Test zero/one/multiple budgets and EOS on the first sample. Budget zero must
  record zero sampler invocations while prefill cache/model work is observable.
  Prove real CUDA variant selection, logits/cache placement and M08/M09 dispatch;
  value parity or a CUDA-enabled build alone is not enough.

### K.9 Verification and approval gates

- Native/programmatic and package-created C/C++ chains must agree on outputs,
  complete token history, invocation/iteration counts, exits, sample-only Steps,
  retained values, reset/cancellation and errors for equivalent configurations.
- Cover ordinary Session steps, generic non-generation loops, all supported
  binding/exit kinds, old prefill-sampling mode and the new decoder-only mode.
  Include invalid JSON/policies/references, missing conditions/feeds/variants,
  malformed tensor types/shapes, model failures and resource teardown.
- Cover 0/1/N budgets, first/later EOS, simultaneous exits, no duplicate events,
  prompt validation/position/mask/cache counts and cache-free handoff. Exercise
  per-step options on repeated components, default/advanced option precedence,
  context/options release, construction rollback and multiple executions.
- Exercise portable/installed packages, external components/metadata/data,
  shared assets and source-independent M10 exports with actual local tiny models.
  No automatic real-model download or committed model weights.
- Run CPU native/public/package regressions and real CUDA-plugin conformance/
  package parity. Include legacy CUDA unsupported-capability regressions.
  Validate genuine C11 and C++17 consumers and the existing no-exceptions-client
  surface; verify baseline single-component and standalone package tests.
- Measure 100% of new executable production host/parser/loader/native/C/C++
  wrapper/glue lines, including GPU routing/error paths, with fresh matching
  artifacts and profiles. Combine evidence only with verified source mappings;
  do not hide incompatible profile hashes or round partial coverage up.
- No new CUDA kernel is intended. Verify reused device-code identity and run
  existing device conformance; if device implementation changes, independently
  measure every new executable device line. Host coverage and successful
  launches are not substitutes. Use applicable memory/lifetime/stream checks.
- Keep one independently documented/tested M11 PR, as requested. Record exact
  CPU/CUDA selectors, nonzero counts/skips, coverage totals, actual dispatch/
  transfer evidence and any instrumentation limitation without weakening tests.
- Review and approve this appendix first. Only then prepare and separately
  review the detailed TASKS plan. Do not implement, publish or begin M12 merely
  because the user requested starting this milestone.

## Appendix L. M12 DDIM denoising policy (approved)

**Interview completed:** 2026-10-30. The user explicitly confirmed there were no
additional requirements before drafting. This appendix refines SD-3/SD-4 and the
M12 milestone; it does not authorize implementation.
**Specification approved:** 2026-10-30, for detailed execution-plan preparation.
The detailed M12 execution plan was separately approved on 2026-10-30.

### L.1 Confirmed outcome and scope

- Implement one SD 1.5-compatible, deterministic DDIM policy, using a pinned
  Diffusers DDIMScheduler convention and `eta = 0`.
- Support both native seeded Gaussian initialization and caller-supplied initial
  latents. Seed/initial latents are per-execution data; step count, guidance scale
  and guidance convention belong to the immutable definition/package policy.
- Support FP32 only, with rank-four `[N, C, H, W]` latents and any positive batch
  N compatible with the selected model. Positive C/H/W are explicitly configured,
  not inferred from a filename or an assumed image resolution.
- Support both concatenated classifier-free guidance (one model invocation per
  timestep) and separate unconditional/conditional invocations (two per timestep).
- Execute scheduler, initialization and guidance arithmetic on CPU, including
  when the denoiser Session uses CUDA. Full-tensor transfers are expected and
  must be documented and measured, not presented as GPU-resident scheduling.
- One successful Step is one complete guidance/prediction/DDIM latent update.
  Return an immutable CPU latent snapshot and progress. Reset replays the
  original initialization. Cancellation is observed at complete-update boundaries.
- Reject zero inference steps. Do not add stochastic DDIM, FP16, other prediction
  types, checkpoint/resume, image-to-image, VAE scaling, tokenization, image
  preprocessing/postprocessing, model downloads, GPU-native scheduler kernels or
  a general callback/script engine. The full Stable Diffusion demo remains M15.
- Reuse SessionLoop/SessionChain, the existing execution/result APIs and ordinary
  retained state. Do not represent diffusion state as KV cache or token history.
  Ordinary stateless Sessions and existing generation policies remain unchanged.

### L.2 Pinned numerical convention

The reference is Diffusers **v0.35.1**, commit
`0f252be0ed42006c125ef4429156cb13ae6c1d60`, specifically
[DDIMScheduler](https://github.com/huggingface/diffusers/blob/0f252be0ed42006c125ef4429156cb13ae6c1d60/src/diffusers/schedulers/scheduling_ddim.py).
This is an explicit reproducibility pin, not a claim to track the latest release.
Its use as a numerical reference does not add a production Python dependency.

Use the following fixed profile, identified as `sd15_ddim_v1`:

| Parameter | Value |
| --- | --- |
| Training timesteps | 1000 |
| Beta schedule | `scaled_linear` |
| Beta start / end | 0.00085 / 0.012 |
| Prediction interpretation | `epsilon` |
| Timestep spacing / offset | `leading` / 1 |
| `set_alpha_to_one` | false |
| `clip_sample` / thresholding | false / false |
| Zero-terminal-SNR rescaling | false |
| `eta` | 0 |
| Initial-noise sigma / model-input scaling | 1 / identity |

The public [SD 1.5 scheduler configuration](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/blob/main/scheduler/scheduler_config.json)
is a **PNDM** configuration. Only the explicitly listed compatible parameters
above inform this DDIM profile; do not interpret PNDM-only settings such as
`skip_prk_steps` as DDIM behavior or claim to reproduce PNDM trajectories.

For S inference steps, let `r = floor(1000 / S)` and define
`t_i = (S - 1 - i) * r + 1`, for `i = 0 .. S - 1`.
Reject any schedule containing a timestep outside `[0, 999]` before execution.
With this fixed offset, the supported range is **1 through 999 steps**;
1000 steps would start at invalid index1000 and is explicitly rejected rather
than reproducing an out-of-range reference access. Examples:

- S=1: `[1]`.
- S=2: `[501, 1]`.
- S=50: `[981, 961, ..., 1]`.

Construct FP32 betas by squaring an FP32 linear spacing between the square roots
of the beta endpoints. Let `A_t` be the cumulative product of `1 - beta_j`
through timestep t. Use the pinned FP32 reference within documented tolerances,
not a requirement for cross-library bitwise identity.

For current latents x, guided prediction e, timestep t and `p = t - r`, use
`A_p` when p is nonnegative, otherwise use `A_0`:

```text
x0 = (x - sqrt(1 - A_t) * e) / sqrt(A_t)
x_next = sqrt(A_previous) * x0 + sqrt(1 - A_previous) * e
```

Do not add variance noise, clip x0, rescale guidance, apply VAE normalization or
recompute epsilon from a clipped prediction. Reject nonfinite policy inputs,
predictions or computed results explicitly rather than publishing invalid state.

### L.3 Initialization, shapes and model ports

- The definition fixes `latent_shape = [N,C,H,W]`. Any positive N is supported;
  batch and dimensions must match actual model metadata and execution tensors.
  Dynamic model dimensions do not remove execution-time shape checks.
- Explicitly name the model latent input, timestep input, conditioning input and
  epsilon prediction output. Validate FP32 latent/conditioning/prediction
  metadata. Do not guess conventional graph-port names.
- Explicitly select a timestep format: int64 or FP32, each as a scalar or
  one-element vector. Feed the exact selected schedule value. Other timestep
  encodings are unsupported rather than broadcast or converted implicitly.
- Conditioning inputs are separate positive/conditional and negative/
  unconditional CPU FP32 tensors with identical nonempty shapes and leading
  dimension N. Validate their rank and dimensions against the model after the
  chosen guidance convention's batch transformation.
- Additional model inputs can use existing invariant bindings; they are not
  automatically duplicated, guessed or modified by this policy.
- Use one explicitly named logical `initialization_input` to avoid optional
  chain bindings or new execution-creation APIs. Its typed value is exactly one
  of: a CPU scalar **uint64 seed**, or a CPU FP32 tensor matching `latent_shape`.
  This is a declared two-type input contract, not shape/name-based inference.
  Other dtypes/ranks/devices fail explicitly.
- Likewise, explicitly name logical positive/negative conditioning inputs.
  Logical inputs are policy-owned feeds, not extra ONNX graph inputs; reject
  collisions and attempts to bind policy-managed model ports separately.
- Copy supplied initial latents into private execution state. Do not mutate
  caller storage. They are already diffusion-space x at the initial timestep,
  not a clean image or a tensor requiring automatic noise addition.
- For seeded initialization, use a private `std::mt19937_64` engine initialized
  from the full uint64 seed, never a global RNG. Convert consecutive engine
  outputs to double uniforms using their upper53 bits:
  `u1 = ((a >> 11) + 1) / 2^53`, `u2 = (b >> 11) / 2^53`.
  Use the Box-Muller cosine then sine pair, cast to FP32, and fill in contiguous
  NCHW order; discard the final partner for an odd element count.
- Reset must reproduce the original initialization and trajectory on the same
  build/platform with the same deterministic model/configuration. CPU and CUDA
  use the same host initializer, but CUDA arithmetic may differ within the
  approved tolerances. Do not promise PyTorch RNG equivalence or cross-platform
  transcendental-function bit identity.
- Existing external-buffer, Env and programmatic Session lifetime obligations
  remain applicable. Initial latent storage and published state/results are
  owned independently of temporary caller handles.

### L.4 Guidance conventions

For finite, nonnegative guidance scale g, combine unconditional and conditional
predictions using `e = e_unconditional + g * (e_conditional - e_unconditional)`.
Test g=0, g=1 and values greater than1. Negative/nonfinite scale is invalid.

- **`batched`:** concatenate two copies of x in unconditional-then-conditional
  order along batch, and concatenate conditioning tensors in that same order.
  Run the denoiser once with batch2N and split its prediction into equal halves.
- **`sequential`:** run the same denoiser Session twice at the same timestep
  with identical, unchanged x: unconditional conditioning first, conditional
  conditioning second, each with batch N. The first call cannot advance x.
- In either mode, validate all predictions before combining them and publishing
  the latent update. Do not omit a branch for scale0/1: invocation counts remain
  exactly one or two per complete update.
- Guidance mode is definition-time policy. Validate the selected model's static
  batch dimensions accordingly; do not switch conventions or retry another
  backend/model after failure.

### L.5 Loop lifecycle, exits, state and counters

- Require an explicit existing `max_iterations` exit whose limit equals S.
  Do not silently insert an exit or let conflicting limits truncate the
  configured schedule. For this first DDIM profile, require exactly that exit;
  finite input sequences, boolean/token exits, generic feedback, generation,
  KV cache and sampler policies cannot be mixed into the same DDIM loop.
  This restriction does not change ordinary loops' supported exits/features.
- Preparation/initialization does not invoke the model or increment iterations.
  On each successful Step, run the required prediction(s), combine guidance,
  update latents, advance one timestep and then publish the complete update.
- `executed=true` means model work occurred, as in existing APIs. Iterations
  count complete denoising updates; the existing invocation counter counts actual
  denoiser calls. After k successful updates, counts are k/k in batched mode and
  k/2k in sequential mode. Chain totals include actual calls in other components.
  Generated-token counters remain zero; do not infer invocations from iterations.
- After update S, report completion with the explicit MaxIterations reason.
  A subsequent Step does no model/RNG/update work, reports `executed=false`,
  preserves counters and can return the same retained final latent value.
- Configure a distinct logical `latent_output` name for the policy-produced
  updated tensor. It is not an alias for the raw ONNX epsilon output.
  Expose it through ordinary named loop/chain outputs, retained-state inspection
  and explicit downstream chain/state bindings. For M12, DDIM's selected output
  is this latent value; raw prediction/debug-output export is not added.
- Initial/current latents and schedule position belong to the execution, not
  the immutable definition or KV manager. Definitions support independent
  executions and RNG state. Existing results remain immutable across later
  Step, Run, Reset, cancellation, failure and execution/definition release.
- Cancel may be called concurrently; Step/Run/Reset remain sequential. Observe
  cancellation before starting the next complete update, not between sequential
  guidance calls or through GPU preemption. The existing cancellation/error
  invalidation contract applies; failed/possibly advanced executions require
  Reset, and a failed Reset remains invalid.
- If prediction two, validation, allocation or scheduler arithmetic fails, do
  not publish a half-updated latent or success result. Preserve prior immutable
  results, report contextual errors and require Reset before further execution.
- Reset restores initial latents/RNG, schedule index, counters, completion and
  cancellation state without altering prior snapshots. Reuse original feeds;
  changing seed/initial latents requires a new execution.

### L.6 Native and public C/C++ integration

- Place reusable scheduler/guidance/initialization code under
  `onnxruntime/core/diffusion/`, separate from token generation and Session
  orchestration. Keep SessionLoop/SessionChain responsible for invocation,
  wiring, lifecycle and progress; package parsing is not another DDIM engine.
- Add one optional typed DDIM configuration to native loop options. Provide a
  size-tagged `OrtStatefulDdimConfig` carrying the fixed profile identifier,
  configured dimensions, step count, guidance mode/scale, explicit model ports,
  timestep format, logical input names and latent output name.
- Append `SessionLoopOptionsSetDdim` to `OrtStatefulApi`, copying configuration
  and allowing NULL to clear it. Add conventional C++ Set/Clear wrappers. Do not
  change old table prefixes, existing config-array strides, public result layouts
  or ORT_API_VERSION. Apply the API version's normal Doxygen/ownership rules.
- Setting DDIM is transactional. Replacing the complete base loop configuration
  clears any attached DDIM configuration; configuring a new one requires its
  explicit setter. Existing definitions are unaffected by option mutations.
- Reuse existing named execution inputs, Step/Run/Reset/Cancel, state/output
  access and size-safe progress/staging queries. No separate public scheduler
  object or specialized execution-creation API is required.
- Package-created definitions retain Sessions using M11 ownership. Caller-owned
  programmatic Sessions retain their existing lifetime obligations.
- Cover genuine C11, public C++17 and no-exceptions consumers, ABI-prefix/
  unsupported-build behavior, and error paths with unchanged failed out-parameters.

### L.7 Declarative Model Package policy

Extend only the selected ORT `session_loop` configuration with a strict `ddim`
object. Its fields correspond to the same typed native configuration; use
existing named-chain selection, component variant selection and SessionOptions
precedence. Standalone Model Package metadata remains ORT-independent.

Illustrative ORT step execution payload, not a full package:

```json
{
  "kind": "session_loop",
  "inputs": {
    "init": {"kind": "external_input", "name": "init"},
    "positive": {"kind": "external_input", "name": "positive"},
    "negative": {"kind": "external_input", "name": "negative"}
  },
  "loop": {
    "ddim": {
      "profile": "sd15_ddim_v1",
      "latent_shape": [1, 4, 64, 64],
      "num_inference_steps": 50,
      "guidance_scale": 7.5,
      "guidance_mode": "batched",
      "latent_input": "sample",
      "timestep_input": "timestep",
      "timestep_format": "float32_vector",
      "conditioning_input": "encoder_hidden_states",
      "prediction_output": "out_sample",
      "initialization_input": "init",
      "positive_conditioning_input": "positive",
      "negative_conditioning_input": "negative",
      "latent_output": "denoised_latents"
    },
    "exit_conditions": [{"kind": "max_iterations", "limit": 50}]
  }
}
```

The component's actual model must match these explicitly chosen port names and
shapes. An execution supplies `init` as either the declared seed scalar or latent
tensor. Positive/negative conditioning may also be bound from earlier component
outputs or retained state using existing declarative sources. Downstream
components consume `denoised_latents` as an explicit named loop output/state,
not hidden scheduler state or an executable transition.

Reject unknown fields/profiles/enums, contradictory policies/limits, bad
references, logical/model name collisions and incompatible model metadata.
Reject incompatible execution tensor data before advancement when available.
Preserve legacy package behavior and opaque unrelated consumer namespaces.
No inference-time seed/step/guidance override is read from private metadata.
Declared initialization, conditioning and invariant feeds may vary between
executions; step count and guidance configuration do not.

### L.8 CPU/CUDA responsibilities and observability

- The policy's working latents, coefficients, guidance and published results are
  CPU FP32. Required logical initialization/conditioning feeds are CPU tensors.
  A GPU-only logical feed is explicitly unsupported in this first profile.
- CPU or CUDA denoiser selection uses ordinary Session configuration, not cache/
  sampler provider capabilities. Do not imply that legacy CUDA gains native
  stateful cache/sampler hooks; this policy requires none.
- With a CUDA denoiser, stage model inputs and read required predictions back
  through supported ORT Session I/O. Distinguish unconditional/conditional calls,
  model/weight transfers, timestep/conditioning transfers, prediction readback
  and any snapshot copies. Do not replace real measurements with payload counters.
- Existing CPU staging counters remain payload observations, not DMA counters.
  Include both sequential guidance calls and actual snapshot-copy work accurately;
  report actual HtoD/DtoH counts and bytes separately from profiler evidence.
- Respect Session task streams and host readiness. Do not add a device-wide
  synchronization or global CUDA/driver setting change as a shortcut.
- Prove that the tiny denoiser's meaningful work actually runs on CUDA, with
  distinct CPU/CUDA variants or assignment/dispatch evidence preventing a
  silent CPU-fallback parity pass. Exercise a nondefault stream and actual
  resource teardown. No GPU-native DDIM or device-only pipeline claim is made.

### L.9 Acceptance and approval gates

- Independently verify coefficient tables, leading timesteps, previous/final
  alpha and analytical updates, plus pinned-reference multi-step trajectories.
  Use explicit supplied latents for numerical comparison so RNG differences
  cannot masquerade as scheduler differences.
- For finite deterministic FP32 fixtures, require per-element
  `abs(actual - expected) <= atol + rtol * abs(expected)`: `atol=rtol=1e-5`
  for coefficients/single-step comparisons and `atol=rtol=1e-4` for trajectories
  and CPU/CUDA denoiser parity. Record fixtures and intermediate checkpoints.
  A failure requires investigation, not automatic tolerance relaxation.
- Cover both guidance conventions, N=1 and N>1, nontrivial C/H/W, g=0/1/>1,
  all four timestep encodings, supplied/seeded initialization, seed0 and large
  uint64 seeds, reproducible reset and independent/concurrent executions.
- Cover S=0 rejection, S=1/2/50/999, invalid1000/negative counts, malformed
  metadata/configuration/bindings, NaN/Inf, shape/type/device mismatch,
  arithmetic/allocation overflow and failures in either guidance invocation.
- Verify identical programmatic/package C/C++ behavior, actual 1/2-call
  invocation counts, final/intermediate CPU latents, ordinary chain handoff,
  completion-only calls, boundary cancellation, failed-reset invalidation and
  immutable retained results after all supported owner-release sequences.
- Run CPU qualification and real CUDA denoiser parity, transfer, stream and
  lifetime checks. Exercise applicable legacy CUDA compatibility without
  broadening previous cache/sampler capability claims.
- Measure 100% of all new executable production host/native/parser/C/C++ inline/
  glue lines with matching source/artifacts/profiles, including CUDA I/O and
  failure paths. No new device implementation is planned; any later device-code
  scope change requires revised approval and independent device-line coverage.
- Maintain the shared scratchpad and task checklist. Deliver one intended,
  independently documented/tested M12 PR stacked on verified M11, with exact
  selectors/results, coverage and transfer evidence. No automatic merge or M13.
- Review and approve this appendix first. Only then prepare and separately
  review the detailed M12 TASKS plan; do not infer either approval from the
  instruction to start M12.

## Appendix M. M13 Python stateful bindings (approved)

**Interview completed:** 2026-11-02. The user explicitly confirmed there were no
additional requirements before drafting. This appendix refines API-1/API-2,
API-5/API-6 and M13's Python/native/package parity.
**Specification approved:** 2026-11-02, for detailed execution-plan preparation.
The detailed M13 execution plan was separately approved on 2026-11-02.
**Stack:** M13 [#27](https://github.com/onnxruntime/ort-state/pull/27), OPEN/ready on M12
`09d169b3de6001f9ac5a2a5e52747bf1cadfa003` /
[#26](https://github.com/onnxruntime/ort-state/pull/26).
**Implementation qualification/publication:** Completed 2026-11-02.
New native binding/glue coverage is660/660 mapped lines, new Python binding/
helper coverage769/769 statements, and executable examples63/63 statements.
CPU/plugin/legacy, installed-wheel, explicit transfer and sanitizer results,
including the separately reproduced pre-existing stateless plugin test failure,
are recorded in [M13 qualification](TASKS.md#final-qualification-2026-11-02).

### M.1 Confirmed scope and Python surface

- Publish the new API under **`onnxruntime.stateful`**, with typed configuration
  objects/enums and conventional snake_case methods. Do not populate the root
  namespace with every stateful type or change ordinary InferenceSession APIs.
- Expose programmatic loops/chains, executions, sources/bindings/exits, native
  generation/bootstrap, DDIM, standalone cache management/greedy sampling,
  provider selection/capabilities/diagnostics and retained results/state.
- Expose read-only Model Package loading/inspection, variant selection and
  executable named-chain construction. Package authoring/export wrappers are
  deferred; JSON metadata may be authored separately by applications.
- Accept NumPy arrays and existing public OrtValue inputs, retaining owners
  without additional implicit copies. Return **read-only stateful value
  wrappers backed by OrtValue**, not mutable aliases to native result storage.
- Resource owners support automatic cleanup, explicit idempotent `close()` and
  context managers. Independent children retain required resources even after
  a parent handle is closed or collected.
- Release the GIL during blocking native operations. Permit cancellation from
  another Python thread, with explicit rejection of overlapping operations on
  one mutable resource. No async API, Step-result iterator, Python scheduler,
  callback predicate or new provider-callback implementation API is introduced.
- Preserve native execution policies and limits, including M11 sample-only
  bootstrap and M12 one/two-invocation DDIM updates. Do not infer completion
  from `executed=False` or implement another scheduler in Python.
- Real Phi-3 and Stable Diffusion package/text/image demonstrations remain
  M14/M15; M13 examples/tests use small locally generated models.

### M.2 Typed configuration and construction

Provide public typed records covering these roles; names below define the
intended vocabulary, not a requirement to mirror C resource allocation:

| Record/type family | Required configuration |
| --- | --- |
| `LoopConfig`, `ExitCondition`, `FeedbackBinding` | Explicit selected outputs, finite inputs, feedback, exits, optional model-token/nonempty-input contracts and native policies. |
| `ValueSource`, `InputBinding`, `StateBinding`, `ChainOutput` | Explicit external input/sequence, earlier component output and retained-state sources; named final outputs/token history. |
| `ChainComponent`, `ChainConfig` | Ordered named Session/SessionLoop components, bindings, optional state/cache/provider/bootstrap configuration. |
| `GenerationConfig`, `GenerationBootstrap` | Existing prefill/decode ports, optional mask/position/row and explicit initial logits/original prompt. |
| `DdimConfig` | Exact approved M12 profile, dimensions, steps, guidance, ports, timestep encoding and logical feeds/output. |
| `CacheSpec`, `CacheBinding`, `SamplerConfig` | Native layout/update semantics and supported greedy algorithm/row selection. |
| `ProviderSelection` and native enums | CPU/prefer/require policies, explicit provider/device identity, independent cache/sampler selection. |

- Configuration records use explicit fields and type annotations. Normalize
  nested collections into owned immutable records/tuples/read-only mappings;
  changing a caller's input list/dict later cannot mutate a definition.
  Configurations are ordinary values, not resources requiring `close()`.
- Constructors capture a validated configuration, using existing native
  validation/status paths. Reconfiguration means creating a new record/
  definition, not mutating a published execution definition.
- A `SessionLoop` captures the initialized native Session behind the supplied
  public `onnxruntime.InferenceSession`. A `SessionChain` captures its supplied
  component definitions and exact initialized Sessions.
- Pin the **actual low-level Python Session object**, not only the replaceable
  high-level wrapper. If the caller later calls ordinary `set_providers()` or
  releases that wrapper, existing stateful definitions still use the originally
  captured initialized Session; only newly constructed definitions see a
  replacement. Do not alter or silently reconstruct a captured Session.
- Preserve existing environment, allocator, custom-op and SessionOptions
  callback-registration lifetimes. Package options capture existing Python
  SessionOptions snapshots/registrations, not naked native callback pointers.
- Source/output names and component identity remain explicit. Unknown fields/
  enums, embedded NUL names, duplicate/conflicting producers and invalid counts
  raise errors. Do not configure native extension properties via arbitrary
  reflected kwargs or `setattr`.
- Standalone cache/sampler objects use safe ORT/default or retained Session
  allocator ownership; do not ask Python callers to pass raw allocator or
  callback addresses. Expose existing provider capability queries/selection,
  not Python-authored C execution-provider callbacks.
- No production dependency on ONNX, PyTorch, Diffusers, tokenizer libraries or
  the standalone Python Model Package ecosystem is added.

### M.3 Execution inputs and retained ownership

- `SessionLoop.create_execution(inputs=..., batches=...)` and
  `SessionChain.create_execution(inputs=..., sequences=..., initial_state=...)`
  use explicit named collections corresponding to the existing native input
  kinds. Do not silently reinterpret finite batches as invariant inputs.
- Values are public OrtValue, NumPy ndarray, or a read-only stateful value from
  this API. Accept the latter as a **read-only feed** without a copying mutable
  conversion; keep its owner alive transitively.
- NumPy inputs must have supported dense numeric/bool dtype, native byte order
  and contiguous storage. Reject noncontiguous arrays, object/string arrays and
  unsupported types rather than silently making contiguous copies, casting,
  flattening, accepting arbitrary array protocols or inferring token/seed dtype.
  An explicit zero-dimensional uint64 ndarray represents a DDIM seed.
- Preserve native type/shape/device checks. Generic host-only bindings remain
  host-only; do not read back a GPU value simply to make validation succeed.
  Existing CUDA cache/logit routes may consume compatible device OrtValues.
- Keep the original NumPy arrays, public OrtValues, low-level Session owners and
  relevant retained stateful values alive for as long as execution/results need
  their storage. Temporary Python handles/GC must not cause dangling buffers.
- Callers must not mutate borrowed arrays/OrtValue buffers while an execution
  may use them, including future Steps and Reset replay. This lifetime contract
  is explicit; holding a Python reference does not prevent external writes.
- Do not add binding-level input copies. Native algorithm-required copies,
  including DDIM's private initial-latent copy, retain their existing behavior
  and diagnostics. Reset uses the originally captured feeds, not a new dict.

### M.4 Results, immutability and explicit copying

- `StepResult` provides immutable copied progress metadata and named output
  collections corresponding to selected outputs, component outputs and retained
  state. Collections do not expose mutable native metadata or writable storage.
- Progress includes native result kind, executed/completed, actual invocations,
  iterations, generated-token count, completed-component count/name, exit
  reasons and an optional generated token. Sample-only decode Steps may advance
  token history with `executed=False`; sequential DDIM counts two calls/update.
- `StatefulValue` supports safe shape/dtype/device metadata access. It retains
  its result/cache-view/storage/allocator/provider dependencies independently
  of Python parent handle release.
- `StatefulValue.numpy()` returns an **independent CPU copy** for CPU values,
  never a mutable view of retained native storage. Calling it on a device value
  fails with an explicit instruction to request a CPU copy first.
- `StatefulValue.to_ortvalue()` returns a separate owned mutable OrtValue copy
  on the **same device**. `StatefulValue.copy_to_cpu()` explicitly returns a
  separate owned CPU OrtValue; callers may then use the existing `.numpy()`.
  CPU/CUDA copying must respect supported ORT allocator/transfer paths and
  retained provider/session lifetimes. Unsupported copy paths report errors.
- Do not expose a borrowed mutable OrtValue, NumPy view, writable buffer,
  DLPack capsule or raw device pointer from the read-only wrapper. No implicit
  `__array__`/device conversion or silent CPU transfer.
- Mutating any returned copying conversion must not affect earlier/later
  results, cache state, Reset replay or downstream read-only feeds. Output access
  and metadata inspection alone perform no tensor copies or device transfer.
- Cache views include immutable logical value plus copied length/capacity.
  Expose existing cache stats, dispatch/capability/failure data and CPU staging
  payload counters with their native meanings, not as measured DMA counters.

### M.5 Lifecycle, GIL and concurrency

- Resource owners include loop/chain definitions, executions, package contexts,
  standalone caches/samplers, results and retained-value/view handles.
  Each supports idempotent close and context-manager exit. Closing a handle
  releases that handle's reference only; already returned children remain valid.
- Calls on a closed resource fail clearly rather than returning empty outputs.
  Previously copied immutable metadata remains ordinary Python data.
- Acquire operation/resource ownership and Python keepalive references while
  holding the GIL. Release the GIL for model/session/package initialization,
  Step/Run, blocking Reset/cache/sampler/copy operations and safe native teardown;
  reacquire it before Python object access, callback handling or refcount cleanup.
  Follow existing ORT Python callback/GIL conventions for SessionOptions; do not
  invent a new Python callback protocol.
- Native work never accesses Python containers/objects without the GIL.
  Do not destroy Python callback/owner references from an unguarded native
  destructor while the GIL is released.
- Use a per-resource operation guard: reject concurrent Step/Run/Reset, mutable
  cache/sampler changes, execution-state/diagnostic access or explicit close on
  a resource in use. Do not rely on the GIL for native mutual exclusion or block
  another thread while holding the GIL needed by a callback.
- `cancel()` is the permitted concurrent operation on an execution. It requests
  native boundary cancellation, not GPU preemption. Resource ownership remains
  stable during Cancel; close cannot invalidate a running native operation.
- Separate executions may run concurrently, including definitions sharing the
  same initialized Session where supported by existing Session behavior.
  Frozen results/values may be inspected independently of mutable executions;
  copying/closing a particular handle must still have safe ownership.
- Preserve native failure/reset invalidation. Convert/validate inputs before
  advancement; acquire native result ownership before execution. If a
  post-advancement Python wrapping failure prevents delivery, propagate the
  error and require Reset rather than silently retrying or losing progress.
- Do not change global GIL/provider/driver settings or introduce a background
  worker, async iterator, callback scheduler or implicit retry/fallback.

### M.6 Read-only Model Package loading

- Provide `ModelPackage` in the same namespace, accepting a filesystem path
  using existing Python path conventions. Reuse the native package context,
  variant selection, confinement and initialization paths.
- Return owned inspection metadata: schema/component/variant/provider identity,
  selected variant names/paths and existing resolvable metadata. No borrowed
  char pointers or mutable views into the live package context.
- Select executable chains by **explicit name**. Support shared default
  SessionOptions and optional per-step complete SessionOptions replacements,
  preserving native precedence, retained Python registrations and exact step IDs.
  Empty/default options use existing CPU defaults, not automatic GPU selection.
- Support read-only component selection and ordinary Session creation using the
  same native variant-selection mechanism. Return an ordinary public
  InferenceSession without reloading/reinitializing a second copy of the model.
- Owned package chains retain their initialized Sessions. Closing the package,
  options or chain handles does not invalidate existing executions or retained
  results/values. Package component selections retain needed package metadata.
- Manifest policies remain authoritative, including DDIM steps/guidance and
  generation exits. Per-execution data are explicit feeds, not a private
  override dictionary. Unknown/invalid selected ORT policies fail.
- Do not introduce chain mutation, package commit/export/prune, default-chain
  selection, Python execution of manifest scripts or another path resolver.
  Standalone Model Package remains independent of ORT/Python.

### M.7 Errors, typing, packaging and documentation

- Reuse repository-standard native Python exception translation, preserving
  status codes/messages and available component/operation context. Wrong Python
  argument types, closed/busy handles and unsupported conversion/build paths
  raise clear exceptions; no print-and-retry, broad catch or empty success value.
- Do not automatically retry a failed stateful run through ordinary Session
  fallback or silently rebuild its providers. Existing stateless behavior is
  unchanged.
- Expose type annotations/docstrings for every new public symbol, with explicit
  ownership/no-mutation/device-copy/cancellation/Reset contracts. Follow existing
  Python/C++ naming/style and export/packaging conventions.
- The new namespace is part of the ordinary built ORT Python package; no
  separate distribution or runtime-generated import path is required.
  Unsupported minimal/reduced builds report feature unavailability explicitly
  without breaking ordinary package import or claiming feature parity.
- Tests must import the actual new Python extension and wrapper sources, not a
  stale installed wheel or source tree missing its compiled `capi` library.
  Verify loaded paths/build identity and CPU/plugin/legacy availability.
- Document executable tiny-model examples for generic loop/chain, sampling-free
  LLM bootstrap and DDIM package execution. Examples use explicit policy/ports/
  exits, copying conversions and cleanup; they do not supply a tokenizer.
  Explain application-provided tokenizer/detokenizer boundaries for M14/M15.

### M.8 Acceptance and approval gates

- Verify native/programmatic/package Python parity for generic loop/chain,
  all supported binding/exit families, standalone cache/sampler operations,
  provider policies, generation/bootstrap/DDIM, result/state/diagnostic access,
  failed/reset-invalid contexts, close and GC.
- Test zero/one/multiple generation budgets, first EOS and sample-only Step;
  compare complete token history/counts without duplicate emission.
  Verify DDIM both guidance modes, supplied/seeded initialization, batch>1,
  explicit timestep encoding and actual one/two-call counts against M12
  reference values/tolerances, not a rewritten Python scheduler.
- Exercise temporary input owners, captured Session replacement, parent close/
  GC, retained outputs/cache views, Python option/callback lifetimes and no
  writable-result alias. Negative tests include wrong dtypes/shapes/devices,
  noncontiguous/unsupported arrays, bad policy/reference/name/configuration and
  errors before/after possible advancement.
- Prove GIL release with a second Python thread, deterministic cancellation at
  native boundaries, rejection of overlapping mutable operations and independent
  concurrent executions. Avoid sleep-only races as proof.
- Run CPU tests for every outcome and meaningful CUDA tests with real provider
  selection/native cache/sampler dispatch. Device values stay resident during
  normal access; trace dedicated cases separately from explicit caller copies.
  DDIM retains M12's intentional CPU scheduler/full-prediction readback.
- Preserve CUDA-plugin-only native cache/greedy capability boundaries and
  applicable legacy CUDA DDIM/unsupported-capability behavior. A CPU-fallback
  value comparison or CUDA-enabled extension alone is not device proof.
- Measure **100% new executable production Python and native binding/glue
  lines**, with Python coverage plus matching instrumented native artifacts.
  Count runtime wrapper/config/error/copy/lifetime code, not tests alone; report
  covered/total counts and any uncovered locations without exclusions/rounding.
  Re-exercise modified earlier paths and report cumulative coverage with safe
  source mappings. No new CUDA kernel is planned.
- Validate installed/built package import and public examples, relevant existing
  Python/native/package regressions, applicable sanitizers, style and type checks.
  Record exact commands/results, nonzero tests/skips, loaded extension identity,
  provider/device placement and separate transfer evidence.
- Maintain TASKS/SCRATCHPAD throughout. Deliver one intended independently
  documented/tested M13 PR on verified M12; no automatic merge or M14 work.
- Review and approve this appendix first; then prepare and separately review
  the detailed M13 execution plan. The request to move to M13 is not a waiver
  of either approval gate.

## Appendix N. M14 Phi-3 package and text demo (approved)

**Interview completed:** 2026-11-02. The user explicitly confirmed no additional
requirements after selecting the model variants, local asset directory,
Python interface, tokenizer adapter, prompt handling and reference approach.
**Specification approved:** 2026-11-02. The user authorized detailed M14 plan
preparation, then separately approved the detailed execution plan on 2026-11-02.
Implementation and the explicitly scoped local asset acquisition are authorized.
**Qualification:** CPU INT4 and CUDA FP16 real runs, per-variant stateless
reference/checkpoint/replay and typed/package parity pass for budgets0/1/16.
New executable production Python reaches377/377 with zero exclusions; actual
CUDA memory/lifetime/race/sync and transfer gates pass. Exact results and known
parent limitations are recorded in [M14 qualification](TASKS.md#final-m14-qualification-2026-11-03).
**Stack:** M14 [#28](https://github.com/onnxruntime/ort-state/pull/28), OPEN/ready on M13
`42ab052f08c8dc44a76a8012e3e70a053fe64b1a` /
[#27](https://github.com/onnxruntime/ort-state/pull/27).
The local planning branch is `state/m14-phi3-demo`.

### N.1 Confirmed assets and explicit acquisition

- Use the user's supplied
  [Microsoft Phi-3-mini-4k-instruct ONNX repository](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx),
  pinned to revision `5f5f794c1c23c9d5ee142af85df02a6cc52d6945`.
  Metadata and tokenizer configuration at that revision were inspected without
  downloading model graphs or external weights.
- The user explicitly authorizes downloading the following two variants and
  their tokenizer/configuration/license assets after both approval gates:

  | Target | Repository variant | External weight size |
  | --- | --- | --- |
  | CPU EP | `cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4` | 2,722,861,056 bytes |
  | CUDA EP | `cuda/cuda-fp16` | 7,642,945,536 bytes |

  The graphs, weights and tokenizer metadata total approximately 10.4GB.
  Additional CUDA INT4, DirectML and alternate CPU variants are not included.
- Store assets under the user-selected directory
  `/home/devpatel/onnxruntime-stateful/phi3-mini-4k-instruct`, outside Git.
  Do not put large models in the repository, root filesystem build trees,
  commits or PR attachments. Avoid unnecessary duplicate weight copies.
- Acquisition must be an explicit setup operation, not a side effect of normal
  package loading, tokenization or inference. Normal runs use local assets only.
  Pin every fetch to the approved revision; do not silently update to `main`.
- Download only allowlisted model/data, tokenizer/configuration and license
  files. Do not fetch or execute the repository's Python model/configuration
  implementations. Verify expected sizes and published LFS SHA-256 values for
  graph/weight files; record local hashes/provenance for supporting assets.
  Incomplete, corrupt or missing downloads must fail explicitly.
- Respect the repository's MIT license and retain its attribution/license with
  generated packages. Acquisition failures, insufficient space, authentication
  requirements or unavailable files must be reported, not silently substituted.

### N.2 Package and Python demo surface

- Deliver a Python command-line interface plus reusable Python helpers for
  explicit asset preparation, package construction and local text generation.
  Keep the demo/tooling outside the runtime's production dependencies.
- Produce runnable CPU and CUDA Model Packages with declarative manifests,
  model/external-data references, tokenizer assets and provenance. Prefer one
  portable package per selected EP under the asset directory so each package
  can declare its actual cache element types and policies without pretending
  CPU INT4 weights and CUDA FP16 weights imply identical tensor signatures.
- Validate generated manifests through the existing native Model Package
  parser/factories. Reuse existing schema/authoring helpers where applicable;
  do not implement a competing package parser or execution engine.
  Public Python package authoring/export bindings remain deferred.
- Each package declares a sampler-free full-prompt prefill SessionLoop followed
  by a decode SessionLoop in a SessionChain. Both stages share the execution's
  managed KV caches. The same selected ONNX file may serve both stages when
  inspection confirms it supports both, without duplicating its external weights.
- Main demo execution must use the M13 public `onnxruntime.stateful` bindings
  and existing native policies. Python may handle CLI arguments, tokenization,
  construction, reporting and detokenization, but must not replace the native
  generation loop, cache manager or greedy sampler.
- Accept a text prompt and an explicit package/EP selection; return generated
  text and expose token IDs plus termination/invocation information to reusable
  callers. Keep generated continuation distinguishable from the original prompt.
  Record preprocessing and measured timings without adding performance gates.
- A declared generation budget defaults to 16 new tokens for the demonstration.
  Budget/EOS policy belongs to the constructed loop/package definition, not an
  undocumented per-execution override.

### N.3 Tokenizer, chat formatting and generation semantics

- Preserve a caller-provided encode/decode adapter interface. Provide a
  demo-only Hugging Face adapter using the downloaded local tokenizer files;
  tokenizer libraries are optional demo/test dependencies, not ORT runtime
  requirements. Do not depend on a separate ONNX Runtime GenAI engine.
- Use local-only loading with remote Python code execution disabled. Apply the
  pinned tokenizer's chat template by default with an assistant generation
  prompt; provide an explicit raw-prompt mode. Avoid double insertion of BOS,
  EOS or other special tokens after template application.
- Derive and validate special-token configuration from the pinned assets.
  The inspected generation metadata declares EOS IDs 32000,32001 and32007;
  tokenizer metadata alone names 32000 as EOS. Reconcile this explicitly and
  declare all intended termination IDs rather than assuming a single ID.
- Use batch1, full-prompt prefill and greedy decode. Reject empty encoded
  prompts and unsupported inputs explicitly; do not silently truncate prompts,
  change precision, select another model or ignore the 4096-token context bound.
  Validate the prompt/generation context budget before execution.
- Prefill exits on explicit EndOfInputs and does not sample. Decode's first
  step samples the prefill logits without a decode-model invocation. Every
  generated token, including an observed EOS, is counted exactly once.
  EOS and the declared generated-token limit are explicit OR-composed exits.
- Preserve native zero-budget behavior: the declared prefill may run, while
  decode emits no tokens or decode invocations. A first generated EOS terminates
  immediately. Expose EOS in token-level results while omitting special tokens
  from rendered text as configured by the adapter.
- Preserve close/context-manager, retained-input/value, cancellation and
  Reset-required-on-failure contracts. Reset/replay must start from the same
  captured prompt and produce the same deterministic result for that variant.
  No streaming iterator, async API or multi-turn chat-session abstraction is added.

### N.4 Actual model compatibility and execution placement

- Inspect downloaded ONNX graphs, external-data references, opsets, ports,
  ranks, dtypes, cache dimensions and aliasing behavior before package authoring.
  Quantized weight precision is not sufficient evidence for cache/logit dtype.
- Pinned metadata advertises `input_ids`, `attention_mask`,
  `past_key_values.%d.key/value`, `present.%d.key/value` and `logits`, with
  32 layers,32 attention/KV heads, head size 96 and vocabulary 32064.
  Treat these as inspection expectations, not substitutes for actual graph checks.
- Explicitly inspect past/present buffer sharing and attention-mask/sequence
  length conventions. Do not blindly enable the upstream GenAI application's
  `past_present_share_buffer` optimization or weaken immutable snapshot/input
  ownership rules to accommodate a model.
- CPU qualification uses the selected INT4 model with CPU EP and native CPU
  cache/greedy. CUDA qualification uses the selected FP16 model with CUDA EP,
  native CUDA-plugin cache/greedy and an explicitly requested device.
  Require the requested cache/sampler implementations rather than silently
  falling back to CPU.
- Prove CUDA compute placement and native cache/sampler dispatch independently.
  Any intentional CPU shape/control work must be identified, not mistaken for
  whole-model CPU fallback. Do not claim that a CUDA-enabled wheel or matching
  generated prose proves GPU execution.
- Keep GPU cache/logit tensors resident across normal decode steps. Collect
  dedicated transfer evidence distinguishing compact token/status readback
  from requested validation readbacks; normal decoding must not require copying
  full GPU logits/cache payloads to Python.
- Build an ordinary stateless InferenceSession reference for each selected
  variant using the same tokenized prompt, weights, exits and greedy policy.
  Compare the stateful path against that variant's reference. CPU INT4 and CUDA
  FP16 are different numerical models; identical cross-provider prose is not
  required. Document appropriate logit/cache tolerances and near-tie behavior.
- If an actual graph exposes an unsupported signature, unsafe aliasing or a
  missing native capability, report the concrete mismatch and seek an explicit
  amendment before expanding scope. No silent model rewriting, replacement,
  unsupported fallback or new runtime concept is authorized by this draft.

### N.5 Validation and completion criteria

- Unit tests use small deterministic ONNX fixtures and fake text adapters to
  cover every new executable production line, including acquisition validation,
  builder/CLI/helper errors, tokenization/template failures and cleanup.
  Retain the standing measured100% Python/native line-coverage requirement for
  any new production glue; do not substitute tests-only or stale-profile reports.
- Run real text-in/text-out on both approved local variants with the proposed
  short-prompt16-token case, zero/one-token budgets, EOS/boundary accounting,
  context-bound validation and Reset/replay. Use deterministic fixtures for
  boundary cases that cannot be forced naturally in the real model.
- Compare against each ordinary-Session reference at prefill/decode logit and
  cache checkpoints, including final token/termination/invocation results.
  Document numerical tolerances based on actual precision and measured evidence.
- Verify programmatic/package construction parity, offline execution,
  explicit missing-asset/adapter/EP errors, and correct paths/hashes without
  committing model binaries or depending on untrusted downloaded code.
- Re-run directly related native/public/Python/package regressions and applicable
  CUDA memory/lifetime/race/synchronization checks. Preserve M13 proof and its
  documented sanitizer backtrace workaround and pre-existing stateless plugin
  negative-device test limitation; do not silently hide new failures.
- Record exact commands/results, actual loaded extension/source identities,
  nonzero test/coverage counts, selected model revision/hashes, GPU dispatch,
  transfer evidence and reproducible demo commands in documentation.

### N.6 Scope and approval boundaries

- No model fine-tuning, model re-export/rewrite, automatic asset substitution,
  additional model families/variants, chunked prefill, batching, stochastic
  sampling, serving, new CUDA kernels or Stable Diffusion/M15 work is planned.
- Maintain shared SCRATCHPAD notes and current TASKS status. Publish one
  independently documented/tested, ready-for-review PR stacked on M13 after
  qualification. Do not merge any milestone automatically.
- First review/approve this specification. Then prepare the detailed M14
  execution plan with outcomes, dependencies and CPU/CUDA test gates in TASKS,
  and obtain separate plan approval before implementation or weight downloads.

## Appendix O. M15 CUDA SD1.5 package and image demo (approved)

**Interview completed:** 2026-11-03. The user explicitly confirmed no additional
requirements after clarifying real-model EP scope, DDIM versus upstream PNDM,
boundary wrappers and the required safety checker.
**Specification approved:** 2026-11-03. The user authorized detailed M15 plan
preparation, then separately approved the detailed execution plan on 2026-11-03.
Implementation and the explicitly scoped local asset acquisition are authorized.
**Stack:** One intended ready-for-review M15 PR on M14
`44e512c7719c1e7ebc2e231c47281535c2c52a90` /
[#28](https://github.com/onnxruntime/ort-state/pull/28).
The local planning branch is `state/m15-sd15-demo`.

### O.1 Confirmed model source and acquisition

- Use the user's supplied
  [tlwu/stable-diffusion-v1-5-onnxruntime repository](https://huggingface.co/tlwu/stable-diffusion-v1-5-onnxruntime),
  pinned to revision `f22c0e90205a7a2c08bf3e8927b39dc84e4064c2`.
  Only repository metadata/configuration has been inspected; no ONNX payload
  has been downloaded.
- Its model card explicitly states that these optimized exports require CUDA
  and cannot run on CPU or DirectML. The user selected **CUDA-only real-model
  qualification**, retaining CPU EP deterministic fixture tests and measured
  100% new executable production-line coverage. Do not attempt or claim real
  CPU inference with these exports.
- The approved text-to-image asset set includes:

  | Component | Repository path | Published size |
  |---|---|---|
  | Text encoder | `text_encoder/model.onnx` | 246,178,495 bytes |
  | Denoiser | `unet/model.onnx` | 1,719,631,467 bytes |
  | VAE decoder | `vae_decoder/model.onnx` | 99,072,656 bytes |
  | Required safety checker | `safety_checker/model.onnx` | 608,658,360 bytes |

  Include the associated tokenizer, feature-extractor, scheduler and pipeline
  configuration, attribution and applicable license material. The selected
  set is approximately2.7GB. Do not download the VAE encoder: image-to-image is
  outside scope.
- Store assets under the user-selected directory
  `/home/devpatel/onnxruntime-stateful/sd15`, outside Git. Keep original downloads
  immutable and verify published sizes/LFS SHA-256 plus pinned metadata identity.
  Record original/derived-model and supporting-asset hashes/provenance.
- Acquisition is an explicit, user-approved setup operation. Normal package
  loading, tokenization and inference use local assets only, without remote
  Python code execution. No silent `main` updates, alternate models, incomplete
  downloads or unverified cached files.
- Respect the declared CreativeML OpenRAIL-M license, attribution and use
  restrictions. Pin/retain the authoritative license provenance referenced by
  the model card; do not substitute M14's MIT license or silently omit conditions
  because this repository lacks a top-level license file.

### O.2 Demo inputs, outputs and definition-time policy

- Reuse M14's Python CLI plus reusable helper style, with a caller-provided
  tokenizer boundary and optional demo-only local Hugging Face tokenizer/image
  integrations. Do not add tokenizer, Pillow or Diffusers requirements to core
  ORT runtime dependencies.
- Default configuration: one512x512 image, seed42,20 DDIM inference steps,
  guidance scale7.5, empty negative prompt and batched guidance. Output is an
  explicitly requested RGB PNG file.
- Prompt/negative-prompt text and seed or supported initial latents are
  execution inputs. Shape, step count, guidance convention/scale and model
  interfaces belong to the immutable constructed definition/package.
  Do not add an undocumented execution-time generation-policy override.
- Use the local CLIP tokenizer metadata, including its77-token limit and special
  token/padding rules. Reject an empty positive prompt and malformed/overlength
  inputs explicitly; accept the declared empty negative prompt. Do not silently
  truncate or change tokenization. Any supported explicit truncation option
  must be documented and tested rather than enabled implicitly.
- Report the effective configuration, model/revision identity, safety result,
  native step/model-invocation counts, selected EP, numerical validation and
  timing information. Separate setup, denoising and output handling measurements
  where applicable; no numerical performance gate or speedup claim.
- Image writing must be explicit, validate shape/dtype/range and use a defined
  clipping/rounding/channel-layout convention. Do not silently overwrite an
  existing output or return success when image creation fails.

### O.3 Native pipeline and approved adapters

- Construct a declarative Model Package and one native SessionChain using ordinary
  Session components for text encoding/VAE/safety work and a SessionLoop for
  denoising. Encode positive/negative conditioning separately or use compatible
  declared batch/split adapters according to actual model signatures.
- Scheduler progress, guidance and latents remain native execution state.
  Do not implement the denoising scheduler/loop in Python or misuse KVCacheManager
  as a diffusion-state container. Python handles application inputs, construction,
  reporting and checked final image saving, not a second runtime engine.
- Preserve the existing native `sd15_ddim_v1` policy: deterministic epsilon DDIM,
  eta0, the pinned scaled-linear beta/leading-timestep convention, CPU FP32
  initialization/scheduler/guidance, explicit termination and complete-update
  cancellation/reset semantics. Default20 steps means20 complete latent updates.
- The repository declares **PNDMScheduler**, not DDIM. The user explicitly chose
  native DDIM instead. Preserve upstream metadata for provenance, declare the
  effective DDIM policy clearly and compare with a matching DDIM reference.
  Do not claim parity with the repository's original PNDM trajectory.
- Inspect actual model interfaces before selecting timestep encoding, latent/
  conditioning dtypes, dimensions or guidance batching. The user did not request
  a new FP16 scheduler: CPU scheduler/guidance remains FP32 even if model internals
  are optimized FP16.
- The user permits small declared adapter stages and derived graph-boundary
  wrappers for required casts, layout conversions and scaling. Keep original
  downloaded models immutable and weight values unchanged. Record the exact
  transformations and validate each wrapper against its original model.
- In particular, bridge FP32 DDIM interfaces to any FP16 denoiser interfaces with
  explicit boundary casts where required, without silently changing scheduler
  arithmetic or widening the native policy's accepted types.
- Inspect VAE latent scaling and apply the SD1.5 scale exactly once. Do not
  assume a filename proves whether a decoder already incorporates scaling.
  Likewise define image layout/range and safety preprocessing explicitly.
- Model-only adapters may implement required preprocessing within the chain.
  No new generic graph scheduler, Python-authored runtime callbacks, arbitrary
  package scripts, runtime feature or CUDA kernel is authorized. If actual
  interfaces require a missing native capability or incompatible transformation,
  stop with concrete evidence and seek an explicit amendment.

### O.4 Required safety path and reference behavior

- The pinned pipeline metadata has `requires_safety_checker=true`; the user
  explicitly confirmed retaining the supplied checker and preprocessing.
  Safety checking is part of the output path, not an optional silent omission.
- Derive preprocessing from the pinned feature-extractor configuration and actual
  checker inputs. Validate resize/crop, channel order, rescaling/normalization and
  any image quantization against the matching reference.
- Save only an output that passed the declared safety policy. If the checker
  flags an image, report the safety result explicitly and do not write an
  unfiltered image or return ordinary successful-image status.
  Use benign prompts for qualification and deterministic fixtures for blocked/
  failed-checker behavior; do not disable the checker to obtain an artifact.
- Use ordinary CUDA InferenceSessions plus the pinned Diffusers0.35.1 DDIM
  convention as the differential reference, not the native stateful scheduler
  itself or the source PNDM scheduler.
- Feed identical initial latents to both paths. Native seed initialization uses
  M12's specified RNG, not an assumed equivalence with `torch.randn(seed=42)`.
  Verify seeded versus supplied-initialization behavior and same-config replay.
- Compare text-encoder conditioning, denoiser predictions, timestep/latent
  trajectory, VAE outputs, safety preprocessing/verdict and final image
  postprocessing with documented dtype-aware tolerances. Independently validate
  boundary adapters so a shared wrapper defect cannot masquerade as agreement.
- Deterministic fixtures use exact expected values. Real CUDA comparisons and
  replay use explicit justified tolerances; record observed differences and
  image hashes without claiming bitwise equality across providers/builds.

### O.5 Validation and completion criteria

- CPU EP unit tests with small deterministic ONNX fixtures cover every new
  executable production line: acquisition/inspection, builder/wrappers,
  tokenization, pipeline helpers, reference/verification, safety handling,
  image conversion/writing and reachable error/cleanup paths.
  Retain the standing100% requirement without new exclusions or rounded claims.
- CUDA fixtures cover actual adapter/model execution where meaningful. The real
  selected pretrained components run on CUDA only; CPU scheduler/guidance and
  documented small control/preprocessing work do not imply CPU model support.
- Produce a real512x512 RGB PNG using the default seed/step/guidance configuration,
  with retained safety result, ordinary-reference checkpoints, typed/programmatic
  versus package parity and same-config reset/replay evidence.
  Verify both native guidance conventions when the selected interfaces permit
  them; report any incompatibility explicitly rather than silently changing the
  requested default or claiming an unexecuted path.
- Prove actual CUDA denoiser/large-compute placement separately from numerical
  equality. Measure dedicated transfers with setup/warmup/reference reads
  distinguished. Full denoiser prediction readbacks are intentional under the
  inherited CPU FP32 policy and must be counted, not described as GPU-resident
  scheduling.
- Run applicable real CUDA memory/lifetime/race/synchronization checks and
  directly related native/public/Python/package regressions. Preserve previous
  milestone proofs and documented parent limitations; do not hide new errors.
- Validate local-only installed-wheel execution, image shape/encoding/safety,
  missing assets/adapters/devices, invalid dimensions/seed/steps, cancellation,
  failure requiring Reset and downstream/image-write suppression after errors.
- Record exact commands/results, actual artifact/source/model identities,
  nonzero test/coverage counts, output image/provenance, scheduler/reference
  tolerances, device placement and explicit transfer/sanitizer evidence.

### O.6 Scope and approval boundaries

- Exclude real CPU inference with these CUDA-only exports, DirectML, extra model
  families, model-weight changes/retraining, PNDM implementation, stochastic
  DDIM, image-to-image, VAE encoding, serving and new runtime/kernel features.
  This approved EP scope does not waive CPU fixture coverage.
- Maintain TASKS/SCRATCHPAD throughout. Publish one independently documented/
  tested ready-for-review PR stacked on M14 after qualification; do not merge
  automatically or begin M16.
- First review/approve this specification. Then prepare the detailed M15 plan
  with intended outcomes, actual prerequisites and CPU/CUDA test gates in TASKS,
  and obtain separate plan approval before model downloads or implementation.

### O.7 CUDA-plugin compatibility repair (specification approved)

Actual approved-asset execution identified a native runtime blocker on2026-11-03:

- The unmodified original UNet, executed through an ordinary CUDA-plugin
  InferenceSession outside the new stateful pipeline and outside all boundary
  wrappers, segfaults. GDB identifies CUDA `Shape::ComputeInternal` with a null
  input tensor dereferenced while reading shape metadata.
- Disabling memory patterns reproduces the same native failure. Disabling graph
  optimizations changes the failure to a CUDA Transpose `input count mismatch`;
  it does not produce a working model. These are failed diagnostics, not passing
  compatibility results.
- Original/wrapper text-encoder inference agrees exactly. All downloaded hashes,
  original initializer preservation and the CPU ONNX safety preprocessing
  reference checks pass. No successful full image is claimed.

**Scope amendment approved:** 2026-11-03. Permit a narrowly targeted root-cause repair in the
existing CUDA-plugin Shape/tensor-input adapter path, with no new public API,
runtime concept, scheduler, model substitution or safety bypass. Resolve the
actual input/constant/metadata handling defect rather than only adding a null
guard that turns the crash into an unusable-model error.

The repair must include a minimal regression derived from the failure and
relevant Shape/Transpose/constant/empty-input coverage, preserve CPU/legacy
behavior, pass applicable CPU/CUDA tests, and measure100% of any new/modified
executable native/glue lines with fresh matching binaries/profiles.
Then repeat original/wrapper equivalence and all approved M15 qualification.
Preserve previous milestone proofs before any rebuild.

The user approved this specification amendment, then separately approved its
targeted R0-R4 execution plan on2026-11-03. The focused repair, proof-preserving
task-owned build relocation and accurate rebuild/qualification are authorized.

The scoped diagnosis subsequently identified the upstream contract violation:
adapter `KernelDef::OpName()` exposes a node display name, not the operator type.
The existing CUDA GroupNorm implementation therefore misclassifies arbitrarily
named SkipGroupNorm nodes and omits the residual consumed by the failing Shape.
The focused repair uses the existing `GetOperatorType()` accessor. It does not
change Shape/Transpose input rules, add a null guard or introduce a new API.
Named-node/residual/downstream regressions and the original UNet remain required
proofs before the repair or M15 can be accepted.

### O.8 FP16 differential-validation precision amendment (approved)

The repaired fresh local wheel produces the default 512x512 RGB image with
25 model invocations, seed42 and an unflagged mandatory safety verdict.
Original/wrapper component checks and actual CUDA large-op placement pass.
The source safety model reuses `height`/`width` symbols for both224px CLIP
features and512px images; the declared boundary wrapper now distinguishes the
image symbols without changing any weights or computation.

Full closed-loop comparisons still fail the initially documented per-element
`rtol=0.002, atol=0.02` latent gate. Independent ordinary CUDA Sessions also
disagree with each other, more than the native/reference paths do. Diagnostics
using identical starting noise and the same Diffusers DDIM arithmetic measured:

| Metric across20 updates | Batched | Sequential |
|---|---:|---:|
| Native/reference largest latent RMSE |0.005061|0.002716|
| Ordinary/ordinary largest latent RMSE |0.006517|0.003537|
| Native/reference largest absolute latent difference |0.084934|0.027581|
| Ordinary/ordinary largest absolute latent difference |0.133366|0.097084|
| Largest same-input local-update absolute difference |0.007172|0.008536|
| Native/reference normalized-image RMSE |0.001608|0.000888|
| Ordinary/ordinary normalized-image RMSE |0.002212|0.001375|
| Ordinary/ordinary largest absolute image difference |0.120362|0.134766|

An unmodified original VAE repeats with maximum differences around0.0054;
its boundary cast agrees exactly with independently prepared FP16 inputs.
Setting the existing deterministic-compute option did not remove the variation.
These are measured model/kernel numerical effects, not evidence that a seed
produces byte-identical CUDA images. The current failed strict comparisons remain
failed evidence and must not be relabeled as passing.

**Validation specification approved:** 2026-11-03. The user approved the following
policy and requested a separately reviewed execution plan before implementation:

- Keep all CPU deterministic fixtures exact; do not relax their results,
  state/ownership/cancellation tests or100% new-line coverage.
- Keep original initializer identity, explicit CUDA placement, stage/update/
  invocation counts, safety BOOL verdicts, same-decoded-image preprocessing and
  retained-snapshot immutability exact.
- Check each complete native DDIM update against an independent ordinary-model/
  Diffusers update from the **same preceding native latent tensor**, retaining
  `rtol=0.002, atol=0.02`. This isolates scheduler/guidance correctness rather
  than masking it behind cumulative model variation.
- Also retain the fully independent closed-loop reference from identical initial
  noise. At every update require latent RMSE<=0.01 and absolute max error<=0.2.
  For normalized final images require RMSE<=0.005 and absolute max error<=0.2.
  Apply these same declared bounds to real typed/package, seeded/supplied and
  reset/replay comparisons; record both RMSE and maxima, not just success flags.
- Continue exact safety verdict comparisons and save only an unflagged output.
  Record individual image hashes without requiring identical encoded PNG bytes
  from separate nondeterministic real-model executions.
- Fix and document these bounds before final qualification; never auto-calibrate
  or increase them to make a run pass. Repeat both guidance modes and retain
  ordinary/ordinary baselines, original/wrapper checks and observed differences.
- No new runtime/kernel/model-compute changes are included. A request instead
  to enforce bitwise CUDA determinism requires a separately approved scope and
  execution plan, not a silent expansion of the current repair.

Separately review the concrete verifier/test/qualification plan before
implementing this amendment. M15 publication remains
blocked until the approved policy and all original qualification gates pass.

### O.9 cuDNN search and accumulated-error amendment (approved)

O.8 and its Q0-Q5 plan were approved and implemented without increasing their
limits. CPU tests and501/501 new Python lines pass. Real qualification exposed
two remaining blockers:

- A typed/package image pair has RMSE0.001992 (below0.005), but one absolute
  difference reaches0.202148 (above the fixed0.2 cap). With an experimental
  HEURISTIC convolution search, one closed-loop latent comparison has
  RMSE0.008329 (below0.01), but maximum0.222359. These remain failed checks.
  The existing CUDA GroupNorm code explicitly documents a nondeterministic scan/
  atomic reduction and warns that no deterministic CUDA kernel exists.
- Full real-pipeline memcheck reports20,744 leaked bytes in three allocations.
  The **unmodified original VAE through an ordinary stateless CUDA Session**
  reproduces the same leak. Allocation traces point into `libcudnn_graph.so.9`.
  Non-exhaustive `DEFAULT` and `HEURISTIC` convolution-search diagnostics both
  eliminate the leak without changing weights or adding runtime code.

**Specification and separate execution plan approved:** 2026-11-03. The user
approved the following revision with no additional requirements, then approved
its N0-N4 implementation/requalification plan. The configuration is implemented:

- Select the existing CUDA provider option
  `cudnn_conv_algo_search=HEURISTIC` explicitly for this demo's CUDA Sessions.
  Record it in package/demo provenance and documentation; rebuild packages
  explicitly for the qualified configuration. CPU SessionOptions are unchanged.
  This is an application-level choice, not a global driver/library modification
  or a new runtime/kernel API. Continue proving CUDA large-op placement.
- Preserve the approved hard closed-loop RMSE bounds: latent<=0.01 and
  normalized image<=0.005. Continue measuring/reporting absolute maxima, but
  do not use accumulated trajectory/image/parity maxima as a hard gate.
  Keep the strict same-input local-update pointwise check
  (`rtol=0.002, atol=0.02`), the matching-input component checks, exact CPU
  fixtures, exact original initializer identity, exact safety/preprocessing/
  immutable-snapshot checks and all accounting requirements.
- No auto-calibration, retry-until-green policy, changed scheduler/model math,
  safety bypass, memory suppression or `cudaDeviceReset` leak masking.
  Rerun both real modes, all lifecycle tools and transfer/coverage/regression
  gates with this explicit configuration before publication.

An earlier **exploratory, test-process-only** configuration with HEURISTIC search and
RMSE-only accumulated checks passes both real guidance-mode tests and full
real-pipeline memcheck with zero errors/leaks. That exploratory result was not
accepted as production qualification. Subsequent production runs use the
approved configuration and pass both real guidance modes repeatedly.
Requiring bitwise CUDA outputs would still call for deterministic-kernel work
outside M15's approved scope.

### O.10 Bounded real-model race qualification (approved)

**Qualification refinement approved:** 2026-11-03. After an unfiltered
two-update racecheck timed out at ten minutes, the user explicitly approved
bounded sampling of each actual GroupNorm kernel variant affected by the
operator-identity repair. This is a test-scope refinement, not a production,
numerical, model, scheduler or safety-policy change.

- Retain full 20-update, full-leak/API memcheck and all-kernel synchronization
  checking of a full-size two-update pipeline, including replay, cancellation,
  required Reset, retained values and downstream safety handling.
- Sample the actual FP16 GroupNorm scale and sum160/sum256 variants with
  case-correct mangled-name filters. Correlate selected names with actual CUDA
  launch records; do not accept a zero-match filter as evidence.
- Record launch bounds, instrumented scope, zero errors/hazards and incomplete
  attempts explicitly. Do not claim unfiltered/full-model race freedom.
- An earlier lowercase-only filter did not match the uppercase GroupNorm
  symbols. Its zero-hazard result is withdrawn, not counted as qualification.
  Correctly targeted but interrupted/unbounded attempts likewise are not passes.
- Preserve the fixed RMSE/local/exact tests, original weights, mandatory safety,
  scoped 100% new-line coverage and stacked-publication gates.
