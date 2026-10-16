# Stateful Session Execution for ONNX Runtime

**Status:** Approved by the user on 2026-10-15 for execution-plan preparation.
Implementation requires separate approval of `TASKS.md`.
**Requirements captured:** 2026-10-15.

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
  weights. Do not commit large model binaries or automatically download them.
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
- **SD-5:** Provide package construction and execution instructions for CPU and
  CUDA, including model/export assumptions, seed, step count, guidance settings,
  preprocessing, and output postprocessing.

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
| Diffusion | Tiny deterministic fixtures plus a real local text-to-image run on CPU and CUDA, exercising DDIM and guidance. |

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
                { "kind": "end_of_inputs" }
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
