# Adaptive CUDA expert offloading for Qwen 3.6 MoE

**Status:** Implementation planned

**Date:** 2026-08
**Updated:** 2026-09-22

## Objective

Implement adaptive expert placement for Qwen 3.6 and other Mixture-of-Experts (MoE) models. Expert-routing
instrumentation and the reproducible evaluation tooling are complete. The trace-simulation and predictive-analysis
steps are skipped; implementation proceeds directly with the policy described below.

The eventual operator must execute models whose expert weights do not all fit in GPU memory. CPU memory keeps the canonical copy of every expert. A bounded number of experts are also cached on CUDA without removing their CPU copy, so an evicted expert remains immediately available to the CPU path.

The adaptive policy maintains an exponentially decayed counter for every expert of every `MoE` and `QMoE` node.
The highest counters determine CUDA residency within each node. A global setting specifies how many experts, or what
proportion of experts, may be offloaded. Initial counter values may be loaded from a text file; otherwise all counters
start at zero and the initial placement is uniform across nodes.

Each MoE invocation uses an immutable placement snapshot. After it completes, the runtime may asynchronously exchange
one hot CPU expert with one cold CUDA expert. The exchange must be complete before that node's next invocation. After
the complete model inference, the runtime redistributes the global CUDA budget across nodes, prioritizing a placement
that allows as many complete `MoE`/`QMoE` nodes as possible to execute on CUDA.

## Scope

The initial work targets the Qwen 3.6 MoE graph and its top-k routing pattern. Expert identity is the pair `(layer_id, expert_id)` because experts from different layers do not share weights or statistics.

The work proceeds in this order:

- Add opt-in logging of complete expert-selection sequences. **Complete.**
- Add reproducible routing-trace collection and statistical-analysis tooling. **Complete.**
- Skip the offline cache simulator and measured-results step.
- Skip predictive-strategy analysis.
- Implement the selected exponentially decayed counter policy, global budget redistribution, persistent CPU weights,
  CUDA slots, asynchronous exchanges, and CPU fallback.

Training, router-logit changes, and expert-weight quantization are outside the initial implementation. The operator must preserve model output within the tolerance of the existing CPU or ONNX Runtime implementation.

## Memory and execution model

Each expert has one permanent CPU allocation. CUDA slots contain copies of expert weights. CPU weights are never moved
or released when an expert is copied to a slot.

The implementation adds one global session setting:

```text
session.moe_cpu_offload_experts=<positive number>
```

When the value is an integer greater than or equal to `1`, it is the global number of experts to offload to CPU. When
it is strictly between `0` and `1`, it is the proportion of all experts to offload to CPU; the implementation documents
and applies one deterministic rounding rule. Zero, negative values, non-integral values greater than `1`, and counts
larger than the model's total expert count are rejected. When the option is absent, expert offloading is disabled and
the existing CPU or CUDA `MoE`/`QMoE` behavior remains unchanged.

The complement of the offload target is the global CUDA-resident budget. The cache manager distributes this budget
among all `MoE` and `QMoE` nodes. Within a node, experts are ranked by descending counter with `(node_index,
expert_id)` as the deterministic tie-breaker. If all counters are zero, the initial CUDA slots are distributed as
uniformly as possible across the nodes; any remainder is assigned in node-index order.

```text
CPU expert weights (canonical, always resident)
    expert 0 ---------------------------+
    expert 1 --------------+            |
    ...                    | copy       | copy
    expert N -------+      v            v
                    |  CUDA slot 0   CUDA slot 1  ...  CUDA slot C-1
                    +-> CPU fallback
```

Cache policy, counters, and transfers belong to a runtime cache manager. The execution path receives an immutable
snapshot of the current expert-to-slot mapping:

- A cache hit dispatches the token to the CUDA expert in its assigned slot.
- A cache miss executes the current token from the CPU weights.
- After the complete MoE invocation finishes, the policy compares the hottest CPU expert with the coldest CUDA expert.
- If the exchange threshold is met, the CPU expert's permanent weights are copied to the selected slot on a dedicated
  transfer stream.
- A slot cannot be reused until all CUDA work referencing its previous expert has completed.

The cache manager never changes the placement used by the current invocation. It never moves or removes CPU weights:
every expert remains executable on CPU before, during, and after CUDA residency. A host-to-device copy starts only
after the current MoE invocation has finished. If the destination slot is still in use, the transfer stream waits for
the previous expert's completion event before starting the copy.

An expert is `loading` while its copy is in flight. Before the next invocation of the same `MoE`/`QMoE` node, the
manager waits for the transfer-completion event if necessary, then atomically publishes the new mapping. The next
invocation must not run with a partially exchanged slot or the old mapping.

The intended timeline is:

```text
token t, layer L router selects expert E
    -> execute the complete MoE using CUDA hits and CPU misses
    -> update every counter for layer L using decay and the used/not-used indicator
    -> compare the hottest CPU expert with the coldest CUDA expert
    -> after the MoE completes, enqueue one qualifying exchange asynchronously
    -> finish the remaining layers of token t
token t+1, before layer L
    -> wait for the exchange to complete if it is still in flight
    -> publish the new mapping and execute with the new placement
end of model inference
    -> redistribute the global CUDA budget across all MoE/QMoE nodes
    -> maximize the number of nodes whose experts are all CUDA-resident
```

This gives the copy an overlap window from completion of layer `L`'s MoE for token `t` until layer `L` is reached for
token `t + 1`. A synchronous transfer mode is retained only for correctness tests.

## Operator strategy

ONNX Runtime already provides `com.microsoft::MoE` and `com.microsoft::QMoE` kernels for both the CPU and CUDA execution providers. The initial implementation should extend these existing operators instead of adding a public operator whose contract is tied to two devices.

An ORT graph node is assigned to one execution provider; an operator is not jointly owned by the CPU and CUDA EPs. In hybrid mode, the node remains assigned to the CUDA EP. Its CUDA kernel owns the expert cache, keeps the canonical expert weights in CPU memory, launches cached experts on CUDA, and invokes shared CPU MoE computation for misses. CPU execution is therefore an internal fallback path, not a second EP assignment.

This approach preserves the existing operator schema and exported models:

- when `session.moe_cpu_offload_experts` is absent, current CPU and CUDA behavior is unchanged;
- when the option is present, the CUDA kernel selects the hybrid implementation;
- CPU and CUDA implementations share routing validation and expert-compute helpers instead of duplicating numerical logic;
- placement policy and cache state remain runtime concerns rather than ONNX attributes.

PR 5 must first verify that initializer prepacking and ORT's memory planner can retain the canonical expert weights on CPU without also materializing every expert on CUDA. If that cannot be done without changing the existing kernel's input-memory contract or regressing its normal CUDA path, the fallback is an internal experimental `MoEWithCPUOffload` contrib operator inserted by an ORT graph transformer only when the session option is present. It must reuse the existing `MoE`/`QMoE` schema semantics and kernels and must not become the exported model contract unless the experiment proves that a separate operator is necessary.

## Selected placement strategy

### Counters and initial state

Each `MoE` and `QMoE` node owns one counter per expert. After the node executes, all counters for that node are updated
once:

```text
count_{t+1}(e) = alpha * count_t(e) + beta * (1 if expert e was used, otherwise 0)
```

`alpha`, `beta`, and `epsilon` are non-negative session-policy parameters with documented defaults. `alpha` is at most
`1`. The used term is binary for one invocation even when several rows select the same expert; it measures whether an
expert participated in that invocation rather than the number of tokens routed to it.

An optional session setting names a UTF-8 text file containing initial counter values:

```text
session.moe_expert_counter_state_file=<path>
```

The file contains one record per `(node_index, node_type, expert_id)` and is validated strictly against the resolved
graph: duplicate records, unknown nodes or experts, non-finite values, and negative counters are errors. Missing expert
records are initialized to zero. When the option is absent, every counter starts at zero. The exact line-oriented
format and version marker are specified with the implementation so state can be generated and reviewed without a
binary tool.

### Per-node exchange

For a node with both CPU and CUDA experts, let `cpu_max` be the maximum counter among CPU experts and `cuda_min` the
minimum counter among CUDA experts. After the node finishes executing, exchange the corresponding experts when:

```text
cpu_max > (1 + epsilon) * cuda_min
```

At most one exchange is started for a node after one invocation. Ties use expert ID order. The current invocation is
unaffected. The old CUDA expert remains valid until its CUDA work completes, then the transfer stream overwrites that
slot with the selected CPU expert. The new mapping is published only after the copy completes, and completion is
required before the node executes again.

### Global redistribution

After each complete model inference, recompute how many CUDA slots belong to each `MoE` and `QMoE` node while preserving
the global offload target. The allocation objective is lexicographic:

1. Maximize the number of nodes whose complete expert set is CUDA-resident.
2. Among allocations with the same number of complete CUDA nodes, maximize retained counter mass.
3. Break remaining ties by node index and expert ID.

Within each node, keep the experts with the largest counters. Rebalancing copies are asynchronous, but every affected
node must finish its pending copies before its next invocation. This end-of-inference redistribution changes slot
ownership between nodes; the per-node exchange above changes expert identity without changing the node's slot count.

## Expert-sequence logging

Expert-statistics collection is disabled by default. It is enabled only when the following session configuration entry is set:

```text
session.enable_moe_expert_statistics=1
```

The default value is `0`. When it is `0`, the routing path must not allocate statistics buffers, collect expert
identifiers, or add measurable synchronization overhead. `qmoe_prompt_runner.py` enables the setting automatically.

The delivered evaluation protocol is intentionally limited to sequential, single-prompt generation. The runner
processes one prompt at a time, and expert-statistics logging rejects a 3D MoE/QMoE input whose explicit batch dimension
is not 1. A 2D input remains valid because the operator schema defines it as an unbatched token matrix, which is needed
for a single prompt's multi-token prefill. Concurrent `Run()` calls are outside this protocol.

The implementation writes only MoE routing decisions through the normal ONNX Runtime logger at INFO severity. It does
not enable `profiling::Profiler`, retain a Chrome trace in memory, or record unrelated node and kernel events. Consumers
can redirect the logger output to a file and select lines beginning with `moe_routing `; the remainder of each such line
is a standalone JSON object.

Each current routing record contains `request_id`, `node_index`, `node_type`, `node_name`, `expert_ids`,
`router_weights`, `num_rows`, `top_k`, and `execution_device_id`. The three node fields form the routing record's node
identity; `node_name` may be empty. ORT initially assigns node indices in raw `GraphProto.node` order and does not
renumber surviving nodes when graph optimization removes other nodes. The analyzer also requires the recorded operator
type and name to match, so a transformation that replaces a QMoE node fails validation instead of attributing its
records to another raw node. CUDA routing buffers are copied asynchronously and retained until the normal end-of-run
execution-provider synchronization, then serialized. Per-run record and routing-element limits bound pinned host
memory; a `moe_routing_truncated` warning reports any dropped decisions explicitly.

The runner surrounds every prompt with ordered `prompt_start` and `prompt_end` markers and writes a final
`moe_routing_complete` footer containing the prompt, prompt-run, and routing-record counts. The analyzer requires all
markers, rejects overlapping or out-of-order prompts, validates the footer counts against the records, and cross-checks
the completed prompt count with the benchmark JSON before writing CSV or plot artifacts. An interrupted, truncated, or
non-sequential run therefore fails closed.

PR 2 adds a dedicated JSON Lines destination:

```text
session.moe_expert_statistics_file=<path>
```

When this entry is present together with `session.enable_moe_expert_statistics=1`, ORT writes only routing records to
the specified file. It does not redirect `stdout` or `stderr`, include unrelated ORT messages, or enable profiling.
The normal logger remains the backward-compatible destination when the file entry is absent. File-open and write
failures are reported as run errors rather than silently disabling trace collection.

A later output-prediction study extends these events with sampled MoE outputs. This extension remains behind `session.enable_moe_expert_statistics=1` and is disabled unless an explicit sampling configuration is provided. It records the source `(request_id, input_shapes, token_index, layer_id)`, output shape and type, and either the sampled output vector or a documented deterministic projection. It must not emit every full activation by default because that would make the JSON traces impractically large.

The planned extended routing record contains:

| Field | Meaning |
|---|---|
| `run_id`, `request_id` | Reproducible benchmark and sequence identifiers. |
| `input_shapes` | Concrete model-input dimensions for detecting sequence growth and reset. |
| `input_token_hash` | Stable identifier for joining CPU and CUDA iterations with identical token inputs. |
| `token_index`, `layer_id` | Position of the routing decision in the generated sequence. |
| `expert_ids`, `router_weights` | Ordered top-k experts and their router weights. |
| `execution_device_id` | Device that executed the expert; `-1` denotes CPU and non-negative values denote CUDA device IDs. |
| `resident_experts` | Placement snapshot including `(layer_id, expert_id, cuda_device_id, slot_id)` tuples. |
| `cache_hit` | Whether each selected expert executed on CUDA. |
| `admitted`, `evicted` | Placement changes decided after the routing update. |
| `moe_completed_ns`, `copy_enqueued_ns`, `copy_ready_ns` | MoE completion, copy interval, and readiness before the next token. |
| `copy_source_device_id`, `copy_destination_device_id` | Transfer endpoints; CPU is identified by `-1`. |
| `copy_bytes`, `copy_duration_ns` | Host-to-device or future peer-to-peer traffic caused by cache changes. |
| `iteration_duration_us` | Complete `model_run` duration, including all non-MoE work. |
| `cpu_duration_ns`, `cuda_duration_ns` | MoE execution time split by device when the cache is implemented. |

Run metadata records the model revision, ONNX Runtime and `onnxruntime-genai` versions, CUDA version, GPU and CPU models, visible CUDA device IDs and topology, capacity, policy parameters, random seed, warm-up length, and benchmark configuration. Prompts and generated text are not logged by default; stable request identifiers are sufficient for joins when explicitly required.

## Evaluation scripts

`tools/python/qmoe_prompt_runner.py` submits a JSON list of prompts directly through the `onnxruntime-genai`
generation API. It does not import or invoke `locodellm`. Until the dedicated routing-file session option is
implemented, it redirects the native ORT `stderr` file descriptor to the requested routing log and emits explicit
`N/total` prompt start/end boundaries. After the result JSON is written, it appends a counted
`moe_routing_complete` footer. It writes generated text, token counts, durations, and throughput to a separate JSON
file.

```bash
python tools/python/qmoe_prompt_runner.py /path/to/model --prompts-file prompts.json --provider cuda --max-new-tokens 256 --output qmoe-prompt-results.json --routing-log qmoe-routing.log
```

The prompt file is either a JSON array of strings or JSON Lines containing strings or objects with a `prompt` field.
Use `--prompt` repeatedly instead of `--prompts-file` for small manual runs. The runner applies the model tokenizer's
chat template by default; `--raw-prompts` disables that behavior.

`tools/python/qmoe_expert_distribution.py` validates and streams the routing records, computes prompt, layer, and global
expert distributions across every routing row, ranks experts by frequency, and maps the final row's selected top-k
experts to their zero-based frequency ranks for the current decode-token view. It also generates threshold aggregates,
derives expert bytes from the ONNX external initializers, and writes the result plots. Logs containing a
`moe_routing_truncated` warning, incomplete prompt boundaries, an absent or inconsistent completion footer, or
malformed routing and external-data metadata are rejected before generating analysis artifacts. The benchmark JSON is
required, and its prompt count and indices must match the completed trace. Matplotlib is required for plotting, but not
for importing the analysis helpers.

```bash
python tools/python/qmoe_expert_distribution.py qmoe-routing.log --benchmark-json qmoe-prompt-results.json --model /path/to/model/model.onnx --output-prefix qmoe-routing-analysis
```

Both scripts keep the raw routing trace separate from aggregate CSV and PNG artifacts. The analyzer tests are registered
explicitly in Python CI and cover prompt completeness and ordering, top-k extraction, rank ties, threshold totals,
routing-schema validation, and ONNX expert-size calculation.

## First results

An exploratory trace was collected from Qwen3.5-35B-A3B INT4 on CUDA using 10 prompts. It contains 59,880 valid routing
records from 40 QMoE layers, with 256 experts per layer and top-k 8. General profiling was disabled. The frequency ranks
below are zero-based and learned from this complete 10-prompt trace; these preliminary results demonstrate the analysis
pipeline but are not sufficient to select a cache policy.

Five generated routing-analysis rows are shown below. The prefill record retains only the final token row so every event
contains exactly eight selected experts and eight corresponding frequency ranks.

| Prompt | Inference | QMoE | Selected expert IDs | Frequency ranks | Maximum rank |
|---:|---:|---|---|---|---:|
| 1 | 1 | `layers.0` | `[81,206,140,200,67,95,30,187]` | `[2,87,41,11,178,21,109,54]` | 178 |
| 1 | 1 | `layers.1` | `[94,224,128,233,112,33,11,172]` | `[15,2,16,23,4,1,8,3]` | 23 |
| 1 | 1 | `layers.2` | `[167,158,153,34,109,93,217,179]` | `[7,1,4,42,46,69,0,18]` | 69 |
| 1 | 1 | `layers.3` | `[6,214,196,69,233,117,230,225]` | `[20,73,6,0,68,22,33,32]` | 73 |
| 1 | 1 | `layers.4` | `[15,154,163,222,108,195,129,87]` | `[35,4,26,28,17,2,20,23]` | 35 |

The first figure compares observed routing coverage with the normalized bytes excluded by a frequency-ranked shortlist.
One expert occupies 1,775,616 bytes per QMoE layer, or 71,024,640 bytes across all 40 layers for one additional rank.

![Normalized routing coverage and expert bytes](images/08-moe-cpu-offload/normalized-total-vs-expert-bytes.png)

The second figure compares normalized rank-threshold curves for representative early, middle, and late QMoE layers.
Layer-specific differences motivate the per-node counters and end-of-inference budget redistribution.

![Selected QMoE layer expert-rank distributions](images/08-moe-cpu-offload/selected-layers-expert-ranks.png)

## Post-implementation evaluation

Evaluate the model on a fixed set of 1,000 prompts using `onnxruntime-genai`. The evaluation driver must use the `onnxruntime-genai` generation API rather than a custom token-generation loop around `InferenceSession`. Run exactly the same prompts and generation limits once with CUDA and once with CPU. Pin and record the `onnxruntime-genai` revision, model configuration, provider configuration, tokenizer, sampling parameters, and random seed. The prompt set should contain long single-request generations and heterogeneous conversational or instruction prompts so the traces expose different routing-locality patterns.

For every prompt and execution provider, record:

- complete per-iteration `model_run` time and, where available, per-expert execution time;
- the concrete dimensions of every input for every iteration;
- the complete ordered sequence of selected `(layer_id, expert_id)` pairs;
- router weights and generated-token counts;
- all metadata required to reproduce and compare the CPU and CUDA runs.

Trace collection does not require an adaptive cache. Running the 1,000-prompt evaluation is the only activity in this plan that is not delivered through a pull request. The raw traces remain evaluation artifacts. All scripts used to process them, all aggregate results, and every update to this document are committed to the repository through pull requests.

After implementation, report:

- expert-frequency distributions and complete routing sequences;
- measured cache-hit rate per layer and overall;
- host-to-device bytes, transfer count, and copies completed before the next token;
- exchanges, global redistributions, and CPU fallbacks;
- time spent waiting for an exchange at the next invocation;
- measured latency, throughput, and peak CPU and CUDA memory;
- output agreement with the existing CPU and CUDA implementations.

Report kernel-only timing separately; it is not the decision metric.

## Deferred analyses

The following trace analyses are not prerequisites for PR 5 and are not planned as PR 3 or PR 4 work. They remain
possible follow-up investigations if measured implementation results justify them:

- expert-frequency concentration and capacity required for a target coverage;
- frequency drift across requests and generation phases;
- run lengths, reuse distance, and achievable LRU and LFU hit rates;
- per-layer differences in expert popularity;
- first- and higher-order transitions between selected experts;
- correlation between router weight and near-future reuse;
- an offline optimal cache trace as an upper bound.

Any future policy comparison should use a trace prefix for parameter selection and held-out tokens and workloads for
evaluation.

### Mixing CPU and CUDA measurements

A checked-in script combines the CPU and CUDA measurements to estimate hybrid execution. For each expert decision, it applies the measured CPU cost to a simulated miss, the measured CUDA cost to a hit, and the measured host-to-device cost to an admission. It then compares the resulting estimated iteration time with the CPU-only and CUDA-only baselines.

CPU and CUDA may generate slightly different responses from the same prompt. Once a generated token differs, subsequent model inputs, routing decisions, and expert sequences may also differ, so records must not be joined only by prompt and iteration index.

The evaluation and mixing scripts therefore:

- record a stable hash of the complete token input for every iteration, without storing prompt text;
- join CPU and CUDA iterations only when `request_id`, input dimensions, and input-token hash all match;
- record the first divergent iteration for each prompt and stop paired comparisons after that point;
- report generated-token and expert-selection agreement before divergence;
- simulate the hybrid policy independently on the complete CPU and CUDA routing traces;
- report both simulation results rather than presenting one merged trace when expert choices differ.

For a directly paired CPU/CUDA cost comparison, the evaluation script also supports deterministic replay of one canonical generated-token sequence on both execution providers. Free-running generation remains the end-to-end correctness measurement; canonical replay isolates execution-provider timing from autoregressive output divergence.

A separate predictive analysis measures:

- `P(expert_n | expert_{n-1})` and mutual information between adjacent layers;
- prediction accuracy and copy lead time for inter-layer prefetching;
- the relation between experts in layer `n` and the top-1 token predicted from layer `n - 1`;
- the ability to predict the next selected expert from the current MoE output;
- the additional gain of token prediction over expert correlation alone.

The intermediate top-1 token requires an extra projection through final normalization and the language-model head. Log it only on sampled tokens or compute it offline, and exclude its cost from policy timing.

Output-based prediction treats two targets separately:

- the expert selected by the next MoE layer for the same token;
- the expert selected by the same MoE layer for the next token.

Train candidate predictors on a trace prefix and evaluate them on held-out requests. Report prediction accuracy, copy lead time, transfer waste from incorrect predictions, predictor runtime, and incremental gain over expert-transition statistics alone.

## Literature-informed candidates

No replacement policy dominates across all MoE models, workloads, capacities, and hardware. Published work does consistently indicate that routing contains exploitable structure and that transfer scheduling matters at least as much as replacement policy.

| Work | Implication for this plan |
|---|---|
| [Fast Inference of Mixture-of-Experts Language Models with Offloading](https://arxiv.org/abs/2312.17238) | Include recency as a baseline and measure consecutive-token locality. |
| [MoE-Infinity](https://arxiv.org/abs/2401.14361) | Compare the proposed cheap online counters with activation-aware historical trace matching and prefetching. |
| [Fiddler](https://arxiv.org/abs/2402.07033) | Model CPU execution as a credible non-blocking miss path, not every miss as a mandatory weight-transfer stall. |
| [SiDA-MoE](https://arxiv.org/abs/2310.18859) | Evaluate distinct workloads because expert popularity may be input-dependent. |
| [Pre-gated MoE](https://arxiv.org/abs/2308.12066) | Model when predictions become available and whether transfers finish before the target layer. |
| [ExFlow](https://arxiv.org/abs/2401.08383) | Preserve layer ordering and evaluate conditional transitions. |
| [ProMoE](https://arxiv.org/abs/2410.22134) | Consider chunked and cancellable copies; include predictor training and runtime costs. |
| [HOBBIT](https://arxiv.org/abs/2411.01433) | Evaluate multiple time scales; keep mixed-precision fallback outside the exact-weight experiment. |
| [Klotski](https://arxiv.org/abs/2502.06888) | Keep multi-request throughput results separate from single-request token latency. |
| [HybriMoE](https://arxiv.org/abs/2504.05897) | Consider impact-aware scoring based on miss cost and expected reuse. |
| [FreeToken](https://arxiv.org/abs/2608.16157) ([code](https://github.com/FlashML-org/FreeToken)) | Refines the same CPU/GPU expert split with a bandwidth-adaptive policy selecting how many experts run on CPU, global LRU expert caching, and runtime re-allocation of device memory between expert cache and KV cache; use it as a reference point for the cache-sizing and CPU-execution-versus-transfer trade-off. |

If simulation work is resumed, its minimum set is hindsight static, cumulative LFU, LRU, decayed or windowed LFU, and
an offline optimal bound. Inter-layer transition prediction and impact-aware scoring are the first advanced candidates.

## Correctness and concurrency

The cache manager owns all mutable state and exposes a mapping snapshot for one inference. Concurrent requests may share immutable CPU weights but must not mutate a session cache without synchronization. The initial implementation serializes placement updates; later parallel execution may use versioned snapshots and per-slot events.

Tests must cover:

- partial and full CPU offload targets expressed as counts and proportions;
- invalid, missing, and partially specified initial counter state;
- all-zero uniform placement and deterministic ties;
- exponential counter updates and the exact epsilon threshold boundary;
- deterministic exchanges;
- repeated hits without additional copies;
- eviction without releasing or modifying CPU weights;
- no cache copy before the current MoE invocation completes;
- asynchronous exchange enqueue immediately after MoE completion;
- waiting for an incomplete exchange before the node's next invocation;
- atomic publication after the copy completes;
- end-of-inference redistribution that maximizes complete CUDA-resident nodes;
- safe slot reuse after CUDA completion;
- counter reset, export, and replay;
- complete logs and explicit buffer-overflow errors;
- numerical agreement for mixed CPU/CUDA expert execution.

## Pull request plan

Every persistent change is delivered through one of the following pull requests. Model evaluations produce raw
measurements outside the repository; scripts and aggregate results remain pull-request changes.

### PR 1: expert-routing instrumentation (complete)

- Add the `session.enable_moe_expert_statistics` session configuration entry, disabled by default.
- Emit compact JSON routing records through the normal ORT logger without enabling general profiling.
- Record the request identifier, MoE node, selected experts, router weights, row count, top-k, and execution device.
- Test the disabled path, JSON schema, CUDA and CPU routing, and explicit overflow behavior.

### PR 2: reproducible evaluation and statistical-analysis scripts (complete)

- Add an `onnxruntime-genai` script that runs a fixed set of 1,000 prompts with identical generation settings on CPU and CUDA and enables expert statistics in its generated session configuration.
- Pin and validate the supported `onnxruntime-genai` revision and record its generation and provider configurations in every run.
- Add checked-in scripts that validate, normalize, and join the CPU and CUDA traces.
- Detect sequence boundaries from decreases in the sequence-length input dimension and validate them against request identifiers.
- Add input-token hashes, divergence detection, and canonical-sequence replay.
- Add a script that mixes measured CPU execution, CUDA execution, and transfer costs to estimate hybrid execution.
- Compute expert frequencies, per-layer distributions, reuse distance, transitions, full-iteration timing summaries, and CPU/CUDA comparisons.
- Add small synthetic fixtures and tests so the analysis is reproducible without the full evaluation artifacts.
- Document the exact commands, inputs, outputs, model revision, and hardware metadata.

**PRs 1 and 2 are independent deliverables and remain useful independently of the cache implementation.**
They provide reusable MoE routing instrumentation, reproducible `onnxruntime-genai` evaluation, CPU/CUDA comparison, and statistical-analysis
tooling for testing other placement, scheduling, prefetching, quantization, or kernel ideas.
They must not depend on the cache implementation introduced by later PRs.

### Model evaluation outside a PR (no longer required before implementation)

The checked-in driver remains available for future validation, but the 1,000-prompt CPU/CUDA evaluation is no longer a
prerequisite for the runtime implementation.

### PR 3: skipped

The trace simulator and measured-results gate are intentionally skipped. The selected runtime policy uses exponentially
decayed per-expert counters and is implemented directly in PR 5.

### PR 4: skipped

Predictive-strategy analysis, sampled intermediate outputs, and predictor training are outside the implementation path.

### PR 5: runtime cache manager

- Add and validate the global `session.moe_cpu_offload_experts` count-or-proportion option; absence preserves current
  behavior.
- Add configurable `alpha`, `beta`, and `epsilon` policy parameters.
- Maintain a decayed counter for every expert of every `MoE` and `QMoE` node.
- Load optional initial counter values from `session.moe_expert_counter_state_file`; initialize all unspecified values
  to zero.
- For all-zero state, distribute CUDA residency uniformly across nodes with deterministic remainder handling.
- Verify that prepacking and memory planning retain canonical weights on CPU without allocating all expert weights on CUDA.
- Extend the existing CUDA `MoE`/`QMoE` implementation with hybrid dispatch and shared CPU expert-compute helpers.
- Use an internal graph-transformer-inserted `MoEWithCPUOffload` operator only if the existing input-memory contract makes the schema-preserving approach infeasible.
- Keep every expert's canonical weights permanently resident and executable on CPU, regardless of CUDA residency.
- Add CUDA slots containing copies only, immutable mapping snapshots, and deterministic ranking.
- After each node invocation, apply the `cpu_max > (1 + epsilon) * cuda_min` threshold and enqueue at most one
  asynchronous expert exchange.
- Require an in-flight exchange to finish before that node's next invocation, then atomically publish the new mapping.
- After each complete inference, redistribute the global CUDA budget across nodes, first maximizing the number of
  complete nodes that can execute entirely on CUDA and then maximizing retained counter mass.
- Cover option parsing, initial-state loading, zero-state uniform placement, counter decay, threshold boundaries,
  global redistribution, asynchronous transfer, synchronization, concurrency, and CPU fallback with tests.

### PR 6: fused Qwen 3.6 MoE integration

- Integrate the cache manager with the fused Qwen 3.6 CUDA MoE operator.
- Preserve a permanent CPU copy of every expert and use CUDA slots only as disposable cache copies.
- Execute all non-resident experts on CPU for the current MoE invocation, then schedule a qualifying exchange after
  that invocation completes.
- Add mixed CPU/CUDA expert dispatch and numerical correctness tests.
- Add end-to-end bounded-memory tests for partial and full CPU offload targets.

### PR 7: end-to-end results

- Use measurements produced by repeating the out-of-PR CPU and CUDA evaluation against the completed implementation.
- Compare measured behavior with the CPU-only and CUDA-only baselines.
- Add time to first token, inter-token latency, throughput, memory, transfer, hit-rate, and output-agreement results.
- Update this document with the final conclusion and move it to the appropriate next-step status.

## Going further: per-expert limits and MoE-output prediction

This is not a planned pull request or a requirement for completing the implementation. It describes possible follow-up research after PR 7 establishes a correct end-to-end baseline.

- Study whether one global cache policy should be supplemented by limits keyed by `(layer_id, expert_id)`.
- Keep the global CUDA memory capacity as a hard safety bound while evaluating per-expert admission, retention, or replacement limits derived from expert size, copy cost, CPU cost, CUDA speedup, and observed reuse.
- Extend the existing profiler-based JSON logging with explicitly sampled MoE outputs, including their source iteration, token, layer, shape, and type.
- Add checked-in scripts that predict the next-layer expert and the same-layer next-token expert from those outputs.
- Compare raw sampled outputs with compact deterministic projections to quantify trace size, logging overhead, and predictive accuracy.
- Simulate prefetching driven by the predictor and include late copies, unused copies, and predictor cost.
- If this research is pursued, document its results separately and propose implementation work only when it improves the PR 7 baseline.

## Going further: multiple CUDA devices

The initial implementation remains limited to one CUDA device, but the same trace-first plan can be extended to several CUDA devices installed in one machine. Every expert still has one canonical CPU copy. Each CUDA device owns an independent bounded cache containing copies of selected experts.

PR 1 records the execution device only. Cache-slot devices, transfer endpoints, and visible-device topology belong to
the later cache and multi-device phases because no such placement or transfer exists in PR 1.

The multi-device study should distinguish two use cases:

- **Independent request placement:** each request runs on one CUDA device and uses only that device's expert cache.
- **Cross-device expert execution:** one request may dispatch experts to several CUDA devices and must transfer activations and outputs between them.

The first mode extends the single-device cache directly. The second requires explicit modeling of PCIe or NVLink topology, peer-to-peer availability, activation-transfer cost, synchronization, and output aggregation. It must not be assumed beneficial merely because aggregate CUDA memory increases.

Extend the study in the same order:

1. Measure CPU-to-device and device-to-device bandwidth and latency for every relevant pair.
2. Replay the existing traces with one capacity and cache state per CUDA device.
3. Compare replicated, statically sharded, and adaptive expert placement.
4. Include device assignment in every hit, miss, admission, eviction, and transfer.
5. Estimate end-to-end iteration time including activation movement, expert computation, weight copies, and synchronization.
6. Implement multi-device dispatch only if the simulator improves the best single-device result.

A future configuration may generalize `session.moe_cpu_offload_experts` to a per-device placement map while preserving
the existing single-device option. Cache identity then becomes `(cuda_device_id, slot_id)`, and one expert may have
copies on zero, one, or several CUDA devices while its CPU weights remain permanently resident.

## Decision criteria

Success requires a correct bounded-memory MoE implementation that honors the global offload target, keeps canonical
weights on CPU, updates counters deterministically, completes asynchronous exchanges before the affected node runs
again, and maximizes complete CUDA-resident nodes during end-of-inference redistribution. End-to-end measurements must
show the memory, latency, throughput, transfer, hit-rate, and output-agreement effects relative to CPU-only and
CUDA-only baselines.

PRs 1 and 2 remain the common instrumentation and evaluation foundation if the selected policy later needs tuning.
