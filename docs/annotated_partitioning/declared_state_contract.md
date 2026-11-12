# KV-Cache Declaration Contract for CPU/GPU Placement (K1)

The KV owner declares storage requirements **before allocation**; ORT plans placement; the owner allocates at the
committed locations. GenAI, an external harness, and ORT-managed KV use the same contract. This is K1 in the
[placement roadmap](https://github.com/microsoft/onnxruntime/pull/33372), built on [S0](capability_planning_interfaces.md).

**Implemented:** strict JSON parsing, graph resolution for dense GQA and unquantized paged pools/block tables,
checked backing bounds, and capacity/alias projection into S0. These are internal helpers in
[`declared_state.h`](../../onnxruntime/core/framework/declared_state.h), not an enabled session feature.
**Not implemented:** producers, paged allocation, estimator adaptation, M2 accounting, admission, placement selection,
or bind-time enforcement. Successful parsing/resolution does not admit a plan or prove kernel support on either device.

## KV-colocation experiment: performance and transfer savings

The reported GPT-OSS-20B v1 experiment kept GQA layers 12-15 on CPU and MoE/projections on GPU.
For the same mixed-placement cut, it compared keeping KV on GPU with colocating KV with each attention consumer:

| KV capacity (tokens) | GPU-resident KV (ms/token) | Colocated KV (ms/token) |
|:--------------------|--------------------------:|-----------------------:|
| 256 | 14.04 | 13.20 |
| 2,048 | 20.02 | 12.42 |
| 8,192 | 32.92 | 11.75 |
| 32,768 | 84.91 | 12.49 |

At 32,768 capacity, colocation gave a **6.8x speedup over the same CPU cut with GPU-resident KV**,
eliminating **512 MiB of KV transfers per token** and moving **256 MiB of KV backing off GPU**.
GPU-only execution remained faster at **10.39 ms/token**; this is evidence for reducing offload's transfer cost,
not a claim that mixed placement outperforms GPU-only execution.

These are synchronized capacity measurements with a **99-token prompt and 64 decode steps**, not 32K active-context
inference. The two KV layouts produced bit-identical logits and unforced generation for the same mixed-placement cut.
CPU-versus-GPU compute still differed in one top-1 prediction out of 65, so cross-device compute equivalence is not
certified. The experiment is separate prototype evidence, not a K1/GenAI integration test or an admission guarantee.

This motivates K1's declare/plan/query/allocate handshake: moving attention compute to CPU is insufficient if its KV
still requires GPU/CPU copies at every step. `follow_consumer` must guide per-cache allocation to the selected
attention implementation's required memory location, independently of whether GenAI, a harness, or ORT owns the buffers.

## Simplified end-to-end workflow

The integration below is proposed; this PR implements only declaration/resolution/bounds/projection.

| Step | Responsibility |
|:-----|:---------------|
| 1. Declare, before session creation | The owner translates configuration/package metadata and caller workload bounds into K1. It does not choose the CPU/GPU cut. |
| 2. Resolve | ORT matches graph ports and derives dtype, layout, heads, and head size; malformed or unsupported state fails. |
| 3. Plan and commit | For each candidate, KV follows the consumer's input-memory requirements. M2 attributes backing, R3 prices transfers, I1 checks budgets, I3 selects, and I2 rechecks before commitment/allocation. |
| 4. Query | The owner reads committed per-cache locations. Dense pairs use `SessionGetMemoryInfoForInputs/Outputs`; paged integration must also expose pool/table geometry, limits, and plan identity. |
| 5. Allocate and bind | The owner selects allocators/providers per cache/device, enforces the envelope, and binds the declared storage. |
| 6. Validate and run | ORT checks dtype, geometry, alias, lifetime, capacity, locations, and paged mappings/shared-write rules. No unplanned staging or growth above admitted bounds. |

GenAI's producer translates `genai_config.json`; a harness translates CLI/config; ORT chain/session-creation glue
translates package cache specifications **before creating each step's session**. All require the caller's envelope.
A package's `initial_capacity` is not a maximum; `update: share` may mean output adoption, not input/output aliasing.

`follow_consumer` means the selected kernel's memory requirements, not merely its EP label. Query both cache input
and output locations. Identical pairs require compatible device, device ID, and memory type. A shared block table's
consumers must also agree on its location; otherwise integration must reject the cut or explicitly model separate
backing/copies. One loop-wide `cache_provider` cannot implement mixed CPU/GPU cache placement.

## JSON contract

**Version 1 remains dense-only. Version 2 also accepts `paging`.** Omitting `paging` selects dense GQA in either version.
Each state entry describes one K **or** V input/output pair; declare both separately. Example: two layers sharing one
block-table graph input, with session-wide pools covering two resident generators:

```json
{
  "version": 2,
  "max_batch_size": 2,
  "max_sequence_length": 64,
  "max_live_instances": 2,
  "states": [
    {
      "id": "key", "kind": "kv_cache",
      "input": "past.{layer}.key", "output": "present.{layer}.key",
      "layers": [0, 1], "capacity": 64, "alias": "past_present",
      "lifetime": "per_session", "location": "follow_consumer",
      "paging": {"block_size": 16, "num_blocks": 16, "block_table": "block_table"}
    },
    {
      "id": "value", "kind": "kv_cache",
      "input": "past.{layer}.value", "output": "present.{layer}.value",
      "layers": [0, 1], "capacity": 64, "alias": "past_present",
      "lifetime": "per_session", "location": "follow_consumer",
      "paging": {"block_size": 16, "num_blocks": 16, "block_table": "block_table"}
    }
  ]
}
```

| Field | Meaning |
|:------|:--------|
| `version` | Required integer `1` or `2`. |
| `max_batch_size` | Positive per-instance batch bound, including beam expansion. |
| `max_sequence_length` | Positive absolute total-length bound, past plus new tokens; not a sliding-window size. |
| `max_live_instances` | Positive bound on resident generator/state sets, including retained versions. Serialized Runs do not imply a single resident set. |
| `id`, `kind`, `layers` | Unique nonempty group ID; `kind: kv_cache`; nonempty, distinct nonnegative layer IDs. |
| `input`, `output` | Literal graph names or templates with one `{layer}` token. Expanded cache names must be globally distinct. |
| `capacity` | Dense: physical sequence-dimension bound. Paged: per-sequence token bound used to size tables and check pool coverage. |
| `alias` | Dense: `past_present` (identical base storage, no offset) or `none` (disjoint). Paged requires `past_present`. Output adoption is not identical storage. |
| `lifetime` | `per_generator`: one cache/pool per resident instance. `per_session`: one cache/pool shared across instances. |
| `location` | Required `follow_consumer`. Pins and cross-device copy-through are not supported. |
| `window` | Optional nonnegative assertion against graph `local_window_size`; omitted means derive it. A local mask alone does not permit eviction. |
| `paging` | V2-only object: positive `block_size` (tokens/block), positive `num_blocks` (physical pool limit), and `block_table` graph name/template. |

All fields except `window` and V2 `paging` are required. Unknown fields, duplicate object keys/groups/cache names/layers,
invalid enums/types/bounds, and conflicting table declarations fail. Unsupported versions/kinds/location policies return
`NOT_IMPLEMENTED`; malformed declarations return `INVALID_ARGUMENT`. Helpers preserve outputs on failure.

## Storage and backing bounds

Dense GQA derives FP32/FP16/BF16 BNSH shapes from `com.microsoft::GroupQueryAttention` input 3/output 1 and input 4/output 2.
Full caches require `capacity >= max_sequence_length`. Evicting GQA caches require `capacity >= local_window_size`;
the graph's `sliding_window_cache` must enable eviction. CUDA-specific alias/window eligibility remains separate.

V2 paging resolves the same port pairs on `com.microsoft::PagedAttention`, with unquantized FP16/BF16 `SEPARATE` pools.
Input 7 must match a rank-two int32 block table. Pools use `[num_blocks, block_size, kv_heads, head_size]`, not BNSH.
The first slice retains full history even with a local mask: `capacity >= max_sequence_length`; block/sequence bounds
must fit int32. Graph-fixed dimensions and K/V pool shapes must agree, and each cache has one consumer.

Let `B = max_batch_size`, `L = max_live_instances`, `C = capacity`, `P = block_size`, and `N = num_blocks`:

```text
dense tensor S = B * kv_heads * C * head_size * sizeof(dtype)
paged pool   S = N * P * kv_heads * head_size * sizeof(dtype)
table        T = B * ceil(C / P) * sizeof(int32)
```

`ComputeDeclaredStateBackingBounds` checks overflow and reports cache bytes, block-table bytes, and their sum:

- Dense pairs charge `S` when identical or `2*S` when disjoint. Multiply per-generator backing by `L`.
- Paged pairs charge the **whole pool**, including free blocks: `S` per-session or `L*S` per-generator. K and V are separate pools.
- A per-generator pool requires `N >= B*ceil(C/P)`; a session pool requires `N >= L*B*ceil(C/P)`. No speculative prefix-sharing credit.
- Block tables are per-generator backing: charge `L*T` once per unique graph input, even when K/V or layers share it.

With FP16 caches, `kv_heads = 2`, and `head_size = 4`, the example bounds four 4096-byte pools plus two 32-byte table
instances: **16448 bytes**. This is a declared tensor-backing bound, not a reservation or whole-session budget.
Allocator padding/overhead, CPU/device table mirrors,
other metadata, workspace, snapshots outside the envelope, and initialization/growth/copy overlap need separate accounting.
Copy-on-write blocks inside an already-counted pool consume its free capacity, not an additional backing charge.
Growth or retained state beyond the limits requires new admission.

## S0 projection and remaining integration

`ProjectDeclaredStateToPlanningQuery` adds actual dense/pool/table shapes to `max_input_shapes`, per-pair alias facts to
`node_constraints`, and batch/absolute-length bounds to `workload_bounds`. It does not charge KV as L1 workspace.
Owner, lifetime, and instance multiplicity stay with ORT. Shared tables retain all consumer indices for later placement checks.
Projection is atomic, preserves EP pins/policies, and rejects conflicting facts or stale graph identity/revision.
Use an unpublished query and a fresh immutable snapshot whenever estimation facts change; backing changes require re-admission.

Resolution covers one resolved, unfiltered graph with cache pairs and tables at graph boundaries, not initializer-backed state.
Paged table sharing is accepted only among recognized PagedAttention input-7 uses. Nested/internal state, quantized caches,
LATENT/MLA, sparse paging, and other recurrent adapters remain unsupported. Strict whole-model integration must reject
unsupported state instead of treating it as zero; these helpers alone do not certify an arbitrary model.

K3 connects session creation, commitment, and bind-time validation; M2/R3/I1 consume backing facts; C1/C2 adapt estimators.
K4/K5 add GenAI/ORT owner allocation, including block reservation, reference counts, copy-on-write, and completion-safe reuse.
Paged declarations do not add CPU kernels or prove EP eligibility. K6 must explicitly price any host-storage/GPU-compute
offload; ordinary I/O staging is not a repair for a placement mismatch.

The tentative transport remains `session.declared_state`, **not registered or consumed by this PR**. Integration must
handle the current 8192-byte config-value limit and provide positive versioned support negotiation. Adding an arbitrary
config entry, checking the ORT version, or finding the location APIs is not feature detection. Legacy behavior is unchanged.
