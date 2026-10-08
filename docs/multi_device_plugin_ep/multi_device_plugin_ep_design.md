# Multi-Device Plugin EP: Design & Change Plan

Status: Draft for review
Scope owner: (TBD)
Related areas: plugin EP (`OrtEp`/`OrtEpFactory`), graph partitioning, allocation planning, public session device-query API, allocators, data transfer.

## 1. Problem statement

Today an `IExecutionProvider` is registered with a **single** `default_device_`
(`include/onnxruntime/core/framework/execution_provider.h`). Per-value device placement is derived
entirely from that one device: the allocation planner calls
`exec_provider->GetOrtDeviceByMemType(mem_type)` for every input/output `NodeArg`
(`onnxruntime/core/framework/allocation_planner.cc`), and `GetOrtDeviceByMemType` only ever returns
`default_device_` (for `Default`) or a CPU `OrtDevice` (for the CPU mem types).

Modern accelerators expose **multiple internal devices** behind one EP (e.g. an SoC EP that owns a
GPU *and* an NPU, or a discrete card with device + host-pinned memory as distinct placement targets).
Such an EP currently cannot:

1. Tell ORT that *this fused node's input 0 lives on the NPU while input 1 lives on the GPU*.
2. Have the public device-query API (`SessionGetEpDeviceForInputs/Outputs`) report the *specific*
   internal device for each I/O, so the user can pre-place data and avoid host round-trips.
3. Cleanly expose a **shared allocator per non-CPU device** it uses so the user can allocate on-device
   and drive `CopyTensors` to the right target.

Much of the plumbing already exists in pieces:

- `OrtEpDevice` already carries `device_memory_info` / `host_accessible_memory_info`
  (`onnxruntime/core/session/abi_devices.h`).
- A `PluginExecutionProvider` is already constructed with a *list* of `OrtEpDevice`s and already
  collects `allocator_mem_infos_` for all of them
  (`onnxruntime/core/session/plugin_ep/ep_plugin_provider_interfaces.cc`).
- The environment already registers a shared allocator per device / host memory info
  (`onnxruntime/core/session/environment.cc`).
- `CreateSyncStreamForDevice` and `CreateAllocator` are already parameterized by
  `OrtMemoryDevice` / `OrtMemoryInfo`.

What is missing is a way to **plumb a per-I/O device choice from `GetCapability` down through the
plan**, plus the public-API refinements and validation.

## 2. Current-state analysis (single-device assumptions to break)

| Layer | File | Current behavior | Blocks multi-device |
|---|---|---|---|
| EP device model | `include/onnxruntime/core/framework/execution_provider.h` | one `default_device_`; `GetOrtDeviceByMemType` returns it | yes |
| Plugin EP device | `onnxruntime/core/session/plugin_ep/ep_plugin_provider_interfaces.cc` | collapses ep_devices to **one** `OrtDevice` (`GetOrtDeviceForPluginEp`) | yes |
| Capability expression | `onnxruntime/core/framework/compute_capability.h`, `onnxruntime/core/session/abi_ep_types.h` | `IndexedSubGraph` + `MetaDef`; plugin `OrtEpGraphSupportInfo` groups nodes only | yes — no place to put a device |
| Placement planning | `onnxruntime/core/framework/allocation_planner.cc` | device = `GetOrtDeviceByMemType(kernel mem type)` | yes |
| Node info / session state | `onnxruntime/core/framework/session_state.h` (`NodeInfo.device`); `onnxruntime/core/session/inference_session.cc` | derives one `OrtDevice` per value from the plan | works once plan is correct |
| Public: memory info per I/O | `inference_session.cc` -> `SessionGetMemoryInfoForInputs/Outputs` | already returns a per-I/O `OrtMemoryInfo` | **already device-granular** |
| Public: EpDevice per I/O | `inference_session.cc` (`GetEpDeviceForInputs/Outputs`) | matches by **ep_name only** | yes — ambiguous when EP has >1 device |
| Input single-device rule | `onnxruntime/core/framework/session_state.cc` (`AddInputNameToNodeInfoMapping`) | input consumed on two devices -> `NOT_IMPLEMENTED` | constrains fan-out |
| Cross-device copies | `onnxruntime/core/optimizer/transformer_memcpy.cc` | inserts copies only between provider vs non-provider | intra-EP copies not modeled |
| Shared allocators | `onnxruntime/core/session/environment.cc` | one per `device_memory_info` / `host_accessible_memory_info` | works; needs per-device coverage + validation |

**Key scoping insight:** the story is *dramatically* simpler for **compiling / fused EPs** than for
kernel-based EPs. A fused node is opaque to ORT — any device-to-device movement *inside* it is the
EP's responsibility. ORT only needs correct device tags on the **boundary inputs/outputs** of each
fused node. Kernel-based multi-device placement (per standard-op kernel) would require the allocation
planner, stream assignment, and MemcpyTransformer to reason about two non-CPU devices owned by one EP,
which is a much larger change.

Recommendation: scope Phase 1-3 to **compiled / fused plugin EPs**, and treat kernel-based
multi-device as a later, optional phase.

## 3. Options considered

### Option A — Per-I/O device carried on `ComputeCapability` (recommended)

Attach an optional per-I/O device list to the returned capability: for the capability's node(s), a
device (as `OrtMemoryInfo`) per input and per output. Plumb through the plugin API (for both the
single-node and fused grouping paths), store on `ComputeCapability` (positional per I/O slot), harvest
it into a `SessionState` override map at partition time, and have the allocation planner honor the
override instead of `GetOrtDeviceByMemType`.

- Pros: push model matches how EPs already describe capability; naturally per-value; localized to
  capability creation + planner; no new EP callbacks; directly answers the "each input/output" ask.
- Cons: requires a new internal field + ABI struct + planner branch; must validate every referenced
  device has a registered allocator.

### Option B — EP-level device set + extended kernel mem-type mapping

Give the EP a set of devices and let `GetOrtDeviceByMemType` (or a widened variant) return arbitrary
devices per mem-type slot.

- Pros: smallest change to the planner (it already calls `GetOrtDeviceByMemType`).
- Cons: **fundamentally cannot address a specific value.** `GetOrtDeviceByMemType(OrtMemType)`
  receives only a coarse enum (`Default`/`CPUInput`/`CPUOutput`) — no node, no I/O slot, no scope. It
  therefore cannot express *different* devices for two different `Default` inputs of the same node, and
  it has no way to identify *which* value it is being asked about. This matters because value **names
  can be shadowed across nested subgraph scopes** (If/Loop/Scan bodies reuse names), so the only place
  that can correctly resolve a name to a concrete value is the planner's per-graph-level `Index(name)`
  (which builds on the scope-aware `OrtValueNameIdxMap`). A function that receives just a mem-type enum
  has none of that context. Fails the "per input/output" requirement outright. Rejected as the primary
  mechanism; kept only as the **default fallback** when no override is supplied.

### Option C — Pull callback on the EP

New callback `OrtEp::GetInputOutputMemoryDevice(fused_node, io_kind, index, OrtMemoryDevice**)`
consulted by the planner.

- Pros: no capability-struct changes; lazy.
- Cons: planner must call back into the EP mid-planning (ABI cost, ordering); harder to validate up
  front; duplicates info the EP already has at `GetCapability` time. Keep as a fallback shape but
  prefer A.

Recommendation: **Option A** as the primary mechanism, reusing Option B's `GetOrtDeviceByMemType`
only as the default when no override is supplied, keeping the existing single-device path 100% intact
for legacy EPs.

## 4. Detailed change list (Option A)

### 4.1 Internal capability representation

- Add an optional structure on **`ComputeCapability`** (`compute_capability.h`) describing boundary
  placement. Key it **positionally by I/O slot**, not by NodeArg name:

  ```cpp
  // Optional. Device the EP wants each I/O of the (fused or single) node placed on.
  // Indexed positionally to match the node's input/output defs:
  //   input_devices[i]  applies to the node's input slot  i
  //   output_devices[j] applies to the node's output slot j
  // A default-constructed OrtDevice entry means "no override; use GetOrtDeviceByMemType".
  struct DevicePlacement {
    InlinedVector<std::optional<OrtDevice>> input_devices;
    InlinedVector<std::optional<OrtDevice>> output_devices;
  };
  std::optional<DevicePlacement> device_placement;
  ```

  **Why positional, not name-keyed.** NodeArg names can be **shadowed across nested subgraph scopes**,
  so a bare name is not a globally unique key. The only component that resolves a name to a concrete
  value is the planner's per-graph-level `Index(name)` (backed by the scope-aware `OrtValueNameIdxMap`).
  At every device-assignment site the planner already iterates the node's defs *with their slot index*
  (`Node::ForEachWithIndex(pnode->InputDefs(), ...)` and the output `for i`), so `(node, slot)` is the
  unambiguous identity we should carry, and the planner maps it to the scope-correct `OrtValueIndex`
  for us. For a fused capability slot `i` corresponds to `MetaDef.inputs[i]`; for a single-node
  capability it is the node's own def at slot `i`.

  **Store on `ComputeCapability`, not on `MetaDef`.** `MetaDef` exists *only* for fused/compiled
  capabilities (multi-node `IndexedSubGraph`); a single-node capability has no `MetaDef`. Hanging the
  map on `ComputeCapability` keeps the structure present for **both** fused and single-node
  capabilities, so it does not architecturally exclude non-compiling / kernel-based EPs (see the
  "structural support vs. execution reconciliation" note below).
- **Rejected: storing the device on `NodeArg`.** A `NodeArg` is shared by the producer and *all*
  consumers and is part of the serialized IR; it cannot represent per-consumer / multi-device placement
  (that is precisely why the planner models `OrtValue -> OrtDevice` after reconciling all consumers),
  and device is an execution concern that must not leak into or persist in the model.
- Provider-bridge wrappers: add accessors in
  `onnxruntime/core/providers/shared_library/provider_wrappedtypes.h` and the host impl in
  `onnxruntime/core/session/provider_bridge_ort.cc` so out-of-tree / provider-bridge EPs can set it.

### 4.2 Plugin C API surface (`include/onnxruntime/core/session/onnxruntime_ep_c_api.h`)

Device placement is a property of the **grouping**, not of fusion. `OrtEpGraphSupportInfo` produces a
`ComputeCapability` for **either** grouping kind
(`ep_plugin_provider_interfaces.cc::GetCapability`):

- `EpGraphSupportInfo_AddSingleNode` -> `kSingleAssignedNode` -> single-node capability (no `MetaDef`);
  executed by a **static `OpKernel`** from the EP's kernel registry (`TryAssignSingleNode` in
  `graph_partitioner.cc`).
- `EpGraphSupportInfo_AddNodesToFuse` -> `kFusedNode` -> capability **with** `MetaDef`; either
  `Compile()`-d (opaque) or matched to a predefined fused kernel. A single node may also be wrapped
  here and compiled ("even with single node, EP might still want to compile it").

Because `device_placement` lives on `ComputeCapability` (§4.1), the C API should let the EP attach it
to **both** paths, not just fusion. Preferred shape \u2014 a grouping-agnostic call the EP issues after
adding a node/group:

- `EpGraphSupportInfo_AddNodeIODevice(graph_support_info, const OrtNode* node, bool is_input,
  size_t io_index, const OrtMemoryDevice* device)`. ORT records `(node, is_input, io_index) ->
  device` on `OrtEpGraphSupportInfo`, then when it builds each capability it matches these entries to
  whichever grouping contains `node` and writes them positionally into `ComputeCapability::device_placement`
  (aligned to `MetaDef.inputs/outputs` for a fused grouping, or the node's own `InputDefs/OutputDefs`
  for a single assigned node). This decouples placement from fusion entirely and works uniformly.
  - The EP specifies device **per `(OrtNode*, is_input, io_index)`** \u2014 concrete node + slot, which is
    unambiguous and matches how the EP sees the graph; the EP never keys by, or reasons about the
    uniqueness of, NodeArg names.
  - For a fused grouping, an entry whose `(node, slot)` is *interior* to the group (not on the group's
    outer boundary) is rejected with a clear error \u2014 only the group's outer I/O placement is meaningful.
  - The `OrtMemoryDevice*` must correspond to an `OrtMemoryInfo` the EP registered via
    `EpDevice_AddAllocatorInfo`.
- Alternative shape (if a per-call carrier is preferred over a separate call): `_V2` variants of both
  `AddSingleNode` / `AddNodesToFuse` that take an `const OrtNodeIODevice* io_devices` array. Two V2
  entry points vs. one orthogonal call \u2014 the orthogonal call is less surface area.
- Update `abi_ep_types.h` `NodeGrouping` to store the collected per-slot devices for its nodes, and
  `abi_ep_types.cc` to carry them; translate into `ComputeCapability::device_placement` in
  `PluginExecutionProvider::GetCapability` (`ep_plugin_provider_interfaces.cc`).

> **Which groupings actually honor differing per-I/O devices (execution model, not API).** An
> **EP-compiled** grouping (fused, single or multi-node) is opaque, so per-I/O device tags are honored
> in Phase 1. A **single assigned node run by a static `OpKernel`** expects all its inputs on the
> kernel's one compute device; Phase 1 honors only a *uniform* device for such a node (place all its
> I/O on one of the EP's devices), and differing per-input devices for it are the Phase 4
> reconciliation (§4.4, §4.8). The C API accepts placement for both kinds; the *honoring* depends on
> the execution model, consistent with §4.4.

- Document that all referenced devices **must** be backed by an `EpDevice_AddAllocatorInfo`
  registration and a working `CreateDataTransfer` that `CanCopy` to/from those devices.

### 4.3 Multiple devices on the (plugin) EP itself

- Relax `GetOrtDeviceForPluginEp` (`ep_plugin_provider_interfaces.cc`): keep choosing one *default*
  device (needed for stream/LogicStream defaults and legacy paths), but retain the **full device set**
  (already available via `ep_devices_` / `allocator_mem_infos_`).
- Add an internal accessor on `IExecutionProvider` (or the plugin subclass) to enumerate all supported
  `OrtDevice`s, used by validation and by the planner override path. Consider a virtual
  `GetAllOrtDevices()` defaulting to `{default_device_}`.

### 4.4 Placement: partitioning harvests intent, planner resolves it

Use a **hybrid**: partitioning *records* the EP's requested placement into a SessionState override map;
the allocation planner remains the single authority that *resolves* each value's device.

- **Harvest at partition time, keyed by node identity + slot (never a bare name).** When a capability
  is applied, copy its positional `device_placement` into a `SessionState` side-table keyed by
  the **fused `const Node*`** (pointer-stable across the session's graph lifetime) plus I/O slot:
  `map<const Node*, {InlinedVector<optional<OrtDevice>> inputs, outputs}>`. Note the planner's
  `OrtValueIndex` does **not** exist yet at partition time (the `OrtValueNameIdxMap` is built later
  during SessionState finalization), which is another reason the side-table cannot be keyed by value
  index or by an unscoped name here \u2014 it is keyed by the node, which *is* stable, and the planner does
  the value resolution.
- **Planner consults the override first, resolving identity itself.** At each existing `SetLocation`
  site the planner already has the node (`pnode`) and the slot index (`arg_idx` for inputs via
  `Node::ForEachWithIndex`, `i` for outputs) and has already computed the scope-correct
  `OrtValueIndex index = Index(name)`. Look up `override[pnode]` and, if it has an entry for that slot,
  `SetLocation(index, that_device)`; otherwise fall back to `GetOrtDeviceByMemType(...)` exactly as
  today. Because the name is only ever resolved by the planner's own per-graph-level `Index()`, the
  subgraph name-shadowing hazard never arises.
- **Precedence vs. the mem-type defaults.** An explicit boundary override wins over the
  `GetOrtDeviceByMemType(Default)` default. It should also take precedence over the
  `utils::IsInputOnCpu` / `OrtMemTypeCPUInput` heuristic for a **fused** node's boundary I/O (the node
  is opaque; ORT does not itself read those boundary tensors as shape/index inputs). The override must
  still never resolve to a device lacking a registered allocator + data-transfer path (validated up
  front \u2014 see 4.6), so a bad override fails fast rather than silently mis-placing.
- This keeps reconciliation (multi-consumer values, implicit subgraph inputs, and MemcpyTransformer
  copy insertion) in/around the planner where it already lives \u2014 we record intent in partitioning but
  do **not** move resolution there.

  Data flow:
  `EP.GetCapability -> ComputeCapability.device_placement (positional) -> partitioning harvest ->
  SessionState override map (const Node* + slot -> OrtDevice) ->
  AllocationPlanner (has pnode + slot + scope-correct OrtValueIndex; override ? SetLocation : GetOrtDeviceByMemType)`.

- Preserve existing behavior when no override is present (all current EPs unaffected).
- Ensure `GetAllocator(OrtDevice)` in `session_state.cc` can resolve every overridden device (it will,
  provided the shared/NP allocator was registered — see 4.6).

> **Graph-I/O boundary vs. internal-edge boundary (what Phase 1 actually gets for free).**
> Setting a value's planned location also drives the runtime copy. A boundary value that is a **graph
> input/output** is copied to/from the planned device by the existing `utils::CopyInputsAcrossDevices`
> / output-copy machinery, using the EP's data transfer. So for compiled EPs, correctly setting the
> location + registering a `CanCopy` CPU<->device transfer makes pre-placement *optional* and IOBinding
> truthful **with no MemcpyTransformer change**. The harder case is a boundary value that is an
> **internal graph edge** (produced by another node, consumed by a fused node on a *different* device):
> inserting that device-to-device copy is MemcpyTransformer work (§4.8, Phase 4). Phase 1 can require
> the EP to accept such an edge on its default device and move it internally (opaque).

> **Structural support vs. execution reconciliation (why non-compiling EPs are not blocked).**
> Because the override map is fed from `ComputeCapability` (present for single-node capabilities too),
> a kernel-based EP *can* express per-I/O device intent structurally today — nothing about the storage
> choice excludes it. What remains gated is **execution semantics**: a single standard-op `OpKernel`
> runs on one compute stream/device and assumes its inputs are already there. Tagging two inputs of one
> kernel to different non-CPU devices only works once ORT inserts intra-EP device-to-device copies and
> assigns per-device streams — the Phase 4 reconciliation (§4.7, §4.8), which is independent of where
> the device is stored. A **fused** node is opaque, so boundary tags alone suffice — hence Phase 1 is
> scoped to fused/compiled EPs, not because single-node capabilities *can't* carry the map.

### 4.5 Public API — device query per input/output

- **Memory info API already works**: `SessionGetMemoryInfoForInputs/Outputs` returns a per-I/O
  `OrtMemoryInfo` sourced from `NodeInfo.device` (`inference_session.cc`). Once the plan assigns the
  correct device, this API automatically reports it. **No signature change** — it simply gains
  resolution from the planner fix.
- **Fix `SessionGetEpDeviceForInputs/Outputs`**: current lookup matches by `ep_name` only
  (`inference_session.cc`), so a multi-device EP returns an arbitrary `OrtEpDevice`. Change the match
  to `(ep_name AND OrtDevice)`: resolve the planned `NodeInfo.device` for the value, then pick the
  `OrtEpDevice` whose `device_memory_info->device` (or `host_accessible_memory_info->device`) equals
  it. Fall back to ep_name-only when the EP has a single device (legacy behavior preserved).
- No new public function is strictly required. Ensure both inputs and outputs variants consult the
  refined matcher.
- C# / C++ wrappers: existing `GetMemoryInfoForInputs/Outputs` and `GetEpDeviceForInputs` wrappers
  (`include/onnxruntime/core/session/onnxruntime_cxx_inline.h`) need no signature change.

### 4.6 Shared allocators for every non-CPU device the EP returns

- Requirement: the EP must provide a **shared allocator for each non-CPU device** appearing in any
  `ComputeCapability`, so users can `GetSharedAllocator(memory_info)` + `CopyTensors`.
- Environment already creates a shared allocator per `device_memory_info` /
  `host_accessible_memory_info` at registration (`environment.cc`). **Gap:** it only iterates the two
  well-known slots; a genuinely multi-device EP needs an allocator per *each* device it can place
  boundary values on. Options:
  - Require each internal device to be exposed as its own `OrtEpDevice` (already supported — the plugin
    EP takes a *list*), so each gets its `device_memory_info` and thus a shared allocator.
    **Recommended** (least code, fits the existing model).
  - Or generalize `EpDevice_AddAllocatorInfo` to accept N device memory infos per `OrtEpDevice`
    (bigger change to `OrtEpDevice`).
- **Add validation** at session initialization / partition time: for every device referenced in a
  capability's `device_placement`, assert a shared allocator + a data-transfer path
  (`CanCopy`) exists; otherwise fail fast with a clear message (`ORT_INVALID_ARGUMENT`). This is where
  a subtle multi-device bug would otherwise surface as a silent wrong-device copy.

### 4.7 Data transfer & streams

- `plugin_ep::DataTransfer` is registered per factory (`environment.cc`) and its `CanCopy` uses
  `OrtMemoryDevice`. A multi-device EP's `CreateDataTransfer` must return an `OrtDataTransferImpl`
  whose `CanCopy` covers **all pairs** among its devices + CPU that boundary placement can require.
  Document + validate.
- Streams: `IsStreamAware` / `CreateSyncStreamForDevice(memory_device)` are already device-
  parameterized (`onnxruntime_ep_c_api.h`). The stream-assignment logic in the planner keys
  LogicStreams off `GetOrtDeviceByMemType(Default)`; for compiled nodes this is fine (one fused node =
  one stream on its default device). Multi-device *kernel* placement would need per-device streams —
  defer to the kernel-based phase.

### 4.8 The "input consumed on two devices" constraint

- `session_state.cc` (`AddInputNameToNodeInfoMapping`) returns `NOT_IMPLEMENTED` if one graph input
  feeds two different devices. With a multi-device EP, a single graph input could legitimately be
  required on two of the EP's devices (two different fused nodes). Today the MemcpyTransformer only
  mediates provider-vs-non-provider. Decisions:
  - **Phase 1:** keep the constraint; require the EP to internally accept the input on its default
    device and move it (opaque). Document that boundary placement should be self-consistent for shared
    inputs.
  - **Phase 2 (optional):** teach MemcpyTransformer / the copy insertion to insert intra-EP
    device-to-device copies using the EP's data transfer, lifting the restriction.

## 5. Backward compatibility & versioning

- All internal additions are optional (`std::optional` field, new API entries appended to the current
  `ort_api_1_to_N` / EP API tables — no version bump required during development per repo policy).
- Legacy EPs (single device, no override) hit the exact same code paths as today:
  `device_placement` is empty -> planner uses `GetOrtDeviceByMemType`.
- `SessionGetEpDeviceForInputs/Outputs` refinement is a strict improvement; single-device EPs still
  resolve to the same `OrtEpDevice`.
- New plugin API (`..._V2` fusion / IO devices) is opt-in and guarded by `ep.ort_version_supported`.

## 6. Testing

- **Unit (framework):** extend `onnxruntime/test/framework/ep_plugin_provider_test.cc` — a fake
  multi-device plugin EP that fuses a subgraph and tags input0->deviceA, input1->deviceB,
  output->deviceA; assert the plan places values correctly and `SessionGetMemoryInfoForInputs/Outputs`
  + refined `SessionGetEpDeviceForInputs/Outputs` report the right devices.
- **Example EP:** extend `onnxruntime/test/autoep/library/example_plugin_ep_virt_gpu/ep.cc` to expose
  two virtual devices and exercise `CopyTensors` to each via `GetSharedAllocator`.
- **Data-copy end-to-end:** extend `onnxruntime/test/shared_lib/test_data_copy.cc` /
  `onnxruntime/test/autoep/test_data_transfer.cc` for pre-placing inputs on the reported device and
  confirming no host round-trip.
- **Validation / negative:** capability references a device with no registered allocator -> clear init
  failure.
- **Regression:** confirm all existing single-device EP tests are unchanged.

## 7. Risks & open questions

- **Boundary-arg keying (resolved):** use **positional `(node, slot)`** identity end-to-end, never a
  bare NodeArg name. Names can be shadowed across nested subgraph scopes, and the only scope-correct
  name resolver is the planner's per-level `Index()`. The EP passes `(OrtNode*, is_input, io_index)`;
  the harvest keys the side-table by `const Node*` + slot; the planner resolves slot -> scope-correct
  `OrtValueIndex` itself.
- **Override storage decided:** on `ComputeCapability` (not `MetaDef`), harvested into a SessionState
  side-table keyed by `const Node*` + slot for the planner. `OrtValueIndex` does not exist at harvest
  time, which reinforces node-identity keying over value-index/name keying there.
- **Graph-I/O vs internal-edge boundary:** graph inputs/outputs get device placement + copies via the
  existing `CopyInputsAcrossDevices` path (Phase 1); internal edges crossing devices need
  MemcpyTransformer work (Phase 4). Confirm the compiled-EP targets only need the former initially.
- **Kernel-based multi-device** (non-fused) is explicitly **out of Phase 1**; confirm no in-tree EP
  needs it immediately.
- **`GetEpDeviceForInputs` device match** relies on `OrtDevice` equality between the planned device and
  the `OrtEpDevice`'s memory-info device — verify vendor/id/type are all populated consistently for
  virtual devices (`onnxruntime/core/framework/layering_annotations.cc` shows device-id sourcing
  nuances).
- **Signed/unsigned & span discipline** per `AGENTS.md` when adding index math in the planner override.

## 8. Suggested phasing

1. **Phase 0 — spike:** confirm the partition-time harvest hook and the positional `(node, slot)`
   keying (harvest side-table keyed by `const Node*`; planner resolves slot -> `OrtValueIndex`). Small
   PoC in the planner with a hand-built `ComputeCapability` + SessionState override map.
2. **Phase 1 — internal + compiled EPs:** `ComputeCapability::device_placement` field,
   provider-bridge wrappers, partition-time harvest into the SessionState override map, planner
   honoring overrides, per-device shared-allocator validation. Fake-EP unit tests.
3. **Phase 2 — plugin C API:** `EpGraphSupportInfo_AddNodesToFuse_V2` (+ IO-device carrier),
   translation in `PluginExecutionProvider::GetCapability`, example-EP coverage.
4. **Phase 3 — public device query refinement:** fix `SessionGetEpDeviceForInputs/Outputs` matcher;
   C# / C++ / Python surface tests; docs.
5. **Phase 4 (optional) — intra-EP copies / kernel-based multi-device:** lift the single-device-per-
   input restriction via MemcpyTransformer + per-device streams.

Each phase is independently shippable and keeps every existing EP on the current code path.
