# Portable MoE / QMoE CPU/GPU Offload Architecture (`session.moe.offload.*`)

This document describes the ONNX Runtime framework, session planner, memory budget ledger, and
CUDA execution provider lifecycle hooks for **portable runtime MoE/QMoE CPU/GPU offload**.

Related operator and kernel documentation:
- **CPU shared-layout kernels & `FileExpertStore`**: [docs/contrib_ops/cpu/qmoe_shared_layout.md](../contrib_ops/cpu/qmoe_shared_layout.md)
- **CUDA `MoE` / `QMoE` operator & `QMoEOffload` backend**: [docs/contrib_ops/cuda/moe_qmoe.md#12b-portable-cpugpu-offload-sessionmoeoffload](../contrib_ops/cuda/moe_qmoe.md#12b-portable-cpugpu-offload-sessionmoeoffload)

---

## 1. Overview

Large Mixture-of-Experts models (such as 48-layer $\times$ 512-expert NVFP4 or INT4/INT3 models with
45 GiB–63 GiB of routed expert weights) often exceed the VRAM of 24 GiB–32 GiB consumer and
workstation GPUs. Rather than exporting a separate hardware-specific operator or baking GPU slot
counts into the ONNX graph, ONNX Runtime supports **portable runtime CPU/GPU offload** on standard
`com.microsoft::QMoE` nodes (row-major NVFP4 with `weights_prepacked=1`, SM80-prepacked INT4 with
`weights_prepacked=1` and `block_size=32`, and raw INT3 with `weights_prepacked=0` and
`block_size` in `{32, 64, 128}`) controlled entirely by `session.moe.offload.*`
session options.

A **single canonical `com.microsoft::QMoE` model package** runs unchanged across:

- **Full-HBM GPUs** (e.g., 80 GiB–141 GiB H100/H200): `session.moe.offload.mode="auto"` admits all
  $E$ experts per layer into GPU HBM during session initialization with zero runtime offload overhead.
- **VRAM-Constrained GPUs with Ample Host RAM** (e.g., 24 GiB / 32 GiB GPU + 64 GiB–128 GiB RAM):
  caches the hottest $C_l < E$ experts per layer in GPU HBM, executes cold routed experts on the
  CPU via shared-layout AVX-512 / AVX2 microkernels overlapped with GPU resident GEMVs inside a
  single captured CUDA Graph, and promotes newly hot experts asynchronously in the background.
- **VRAM- and Host-RAM-Constrained Systems** (e.g., 24 GiB / 30 GiB GPU + 40 GB–48 GB RAM): bounds
  host RAM usage via `FileExpertStore` (`session.moe.offload.host_expert_bytes`), maintaining
  disjoint GPU and host residency and streaming uncached cold experts from the backing weight files
  on demand without online weight prepacking or scale transposition.

---

## 2. Export Storage, File Mapping & Popularity Profile

### 2.1 External Data Layout & Page Alignment

Exporters emit standard `com.microsoft::QMoE` nodes with no offload-specific attributes or inputs.
Expert initializers (slots `2..13, 15, 16`) are saved as standard ONNX external data
(`TensorProto.DataLocation.EXTERNAL`) in one or more regular files relative to the directory
containing `model.onnx`:

- **Expert-Major Contiguity**: Within each bank tensor, expert $e \in [0, E)$ occupies a contiguous
  byte range `[e * bytes_per_expert, (e + 1) * bytes_per_expert)`.
- **4096-Byte File Offset Alignment**: Aligning the start `offset` of each external tensor to a
  `4096`-byte OS page boundary allows ORT's `PosixRandomAccessFile::Map` / `WindowsEnv::MapFileIntoMemory`
  to memory-map the file read-only (`PROT_READ, MAP_SHARED` on Linux; `FILE_MAP_READ` on Windows)
  with zero heap allocation, stripe pages evenly across NUMA sockets, and release host OS pages for
  GPU-resident experts at `4 KiB` granularity.
- **Clean Private-to-Shared Mapping Promotion (`PromoteFileMappingToSharedReadOnly`)**: On Linux,
  `PosixRandomAccessFile::Map` creates `MAP_PRIVATE` (`rw-p`) mappings so mutable initializers can be
  modified in place. During Stage 3 admission, `PromoteFileMappingToSharedReadOnly` verifies via
  `/proc/self/maps`, `/proc/self/pagemap`, and `fstat` (`st_ino == inode`) that no page in the expert
  range has been Copy-On-Write faulted into anonymous or swap memory, and converts the range in-place
  via `mmap(..., PROT_READ, MAP_SHARED | MAP_FIXED)` (`r--s`) so `FileExpertStore`,
  `EvictUploadedHostPages`, and `cudaHostRegister` operate on shared page-cache pages with zero
  `RssAnon` inflation.

### 2.2 Optional Offline Popularity Profile (`expert-profile.bin`)

A model package may optionally ship a compact binary popularity profile (`expert-profile.bin`,
`STRP` v1 format) in the model directory (or referenced via `session.moe.offload.popularity_file`).
The file has a 24-byte little-endian header:

```python
struct.pack("<6I", 0x50525453, 1, num_layers, num_experts, total_pairs, num_ranked_pairs)
```

followed by `num_ranked_pairs` 4-byte `<2H` `(layer_id, expert_id)` pairs ordered from globally
most frequent to least frequent.

When present, `QMoESessionController::PlanAndAdmitSession()` uses this global ranking to:
1. Allocate **non-uniform per-layer GPU resident capacities** $C_l$ (after reserving a floor of
   `min_slots_per_layer = 4` experts per layer) so layers with flatter routing distributions receive
   proportionally more GPU HBM slots.
2. Seed the initial GPU resident slots `[0, C_l)` and bounded host cache slots (`FileExpertStore`)
   with the hottest cold experts before the first inference call.

When no popularity file is provided, the planner distributes the GPU expert byte budget uniformly
across layers (`C_l = floor(budget / L)`) and seeds experts `0 .. C_l - 1`.

---

## 3. Session Options (`session.moe.offload.*`) & GenAI Runtime Profiles

All offload policies and memory budgets are configured through standard `OrtSessionOptions`
configuration entries ([onnxruntime_session_options_config_keys.h](../../include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h))
and parsed by `MoeOffloadSessionPolicy::ParseFromConfigOptions` ([moe_offload_plan.h](../../onnxruntime/core/framework/moe_offload_plan.h)).
Unknown `session.moe.offload.*` keys or legacy prototype keys (`moe.cpu_offload.*`) fail fast at
session creation with `ORT_INVALID_ARGUMENT`.

### 3.1 Session Configuration Keys

| Config Key (`session.moe.offload.*`) | Allowed Values | Default | Description |
| :--- | :--- | :---: | :--- |
| `session.moe.offload.mode` | `"disabled"`, `"auto"`, `"required"` | `"disabled"` | `"disabled"`: standard GPU-only QMoE. `"auto"`: places initializers on CPU first; if all experts fit in usable GPU VRAM, uploads all experts to GPU (`C_l = E`), otherwise enables hybrid offload. `"required"`: enforces offload admission and fails fast if any QMoE node is ineligible or cannot fit minimum capacity. |
| `session.moe.offload.placement` | `"static"`, `"adaptive"` | `"adaptive"` | `"static"`: fixed GPU resident set (`spare_slots = 0`). `"adaptive"`: reserves `spare_slots = 2` per partially offloaded layer and promotes hot cold experts into GPU HBM between steps. |
| `session.moe.offload.cpu_execution` | `"auto"`, `"enabled"`, `"disabled"` | `"auto"` | Controls CPU cold-expert execution during decode (`rows <= async_max_tokens`). |
| `session.moe.offload.device_expert_bytes` | `"auto"` or integer bytes | `"auto"` | Maximum GPU HBM bytes for all MoE resident + spare expert slabs and lookup tables in the session. `"auto"` uses `usable_free_vram - reserve_device_bytes - device_working_bytes`. |
| `session.moe.offload.host_expert_bytes` | `"auto"` or integer bytes | `"auto"` | Maximum host RAM bytes for cold experts. `"auto"` uses direct `MAP_SHARED` `mmap` over the external weight files (or cgroup-bounded cache when host RAM is constrained). A finite value smaller than total expert bytes enables the bounded `FileExpertStore` LRU host cache and evicts `mmap` source pages via `madvise(MADV_DONTNEED)`. |
| `session.moe.offload.reserve_device_bytes` | `"auto"` or integer bytes | `0` | GPU VRAM bytes reserved for post-initialization KV/GDN caches, child sessions (`mtp.onnx`), and CUDA Graph pools when `device_expert_bytes="auto"`. |
| `session.moe.offload.device_working_bytes` | Positive integer bytes | `268435456` (`256 MiB`) | Upper bound on the aggregate transient GPU working set of one prefill call: runner scratch, expert-compaction maps, compacted biases, multi-pass accumulators, DMA staging, and the dequantized expert weights (counted once, whether they live in the arena or in the DMA workspace). A call whose aggregate does not fit fails with an error naming this option. |
| `session.moe.offload.host_working_bytes` | Positive integer bytes | `67108864` (`64 MiB`) | Upper bound on transient CPU thread-pool scratch buffers. |
| `session.moe.offload.pinned_bytes` | `"auto"` or non-negative integer bytes | `67108864` (`64 MiB`) | Upper bound on pinned host DMA staging and promotion bounce buffers. |
| `session.moe.offload.budget_group` | String identifier | `""` (per-session) | Optional cross-session budget group name shared across target and draft sessions on the same GPU. |
| `session.moe.offload.gpu_device_mem_limit` | Non-negative integer bytes | `0` (no limit) | Per-session cap on the bytes this session may allocate on the GPU through its CUDA allocators (BFC arena growth and MoE offload buffers). It is read when the CUDA EP is created, so it must be set before the session is created. Honored by both the bundled CUDA EP and the CUDA plugin EP; it is not part of `OrtCUDAProviderOptionsV2`. Not the same as `gpu_mem_limit` (see §5.1). Not supported together with an external CUDA allocator or a CUDA memory pool. |
| `session.moe.offload.allow_file_backing` | `"0"`, `"1"` | `"1"` | Allows `FileExpertStore` to read uncached cold experts from the backing weight files when `host_expert_bytes` is bounded. |
| `session.moe.offload.cpu_threads` | Non-negative integer | `0` (`auto`) | Number of `SpinPool` worker threads for cold-expert CPU execution (`0` auto-detects physical cores via `DiscoverPhysicalCoreCount()`). |
| `session.moe.offload.async_max_tokens` | Non-negative integer | `8` | Token threshold separating hybrid doorbell decode (`rows <= async_max_tokens`) from GPU streaming DMA prefill (`rows > async_max_tokens`). |
| `session.moe.offload.prefill_wave_experts` | Positive integer | `32` | Maximum experts per compaction/DMA wave during prefill (further bounded by `device_working_bytes`). |
| `session.moe.offload.skip_zero_cold_doorbell` | `"0"`, `"1"` | `"0"` | Complete zero-cold partial-residency decode calls on the GPU without dispatching CPU expert work (`ORT_QMOE_SKIP_ZERO_COLD_DOORBELL` fallback). Applies to static placement with cross-layer promotions disabled. |
| `session.moe.offload.popularity_file` | Relative or absolute path | `""` | Optional path to `expert-profile.bin` (`STRP` v1 format). Relative paths resolve against the model directory. |
| `session.moe.offload.source_root` | Relative or absolute path | Model directory | Root directory used to resolve external weight files and relative profile paths. |
| `session.moe.offload.derived_cache_dir` | Relative or absolute path | `""` | Optional directory for derived cache artifacts. |
| `session.moe.offload.node_overrides` | Semicolon-separated spec | `""` | Optional per-node overrides (`"node_name:max_resident_slots=<N>,device_expert_bytes=<B>"`). |

### 3.2 Standalone Python Usage Example

```python
import onnxruntime as ort

so = ort.SessionOptions()
so.add_session_config_entry("session.use_device_allocator_for_initializers", "1")
so.add_session_config_entry("session.moe.offload.mode", "required")
so.add_session_config_entry("session.moe.offload.placement", "adaptive")
so.add_session_config_entry("session.moe.offload.device_expert_bytes", str(18 * 1024**3))  # 18 GiB GPU slab budget
so.add_session_config_entry("session.moe.offload.host_expert_bytes", str(38 * 1024**3))    # 38 GiB bounded host cache
so.add_session_config_entry("session.moe.offload.allow_file_backing", "1")
so.add_session_config_entry("session.moe.offload.popularity_file", "expert-profile.bin")

sess = ort.InferenceSession(
    "model.onnx",
    sess_options=so,
    providers=[("CUDAExecutionProvider", {"enable_cuda_graph": "1"})],
)
```

### 3.3 `onnxruntime-genai` `runtime_profiles` Example (`genai_config.json`)

In `onnxruntime-genai`, a single portable `genai_config.json` sets `session.moe.offload.mode="auto"`
in the base decoder config and overrides `session.moe.offload.*` per hardware tier via
`runtime_profiles` (matching device VRAM and effective host RAM bounds, including Linux cgroup v2
`memory.max` limits):

```json
{
  "model": {
    "decoder": {
      "filename": "model.onnx",
      "session_options": {
        "session.use_device_allocator_for_initializers": "1",
        "session.moe.offload.mode": "auto",
        "session.moe.offload.placement": "adaptive",
        "session.moe.offload.device_expert_bytes": "auto",
        "session.moe.offload.host_expert_bytes": "auto",
        "session.moe.offload.allow_file_backing": "1",
        "session.moe.offload.popularity_file": "expert-profile.bin"
      }
    }
  },
  "runtime_profiles": [
    {
      "id": "consumer-gpu-32g",
      "eligibility": {
        "minimum_total_device_memory_bytes": 25769803777,
        "maximum_total_device_memory_bytes": 34359738368,
        "minimum_effective_host_memory_bytes": 51539607553
      },
      "overlay": {
        "model": {
          "decoder": {
            "filename": "model_s4.onnx",
            "session_options": {
              "session.moe.offload.mode": "required",
              "session.moe.offload.device_expert_bytes": "19231076352"
            }
          }
        }
      }
    },
    {
      "id": "consumer-gpu-30g-ram-48g-32k",
      "eligibility": {
        "minimum_total_device_memory_bytes": 25769803777,
        "maximum_total_device_memory_bytes": 32212254720,
        "minimum_effective_host_memory_bytes": 0,
        "maximum_effective_host_memory_bytes": 51539607552
      },
      "overlay": {
        "model": {
          "decoder": {
            "filename": "model_s4.onnx",
            "session_options": {
              "session.moe.offload.mode": "required",
              "session.moe.offload.device_expert_bytes": "18289511424",
              "session.moe.offload.host_expert_bytes": "41339060224",
              "session.moe.offload.allow_file_backing": "1"
            }
          }
        }
      }
    }
  ]
}
```

---

## 4. Five-Stage `SessionState` Planning & Bounded Materialization Pipeline

On 24 GiB–32 GiB GPUs, **ORT must never materialize or prepack all $E$ experts onto the GPU during
session creation** (which would immediately OOM on a 45 GiB–63 GiB MoE model). Offload initialization
is split across five deterministic stages in [session_state.cc](../../onnxruntime/core/framework/session_state.cc),
[qmoe_session_controller.h](../../onnxruntime/core/providers/cuda/qmoe_session_controller.h), and
`IMoeOffloadNode` implementations:

```
Stage 1: PlaceMoeOffloadInitializersOnCpu (SessionState::FinalizeSessionState, before SaveInitializedTensors)
  └─ Overrides exec_plan location of eligible NVFP4 / prepacked INT4 / raw INT3 QMoE (slots 2..7, 15, 16)
     initializers to OrtDevice::CPU and skips memory-pattern tracing for external CPU initializers.
     SaveInitializedTensors() maps external files on CPU with 0 B GPU VRAM allocated for experts.
        │
Stage 2: Kernel Construction & Zero-Copy PrePack Bypass (CreateKernel -> OpKernel::PrePack)
  └─ QMoE::QMoE registers MoeExpertSourceDescriptor and MoeBackendCapabilities (gpu_slot_bytes,
     cpu_slot_bytes, gpu_fixed_table_bytes) with QMoESessionController.
  └─ QMoE::PrePack sets is_packed = false when offload_policy_.mode != kDisabled so the CPU mmap
     OrtValues remain alive in SessionState::constant_initialized_tensors_ with zero copies in both
     monolithic and plugin CUDA EP builds.
        │
Stage 3: Session-Wide Byte Planner (ep->OnSessionInitializationEnd -> PlanAndAdmitSession)
  └─ Queries usable free VRAM after all non-MoE weights (attention, norms, shared experts, LM head)
     are already on GPU. On Windows WDDM, intersects cudaMemGetInfo with DXGI
     QueryVideoMemoryInfo(DXGI_MEMORY_SEGMENT_GROUP_LOCAL) minus a 12% anti-spillover reserve so
     driver allocations never spill into slow Shared GPU Memory (dxgmms2.sys).
  └─ Runs profile-guided water-filling across all registered QMoE layers (with min_slots_per_layer = 4
     floor, spare_slots in {0, 2}, dummy_slots = 0) and water-fills bounded host_cache_slots.
  └─ Promotes clean private file mappings to shared read-only (PromoteFileMappingToSharedReadOnly)
     and on Linux unlimited-RAM offload stripes mapped expert pages across NUMA sockets
     (PrepareLinuxHostExpertMappings).
        │
Stage 4: Per-Node Materialization & Host Page Eviction (IMoeOffloadNode::InitializeAdmittedPlan)
  └─ Full GPU Residency (resident_slots == E): uploads CPU tensors to GPU buffers and uses the stock
     GPU QMoE path with zero offload overhead.
  └─ Partial Offload (resident_slots < E): constructs QMoEOffload(info, policy, source_desc, plan),
     allocates the [C_l + S_l] GPU resident slab and double-buffered slot_table_dev_ (2 * E * sizeof(int)),
     seeds initial residents, initializes byte-preserving FileExpertStore when host_expert_bytes is
     bounded, and evicts uploaded host pages via madvise(MADV_DONTNEED) / VirtualUnlock.
        │
Stage 5: Runtime Execution
  └─ rows <= async_max_tokens (decode): dispatches to QMoEOffload doorbell CUDA-graph execution.
  └─ rows > async_max_tokens (prefill): syncs slot tables and runs streaming DMA + UVA/HBM dequant.
```

---

## 5. CUDA Memory Budget Ledger & Session Lifecycle Hooks

### 5.1 Device & Host Pinned Memory Ledgers ([cuda_device_memory_budget.h](../../onnxruntime/core/providers/cuda/cuda_device_memory_budget.h))

- **Device Memory Budget (`CudaDeviceMemoryBudget` & `SessionBudgetAccount`)**:
  Tracks per-device and per-session (or named `budget_group`) committed and in-flight GPU HBM
  allocations (`session.moe.offload.gpu_device_mem_limit` and `session.moe.offload.device_expert_bytes`).
  Reservations use RAII `Reservation` tokens (`Reserve` $\to$ `Commit`, or automatic rollback on failure)
  and two-phase `BeginFree` / `FinishFree` around `cudaFree`.
  `session.moe.offload.gpu_device_mem_limit` (session config key, bytes, `0` or unset = unlimited) is not
  the same as the older `gpu_mem_limit` CUDA provider option: `gpu_mem_limit` only caps the size of the
  BFC arena, whereas `session.moe.offload.gpu_device_mem_limit` is charged on every `cudaMalloc` made
  through the session's CUDA allocators (BFC arena growth and MoE offload buffers) and an allocation that
  would exceed it fails. It is a session config key rather than a CUDA provider option so that
  `OrtCUDAProviderOptionsV2` (public API) keeps its layout; both the bundled CUDA EP and the CUDA plugin EP
  read it when they are created.
- **Host Pinned Memory Ledger (`HostPinnedBudgetLedger`)**:
  Tracks both `cudaHostAlloc` buffers (`ReserveAllocated` / `BeginFreeAllocated` / `FinishFreeAllocated`)
  and `cudaHostRegister` page-aligned intervals (`RegisterHostRange` / `UnregisterHostRange`), merging
  overlapping `4 KiB` page ranges so shared pages are never double-counted or prematurely unregistered.

### 5.2 Execution Provider Lifecycle & CUDA Graph Coordination ([qmoe_session_controller.h](../../onnxruntime/core/providers/cuda/qmoe_session_controller.h))

Both the monolithic CUDA EP ([cuda_execution_provider.cc](../../onnxruntime/core/providers/cuda/cuda_execution_provider.cc))
and the plugin CUDA EP ([cuda_ep.cc](../../onnxruntime/core/providers/cuda/plugin/cuda_ep.cc))
own a `QMoESessionController` instance and invoke:

1. `OnSessionInitializationEnd()` $\to$ `PlanAndAdmitSession()`: runs Stage 3–4 planning and admission
   once all non-MoE initializers have been placed on GPU.
2. `OnRunStart(run_options)` / `OnRunEnd(sync_stream, run_options)`: enforces single-active-run
   serialization per session controller, drains in-flight CPU/GPU invocation records on run failure
   (`__ort.run_outcome = "failed"` propagated by [inference_session.cc](../../onnxruntime/core/session/inference_session.cc)),
   and commits deferred routing observations only after synchronized successful runs.
3. `OnBeforeGraphReplay(stream)` / `OnAfterGraphReplay(stream, sync)`: executes between-step background
   expert promotions (`StepBetweenRuns`) and increments `GraphGeneration()` only when a captured graph's
   device pointer bindings change, allowing in-slab slot-table flips to reuse captured CUDA Graphs
   without recapture.

Each logical execution of a QMoE node is one invocation record, and its routing observation is applied exactly
once. A run that captures a CUDA graph registers the record while capturing and again for the immediate replay of
the captured graph (no `OnRunStart` in between); `RegisterInvocation` keeps a single entry, so that execution is
drained and observed once. Observations are staged on the record and taken out of it when the run ends, so they
are applied at most once and never for a failed run. Note that ORT keeps re-running the first `Run()` until the
graph is captured (`min_num_runs_before_cuda_graph_capture` regular runs plus the capture run), and each of those
executions is a separate logical execution that is observed on its own.

### 5.3 Per-Run Controls (`ep.cuda.qmoe_*` Run Config Entries) ([qmoe_run_controls.h](../../onnxruntime/core/providers/cuda/qmoe_run_controls.h))

An application steers the prefill/decode hand-off by attaching run config entries to `Run()`
(`RunOptions.add_run_config_entry(...)` in Python, `AddRunConfigEntry` in the C API). `OnRunStart` validates and
applies them to the session's `QMoERunControls`. The values live on the session, so one session never observes or
consumes another session's signals, and the library never calls `setenv` while other threads read the environment.

| Run Config Key | Allowed Values | Lifetime | Description |
| :--- | :--- | :--- | :--- |
| `ep.cuda.qmoe_in_warmup` | `"0"`, `"1"` | This `Run()` only | `"1"` marks a warm-up run: it does not feed expert scores, route statistics, or promotions. |
| `ep.cuda.qmoe_prefill_new_turn` | `"0"`, `"1"` | This `Run()` only; consumed once, by layer 0 | `"1"` marks the first run of a new conversation turn: prefill state left over from the previous turn is finalized and reset once. |
| `ep.cuda.qmoe_prefill_total_tokens` | Non-negative 32-bit integer | Sticky: stays on the session until a later run supplies a new value | Total prompt tokens of the turn being prefilled. |
| `ep.cuda.qmoe_prefill_handoff_done` | `"0"`, `"1"` | Armed only when the run completes successfully; consumed once | `"1"` marks the last prefill run of a turn. After it completes successfully, the hand-off to decode stays pending until the next prefill-to-decode finalization consumes it. A failed run never arms it. |

- An entry that is not supplied takes its default (`"0"`) for that run, so a run never inherits the flags of an
  earlier run.
- A malformed value (a flag other than `"0"`/`"1"`, or a negative or non-numeric token count) fails that run with
  `ORT_INVALID_ARGUMENT` before any QMoE node executes, applies none of the entries, and leaves the session usable.
- The process environment variables `ORT_QMOE_IN_WARMUP`, `ORT_QMOE_PREFILL_NEW_TURN`,
  `ORT_QMOE_PREFILL_TOTAL_TOKENS`, and `ORT_QMOE_PREFILL_HANDOFF_DONE` used by earlier prototypes are no longer read.
- Limitation: a `Run()` that replays an already captured CUDA graph does not reach `OnRunStart`, so it cannot carry
  these entries. Attach them to runs that execute eagerly (warm-up and prefill runs).

### 5.4 Background Services & Provider Shutdown ([qmoe_background_services.h](../../onnxruntime/contrib_ops/cuda/moe/qmoe_background_services.h))

Cold-expert CPU execution uses process-wide background threads that are started on demand by the first QMoE
offload run and shared by every session in the process: the doorbell service (`qmoe-doorbell`), the shared CPU spin
pools (`qmoe-spin`, see `session.moe.offload.cpu_threads`), and, only while `ORT_QMOE_CROSS_LAYER_PROMO=1`, the
promotion worker (`qmoe-promo`). The threads are named, so they can be told apart in a debugger or in
`/proc/<pid>/task/*/comm`.

Joining a thread from a static destructor while `FreeLibrary` holds the Windows loader lock deadlocks, so the
provider does not rely on static destruction. `CUDA_Provider::Shutdown()` (bundled EP) and `~CudaEpFactory()`
(plugin EP) call `ShutdownQMoEBackgroundServices()` before the library can be unloaded. The call is idempotent,
stops the doorbell service first and the spin pools after it, and does nothing while any QMoE offload backend is
still alive (each backend holds a live-backend lease). The services start again on demand if a QMoE kernel is
created afterwards. The bounded `FileExpertStore` read workers are owned per kernel and stop with the kernel.

### 5.5 Cross-Layer Promotion Look-Ahead (Experimental, `ORT_QMOE_CROSS_LAYER_PROMO=1`)

Cross-layer promotion is off by default. When enabled, the doorbell worker can rewrite the slot table and host
maps of the next layer while it services the current layer. That is only sound for a layer that has not classified
its routes for the current step, so a promotion request is published only when all of the following hold
(`IsCrossLayerLookaheadTarget` / `CanPublishCrossLayerPromotion` in
[qmoe_offload_plan.h](../../onnxruntime/contrib_ops/cuda/moe/qmoe_offload_plan.h)):

- The target is a later layer of the same session as the servicing layer. One session runs on one stream and
  rejects overlapped runs, so a later layer is still queued behind the servicing layer's merge kernel. A layer of
  another session has no such ordering and may already have classified with the old placement.
- The target has no pending doorbell request, which would mean it has already classified for this step.
- The target's placement epoch still equals the epoch the request was planned against. If the target changed its
  own placement in the meantime, the request is cancelled (after its staged copy completes) instead of landing in a
  reused slot.
