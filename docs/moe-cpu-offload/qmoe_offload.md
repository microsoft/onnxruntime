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
`com.microsoft::QMoE` (and `com.microsoft::MoE`) nodes controlled entirely by `session.moe.offload.*`
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
| `session.moe.offload.device_working_bytes` | Positive integer bytes | `268435456` (`256 MiB`) | Upper bound on transient GPU streaming/dequant workspace. |
| `session.moe.offload.host_working_bytes` | Positive integer bytes | `67108864` (`64 MiB`) | Upper bound on transient CPU thread-pool scratch buffers. |
| `session.moe.offload.pinned_bytes` | `"auto"` or non-negative integer bytes | `67108864` (`64 MiB`) | Upper bound on pinned host DMA staging and promotion bounce buffers. |
| `session.moe.offload.budget_group` | String identifier | `""` (per-session) | Optional cross-session budget group name shared across target and draft sessions on the same GPU. |
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
  └─ Overrides exec_plan location of MoE (slots 2..7) and QMoE (slots 2..13, 15, 16) initializers
     to OrtDevice::CPU and skips memory-pattern tracing for external CPU initializers.
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
  allocations (`gpu_device_mem_limit` and `session.moe.offload.device_expert_bytes`). Reservations
  use RAII `Reservation` tokens (`Reserve` $\to$ `Commit`, or automatic rollback on failure) and
  two-phase `BeginFree` / `FinishFree` around `cudaFree`.
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
