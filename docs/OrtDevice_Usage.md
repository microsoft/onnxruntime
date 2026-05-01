# OrtDevice: Usage in ONNX Runtime

## Overview

`OrtDevice` (defined in `include/onnxruntime/core/framework/ortdevice.h`) is a lightweight struct that represents a **combination of a physical device and memory type**. It is the fundamental abstraction ONNX Runtime uses to describe *where* a memory allocation lives, and is central to allocation planning, data transfer decisions, and execution provider (EP) identity.

## Structure

```cpp
struct OrtDevice {
  DeviceType device_type;  // CPU=0, GPU=1, FPGA=2, NPU=3
  MemoryType memory_type;  // DEFAULT=0, HOST_ACCESSIBLE=5
  VendorId   vendor_id;    // PCI vendor (NVIDIA=0x10DE, AMD=0x1002, MICROSOFT=0x1414, etc.)
  DeviceId   device_id;    // Device index (e.g., GPU 0, GPU 1)
  Alignment  alignment;    // Required memory alignment
};
```

### Key Helper

- `UsesCpuMemory()` — returns `true` if `device_type == CPU` OR `memory_type == HOST_ACCESSIBLE` (e.g., CUDA pinned memory). This means the data is readable from the host without an explicit device-to-host transfer.

---

## The Allocator Registry

Allocators are registered in an `AllocatorMap`, which is simply:

```cpp
using AllocatorMap = std::map<OrtDevice, AllocatorPtr>;
```

This map is owned by `SessionState` and contains one entry per unique device/memory combination.

### Registration

During `SessionState` initialization:
1. Each EP's `CreatePreferredAllocators()` is called (in EP registration order).
2. Each returned allocator has an `OrtMemoryInfo` containing a `.device` field (an `OrtDevice`).
3. The allocator is inserted into the map keyed by that `OrtDevice`.
4. **Priority rule**: The EP registered first gets priority — `insert()` won't overwrite an existing key.

### Lookup

```cpp
AllocatorPtr SessionState::GetAllocator(const OrtDevice& device);
AllocatorPtr SessionState::GetAllocator(const OrtMemoryInfo& location);  // delegates to location.device
```

---

## IExecutionProvider: Two Device APIs

### `default_device_` (via `GetDevice()`)

A `const OrtDevice` set once in the EP constructor. It represents the EP's **identity device** — "what hardware does this EP own?"

### `GetOrtDeviceByMemType(OrtMemType)` (virtual)

Maps an `OrtMemType` to the correct `OrtDevice` for that memory category. It expresses that an EP manages **multiple memory spaces**.

**Example (CUDA EP):**

| `OrtMemType`         | Returns                                      |
|----------------------|----------------------------------------------|
| `OrtMemTypeDefault`  | `default_device_` (GPU, NVIDIA, device N)    |
| `OrtMemTypeCPUInput` | `OrtDevice()` (plain CPU)                    |
| `OrtMemTypeCPUOutput`| GPU + HOST_ACCESSIBLE (CUDA pinned memory)   |

The base class implementation returns `default_device_` for DEFAULT, and plain CPU for CPU_INPUT/CPU_OUTPUT. EPs override this to provide EP-specific memory spaces (e.g., pinned memory).

---

## How `default_device_` Is Used

| Use Case | What's Checked |
|----------|----------------|
| **CPU-based EP check** | `GetDevice().Type() == OrtDevice::CPU` — determines if memcpy/sync is needed |
| **MemcpyTransformer** | `Type() + Vendor()` compatibility between two EPs — decides whether to insert copy nodes |
| **EP combination validation** | e.g., DML EP cannot coexist with other non-CPU EPs |
| **Stream device binding** | `Stream` stores the device; used for `SetDevice()` on new threads |
| **Layering annotations** | Matches EPs to device-targeting rules |
| **`GetDeviceId()`** | Returns `default_device_.Id()` — used by kernels (e.g., `cudaSetDevice`) |
| **Fallback in `GetOrtDeviceByMemType()`** | Returned for `OrtMemTypeDefault` in the base implementation |

---

## How `GetOrtDeviceByMemType()` Is Used

The primary consumers are:

### 1. Allocation Planner (`allocation_planner.cc`)

For each tensor in the graph, the planner determines its memory location:

```
KernelDef says input/output has OrtMemType X
  → EP::GetOrtDeviceByMemType(X) → returns an OrtDevice
  → plan_.SetLocation(ort_value_index, that OrtDevice)
```

### 2. Allocator Lookup (at execution time)

```
SessionState::GetAllocator(OrtDevice) → returns the allocator for that location
```

### 3. OpKernelInfo

When a kernel needs an allocator for a specific memory type:

```cpp
auto it = allocators_.find(execution_provider_->GetOrtDeviceByMemType(mem_type));
```

---

## The Two Phases: MemcpyTransformer vs. Allocation Planner

These two components both deal with "where does a tensor live?" but operate at different phases:

### Phase 1: MemcpyTransformer (Graph Optimization)

- **When**: During graph optimization, before execution planning.
- **Purpose**: Insert explicit `MemcpyFromHost`/`MemcpyToHost` nodes where data must cross EP boundaries.
- **Uses `GetDevice()`**: Coarse-grained check — are two EPs on the same hardware (same device type + vendor)? If yes, no copy needed.
- **Uses `KernelDef` memory types**: `IsInputOnCpu()`/`IsOutputOnCpu()` checks if a specific input/output is declared to live on CPU (e.g., shape tensors). These don't need copies.
- **Output**: Modified graph with Memcpy nodes.

### Phase 2: Allocation Planner (Execution Planning)

- **When**: After graph optimization, during session initialization.
- **Purpose**: Assign a **concrete `OrtDevice`** to every tensor and plan allocator usage.
- **Uses `GetOrtDeviceByMemType()`**: Fine-grained — converts abstract `OrtMemType` into an exact `OrtDevice` (including vendor, device ID, and whether memory is pinned).
- **Output**: Location assignment for every OrtValue; allocator selection.

### How They Complement Each Other

The MemcpyTransformer decides **whether a copy is needed** and makes it explicit in the graph. The allocation planner decides **where each tensor actually lives**, including the inputs/outputs of the Memcpy nodes themselves.

**Example with CUDA:**
1. MemcpyTransformer sees data flowing CPU EP → CUDA EP → inserts `MemcpyFromHost` node.
2. Allocation planner assigns:
   - `MemcpyFromHost` input → `GetOrtDeviceByMemType(OrtMemTypeCPUInput)` → plain CPU
   - `MemcpyFromHost` output → `GetOrtDeviceByMemType(OrtMemTypeDefault)` → GPU (NVIDIA, device 0)

---

## Plugin EPs (OrtEp) and `GetOrtDeviceByMemType()`

### Current Behavior

`PluginExecutionProvider` (the internal wrapper for plugin EPs using the OrtEp C API) **overrides** `GetOrtDeviceByMemType()` by delegating to two optional OrtEp hooks:

1. **`OrtEp::GetDefaultMemoryDevice()`** — explicit control over the EP's default memory device (used as the EP's identity for memcpy decisions, stream binding, etc.). If not implemented, ORT infers it from `OrtEpDevice::device_memory_info`.
2. **`OrtEp::GetMemoryDeviceByMemType(OrtMemType)`** — allows the plugin EP to map each `OrtMemType` to a specific device. If not implemented, ORT uses the default behavior (CPU for `OrtMemTypeCPUInput`/`OrtMemTypeCPUOutput`, EP's default device for `OrtMemTypeDefault`).

Both hooks are optional (NULL = use default behavior). They are version-gated at ORT API version 27+.

### What's Registered vs. What's Used

| OrtEpDevice field | Allocator registered? | Selected by planner? |
|---|---|---|
| `device_memory_info` | ✅ Yes (in `allocators_`) | ✅ Yes — selected when `GetOrtDeviceByMemType()` returns the EP's default device |
| `host_accessible_memory_info` | ✅ Yes (in `allocators_`) | ✅ Yes — selected when `GetOrtDeviceByMemType()` returns the host-accessible device (e.g., for `OrtMemTypeCPUOutput`) |
| `read_only_device_memory_info` | ✅ Yes (in `initializer_allocators_`) | Partially — see below |

### Background: Previous Limitation

Prior to the addition of these hooks, `PluginExecutionProvider` did **not** override `GetOrtDeviceByMemType()`. The base class always returned `default_device_` for `OrtMemTypeDefault` and plain CPU for `OrtMemTypeCPUInput`/`OrtMemTypeCPUOutput`. This meant that even if a plugin EP registered a host-accessible allocator, the allocation planner would never route tensors to it.

### Allocation Planner Call Patterns

The allocation planner calls `GetOrtDeviceByMemType()` at these points:

1. **Graph inputs** (line 807): `ep->GetOrtDeviceByMemType(kernel_def->InputMemoryType(idx))` → always `DEFAULT` for fused/compiled nodes.
2. **Node outputs** (line 914): `ep->GetOrtDeviceByMemType(kernel_def->OutputMemoryType(i))` → always `DEFAULT` for fused/compiled nodes.
3. **Downstream consumer optimization** (lines 927-928): `consumer_ep->GetOrtDeviceByMemType(OrtMemTypeCPUInput)` — only when producer output is on CPU.
4. **Node inputs** (line 957): `ep->GetOrtDeviceByMemType(IsInputOnCpu() ? CPUInput : DEFAULT)` → always `DEFAULT` for fused/compiled nodes.
5. **Never** calls with `OrtMemTypeCPUOutput` — that value is only used by EP-internal code.

### Compiling EP Constraints

For compiling EPs (those using `IExecutionProvider::Compile()`):

- `BuildFusedKernelDef()` in `graph_partitioner.cc` sets **NO** memory type annotations on fused kernel defs (only name/domain/version/provider).
- All fused node I/O defaults to `OrtMemTypeDefault`.
- Only `DEFAULT` and `CPUInput` (as consumer of upstream CPU output) are ever queried via `GetOrtDeviceByMemType()`.
- `OrtMemTypeCPUOutput` is never queried by the planner for any EP.

### Initializer Allocator Bypass for CPU-Default EPs

If an OrtEp uses `OrtDevice()` (CPU) as its default device and registers an initializer allocator (via `read_only_device_memory_info`), the allocator is **bypassed** for external initializers:

- In `session_state_utils.cc`, `DeserializeTensorProto()` checks if `device == OrtDevice()` AND `HasExternalData()` → memory-maps the file directly, ignoring any provided allocator.
- The initializer allocator IS used for: PrePack operations, and inline initializers routed through the memory pattern planner.

---

## Summary Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Session Initialization                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. EP Registration                                                 │
│     └─ Each EP constructed with default_device_ (identity)          │
│                                                                     │
│  2. Graph Optimization (MemcpyTransformer)                          │
│     └─ Uses GetDevice() for coarse EP compatibility                 │
│     └─ Inserts MemcpyFromHost / MemcpyToHost nodes                  │
│                                                                     │
│  3. Allocator Registration                                          │
│     └─ EP::CreatePreferredAllocators()                              │
│     └─ AllocatorMap[OrtDevice] = allocator (first EP wins)          │
│                                                                     │
│  4. Allocation Planning                                             │
│     └─ Uses GetOrtDeviceByMemType() for precise tensor placement    │
│     └─ Each OrtValue gets an assigned OrtDevice location            │
│                                                                     │
│  5. Execution                                                       │
│     └─ GetAllocator(OrtDevice) → allocate tensors                   │
│     └─ Stream::GetDevice() → thread device binding                  │
│     └─ DataTransfer triggered when source/dest OrtDevices differ    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```
