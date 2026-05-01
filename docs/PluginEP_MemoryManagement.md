# Memory Devices, Memory Info, and Allocators for Plugin EPs

This document explains how ONNX Runtime manages memory through the plugin EP (OrtEp) C API.
It covers the key concepts — **OrtMemoryDevice**, **OrtMemoryInfo**, and **OrtAllocator** — and
how they are used during session initialization and execution.

> **Audience**: Plugin EP authors who work with the public OrtEp C API. Internal ORT implementation
> details are called out explicitly so EP authors know what happens behind the scenes.

---

## Core Concepts

### OrtMemoryDevice

An `OrtMemoryDevice` represents a **specific memory space** on a device. It combines four properties:

| Property | Description | Example |
|----------|-------------|---------|
| **Device type** | The kind of hardware | GPU, CPU, NPU |
| **Vendor ID** | PCI vendor or custom identifier | `0x10DE` (NVIDIA), `0xBE57` (custom) |
| **Device ID** | Index of the device | 0, 1, 2 |
| **Memory type** | The kind of memory on that device | `DEFAULT`, `HOST_ACCESSIBLE` |

Two `OrtMemoryDevice` values are equal if all five properties match (device type, vendor ID,
device ID, memory type, and alignment). Use `OrtEpApi::MemoryDevice_AreEqual()` to compare them.

> **Important**: Alignment participates in equality checks and allocator map lookups. If you create
> two `OrtMemoryInfo` instances that differ only in alignment, they will have different
> `OrtMemoryDevice` values and require separate allocators.

**Query functions** (all in `OrtEpApi`):

```
MemoryDevice_GetDeviceType(device)   → OrtMemoryInfoDeviceType
MemoryDevice_GetMemoryType(device)   → OrtDeviceMemoryType
MemoryDevice_GetVendorId(device)     → uint32_t
MemoryDevice_GetDeviceId(device)     → uint32_t
```

> **Internal note**: `OrtMemoryDevice` is a thin wrapper around the internal `OrtDevice` struct.
> Plugin EPs never construct an `OrtMemoryDevice` directly — they obtain one from an `OrtMemoryInfo`
> via `OrtEpApi::MemoryInfo_GetMemoryDevice()`.

#### Memory Types

| `OrtDeviceMemoryType` | Meaning |
|-----------------------|---------|
| `OrtDeviceMemoryType_DEFAULT` | Standard device memory (e.g., GPU VRAM). Not directly accessible from the CPU. |
| `OrtDeviceMemoryType_HOST_ACCESSIBLE` | Device-associated memory that is also accessible from the CPU without explicit copies (e.g., CUDA pinned memory, shared memory). |

An `OrtMemoryDevice` with `HOST_ACCESSIBLE` memory type means the CPU can read/write the memory
directly, even though it is associated with a non-CPU device.

### OrtMemoryInfo

An `OrtMemoryInfo` is a **descriptor for an allocator's memory space**. It includes:

- A human-readable **name** (e.g., `"MyEP GPU"`, `"MyEP GPU pinned"`)
- An **`OrtMemoryDevice`** (device type + vendor + device ID + memory type)
- An **allocator type** (`OrtDeviceAllocator` or `OrtReadOnlyAllocator`)
- A memory **alignment**

Plugin EPs create `OrtMemoryInfo` objects to describe each memory space their allocators manage.

**Creating an OrtMemoryInfo** (C++ helper API):

```cpp
auto device_memory = Ort::MemoryInfo{
    "MyEP GPU",                           // allocator name
    OrtMemoryInfoDeviceType_GPU,          // device type
    /*vendor*/ 0xBE57,                    // vendor ID
    /*device_id*/ 0,                      // device index
    OrtDeviceMemoryType_DEFAULT,          // memory type
    /*alignment*/ 0,                      // 0 = use default
    OrtAllocatorType::OrtDeviceAllocator  // allocator type
};
```

**Extracting the OrtMemoryDevice**:

```cpp
const OrtMemoryDevice* device = ep_api.MemoryInfo_GetMemoryDevice(device_memory);
```

#### Allocator Types

| `OrtAllocatorType` | Usage |
|-------------------|-------|
| `OrtDeviceAllocator` | Standard read/write allocator for intermediate tensors, outputs, and initializers. |
| `OrtReadOnlyAllocator` | Read-only allocator used only for model initializers (weights). Allows ORT to use optimized placement (e.g., memory mapping). |

> **Important**: Do not use `OrtArenaAllocator` — that value is reserved for ORT's internal arena
> implementation.

### OrtAllocator

An `OrtAllocator` is the interface that actually performs memory allocation and deallocation. Plugin
EPs implement allocators that ORT calls to allocate memory for tensors.

An EP creates allocators via either:

- **`OrtEp::CreateAllocator`** — instance-level allocator creation (preferred if the allocator depends on EP instance state)
- **`OrtEpFactory::CreateAllocator`** — factory-level allocator creation (preferred if the allocator only depends on device info)

Each allocator is associated with an `OrtMemoryInfo` that describes where its memory lives.
ORT calls the EP's allocator creation function once per registered `OrtMemoryInfo` during session
initialization.

---

## Plugin EP Setup

### Registering Memory Info with OrtEpDevice

During `OrtEpFactory::GetSupportedDevices()`, the EP creates `OrtEpDevice` objects and registers
`OrtMemoryInfo` descriptors with each one. These descriptors tell ORT what memory spaces the EP
manages.

```cpp
// In GetSupportedDevices():
OrtEpDevice* ep_device = nullptr;
ep_api.CreateEpDevice(factory, &hardware_device, metadata, options, &ep_device);

// Register DEFAULT device memory (required)
ep_api.EpDevice_AddAllocatorInfo(ep_device, device_memory_info);

// Register HOST_ACCESSIBLE memory (optional — e.g., pinned memory)
ep_api.EpDevice_AddAllocatorInfo(ep_device, host_accessible_memory_info);

// Register read-only memory for initializers (optional)
ep_api.EpDevice_AddAllocatorInfo(ep_device, readonly_memory_info);
```

Each `OrtMemoryInfo` registered via `EpDevice_AddAllocatorInfo()` is stored in one of three
slots on the `OrtEpDevice`, based on its allocator type and memory type. Each slot holds at most
one `OrtMemoryInfo` — a later call in the same category overwrites the previous one.

| Allocator Type | Memory Type | OrtEpDevice field | Purpose |
|---------------|-------------|-------------------|---------|
| `OrtDeviceAllocator` | `DEFAULT` | `device_memory_info` | Primary device memory |
| `OrtDeviceAllocator` | `HOST_ACCESSIBLE` | `host_accessible_memory_info` | CPU-accessible device memory |
| `OrtReadOnlyAllocator` | `DEFAULT` | `read_only_device_memory_info` | Read-only initializer memory |

### Creating Allocators

ORT calls `OrtEp::CreateAllocator` (or `OrtEpFactory::CreateAllocator` as fallback) for each
`OrtMemoryInfo` collected from the selected `OrtEpDevice`(s). The EP receives the `OrtMemoryInfo`
and must return a matching `OrtAllocator`.

```
OrtEp::CreateAllocator(this_ptr, memory_info, &allocator)
// -or-
OrtEpFactory::CreateAllocator(this_ptr, memory_info, allocator_options, &allocator)
```

### Specifying the EP's Default Memory Device (Optional)

The EP's **default memory device** identifies the primary hardware the EP operates on. ORT uses it
to:

- Determine if data copies (memcpy nodes) are needed at EP boundaries
- Determine if the EP is CPU-based (affects synchronization and data transfer)
- Bind execution streams to the correct device

By default, ORT infers the default memory device from the `OrtMemoryInfo` registered with
`OrtDeviceMemoryType_DEFAULT` and `OrtDeviceAllocator` (i.e., the `device_memory_info` slot).
If there are multiple `OrtEpDevice` instances, their `device_memory_info` values must be
equivalent; otherwise, EP construction fails. If no `device_memory_info` is registered, ORT
falls back to a CPU device.

To override this, implement `OrtEp::GetDefaultMemoryDevice()`:

```cpp
OrtStatus* ORT_API_CALL GetDefaultMemoryDevice(const OrtEp* this_ptr,
                                                const OrtMemoryDevice** device) noexcept {
    auto* ep = static_cast<const MyEp*>(this_ptr);
    *device = ep_api.MemoryInfo_GetMemoryDevice(ep->my_preferred_memory_info);
    return nullptr;  // success
}
```

The returned `OrtMemoryDevice` must correspond to one of the `OrtMemoryInfo` instances registered
via `EpDevice_AddAllocatorInfo()` (specifically, matching `device_memory_info` or
`host_accessible_memory_info`). ORT validates this at EP construction time.

Setting `*device = nullptr` tells ORT to use the default inference behavior.

> **Since**: ORT API version 1.27. Optional — if not implemented (NULL function pointer), ORT
> infers the default from `device_memory_info`.

### Mapping Memory Types to Devices (Optional)

ORT categorizes tensor memory requirements using `OrtMemType` values:

| `OrtMemType` | Meaning |
|-------------|---------|
| `OrtMemTypeDefault` | Tensor lives on the EP's primary compute device |
| `OrtMemTypeCPUInput` | Input tensor declared as CPU-resident (e.g., shape tensors) |
| `OrtMemTypeCPUOutput` | Output tensor that should be CPU-accessible |

By default, ORT maps these as:

- `OrtMemTypeDefault` → EP's default memory device
- `OrtMemTypeCPUInput` → CPU device
- `OrtMemTypeCPUOutput` → CPU device

To customize this mapping, implement `OrtEp::GetMemoryDeviceByMemType()`:

```cpp
OrtStatus* ORT_API_CALL GetMemoryDeviceByMemType(const OrtEp* this_ptr,
                                                  OrtMemType mem_type,
                                                  const OrtMemoryDevice** device) noexcept {
    auto* ep = static_cast<const MyEp*>(this_ptr);

    switch (mem_type) {
        case OrtMemTypeCPUOutput:
            // Use pinned memory for CPU-accessible outputs instead of plain CPU
            *device = ep_api.MemoryInfo_GetMemoryDevice(ep->pinned_memory_info);
            return nullptr;
        default:
            // Return nullptr to use default behavior for other mem types
            *device = nullptr;
            return nullptr;
    }
}
```

Setting `*device = nullptr` for a given `OrtMemType` tells ORT to use the default mapping for that
type.

> **No validation**: ORT does not validate the returned `OrtMemoryDevice` against registered
> allocator infos. If the EP returns a device that has no matching allocator, the error will
> surface later at execution time when the allocator lookup fails.

> **Since**: ORT API version 1.27. Optional — if not implemented (NULL function pointer), ORT uses
> the default mapping for all memory types.

---

## Session Initialization

When a user creates an `Ort::Session` (C++ API) or calls `OrtApi::CreateSession` (C API), ORT
loads the model and initializes the session. During initialization, ORT performs the following
steps internally. This section explains what happens and how the EP's memory configuration
is used.

### 1. EP Registration and Default Memory Device

ORT constructs an internal `PluginExecutionProvider` wrapper for each plugin EP. During
construction, ORT determines the EP's **default memory device**:

1. If the EP implements `GetDefaultMemoryDevice()` and returns a non-NULL device, that device is
   used (after validation against registered memory infos).
2. Otherwise, ORT extracts the `OrtMemoryDevice` from the `device_memory_info` (the `OrtMemoryInfo`
   registered with `OrtDeviceMemoryType_DEFAULT` + `OrtDeviceAllocator`).

> **Internal detail**: The default memory device is stored as `default_device_` on the internal
> `IExecutionProvider` base class. It is returned by `IExecutionProvider::GetDevice()`.

### 2. Graph Optimization: Inserting Memcpy Nodes

During graph optimization, ORT's **MemcpyTransformer** examines data flowing between nodes assigned
to different EPs. The transformer only runs for non-CPU-based EPs (i.e., EPs whose default memory
device type is not CPU).

For each non-CPU-based EP, the transformer classifies every node as a **provider node** (compatible
with the EP) or a **non-provider node** (incompatible). A node is considered compatible if:

- It is assigned to the **same EP type** (e.g., both are `"MyPluginEP"`), **or**
- It is assigned to a different EP type but shares the **same device type and vendor ID**.

For data flowing between compatible and incompatible nodes, ORT inserts `MemcpyFromHost` or
`MemcpyToHost` copy nodes to make the data transfer explicit.

The compatibility check also considers kernel-level annotations — if a kernel declares a specific
input as CPU-resident (e.g., shape tensors via `IsInputOnCpu`), no copy is inserted for that input.

#### Initializer duplication

If an initializer (e.g., a weight tensor `W`) is referenced by both provider and non-provider
nodes, the MemcpyTransformer **duplicates** it. It creates a new initializer `W2` with the same
data and rewrites all provider-node references to point to `W2`. After this transformation, each
initializer is consumed by nodes on only one device type. This means the weight data will exist
twice in memory — once on the CPU (original `W`) and once on the EP's device (duplicate `W2`,
allocated and copied during initializer allocation).

> **Subgraph exception**: Initializers shared across different graph levels (e.g., main graph vs.
> a subgraph inside an `If` or `Loop` node) are **not** duplicated. Instead, ORT copies them to
> the correct device at subgraph execution time.

> **Internal detail**: The MemcpyTransformer uses `IExecutionProvider::GetDevice()` (which returns
> the EP's default memory device) for the device type and vendor comparison. It does not use
> `GetMemoryDeviceByMemType()`.

### 3. Allocator Registration

ORT calls `CreatePreferredAllocators()` on each EP (in registration order). For plugin EPs, this
iterates over all `OrtMemoryInfo` instances registered via `EpDevice_AddAllocatorInfo()` and calls
the EP's `CreateAllocator` for each one.

The returned allocators are stored in an internal allocator map keyed by `OrtMemoryDevice`:

```
AllocatorMap[OrtMemoryDevice] = allocator
```

**Priority rule**: The first EP to register an allocator for a given `OrtMemoryDevice` wins.
Later EPs cannot overwrite an existing entry. This means EP registration order matters when
multiple EPs use overlapping device types (e.g., two GPU EPs on the same device).

### 4. Allocation Planning

After graph optimization, ORT's **allocation planner** assigns a concrete `OrtMemoryDevice`
location to every tensor (OrtValue) in the graph. For each tensor, the planner:

1. Looks up the EP assigned to the node that produces or consumes the tensor.
2. Checks the kernel definition for the tensor's declared `OrtMemType` (e.g., `OrtMemTypeDefault`
   for most tensors, `OrtMemTypeCPUInput` for shape tensors).
3. Calls the EP's memory type mapping to resolve the `OrtMemType` to a specific `OrtMemoryDevice`.

For plugin EPs, this resolution works as follows:

- If `OrtEp::GetMemoryDeviceByMemType()` is implemented and returns a non-NULL device, that device
  is used.
- Otherwise, ORT uses the default mapping: `OrtMemTypeDefault` → EP's default device, `OrtMemTypeCPUInput`/`OrtMemTypeCPUOutput` → CPU.

> **Internal detail**: The allocation planner calls `IExecutionProvider::GetOrtDeviceByMemType()`,
> which the `PluginExecutionProvider` wrapper overrides to delegate to the OrtEp hooks described
> above.

#### Downstream consumer CPU location override

When a producer node's output is placed on CPU, the planner additionally consults **downstream
consumer** nodes. For each consumer, it calls `GetMemoryDeviceByMemType(OrtMemTypeCPUInput)` to
ask: "Where would you prefer this CPU-resident data to live?" This allows an EP to express a
preference for a specific CPU-accessible memory type (e.g., pinned host memory) without requiring
an explicit copy node.

If the consumer's suggested device is CPU-typed, the planner selects the "best" device among all
consumers. If the suggested device uses CPU-accessible memory but is not strictly a CPU device
(e.g., pinned memory on a GPU device), the planner uses that device directly.

> **Compiling EPs with a CPU default device**: Because the fused node's kernel definition does not
> declare per-input memory types, the planner resolves the fused node's own inputs with
> `OrtMemTypeDefault`. However, if the compiling EP uses a CPU default memory device (and thus no
> Memcpy nodes are inserted), the consumer override path above still applies — the planner calls
> `GetMemoryDeviceByMemType(OrtMemTypeCPUInput)` to determine the preferred CPU-accessible
> placement for upstream CPU outputs consumed by the fused node.

### 5. Initializer Allocation

Model initializers (weights, biases, etc.) are allocated during session initialization — before
any inference runs. The allocation strategy depends on where the initializer data resides and
where it needs to be placed.

#### Embedded initializers (data stored inside the ONNX model)

- **CPU-targeted**: ORT allocates memory using the CPU allocator (not the arena) and deserializes
  the data directly into the buffer.
- **Non-CPU-targeted** (e.g., GPU): ORT allocates memory on the target device using the EP's
  allocator, deserializes the data into a temporary CPU buffer, and then copies it to the device
  via the EP's **data transfer** implementation.

#### External initializers (data stored in separate files)

- **CPU-targeted**: ORT **memory-maps** the external file and uses the mapped buffer directly —
  no allocator is involved and no copy occurs. This is an important optimization for large models.
- **Non-CPU-targeted** (e.g., GPU): ORT allocates memory on the target device using the EP's
  allocator, memory-maps the external file on the CPU, and then copies the data to the device
  via the EP's **data transfer** implementation.

> **Note**: Built-in EPs can register a custom `ExternalDataLoader` to override the default
> memory-mapping behavior for external initializers. This capability is not yet available to
> plugin EPs.

If the EP registered a read-only allocator (`OrtReadOnlyAllocator`), ORT uses it preferentially
for initializers instead of the standard device allocator.

#### Compiling EPs and `drop_constant_initializers`

A compiling EP (one that implements `OrtEp::Compile()`) fuses supported nodes into a single
compiled subgraph. The `OrtNodeFusionOptions.drop_constant_initializers` field controls how
initializers used by the fused subgraph are handled:

- **`drop_constant_initializers = false`** (default): Constant initializers are listed as inputs
  to the fused node. ORT allocates and retains the initializer OrtValues so they can be passed
  to the fused node's `Compute` function at inference time.

- **`drop_constant_initializers = true`**: The EP signals that it has internally copied or
  embedded the initializer data within its compiled representation. ORT excludes these constant
  initializers from the fused node's input list. If no other EP or node references the
  initializer, ORT can release the associated OrtValue, freeing memory.

> For more details on graph partitioning, node fusion, and the `Compile` workflow, refer to
> a separate graph partitioning document (TODO).

---

## Execution

### Intermediate Tensor Allocation

During inference (`Session::Run()`), ORT allocates memory for intermediate and output tensors
according to the allocation plan created during session initialization. For each tensor:

1. ORT looks up the `OrtMemoryDevice` assigned during allocation planning.
2. ORT retrieves the allocator from the allocator map for that device.
3. The allocator allocates the required memory.

If no allocator is found for the assigned `OrtMemoryDevice`, execution fails with an error like:
`"Failed to get allocator for Device:[DeviceType:X MemoryType:Y DeviceId:Z]"`.

### Streams

For EPs that support stream-based execution (e.g., GPU compute streams), ORT creates execution
streams bound to specific `OrtMemoryDevice` values. Each stream is associated with a non-CPU
memory device from the EP's registered allocator infos.

- ORT calls `OrtEp::CreateSyncStreamForDevice()` (or `OrtEpFactory::CreateSyncStreamForDevice()`)
  with the target `OrtMemoryDevice`.
- During execution, ORT binds kernel execution to the appropriate stream based on the
  `OrtMemoryDevice` assigned to the node's tensors.
- Streams enable concurrent kernel execution and asynchronous data transfers on devices that
  support them.

> **Opt-in**: Stream support requires the EP factory to implement `OrtEpFactory::IsStreamAware()`
> returning `true`. CPU-accessible memory devices are excluded from stream creation.

### Data Transfer

When data must move between different `OrtMemoryDevice` locations (e.g., CPU → GPU, or between
two different memory types on the same device), ORT uses the EP's **data transfer** implementation.

The EP implements data transfer via `OrtDataTransferImpl`:

```cpp
struct OrtDataTransferImpl {
    // Can this implementation copy between src and dst?
    bool CanCopy(const OrtMemoryDevice* src, const OrtMemoryDevice* dst);

    // Perform the copy
    OrtStatus* CopyTensors(const OrtValue** src, OrtValue** dst,
                           OrtSyncStream** streams, size_t count);
};
```

Data transfers are triggered automatically when the source and destination `OrtMemoryDevice` values
differ — for example, when a CPU EP produces a tensor that a GPU EP consumes, or when initializer
data on the CPU must be copied to device memory.

---

## Summary Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Session Initialization                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. EP Registration                                                 │
│     └─ Default memory device determined                             │
│        (from GetDefaultMemoryDevice or device_memory_info)          │
│                                                                     │
│  2. Graph Optimization (MemcpyTransformer)                          │
│     └─ Compares default memory devices across EPs                   │
│     └─ Inserts MemcpyFromHost / MemcpyToHost nodes                  │
│                                                                     │
│  3. Allocator Registration                                          │
│     └─ CreateAllocator() called for each registered OrtMemoryInfo   │
│     └─ AllocatorMap[OrtMemoryDevice] = allocator                    │
│                                                                     │
│  4. Allocation Planning                                             │
│     └─ GetMemoryDeviceByMemType() maps each tensor to a device      │
│     └─ Each OrtValue gets an assigned OrtMemoryDevice location      │
│                                                                     │
│  5. Initializer Allocation                                           │
│     └─ Embedded: allocate + deserialize (+ copy if non-CPU)         │
│     └─ External on CPU: memory-mapped directly (no allocator)       │
│     └─ External on non-CPU: memory-map + copy via DataTransfer      │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                             Execution                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  6. Tensor Allocation & Kernel Execution                             │
│     └─ Allocators allocate intermediate/output tensors per plan     │
│     └─ Streams bind kernel execution to target device               │
│     └─ DataTransfer copies data between different OrtMemoryDevices  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```
