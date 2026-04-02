# CUDA Plugin EP — Arena Allocator Integration Design

## 1. Problem Statement

The CUDA plugin EP currently uses raw `cudaMalloc`/`cudaFree` through `CudaDeviceAllocator` (an `OrtAllocator*` wrapper). The in-tree (bridge-based) CUDA EP wraps its allocators in arenas by default:

| Allocator | In-Tree CUDA EP | Plugin CUDA EP (today) |
|-----------|----------------|----------------------|
| GPU device | `CUDAAllocator` → arena (stream-aware) | `CudaDeviceAllocator` → raw `cudaMalloc`/`cudaFree` |
| GPU device (mempool) | `CudaMempoolArena` (native CUDA mempool) | Not available |
| Pinned (host) | `CUDAPinnedAllocator` → arena (non-stream-aware) | `CudaPinnedAllocator` → raw `cudaHostAlloc`/`cudaFreeHost` |

This gap means the plugin EP has significantly worse allocation performance for typical workloads.

---

## 2. Reference Implementation: Example Plugin EP Arena

The ORT test suite contains a complete reference implementation of a plugin-hosted arena in `onnxruntime/test/autoep/library/example_plugin_ep/`:

| File | Purpose |
|------|---------|
| `ep_arena.h` | `ArenaConfig`, `ArenaImpl` (arena allocator — ~632 lines), `ArenaAllocator` (OrtAllocator wrapper) |
| `ep_arena.cc` | `ArenaImpl` implementation: bins, chunks, region management, stream-aware allocation |
| `ep_allocator.h` | `BaseAllocator` (virtual dtor for `OrtAllocator`), `CustomAllocator` (raw malloc/free device allocator), `AllocatorStats` |
| `ep_factory.cc` | `CreateAllocatorImpl` — creates shared `ArenaAllocator` wrapping `CustomAllocator`; ref-counted lifecycle |
| `ep_stream_support.cc` | `StreamImpl::OnSessionRunEndImpl` — calls `arena->ResetChunksUsingStream()` |

### 2.1 Key Design Patterns

**Arena lives inside the plugin.** The arena implementation is self-contained in the plugin library. ORT core sees only an `OrtAllocator*` with `OrtDeviceAllocator` type — it is unaware that the allocator internally manages an arena. This is the intended plugin EP architecture: the EP library owns its allocation strategy.

**Factory creates a shared arena.** `ExampleEpFactory::CreateAllocatorImpl` creates one `ArenaAllocator` instance on first call and returns the same pointer on subsequent calls, with reference counting:

```cpp
// ep_factory.cc — CreateAllocatorImpl (simplified)
if (!factory.arena_allocator_) {
  AllocatorUniquePtr ep_allocator = std::make_unique<CustomAllocator>(memory_info, factory);
  factory.arena_allocator_using_default_settings_ = allocator_options == nullptr;
  ArenaAllocator::CreateOrtArenaAllocator(std::move(ep_allocator), allocator_options,
                                          factory.ort_api, factory.default_logger_,
                                          factory.arena_allocator_);
} else {
  if (factory.arena_allocator_using_default_settings_ && allocator_options) {
    // arena settings may have changed — EP decides how to handle
  }
}
++factory.num_arena_users_;
*allocator = factory.arena_allocator_.get();
```

**Arena config via `OrtKeyValuePairs`.** `ArenaConfig::FromKeyValuePairs()` parses standard `arena.*` keys:

| Key | Type | Default |
|-----|------|---------|
| `arena.extend_strategy` | `"0"` (power of two) or `"1"` (same as requested) | `kNextPowerOfTwo` |
| `arena.initial_chunk_size_bytes` | int | 1 MB |
| `arena.max_dead_bytes_per_chunk` | int | 128 MB |
| `arena.initial_growth_chunk_size_bytes` | int | 2 MB |
| `arena.max_power_of_two_extend_bytes` | int64 | 1 GB |
| `arena.max_mem` | size_t | `SIZE_MAX` |

**Stream-aware allocation.** `ArenaImpl::AllocOnStream(size, stream)` tracks which chunks are assigned to which stream. `ResetChunksUsingStream(stream_impl)` is called from `OrtSyncStreamImpl::OnSessionRunEnd` to release chunk-to-stream assignments when a session run completes.

**Read-only allocator bypasses arena.** The factory creates a plain `CustomAllocator` (no arena) for `OrtReadOnlyAllocator` (initializers), since initializer memory doesn't benefit from arena allocation.

### 2.2 How ORT Core Calls the Factory

**Path 1: Shared allocators (environment level)**
```
RegisterExecutionProviderLibrary()
  → CreateSharedAllocatorImpl(ep_device, memory_info, OrtDeviceAllocator, nullptr, ...)
    → ep_factory->CreateAllocator(factory, &mem_info, /*options=*/ nullptr, &alloc)
      → [factory creates ArenaAllocator wrapping raw allocator]
    → IAllocatorImplWrappingOrtAllocator(alloc)
    → shared_allocators_.push_back(wrapped)
```

**Path 2: Per-session allocators**
```
SessionState constructor
  → ep->CreatePreferredAllocators()
    → PluginExecutionProvider::CreatePreferredAllocators()
      → OrtEp::CreateAllocator(ep, &mem_info, &alloc)   [if set]
        OR ep_factory.CreateAllocator(&factory, &mem_info, /*options=*/ nullptr, &alloc)
        → [factory returns same shared ArenaAllocator]
      → IAllocatorImplWrappingOrtAllocator(alloc)
    → session allocator maps
```

**Path 3: User-created allocators (public API)**
```
OrtApi::CreateSharedAllocator(env, ep_device, mem_type, alloc_type, allocator_options, &alloc)
  → Environment::CreateSharedAllocator()
    → CreateSharedAllocatorImpl(ep_device, mem_info, alloc_type, allocator_options, &alloc, replace=true)
      → ep_factory->CreateAllocator(factory, &mem_info, allocator_options, &alloc)
        → [factory creates ArenaAllocator with user-provided config]
```

**Key point:** `CreateSharedAllocatorImpl` explicitly rejects `OrtArenaAllocator` type from plugin factories and verifies the returned allocator doesn't use it either. The arena is opaque — ORT core sees `OrtDeviceAllocator`.

---

## 3. Applying the Pattern to CUDA Plugin EP

The CUDA plugin EP should follow the example plugin's architecture: **the arena lives inside the plugin library**. The previous design explored ORT-core-wrapping approaches (wrapping plugin allocators in ORT's internal arena). The example plugin EP demonstrates the intended approach: the EP library includes its own arena and wraps its raw allocators (both device and pinned) internally.

### 3.1 What Needs to Change in the CUDA Plugin Factory

`CudaEpFactory::CreateAllocatorImpl` currently creates raw `CudaDeviceAllocator` or `CudaPinnedAllocator` and returns them directly. The change:

```cpp
// Current (cuda_ep_factory.cc — CreateAllocatorImpl):
if (strcmp(name, "Cuda") == 0) {
  auto cuda_allocator = std::make_unique<CudaDeviceAllocator>(memory_info, req_device_id);
  *allocator = cuda_allocator.release();  // raw cudaMalloc/cudaFree
}

// Target: wrap in ArenaAllocator, following the example plugin pattern.
// NOTE: The factory must maintain a separate arena per device_id, since each GPU
// has its own memory space. The factory already has a device_cache_ mapping
// HardwareDeviceKey → DeviceCacheEntry; the arena is stored there.
if (strcmp(name, "Cuda") == 0) {
  auto& entry = factory.GetOrCreateDeviceCacheEntry(req_device_id);
  std::lock_guard<std::mutex> lock{entry.arena_mutex};

  if (/* use_cuda_mempool option */) {
    // CudaMempoolArena path — see Section 4
  } else if (!entry.device_arena) {
    // Arena path — first call for this device:
    auto raw_allocator = std::make_unique<CudaDeviceAllocator>(memory_info, req_device_id);
    entry.device_arena_using_defaults = (allocator_options == nullptr);
    ArenaAllocator::CreateOrtArenaAllocator(std::move(raw_allocator), allocator_options,
                                            factory.ort_api_, factory.default_logger_,
                                            entry.device_arena);
  }
  ++entry.num_device_arena_users;
  *allocator = entry.device_arena.get();
}

if (strcmp(name, "CudaPinned") == 0) {
  // Pinned memory is CPU-side and technically shared, but each device's pinned
  // allocator has a distinct OrtMemoryInfo (device_id). Keep per-device.
  auto& entry = factory.GetOrCreateDeviceCacheEntry(req_device_id);
  std::lock_guard<std::mutex> lock{entry.arena_mutex};

  if (!entry.pinned_arena) {
    auto raw_allocator = std::make_unique<CudaPinnedAllocator>(memory_info);
    ArenaAllocator::CreateOrtArenaAllocator(std::move(raw_allocator), allocator_options,
                                            factory.ort_api_, factory.default_logger_,
                                            entry.pinned_arena);
  }
  ++entry.num_pinned_arena_users;
  *allocator = entry.pinned_arena.get();
}
```

### 3.2 Adapting the Arena Code for CUDA

The `ep_arena.h`/`ep_arena.cc` from the example plugin are designed to be copied and adapted. For the CUDA plugin EP, the raw allocator (`CustomAllocator` in the example) is replaced with `CudaDeviceAllocator` (for GPU) or `CudaPinnedAllocator` (for pinned). Since `ArenaImpl` takes an `AllocatorUniquePtr` (a `std::unique_ptr<BaseAllocator>`) — and `BaseAllocator` inherits from `OrtAllocator` — the CUDA allocators need to either:

**(a) Inherit from `BaseAllocator`** instead of inheriting from `OrtAllocator` directly (preferred — minimal change, adds virtual dtor), or

**(b) Create thin adapters** wrapping `CudaDeviceAllocator`/`CudaPinnedAllocator` in `BaseAllocator`.

Option (a) is simpler. `CudaAllocatorBase` (the current common base for CUDA allocators) would change from `OrtAllocator` to `BaseAllocator`:

```cpp
// Current:
class CudaAllocatorBase : public OrtAllocator { ... };
// Change to:
class CudaAllocatorBase : public BaseAllocator { ... };
```

This is a non-breaking change since `BaseAllocator` only adds a virtual destructor.

### 3.3 Shared Arena Lifecycle and Reference Counting

**Multi-GPU consideration.** A system may have multiple CUDA devices. Each GPU has its own device memory, so each needs its own arena. The CUDA plugin factory already maintains a per-device cache (`device_cache_`) mapping `HardwareDeviceKey → DeviceCacheEntry` that stores `OrtMemoryInfo` instances per GPU. The arena pointers and ref counts are added to this existing cache structure:

```cpp
// Existing structure in cuda_ep_factory.h — extended with arena members:
struct DeviceCacheEntry {
  int cuda_device_id{-1};
  Ort::MemoryInfo device_memory_info{nullptr};      // GPU device memory
  Ort::MemoryInfo pinned_memory_info{nullptr};      // CPU pinned memory for this GPU

  // Arena members (new):
  std::mutex arena_mutex;
  std::unique_ptr<ArenaAllocator> device_arena;
  std::unique_ptr<ArenaAllocator> pinned_arena;
  int num_device_arena_users = 0;
  int num_pinned_arena_users = 0;
  bool device_arena_using_defaults = true;
};
```

The factory's `device_cache_` is populated during `GetSupportedDevicesImpl` (one entry per GPU discovered). `CreateAllocatorImpl` extracts the `device_id` from the incoming `OrtMemoryInfo`, locates the corresponding `DeviceCacheEntry`, and creates/returns the arena for that device. Each GPU gets independent arena instances with independent lifecycle.

`CreateAllocatorImpl` creates the arena on first call for a given device and increments its ref count. `ReleaseAllocatorImpl` decrements; when zero, the arena is destroyed. This handles both:
- **Shared allocators** — `RegisterExecutionProviderLibrary` iterates over each `OrtEpDevice` and calls `CreateAllocator` for each device's memory infos. Each device gets its own shared arena.
- **Per-session allocators** — each session calls `CreateAllocator` (returning the same shared arena for the device) and `ReleaseAllocator` on session teardown.

The `OrtApi::CreateSharedAllocator` public API also flows through `CreateAllocatorImpl` with `replace_existing=true`. When replacing, `ReleaseAllocator` is called on the old allocator first (dropping that device's arena if ref count hits zero), then `CreateAllocator` is called again with the new options — potentially creating a new arena with different config for that specific device.

**Note:** The example plugin EP uses single `arena_allocator_` / `num_arena_users_` members because it only registers for one device (`device_id=0`). The CUDA plugin must generalize this to per-device storage.

### 3.4 Stream Integration

The CUDA plugin's `StreamImpl` (from `OrtSyncStreamImpl`) must call `ResetChunksUsingStream` on the device arena at session run end, following the example. Since there may be multiple GPUs, the stream must know which device's arena to reset. Each stream is created for a specific `OrtMemoryDevice`, which has a device_id — this maps to the corresponding `DeviceCacheEntry`:

```cpp
// cuda stream_support.cc — OnSessionRunEndImpl:
OrtStatus* ORT_API_CALL CudaStreamImpl::OnSessionRunEndImpl(OrtSyncStreamImpl* this_ptr) noexcept {
  auto& impl = *static_cast<CudaStreamImpl*>(this_ptr);
  // impl.device_id_ was set at stream creation from the OrtMemoryDevice
  auto* arena = impl.factory_->GetDeviceArenaAllocator(impl.device_id_);
  if (arena) {
    arena->ResetChunksUsingStream(this_ptr);
  }
  return nullptr;
}
```

`GetDeviceArenaAllocator(device_id)` looks up the `DeviceCacheEntry` for the given device and returns its `device_arena.get()`.

The pinned allocator is also wrapped in the same `ArenaAllocator` but does not need stream-aware allocation (matching the in-tree EP where pinned uses a non-stream-aware arena). `AllocOnStream` is not invoked for pinned memory, and `ResetChunksUsingStream` is not called for the pinned arena at session run end.

### 3.5 Arena Config Flow

**Shared allocators (environment level):**

`RegisterExecutionProviderLibrary` calls `CreateSharedAllocatorImpl` with `allocator_options = nullptr`. This means the factory's first arena creation uses default `ArenaConfig` values. This is acceptable:
- The defaults (1 MB initial chunk, 128 MB max dead, kNextPowerOfTwo growth) are reasonable.
- If the user configures arena options via `OrtApi::CreateSharedAllocator` later, the old allocator is released and a new one is created with the provided options (because `replace_existing=true`).

**Per-session allocators:**

`CreatePreferredAllocators` also calls with `allocator_options = nullptr` today. Options arrive at the factory if the user calls `OrtApi::CreateSharedAllocator` with explicit options. Since per-session calls reuse the shared arena (ref counting), the arena config is effectively set at first creation time.

**User-provided config via `CreateEnvWithOptions`:**

Environment-level config can be passed via `OrtEnvCreationOptions::config_entries`:

```cpp
api->AddKeyValuePair(kvps, "ep_factory.cudapluginexecutionprovider.arena.extend_strategy", "1");
api->AddKeyValuePair(kvps, "ep_factory.cudapluginexecutionprovider.arena.max_mem", "4294967296");

OrtEnvCreationOptions options{};
options.config_entries = kvps;
api->CreateEnvWithOptions(&options, &env);
```

**Current gap:** `RegisterExecutionProviderLibrary` does not extract env config entries and pass them as `allocator_options` to `CreateSharedAllocatorImpl`. To support env-level arena config, this needs to be plumbed:

1. `RegisterExecutionProviderLibrary` constructs a prefix via `"ep_factory." + GetLowercaseString(factory->GetName()) + "."` and scans `Environment::config_entries_` for matching `arena.*` keys (see Section 3.6 for casing convention)
2. Strips the prefix and builds `OrtKeyValuePairs` with bare `arena.*` keys
3. Passes to `CreateSharedAllocatorImpl` as `allocator_options`
4. `CreateSharedAllocatorImpl` forwards to `ep_factory->CreateAllocator`

This is a small ORT core change that enables the existing config mechanism to reach the plugin's arena.

### 3.6 Environment vs. Session Config

ORT has two separate configuration namespaces for EP-specific options.

#### Current state

| | Environment-level | Session-level |
|---|---|---|
| **Prefix pattern** | `ep_factory.<ep_name>.` | `ep.<ep_name>.` |
| **Who constructs the prefix?** | No one — convention from C API doc comments only | ORT core (`GetProviderOptionPrefix`) |
| **Lowercasing applied?** | **Not defined** — ORT never constructs or parses this prefix today | **Yes** — `GetLowercaseString(GetName())` |
| **Backing store** | `std::map<string,string>` (case-sensitive) | `std::unordered_map<string,string>` (case-sensitive) |
| **Set via** | `CreateEnvWithOptions` (`OrtEnvCreationOptions.config_entries`) | `SessionOptionsAppendExecutionProvider_V2` |
| **CUDA plugin `GetName()`** | `"CudaPluginExecutionProvider"` | `"CudaPluginExecutionProvider"` |

The C API documentation (`onnxruntime_c_api.h`) describes the environment-level prefix as `ep_factory.<ep_name>.` where `<ep_name>` is the factory's own name (from `OrtEpFactory::GetName()`), **not** the user-provided registration name passed to `RegisterExecutionProviderLibrary`. However, ORT core does not currently construct, parse, or normalize this prefix — it is purely a documentation convention. The design (Section 3.5 / 5.3) proposes new code in `RegisterExecutionProviderLibrary` that would extract these keys for the first time, which requires deciding on a casing convention.

The session-level prefix is always lowercased by ORT via `GetLowercaseString`:

```cpp
// abi_session_options.cc — GetProviderOptionPrefix
std::string key_prefix = "ep.";
key_prefix += onnxruntime::utils::GetLowercaseString(provider_name);
key_prefix += ".";
```

Both backing stores (`std::map` and `std::unordered_map`) use exact string comparison — key lookup is case-sensitive.

#### Casing convention for `ep_factory.` prefix

Since new code must be written to extract `ep_factory.` keys, we must decide how the `<ep_name>` portion is matched:

| Option | Env-level example key | Pros | Cons |
|--------|----------------------|------|------|
| **(A) Use `GetName()` as-is** | `ep_factory.CudaPluginExecutionProvider.arena.*` | Exact match to factory identity; unambiguous | Inconsistent with session-level (lowercase); users must get casing exactly right; error-prone |
| **(B) Lowercase like session-level** | `ep_factory.cudapluginexecutionprovider.arena.*` | Consistent with `ep.cudapluginexecutionprovider.*`; users see one pattern | Diverges from C API doc comment which doesn't specify lowercasing; slight surprise if user reads `GetName()` |
| **(C) Case-insensitive matching** | Either casing works | Most forgiving for users | Requires scanning all map entries (can't use `std::map::find`); unusual; extra code |

**Recommendation: Option B** — lowercase the `<ep_name>` when constructing the env-level prefix, matching the session-level convention. Both paths then use `GetLowercaseString(GetName())`:

```
Environment: ep_factory.cudapluginexecutionprovider.arena.extend_strategy
Session:     ep.cudapluginexecutionprovider.arena.extend_strategy
```

This means the new code in `RegisterExecutionProviderLibrary` would construct the prefix as:

```cpp
std::string prefix = "ep_factory." + onnxruntime::utils::GetLowercaseString(factory->GetName()) + ".";
```

#### Conflict between namespaces

The EP is blind to conflicts between these two namespaces. This is acceptable because:
- Shared allocators run before any session exists — only env config applies.
- Per-session allocators reuse the factory's shared arena — the arena config is determined at first creation.
- The two config paths are independent and serve different lifecycle scopes.

**Runtime validation (recommended):** When `CreateAllocatorImpl` receives `allocator_options` and the factory already holds a shared arena for that device, log a warning if the incoming keys differ from the keys used at first creation. This makes misconfiguration visible without silently ignoring the second set of options.

---

## 4. Migrating `CudaMempoolArena` to the Plugin

### 4.1 Overview

`CudaMempoolArena` is CUDA's native memory pool (`cudaMallocFromPoolAsync`/`cudaFreeAsync`). It is an alternative to the plugin's arena for GPU device memory — mutually exclusive, selected by config. It is self-contained (CUDA SDK only) and already stream-aware.

### 4.2 Current Dependencies

| Dependency | Plugin-Safe? | Notes |
|-----------|-------------|-------|
| `<cuda_runtime_api.h>` | ✅ | CUDA SDK — always available |
| `core/common/common.h` | ✅ | `ORT_THROW`, `ORT_ENFORCE` — no framework deps |
| `core/providers/cuda/cuda_stream_handle.h` | ✅ | Only for `Stream::GetHandle()` → `cudaStream_t` |
| `core/providers/cuda/shared_inc/cuda_call.h` | ✅ | CUDA error-handling macros |
| `core/providers/shared_library/provider_api.h` | ❌ | Provider-bridge header defining `logging::Logger` forward decl used by `CudaMempoolArena`; must be removed/guarded in plugin build |
| `logging::Logger*` | ❌ | **Primary blocker** — provider-bridge logger type (from `provider_api.h`), not available in plugin build |

### 4.3 Logger Adaptation

Replace `const logging::Logger* logger_` with a build-conditional type using `#ifdef BUILD_CUDA_EP_AS_PLUGIN`. This follows the established pattern already used across 20+ CUDA provider files (`cuda_common.h`, `cuda_kernel.h`, `cudnn_common.h`, `space_depth_ops.h`, `identity_op.cc`, `pad.cc`, `scatter_nd.cc`, etc.) where shared headers use `#ifdef BUILD_CUDA_EP_AS_PLUGIN` to adapt between in-tree and plugin builds:

```cpp
#ifdef BUILD_CUDA_EP_AS_PLUGIN
  const OrtApi& ort_api_;                  // stored reference to OrtApi (set at construction)
  const OrtLogger* logger_;                // plugin: OrtLogger from EP C API
  // Logger_LogMessage returns OrtStatus* which must be released if non-null.
  #define MEMPOOL_LOG(ort_api_ref, logger, level, msg) do {          \
    OrtStatus* _s = (ort_api_ref).Logger_LogMessage(                 \
        (logger), ORT_LOGGING_LEVEL_##level,                         \
        (msg).c_str(), ORT_FILE, __LINE__, __FUNCTION__);            \
    if (_s) (ort_api_ref).ReleaseStatus(_s);                         \
  } while (0)
#else
  const logging::Logger* logger_;          // in-tree: ORT internal logger
  #define MEMPOOL_LOG(ort_api_ref, logger, level, msg) LOGS(*logger, level) << msg
#endif
```

The plugin build stores a `const OrtApi&` reference (passed at construction from the factory) so the macro can call `Logger_LogMessage`. The returned `OrtStatus*` is released if non-null — logging failures are not propagated.

**Decision:** Use the `#ifdef` macro approach (not a virtual `ICudaMempoolLogger` interface) for consistency with the existing codebase convention.

### 4.4 OrtAllocator Wrapper

The factory returns `CudaMempoolArena` wrapped behind `OrtAllocator*`, not inheriting from `IArena`/`IAllocator`. Following the same pattern as the device arena:

```cpp
struct CudaMempoolOrtAllocator : BaseAllocator {
  static OrtStatus* Create(const OrtMemoryInfo* memory_info,
                           const OrtKeyValuePairs* options,
                           const OrtApi& api,
                           const OrtLogger& logger,
                           std::unique_ptr<CudaMempoolOrtAllocator>& out);

  // OrtAllocator callbacks — delegate to CudaMempoolArena
  static void* ORT_API_CALL AllocImpl(OrtAllocator* this_, size_t size);
  static void* ORT_API_CALL AllocOnStreamImpl(OrtAllocator* this_, size_t size, OrtSyncStream* stream);
  static void ORT_API_CALL FreeImpl(OrtAllocator* this_, void* p);
  static void* ORT_API_CALL ReserveImpl(OrtAllocator* this_, size_t size);
  static const OrtMemoryInfo* ORT_API_CALL InfoImpl(const OrtAllocator* this_);
  static OrtStatus* ORT_API_CALL GetStatsImpl(const OrtAllocator* this_, OrtKeyValuePairs** out) noexcept;

 private:
  const OrtApi& ort_api_;                    // needed for SyncStream_GetHandle, KVP creation
  std::unique_ptr<CudaMempoolArena> arena_;
  const OrtMemoryInfo& memory_info_;
};
```

`AllocOnStreamImpl` resolves `OrtSyncStream*` → `cudaStream_t` via `OrtApi::SyncStream_GetHandle()`. This requires the wrapper to store a reference to `const OrtApi&` (already present via the `Create` factory method's `api` parameter). The stored `OrtApi` reference is also needed for `GetStatsImpl` (to create `OrtKeyValuePairs`) and for `Create` itself (to parse config options). The `OrtApi` pointer is available in all allocator callback contexts because it is captured in the `CudaMempoolOrtAllocator` instance that `this_` points to.

**OrtMemoryInfo type:** Must be `OrtDeviceAllocator` (ORT core rejects `OrtArenaAllocator` from plugins).

### 4.5 Arena Mode Selection in CreateAllocatorImpl

The factory selects between the plugin's arena and CUDA mempool based on allocator options:

```cpp
OrtStatus* CudaEpFactory::CreateAllocatorImpl(OrtEpFactory* this_ptr,
                                              const OrtMemoryInfo* memory_info,
                                              const OrtKeyValuePairs* allocator_options,
                                              OrtAllocator** allocator) noexcept {
  auto& factory = *static_cast<CudaEpFactory*>(this_ptr);
  // ...
  if (strcmp(name, "Cuda") == 0) {
    bool use_mempool = false;
    if (allocator_options) {
      const char* v = factory.ort_api_.GetKeyValue(allocator_options, "arena.use_cuda_mempool");
      use_mempool = v && std::string(v) == "1";
    }

    if (use_mempool) {
      return CudaMempoolOrtAllocator::Create(memory_info, allocator_options,
                                             factory.ort_api_, factory.default_logger_,
                                             factory.mempool_arena_);
      // ... ref counting as for the arena
    } else {
      // Arena path (Section 3.1)
    }
  }
}
```

### 4.6 Config Keys for Mempool

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `arena.use_cuda_mempool` | `"0"` or `"1"` | `"0"` | Enable CUDA native mempool instead of the plugin arena |
| `arena.cuda_mempool_release_threshold` | uint64 bytes | `0` | `cudaMemPoolAttrReleaseThreshold` value |
| `arena.cuda_mempool_bytes_to_keep_on_shrink` | size_t bytes | `0` | Target for `cudaMemPoolTrimTo()` on `Shrink()` |

---

## 5. Summary of Changes

### 5.1 Files Copied from Example Plugin EP

The arena implementation in `onnxruntime/test/autoep/library/example_plugin_ep/` is the reference. Two files are copied into the CUDA plugin directory and adapted:

| Source | Target | What to copy | Adaptations needed |
|---|---|---|---|
| `ep_arena.h` (~632 lines) | `plugin/cuda_arena.h` | `ArenaExtendStrategy` enum, `ArenaConfig` struct (with `FromKeyValuePairs` parser and `ConfigKeyNames`), `ArenaImpl` class (full arena implementation), `ArenaAllocator` struct (OrtAllocator wrapper) | **Namespace:** Wrap in `onnxruntime::cuda_plugin`. **Includes:** Replace `#include "ep_allocator.h"` and `#include "../plugin_ep_utils.h"` with `#include "cuda_allocator_plugin.h"` and `#include "cuda_plugin_utils.h"`. **`AllocatorUniquePtr`:** Already defined as `std::unique_ptr<BaseAllocator>` — redefine in this file or in `cuda_allocator_plugin.h` (see 5.2). **Macros:** The `EP_ENFORCE`, `LOG`, `RETURN_ERROR` macros come from `plugin_ep_utils.h`; replace with equivalents from `cuda_plugin_utils.h` or define locally (see 5.2). **No CUDA-specific changes** — the arena operates on the `OrtAllocator` C interface and is CUDA-agnostic. |
| `ep_arena.cc` (~750 lines) | `plugin/cuda_arena.cc` | Full `ArenaImpl` implementation: constructor, destructor, `Alloc`, `AllocOnStream`, `Free`, `Reserve`, `Extend`, `FindChunkPtr`, `SplitChunk`, `Merge`, `FreeAndMaybeCoalesce`, `Coalesce`, `ResetChunksUsingStream`, `DumpMemoryLog`, `GetStats` | **Namespace:** Wrap in `onnxruntime::cuda_plugin`. **Include:** `#include "cuda_arena.h"`. **Macros:** Same as header. No other changes needed — the implementation is allocator-agnostic (delegates to `device_allocator_->Alloc/Free`). |

**Not copied** — `ep_allocator.h`. The CUDA plugin already has `cuda_allocator_plugin.h` with `CudaAllocatorBase`, `CudaDeviceAllocator`, `CudaPinnedAllocator`. We add the missing types (`BaseAllocator`, `AllocatorStats`, `AllocatorUniquePtr`) to this existing file (see 5.2).

**CMake:** No changes needed. The plugin CMake uses `file(GLOB_RECURSE ... "core/providers/cuda/*.cc")` which automatically picks up new `.cc` files in the `plugin/` directory.

### 5.2 CUDA Plugin Changes

| File | Change |
|------|--------|
| `plugin/cuda_arena.h` | **New file.** Copied from `ep_arena.h` with namespace/include adaptations per 5.1. Contains `ArenaExtendStrategy`, `ArenaConfig`, `ArenaImpl`, `ArenaAllocator`. |
| `plugin/cuda_arena.cc` | **New file.** Copied from `ep_arena.cc` with namespace/include adaptations per 5.1. |
| `plugin/cuda_allocator_plugin.h` | **(a)** Add `BaseAllocator` struct (inherits `OrtAllocator`, adds virtual dtor) — or make `CudaAllocatorBase` inherit from a new `BaseAllocator`. **(b)** Add `AllocatorStats` struct (POD with `ToKeyValuePairs` helper, copied from `ep_allocator.h`). **(c)** Add `using AllocatorUniquePtr = std::unique_ptr<BaseAllocator>;` typedef. **(d)** Add arena-support macros: `EP_ENFORCE` (ostringstream + throw), `LOG` (delegates to `OrtApi::Logger_LogMessage`), `RETURN_ERROR` (creates OrtStatus). These can go in `cuda_plugin_utils.h` instead if preferred. |
| `plugin/cuda_ep_factory.h` | Extend `DeviceCacheEntry` with per-device arena members: `std::mutex arena_mutex; std::unique_ptr<ArenaAllocator> device_arena; std::unique_ptr<ArenaAllocator> pinned_arena; int num_device_arena_users = 0; int num_pinned_arena_users = 0; bool device_arena_using_defaults = true;`. Add `#include "cuda_arena.h"`. Add helper `ArenaAllocator* GetDeviceArenaForDevice(int device_id)` for stream integration. |
| `plugin/cuda_ep_factory.cc` | Rewrite `CreateAllocatorImpl`: extract `device_id` from `OrtMemoryInfo`, find `DeviceCacheEntry`, create/return shared `ArenaAllocator` wrapping `CudaDeviceAllocator` or `CudaPinnedAllocator` per device (Section 3.1 pseudocode). Rewrite `ReleaseAllocatorImpl`: detect arena allocator (compare pointer against `DeviceCacheEntry` arenas), decrement ref count, destroy if zero; fall back to `delete` for non-arena allocators. |
| `plugin/cuda_stream_plugin.cc` | Update `CudaSyncStream::OnSessionRunEndImpl`: after stream synchronization and deferred buffer cleanup, call `factory.GetDeviceArenaForDevice(stream->device_id_)->ResetChunksUsingStream(this_ptr)` to release chunk-to-stream assignments (Section 3.4). |

### 5.3 ORT Core Changes (Minimal)

| File | Change |
|------|--------|
| `environment.cc` | `RegisterExecutionProviderLibrary`: construct prefix `"ep_factory." + GetLowercaseString(factory->GetName()) + "."`, extract matching `arena.*` keys from `config_entries_`, strip prefix, build `OrtKeyValuePairs` with bare `arena.*` keys, pass as `allocator_options` to `CreateSharedAllocatorImpl` instead of `nullptr` (see Section 3.6 for casing convention). |

This is the only ORT core change needed — it enables env-level arena config to reach the plugin factory. The arena wrapping itself happens entirely inside the plugin.

---

## 6. Implementation Plan

### Phase 1: Arena in CUDA Plugin

1. **Add support types to `cuda_allocator_plugin.h`:** Add `BaseAllocator` (OrtAllocator + virtual dtor), `AllocatorStats` (POD), `AllocatorUniquePtr` typedef. Make `CudaAllocatorBase` inherit from `BaseAllocator` instead of `OrtAllocator` directly.
2. **Add arena macros to `cuda_plugin_utils.h`:** Add `EP_ENFORCE` (ostringstream throw), `LOG` (delegates to `OrtApi::Logger_LogMessage`), `RETURN_ERROR` (creates OrtStatus). These are needed by the arena code copied from the example plugin.
3. **Copy `ep_arena.h` → `plugin/cuda_arena.h`:** Wrap in `onnxruntime::cuda_plugin` namespace. Replace includes with `cuda_allocator_plugin.h` and `cuda_plugin_utils.h`. No other changes needed — the arena is allocator-agnostic.
4. **Copy `ep_arena.cc` → `plugin/cuda_arena.cc`:** Wrap in `onnxruntime::cuda_plugin` namespace. Replace includes. No other changes needed.
5. **Extend `DeviceCacheEntry` in `cuda_ep_factory.h`:** Add per-device arena members (`device_arena`, `pinned_arena`, ref counts, mutex) as described in Section 3.3. Add `#include "cuda_arena.h"`. Add `GetDeviceArenaForDevice(int device_id)` accessor.
6. **Rewrite `CreateAllocatorImpl` in `cuda_ep_factory.cc`:** Look up `DeviceCacheEntry` by `device_id`, create shared `ArenaAllocator` wrapping `CudaDeviceAllocator`/`CudaPinnedAllocator` on first call per device, return same pointer on subsequent calls (Section 3.1 pseudocode).
7. **Rewrite `ReleaseAllocatorImpl` in `cuda_ep_factory.cc`:** Detect arena allocator (compare pointer against device cache entries), decrement ref count, destroy if zero. Fall back to `delete` for non-arena types.
8. **Update `OnSessionRunEndImpl` in `cuda_stream_plugin.cc`:** After existing stream sync and deferred buffer cleanup, call `arena->ResetChunksUsingStream(this_ptr)` for the device's arena (Section 3.4).
9. **No CMake changes needed:** The glob picks up new `.cc` files in `plugin/` automatically.
10. **Update `RegisterExecutionProviderLibrary` in `environment.cc`:** Construct prefix via `GetLowercaseString(factory->GetName())`, extract `ep_factory.<lowercase_ep_name>.arena.*` keys from env config, pass as `allocator_options` to `CreateSharedAllocatorImpl` (see Section 3.6).

### Phase 2: CudaMempoolArena Migration

1. Add conditional logger abstraction to `cuda_mempool_arena.h/.cc`
2. Create `CudaMempoolOrtAllocator` wrapper following `ArenaAllocator` pattern
3. Add mempool arena mode selection in `CreateAllocatorImpl` based on `arena.use_cuda_mempool` option
4. Remove `cuda_mempool_arena.cc` from plugin CMake exclusion list

### Phase 3: Validation

1. Verify default arena gives same allocation behavior as in-tree EP
2. Test mempool mode with `arena.use_cuda_mempool=1`
3. Test env-level arena config via `CreateEnvWithOptions`
4. Test shared allocator replacement via `OrtApi::CreateSharedAllocator`
5. Benchmark allocation performance vs. in-tree EP
6. Verify `use_env_allocators=1` works correctly (shared arena replaces per-session)

---

## 7. Open Questions

1. **Arena code sharing vs. copying.** Should the CUDA plugin copy `ep_arena.h/cc` verbatim, or should there be a shared location for the arena code that multiple plugin EPs can use? Copying is simpler and avoids coupling, but risks divergence if bugs are found. A shared `plugin_arena/` directory under `onnxruntime/test/autoep/library/` (or a new location) could be consumed by multiple plugin EPs.
