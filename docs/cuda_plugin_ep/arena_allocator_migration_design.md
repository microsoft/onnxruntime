# CUDA Plugin EP — Arena Allocator Integration Design

## 1. Problem Statement

The CUDA plugin EP currently uses raw `cudaMalloc`/`cudaFree` through `CudaDeviceAllocator` (an `OrtAllocator*` wrapper). The in-tree (bridge-based) CUDA EP wraps its allocators in arenas by default:

| Allocator | In-Tree CUDA EP | Plugin CUDA EP (today) |
|-----------|----------------|----------------------|
| GPU device | `CUDAAllocator` → `StreamAwareBFCArena` | `CudaDeviceAllocator` → raw `cudaMalloc`/`cudaFree` |
| GPU device (mempool) | `CudaMempoolArena` (native CUDA mempool) | Not available |
| Pinned (host) | `CUDAPinnedAllocator` → `BFCArena` | `CudaPinnedAllocator` → raw `cudaHostAlloc`/`cudaFreeHost` |

This gap means the plugin EP has significantly worse allocation performance for typical workloads. Two arena types must be integrated:

1. **`CudaMempoolArena`** — native CUDA mempool (`cudaMallocFromPoolAsync`/`cudaFreeAsync`). Self-contained, CUDA-only dependencies.
2. **`BFCArena`** — ORT's bin-based arena allocator. Lives in `onnxruntime/core/framework/`, not available in the plugin binary.

---

## 2. Three Arena Modes

The CUDA EP has three mutually exclusive arena modes for the **device** allocator:

| Mode | Trigger | Arena Type | BFCArena Wrapping? |
|------|---------|-----------|-------------------|
| **Default** | Always (unless mempool configured) | `StreamAwareBFCArena` wrapping `CUDAAllocator` | Yes — with default `OrtArenaCfg{0, -1, -1, -1, -1, -1L}` |
| **CUDA Mempool** | `OrtArenaCfg::use_cuda_mempool == 1` | `CudaMempoolArena` (native CUDA pool) | No — is its own arena |
| **No Arena** | `DisableCpuMemArena()` API | N/A | **CPU-only** — CUDA device allocator is unaffected |

The **pinned allocator** is always wrapped in `BFCArena` (non-stream-aware) in the in-tree EP.

The `DisableCpuMemArena()` public API sets `SessionOptions::enable_cpu_mem_arena = false` but only affects the CPU EP. The CUDA EP always uses arena: *"CUDA malloc/free is expensive so always use an arena"* (comment in `cuda_execution_provider.cc`).

---

## 3. Part A — Migrating `CudaMempoolArena` to the Plugin

### 3.1 Current Dependencies

`CudaMempoolArena` in `cuda_mempool_arena.h/.cc` has these dependencies:

| Dependency | Plugin-Safe? | Notes |
|-----------|-------------|-------|
| `<cuda_runtime_api.h>` | ✅ | CUDA SDK — always available |
| `core/common/common.h` | ✅ | `ORT_THROW`, `ORT_ENFORCE` — no framework deps |
| `core/common/inlined_containers.h` | ✅ | STL-based containers, no framework deps |
| `core/providers/cuda/cuda_stream_handle.h` | ✅ | But only for `Stream::GetHandle()` → `cudaStream_t` |
| `core/providers/shared_library/provider_api.h` | ⚠️ | **No-op in plugin build** (`BUILD_CUDA_EP_AS_PLUGIN`) |
| `core/providers/cuda/shared_inc/cuda_call.h` | ✅ | CUDA error-handling macros |
| `IArena` base class | ⚠️ | Defined in framework `allocator.h` — available in plugin (not behind `SHARED_PROVIDER`) |
| `OrtMemoryInfo` | ✅ | Public framework struct |
| `AllocatorStats` | ✅ | Plain POD struct in public header |
| `logging::Logger*` | ❌ | **Primary blocker** — `provider_api.h` forward-declares `Logger` as struct; `LoggingManager::DefaultLogger()` not available in plugin |
| `Stream*` | ✅ | Only uses `stream->GetHandle()` → `void*` → `cudaStream_t` |

### 3.2 The Logger Problem

`CudaMempoolArena` uses `LOGS(*logger_, ...)` in 6 locations:
- Constructor (INFO): pool creation message
- `Alloc()` (VERBOSE): per-allocation trace
- `AllocOnStream()` (VERBOSE): per-allocation trace
- `Free()` (WARNING): unknown pointer warning
- `Shrink()` (INFO): pool trim stats

The plugin has its own logger type: `OrtLogger` (from the EP C API). The factory stores `const OrtLogger& default_logger_`.

### 3.3 Proposed Changes

**Approach: Make `CudaMempoolArena` compilable in both in-tree and plugin builds.**

The class itself is almost entirely CUDA SDK code. Only the logging needs adaptation.

#### Option 1: Conditional Logger (Recommended)

Replace `const logging::Logger* logger_` with a thin logging abstraction that works in both builds:

```cpp
// In cuda_mempool_arena.h:
#ifdef BUILD_CUDA_EP_AS_PLUGIN
  // Plugin build: use OrtLogger-based logging
  #include "cuda_plugin_utils.h"  // provides LOG_INFO, LOG_VERBOSE, LOG_WARNING macros
  // No logger_ member needed — macros use the factory/EP logger directly
  // OR: store an OrtLogger* and define thin macros
#else
  // In-tree build: use existing logging::Logger
  const logging::Logger* logger_;
#endif
```

**Concrete steps:**
1. Replace `#include "core/providers/shared_library/provider_api.h"` with a conditional include for the logger type.
2. Make the `logger_` member type conditional: `const logging::Logger*` in-tree, `const OrtLogger*` in plugin.
3. Define a `MEMPOOL_LOG(level, msg)` macro that dispatches to either `LOGS()` or OrtLogger-based logging.
4. Add `cuda_mempool_arena.cc` to the plugin CMake source list (remove from exclusion list in `onnxruntime_providers_cuda_plugin.cmake`).

#### Option 2: Template on Logger Type

Make the constructor accept a callable/functor for logging, avoiding compile-time branching.

#### Option 3: Strip Logging Entirely in Plugin Build

Wrap all `LOGS()` calls in `#ifndef BUILD_CUDA_EP_AS_PLUGIN` guards. Simplest, but loses diagnostic capability.

**Recommendation:** Option 1. The logging is genuinely useful for diagnosing mempool behavior. The plugin already has `OrtLogger` available; we just need a thin macro bridge.

### 3.4 OrtAllocator Wrapper

The plugin factory's `CreateAllocatorImpl` returns `OrtAllocator*`. `CudaMempoolArena` is an `IArena`. A thin wrapper is needed:

```cpp
class CudaMempoolOrtAllocator : public OrtAllocator {
  std::unique_ptr<CudaMempoolArena> arena_;
  const OrtMemoryInfo* memory_info_;

  // OrtAllocator callbacks:
  static void* AllocImpl(OrtAllocator* this_, size_t size);
  static void FreeImpl(OrtAllocator* this_, void* p);
  static void* ReserveImpl(OrtAllocator* this_, size_t size);
  static void* AllocOnStreamImpl(OrtAllocator* this_, size_t size, OrtSyncStream* stream);
  static const OrtMemoryInfo* InfoImpl(const OrtAllocator* this_);
};
```

The `AllocOnStream` callback must resolve `OrtSyncStream*` → `cudaStream_t`. The `OrtEpApi::SyncStream_GetHandle()` function provides this.

**Important:** The `OrtMemoryInfo::alloc_type` must be `OrtDeviceAllocator`, not `OrtArenaAllocator`. Both `CreatePreferredAllocators` and `CreateSharedAllocatorImpl` reject `OrtArenaAllocator` from plugin factories.

### 3.5 Arena Config Parsing

The plugin factory's `CreateAllocatorImpl` receives `const OrtKeyValuePairs* allocator_options` (currently ignored). The relevant keys:
- `arena.use_cuda_mempool` — `"1"` to enable
- `arena.cuda_mempool_release_threshold` — bytes; `0` disables threshold
- `arena.cuda_mempool_bytes_to_keep_on_shrink` — bytes retained after `Shrink()`

These can be parsed via `OrtArenaCfg::FromKeyValuePairs()` or directly from the key-value pairs using the `OrtApi`.

**Problem:** `CreateAllocatorImpl` currently receives `nullptr` for `allocator_options` from both callers (see Part B). The plugin can work around this by parsing arena config from session/provider options in `CudaEpFactory` and storing them for later use by `CreateAllocatorImpl`.

### 3.6 Summary of Changes for CudaMempoolArena Migration

| File | Change |
|------|--------|
| `cuda_mempool_arena.h` | Conditional logger type; add `#ifdef BUILD_CUDA_EP_AS_PLUGIN` for logger include |
| `cuda_mempool_arena.cc` | Replace `LOGS()` with build-conditional macro |
| `cmake/onnxruntime_providers_cuda_plugin.cmake` | Remove `cuda_mempool_arena.cc` from exclusion list |
| `plugin/cuda_allocator_plugin.h` | Add `CudaMempoolOrtAllocator` wrapper class |
| `plugin/cuda_allocator_plugin.cc` | Implement wrapper callbacks |
| `plugin/cuda_ep_factory.cc` | Parse mempool options; create `CudaMempoolOrtAllocator` in `CreateAllocatorImpl` when configured |
| `plugin/cuda_ep_factory.cc` | Handle `CudaMempoolOrtAllocator` in `ReleaseAllocatorImpl` |

---

## 4. Part B — Integrating BFCArena for the Plugin EP

`BFCArena` lives in `onnxruntime/core/framework/bfc_arena.h/.cc` and is part of the ORT core framework. Duplicating it into the plugin would be a significant code duplication burden. Instead, the framework should wrap the plugin's raw allocator in BFCArena on the ORT core side.

### 4.1 Current Allocator Lifecycle

There are two paths through which plugin allocators are created and used:

**Path 1: Shared allocators (environment level)**
```
RegisterExecutionProviderLibrary()
  → CreateSharedAllocatorImpl(ep_device, memory_info, OrtDeviceAllocator, nullptr, ...)
    → ep_factory->CreateAllocator(factory, &mem_info, /*options=*/ nullptr, &alloc)
    → IAllocatorImplWrappingOrtAllocator(alloc)
    → shared_allocators_.push_back(wrapped)

Session::Initialize() [if use_env_allocators="1"]
  → UpdateAllocatorsWithEnvAllocators(env.GetRegisteredSharedAllocators())
    → replaces per-session allocators by device key
```

**Path 2: Per-session allocators**
```
SessionState constructor
  → ep->CreatePreferredAllocators()
    → PluginExecutionProvider::CreatePreferredAllocators()
      → OrtEp::CreateAllocator(ep, &mem_info, &alloc)   [if set]
        OR ep_factory.CreateAllocator(&factory, &mem_info, /*options=*/ nullptr, &alloc)
      → IAllocatorImplWrappingOrtAllocator(alloc)
    → session allocator maps
```

**Key gap:** Neither path passes arena configuration (`allocator_options` is always `nullptr`), and neither path wraps the result in BFCArena.

### 4.2 Three Options for BFCArena Integration

#### Option A: Wrap at All Callers

**Where:** Every ORT core call site that creates allocators from plugin factories wraps the result in BFCArena.

**Changes needed:**
- `SessionState` constructor — after `ep->CreatePreferredAllocators()`, wrap each returned allocator in BFCArena via `CreateAllocator(AllocatorCreationInfo{...})`
- `Environment::CreateSharedAllocatorImpl()` — after creating `IAllocatorImplWrappingOrtAllocator`, wrap in BFCArena with default arena config

**Arena config source:** Must be parsed from session options or hardcoded defaults at each call site independently.

| Pros | Cons |
|------|------|
| No plugin code changes | Multiple ORT core sites to modify — fragile, hard to maintain |
| Reuses existing `BFCArena` and `CreateAllocator()` utility | Arena config plumbing is ad-hoc per call site |
| | `CreateSharedAllocatorImpl` receives `nullptr` for options — requires hardcoded defaults or new plumbing |
| | Must distinguish "plugin EP that wants arena wrapping" from one that doesn't at each site |
| | Every new consumer of plugin allocators must know to wrap — doesn't scale |
| | Risk of inconsistency between the two paths |

#### Option B: Wrap at the Two ORT Core Entry Points

**Where:** BFCArena wrapping is added at the two ORT core entry points that create allocators from plugin factories:

1. `PluginExecutionProvider::CreatePreferredAllocators()` — per-session allocators
2. `Environment::CreateSharedAllocatorImpl()` — shared (environment-level) allocators

`CreateSharedAllocatorImpl` already accepts `const OrtKeyValuePairs* allocator_options` and has full access to the `OrtEpDevice` and `OrtMemoryInfo`. Today the caller (`RegisterExecutionProviderLibrary`) passes `nullptr` for options. The fix is:
1. Pass default arena options from `RegisterExecutionProviderLibrary` instead of `nullptr`
2. Inside `CreateSharedAllocatorImpl`, after creating `IAllocatorImplWrappingOrtAllocator` (line 864), conditionally wrap in BFCArena using `CreateAllocator(AllocatorCreationInfo{...})` before pushing to `shared_allocators_`

**Changes needed:**
- `PluginExecutionProvider::CreatePreferredAllocators()` — after creating the `IAllocator` wrapper, conditionally wrap in BFCArena using `CreateAllocator(AllocatorCreationInfo{...})`
- `Environment::CreateSharedAllocatorImpl()` — parse `allocator_options` for arena config, wrap returned allocator in BFCArena when appropriate
- `Environment::RegisterExecutionProviderLibrary()` — construct and pass default arena options (`OrtArenaCfg{0, -1, -1, -1, -1, -1L}`) instead of `nullptr`
- Arena config stored on `PluginExecutionProvider` for the per-session path (populated during EP creation from session/provider options)

| Pros | Cons |
|------|------|
| Covers both per-session and shared allocator paths | Two ORT core sites to modify |
| Clean — wrapping happens at the adapter/infrastructure boundary | Arena wrapping decision logic must be present in both sites (can share a helper) |
| Arena config naturally available from EP's parsed options (per-session) and from `allocator_options` param (shared) | |
| Reuses existing `CreateAllocator(AllocatorCreationInfo)` utility | |
| `use_env_allocators` works correctly — shared allocators are also arena-wrapped | |
| **Naturally gated by EP type** — arena options (`arena.extend_strategy`, `arena.max_mem`, etc.) are only recognized by CUDA EP. Non-CUDA plugin EPs don't pass arena keys, so no wrapping occurs. The presence of arena keys in `allocator_options` is the signal — no device-type checks needed in ORT core. | |
| **No new public API surface** — uses existing `allocator_options` parameter. It is always easier to add a new API later (Option C) than to remove a wrong one. Option B can be promoted to Option C if the convention proves insufficient. | |

#### Option C: Declarative Arena Request via `OrtEpDevice` API

**Where:** The plugin declares at device-registration time (in `GetSupportedDevices`) that allocators for a given memory type should be BFCArena-wrapped by ORT, including the arena config. ORT core reads this declaration and wraps after receiving the raw `OrtAllocator*`.

**API changes:**
```c
// New OrtEpApi function:
ORT_API2_STATUS(EpDevice_RequestArenaWrapping,
                _In_ OrtEpDevice* ep_device,
                _In_ const OrtMemoryInfo* allocator_memory_info,
                _In_opt_ const OrtKeyValuePairs* arena_config);
```

**Internal changes:**
- `OrtEpDevice` gains a `std::vector<ArenaRequest>` field storing per-memory-info arena configuration
- `Environment::CreateSharedAllocatorImpl()` checks `OrtEpDevice` for arena request → wraps with the declared config (or defaults)
- `PluginExecutionProvider::CreatePreferredAllocators()` does the same check and wrap

**Plugin-side changes:**
- `CudaEpFactory::GetSupportedDevicesImpl` calls `EpDevice_RequestArenaWrapping` for device memory (with default BFCArena config) and for pinned memory

| Pros | Cons |
|------|------|
| **Covers both paths uniformly** — same `OrtEpDevice` declaration drives wrapping in both shared and per-session paths | New public API surface on `OrtEpApi` — requires API review |
| **Config plumbing solved cleanly** — plugin declares arena needs upfront with full config | Medium effort: new API + two wrapping callsites + plugin callsite |
| **Fully opt-in** — zero behavior change for existing EPs or the bridge-based CUDA EP | |
| **Preserves environment shared allocators** — shared allocators are arena-wrapped → `use_env_allocators` works correctly | |
| **Extensible** — any future plugin EP can request arena wrapping the same way | |
| Reuses existing `CreateAllocator(AllocatorCreationInfo)` — no BFCArena code duplication | |
| `OrtArenaAllocator` rejection stays unchanged — raw allocator from factory is still `OrtDeviceAllocator` | |
| Plugin controls arena mode: BFCArena, CudaMempoolArena, or no arena per memory type | |
| Natural API idiom — mirrors existing `EpDevice_AddAllocatorInfo` | |

### 4.3 Allocator Config Flow — In-Tree vs. Plugin

The in-tree CUDA EP receives arena config through `OrtCUDAProviderOptionsV2`, which contains `OrtArenaCfg* default_memory_arena_cfg`. This is stored in `CUDAExecutionProviderInfo` and cached on the EP instance as `info_`. Both allocator creation paths read from this cached config:

- **Factory path (shared allocators):** `ProviderInfo_CUDA_Impl::CreateCudaAllocator()` accepts `OrtArenaCfg*` directly.
- **Per-session path:** `CUDAExecutionProvider::CreatePreferredAllocators()` reads `info_.default_memory_arena_cfg` into `CUDAAllocatorParams.arena_cfg` and passes it to `CreateCudaAllocator()`.

For the plugin CUDA EP, configuration arrives through `session_options` as key-value pairs with an EP-specific prefix (e.g., `"ep.cudapluginexecutionprovider.prefer_nhwc"`). The factory's `CreateEpImpl` extracts these via `GetSessionConfigEntry(session_options, prefixed_key, ...)`. This is the existing config pipeline for all plugin EP settings.

**Per-session allocator config flow (Path 2 — `CreatePreferredAllocators`):**

`PluginExecutionProvider::CreatePreferredAllocators()` currently passes `nullptr` for allocator options when calling `ep_factory_.CreateAllocator()`. The fix:

1. `PluginExecutionProvider` already receives `session_options` at construction time.
2. At `CreatePreferredAllocators()` time, extract arena keys from `session_options` using the EP prefix, build an `OrtKeyValuePairs` with bare `"arena.*"` keys, and pass it to `ep_factory_.CreateAllocator()`.
3. The same `OrtKeyValuePairs` is used by ORT core to decide BFCArena wrapping (under Option B).

**Shared allocator config flow (Path 1 — `CreateSharedAllocatorImpl`):**

`RegisterExecutionProviderLibrary()` is called at environment level — no session exists yet, so no session-specific arena config is available. The fix is to pass default arena options (`OrtArenaCfg{0, -1, -1, -1, -1, -1L}` expressed as `OrtKeyValuePairs` with bare `"arena.*"` keys) to `CreateSharedAllocatorImpl()`. The function already accepts `const OrtKeyValuePairs* allocator_options` — it just needs the caller to provide defaults.

### 4.4 Key Name Prefix Mismatch

**Issue:** `OrtArenaCfg::FromKeyValuePairs()` expects bare key names (e.g., `"arena.extend_strategy"`, `"arena.max_mem"`). However, session options store EP config with an EP-specific prefix:

```
Session options key:  "ep.cudapluginexecutionprovider.arena.extend_strategy"
OrtArenaCfg expects:  "arena.extend_strategy"
```

`FromKeyValuePairs()` uses exact key lookup (`kvps_entries.find(ConfigKeyNames::ArenaExtendStrategy)`) — prefixed keys will not match.

**Resolution:** The ORT core code that builds `OrtKeyValuePairs` for `CreateAllocator` must strip the EP prefix. Since both `CreatePreferredAllocators` and `CreateSharedAllocatorImpl` are ORT core code, they control the KVP construction:

- **Per-session path:** Read prefixed keys from `session_options` via `GetSessionConfigEntry()`, write bare `"arena.*"` keys into the `OrtKeyValuePairs` passed to `CreateAllocator`.
- **Shared path:** `RegisterExecutionProviderLibrary` constructs KVPs from scratch with bare keys and default values — no prefix issue.

The plugin factory's `CreateAllocatorImpl` then calls `OrtArenaCfg::FromKeyValuePairs()` on the received KVPs and gets correct parsing.

### 4.5 Arena-Already-Handled Signal Problem

Under Option B, ORT core wraps raw allocators from the factory in BFCArena. But when the factory returns a self-contained arena (CudaMempoolArena), ORT must **not** double-wrap it.

**The easy case — default options:** When default arena options are passed (no `use_cuda_mempool` key or `use_cuda_mempool=-1`), the factory returns a raw `CudaDeviceAllocator` and ORT core wraps it in BFCArena. This is straightforward.

**The hard case — CudaMempoolArena:** When `use_cuda_mempool=1`, the factory returns a `CudaMempoolOrtAllocator` that is already an arena. ORT core must know not to wrap it. But both the raw allocator and the mempool allocator return `OrtDeviceAllocator` type — the `OrtArenaAllocator` type is currently rejected by both `CreateSharedAllocatorImpl` and `CreatePreferredAllocators`.

ORT core could read `use_cuda_mempool` from the same `OrtKeyValuePairs` it passes to the factory and skip BFCArena wrapping. However, `use_cuda_mempool` is a CUDA-specific concept — having ORT core interpret it undermines the EP abstraction.

**Considered signals:**

| Signal Mechanism | Pros | Cons |
|---|---|---|
| **(a) ORT reads `use_cuda_mempool` from options** | Simple, no API changes | ORT core has CUDA-specific knowledge |
| **(b) Factory omits arena keys when mempool active** — absence = no BFCArena wrapping | Clean "keys-as-signal" convention | Doesn't generalize; ORT must still pass default options for the common case |
| **(c) Allow `OrtArenaAllocator` type from plugin factories** | Clean, explicit signal — ORT skips wrapping when it sees this type | Reverses current restriction; changes API contract |
| **(d) Check the returned allocator's `OrtMemoryInfo` name** | No API changes; uses existing data | Convention-based; fragile if names change |

**Decision: Option (d) — check the allocator's `OrtMemoryInfo` name.**

ORT core compares the returned allocator's `OrtMemoryInfo` name against the name from the `OrtEpDevice`'s `device_memory_info` (or `host_accessible_memory_info`). If the names match, the allocator is a raw device allocator and ORT wraps it in BFCArena. If the name differs, the factory returned a specialized allocator (e.g., `CudaMempoolArena` with name `"CUDAMemPoolArena"` instead of `"Cuda"`) and ORT skips wrapping.

This approach:
- Requires **no API changes** — uses existing `OrtMemoryInfo` data already available to both the factory and ORT core.
- Is **EP-agnostic** — any plugin EP can use a distinct allocator name to signal "I handle my own arena."
- The in-tree CUDA EP already follows this pattern: `CudaMempoolArena` uses `"CUDAMemPoolArena"` while the raw allocator uses `"Cuda"`.
- The `OrtEpDevice` already declares the expected memory info names at device registration time, so ORT core has the baseline to compare against.

### 4.6 Default Arena Options Fix (Applies to All Options)

Today, `Environment::RegisterExecutionProviderLibrary()` calls `CreateSharedAllocatorImpl()` with `nullptr` for `allocator_options`. This means shared allocators for plugin EPs are never arena-wrapped, even when they should be.

**Required fix (independent of which option is chosen for BFCArena integration):**

`RegisterExecutionProviderLibrary` must construct and pass default arena options (`OrtArenaCfg{0, -1, -1, -1, -1, -1L}` as bare-key `OrtKeyValuePairs`) to `CreateSharedAllocatorImpl()` instead of `nullptr`.

For **Option A**: Each caller site constructs options and does its own wrapping.

For **Option B**: `CreateSharedAllocatorImpl` uses the options it already receives to decide on wrapping. `RegisterExecutionProviderLibrary` passes defaults. `CreatePreferredAllocators` extracts arena keys from session_options.

For **Option C**: The `OrtEpDevice` arena declaration is available to `CreateSharedAllocatorImpl` — default arena config is carried by the declaration, so the fix is automatic.

### 4.7 Comparison Matrix

| Criterion | A (Callers wrap) | B (Adapter wraps) | C (Declarative API) |
|-----------|:-:|:-:|:-:|
| Covers per-session allocators | ✅ | ✅ | ✅ |
| Covers shared (environment) allocators | ✅ (with fix) | ✅ (via `allocator_options` param) | ✅ (built-in) |
| `use_env_allocators` works correctly | ⚠️ fragile | ✅ (shared allocators arena-wrapped) | ✅ |
| Arena config plumbing | Ad-hoc per site | `allocator_options` (shared) + EP-stored (per-session) | Declared upfront per device |
| ORT core change surface | Multiple files | 2 files (`CreatePreferredAllocators` + `CreateSharedAllocatorImpl`) + caller fix | 2 files + new API |
| Plugin code changes | None | None | Small (1 API call) |
| Backward compatible | ⚠️ all plugin EPs affected | ✅ gated by arena options — only EPs that pass arena keys get wrapping | ✅ fully opt-in |
| Future EP extensibility | Poor | Good — any EP can pass arena keys | Good |
| Supports both BFC and CudaMempool modes | Must distinguish externally | Must distinguish externally | Plugin declares what it wants |
| Stream-aware BFCArena support | Must plumb stream-awareness flag | Must plumb stream-awareness flag | Config key (`arena.stream_aware`) |
| Effort | Medium | Low-Medium | Medium |

---

## 5. Recommended Plan

### Phase 1: Migrate `CudaMempoolArena` to Plugin Build

1. Add conditional logger abstraction to `cuda_mempool_arena.h/.cc` (Option 1 from Section 3.3)
2. Create `CudaMempoolOrtAllocator` wrapper in `plugin/cuda_allocator_plugin.h/.cc`
3. Update `CudaEpFactory::CreateAllocatorImpl` to create mempool allocator when configured
4. Parse mempool options from provider/session options in `CudaEpFactory`
5. Remove `cuda_mempool_arena.cc` from plugin CMake exclusion list
6. Test with `arena.use_cuda_mempool=1` provider option

### Phase 2: BFCArena Integration (Option B Recommended)

Option B is recommended as the starting point because it requires no new public API surface, uses existing `allocator_options` plumbing, covers both shared and per-session allocator paths, and is naturally gated by arena config keys (only EPs that pass them get wrapping). Option C (declarative API) can be added later if a more formal mechanism proves necessary — it is always easier to add a new API than to remove a wrong one.

1. Update `Environment::RegisterExecutionProviderLibrary()` to construct and pass default arena options (`OrtArenaCfg{0, -1, -1, -1, -1, -1L}`) to `CreateSharedAllocatorImpl()` instead of `nullptr`
2. Update `Environment::CreateSharedAllocatorImpl()` to parse `allocator_options` for arena config keys and wrap the returned `IAllocator` in BFCArena via `CreateAllocator(AllocatorCreationInfo{...})` when arena keys are present
3. Update `PluginExecutionProvider::CreatePreferredAllocators()` to wrap returned allocators in BFCArena using EP-stored arena config (populated during EP creation from session/provider options)
4. Extract a shared helper for the arena-wrapping logic so both sites stay consistent
5. Test both shared allocator path and per-session path; verify `use_env_allocators` works correctly

### Phase 3: Parity Validation

1. Verify arena mode selection matches in-tree EP: default BFCArena, CUDA mempool if configured
2. Benchmark allocation performance vs. in-tree EP
3. Verify `DisableCpuMemArena()` does not affect CUDA plugin allocators (it shouldn't)
4. Test shared allocator replacement (environment allocators replacing per-session)

---

## 6. Open Questions

1. **Stream-aware BFCArena for shared allocators.** The per-session GPU allocator in the in-tree EP uses `StreamAwareBFCArena`. Should `CreateSharedAllocatorImpl` also create stream-aware arenas when wrapping? The in-tree EP only creates arenas in `CreatePreferredAllocators()` (per-session), so there is no precedent for shared stream-aware arenas. A `stream_aware` key in `allocator_options` could control this — decide whether to add it now or default to non-stream-aware for shared allocators.

2. **Arena wrapping for shared allocators at `RegisterExecutionProviderLibrary` time.** Wrapping shared allocators in BFCArena at EP library registration ensures that when `use_env_allocators=1` replaces per-session allocators with shared ones, the shared allocators already have arena behavior — otherwise the session loses arena wrapping entirely. However, BFCArena may pre-allocate significant GPU memory at registration time, before any session exists. This is a trade-off:
   - **If we wrap:** Shared allocators are arena-backed. `use_env_allocators` works correctly. But memory is committed early (at `RegisterExecutionProviderLibrary` time), potentially wasting resources if no session is ever created, or if the arena config (e.g., `max_mem`) is too aggressive for a shared context.
   - **If we don't wrap:** Shared allocators remain raw. `use_env_allocators` replaces arena-wrapped per-session allocators with raw shared ones, losing arena performance. Users who set `use_env_allocators=1` get worse allocation behavior than without it.
   - **Pinned allocator:** The in-tree EP wraps pinned in `BFCArena` (non-stream-aware) using the same arena options as the device allocator — defaults are `OrtArenaCfg{0, -1, -1, -1, -1, -1L}`. The plugin should use the same arena options for pinned allocators to maintain parity.
   - **Needs validation:** Confirm that default arena options (`OrtArenaCfg{0, -1, -1, -1, -1, -1L}`) do not cause excessive upfront memory allocation in BFCArena. The `max_mem=0` default means "ORT chooses" — verify what BFCArena actually allocates at construction time vs. on first `Alloc()` call.

3. **Helper function for arena wrapping.** Both `CreateSharedAllocatorImpl` and `CreatePreferredAllocators` need the same wrapping logic: parse `OrtArenaCfg` from options, call `CreateAllocator(AllocatorCreationInfo{...})`. Extract a shared helper (e.g., `MaybeWrapInArena(AllocatorPtr, OrtArenaCfg)`) to keep both sites consistent and avoid logic duplication.

4. **Default arena config values.** The in-tree EP uses `OrtArenaCfg{0, -1, -1, -1, -1, -1L}` as defaults for GPU and pinned. Confirm these defaults are appropriate for the plugin path, or whether any should differ (e.g., different `max_mem` for multi-session shared allocators).
