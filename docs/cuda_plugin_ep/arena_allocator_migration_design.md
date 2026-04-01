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

## 3. Part A — Integrating BFCArena for the Plugin EP

`BFCArena` lives in `onnxruntime/core/framework/bfc_arena.h/.cc` and is part of the ORT core framework. Duplicating it into the plugin would be a significant code duplication burden. Instead, the framework should wrap the plugin's raw allocator in BFCArena on the ORT core side.

### 3.1 Current Allocator Lifecycle

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

### 3.2 Two Options for BFCArena Integration

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
| **No new public API surface** — uses existing `allocator_options` parameter and the existing `CreateEnvWithOptions` API with `ep_factory.<registration_name>.*` config entries for environment-level config. | |

### 3.3 Allocator Config Flow — In-Tree vs. Plugin

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

`RegisterExecutionProviderLibrary()` is called at environment level — no session exists yet, so no session-specific arena config is available. Today it passes `nullptr` for `allocator_options` to `CreateSharedAllocatorImpl()`, which means shared allocators for plugin EPs are never arena-wrapped.

**Resolution:** `RegisterExecutionProviderLibrary` must always extract arena options and pass them to `CreateSharedAllocatorImpl()` instead of `nullptr`. The logic is:

1. **Check environment config entries** (`Environment::config_entries_`) for `ep_factory.<registration_name>.arena.*` keys.
2. **If found:** Extract matching arena keys, strip the `ep_factory.<registration_name>.` prefix, and build an `OrtKeyValuePairs` with bare `"arena.*"` keys.
3. **If not found:** Construct default arena options (`OrtArenaCfg{0, -1, -1, -1, -1, -1L}` expressed as bare-key `OrtKeyValuePairs`).
4. **Pass the resulting `OrtKeyValuePairs*`** to `CreateSharedAllocatorImpl()` as `allocator_options`.

This leverages the existing `CreateEnvWithOptions` API — the application provides arena config at environment creation time via `OrtEnvCreationOptions::config_entries`:

```cpp
// Application provides arena config at env creation:
api->AddKeyValuePair(kvps, "ep_factory.cuda.arena.extend_strategy", "1");
api->AddKeyValuePair(kvps, "ep_factory.cuda.arena.max_mem", "0");
api->AddKeyValuePair(kvps, "ep_factory.cuda.arena.use_cuda_mempool", "1");

OrtEnvCreationOptions options{};
options.config_entries = kvps;
// ...
api->CreateEnvWithOptions(&options, &env);
```

For **Option A**: Each caller site constructs options and does its own wrapping.

For **Option B**: `CreateSharedAllocatorImpl` uses the options it already receives to decide on wrapping. `RegisterExecutionProviderLibrary` extracts from env config or uses defaults. `CreatePreferredAllocators` extracts arena keys from session_options (with env config as fallback).

### 3.4 Key Name Prefix Mismatch

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

### 3.5 Arena-Already-Handled Signal Problem

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

### 3.6 Comparison Matrix

| Criterion | A (Callers wrap) | B (Adapter wraps) |
|-----------|:-:|:-:|
| Covers per-session allocators | ✅ | ✅ |
| Covers shared (environment) allocators | ✅ (with fix) | ✅ (via `allocator_options` param) |
| `use_env_allocators` works correctly | ⚠️ fragile | ✅ (shared allocators arena-wrapped) |
| Arena config plumbing | Ad-hoc per site | `allocator_options` (shared) + EP-stored (per-session) |
| ORT core change surface | Multiple files | 2 files (`CreatePreferredAllocators` + `CreateSharedAllocatorImpl`) + caller fix |
| Plugin code changes | None | None |
| Backward compatible | ⚠️ all plugin EPs affected | ✅ gated by arena options — only EPs that pass arena keys get wrapping |
| Future EP extensibility | Poor | Good — any EP can pass arena keys |
| Supports both BFC and CudaMempool modes | Must distinguish externally | Must distinguish externally |
| Stream-aware BFCArena support | Must plumb stream-awareness flag | Must plumb stream-awareness flag |
| Effort | Medium | Low-Medium |

### 3.7 Environment vs. Session Config: Conflict Blindness

ORT has two separate configuration namespaces for EP-specific options:

| | Environment-level | Session-level |
|---|---|---|
| **Prefix** | `ep_factory.<registration_name>.` | `ep.<ep_name>.` |
| **Example** | `ep_factory.cuda.arena.extend_strategy` | `ep.cudapluginexecutionprovider.arena.extend_strategy` |
| **Set via** | `CreateEnvWithOptions` (`OrtEnvCreationOptions.config_entries`) | `SessionOptionsAppendExecutionProvider_V2` |
| **Storage** | `Environment::config_entries_` | `SessionOptions::config_options` |
| **Read by EP** | `GetEnvConfigEntries()` — returns all entries unfiltered | `GetSessionConfigEntry(session_options, key)` |

**The EP is blind to conflicts.** At each point in its lifecycle, the EP only sees one source of config:

- **Shared allocator creation** (`RegisterExecutionProviderLibrary` → `CreateSharedAllocatorImpl`): happens at environment level, before any session exists. Only environment config (`ep_factory.*`) is available. The EP factory's `CreateAllocatorImpl` receives `allocator_options` derived from env config. **No session options exist yet — no conflict possible.**

- **Per-session allocator creation** (`CreatePreferredAllocators`): happens at session creation time. ORT core builds `allocator_options` from session options (stripping the EP prefix). The factory's `CreateAllocatorImpl` receives these options. **The EP does not simultaneously see env config — it only sees whatever ORT core passes.**

- **EP instance creation** (`CreateEpImpl`): receives `session_options` only. The factory *could* also call `GetEnvConfigEntries()`, but the CUDA plugin factory does not do this today.

This means:
1. An EP cannot detect that `ep_factory.cuda.arena.max_mem=1073741824` (env) conflicts with `ep.cudapluginexecutionprovider.arena.max_mem=2147483648` (session).
2. The effective config depends on which path creates the allocator — shared allocators use env config, per-session allocators use session config.
3. The existing API documentation states: *"If an environment-level configuration conflicts with a session-level configuration, then precedence is determined by the execution provider library itself."* In practice, this is aspirational — the EP lacks the mechanism to implement precedence because it sees only one source at each decision point.

**Implication for arena config:** This is acceptable for the arena use case because:
- Shared allocators are environment-scoped and should use environment config.
- Per-session allocators are session-scoped and should use session config.
- The two allocator sets are independent — they don't compete for the same resources at the same time.
- If `use_env_allocators=1` causes shared allocators to replace per-session ones, the shared allocators already carry their env-configured arena behavior.

**Prefix schema mismatch:** Note that the two namespaces use different `<ep_name>` values — environment uses the `registration_name` passed to `RegisterExecutionProviderLibrary` (e.g., `"cuda"`), while session uses the lowercased EP type name (e.g., `"cudapluginexecutionprovider"`). This inconsistency is a guaranteed source of user confusion. However, both prefix schemes are already published and in use — they cannot be changed without breaking backward compatibility. Documentation and examples must clearly explain which prefix to use in which context.

---

## 4. Part B — Migrating `CudaMempoolArena` to the Plugin

### 4.1 Current Dependencies

`CudaMempoolArena` in `cuda_mempool_arena.h/.cc` has these dependencies:

| Dependency | Plugin-Safe? | Notes |
|-----------|-------------|-------|
| `<cuda_runtime_api.h>` | ✅ | CUDA SDK — always available |
| `core/common/common.h` | ✅ | `ORT_THROW`, `ORT_ENFORCE` — no framework deps |
| `core/common/inlined_containers.h` | ✅ | STL-based containers, no framework deps |
| `core/providers/cuda/cuda_stream_handle.h` | ✅ | But only for `Stream::GetHandle()` → `cudaStream_t` |
| `core/providers/shared_library/provider_api.h` | ⚠️ | **No-op in plugin build** (`BUILD_CUDA_EP_AS_PLUGIN`) |
| `core/providers/cuda/shared_inc/cuda_call.h` | ✅ | CUDA error-handling macros |
| `IArena` base class | ✅ | Defined in `include/onnxruntime/core/framework/allocator.h` — public header, no `SHARED_PROVIDER` guard. `onnxruntime_framework` static lib is linked into the plugin, so vtable and `SafeArenaCast()` are available at link time. |
| `OrtMemoryInfo` | ✅ | Public framework struct |
| `AllocatorStats` | ✅ | Plain POD struct in public header |
| `logging::Logger*` | ❌ | **Primary blocker** — `provider_api.h` forward-declares `Logger` as struct; `LoggingManager::DefaultLogger()` not available in plugin |
| `Stream*` | ✅ | Only uses `stream->GetHandle()` → `void*` → `cudaStream_t` |

### 4.2 The Logger Problem

`CudaMempoolArena` uses `LOGS(*logger_, ...)` in 6 locations:
- Constructor (INFO): pool creation message
- `Alloc()` (VERBOSE): per-allocation trace
- `AllocOnStream()` (VERBOSE): per-allocation trace
- `Free()` (WARNING): unknown pointer warning
- `Shrink()` (INFO): pool trim stats

The plugin has its own logger type: `OrtLogger` (from the EP C API). The factory stores `const OrtLogger& default_logger_`.

### 4.3 Proposed Changes

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

### 4.4 OrtAllocator Wrapper

`IArena` (and `IAllocator`) are fully available in the plugin binary — the header is public and `onnxruntime_framework` is statically linked. `CudaMempoolArena` can inherit from `IArena` without issue.

However, the plugin factory's `CreateAllocatorImpl` must return `OrtAllocator*` (C API struct), not `IAllocator*`. This is the standard plugin C API boundary: plugin factories communicate through C structs, not C++ class hierarchies. A thin wrapper bridges the two:

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

### 4.5 Arena Config Parsing

The plugin factory's `CreateAllocatorImpl` receives `const OrtKeyValuePairs* allocator_options` (after the Part A fix — previously `nullptr`). The relevant keys:
- `arena.use_cuda_mempool` — `"1"` to enable
- `arena.cuda_mempool_release_threshold` — bytes; `0` disables threshold
- `arena.cuda_mempool_bytes_to_keep_on_shrink` — bytes retained after `Shrink()`

These can be parsed via `OrtArenaCfg::FromKeyValuePairs()` or directly from the key-value pairs using the `OrtApi`.

### 4.6 Summary of Changes for CudaMempoolArena Migration

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

## 5. Recommended Plan

### Phase 1: BFCArena Integration (Option B — ORT Core Changes)

Option B is recommended because it requires no new public API surface, uses existing `allocator_options` plumbing, covers both shared and per-session allocator paths, and is naturally gated by arena config keys (only EPs that pass them get wrapping).

1. Update `Environment::RegisterExecutionProviderLibrary()` to extract `ep_factory.<registration_name>.arena.*` keys from `config_entries_`; if found, strip the prefix and build `OrtKeyValuePairs` with bare `"arena.*"` keys; if not found, construct default arena options (`OrtArenaCfg{0, -1, -1, -1, -1, -1L}`). Pass the result to `CreateSharedAllocatorImpl()` instead of `nullptr`.
2. Update `Environment::CreateSharedAllocatorImpl()` to parse `allocator_options` for arena config keys and wrap the returned `IAllocator` in BFCArena via `CreateAllocator(AllocatorCreationInfo{...})` when arena keys are present
3. Update `PluginExecutionProvider::CreatePreferredAllocators()` to wrap returned allocators in BFCArena using EP-stored arena config (populated during EP creation from session/provider options)
4. Extract a shared helper for the arena-wrapping logic so both sites stay consistent
5. Test both shared allocator path and per-session path; verify `use_env_allocators` works correctly

### Phase 2: Migrate `CudaMempoolArena` to Plugin Build

This phase requires ORT core changes from Phase 1 to be in place (arena-already-handled signal from Section 3.5).

1. Add conditional logger abstraction to `cuda_mempool_arena.h/.cc` (Option 1 from Section 4.3)
2. Create `CudaMempoolOrtAllocator` wrapper in `plugin/cuda_allocator_plugin.h/.cc`
3. Update `CudaEpFactory::CreateAllocatorImpl` to create mempool allocator when configured
4. Parse mempool options from provider/session options in `CudaEpFactory`
5. Remove `cuda_mempool_arena.cc` from plugin CMake exclusion list
6. Test with `arena.use_cuda_mempool=1` provider option

### Phase 3: Parity Validation

1. Verify arena mode selection matches in-tree EP: default BFCArena, CUDA mempool if configured
2. Benchmark allocation performance vs. in-tree EP
3. Verify `DisableCpuMemArena()` does not affect CUDA plugin allocators (it shouldn't)
4. Test shared allocator replacement (environment allocators replacing per-session)

---

## 6. Decisions and Open Questions

### Decided

1. **Stream-aware BFCArena: match in-tree behavior by memory type.** The in-tree CUDA EP hardcodes the stream-awareness decision per allocator type: GPU device allocator → `StreamAwareBFCArena` (`use_stream_aware_arena = true`), pinned allocator → `BFCArena` (`use_stream_aware_arena = false`). The plugin path will follow the same convention. The arena-wrapping helper (used by both `CreateSharedAllocatorImpl` and `CreatePreferredAllocators`) determines stream-awareness from the `OrtMemoryInfo` of the allocator being wrapped: if the memory is on a GPU device, create `StreamAwareBFCArena`; if it is host-accessible (pinned), create `BFCArena`. This matches the in-tree EP's `AllocatorCreationInfo` parameters without introducing a new config key.

2. **Arena wrapping for shared allocators at `RegisterExecutionProviderLibrary` time.** Shared allocators will be wrapped in BFCArena at EP library registration, matching the behavior of per-session allocators for uniformity. The rationale:
   - Without arena wrapping, `use_env_allocators=1` replaces arena-backed per-session allocators with raw shared ones, silently degrading performance.
   - If the default arena config causes excessive upfront memory usage, the application can correct this by providing explicit arena options via `CreateEnvWithOptions` environment config (e.g., `ep_factory.cuda.arena.max_mem`).
   - **Pinned allocator exception:** The pinned allocator arena is always created with default `AllocatorCreationInfo` settings regardless of env or session options. This means: `use_stream_aware_arena = false`, `use_arena = true`, and `OrtArenaCfg{0, -1, -1, -1, -1, -1L}`. This behavior must be preserved — the pinned allocator arena config is not configurable via `ep_factory.*` or `ep.*` keys. Only the device allocator's arena config is driven by options.
   - **Needs validation:** Confirm that default arena options (`OrtArenaCfg{0, -1, -1, -1, -1, -1L}`) do not cause excessive upfront memory allocation in BFCArena. The `max_mem=0` default means "ORT chooses" — verify what BFCArena actually allocates at construction time vs. on first `Alloc()` call.

3. **Default arena config values: use in-tree defaults.** The plugin path will use the same defaults as the in-tree EP (`OrtArenaCfg{0, -1, -1, -1, -1, -1L}`) for both GPU device and pinned allocators. This is already captured in Decided 2 (pinned always uses defaults; device uses env/session options or falls back to defaults). The "Needs validation" item in Decided 2 covers confirming that `max_mem=0` does not cause excessive upfront allocation.

4. **Helper function for arena wrapping: yes, extract a shared helper.** Both `CreateSharedAllocatorImpl` and `CreatePreferredAllocators` need the same wrapping logic: parse `OrtArenaCfg` from options, determine stream-awareness from `OrtMemoryInfo`, check allocator name against `OrtEpDevice` baseline to detect self-contained arenas (Section 3.5), and call `CreateAllocator(AllocatorCreationInfo{...})`. A shared helper (e.g., `MaybeWrapInArena(AllocatorPtr, const OrtKeyValuePairs*, const OrtEpDevice&)`) keeps both sites consistent. This is an implementation detail, not a design question.
