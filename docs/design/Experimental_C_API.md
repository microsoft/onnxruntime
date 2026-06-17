# Experimental C API Design

## Problem Statement

The ORT C API (`OrtApi` struct) provides a stable binary interface with strict backward compatibility guarantees: functions are append-only, never reordered or removed, and versioned so that older clients work against newer libraries.

This stability comes at a cost: new public APIs cannot be iterated on. Once a function is added to the stable struct, its signature and slot are permanent. We need a mechanism to expose new APIs for experimental/preview usage before committing them to the stable surface.

Requirements:
- Allow test usage of new public APIs before promotion to the stable API
- No stability guarantee for experimental functions
- Minimal impact on the stable API surface
- Reasonable ergonomics for C and C++ consumers
- Functions may persist across multiple releases if unchanged
- Clean promotion path to the stable API

## Approaches Considered

### Approach A: Versioned Experimental Struct (Exact Match)

Add a stable API entry point that returns a `const OrtExperimentalApi*` struct, similar to the existing sub-API pattern (`GetCompileApi`, `GetEpApi`, etc.). The struct would require an exact version match—any change to the struct layout bumps the version, and the runtime only satisfies the exact version it was built with.

**Pros:**
- Same ergonomics as the existing stable API (typed struct, IDE autocomplete)
- Full type safety with no casts after the initial retrieval
- Familiar pattern for existing ORT API consumers

**Cons:**
- A struct version bump due to *any* function changing breaks *all* consumers, even those only using unchanged functions
- Mimics the stable API pattern but intentionally breaks its contract—semantically confusing
- Requires either per-version headers or a single "latest only" header
- Doesn't naturally support functions that persist unchanged across releases
- If a user doesn't know the runtime ORT version, they'd have to probe multiple versions

### Approach B: Name-Based Function Pointer Lookup

Add a single stable API entry point that retrieves an experimental function pointer by name. A companion header provides typedefs, name constants, and (for C++) typed accessor helpers.

**Pros:**
- Each function is independently addressable—unchanged functions keep resolving across releases
- Signature changes use a new name (API version-introduced suffix guarantees uniqueness); old name can be removed independently
- Minimal stable API cost (one slot)
- The instability contract is semantically clear: "is this specific thing available?"
- Promotion to stable is clean: move to `OrtApi`, optionally keep the name as a redirect
- Adding/removing experimental functions doesn't affect unrelated consumers

**Cons:**
- Requires one cast from the generic function pointer to the correct function pointer type
- Less discoverable without the companion header
- String-based lookup has minor runtime cost (irrelevant—done once at init)

## Chosen Approach: Name-Based Lookup with Typed C++ Helpers

The name-based approach (B) is the better fit because:

1. **Individual function longevity**: Experimental functions may persist unchanged for several releases before promotion. The struct approach breaks all consumers on any change; name-based lookup only affects the specific function that changed.

2. **Honest contract**: A struct *looks* stable but isn't. A per-function lookup makes instability semantically obvious.

3. **Simpler maintenance**: No struct layout to track, no version bumps to coordinate. Just add/remove entries from a registration table.

4. **Ergonomics are solvable**: The C++ wrapper with macro-generated typed accessors provides essentially the same user experience as a struct, minus one initial cast.

## Design Details

### Stable API Addition

One function pointer slot added to `OrtApi`:

```c
// Generic function pointer type used as an opaque handle.
// Cast to the correct function pointer type before calling.
typedef void (ORT_API_CALL* OrtExperimentalFnPtr)(void);

// Returns nullptr if the named function is not available in this build.
OrtExperimentalFnPtr(ORT_API_CALL* GetExperimentalFunction)(_In_ const char* name) NO_EXCEPTION;
```

Using a function pointer type (rather than `void*`) ensures that casting to the correct
function pointer type and back is well-defined in both C and C++ per the standard.
Consumers must cast the returned value to the exact typedef before calling—calling through
any other type is undefined behavior. Including `ORT_API_CALL` in the typedef matches the
calling convention of all ORT API functions, which avoids compiler warnings when casting
between the generic and typed pointers.

### Single Source of Truth: The `.inc` File

All experimental functions are declared in one [X-macro](https://en.wikipedia.org/wiki/X_macro) include file.
The first argument is the ORT API version in which the function was introduced. The macro
mechanically constructs the lookup name as `<Name>_SinceV<API Version>`, guaranteeing uniqueness
by construction—no two entries can collide unless they share both the same version and the
same base name, which is trivially avoided during review.

```c
// onnxruntime_experimental_c_api.inc
//
// ORT_EXPERIMENTAL_API(SinceVersion, ReturnType, Name, Params...)

ORT_EXPERIMENTAL_API(22, OrtStatusPtr, OrtApi_SomeNewThing,
    _In_ const OrtSession* session, _Out_ int64_t* result)

ORT_EXPERIMENTAL_API(22, OrtStatusPtr, OrtApi_AnotherThing,
    _In_ const OrtEnv* env, _In_ const char* name, _Out_ OrtValue** out)
```

### Experimental Consumer Headers (generated from `.inc`)

Four headers serve experimental API consumers, mirroring the `onnxruntime_c_api.h` / `onnxruntime_cxx_api.h` split. Each
public header is paired with a `_fns.h` detail header that holds only the mechanical, X-macro-generated per-function
plumbing, so the public header stays focused on the curated, hand-written surface:

- `onnxruntime_experimental_c_api.h` — pure C, public. Holds hand-written auxiliary declarations (e.g. opaque types the
  experimental APIs require) and includes the C detail header.
  - `onnxruntime_experimental_c_api_fns.h` — detail header: the X-macro pass that produces the function pointer
    typedefs and name constants. Not included directly by consumers.
- `onnxruntime_experimental_cxx_api.h` — C++ companion, public. Includes the C header (and `onnxruntime_cxx_api.h`),
  includes the C++ detail header, and is where hand-written C++ helpers live.
  - `onnxruntime_experimental_cxx_api_fns.h` — detail header: the X-macro passes that produce the typed accessors in
    the `Ort::Experimental` namespace. Not included directly by consumers.

The `.inc` file remains the single source of truth; the `_fns.h` headers are "do not hand-edit except the macro
template." Splitting the generated plumbing out keeps each public header as the place a contributor edits by hand
(auxiliary type declarations, convenience helpers) without wading through macro expansions. Direct inclusion of a
`_fns.h` header triggers a `#error`: each public header defines a short `ORT_INCLUDING_*` guard macro around its detail
include, and the detail header checks for it.

```c
// onnxruntime_experimental_c_api.h  (public)
#pragma once

#include "onnxruntime_c_api.h"

// Declare any new, auxiliary opaque types required by the experimental APIs here, before the detail include.
// ORT_RUNTIME_CLASS(...);

// Per-function typedefs and name constants (X-macro pass over the .inc) live in the detail header. The define/undef
// guard enforces that the detail header is only included through this header.
#define ORT_INCLUDING_EXPERIMENTAL_C_API_FNS
#include "onnxruntime_experimental_c_api_fns.h"
#undef ORT_INCLUDING_EXPERIMENTAL_C_API_FNS
```

```c
// onnxruntime_experimental_c_api_fns.h  (detail — included by the header above)
#pragma once

#ifndef ORT_INCLUDING_EXPERIMENTAL_C_API_FNS
#error "Include onnxruntime_experimental_c_api.h; do not include this detail header directly."
#endif

// --- C: function pointer typedefs and name constants ---
#define ORT_EXPERIMENTAL_API(VER, RET, NAME, ...)                                                    \
  typedef RET(ORT_API_CALL* OrtExperimental_##NAME##_SinceV##VER##_Fn)(__VA_ARGS__) NO_EXCEPTION; \
  static const char* const kOrtExperimental_##NAME##_SinceV##VER##_FnName = #NAME "_SinceV" #VER;
#include "onnxruntime_experimental_c_api.inc"
#undef ORT_EXPERIMENTAL_API

// Produces (for SinceVersion=22, Name=OrtApi_SomeNewThing):
//   typedef OrtStatusPtr(ORT_API_CALL* OrtExperimental_OrtApi_SomeNewThing_SinceV22_Fn)(
//       ...) NO_EXCEPTION;
//   static const char* const kOrtExperimental_OrtApi_SomeNewThing_SinceV22_FnName =
//       "OrtApi_SomeNewThing_SinceV22";
```

The C++ header generates two accessor flavors per function: a nullable accessor (returns `nullptr` if the function is
unavailable) and a throwing accessor (`...FnOrThrow`, throws `Ort::Exception` with `ORT_NOT_IMPLEMENTED` if the function
is unavailable). The nullable accessor is for runtime availability checks; the throwing accessor is for when the
function is required.

```cpp
// onnxruntime_experimental_cxx_api.h  (public)
#pragma once

#include "onnxruntime_experimental_c_api.h"
#include "onnxruntime_cxx_api.h"  // for Ort::Exception / ORT_CXX_API_THROW

// Typed accessors (nullable + throwing) live in the detail header, which declares them in Ort::Experimental.
// Hand-written C++ helpers can be added in their own Ort::Experimental namespace block.
#define ORT_INCLUDING_EXPERIMENTAL_CXX_API_FNS
#include "onnxruntime_experimental_cxx_api_fns.h"
#undef ORT_INCLUDING_EXPERIMENTAL_CXX_API_FNS
```

```cpp
// onnxruntime_experimental_cxx_api_fns.h  (detail — included by the header above)
#pragma once

#ifndef ORT_INCLUDING_EXPERIMENTAL_CXX_API_FNS
#error "Include onnxruntime_experimental_cxx_api.h; do not include this detail header directly."
#endif

namespace Ort {
namespace Experimental {

// --- C++: nullable typed inline accessors (reuses the C typedefs) ---
#define ORT_EXPERIMENTAL_API(VER, RET, NAME, ...)                                       \
  inline OrtExperimental_##NAME##_SinceV##VER##_Fn Get_##NAME##_SinceV##VER##_Fn(       \
      const OrtApi* api) {                                                              \
    return reinterpret_cast<OrtExperimental_##NAME##_SinceV##VER##_Fn>(                 \
        api->GetExperimentalFunction(kOrtExperimental_##NAME##_SinceV##VER##_FnName));  \
  }
#include "onnxruntime_experimental_c_api.inc"
#undef ORT_EXPERIMENTAL_API

// --- C++: throwing typed inline accessors (reuse the nullable accessors) ---
#define ORT_EXPERIMENTAL_API(VER, RET, NAME, ...)                                          \
  inline OrtExperimental_##NAME##_SinceV##VER##_Fn Get_##NAME##_SinceV##VER##_FnOrThrow(   \
      const OrtApi* api) {                                                                 \
    auto* fn = Get_##NAME##_SinceV##VER##_Fn(api);                                         \
    if (fn == nullptr) {                                                                   \
      ORT_CXX_API_THROW(                                                                   \
          "Experimental function " #NAME "_SinceV" #VER " is not available in this build", \
          ORT_NOT_IMPLEMENTED);                                                            \
    }                                                                                      \
    return fn;                                                                             \
  }
#include "onnxruntime_experimental_c_api.inc"
#undef ORT_EXPERIMENTAL_API

// Produces (for SinceVersion=22, Name=OrtApi_SomeNewThing):
//   inline OrtExperimental_OrtApi_SomeNewThing_SinceV22_Fn
//   Get_OrtApi_SomeNewThing_SinceV22_Fn(const OrtApi* api) {
//     return reinterpret_cast<OrtExperimental_OrtApi_SomeNewThing_SinceV22_Fn>(
//         api->GetExperimentalFunction(kOrtExperimental_OrtApi_SomeNewThing_SinceV22_FnName));
//   }
//   inline OrtExperimental_OrtApi_SomeNewThing_SinceV22_Fn
//   Get_OrtApi_SomeNewThing_SinceV22_FnOrThrow(const OrtApi* api) {
//     auto* fn = Get_OrtApi_SomeNewThing_SinceV22_Fn(api);
//     if (fn == nullptr) {
//       ORT_CXX_API_THROW(
//           "Experimental function OrtApi_SomeNewThing_SinceV22 is not available in this build",
//           ORT_NOT_IMPLEMENTED);
//     }
//     return fn;
//   }

}  // namespace Experimental
}  // namespace Ort
```

C usage:

```c
OrtExperimental_OrtApi_SomeNewThing_SinceV22_Fn fn =
    (OrtExperimental_OrtApi_SomeNewThing_SinceV22_Fn)api->GetExperimentalFunction(
        kOrtExperimental_OrtApi_SomeNewThing_SinceV22_FnName);
if (fn) {
  OrtStatusPtr status = fn(session, &result);
}
```

C++ usage (nullable):

```cpp
if (auto* fn = Ort::Experimental::Get_OrtApi_SomeNewThing_SinceV22_Fn(api)) {
  Ort::Status status(fn(session, &result));
}
```

C++ usage (throwing):

```cpp
auto* fn = Ort::Experimental::Get_OrtApi_SomeNewThing_SinceV22_FnOrThrow(api);
Ort::Status status(fn(session, &result));
```

### Implementation Side (generated from `.inc`)

```cpp
// experimental_c_api.cc

// Function implementations use the full constructed name.
ORT_API_STATUS_IMPL(OrtExperimentalApis::OrtApi_SomeNewThing_SinceV22,
                    _In_ const OrtSession* session, _Out_ int64_t* result) {
  API_IMPL_BEGIN
  // ...
  API_IMPL_END
}

// Registration table (auto-generated from .inc)
struct ExperimentalEntry {
  std::string_view name;
  OrtExperimentalFnPtr fn;
};

static const ExperimentalEntry kExperimentalFunctions[] = {
#define ORT_EXPERIMENTAL_API(VER, RET, NAME, ...) \
  { #NAME "_SinceV" #VER, reinterpret_cast<OrtExperimentalFnPtr>(&OrtExperimentalApis::NAME##_SinceV##VER) },
#include "onnxruntime_experimental_c_api.inc"
#undef ORT_EXPERIMENTAL_API
};

// Lookup implementation
ORT_API(OrtExperimentalFnPtr, OrtApis::GetExperimentalFunction, _In_ const char* name) {
  if (name == nullptr) return nullptr;
  std::string_view target(name);
  for (const auto& entry : kExperimentalFunctions) {
    if (entry.name == target) return entry.fn;
  }
  return nullptr;
}
```

### Lifecycle Rules

1. **Adding an experimental function**: Add one line to the `.inc` file with the current ORT API version, implement it.
2. **Removing a function**: Delete the line from the `.inc`. No retirement tracking is needed—the versioned name is inherently unique and cannot be accidentally reused.
3. **Changing a signature**: Add a new entry with the current ORT API version (producing a new unique name) and optionally delete the old entry. Both can coexist if the old signature is still supported.
4. **Promoting to stable**: Add the function to the stable API struct (append-only, name drops the `_SinceV<ver>` suffix). Delete the experimental entry. Optionally keep the experimental entry for a transitional period, resolving it as a redirect.

### Naming Convention

Experimental function names follow the pattern `<TargetStruct>_<Name>_SinceV<API Version>`.
The API version is the ORT API version in which the function was first introduced. The target
struct prefix indicates the intended promotion destination. This naming scheme guarantees
uniqueness by construction—a signature change requires a new `.inc` entry at the current
API version, which produces a distinct name.

Examples:

- `OrtApi_SomeNewThing_SinceV22` — introduced in API v22, destined for `OrtApi`
- `OrtApi_SomeNewThing_SinceV23` — signature changed in API v23, replaces the v22 entry
- `OrtEpApi_SomeNewEpThing_SinceV22` — destined for `OrtEpApi`
- `OrtCompileApi_SomeNewCompileThing_SinceV22` — destined for `OrtCompileApi`

At promotion, the stable struct member drops the `_SinceV<API Version>` suffix (e.g., the stable
slot is named `SomeNewThing` in `OrtApi`).

Names are flat strings matched exactly. No separate retirement tracking is needed because
the version suffix makes accidental name reuse impossible.

### Rejected: Enumeration Helper

A runtime enumeration API (`GetExperimentalFunctionNames`) was considered but rejected.
Any consumer that wants to *call* an experimental function already needs the header for the
typedef—and the header *is* the enumeration. A consumer without the header could discover
names at runtime but couldn't safely call them (no type information). This would cost a
stable API slot for no practical benefit.

## Open Questions

- Should the Python/C#/Java bindings expose experimental functions, or keep them C/C++ only initially?
  - We can start with C/C++ and prove it out first.
- Should we document an "epoch" expectation (e.g., "experimental functions are expected to be promoted or removed within 2 releases")?
  - We can set a general expectation without enforcing it.
