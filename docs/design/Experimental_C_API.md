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
- Signature changes use a new name (`Foo_v2`); old name can be removed independently
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

All experimental functions are declared in one [X-macro](https://en.wikipedia.org/wiki/X_macro) include file:

```c
// onnxruntime_experimental_api.inc
//
// ORT_EXPERIMENTAL_FUNC(Name, ReturnType, Params...)

ORT_EXPERIMENTAL_FUNC(OrtApi_SomeNewThing, OrtStatusPtr,
    _In_ const OrtSession* session, _Out_ int64_t* result)

ORT_EXPERIMENTAL_FUNC(OrtApi_AnotherThing, OrtStatusPtr,
    _In_ const OrtEnv* env, _In_ const char* name, _Out_ OrtValue** out)
```

### C Header (generated from `.inc`)

```c
// onnxruntime_experimental_api.h

// --- Function pointer typedefs and name constants (auto-generated from .inc) ---
#define ORT_EXPERIMENTAL_FUNC(NAME, RET, ...)                                         \
  typedef RET(ORT_API_CALL* OrtExperimental_##NAME##_Fn)(__VA_ARGS__) NO_EXCEPTION;   \
  static const char* const kOrtExperimental_##NAME = #NAME;
#include "onnxruntime_experimental_api.inc"
#undef ORT_EXPERIMENTAL_FUNC

// Produces:
//   typedef OrtStatusPtr(ORT_API_CALL* OrtExperimental_OrtApi_SomeNewThing_Fn)(...) NO_EXCEPTION;
//   static const char* const kOrtExperimental_OrtApi_SomeNewThing = "OrtApi_SomeNewThing";
//
//   typedef OrtStatusPtr(ORT_API_CALL* OrtExperimental_OrtApi_AnotherThing_Fn)(...) NO_EXCEPTION;
//   static const char* const kOrtExperimental_OrtApi_AnotherThing = "OrtApi_AnotherThing";
```

C usage:

```c
OrtExperimental_OrtApi_SomeNewThing_Fn fn =
    (OrtExperimental_OrtApi_SomeNewThing_Fn)api->GetExperimentalFunction(kOrtExperimental_OrtApi_SomeNewThing);
if (fn) {
  OrtStatusPtr status = fn(session, &result);
}
```

### C++ Header (generated from `.inc`)

Uses a macro to declare the typedef and a typed inline accessor in one shot:

```cpp
// onnxruntime_experimental_cxx_api.h
namespace Ort::Experimental {

#define ORT_EXPERIMENTAL_FUNC(NAME, RET, ...)                                    \
  typedef RET(ORT_API_CALL* NAME##_Fn)(__VA_ARGS__) NO_EXCEPTION;                \
  inline NAME##_Fn Get_##NAME##_Fn(const OrtApi* api) {                          \
    return reinterpret_cast<NAME##_Fn>(api->GetExperimentalFunction(#NAME));     \
  }
#include "onnxruntime_experimental_api.inc"
#undef ORT_EXPERIMENTAL_FUNC

}  // namespace Ort::Experimental

// Produces:
// namespace Ort::Experimental {
//
//   typedef OrtStatusPtr(ORT_API_CALL* OrtApi_SomeNewThing_Fn)(
//       _In_ const OrtSession* session, _Out_ int64_t* result) NO_EXCEPTION;
//   inline OrtApi_SomeNewThing_Fn Get_OrtApi_SomeNewThing_Fn(const OrtApi* api) {
//     return reinterpret_cast<OrtApi_SomeNewThing_Fn>(api->GetExperimentalFunction("OrtApi_SomeNewThing"));
//   }
//
//   typedef OrtStatusPtr(ORT_API_CALL* OrtApi_AnotherThing_Fn)(
//       _In_ const OrtEnv* env, _In_ const char* name, _Out_ OrtValue** out) NO_EXCEPTION;
//   inline OrtApi_AnotherThing_Fn Get_OrtApi_AnotherThing_Fn(const OrtApi* api) {
//     return reinterpret_cast<OrtApi_AnotherThing_Fn>(api->GetExperimentalFunction("OrtApi_AnotherThing"));
//   }
//
// }  // namespace Ort::Experimental
```

C++ usage:

```cpp
if (auto fn = Ort::Experimental::Get_OrtApi_SomeNewThing_Fn(api)) {
  Ort::Status status(fn(session, &result));
}
```

### Implementation Side (generated from `.inc`)

```cpp
// experimental_api.cc

// Function implementations
ORT_API_STATUS_IMPL(OrtExperimentalApis::OrtApi_SomeNewThing,
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
#define ORT_EXPERIMENTAL_FUNC(NAME, ...) { #NAME, reinterpret_cast<OrtExperimentalFnPtr>(&OrtExperimentalApis::NAME) },
#include "onnxruntime_experimental_api.inc"
#undef ORT_EXPERIMENTAL_FUNC
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

### Name Reuse Prevention

When a function name is retired (removed or superseded by a `_v2`), it must never be
reused with a different signature—a stale client holding a cached pointer to the old name
would call it with the wrong arguments. A compile-time check is sufficient to enforce this:

```c
// onnxruntime_experimental_retired.inc
//
// Names that were once registered and must not be reused.
// ORT_RETIRED_EXPERIMENTAL_FUNC(Name)

ORT_RETIRED_EXPERIMENTAL_FUNC(OrtApi_SomeOldThing)
ORT_RETIRED_EXPERIMENTAL_FUNC(OrtApi_SomeNewThing_v1)
```

In the implementation file, after building the active registration table:

```cpp
// Compile-time check: no active name may collide with a retired name.
#define ORT_RETIRED_EXPERIMENTAL_FUNC(NAME)                                          \
  static_assert(                                                                     \
      std::none_of(std::begin(kExperimentalFunctions), std::end(kExperimentalFunctions), \
                   [](const ExperimentalEntry& e) { return e.name == #NAME; }),       \
      "Experimental function name '" #NAME "' is retired and must not be reused.");
#include "onnxruntime_experimental_retired.inc"
#undef ORT_RETIRED_EXPERIMENTAL_FUNC
```

Runtime enforcement is unnecessary: if the `static_assert` passes, the retired name
cannot exist in the lookup table, so no query for it can ever succeed. The compile-time
check alone is the single enforcement point.

### Lifecycle Rules

1. **Adding an experimental function**: Add one line to the `.inc` file, implement it.
2. **Removing a function**: Delete the line from the `.inc`; add the name to the retired `.inc`.
3. **Changing a signature**: This requires a new function name. Add a new function with an incremented version suffix (`_v2`) and optionally remove the old function (see cases 1 and 2). If the old function is kept around, both signatures are supported.
4. **Promoting to stable**: Add the function to the stable API (append-only). Remove the experimental function (see case 2). Optionally keep the experimental function around for a short period, resolving it as a redirect.

### Naming Convention

Experimental function names are prefixed with the target stable API struct name, making
the intended promotion destination clear. If a signature changes, append `_v2`, `_v3`, etc.
Examples:

- `OrtApi_SomeNewThing` — destined for `OrtApi`
- `OrtApi_SomeNewThing_v2` — updated signature, replaces the original
- `OrtEpApi_SomeNewEpThing` — destined for `OrtEpApi`
- `OrtCompileApi_SomeNewCompileThing` — destined for `OrtCompileApi`

Names are flat strings matched exactly. No formal namespace separator beyond this prefix
convention is needed.

### Rejected: Enumeration Helper

A runtime enumeration API (`GetExperimentalFunctionNames`) was considered but rejected.
Any consumer that wants to *call* an experimental function already needs the header for the
typedef—and the header *is* the enumeration. A consumer without the header could discover
names at runtime but couldn't safely call them (no type information). This would cost a
stable API slot for no practical benefit.

## Open Questions

- Should the Python/C#/Java bindings expose experimental functions, or keep them C/C++ only initially?
- Should we document an "epoch" expectation (e.g., "experimental functions are expected to be promoted or removed within 2 releases")?
