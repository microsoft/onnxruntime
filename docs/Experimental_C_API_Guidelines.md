# Experimental C API Guidelines

## Overview

ONNX Runtime (ORT) provides a mechanism for exposing experimental C API functions. Experimental APIs are a proving ground for new functionality. They may eventually be promoted to the stable ORT API or abandoned, so their availability is not guaranteed across releases. While an experimental function is available under a given name, however, its signature does not change, and its ABI-relevant types and behavioral contract remain backward compatible.

Experimental functions are not part of the stable `OrtApi` struct. Instead, they are resolved by name at runtime through the stable C API [`OrtApi::GetExperimentalFunction`](https://onnxruntime.ai/docs/api/c/struct_ort_api.html#ace33338eb9175cdc92b52f61c9aff9a6), which was introduced in ORT 1.28. Given a name, it returns a generic function pointer, or `NULL` if the named function is not available in the running ORT. A non-null result means the function is present and can be used; a null result means it is absent and the caller must fall back or fail gracefully. This keeps experimental functions off the stable ABI while still making them reachable from any ORT build that supports them.

Every experimental function has a name that ends in a `_SinceV<N>` suffix, where `N` is the ORT API version in which that function was first introduced (for example, `OrtApi_ExperimentalApiTest_SinceV28`). The suffix is part of the identity of the function: it makes each name unique across API versions (which correspond to ORT minor releases). Backward-compatible fixes and behavioral extensions may retain the existing name; a signature change or an incompatible ABI or behavioral-contract change requires a new name.

The lookup names, the function-signature typedefs, and any auxiliary types the experimental functions require are provided by a companion C header, [onnxruntime_experimental_c_api.h](../include/onnxruntime/core/session/onnxruntime_experimental_c_api.h). C++ consumers should instead use the companion C++ header, [onnxruntime_experimental_cxx_api.h](../include/onnxruntime/core/session/onnxruntime_experimental_cxx_api.h), which builds on the C header and adds typed accessors in the `Ort::Experimental` namespace.

## User Guidelines

These guidelines are for consumers who want to *call* experimental functions.

### Expect availability to vary

There is no promise that an experimental function will exist in any future (or past) version of ORT. A newer experimental header may also remove its typedef and name constant. By choosing to use an experimental API, you are committing to adapting your code when updating the header — either following the function to its stable replacement after promotion, or accounting for its removal. At runtime, do not assume that a function declared by the header used to build your application is present in the loaded ORT library.

### Check availability at runtime

Because an experimental function may be absent, always confirm that the lookup returned a usable pointer before calling it. In C, this means checking the pointer returned by `GetExperimentalFunction` against `NULL`. The C++ header offers two accessor flavors per function so you can choose how absence is handled:

- A nullable accessor (`Get_<Name>_Fn`) returns the typed function pointer, or `nullptr` if the function is not available in this build. Use it when the function is optional and you want to branch on availability.
- A throwing accessor (`Get_<Name>_FnOrThrow`) returns a guaranteed-non-null typed function pointer, or throws `Ort::Exception` with `ORT_NOT_IMPLEMENTED` if the function is not available. Use it when the function is required and its absence is a hard error.

### Use the experimental headers for names and signature typedefs

Do not hard-code experimental function names or re-declare their signatures by hand. The companion headers are the source of truth:

- The C header supplies, for each experimental function, a function-pointer typedef (`OrtExperimental_<Name>_Fn`) and a name constant (`kOrtExperimental_<Name>_FnName`). Look the function up with the name constant, then cast the returned generic pointer to the matching typedef before calling it. Calling through any other type is undefined behavior.
- The C++ header wraps this pattern in the typed `Ort::Experimental` accessors described above, so you get the correct type without a manual cast, plus any C++ wrapper types associated with the experimental functions.

The headers provide the known function names and their signatures; the runtime you are calling into will either have a given function or not, which is exactly what the runtime lookup tells you. For C and C++ usage examples, see [onnxruntime_experimental_c_api.h](../include/onnxruntime/core/session/onnxruntime_experimental_c_api.h).

## Implementer Guidelines

These guidelines are for ORT contributors who want to *add* an experimental function.

### Declare the function in one place

Experimental functions are declared in a single X-macro include file, [`onnxruntime_experimental_c_api.inc`](../include/onnxruntime/core/session/onnxruntime_experimental_c_api.inc), that serves as the source of truth for the consumer headers and the runtime registration table. Add one `ORT_EXPERIMENTAL_API(SinceVersion, ReturnType, Name, Params...)` entry, where:

- `SinceVersion` is the current ORT API version, written as a numeric literal (for example, `28`) — not a macro such as `ORT_API_VERSION`.
- `ReturnType` is the function's return type, typically `OrtStatusPtr`.
- `Name` is the base name (`<BaseName>` below) without the version suffix, by convention prefixed with the stable struct the function is destined for.
- `Params` is the parameter list, using the same SAL annotations as other ORT C API declarations.

The lookup name (`<BaseName>_SinceV<SinceVersion>`), the typedefs, the name constants, the C++ accessors, and the registration-table entry are all generated mechanically from this one entry. Implement the function body under the `OrtExperimentalApis` namespace in whatever source location is appropriate.

### Follow the naming convention

Experimental function names follow the pattern `<BaseName>_SinceV<Version>`. `<BaseName>` consists of `<TargetStruct>_<FunctionName>`. The expanded pattern is `<TargetStruct>_<FunctionName>_SinceV<Version>`.

The `<TargetStruct>` prefix (for example, `OrtApi`, `OrtEpApi`, or `OrtCompileApi`) indicates the stable struct the function is intended to be promoted into.

`<FunctionName>` is the intended name of the C API function.

The `_SinceV<Version>` suffix records the ORT API version in which the experimental function first appeared and guarantees the name is unique by construction. This suffix is automatically appended via the `ORT_EXPERIMENTAL_API` X-macro.

At promotion, using the placeholders above, the `<TargetStruct>` struct gets a new function, `<FunctionName>`, appended to it.

### Place auxiliary types in the appropriate header

If an experimental function needs a new auxiliary type, declare it alongside the function's other artifacts: opaque handle types belong in the C header, and any C++ wrapper types belong in the C++ header, in the `Ort::Experimental` namespace.

### Document the expected promotion timeline

Document the expected promotion or abandonment timeline in the new API's documentation. E.g., "Promotion to stable API expected by version X." This is an expectation, not a contract, and it can be updated as plans change.

### Keep a given experimental API name backward compatible

Because an experimental API name may be available across more than one ORT release while consumers fetch the function by name, its signature must not change, and its ABI-relevant auxiliary types and behavioral contract must remain backward compatible for the lifetime of that name. Compatible bug fixes and behavioral extensions are allowed. For a signature change or an incompatible ABI or behavioral-contract change, introduce a *new* name rather than altering the existing one. This should be straightforward to do in a later API version, since the introducing API version is baked into the name.

### Add tests

Add test coverage for new experimental functions, exercising both the lookup mechanism and the function's behavior. Test coverage is expected just as for any other API function.

## Experimental API Lifecycle

1. **Add.** Add one experimental API entry with the current ORT API version and implement the function.
2. **Update.** A backward-compatible fix or behavioral extension may retain the existing name. For a signature change or an incompatible ABI or behavioral-contract change, introduce a new experimental API entry at the current ORT API version — which produces a new lookup name — rather than modifying the existing one. The old and new entries may coexist. Never reuse a name.
3. **Promote or remove.**
   - **Promote:** Add the function to the stable API struct (append-only, dropping the `<TargetStruct>_` prefix and the `_SinceV<Version>` suffix from the name) and remove the experimental entry.
   - **Remove:** Delete the experimental entry. Because the versioned name is unique and never reused, no separate retirement tracking is needed.

## References

- [onnxruntime_experimental_c_api.h](../include/onnxruntime/core/session/onnxruntime_experimental_c_api.h) — C names, typedefs, and auxiliary type declarations.
- [onnxruntime_experimental_cxx_api.h](../include/onnxruntime/core/session/onnxruntime_experimental_cxx_api.h) — C++ `Ort::Experimental` accessors and wrapper types.
- [test_experimental_api.cc](../onnxruntime/test/shared_lib/test_experimental_api.cc) — test coverage and example usage for experimental API function lookup and the C++ accessors.
