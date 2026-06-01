// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Experimental C API consumer header.
//
// This header provides typedefs, name constants, and (for C++) typed inline accessors
// for experimental ORT functions. It is generated from onnxruntime_experimental_c_api.inc.
//
// IMPORTANT: Experimental functions are NOT part of the stable ABI. They may be added,
// changed, or removed between releases without notice. A function's availability should
// always be checked at runtime (the lookup returns nullptr if the function is not present).
//
// C usage:
//   OrtExperimental_OrtApi_ExperimentalApiTest_ExpSinceV27_Fn fn =
//       (OrtExperimental_OrtApi_ExperimentalApiTest_ExpSinceV27_Fn)api->GetExperimentalFunction(
//           kOrtExperimental_OrtApi_ExperimentalApiTest_ExpSinceV27_FnName);
//   if (fn) {
//     OrtStatusPtr status = fn(&result);
//   }
//
// C++ usage:
//   if (auto* fn = Ort::Experimental::Get_OrtApi_ExperimentalApiTest_ExpSinceV27_Fn(api)) {
//     Ort::Status status(fn(&result));
//   }

#pragma once

#include "onnxruntime_c_api.h"

// --- C: function pointer typedefs and name constants ---
//
// For each ORT_EXPERIMENTAL_FUNC(VER, NAME, RET, ...) entry in the .inc file, this generates:
//
//   // Function pointer typedef:
//   typedef RET(ORT_API_CALL* OrtExperimental_<NAME>_ExpSinceV<VER>_Fn)(...) NO_EXCEPTION;
//
//   // Name constant for lookup:
//   static const char* const kOrtExperimental_<NAME>_ExpSinceV<VER>_FnName = "<NAME>_ExpSinceV<VER>";
//
// Example: ORT_EXPERIMENTAL_FUNC(27, OrtApi_ExperimentalApiTest, OrtStatusPtr, _Out_ int64_t* out)
// produces:
//   typedef OrtStatusPtr(ORT_API_CALL* OrtExperimental_OrtApi_ExperimentalApiTest_ExpSinceV27_Fn)(
//       _Out_ int64_t* out) NO_EXCEPTION;
//   static const char* const kOrtExperimental_OrtApi_ExperimentalApiTest_ExpSinceV27_FnName =
//       "OrtApi_ExperimentalApiTest_ExpSinceV27";

#define ORT_EXPERIMENTAL_FUNC(VER, NAME, RET, ...)                                                    \
  typedef RET(ORT_API_CALL* OrtExperimental_##NAME##_ExpSinceV##VER##_Fn)(__VA_ARGS__) NO_EXCEPTION;  \
  static const char* const kOrtExperimental_##NAME##_ExpSinceV##VER##_FnName = #NAME "_ExpSinceV" #VER;
#include "onnxruntime_experimental_c_api.inc"
#undef ORT_EXPERIMENTAL_FUNC

#ifdef __cplusplus

namespace Ort {
namespace Experimental {

// --- C++: typed inline accessors (reuses the C typedefs above) ---
//
// For each .inc entry, this generates a typed accessor in Ort::Experimental:
//
//   inline OrtExperimental_<NAME>_ExpSinceV<VER>_Fn Get_<NAME>_ExpSinceV<VER>_Fn(const OrtApi* api);
//
// Example: for ORT_EXPERIMENTAL_FUNC(27, OrtApi_ExperimentalApiTest, ...) this produces:
//   inline OrtExperimental_OrtApi_ExperimentalApiTest_ExpSinceV27_Fn
//   Get_OrtApi_ExperimentalApiTest_ExpSinceV27_Fn(const OrtApi* api) {
//     return reinterpret_cast<OrtExperimental_OrtApi_ExperimentalApiTest_ExpSinceV27_Fn>(
//         api->GetExperimentalFunction(kOrtExperimental_OrtApi_ExperimentalApiTest_ExpSinceV27_FnName));
//   }

#define ORT_EXPERIMENTAL_FUNC(VER, NAME, RET, ...)                                                     \
  inline OrtExperimental_##NAME##_ExpSinceV##VER##_Fn Get_##NAME##_ExpSinceV##VER##_Fn(                \
      const OrtApi* api) {                                                                             \
    return reinterpret_cast<OrtExperimental_##NAME##_ExpSinceV##VER##_Fn>(                             \
        api->GetExperimentalFunction(kOrtExperimental_##NAME##_ExpSinceV##VER##_FnName));                  \
  }
#include "onnxruntime_experimental_c_api.inc"
#undef ORT_EXPERIMENTAL_FUNC

}  // namespace Experimental
}  // namespace Ort

#endif  // __cplusplus
