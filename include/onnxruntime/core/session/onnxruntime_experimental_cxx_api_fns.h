// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Experimental C++ API accessor declarations (per-function plumbing).
//
// This is an internal detail header for onnxruntime_experimental_cxx_api.h. Do NOT include it directly (enforced via
// the #error below) — include onnxruntime_experimental_cxx_api.h instead. It must be included after
// onnxruntime_experimental_c_api.h (for the C typedefs / name constants) and onnxruntime_cxx_api.h (for Ort::Exception
// / ORT_CXX_API_THROW).
//
// It performs two X-macro passes over onnxruntime_experimental_c_api.inc to declare, for each experimental function,
// a nullable typed accessor and a throwing (FnOrThrow) typed accessor, both in the Ort::Experimental namespace.

#pragma once

#ifndef ORT_INCLUDING_EXPERIMENTAL_CXX_API_FNS
#error "Include onnxruntime_experimental_cxx_api.h; do not include this detail header directly."
#endif

namespace Ort {
namespace Experimental {

//
// Nullable typed accessors
//

// For each .inc entry, this generates a typed accessor in Ort::Experimental:
//
//   inline OrtExperimental_<NAME>_SinceV<VER>_Fn Get_<NAME>_SinceV<VER>_Fn(const OrtApi* api);
//
// Example: ORT_EXPERIMENTAL_API(28, OrtStatusPtr, OrtApi_ExperimentalApiTest, ...) produces:
//   inline OrtExperimental_OrtApi_ExperimentalApiTest_SinceV28_Fn
//   Get_OrtApi_ExperimentalApiTest_SinceV28_Fn(const OrtApi* api) {
//     return reinterpret_cast<OrtExperimental_OrtApi_ExperimentalApiTest_SinceV28_Fn>(
//         api->GetExperimentalFunction(kOrtExperimental_OrtApi_ExperimentalApiTest_SinceV28_FnName));
//   }
#define ORT_EXPERIMENTAL_API(VER, RET, NAME, ...)                                      \
  inline OrtExperimental_##NAME##_SinceV##VER##_Fn Get_##NAME##_SinceV##VER##_Fn(      \
      const OrtApi* api) {                                                             \
    return reinterpret_cast<OrtExperimental_##NAME##_SinceV##VER##_Fn>(                \
        api->GetExperimentalFunction(kOrtExperimental_##NAME##_SinceV##VER##_FnName)); \
  }

#include "onnxruntime_experimental_c_api.inc"

#undef ORT_EXPERIMENTAL_API

//
// Throwing typed accessors
//

// For each .inc entry, this generates a throwing accessor in Ort::Experimental:
//
//   inline OrtExperimental_<NAME>_SinceV<VER>_Fn Get_<NAME>_SinceV<VER>_FnOrThrow(const OrtApi* api);
//
// It returns the non-null typed function pointer, or throws Ort::Exception (ORT_NOT_IMPLEMENTED) if the function is
// not available in this build.
//
// Example: ORT_EXPERIMENTAL_API(28, OrtStatusPtr, OrtApi_ExperimentalApiTest, ...) produces:
//   inline OrtExperimental_OrtApi_ExperimentalApiTest_SinceV28_Fn
//   Get_OrtApi_ExperimentalApiTest_SinceV28_FnOrThrow(const OrtApi* api) {
//     auto* fn = Get_OrtApi_ExperimentalApiTest_SinceV28_Fn(api);
//     if (fn == nullptr) {
//       ORT_CXX_API_THROW(
//           "Experimental function OrtApi_ExperimentalApiTest_SinceV28 is not available in this build",
//           ORT_NOT_IMPLEMENTED);
//     }
//     return fn;
//   }
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

}  // namespace Experimental
}  // namespace Ort
