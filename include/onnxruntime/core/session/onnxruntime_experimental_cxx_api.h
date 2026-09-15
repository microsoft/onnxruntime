// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Experimental C++ API consumer header.
//
// This header provides typed inline accessors in the Ort::Experimental namespace for experimental ORT API functions,
// and any C++ wrapper types associated with the experimental API functions.
//
// It is the C++ companion to onnxruntime_experimental_c_api.h.
//
// IMPORTANT: Experimental functions are NOT part of the stable ABI. They may be added, changed, or removed between
// releases without notice. Anything in this file should be treated as experimental and unstable.
//
// Two accessor flavors are generated for each experimental function:
//
//   1. Get_<NAME>_SinceV<VER>_Fn(api)
//      Returns the typed function pointer, or nullptr if the function is not available in this build.
//      Use this to check availability at runtime.
//
//   2. Get_<NAME>_SinceV<VER>_FnOrThrow(api)
//      Returns the typed function pointer, or throws Ort::Exception (ORT_NOT_IMPLEMENTED) if the function is not
//      available in this build.
//      Use this when the function is required.
//
// C++ usage (nullable):
//   if (auto* fn = Ort::Experimental::Get_OrtApi_ExperimentalApiTest_SinceV28_Fn(api)) {
//     Ort::Status status(fn(&result));
//   }
//
// C++ usage (throwing):
//   auto* fn = Ort::Experimental::Get_OrtApi_ExperimentalApiTest_SinceV28_FnOrThrow(api);
//   Ort::Status status(fn(&result));

#pragma once

#include "onnxruntime_experimental_c_api.h"
#include "onnxruntime_cxx_api.h"

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

//
// Auxiliary types and helpers
//
// C++ wrapper types or helpers go here in the `Ort::Experimental` namespace.
//

// Move-only RAII owner for an OrtEpContextConfig handle, which carries the EPContext read/write callbacks and opaque
// state extracted from an OrtSessionOptions instance. The handle is released via OrtEpApi_ReleaseEpContextConfig when
// the wrapper is destroyed.
//
// Typical EP usage: construct from the session options during CreateEp(), keep the wrapper for the EP's lifetime, and
// query the callbacks via GetReadFunc() / GetWriteFunc().
class EpContextConfig {
 public:
  explicit EpContextConfig(std::nullptr_t) noexcept {}

  explicit EpContextConfig(const SessionOptions& session_options) : EpContextConfig{session_options.GetConst()} {}

  // Extracts the EPContext config from `session_options`. Throws Ort::Exception (ORT_NOT_IMPLEMENTED) if the
  // experimental functions are not available in this build, or propagates any error from the extraction.
  explicit EpContextConfig(ConstSessionOptions session_options) {
    const OrtApi* api = &GetApi();
    // Ensure the release function is available before creating a handle, so the handle can always be freed.
    Get_OrtEpApi_ReleaseEpContextConfig_SinceV28_FnOrThrow(api);
    auto* get_config = Get_OrtEpApi_SessionOptions_GetEpContextConfig_SinceV28_FnOrThrow(api);
    ThrowOnError(get_config(static_cast<const OrtSessionOptions*>(session_options), &config_));
  }

  EpContextConfig(EpContextConfig&& other) noexcept : config_{other.config_} { other.config_ = nullptr; }

  EpContextConfig& operator=(EpContextConfig&& other) noexcept {
    if (this != &other) {
      reset();
      config_ = other.config_;
      other.config_ = nullptr;
    }
    return *this;
  }

  EpContextConfig(const EpContextConfig&) = delete;
  EpContextConfig& operator=(const EpContextConfig&) = delete;

  ~EpContextConfig() { reset(); }

  OrtEpContextConfig* get() const noexcept { return config_; }
  explicit operator bool() const noexcept { return config_ != nullptr; }

  // Relinquishes ownership of the handle without releasing it.
  OrtEpContextConfig* release() noexcept {
    OrtEpContextConfig* released = config_;
    config_ = nullptr;
    return released;
  }

  // Releases any owned handle and resets to empty.
  void reset() noexcept {
    if (config_ != nullptr) {
      if (auto* release_fn = Get_OrtEpApi_ReleaseEpContextConfig_SinceV28_Fn(&GetApi())) {
        release_fn(config_);
      }
      config_ = nullptr;
    }
  }

  // Returns the configured read callback and opaque state (both nullptr if none was set). Throws on failure.
  void GetReadFunc(OrtReadNamedBufferFunc& read_func, void*& state) const {
    auto* get_read_func = Get_OrtEpApi_EpContextConfig_GetEpContextDataReadFunc_SinceV28_FnOrThrow(&GetApi());
    ThrowOnError(get_read_func(config_, &read_func, &state));
  }

  // Returns the configured write callback and opaque state (both nullptr if none was set). Throws on failure.
  void GetWriteFunc(OrtWriteNamedBufferFunc& write_func, void*& state) const {
    auto* get_write_func = Get_OrtEpApi_EpContextConfig_GetEpContextDataWriteFunc_SinceV28_FnOrThrow(&GetApi());
    ThrowOnError(get_write_func(config_, &write_func, &state));
  }

 private:
  OrtEpContextConfig* config_ = nullptr;
};

}  // namespace Experimental
}  // namespace Ort
