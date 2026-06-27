// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Runtime implementation for experimental C API functions.
// See onnxruntime_experimental_c_api.inc for the declaration list and lifecycle rules.

#include <cstring>
#include <array>

#include "core/framework/error_code_helper.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/session/onnxruntime_experimental_c_api.h"
#include "core/session/ort_apis.h"

// ---------------------------------------------------------------------------
// Experimental function implementations
// ---------------------------------------------------------------------------

namespace OrtExperimentalApis {

// Forward declarations driven by the .inc file so the registration table below
// can take the address of every entry, including those defined in other
// translation units linked into onnxruntime_session.
#define ORT_EXPERIMENTAL_API(VER, RET, NAME, ...) \
  RET ORT_API_CALL NAME##_SinceV##VER(__VA_ARGS__) NO_EXCEPTION;
#include "onnxruntime_experimental_c_api.inc"
#undef ORT_EXPERIMENTAL_API

// Test-only experimental function that writes a known sentinel value.
// Exists to exercise the experimental API mechanism end-to-end and to serve as a template for future experimental
// functions.
ORT_API_STATUS_IMPL(OrtApi_ExperimentalApiTest_SinceV28,
                    _Out_ int64_t* out) {
  API_IMPL_BEGIN
  if (out == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "out is null");
  }
  *out = 12345;
  return nullptr;
  API_IMPL_END
}

}  // namespace OrtExperimentalApis

// ---------------------------------------------------------------------------
// Registration table (auto-generated from .inc)
// ---------------------------------------------------------------------------

namespace {

struct ExperimentalEntry {
  const char* name;
  OrtExperimentalFnPtr fn;
};

static const std::array kExperimentalFunctions{

#define ORT_EXPERIMENTAL_API(VER, RET, NAME, ...)                   \
  ExperimentalEntry{kOrtExperimental_##NAME##_SinceV##VER##_FnName, \
                    reinterpret_cast<OrtExperimentalFnPtr>(&OrtExperimentalApis::NAME##_SinceV##VER)},

#include "onnxruntime_experimental_c_api.inc"

#undef ORT_EXPERIMENTAL_API

};

}  // namespace

// ---------------------------------------------------------------------------
// Lookup implementation (wired into OrtApi via ort_apis.h)
// ---------------------------------------------------------------------------

ORT_API(OrtExperimentalFnPtr, OrtApis::GetExperimentalFunction, _In_ const char* name) {
  if (name == nullptr) {
    return nullptr;
  }
  for (const auto& entry : kExperimentalFunctions) {
    if (std::strcmp(entry.name, name) == 0) {
      return entry.fn;
    }
  }
  return nullptr;
}
