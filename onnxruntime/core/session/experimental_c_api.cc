// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Runtime implementation for experimental C API functions.
// See onnxruntime_experimental_c_api.inc for the declaration list and lifecycle rules.

#include <cstring>
#include <array>
#include <limits>
#include <memory>

#include "core/common/common.h"
#include "core/framework/error_code_helper.h"
#include "core/framework/ep_context_options.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/compile_api.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/session/onnxruntime_experimental_c_api.h"
#include "core/session/ort_apis.h"
#include "core/session/plugin_ep/ep_api.h"

#if !defined(ORT_MINIMAL_BUILD)
#include "core/session/model_compilation_options.h"
#endif  // !defined(ORT_MINIMAL_BUILD)

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

ORT_API_STATUS_IMPL(OrtApi_SessionOptions_SetEpContextDataReadFunc_SinceV28, _Inout_ OrtSessionOptions* options,
                    _In_opt_ OrtReadNamedBufferFunc read_func, _In_opt_ void* state) {
  API_IMPL_BEGIN
  ORT_API_RETURN_IF(options == nullptr, ORT_INVALID_ARGUMENT, "'options' parameter must not be NULL");
  options->value.ep_context_data_read_func = read_func;
  options->value.ep_context_data_read_state = read_func != nullptr ? state : nullptr;
  options->value.ep_context_data_read_max_size = std::numeric_limits<size_t>::max();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtCompileApi_ModelCompilationOptions_SetEpContextDataWriteFunc_SinceV28,
                    _In_ OrtModelCompilationOptions* ort_model_compile_options,
                    _In_opt_ OrtWriteNamedBufferFunc write_func, _In_opt_ void* state) {
  return OrtCompileAPI::ModelCompilationOptions_SetEpContextDataWriteFunc(ort_model_compile_options, write_func,
                                                                          state);
}

ORT_API_STATUS_IMPL(OrtEpApi_SessionOptions_GetEpContextConfig_SinceV28,
                    _In_ const OrtSessionOptions* session_options,
                    _Outptr_ OrtEpContextConfig** config) {
  return OrtExecutionProviderApi::SessionOptionsGetEpContextConfig(session_options, config);
}

ORT_API(void, OrtEpApi_ReleaseEpContextConfig_SinceV28, _Frees_ptr_opt_ OrtEpContextConfig* config) {
  OrtExecutionProviderApi::ReleaseEpContextConfig(config);
}

ORT_API_STATUS_IMPL(OrtEpApi_EpContextConfig_GetEpContextDataReadFunc_SinceV28,
                    _In_ const OrtEpContextConfig* config,
                    _Out_ OrtReadNamedBufferFunc* read_func,
                    _Out_ void** state) {
  size_t ignored_max_data_size = 0;
  return OrtExecutionProviderApi::EpContextConfigGetEpContextDataReadFunc(config, read_func, state,
                                                                          &ignored_max_data_size);
}

ORT_API_STATUS_IMPL(OrtEpApi_EpContextConfig_GetEpContextDataWriteFunc_SinceV28,
                    _In_ const OrtEpContextConfig* config,
                    _Out_ OrtWriteNamedBufferFunc* write_func,
                    _Out_ void** state) {
  return OrtExecutionProviderApi::EpContextConfigGetEpContextDataWriteFunc(config, write_func, state);
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
