// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Runtime implementation for experimental C API functions.
// See onnxruntime_experimental_c_api.inc for the declaration list and lifecycle rules.

#include <cstring>
#include <array>
#include <memory>

#include "core/common/common.h"
#include "core/framework/error_code_helper.h"
#include "core/framework/ep_context_options.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/session/onnxruntime_experimental_c_api.h"
#include "core/session/ort_apis.h"

#if !defined(ORT_MINIMAL_BUILD)
#include "core/session/model_compilation_options.h"
#endif  // !defined(ORT_MINIMAL_BUILD)

// Backing definition of the OrtEpContextConfig handle used by the experimental OrtEpApi_* EPContext data functions.
// Holds copies of the application's EPContext read/write callbacks and opaque state extracted from an
// OrtSessionOptions instance.
struct OrtEpContextConfig {
  OrtWriteNamedBufferFunc write_func = nullptr;
  void* write_state = nullptr;
  OrtReadNamedBufferFunc read_func = nullptr;
  void* read_state = nullptr;
};

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

  // Passing a null read_func clears any previously set callback. Clear the state too so a stale state pointer is
  // never paired with a missing callback.
  options->value.ep_context_data_read_func = read_func;
  options->value.ep_context_data_read_state = read_func != nullptr ? state : nullptr;
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtCompileApi_ModelCompilationOptions_SetEpContextDataWriteFunc_SinceV28,
                    _In_ OrtModelCompilationOptions* ort_model_compile_options,
                    _In_opt_ OrtWriteNamedBufferFunc write_func, _In_opt_ void* state) {
  API_IMPL_BEGIN
#if !defined(ORT_MINIMAL_BUILD)
  ORT_API_RETURN_IF(ort_model_compile_options == nullptr, ORT_INVALID_ARGUMENT, "OrtModelCompilationOptions is NULL");

  // A null write_func clears any previously set callback (symmetric with OrtApi_SessionOptions_SetEpContextDataReadFunc
  // and consistent with calling this multiple times to overwrite the callback).
  auto model_compile_options = reinterpret_cast<onnxruntime::ModelCompilationOptions*>(ort_model_compile_options);
  model_compile_options->SetEpContextDataWriteFunc(write_func, state);
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(ort_model_compile_options);
  ORT_UNUSED_PARAMETER(write_func);
  ORT_UNUSED_PARAMETER(state);
  return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, "Compile API is not supported in this build");
#endif  // !defined(ORT_MINIMAL_BUILD)
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtEpApi_SessionOptions_GetEpContextConfig_SinceV28,
                    _In_ const OrtSessionOptions* session_options,
                    _Outptr_ OrtEpContextConfig** config) {
  API_IMPL_BEGIN
  ORT_API_RETURN_IF(session_options == nullptr, ORT_INVALID_ARGUMENT, "OrtSessionOptions is NULL");
  ORT_API_RETURN_IF(config == nullptr, ORT_INVALID_ARGUMENT, "Output OrtEpContextConfig is NULL");

  auto ep_context_config = std::make_unique<OrtEpContextConfig>();
  if (const auto* write_config = session_options->value.ep_context_gen_options.TryGetEpContextDataWriteFunc()) {
    ep_context_config->write_func = write_config->write_func;
    ep_context_config->write_state = write_config->state;
  }
  ep_context_config->read_func = session_options->value.ep_context_data_read_func;
  ep_context_config->read_state = session_options->value.ep_context_data_read_state;

  *config = ep_context_config.release();
  return nullptr;
  API_IMPL_END
}

ORT_API(void, OrtEpApi_ReleaseEpContextConfig_SinceV28, _Frees_ptr_opt_ OrtEpContextConfig* config) {
  delete config;
}

ORT_API_STATUS_IMPL(OrtEpApi_EpContextConfig_GetEpContextDataReadFunc_SinceV28,
                    _In_ const OrtEpContextConfig* config,
                    _Out_ OrtReadNamedBufferFunc* read_func,
                    _Out_ void** state) {
  API_IMPL_BEGIN
  ORT_API_RETURN_IF(config == nullptr, ORT_INVALID_ARGUMENT, "OrtEpContextConfig is NULL");
  ORT_API_RETURN_IF(read_func == nullptr, ORT_INVALID_ARGUMENT, "Output read_func is NULL");
  ORT_API_RETURN_IF(state == nullptr, ORT_INVALID_ARGUMENT, "Output state is NULL");

  *read_func = config->read_func;
  *state = config->read_func != nullptr ? config->read_state : nullptr;
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtEpApi_EpContextConfig_GetEpContextDataWriteFunc_SinceV28,
                    _In_ const OrtEpContextConfig* config,
                    _Out_ OrtWriteNamedBufferFunc* write_func,
                    _Out_ void** state) {
  API_IMPL_BEGIN
  ORT_API_RETURN_IF(config == nullptr, ORT_INVALID_ARGUMENT, "OrtEpContextConfig is NULL");
  ORT_API_RETURN_IF(write_func == nullptr, ORT_INVALID_ARGUMENT, "Output write_func is NULL");
  ORT_API_RETURN_IF(state == nullptr, ORT_INVALID_ARGUMENT, "Output state is NULL");

  *write_func = config->write_func;
  *state = config->write_func != nullptr ? config->write_state : nullptr;
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
