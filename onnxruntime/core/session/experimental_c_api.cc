// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Runtime implementation for experimental C API functions.
// See onnxruntime_experimental_c_api.inc for the declaration list and lifecycle rules.

#include <cstring>
#include <array>

#include "core/common/common.h"
#include "core/framework/error_code_helper.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/session/onnxruntime_experimental_c_api.h"
#include "core/session/ort_apis.h"
#include "core/session/abi_session_options_impl.h"
#include "core/platform/env.h"

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

#if !defined(ORT_MINIMAL_BUILD)
ORT_API_STATUS_IMPL(OrtApi_SessionOptionsSetExternalInitializerFileHandles_SinceV32,
                    OrtSessionOptions* options, const char* const* names,
                    const OrtFileHandle* handles, size_t count) {
#if (defined(_WIN32) || defined(__linux__) || defined(__ANDROID__) || defined(__APPLE__)) && !defined(DISABLE_EXTERNAL_INITIALIZERS)
  API_IMPL_BEGIN
  ORT_API_RETURN_IF(options == nullptr || (count != 0 && (names == nullptr || handles == nullptr)),
                    ORT_INVALID_ARGUMENT, "Invalid external initializer file-source arguments");
  ORT_API_RETURN_IF(!options->value.external_initializer_files_mmap.empty(),
                    ORT_INVALID_ARGUMENT, "Cannot combine external initializer buffers and file sources");
  auto sources = std::make_shared<std::unordered_map<onnxruntime::PathString,
                                                     std::shared_ptr<onnxruntime::RandomAccessFile>>>();
  for (size_t i = 0; i < count; ++i) {
    ORT_API_RETURN_IF(names[i] == nullptr, ORT_INVALID_ARGUMENT, "Null initializer file name");
    const std::string_view name{names[i]};
    ORT_API_RETURN_IF(name.empty() || name == "." || name == ".." ||
                          name.find_first_of("/\\:") != std::string_view::npos,
                      ORT_INVALID_ARGUMENT, "Initializer file names must be flat relative names");
    const auto path = onnxruntime::ToPathString(name);
    ORT_API_RETURN_IF(sources->count(path) != 0, ORT_INVALID_ARGUMENT,
                      "Duplicate initializer file name");
    std::shared_ptr<onnxruntime::RandomAccessFile> file;
    const auto status = onnxruntime::Env::Default().DuplicateReadOnlyFileHandle(handles[i], file);
    if (!status.IsOK()) {
      return onnxruntime::ToOrtStatus(status);
    }
    sources->emplace(path, std::move(file));
  }
  options->value.external_initializer_file_sources = std::move(sources);
  return nullptr;
  API_IMPL_END
#else
  ORT_UNUSED_PARAMETER(options);
  ORT_UNUSED_PARAMETER(names);
  ORT_UNUSED_PARAMETER(handles);
  ORT_UNUSED_PARAMETER(count);
  return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, "File-source loading requires a supported full build");
#endif
}
#endif

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
