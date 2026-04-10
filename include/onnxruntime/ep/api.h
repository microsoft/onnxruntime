// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <charconv>
#include <cstring>
#include <mutex>
#include <optional>
#include <stdexcept>

#pragma push_macro("ORT_API_MANUAL_INIT")
#undef ORT_API_MANUAL_INIT
#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#pragma pop_macro("ORT_API_MANUAL_INIT")

namespace onnxruntime {
namespace ep {

struct ApiPtrs {
  ApiPtrs(const OrtApi& ort_, const OrtEpApi& ep_, const OrtModelEditorApi& model_editor_)
      : ort(ort_), ep(ep_), model_editor(model_editor_) {}
  const OrtApi& ort;
  const OrtEpApi& ep;
  const OrtModelEditorApi& model_editor;
};

namespace detail {
inline std::optional<ApiPtrs> g_api_ptrs;

inline bool TryGetAPIVersionFromVersionString(const char* version_str, uint32_t& api_version) {
  // A valid version string should always be in the format of "1.{API_VERSION}.*".
  if (!version_str || version_str[0] != '1' || version_str[1] != '.') {
    return false;
  }
  const char* begin = version_str + 2;
  const char* end = std::strchr(begin, '.');
  if (!end) {
    return false;
  }
  uint32_t version = 0;
  auto [ptr, ec] = std::from_chars(begin, end, version);
  if (ec != std::errc{} || ptr != end) {
    return false;
  }
  api_version = version;
  return true;
}

inline uint32_t g_current_ort_api_version{};

}  // namespace detail

/// <summary>
/// Get the global instance of ApiPtrs.
/// </summary>
inline const ApiPtrs& Api() {
  if (!detail::g_api_ptrs.has_value()) {
    throw std::logic_error("onnxruntime::ep::Api() called before ApiInit().");
  }
  return *detail::g_api_ptrs;
}

/// <summary>
/// Initialize the EP API pointers and global OrtEnv if not already done.
/// Thread-safe via std::call_once.
/// </summary>
inline void ApiInit(const OrtApiBase* ort_api_base) {
  static std::once_flag init_flag;
  std::call_once(init_flag, [&]() {
    // The following initialization process is composed of 3 steps:
    // 1) Get the ORT API version string
    // 2) Try to parse the ORT API version from the version string. If parsing fails, we assume the version is 24.
    // 3) Get the ORT API for the parsed version and initialize the global API instance with it.
    constexpr uint32_t ORT_BASE_API_VERSION = 24;
    const char* version_str = ort_api_base->GetVersionString();
    if (!version_str) {
      version_str = "unknown";
    }
    uint32_t current_ort_version = 0;
    if (!detail::TryGetAPIVersionFromVersionString(version_str, current_ort_version)) {
      // If we fail to parse the version string, we can still try to get the API for the base version and hope it works.
      current_ort_version = ORT_BASE_API_VERSION;
    }
    if (current_ort_version < ORT_BASE_API_VERSION) {
      throw std::runtime_error("Failed to initialize EP API: the minimum required ORT API version is " + std::to_string(ORT_BASE_API_VERSION) +
                               ", but the current version is \"" + version_str +
                               "\" (parsed API version: " + std::to_string(current_ort_version) + ").");
    }

    const OrtApi* ort_api = ort_api_base->GetApi(current_ort_version);
    if (!ort_api) {
      throw std::runtime_error("Failed to initialize EP API: the current ORT version is \"" + std::string(version_str) +
                               "\" but it does not support the parsed API version " + std::to_string(current_ort_version) + ".");
    }

    detail::g_current_ort_api_version = current_ort_version;

    const OrtEpApi* ep_api = ort_api->GetEpApi();
    const OrtModelEditorApi* model_editor_api = ort_api->GetModelEditorApi();
    if (!ep_api || !model_editor_api) {
      throw std::runtime_error("Failed to initialize EP API: GetEpApi or GetModelEditorApi returned null.");
    }

    // Manual init for the C++ API
    Ort::InitApi(ort_api);

    // Initialize the global API instance
    detail::g_api_ptrs.emplace(*ort_api, *ep_api, *model_editor_api);
  });
}

/// <summary>
/// Get the current ORT API version that the EP API has been initialized with.
///
/// This function should be called after ApiInit() to get the actual API version.
/// </summary>
inline uint32_t CurrentOrtApiVersion() {
  if (!detail::g_api_ptrs.has_value()) {
    throw std::logic_error("onnxruntime::ep::CurrentOrtApiVersion() called before ApiInit().");
  }
  return detail::g_current_ort_api_version;
}

}  // namespace ep
}  // namespace onnxruntime
