// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <charconv>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>

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
inline std::optional<ApiPtrs> g_api_ptrs{};
inline uint32_t g_current_ort_api_version{};

// Strictly parse a "MAJOR.MINOR.PATCH" version string. Each component must be a non-empty sequence of decimal digits.
// Returns false for null/empty input, missing or extra components, non-numeric components, or any trailing characters
// (including a pre-release suffix).
inline bool TryParseVersion(const char* version_str, uint32_t& major, uint32_t& minor, uint32_t& patch) {
  if (version_str == nullptr || version_str[0] == '\0') {
    return false;
  }

  uint32_t values[3] = {0, 0, 0};
  const char* p = version_str;
  for (int i = 0; i < 3; ++i) {
    if (*p < '0' || *p > '9') {
      return false;
    }
    const char* end = p;
    while (*end >= '0' && *end <= '9') {
      ++end;
    }
    auto [next, ec] = std::from_chars(p, end, values[i]);
    if (ec != std::errc{} || next != end) {
      return false;
    }
    p = end;
    if (i < 2) {
      if (*p != '.') {
        return false;
      }
      ++p;
    } else if (*p != '\0') {
      // Last component must consume the whole string.
      return false;
    }
  }
  major = values[0];
  minor = values[1];
  patch = values[2];
  return true;
}

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
///
/// If `min_ort_version` is non-null, it is parsed as a strict "MAJOR.MINOR.PATCH" version string and compared against
/// the runtime ORT version reported by `ort_api_base->GetVersionString()`.
/// The runtime version must be at least `min_ort_version`.
///
/// If initialization fails, this function throws an exception.
/// </summary>
inline void ApiInit(const OrtApiBase* ort_api_base, const char* min_ort_version = nullptr) {
  static std::once_flag init_flag;
  std::call_once(init_flag, [&]() {
    const char* version_str = ort_api_base->GetVersionString();
    if (version_str == nullptr) {
      throw std::runtime_error("Failed to initialize EP API: ort_api_base->GetVersionString() returned null.");
    }

    uint32_t runtime_major = 0, runtime_minor = 0, runtime_patch = 0;
    if (!detail::TryParseVersion(version_str, runtime_major, runtime_minor, runtime_patch)) {
      throw std::runtime_error(std::string("Failed to initialize EP API: could not parse ORT version \"") +
                               version_str + "\". Expected format: \"MAJOR.MINOR.PATCH\".");
    }

    // If a minimum ORT version was specified by the EP, enforce it before any other checks.
    // This is also what defines the floor for the API version below.
    if (min_ort_version != nullptr) {
      uint32_t min_major = 0, min_minor = 0, min_patch = 0;
      if (!detail::TryParseVersion(min_ort_version, min_major, min_minor, min_patch)) {
        throw std::runtime_error(std::string("Failed to parse minimum required ORT version \"") +
                                 min_ort_version + "\". Expected format: \"MAJOR.MINOR.PATCH\".");
      }
      if (std::tie(runtime_major, runtime_minor, runtime_patch) < std::tie(min_major, min_minor, min_patch)) {
        throw std::runtime_error(std::string("ORT runtime version \"") + version_str +
                                 "\" is below the minimum required version \"" + min_ort_version + "\".");
      }
    }

    // Assume ORT versions of the form "1.<API version>.PATCH".
    if (runtime_major != 1) {
      throw std::runtime_error(std::string("Failed to initialize EP API: unsupported ORT major version in \"") +
                               version_str + "\" (expected major version 1).");
    }

    const uint32_t current_ort_api_version = runtime_minor;

    const OrtApi* ort_api = ort_api_base->GetApi(current_ort_api_version);
    if (!ort_api) {
      throw std::runtime_error(
          "Failed to initialize EP API: the current ORT version is \"" + std::string(version_str) +
          "\" but it does not support the parsed API version " + std::to_string(current_ort_api_version) + ".");
    }

    const OrtEpApi* ep_api = ort_api->GetEpApi();
    const OrtModelEditorApi* model_editor_api = ort_api->GetModelEditorApi();
    if (!ep_api || !model_editor_api) {
      throw std::runtime_error("Failed to initialize EP API: GetEpApi or GetModelEditorApi returned null.");
    }

    // Manual init for the C++ API
    Ort::InitApi(ort_api);

    // Initialize globals
    detail::g_api_ptrs.emplace(*ort_api, *ep_api, *model_editor_api);
    detail::g_current_ort_api_version = current_ort_api_version;
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
