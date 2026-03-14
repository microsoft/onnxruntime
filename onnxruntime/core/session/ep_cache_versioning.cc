// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/ep_cache_versioning.h"

#include <array>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/framework/config_options.h"
#include "core/framework/provider_options.h"
#ifdef ORT_VERSION
#include "onnxruntime_config.h"
#endif

namespace onnxruntime {

namespace {

// Option keys (as used in config and provider options) that are EP cache paths.
// When session.ep_cache_use_ort_version is "1", these are suffixed with ORT version.
// Format: (provider_name_lowercase, option_key).
const std::array<std::pair<std::string, std::string>, 5> kEpCachePathOptions = {
    std::pair<std::string, std::string>{"coreml", "ModelCacheDirectory"},
    std::pair<std::string, std::string>{"tensorrt", "trt_engine_cache_path"},
    std::pair<std::string, std::string>{"tensorrt", "trt_timing_cache_path"},
    std::pair<std::string, std::string>{"migraphx", "migraphx_model_cache_dir"},
    std::pair<std::string, std::string>{"nvtensorrtrtx", "nv_runtime_cache_path"},
};

bool IsEpCachePathOption(const std::string& provider_lower, const std::string& option_key) {
  for (const auto& cache_option : kEpCachePathOptions) {
    if (cache_option.first == provider_lower && cache_option.second == option_key) {
      return true;
    }
  }
  return false;
}

std::string GetVersionedCachePath(const std::string& path) {
#ifdef ORT_VERSION
  if (path.empty()) return path;
  std::filesystem::path p(path);
  p /= ORT_VERSION;
  return p.string();
#else
  return path;
#endif
}

std::string GetLowercaseString(const std::string& s) {
  std::string result;
  result.reserve(s.size());
  for (char c : s) {
    result.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
  }
  return result;
}

}  // namespace

void ApplyEpCacheVersionToConfigOptions(ConfigOptions& config_options) {
  if (config_options.GetConfigOrDefault(kOrtSessionOptionsEpCacheUseOrtVersion, "0") != "1") {
    return;
  }
#ifdef ORT_VERSION
  const std::string prefix = "ep.";
  std::vector<std::pair<std::string, std::string>> to_update;
  for (const auto& kv : config_options.GetConfigOptionsMap()) {
    const std::string& key = kv.first;
    if (key.size() <= prefix.size() || key.compare(0, prefix.size(), prefix) != 0) {
      continue;
    }
    size_t dot = key.find('.', prefix.size());
    if (dot == std::string::npos) continue;
    std::string provider_lower = GetLowercaseString(key.substr(prefix.size(), dot - prefix.size()));
    std::string option_key = key.substr(dot + 1);
    if (!IsEpCachePathOption(provider_lower, option_key)) continue;
    const std::string& value = kv.second;
    if (value.empty()) continue;
    to_update.push_back({key, GetVersionedCachePath(value)});
  }
  for (const auto& p : to_update) {
    (void)config_options.AddConfigEntry(p.first.c_str(), p.second.c_str());
  }
#endif
}

ProviderOptions GetProviderOptionsWithVersionedCachePaths(
    const ProviderOptions& provider_options,
    const ConfigOptions& config_options,
    const char* provider_name) {
  ProviderOptions result = provider_options;
  if (config_options.GetConfigOrDefault(kOrtSessionOptionsEpCacheUseOrtVersion, "0") != "1") {
    return result;
  }
  std::string provider_lower = GetLowercaseString(provider_name);
  for (auto& kv : result) {
    if (!IsEpCachePathOption(provider_lower, kv.first)) continue;
    if (kv.second.empty()) continue;
    kv.second = GetVersionedCachePath(kv.second);
  }
  return result;
}

}  // namespace onnxruntime
