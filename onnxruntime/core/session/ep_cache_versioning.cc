// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/ep_cache_versioning.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <mutex>
#include <string>
#include <unordered_map>
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

// Registry of EP cache path option keys, keyed by provider name (lowercased).
std::unordered_map<std::string, EpCachePathOptionKeys>& EpCachePathOptionsRegistry() {
  // Function-local static to avoid static initialization order issues across translation units.
  static std::unordered_map<std::string, EpCachePathOptionKeys> registry;
  return registry;
}

std::mutex& EpCachePathOptionsMutex() {
  // Function-local static to avoid static initialization order issues across translation units.
  static std::mutex mutex;
  return mutex;
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

void RegisterEpCachePathOptions(const char* provider_name,
                                std::initializer_list<const char*> option_keys) {
  if (provider_name == nullptr) {
    return;
  }

  const std::string provider_lower = GetLowercaseString(provider_name);

  std::lock_guard<std::mutex> lock(EpCachePathOptionsMutex());
  auto& registered_keys = EpCachePathOptionsRegistry()[provider_lower];
  for (const char* key : option_keys) {
    if (key == nullptr) {
      continue;
    }
    registered_keys.emplace_back(key);
  }
}

static const EpCachePathOptionKeys* TryGetEpCachePathOptionKeys(const std::string& provider_lower) {
  std::lock_guard<std::mutex> lock(EpCachePathOptionsMutex());
  auto& registry = EpCachePathOptionsRegistry();
  auto it = registry.find(provider_lower);
  if (it == registry.end()) {
    return nullptr;
  }
  return &it->second;
}

void ApplyEpCacheVersionToConfigOptions(ConfigOptions& config_options) {
  const std::string mode = config_options.GetConfigOrDefault(kOrtSessionOptionsEpCacheUseOrtVersion, "0");
  const bool ep_cache_versioning_enabled = (mode == "1");
  if (!ep_cache_versioning_enabled) {
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
    const size_t dot = key.find('.', prefix.size());
    if (dot == std::string::npos) {
      continue;
    }
    const std::string provider_lower = GetLowercaseString(key.substr(prefix.size(), dot - prefix.size()));
    const std::string option_key = key.substr(dot + 1);
    const EpCachePathOptionKeys* cache_keys = TryGetEpCachePathOptionKeys(provider_lower);
    if (cache_keys == nullptr) {
      continue;
    }
    if (std::find(cache_keys->begin(), cache_keys->end(), option_key) == cache_keys->end()) {
      continue;
    }
    const std::string& value = kv.second;
    if (value.empty()) {
      continue;
    }
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
  const std::string mode = config_options.GetConfigOrDefault(kOrtSessionOptionsEpCacheUseOrtVersion, "0");
  const bool ep_cache_versioning_enabled = (mode == "1");
  if (!ep_cache_versioning_enabled) {
    return result;
  }
  if (provider_name == nullptr) {
    return result;
  }
  const std::string provider_lower = GetLowercaseString(provider_name);
  const EpCachePathOptionKeys* cache_keys = TryGetEpCachePathOptionKeys(provider_lower);
  if (cache_keys == nullptr) {
    return result;
  }
  for (auto& kv : result) {
    if (std::find(cache_keys->begin(), cache_keys->end(), kv.first) == cache_keys->end()) {
      continue;
    }
    if (kv.second.empty()) {
      continue;
    }
    kv.second = GetVersionedCachePath(kv.second);
  }
  return result;
}

}  // namespace onnxruntime
