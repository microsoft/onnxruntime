// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/framework/execution_provider.h"
#include "core/graph/graph_viewer.h"
#include "core/common/logging/logging.h"
#ifdef _WIN32
#include <Windows.h>
#include <winmeta.h>
#include <evntrace.h>
#include "core/platform/tracing.h"
#include "core/platform/windows/telemetry.h"
#endif

namespace onnxruntime {

/**
Class for managing lookup of the execution providers in a session.

Add() must not run concurrently with accessors or iteration, except for
GetProviderOptionsSnapshot(), which is explicitly synchronized for ETW capture-state callbacks.
*/
class ExecutionProviders {
 public:
  ExecutionProviders() {
#ifdef _WIN32
    // Register callback for ETW capture state (rundown)
    etw_callback_ = onnxruntime::WindowsTelemetry::EtwInternalCallback(
        [this](
            LPCGUID SourceId,
            ULONG IsEnabled,
            UCHAR Level,
            ULONGLONG MatchAnyKeyword,
            ULONGLONG MatchAllKeyword,
            PEVENT_FILTER_DESCRIPTOR FilterData,
            PVOID CallbackContext) {
          (void)SourceId;
          (void)Level;
          (void)MatchAnyKeyword;
          (void)MatchAllKeyword;
          (void)FilterData;
          (void)CallbackContext;

          // Check if this callback is for capturing state
          if ((IsEnabled == EVENT_CONTROL_CODE_CAPTURE_STATE) &&
              ((MatchAnyKeyword & static_cast<ULONGLONG>(onnxruntime::logging::ORTTraceLoggingKeyword::Session)) != 0)) {
            for (const auto& [provider_id, options] : GetProviderOptionsSnapshot()) {
              LogProviderOptions(provider_id, options, true);
            }
          }
        });
    WindowsTelemetry::RegisterInternalCallback(etw_callback_);
#endif
  }

  ~ExecutionProviders() {
#ifdef _WIN32
    WindowsTelemetry ::UnregisterInternalCallback(etw_callback_);
#endif
  }

  common::Status
  Add(const std::string& provider_id, const std::shared_ptr<IExecutionProvider>& p_exec_provider) {
    // A null provider would crash later when we dereference it (e.g. GetProviderOptions()).
    // Fail with a clear error instead so the caller can diagnose the missing provider.
    if (p_exec_provider == nullptr) {
      auto status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Provider ", provider_id, " is null and cannot be registered.");
      LOGS_DEFAULT(ERROR) << status.ErrorMessage();
      return status;
    }

    const auto check_provider_not_registered = [&]() -> common::Status {
      if (provider_idx_map_.find(provider_id) != provider_idx_map_.end()) {
        auto status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Provider ", provider_id, " has already been registered.");
        LOGS_DEFAULT(ERROR) << status.ErrorMessage();
        return status;
      }

      return Status::OK();
    };

    {
      std::lock_guard<std::mutex> lock(exec_providers_mutex_);
      ORT_RETURN_IF_ERROR(check_provider_not_registered());
    }

    ProviderOptions providerOptions = p_exec_provider->GetProviderOptions();

    {
      std::lock_guard<std::mutex> lock(exec_providers_mutex_);
      // make sure there are no issues before we change any internal data structures
      ORT_RETURN_IF_ERROR(check_provider_not_registered());

      // index that provider will have after insertion
      auto new_provider_idx = exec_providers_.size();

      ORT_IGNORE_RETURN_VALUE(provider_idx_map_.insert({provider_id, new_provider_idx}));

      // update execution provider options
      exec_provider_options_[provider_id] = providerOptions;
      exec_provider_ids_.push_back(provider_id);
      exec_providers_.push_back(p_exec_provider);
    }

#ifdef _WIN32
    LogProviderOptions(provider_id, providerOptions, false);
#endif
    return Status::OK();
  }

  const IExecutionProvider* Get(const onnxruntime::Node& node) const {
    return Get(node.GetExecutionProviderType());
  }

  const IExecutionProvider* Get(onnxruntime::ProviderType provider_id) const {
    auto it = provider_idx_map_.find(provider_id);
    if (it == provider_idx_map_.end()) {
      return nullptr;
    }

    return exec_providers_[it->second].get();
  }

  IExecutionProvider* Get(onnxruntime::ProviderType provider_id) {
    auto it = provider_idx_map_.find(provider_id);
    if (it == provider_idx_map_.end()) {
      return nullptr;
    }

    return exec_providers_[it->second].get();
  }

  bool Empty() const { return exec_providers_.empty(); }

  size_t NumProviders() const { return exec_providers_.size(); }

  using const_iterator = typename std::vector<std::shared_ptr<IExecutionProvider>>::const_iterator;
  const_iterator begin() const noexcept { return exec_providers_.cbegin(); }
  const_iterator end() const noexcept { return exec_providers_.cend(); }

  const std::vector<std::string>& GetIds() const { return exec_provider_ids_; }
  const ProviderOptionsMap& GetAllProviderOptions() const { return exec_provider_options_; }

  using ProviderOptionsSnapshot = std::vector<std::pair<std::string, ProviderOptions>>;

  ProviderOptionsSnapshot GetProviderOptionsSnapshot() const {
    std::lock_guard<std::mutex> lock(exec_providers_mutex_);
    ProviderOptionsSnapshot provider_options_snapshot;
    provider_options_snapshot.reserve(exec_provider_ids_.size());
    for (const auto& provider_id : exec_provider_ids_) {
      auto it = exec_provider_options_.find(provider_id);
      if (it != exec_provider_options_.end()) {
        provider_options_snapshot.emplace_back(provider_id, it->second);
      }
    }

    return provider_options_snapshot;
  }

  bool GetCpuProviderWasImplicitlyAdded() const { return cpu_execution_provider_was_implicitly_added_; }

  void SetCpuProviderWasImplicitlyAdded(bool cpu_execution_provider_was_implicitly_added) {
    cpu_execution_provider_was_implicitly_added_ = cpu_execution_provider_was_implicitly_added;
  }

 private:
  // Some compilers emit incomprehensive output if this is allowed
  // with a container that has unique_ptr or something move-only.
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(ExecutionProviders);

  // Synchronizes provider registration with ETW capture-state snapshots.
  mutable std::mutex exec_providers_mutex_;

  void LogProviderOptions(const std::string& provider_id, const ProviderOptions& options, bool capture_state) {
    const Env& env = Env::Default();
    // Convert ProviderOptions to string for telemetry logging
    std::string provider_options_str;
    for (const auto& config_pair : options) {
      if (!provider_options_str.empty()) {
        provider_options_str += ",";
      }
      provider_options_str += config_pair.first + ":" + config_pair.second;
    }
    env.GetTelemetryProvider().LogProviderOptions(provider_id, provider_options_str, capture_state);
  }

  std::vector<std::shared_ptr<IExecutionProvider>> exec_providers_;
  std::vector<std::string> exec_provider_ids_;
  ProviderOptionsMap exec_provider_options_;

  // maps for fast lookup of an index into exec_providers_
  std::unordered_map<std::string, size_t> provider_idx_map_;

  // Whether the CPU provider was implicitly added to a session for fallback (true),
  // or whether it was explicitly added by the caller.
  bool cpu_execution_provider_was_implicitly_added_ = false;

#ifdef _WIN32
  WindowsTelemetry::EtwInternalCallback etw_callback_;
#endif
};
}  // namespace onnxruntime
