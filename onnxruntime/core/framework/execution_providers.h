// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
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
*/
class ExecutionProviders {
 public:
  ExecutionProviders();

  ~ExecutionProviders();

  common::Status Add(const std::string& provider_id, const std::shared_ptr<IExecutionProvider>& p_exec_provider);

#ifdef _WIN32

  void EtwProvidersCallback(LPCGUID SourceId,
                            ULONG IsEnabled,
                            UCHAR Level,
                            ULONGLONG MatchAnyKeyword,
                            ULONGLONG MatchAllKeyword,
                            PEVENT_FILTER_DESCRIPTOR FilterData,
                            PVOID CallbackContext);

  void LogProviderOptions(const std::string& provider_id, const ProviderOptions& providerOptions,
                          bool captureState);
#endif

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

  bool GetCpuProviderWasImplicitlyAdded() const { return cpu_execution_provider_was_implicitly_added_; }

  void SetCpuProviderWasImplicitlyAdded(bool cpu_execution_provider_was_implicitly_added) {
    cpu_execution_provider_was_implicitly_added_ = cpu_execution_provider_was_implicitly_added;
  }

 private:
  // Some compilers emit incomprehensive output if this is allowed
  // with a container that has unique_ptr or something move-only.
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(ExecutionProviders);

  std::vector<std::shared_ptr<IExecutionProvider>> exec_providers_;
  std::vector<std::string> exec_provider_ids_;
  ProviderOptionsMap exec_provider_options_;

  // maps for fast lookup of an index into exec_providers_
  std::unordered_map<std::string, size_t> provider_idx_map_;

  // Whether the CPU provider was implicitly added to a session for fallback (true),
  // or whether it was explicitly added by the caller.
  bool cpu_execution_provider_was_implicitly_added_ = false;

#ifdef _WIN32
  std::string etw_callback_key_;
#endif
};
}  // namespace onnxruntime
