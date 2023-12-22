// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

// #include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/framework/execution_provider.h"
#include "core/graph/graph_viewer.h"
#include "core/common/logging/logging.h"
#ifdef _WIN32
#include "core/platform/tracing.h"
#endif

namespace onnxruntime {

/**
Class for managing lookup of the execution providers in a session.
*/
class ExecutionProviders {
 public:
  ExecutionProviders() = default;

  common::Status Add(const std::string& provider_id, const std::shared_ptr<IExecutionProvider>& p_exec_provider) {
    // make sure there are no issues before we change any internal data structures
    if (provider_idx_map_.find(provider_id) != provider_idx_map_.end()) {
      auto status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Provider ", provider_id, " has already been registered.");
      LOGS_DEFAULT(ERROR) << status.ErrorMessage();
      return status;
    }

    // index that provider will have after insertion
    auto new_provider_idx = exec_providers_.size();

    ORT_IGNORE_RETURN_VALUE(provider_idx_map_.insert({provider_id, new_provider_idx}));

    // update execution provider options
    auto providerOptions = p_exec_provider->GetProviderOptions();
    exec_provider_options_[provider_id] = providerOptions;

#ifdef _WIN32
    for (const auto& config_pair : providerOptions) {
      TraceLoggingWrite(
          telemetry_provider_handle,
          "ProviderOptions",
          TraceLoggingString(provider_id.c_str(), "ProviderId"),
          TraceLoggingString(config_pair.first.c_str(), "Key"),
          TraceLoggingString(config_pair.second.c_str(), "Value"));
    }
#endif

    exec_provider_ids_.push_back(provider_id);
    exec_providers_.push_back(p_exec_provider);
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
};
}  // namespace onnxruntime
