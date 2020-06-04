// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "core/common/status.h"
#include "core/framework/framework_common.h"
#include "core/framework/ml_value.h"

struct OrtValue;
namespace onnxruntime {
class SessionState;
class TensorShape;
namespace logging {
class Logger;
}

class IExecutor {
 public:
  using CustomAllocator = std::function<Status(const TensorShape&, const OrtMemoryInfo&, OrtValue&, bool& allocated)>;

  IExecutor(const bool& terminate_flag) : terminate_flag_{terminate_flag} {};

  IExecutor(const bool& terminate_flag, const std::unordered_map<std::string, void*>& provider_run_options)
      : terminate_flag_{terminate_flag}, provider_run_options_{provider_run_options} {};

  virtual ~IExecutor() = default;

  /**
   * The lifetime of 'fetches' is limited by 'session_state'
   */
  common::Status Execute(const SessionState& session_state,
                         const std::vector<int>& feed_mlvalue_idxs,
                         const std::vector<OrtValue>& feeds,
                         const std::vector<int>& fetch_mlvalue_idxs,
                         std::vector<OrtValue>& fetches,
                         const logging::Logger& logger) {
    std::unordered_map<size_t, CustomAllocator> fetch_allocators;
    return Execute(session_state, feed_mlvalue_idxs, feeds, fetch_mlvalue_idxs, fetches, fetch_allocators, logger, nullptr);
  }

  // TODO: as fetch_allocators is optional, it should be a pointer instead of reference
  virtual common::Status Execute(const SessionState& session_state,
                                 const std::vector<int>& feed_mlvalue_idxs,
                                 const std::vector<OrtValue>& feeds,
                                 const std::vector<int>& fetch_mlvalue_idxs,
                                 std::vector<OrtValue>& fetches,
                                 // optional custom allocators. key is index in fetches
                                 const std::unordered_map<size_t, CustomAllocator>& fetch_allocators,
                                 const logging::Logger& logger,
                                 const AllocatorPtr custom_cpu_allocator) = 0;
                                 
 protected:
  const bool& terminate_flag_;
  const std::unordered_map<std::string, void*> provider_run_options_;
};
}  // namespace onnxruntime
