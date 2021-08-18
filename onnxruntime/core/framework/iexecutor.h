// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>

#include "core/common/status.h"
#include "core/framework/framework_common.h"
#include "core/framework/ort_value.h"

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
    return Execute(session_state, feed_mlvalue_idxs, feeds, fetch_mlvalue_idxs, fetches, fetch_allocators, logger);
  }

  // TODO: as fetch_allocators is optional, it should be a pointer instead of reference
  virtual common::Status Execute(const SessionState& session_state, const std::vector<int>& feed_mlvalue_idxs,
                                 const std::vector<OrtValue>& feeds, const std::vector<int>& fetch_mlvalue_idxs,
                                 std::vector<OrtValue>& fetches,
                                 // optional custom allocators. key is index in fetches
                                 const std::unordered_map<size_t, CustomAllocator>& fetch_allocators,
                                 const logging::Logger& logger) = 0;
};
}  // namespace onnxruntime
