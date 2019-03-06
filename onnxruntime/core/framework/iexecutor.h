// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>

#include "core/common/status.h"
#include "core/framework/framework_common.h"
#include "core/framework/ml_value.h"

namespace onnxruntime {

class MLValue;
class SessionState;
class TensorShape;
namespace logging {
class Logger;
}

class IExecutor {
 public:
  using CustomAllocator = std::function<Status(const TensorShape&, MLValue&)>;

  virtual ~IExecutor() = default;

  /**
   * The lifetime of 'fetches' is limited by 'session_state'
   */
  common::Status Execute(const SessionState& session_state,
                         const std::vector<int>& feed_mlvalue_idxs,
                         const std::vector<MLValue>& feeds,
                         const std::vector<int>& fetch_mlvalue_idxs,
                         std::vector<MLValue>& fetches,
                         const logging::Logger& logger) {
    return Execute(session_state, feed_mlvalue_idxs, feeds, fetch_mlvalue_idxs, fetches, {}, logger);
  }

  virtual common::Status Execute(const SessionState& session_state,
                                 const std::vector<int>& feed_mlvalue_idxs,
                                 const std::vector<MLValue>& feeds,
                                 const std::vector<int>& fetch_mlvalue_idxs,
                                 std::vector<MLValue>& fetches,
                                 // optional custom allocators. key is index in fetches
                                 const std::unordered_map<size_t, CustomAllocator> fetch_allocators,
                                 const logging::Logger& logger) = 0;
};
}  // namespace onnxruntime
