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

  common::Status Execute(const SessionState& session_state,
                         const NameMLValMap& feeds,
                         const std::vector<std::string>& output_names,
                         std::vector<MLValue>& fetches,
                         const logging::Logger& logger) {
    return Execute(session_state, feeds, output_names, fetches, {}, logger);
  }

  virtual common::Status Execute(const SessionState& session_state,
                                 const NameMLValMap& feeds,
                                 const std::vector<std::string>& output_names,
                                 std::vector<MLValue>& fetches,
                                 // optional custom allocators. key is index in fetches
                                 const std::unordered_map<size_t, CustomAllocator> fetch_allocators,
                                 const logging::Logger& logger) = 0;
};
}  // namespace onnxruntime
