// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>

#include "core/common/status.h"
#include "core/framework/framework_common.h"
#include "core/framework/ml_value.h"

namespace onnxruntime {

class SessionState;
namespace logging {
class Logger;
}

class IExecutor {
 public:
  virtual ~IExecutor() = default;

  virtual common::Status Execute(const SessionState& session_state,
                                 const NameMLValMap& feeds,
                                 const std::vector<std::string>& output_names,
                                 std::vector<MLValue>& fetches,
                                 const logging::Logger& logger) = 0;
};
}  // namespace onnxruntime
