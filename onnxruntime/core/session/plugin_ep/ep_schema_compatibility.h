// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <set>
#include <string>
#include <tuple>

#include "core/common/status.h"
#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime {
class Node;
namespace logging {
class Logger;
}

// Immutable result of negotiating a plugin factory's com.microsoft schema manifest
// against the schemas registered in this runtime.
class PluginEpSchemaCompatibility {
 public:
  using OperatorKey = std::tuple<std::string, std::string, int>;

  static Status Create(OrtEpFactory& factory, const logging::Logger& logger,
                       std::shared_ptr<const PluginEpSchemaCompatibility>& result);

  bool IsNegotiated() const { return is_negotiated_; }
  bool IsCompatible(const std::string& domain, const std::string& op_type, int since_version) const;
  bool IsCompatible(const Node& node) const;

 private:
  bool is_negotiated_ = false;
  std::set<OperatorKey> compatible_operators_;
};

}  // namespace onnxruntime
