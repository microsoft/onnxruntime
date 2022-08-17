// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <string_view>

#include "onnx/defs/schema.h"

#include "core/common/status.h"
#include "core/graph/basic_types.h"

namespace onnxruntime {

class Node;

using common::Status;

namespace utils {

OpIdentifier MakeOpId(std::string_view domain, std::string_view op_type,
                      ONNX_NAMESPACE::OperatorSetVersion since_version);

inline OpIdentifier MakeOpId(const ONNX_NAMESPACE::OpSchema& op_schema) {
  return MakeOpId(op_schema.domain(), op_schema.Name(), op_schema.SinceVersion());
}

OpIdentifier MakeOpId(const Node& node);

Status SplitOpId(const OpIdentifier& op_id,
                 std::string_view& domain, std::string_view& op, ONNX_NAMESPACE::OperatorSetVersion& since_version);

}  // namespace utils

}  // namespace onnxruntime
