// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/op_identifier_utils.h"

#include "core/common/common.h"
#include "core/common/make_string.h"
#include "core/common/string_utils.h"
#include "core/common/parse_string.h"
#include "core/graph/graph.h"

namespace onnxruntime::utils {

static constexpr const char* kOpIdComponentDelimiter = ":";

OpIdentifier MakeOpId(std::string_view domain, std::string_view op_type,
                      ONNX_NAMESPACE::OperatorSetVersion since_version) {
  return MakeString(domain, kOpIdComponentDelimiter, op_type, kOpIdComponentDelimiter, since_version);
}

OpIdentifier MakeOpId(const Node& node) {
  return MakeOpId(node.Domain(), node.OpType(), node.SinceVersion());
}

Status SplitOpId(const OpIdentifier& op_id,
                 std::string_view& domain, std::string_view& op, ONNX_NAMESPACE::OperatorSetVersion& since_version) {
  const auto components = SplitString(op_id, kOpIdComponentDelimiter, true);
  ORT_RETURN_IF_NOT(components.size() == 3, "Expected 3 OpIdentifier components.");
  ORT_RETURN_IF_ERROR(ParseStringWithClassicLocale(components[2], since_version));
  domain = components[0];
  op = components[1];
  return Status::OK();
}

}  // namespace onnxruntime::utils
