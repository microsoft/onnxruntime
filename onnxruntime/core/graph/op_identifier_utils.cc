// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/op_identifier_utils.h"

#include "core/common/make_string.h"
#include "core/graph/graph.h"

namespace onnxruntime {

static constexpr auto kOpIdComponentDelimiter = ':';

OpIdentifier MakeOpId(std::string_view domain, std::string_view op_type,
                      ONNX_NAMESPACE::OperatorSetVersion since_version) {
  return MakeString(domain, kOpIdComponentDelimiter, op_type, kOpIdComponentDelimiter, since_version);
}

OpIdentifier MakeOpId(const Node& node) {
  return MakeOpId(node.Domain(), node.OpType(), node.SinceVersion());
}

}  // namespace onnxruntime
