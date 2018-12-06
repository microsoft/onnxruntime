// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/onnx_protobuf.h"
#include "core/graph/graph.h"

namespace onnxruntime {

namespace utils {
  bool IsSupportedOptypeDomainAndVersion(const Node& node, const std::string& op_type, ONNX_NAMESPACE::OperatorSetVersion version);
}

}