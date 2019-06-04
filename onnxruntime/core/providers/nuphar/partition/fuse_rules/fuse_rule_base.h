// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/codegen/common/common.h"

namespace onnxruntime {
namespace nuphar {

using IsOpTypeSupportedFunc = std::function<bool(const Node& node)>;

class FuseRule {
 public:
  FuseRule() = default;

  virtual ~FuseRule() = default;

  // TODO: This interface is under change. We should modify it accordingly
  // when we add support to multiple fuse rules in Nuphar provider.
  virtual Status Fuse(const onnxruntime::GraphViewer& graph,
                      IsOpTypeSupportedFunc is_op_type_supported_func,
                      std::set<NodeIndex>& claimed_nodes,
                      std::vector<std::unique_ptr<ComputeCapability>>& result) = 0;
};

}  // namespace nuphar
}  // namespace onnxruntime
