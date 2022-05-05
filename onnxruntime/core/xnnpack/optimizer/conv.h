// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <onnx/onnx_pb.h>

#include "core/xnnpack/optimizer/layout_helper.h"

namespace onnxruntime {
namespace xnnpack {
class ConvNodeProcessor : public NodeProcessor {
 public:
  ConvNodeProcessor(const Node& node, const std::unordered_set<const NodeArg*>& graph_const_values)
      : NodeProcessor(node, graph_const_values) {}
  Status Generate(std::unique_ptr<::ONNX_NAMESPACE::GraphProto>& output_graph) override;
};
}  // namespace xnnpack

}  // namespace onnxruntime