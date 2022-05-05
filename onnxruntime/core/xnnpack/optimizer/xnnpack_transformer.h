// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/execution_provider.h"
#include "core/optimizer/graph_transformer.h"
#include "core/xnnpack/optimizer/layout_helper.h"
namespace onnxruntime {

/**
@Class XNNPackTransformer

Transformer a normal graph to XnnPack nodes
*/
class XNNPackTransformer : public GraphTransformer {
 public:
  explicit XNNPackTransformer(AllocatorPtr cpu_allocator) noexcept;

 private:
  using CreateNodeProcessorFn =
      std::function<xnnpack::NodeProcessor*(const Node& node, const std::unordered_set<const NodeArg*>& graph_const_values)>;
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
  AllocatorPtr cpu_allocator_;
  std::map<std::pair<std::string_view, std::string_view>, CreateNodeProcessorFn> processors_;
};

}  // namespace onnxruntime
