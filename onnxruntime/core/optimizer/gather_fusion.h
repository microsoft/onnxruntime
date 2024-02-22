// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
@Class GatherToSplitFusion

Fuse multiple Gather nodes that comsuming one output to one Split node.
*/
class GatherToSplitFusion : public GraphTransformer {
 public:
  GatherToSplitFusion(const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("GatherToSplitFusion", compatible_execution_providers) {}

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;

 private:
  bool IsSupportedGather(const Graph& graph, const Node& node, int64_t& index, int64_t& axis, int64_t& indices_n_dims) const;
};

/**
@Class GatherToGliceFusion

Fuse Range->Gather to Slice node.
*/
class GatherToSliceFusion : public GraphTransformer {
 public:
  GatherToSliceFusion(const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("GatherToSliceFusion", compatible_execution_providers) {}

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
