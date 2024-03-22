// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
@Class GatherSliceToSplitFusion

Fuse multiple Gather/Slice nodes that comsuming one output to one Split node.
*/
class GatherSliceToSplitFusion : public GraphTransformer {
 public:
  GatherSliceToSplitFusion(const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("GatherSliceToSplitFusion", compatible_execution_providers) {}

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;

 private:
  bool IsSupportedGather(const Graph& graph, const Node& node, int64_t rank, int64_t target_axis, int64_t dim_size,
                         InlinedVector<bool>& consumed, int64_t& start, bool& need_squeeze) const;

  bool IsSupportedSlice(const Graph& graph, const Node& node, int64_t rank, int64_t target_axis, int64_t dim_size,
                        InlinedVector<bool>& consumed, int64_t& start, int64_t& end) const;
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
