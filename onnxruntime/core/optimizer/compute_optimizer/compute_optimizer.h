// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef ENABLE_TRAINING_CORE
#pragma once

#include "core/optimizer/compute_optimizer/passthrough_actors.h"
#include "core/optimizer/graph_transformer.h"
#include "core/optimizer/utils.h"

namespace onnxruntime {

/**
 * @brief Graph transformer that helps reduce compute FLOP while maintaining mathematically equivalent result.
 *
 * This graph transformation tries to identify opportunities to reduce unnecessary computations on the graph level.
 * Currently, the major optimization is to bring some slice operators ahead as much as possible, to leave more ops
 * operate on sliced input data. Gather and GatherND are the entry operators that trigger the optimization search.
 *
 * In terms of file dependency, compute_optimizer.h/cc reference structs and utilities defined in
 * passthrough_actors.h/cc.
 */
class ComputeOptimizer : public GraphTransformer {
 public:
  using SliceInfo = onnxruntime::optimizer::compute_optimizer::SliceInfo;
  ComputeOptimizer(const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("ComputeOptimizer", compatible_execution_providers) {}

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;

 private:
  std::optional<SliceInfo> IsSupportedGatherND(Graph& graph, Node& node, const logging::Logger& logger) const;
  std::optional<SliceInfo> IsSupportedGather(Graph& graph, Node& node, const logging::Logger& logger) const;
};

}  // namespace onnxruntime
#endif
