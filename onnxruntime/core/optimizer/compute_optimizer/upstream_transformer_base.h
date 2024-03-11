// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// The optimization here ideally applies to both training and inferencing,
// while so far we mainly validate training during cooking the optimization.
#ifdef ENABLE_TRAINING
#pragma once

#include "core/optimizer/graph_transformer.h"
#include "core/optimizer/utils.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/compute_optimizer/shared_utils.h"

#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

namespace onnxruntime::optimizer::compute_optimizer {

/**
 * @brief Graph transformer base that helps reduce compute FLOP while maintaining mathematically equivalent results.
 *
 * The series of graph transformations (inheriting from this base class) tries to identify opportunities to reduce
 * unnecessary computations on the graph level.
 *
 * Currently, the major optimization is to bring some slice operators ahead as much as possible, to leave more ops
 * operate on sliced input data. Gather and GatherND are the entry operators that trigger the optimization search.
 *
 * T1 defines the operator info type that is used to store the information of the operator to propagate.
 * T2 defines the base operator actor type that is used to support the pass-through.
 */
template <typename T1, typename T2>
class UpStreamGraphTransformerBase : public GraphTransformer {
 public:
  UpStreamGraphTransformerBase(const std::string& name,
                               const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
      : GraphTransformer(name, compatible_execution_providers) {
    // Compile-time check
    static_assert(std::is_base_of<UpstreamOperatorInfoBase, T1>::value,
                  "type parameter of this class must derive from UpstreamOperatorInfoBase");
    static_assert(std::is_base_of<UpStreamOperatorActorBase, T2>::value,
                  "type parameter of this class must derive from UpStreamOperatorActorBase");
  }

  /**
   * @brief The main loop for upstream, which is responsible for finding the entry optimization node
   *     and trigger its upstream.
   */
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;

 protected:
  /**
   * @brief Check if the node is supported for upstream.
   * @param graph  The graph that is to be checked.
   * @param node The node to be checked.
   * @param logger The logger.
   * @return Return nullopt if not supported, otherwise return the operator info.
   */
  virtual std::optional<T1> IsSupportedForUpstream(Graph& graph, Node& node, const logging::Logger& logger) const = 0;

  /**
   * @brief The key function for the child class is to implement the upstream logic for the given node.
   *
   * @param graph The graph that is to be checked.
   * @param queue The upstream operator info queue. When handling current_node's pass-through, for those inputs that
   *     passed through the Gather/Reshape node, it can also be a candidate for further upstream.
   * @param current_node The node that is to be checked.
   */
  virtual bool UpStreamInternal(Graph& graph, std::deque<T1>& queue,
                                Node& current_node, T1& info,
                                const OpPassThroughConfig<T2>& pass_through_config,
                                const logging::Logger& logger) const = 0;

  std::unordered_map<std::string, OpPassThroughConfig<T2>> allowed_passthrough_ops_;

 private:
  /**
   * @brief Handle one given node's upstream and will call UpStreamInternal for implementation details.
   */
  bool Upstream(Graph& graph, std::deque<T1>& queue, Node& current_node, T1& info, const logging::Logger& logger) const;
};

}  // namespace onnxruntime::optimizer::compute_optimizer
#endif
