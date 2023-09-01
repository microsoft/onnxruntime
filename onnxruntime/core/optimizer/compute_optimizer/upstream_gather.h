// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// The optimization here ideally applies to both training and inferencing,
// while so far we mainly validate training during cooking the optimization.
#ifdef ENABLE_TRAINING
#pragma once

#include "core/optimizer/compute_optimizer/upstream_transformer_base.h"
#include "core/optimizer/compute_optimizer/upstream_gather_actors.h"

namespace onnxruntime {

/**
 * @brief Graph transformer that helps reduce compute FLOP while maintaining mathematically equivalent results.
 *
 * Gather and GatherND are the entry operators that trigger the optimization search.
 * The main idea here is: if the number of elements for output is much smaller than the number of elements for input,
 * by upstreaming the slicing operation, we can reduce the number of elements for more operators.
 */
class UpStreamGatherGraphTransformer
    : public optimizer::compute_optimizer::UpStreamGraphTransformerBase<
          optimizer::compute_optimizer::SliceInfo,
          optimizer::compute_optimizer::UpStreamGatherOperatorActorBase> {
 public:
  UpStreamGatherGraphTransformer(
      const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept;

  std::optional<optimizer::compute_optimizer::SliceInfo> IsSupportedForUpstream(Graph& graph, Node& node,
                                                                                const logging::Logger& logger)
      const override;

 private:
  /**
   * @brief Core pass through the logic for Gather and GatherND.
   *
   * @param graph Graph to be transformed.
   * @param queue  Queue to append propagated inputs' SliceInfo.
   * @param current_node The node before slicing node to pass through.
   * @param info The SliceInfo to be passed through.
   * @param pass_through_config The pass-through config for current_node.
   * @param logger Logger.
   * @return true if pass-through is successful, false otherwise.
   */
  bool UpStreamInternal(Graph& graph, std::deque<optimizer::compute_optimizer::SliceInfo>& queue,
                        Node& current_node, optimizer::compute_optimizer::SliceInfo& info,
                        const optimizer::compute_optimizer::OpPassThroughConfig<
                            optimizer::compute_optimizer::UpStreamGatherOperatorActorBase>& pass_through_config,
                        const logging::Logger& logger) const override;

  /**
   * @brief Pass through Slicing op from current_node's output to its specific input.
   *
   * Propagate the slicing operation into current_node's current_input_index-th input, e.g. a slicing op is inserted
   * between current_node's current_input_index-th input and current_node. For example, if current_node is Add,
   * and slice_node is a Gather(axis=1, indices=[1]):
   *
   *    input_0 [M, N, K]    input_1 [M, N, K]
   *                \        /
   *                Add [M, N, K]
   *                     |
   *            Gather0(axis=1, indices=[1])
   *                     |
   *              output [M, 1, K]
   *
   * After the pass-through, the graph will be:
   *
   *   input_0 [M, N, K]                      input_1 [M, N, K]
   *                \                                /
   *     Gather1(axis=1, indices=[1])       Gather2(axis=1, indices=[1])
   *                     \                       /
   *                       \                   /
   *                          \             /
   *                           Add [M, N, K]
   *                               |
   *                       Gather0(axis=1, indices=[1])
   *                               |
   *                         output [M, 1, K]
   *
   * Be noted: Gather1 and Gather2 are inserted on Add's two inputs.
   * Gather0's removal is done in RemoveOriginSlicingOp.
   * Be noted: Add's output shape update is done separately after RemoveOriginSlicingOp is called.
   *
   *
   * @param graph Graph to iterate.
   * @param slice_node Slicing op node that takes current_node's output as input.
   * @param current_node Current node.
   * @param current_node_input_index The current_node_input_index-th input to propagate the Slice op pass through.
   * @param info slice_node's SliceInfo.
   * @param logger Logger.
   * @param new_axis The new axis (for the new Slice op) upon current_node's original current_node_input_index-th input.
   * @return  SliceInfo for newly created slicing op.
   */
  optimizer::compute_optimizer::SliceInfo PropagateSlicingForInput(Graph& graph, Node& slice_node, Node& current_node,
                                                                   int current_node_input_index,
                                                                   optimizer::compute_optimizer::SliceInfo& info,
                                                                   int new_axis, const logging::Logger& logger) const;

  /**
   * @brief Remove the origin slicing op (for example Gather/GatherND) but don't update shapes.
   *
   * In the above example, the graph will be cleaned up to:
   *   input_0 [M, N, K]                      input_1 [M, N, K]
   *                \                                /
   *     Gather1(axis=1, indices=[1])       Gather2(axis=1, indices=[1])
   *                     \                       /
   *                       \                   /
   *                          \             /
   *                           Add [M, N, K]
   *                               |
   *                               |
   *                         output [M, 1, K]
   *
   * Be noted: Gather0 is removed, and Add's output shape is not updated here.
   *
   * @param graph Graph to iterate.
   * @param slice_node Slicing op node that takes current_node's output as input.
   * @param current_node Current node.
   * @param logger Logger.
   * @param info slice_node's SliceInfo.
   * @return
   */
  Status RemoveOriginSlicingOp(Graph& graph, Node& slice_node, Node& current_node,
                               const logging::Logger& logger, optimizer::compute_optimizer::SliceInfo& info) const;
};

}  // namespace onnxruntime
#endif
