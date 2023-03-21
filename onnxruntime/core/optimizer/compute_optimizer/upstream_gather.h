// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// The optimization here ideally is applicable to both training and inferencing,
// while so far we mainly validate on training during cooking the optimization.
#ifdef ENABLE_TRAINING_CORE
#pragma once

#include "core/optimizer/compute_optimizer/upstream_transformer_base.h"
#include "core/optimizer/compute_optimizer/upstream_gather_actors.h"

using namespace onnxruntime::optimizer::compute_optimizer;
namespace onnxruntime {

/**
 * @brief Graph transformer that helps reduce compute FLOP while maintaining mathematically equivalent result.
 *
 * Gather and GatherND are the entry operators that trigger the optimization search.
 * The main idea here is: if the number of elements for output is much smaller than the number of elements for input,
 * by upstreaming the slicing operation, we can reduce the number of elements for more operators.
 */
class UpStreamGatherGraphTransformer
    : public UpStreamGraphTransformerBase<SliceInfo, UpStreamGatherOperatorActorBase> {
 public:
  UpStreamGatherGraphTransformer(
      const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept;

  std::optional<SliceInfo> IsSupportedForUpstream(Graph& graph, Node& node,
                                                  const logging::Logger& logger) const override;

 private:
  /**
   * @brief Core pass through logic for Gather and GatherND.
   *
   * @param graph Graph to be transformed.
   * @param queue  Queue to append propagated inputs' SliceInfo.
   * @param current_node The node before slicing node to pass through.
   * @param info The SliceInfo to be passed through.
   * @param pass_through_config The pass through config for current_node.
   * @param logger Logger.
   * @param entry_node_name The name of the entry node, for logging purpose.
   * @return true if pass through is successful, false otherwise.
   */
  bool UpStreamInternal(Graph& graph, std::deque<SliceInfo>& queue,
                        Node& current_node, SliceInfo& info,
                        const OpPassThroughConfig<UpStreamGatherOperatorActorBase>& pass_through_config,
                        const logging::Logger& logger,
                        const std::string& entry_node_name) const override;

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
   * After the pass through, the graph will be:
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
   * Gather0's removal and Add's output shape update is done in RemoveOriginSlicingOp.
   *
   *
   * @param graph Graph to iterate.
   * @param slice_node Slicing op node the takes current_node's output as input.
   * @param current_node Current node.
   * @param current_node_input_index The current_node_input_index-th input to propagate the Slice op pass through.
   * @param info slice_node's SliceInfo.
   * @param logger Logger.
   * @param new_axis The new axis (for the new Slice op) upon current_node's original current_node_input_index-th input.
   * @return  SliceInfo for new created slicing op.
   */
  SliceInfo PropagateSlicingForInput(Graph& graph, Node& slice_node, Node& current_node, int current_node_input_index,
                                     SliceInfo& info, int new_axis, const logging::Logger& logger) const;

  /**
   * @brief Remove the origin slicing op (for example Gather/GatherND) and update shapes.
   *
   * In the above example, the graph will be cleaned up to:
   *   input_0 [M, N, K]                      input_1 [M, N, K]
   *                \                                /
   *     Gather1(axis=1, indices=[1])       Gather2(axis=1, indices=[1])
   *                     \                       /
   *                       \                   /
   *                          \             /
   *                           Add [M, 1, K]
   *                               |
   *                               |
   *                         output [M, 1, K]
   *
   * Be noted: Gather0 is removed, Add's output shape is updated.
   *
   * @param graph Graph to iterate.
   * @param slice_node Slicing op node the takes current_node's output as input.
   * @param current_node Current node.
   * @param logger Logger.
   * @param info slice_node's SliceInfo.
   * @return
   */
  Status RemoveOriginSlicingOp(Graph& graph, Node& slice_node, Node& current_node,
                               const logging::Logger& logger, SliceInfo& info) const;
};

}  // namespace onnxruntime
#endif
