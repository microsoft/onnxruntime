// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// The optimization here ideally applies to both training and inference,
// while so far we mainly validate training during cooking the optimization.
#ifdef ENABLE_TRAINING_CORE
#pragma once

#include "core/optimizer/compute_optimizer/upstream_transformer_base.h"
#include "core/optimizer/compute_optimizer/upstream_reshape_actors.h"

using namespace onnxruntime::optimizer::compute_optimizer;
namespace onnxruntime {

/**
 * @brief Graph transformer that helps flatten the first two leading dimensions, to make it easier for other
 * transformer passes to do compute optimizer easier.
 *
 * Reshape nodes (from 3D to 2D, with the first two dimensions be flatten) are the entry operators that trigger
 * the optimization search.
 *
 */
class UpStreamReshapeGraphTransformer
    : public UpStreamGraphTransformerBase<ReshapeInfo, UpStreamReshapeOperatorActorBase> {
 public:
  UpStreamReshapeGraphTransformer(const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept;

  /**
   * Only support Reshape node that fulfills the following requirements:
   * > input data rank = 3
   * > input shape is a constant initializer, the untouched dim value MUST be constant.
   * > Reshape is merging the first dimension, so output data rank = 2.
   */
  std::optional<ReshapeInfo> IsSupportedForUpstream(Graph& graph, Node& node,
                                                    const logging::Logger& logger) const override;

  bool UpStreamInternal(Graph& graph, std::deque<ReshapeInfo>& queue,
                        Node& current_node, ReshapeInfo& info,
                        const OpPassThroughConfig<UpStreamReshapeOperatorActorBase>& pass_through_config,
                        const logging::Logger& logger,
                        const std::string& entry_node_name) const override;

 private:
  /**
   * @brief Pass through Reshape op from current_node's output to its specific input.
   *
   * Propagate the reshape operation into current_node's current_input_index-th input, e.g. a Reshape op is inserted
   * between current_node's current_input_index-th input and current_node. For example, if current_node is Add,
   * and reshape_node is a Reshape(shape=[-1, K]):
   *
   *    input_0 [M, N, K]    input_1 [M, N, K]
   *                \        /
   *                Add [M, N, K]
   *                     |
   *            Reshape0(shape=[-1, K])
   *                     |
   *              output [M*N, K]
   *
   * After the passthrough, the graph will be:
   *
   *   input_0 [M, N, K]                      input_1 [M, N, K]
   *                \                                /
   *      Reshape1(shape=[-1, K])         Reshape2(shape=[-1, K])
   *                     \                       /
   *                       \                   /
   *                          \             /
   *                           Add [M, N, K]
   *                               |
   *                       Reshape0(shape=[-1, K])
   *                               |
   *                         output [M*N, K]
   *
   * Be noted: Reshape1 and Reshape2 are inserted on Add's two inputs.
   * Reshape0's removal and Add's output shape update are done in RemoveOriginReshapeOp.
   *
   *
   * @param graph Graph to iterate.
   * @param reshape_node Reshape op node the takes current_node's output as input.
   * @param current_node Current node.
   * @param current_node_input_index The current_node_input_index-th input to propagate the Reshape op pass through.
   * @param info reshape_node's ReshapeInfo.
   * @param logger Logger.
   * @return  ReshapeInfo for the newly created reshape op.
   */
  ReshapeInfo PropagateReshapeForInput(Graph& graph, Node& reshape_node, Node& current_node,
                                       int current_node_input_index, ReshapeInfo& info, std::vector<DimCompareRet>&,
                                       const logging::Logger& logger) const;

  /**
   * @brief Remove the origin Reshape op but don't update shapes.
   *
   * In the above example, the graph will be cleaned up to:
   *   input_0 [M, N, K]                      input_1 [M, N, K]
   *                \                                /
   *     Reshape1(shape=[-1, K])       Reshape2(shape=[-1, K])
   *                     \                       /
   *                       \                   /
   *                          \             /
   *                           Add [M, N, K]
   *                               |
   *                               |
   *                         output [M*N, K]
   *
   * Be noted: Reshape0 is removed, Add's output shape is not updated here.
   *
   * @param graph Graph to iterate.
   * @param reshape_node Reshape op node the takes current_node's output as input.
   * @param current_node Current node.
   * @param logger Logger.
   * @param info reshape_node's ReshapeInfo.
   * @return
   */
  Status RemoveOriginReshapeOp(Graph& graph, Node& reshape_node, Node& current_node,
                               const logging::Logger& logger, ReshapeInfo& info) const;
};

}  // namespace onnxruntime
#endif
