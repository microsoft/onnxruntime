// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/compute_optimizer/passthrough_actors.h"
#include "core/optimizer/graph_transformer.h"
#include "core/optimizer/utils.h"

using SlicingInfo = onnxruntime::optimizer::compute_optimizer::SlicingInfo;
namespace onnxruntime {

/**
 * @brief Graph transformer that helps reduce compute FLOPS while maintaining mathmatically equivalant result.
 *
 * This graph transformation tries to identify opportunaties to reduce unnecessary computations on the graph level.
 * Currently, the major optimization is to bring some sling operators ahead as much as possible, to leave more operators
 * operate on sliced input data. This can reduce the amount of FLOPS required while maintain the equivalent compute
 * result. Gather and GatherND are the entry operators that trigger the optimization search.
 *
 * In terms of file dependency, compute_optimizer.h/cc reference structs and utilities defined in
 * passthrough_actors.h/cc.
 */
class ComputeOptimizer : public GraphTransformer {
 public:
  ComputeOptimizer(const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("ComputeOptimizer", compatible_execution_providers) {}

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;

 private:
  std::optional<SlicingInfo> IsSupportedGatherND(Graph& graph, Node& node, const logging::Logger& logger) const;
  std::optional<SlicingInfo> IsSupportedGather(Graph& graph, Node& node, const logging::Logger& logger) const;
};

namespace optimizer {
namespace compute_optimizer {

/**
 * @brief Functor to trigger the optimization search for a given slicing node
 *   (for example Gather/GatherND node).
 */
struct SliceOperationReorderHandle {
  SliceOperationReorderHandle(const std::string& node_name) : entry_node_name_(node_name) {
    RegisterOperators();
  }

  bool operator()(Graph& graph, Node& current_node, SlicingInfo& info, const logging::Logger& logger,
                  std::deque<SlicingInfo>& queue);

 private:
  void RegisterOperators();

  /**
   * @brief Pass through configuration for specific operator.
   *
   * For each operator,
   * > input_indices can be used to explicitly specify the input indices that Slicing op can be passed through.
   *   This could be helpful if some inputs are not applicable for pass through. If not specified, all inputs
   *   are considered (but there will be checks to ignore those inputs that are not affected by the slicing axis).
   * > actor will be used to perform the actual pass through, including both pre-check stage and actual pass
   *   through stage.
   */
  struct OpPassThroughConfig {
    OpPassThroughConfig() = default;

    OpPassThroughConfig(const std::vector<int>& input_indices, std::shared_ptr<OperatorPassThroughActorBase> actor)
        : input_indices(input_indices), actor(actor) {}

    std::vector<int> input_indices;
    std::shared_ptr<OperatorPassThroughActorBase> actor;
  };

  /**
   * @brief Pass through Slicing op from current_node's output to its specific input.
   *
   * Populate the slicing operation into current_node's current_input_index-th input, e.g. a slicing op is inserted
   * between current_node's current_input_index-th input and current_node. For example, if current_node is Add,
   * and slice_node is a Gather(axis=1, indices=[1]):
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
   * @param current_node_input_index The current_node_input_index-th input to populate the Slice op pass through.
   * @param info slice_node's SlicingInfo.
   * @param logger Logger.
   * @param new_axis The new axis (for the new Slice op) upon current_node's original current_node_input_index-th input.
   * @return  SlicingInfo for new created slicing op.
   */
  SlicingInfo PopulateSlicingForInput(Graph& graph, Node& slice_node, Node& current_node, int current_node_input_index,
                                      SlicingInfo& info, int new_axis, const logging::Logger& logger);

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
   * @param info slice_node's SlicingInfo.
   * @return
   */
  Status RemoveOriginSlicingOp(Graph& graph, Node& slice_node, Node& current_node,
                               const logging::Logger& logger, SlicingInfo& info);

  std::string entry_node_name_;
  std::unordered_map<std::string, OpPassThroughConfig> allowed_passthrough_ops_;
};

}  // namespace compute_optimizer
}  // namespace optimizer

}  // namespace onnxruntime
