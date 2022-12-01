// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/compute_optimizer/passthrough_actors.h"
#include "core/optimizer/graph_transformer.h"
#include "core/optimizer/utils.h"

using SliceInfo = onnxruntime::optimizer::compute_optimizer::SliceInfo;
namespace onnxruntime {

/**
 * @brief Graph transformer that helps reduce compute FLOP while maintaining mathematically equivalent result.
 *
 * This graph transformation tries to identify opportunaties to reduce unnecessary computations on the graph level.
 * Currently, the major optimization is to bring some sling operators ahead as much as possible, to leave more ops
 * operate on sliced input data. Gather and GatherND are the entry operators that trigger the optimization search.
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
  std::optional<SliceInfo> IsSupportedGatherND(Graph& graph, Node& node, const logging::Logger& logger) const;
  std::optional<SliceInfo> IsSupportedGather(Graph& graph, Node& node, const logging::Logger& logger) const;
};

namespace optimizer {
namespace compute_optimizer {

/**
 * @brief Functor to trigger the optimization search for a given slicing node
 *   (for example Gather/GatherND node).
 */
struct SliceOperationReorderHandle {
  using OPSET_VERSION_LIST = std::initializer_list<ONNX_NAMESPACE::OperatorSetVersion>;
  /**
   * @brief Pass through configuration for specific operator.
   *
   * For each operator:
   * > `input_indices` can be used to explicitly specify the input indices that Slicing op can be passed through.
   *   This could be helpful if some inputs are not applicable for pass through. If not specified, all inputs
   *   are considered (but there will be checks to ignore those inputs that are not affected by the slicing axis).
   * > `actor` will be used to perform the actual pass through, including both pre-check stage and post process
   *   stage.
   */
  struct OpPassThroughConfig {
    OpPassThroughConfig() = default;

    OpPassThroughConfig(const std::vector<int>& input_indices,
                        std::shared_ptr<OperatorPassThroughActorBase> actor,
                        const OPSET_VERSION_LIST& opset_list)
        : input_indices(input_indices), actor(actor), opsets(opset_list) {
    }

    std::vector<int> input_indices;
    std::shared_ptr<OperatorPassThroughActorBase> actor;
    const OPSET_VERSION_LIST& opsets;
  };

  static std::string GetFullQualifiedOpName(const std::string& op_type, const std::string& domain) {
    return domain + "::" + op_type;
  }

  constexpr static const OPSET_VERSION_LIST opset_1 = {1};
  constexpr static const OPSET_VERSION_LIST opset_13_1 = {13, 1};
  constexpr static const OPSET_VERSION_LIST opset_13_9_1 = {13, 9, 1};
  constexpr static const OPSET_VERSION_LIST opset_13_11_1 = {13, 11, 1};
  constexpr static const OPSET_VERSION_LIST opset_13_9_6_1 = {13, 9, 6, 1};
  constexpr static const OPSET_VERSION_LIST opset_14_13_5_1 = {14, 13, 5, 1};
  constexpr static const OPSET_VERSION_LIST opset_14_13_7_6_1 = {14, 13, 7, 6, 1};
  constexpr static const OPSET_VERSION_LIST opset_13_12_10_7_6_1 = {13, 12, 10, 7, 6, 1};

  static std::unordered_map<std::string, OpPassThroughConfig>& GetOpPassThroughConfigMap() {
    static std::unordered_map<std::string, OpPassThroughConfig> allowed_passthrough_ops;
    static std::once_flag allowed_ops_init;
    std::call_once(allowed_ops_init, []() {
      allowed_passthrough_ops.insert({
          // Things to consider when more operators are added here:
          // 1. Whether the operator is safe to pass through in term of compute equivalence.
          //    If optype is not enough to guarantee the equivalence, we need to add a customized pre-check function
          //    (as LayerNormalization did).
          // 2. Whether the outputs have the same dim changes if Gather node moves before that operator.
          // 3. Should all inputs be allowed when track back further (bottom-up);
          //    if not, add the input index restriction as MatMul did.
          {GetFullQualifiedOpName("Add", kOnnxDomain),
           OpPassThroughConfig({}, std::make_shared<SimplePassThroughActor>(), opset_14_13_7_6_1)},
          {GetFullQualifiedOpName("BiasGelu", kMSDomain),
           OpPassThroughConfig({}, std::make_shared<SimplePassThroughActor>(), opset_1)},
          {GetFullQualifiedOpName("BitmaskBiasDropout", kMSDomain),
           OpPassThroughConfig({}, std::make_shared<SimplePassThroughActor>(), opset_1)},
          {GetFullQualifiedOpName("Cast", kOnnxDomain),
           OpPassThroughConfig({}, std::make_shared<SimplePassThroughActor>(), opset_13_9_6_1)},
          {GetFullQualifiedOpName("Div", kOnnxDomain),
           OpPassThroughConfig({}, std::make_shared<SimplePassThroughActor>(), opset_14_13_7_6_1)},
          {GetFullQualifiedOpName("Dropout", kOnnxDomain),
           OpPassThroughConfig({}, std::make_shared<SimplePassThroughActor>(), opset_13_12_10_7_6_1)},
          {GetFullQualifiedOpName("Gelu", kMSDomain),
           OpPassThroughConfig({}, std::make_shared<SimplePassThroughActor>(), opset_1)},
          {// Be noted, this is our own implementation of ONNX domain op.
           GetFullQualifiedOpName("LayerNormalization", kOnnxDomain),
           OpPassThroughConfig({0}, std::make_shared<ReductionOpPassThroughActor>(), opset_1)},
          {GetFullQualifiedOpName("MatMul", kOnnxDomain),
           OpPassThroughConfig({}, std::make_shared<MatMulPassThroughActor>(), opset_13_9_1)},
          {GetFullQualifiedOpName("Reshape", kOnnxDomain),
           OpPassThroughConfig({0}, std::make_shared<ReshapePassThroughActor>(), opset_14_13_5_1)},
          {GetFullQualifiedOpName("Softmax", kOnnxDomain),
           OpPassThroughConfig({0}, std::make_shared<ReductionOpPassThroughActor>(), opset_13_11_1)},
          {GetFullQualifiedOpName("Transpose", kOnnxDomain),
           OpPassThroughConfig({}, std::make_shared<TransposePassThroughActor>(), opset_13_1)},
      });
    });

    return allowed_passthrough_ops;
  }

  SliceOperationReorderHandle(const std::string& node_name) : entry_node_name_(node_name) {
  }

  bool operator()(Graph& graph, Node& current_node, SliceInfo& info, const logging::Logger& logger,
                  std::deque<SliceInfo>& queue);

 private:
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
                                     SliceInfo& info, int new_axis, const logging::Logger& logger);

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
                               const logging::Logger& logger, SliceInfo& info);

  std::string entry_node_name_;
};

}  // namespace compute_optimizer
}  // namespace optimizer
}  // namespace onnxruntime
