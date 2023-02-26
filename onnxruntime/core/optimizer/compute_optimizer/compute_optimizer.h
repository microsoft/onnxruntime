// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef ENABLE_TRAINING_CORE
#pragma once

// TODO(pengwa): rename the file name to upstream_gather.h later. Keep original name for now to make it easier to review.

#include "core/optimizer/compute_optimizer/upstream_transformer_base.h"
#include "core/optimizer/compute_optimizer/upstream_gather_actors.h"
#include "core/optimizer/graph_transformer.h"
#include "core/optimizer/utils.h"

using namespace onnxruntime::optimizer::compute_optimizer;
namespace onnxruntime {

/**
 * @brief Graph transformer that helps reduce compute FLOP while maintaining mathematically equivalent result.
 *
 * Gather and GatherND are the entry operators that trigger the optimization search.
 *
 */
class UpStreamGatherGraphTransformer : public UpStreamGraphTransformerBase<SliceInfo, UpStreamGatherOperatorActorBase> {
 public:
  UpStreamGatherGraphTransformer(const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
      : UpStreamGraphTransformerBase("UpStreamGatherGraphTransformer", compatible_execution_providers) {
    allowed_passthrough_ops_.insert({
        // Things to consider when more operators are added here:
        // 1. Whether the operator is safe to pass through in term of compute equivalence.
        //    If optype is not enough to guarantee the equivalence, we need to add a customized pre-check function
        //    (as LayerNormalization did).
        // 2. Whether the outputs have the same dim changes if Gather node moves before that operator.
        // 3. Should all inputs be allowed when track back further (bottom-up);
        //    if not, add the input index restriction as MatMul did.
        {GetFullQualifiedOpName("Add", kOnnxDomain),
         OpPassThroughConfig<UpStreamGatherOperatorActorBase>({}, std::make_shared<SimplePassThroughActor>(),
                                                              opset_14_13_7_6_1)},
        {GetFullQualifiedOpName("BiasGelu", kMSDomain),
         OpPassThroughConfig<UpStreamGatherOperatorActorBase>({}, std::make_shared<SimplePassThroughActor>(), opset_1)},
        {GetFullQualifiedOpName("BitmaskBiasDropout", kMSDomain),
         OpPassThroughConfig<UpStreamGatherOperatorActorBase>({}, std::make_shared<SimplePassThroughActor>(), opset_1)},
        {GetFullQualifiedOpName("Cast", kOnnxDomain),
         OpPassThroughConfig<UpStreamGatherOperatorActorBase>({}, std::make_shared<SimplePassThroughActor>(),
                                                              opset_13_9_6_1)},
        {GetFullQualifiedOpName("Div", kOnnxDomain),
         OpPassThroughConfig<UpStreamGatherOperatorActorBase>({}, std::make_shared<SimplePassThroughActor>(),
                                                              opset_14_13_7_6_1)},
        {GetFullQualifiedOpName("Dropout", kOnnxDomain),
         OpPassThroughConfig<UpStreamGatherOperatorActorBase>({}, std::make_shared<SimplePassThroughActor>(),
                                                              opset_13_12_10_7_6_1)},
        {GetFullQualifiedOpName("Gelu", kMSDomain),
         OpPassThroughConfig<UpStreamGatherOperatorActorBase>({}, std::make_shared<SimplePassThroughActor>(),
                                                              opset_1)},
        {// Be noted, this is our own implementation of ONNX domain op.
         GetFullQualifiedOpName("LayerNormalization", kOnnxDomain),
         OpPassThroughConfig<UpStreamGatherOperatorActorBase>({0}, std::make_shared<ReductionOpPassThroughActor>(),
                                                              opset_1)},
        {GetFullQualifiedOpName("MatMul", kOnnxDomain),
         OpPassThroughConfig<UpStreamGatherOperatorActorBase>({}, std::make_shared<MatMulPassThroughActor>(),
                                                              opset_13_9_1)},
        {GetFullQualifiedOpName("Reshape", kOnnxDomain),
         OpPassThroughConfig<UpStreamGatherOperatorActorBase>({0}, std::make_shared<ReshapePassThroughActor>(),
                                                              opset_14_13_5_1)},
        {GetFullQualifiedOpName("Softmax", kOnnxDomain),
         OpPassThroughConfig<UpStreamGatherOperatorActorBase>({0}, std::make_shared<ReductionOpPassThroughActor>(),
                                                              opset_13_11_1)},
        {GetFullQualifiedOpName("Transpose", kOnnxDomain),
         OpPassThroughConfig<UpStreamGatherOperatorActorBase>({}, std::make_shared<TransposePassThroughActor>(),
                                                              opset_13_1)},
    });
  }

  std::optional<SliceInfo> IsSupportedForUpstream(Graph& graph, Node& node,
                                                  const logging::Logger& logger) const override;

 private:
  std::optional<SliceInfo> IsSupportedGatherND(Graph& graph, Node& node, const logging::Logger& logger) const;
  std::optional<SliceInfo> IsSupportedGather(Graph& graph, Node& node, const logging::Logger& logger) const;

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
