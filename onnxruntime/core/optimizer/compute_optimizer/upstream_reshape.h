// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef ENABLE_TRAINING_CORE
#pragma once

#include "core/optimizer/compute_optimizer/upstream_transformer_base.h"
#include "core/optimizer/compute_optimizer/upstream_reshape_actors.h"
#include "core/optimizer/graph_transformer.h"
#include "core/optimizer/utils.h"

using namespace onnxruntime::optimizer::compute_optimizer;
namespace onnxruntime {

/**
 * @brief Graph transformer that helps flatten the first two leading dimensions, to make it easier for other
 * transformer passes to do compute optimizer easier.
 *
 * Reshape node (from 3D to 2D, with the first two dimension be flatten) are the entry operators that trigger
 * the optimization search.
 *
 */
class UpStreamReshapeGraphTransformer : public UpStreamGraphTransformerBase<ReshapeInfo, UpStreamReshapeOperatorActorBase> {
 public:
  UpStreamReshapeGraphTransformer(const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
      : UpStreamGraphTransformerBase("UpStreamReshapeGraphTransformer", compatible_execution_providers) {
    allowed_passthrough_ops_.insert({
        // Things to consider when more operators are added here:
        // 1. Whether the operator is safe to pass through in term of compute equivalence.
        //    If optype is not enough to guarantee the equivalence, we need to add a customized pre-check function
        //    (as LayerNormalization did).
        // 2. Whether the outputs have the same dim changes if Gather node moves before that operator.
        // 3. Should all inputs be allowed when track back further (bottom-up);
        //    if not, add the input index restriction as MatMul did.
        {GetFullQualifiedOpName("Add", kOnnxDomain),
         OpPassThroughConfig<UpStreamReshapeOperatorActorBase>({}, std::make_shared<SimplePointwiseReshapeActor<true>>(), opset_14_13_7_6_1)},
        {GetFullQualifiedOpName("BiasGelu", kMSDomain),
         OpPassThroughConfig<UpStreamReshapeOperatorActorBase>({}, std::make_shared<SimplePointwiseReshapeActor<true>>(), opset_1)},
        // {GetFullQualifiedOpName("BitmaskBiasDropout", kMSDomain),
        //  OpPassThroughConfig<UpStreamReshapeOperatorActorBase>({}, std::make_shared<SimplePassThroughActor>(), opset_1)},
        {GetFullQualifiedOpName("Cast", kOnnxDomain),
         OpPassThroughConfig<UpStreamReshapeOperatorActorBase>({}, std::make_shared<SimplePointwiseReshapeActor<true>>(), opset_13_9_6_1)},
        // {GetFullQualifiedOpName("Div", kOnnxDomain),
        //  OpPassThroughConfig<UpStreamReshapeOperatorActorBase>({}, std::make_shared<SimplePassThroughActor>(), opset_14_13_7_6_1)},
        {GetFullQualifiedOpName("Dropout", kOnnxDomain),
         OpPassThroughConfig<UpStreamReshapeOperatorActorBase>({}, std::make_shared<SimplePointwiseReshapeActor<true>>(), opset_13_12_10_7_6_1)},
        // {GetFullQualifiedOpName("Gelu", kMSDomain),
        //  OpPassThroughConfig<UpStreamReshapeOperatorActorBase>({}, std::make_shared<SimplePassThroughActor>(), opset_1)},
        {// Be noted, this is our own implementation of ONNX domain op.
         GetFullQualifiedOpName("LayerNormalization", kOnnxDomain),
         OpPassThroughConfig<UpStreamReshapeOperatorActorBase>({}, std::make_shared<LayerNormalizationReshapeActor>(), opset_1)},
        {GetFullQualifiedOpName("MatMul", kOnnxDomain),
         OpPassThroughConfig<UpStreamReshapeOperatorActorBase>({}, std::make_shared<MatMulReshapeActor>(), opset_13_9_1)},
        // {GetFullQualifiedOpName("Reshape", kOnnxDomain),
        //  OpPassThroughConfig<UpStreamReshapeOperatorActorBase>({0}, std::make_shared<ReshapePassThroughActor>(), opset_14_13_5_1)},
        // {GetFullQualifiedOpName("Softmax", kOnnxDomain),
        //  OpPassThroughConfig<UpStreamReshapeOperatorActorBase>({0}, std::make_shared<ReductionOpPassThroughActor>(), opset_13_11_1)},
        // {GetFullQualifiedOpName("Transpose", kOnnxDomain),
        //  OpPassThroughConfig<UpStreamReshapeOperatorActorBase>({}, std::make_shared<TransposePassThroughActor>(), opset_13_1)},
    });
  }

  std::optional<ReshapeInfo>
  IsSupportedForUpstream(Graph& graph, Node& node,
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
   * After the pass through, the graph will be:
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
   * Reshape0's removal and Add's output shape update is done in RemoveOriginReshapeOp.
   *
   *
   * @param graph Graph to iterate.
   * @param reshape_node Reshape op node the takes current_node's output as input.
   * @param current_node Current node.
   * @param current_node_input_index The current_node_input_index-th input to propagate the Slice op pass through.
   * @param info reshape_node's ReshapeInfo.
   * @param logger Logger.
   * @return  ReshapeInfo for new created reshape op.
   */
  ReshapeInfo PropagateReshapeForInput(Graph& graph, Node& reshape_node, Node& current_node, int current_node_input_index,
                                       ReshapeInfo& info, std::vector<DimCompareRet>&, const logging::Logger& logger) const;

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
