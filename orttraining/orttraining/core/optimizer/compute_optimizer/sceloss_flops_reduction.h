// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"
#include "core/optimizer/compute_optimizer/common.h"
#include "core/optimizer/utils.h"
#include "core/graph/graph_utils.h"

namespace onnxruntime::optimizer::compute_optimizer {

/**
 * @brief Graph transformer that inserts ShrunkenGather before SoftmaxCrossEntropyLossInternal.
 *
 * For label sparsity, we can remove them from the compute of loss function, by inserting ShrunkenGather
 * operators for its two inputs.
 *
 *   logits (float) [token_count, classes]       labels (int64), [token_count]
 *                     \                             /
 *                  SoftmaxCrossEntropyLossInternal(ignore_index=-100)
 *                     /                       \
 *   loss (shape is scalar or [token_count])      log_prob [token_count, classes]
 *
 * Be noted in Transformer-based models:
 * > `token_count` usually equals with `batch size` x `sequence length`.
 * > `classes` usually equals with `vocabulary`.
 *
 * Only insert ShrunkGather if all following conditions are met:
 * 1. `SoftmaxCrossEntropyLossInternal`'s reduction MUST not be 'none', to make sure loss is a scalar.
 *   Otherwise, the loss is in shape [token_count], changing on `token_count` will affect subsquent computations.
 * 2. `SoftmaxCrossEntropyLossInternal`'s 2nd output MUST not be graph output and not consumed by other other nodes.
 *
 *
 * After the transformation:
 *                                        labels [token_count]
 *                                            \_______
 *                                             \       \
 *                                              \     Sub(ignore_index)
 *                                               \        \
 *                                                |        |
 *                                                |        |
 *                                                |      NonZero
 *                                                |        |
 *                                                |      Squeeze
 *                                                |        |
 *                                                |   indices of valid token [valid_token_count]
 *                                                 \          |
 *  logits [token_count, classes]                    \       /
 *                      \     /                       \     /
 *                  ShrunkenGather                ShrunkenGather
 *            [valid_token_count, classes]         [valid_token_count]
 *                         \                          /
 *                       SoftmaxCrossEntropyLossInternal(ignore_index=-100)
 *                                /                         \
 *                               /                           \
 *   loss (shape is scalar or [valid_token_count])      log_prob [valid_token_count, classes]
 *
 * In this specific scenario, it is easy to infer that valid_token_count <= token_count.
 * After insertion, loss computation flop is reduced. Additionally, upstream Gather graph optimization
 * will try to reduce flop for other ops further.
 */
class InsertGatherBeforeSceLoss : public GraphTransformer {
 public:
  InsertGatherBeforeSceLoss(const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("InsertGatherBeforeSceLoss", compatible_execution_providers) {
  }

  /**
   * @brief
   */
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime::optimizer::compute_optimizer
