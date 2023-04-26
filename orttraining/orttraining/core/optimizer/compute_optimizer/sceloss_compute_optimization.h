// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef ENABLE_TRAINING
#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
 * @brief Graph transformer that inserts ShrunkenGather before SCE nodes (e.g.
 *  SoftmaxCrossEntropyLossInternal/SoftmaxCrossEntropyLoss nodes).
 *
 * For label sparsity, we can remove them from the compute of loss function, by inserting ShrunkenGather
 * operators for its two inputs.
 *
 *   logits (float) [token_count, classes]       labels (int64), [token_count]
 *                     \                             /
 *                       SCE Node(ignore_index=-100)
 *                     /                       \
 *   loss (shape is scalar or [token_count])      log_prob [token_count, classes]
 *
 * Be noted in Transformer-based models:
 * > `token_count` usually equals with `batch size` x `sequence length`.
 * > `classes` usually equals with `vocabulary`.
 *
 * Only insert ShrunkGather if all following conditions are met for SCE nodes`:
 * 1. Its reduction attribute value is 'sum' or 'mean', to make sure loss is a scalar.
 *   Otherwise, the loss is in shape [token_count], changing on `token_count` will affect subsequent computations.
 * 2. Its 2nd output (log_prob) MUST NOT be graph output and not consumed by other other nodes.
 * 3. Its ignore_index exists and is a constant scalar value.
 * 4. Its 2nd input label's input node is not `ShrunkGather` node (to avoid this transformer duplicated applied).
 *
 *
 * After the transformation:
 *                                        labels [token_count]
 *                                            \_______
 *                                             \       \
 *                                              \     Sub(ignore_index)
 *                                               \          \
 *                                                |           |
 *                                                |           |
 *                                                |        NonZero
 *                                                |           |
 *                                                |         Squeeze
 *                                                |           |
 *                                                |   indices of valid token [valid_token_count]
 *                                                 \          |
 *  logits [token_count, classes]  _________________\ _ _____/
 *                      \         /                  \      /
 *                       \       /                    \    /
 *                        \     /                      \  /
 *                  ShrunkenGather                ShrunkenGather
 *            [valid_token_count, classes]         [valid_token_count]
 *                         \                          /
 *                          SCE Node  (ignore_index=-100)
 *                                /                     \
 *                               /                       \
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

}  // namespace onnxruntime

#endif
