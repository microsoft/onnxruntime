// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <queue>
#include <string>

#include "core/optimizer/graph_transformer.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/compute_optimizer/shared_utils.h"

namespace onnxruntime {

/**
 * @Class PaddingElimination
 *
 * @brief Graph transformer that eliminates unnecessary padding computation induced by embedding sparsity.
 * This transformer is implemented in the following steps:
 * 1. Iterate the graph and find the Embedding node that matches these requirements:
 *     (1) Its 2nd input is a graph input and its rank > 2 with the first two dimensions are dim_params which are
 *         actually batch_size and sequence_length.
 *   Note: Now only support the case of the first two dimensions to merged and remove the padding on the merged
 *         dimension, i.e, [batch_size, sequence_length, ...] -> [batch_size * sequence_length, ...] ->
 *         [valid_token, ... ]. In the future, we may support the case of any two consecutive dimensions to merged,
 *         such as [..., batch_size, sequence_length, ...].
 *     (2) Its 3nd input is a scalar constant initializer which is the padding idx that should >= 0.
 * 2. Iterate the subgraph that starts from the Embedding node and find all inputs and outputs of the subgraph.
 * 3. Insert pattern of 'Reshape + ShrunkenGather' before each input of the subgraph to eliminate the padding.
 * 4. Insert pattern of 'GatherGrad + Reshape' after each output of the subgraph to restore the original shape.
 *    This is needed to ensure not to affect subsequent computations
 *
 * For example, given the following graph:
 *
 *              input_ids [batch_size, seq_length]   padding_idx [1]
 *       \                   \                               /                /
 *         \                  \                             /                /
 *           \                 \                           /                /
 *                                   Aten:embedding
 *                                         |
 *                                         |
 *                     input               |
 *                             \
 *                                     Subgraph
 *
 *                                         |
 *                                       output
 *
 *
 * After the transformation:
 *
 *             input_ids [batch_size, seq_length]
 *                |      \
 *                |       \        [-1]     padding_idx [1]
 *                |        \       /         /
 *                |         Reshape         /
 *                |              \         /
 *                |                  Sub
 *                |    [-1]           |
 *                |    /            NonZero
 *               Reshape              |
 *                     \            Squeeze
 *                      \            /
 *                       \          /(valid_token_index)
 *                        \        /
 *                      ShrunkenGather
 *                              \
 *      \                        \  shape:[valid_token]                padding_idx [1]        /
 *       \                        \                                     /                    /
 *        \                        \                                   /                    /
 *                                          Aten:embedding
 *                                                  |
 *                                                  |
 *       input [batch_size, seq_length]             |
 *           \                                      |
 *            \     [-1]                            |
 *             \    /                               |
 *             Reshape       (valid_token_index)    |
 *                  \           /                   |
 *                   ShrunkenGatherGrad             |  shape:[valid_token]
 *                             \                    |
 *         shape:[valid_token]   \                  |
 *                                 \                |
 *
 *                                             Subgraph
 *
 *                                                  |
 *                                                  | shape:[valid_token]
 *                                                  |
 *                                                  |
 * [batch_size*seq_length]   (valid_token_index)    |
 *             \                  |                 /
 *              \                 |                /
 *                \               |              /
 *
 *                         GatherGrad
 *                              |
 *                           Reshape
 *                              |
 *                              |
 *                      output [batch_size, valid_token]
 *
 *
 *
*/
class PaddingElimination : public GraphTransformer {
 public:
  PaddingElimination(const InlinedHashSet<std::string_view>& compatible_execution_providers = {},
                     const std::vector<std::string>& sparse_embedding_input_names = {}) noexcept
      : GraphTransformer("PaddingElimination", compatible_execution_providers),
        sparse_embedding_input_names_{sparse_embedding_input_names} {}

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
private:
  std::vector<std::string> sparse_embedding_input_names_;
};

}  // namespace onnxruntime
