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
 * @brief Graph transformer that eliminates unnecessary padding computation caused by embedding sparsity.
 *
 * In transformer trainings, input_ids are usually padded to the same length, which is the max sequence length,
 * so its shape is [batch_size, sequence_length] or [sequence_length, batch_size]. This graph transformer
 * tries to MERGE the leading two dimensions and REMOVE the padding on the merged
 * dimension, i.e, [batch_size, sequence_length, ...] -> [batch_size * sequence_length, ...] ->

 *
 * This transformer is implemented in the following steps:
 * 1. Iterate the graph and find the Embedding node that matches these requirements:
 *    1.1 The 2nd input is a graph input and its rank > 2, with the first two dimensions, are:
 *        [batch_size, sequence_length]. Both dimensions can be symbolic or concrete dim values.
 *    1.2 The 3rd input(padding idx) is a scalar constant initializer, and should >= 0.
 * 2. Append embedding node in node_to_scan_list.
 *    Iterate the node_to_scan_list, for each node,
 *    2.1 Check if it is supported for pad elimination (from a pre-defined op list). If no, record this node as output
 *        of the subgraph and continue to next node in node_to_scan_list.
 *    2.2 Find all output_args who have dim params inheriting from embedding input ids' dim params (e.g. batch_size and
 *        sequence_length), if all check passes, append node's output nodes to node_to_scan_list. Otherwise, record this
 *        node as output of the subgraph and continue to next node in node_to_scan_list.
 *    2.3 For input_args that are not affected by embedding output, check the dim params with other input_args that are
 *        affected by embedding output.  If all check passes, append node's output nodes in node_to_scan_list and
 *        check if it's needed to record the input_args that are not affected by embedding as input of the subgraph
 *        that will be inserted 'Reshape + ShrunkenGather' pattern later. Otherwise, record this node as output of the
 *        subgraph and continue to next node in node_to_scan_list.
 * 3. Insert pattern of 'Reshape + ShrunkenGather' before each input of the subgraph to eliminate the padding.
 * 4. Insert pattern of 'GatherGrad + Reshape' after each output of the subgraph to restore the original shape.
 *    This is needed to ensure not to affect subsequent computations
 *
 * For example, given the following graph:
 * 1. `input_0` is a tensor that is an in-direct output of ATen embedding node.
 * 2. `input_1` is a tensor that is NOT a direct or in-direct output of ATen embedding node.
 *
 *  embed.weight    input_ids [batch_size, seq_length]   padding_idx [1]    scale_grad_by_freq   sparse
 *       \                 \                               /                /                      /
 *        \                 \                             /                /                      /
 *         \                 \                           /                /                      /
 *          \_________________\_________________________/________________/______________________/
 *                                         |
 *                                   ATen:embedding
 *                                         |
 *                  - - - - - - - - - - - -|
 *                  |                      |
 *                 input_0                 |                  input_1
 *                      \                  |                   /
 *                       \__________       |       ___________/
 *                                  \      |      /
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
 * embed.weight               \                               padding_idx [1]    scale_grad_by_freq   sparse
 *      \                      \  shape:[valid_token]               /                  /                /
 *       \                      \                                  /                  /                /
 *        \______________________\________________________________/__________________/________________/
 *                                                  |
 *                                            Aten:embedding
 *          - - - - - - - - - - - - - - - - - - - - |
 *         |                                        |
 *       input_0                                    |            input_1
 *          \ [batch_size, seq_length, ...]         |               |
 *           \                                      |     [batch_size, seq_length, ...]
 *            \     [-1]                            |               |
 *             \    /                               |               |
 *             Reshape       (valid_token_index)    |              Reshape      (valid_token_index)
 *                  \           /                   |                 \           /
 *                   ShrunkenGather       shape:[valid_token, ...]     ShrunkenGather
 *                             \                    |                    /
 *    shape:[valid_token, ...]  \                   |                   /
 *                               \                  |                  /
 *                         candidate_input_node     |       candidate_input_node
 *                                 \                |                /
 *                                  \               |               /
 *
 *                                             Subgraph
 *
 *                                                  |
 *                                                  | shape:[valid_token, ...]
 *                                                  |
 *                                                  |  (valid_token_index)
 *                                                  |     /   ________________ (unflatten_dims), shape:[2],
 *                                                  |    /   /                 value:[batch_size, seq_length]
 *                                                  |   /   /
 *                                                 PadAndUnflatten
 *                                                  |
 *                                                  | [batch_size, seq_length, ...]
 *                                          candidate_output_node
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
