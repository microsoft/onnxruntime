// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/graph.h"

namespace onnxruntime {
namespace training {

/**
 * Applies the graph transformation to enable mixed precision training.
 *
 * @param graph The graph.
 * @param weights_to_train The names of the weights to train.
 * @param use_mixed_precision_initializer Whether to introduce separate mixed precision initializers
 *        or add casts from existing FP32 initializers.
 * @param mixed_precision_type The mixed precision element type.
 * @param fp32_weight_name_to_mixed_precision_node_arg Mapping from the original FP32
 *        weight name to the mixed precision NodeArg corresponding to the added mixed precision
 *        initializer. This will be empty if use_mixed_precision_initializer is false.
 *
 * @return The status of the operation.
 */
Status TransformGraphForMixedPrecision(Graph& graph,
                                       const std::unordered_set<std::string>& weights_to_train,
                                       bool use_mixed_precision_initializer,
                                       ONNX_NAMESPACE::TensorProto_DataType mixed_precision_type,
                                       std::unordered_map<std::string, NodeArg*>& fp32_weight_name_to_mixed_precision_node_arg,
                                       bool layernorm_stash_as_fp32);

/**
 * Checks if a node is an fp32-only node.
 *
 * @param node The node to check.
 * @return Whether it's an fp32-only node.
 */
bool IsFP32Node(const Node* node);
}  // namespace training
}  // namespace onnxruntime
