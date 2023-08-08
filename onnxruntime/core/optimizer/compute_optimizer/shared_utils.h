// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// The optimization here ideally is applicable to both training and inferencing,
// while so far we mainly validate on training during cooking the optimization.
#ifdef ENABLE_TRAINING
#pragma once

#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

// Uncomment for debugging Compute optimizer (CO).
// #define CO_NEED_LOG_DEBUG_INFO 1

#ifndef LOG_DEBUG_INFO
#ifdef CO_NEED_LOG_DEBUG_INFO
#define LOG_DEBUG_INFO(logger, message) LOGS(logger, WARNING) << message
#else
#define LOG_DEBUG_INFO(logger, message) \
  ORT_UNUSED_PARAMETER(logger);         \
  do {                                  \
  } while (0)
#endif
#endif

namespace onnxruntime::optimizer::compute_optimizer {

using OPSET_VERSION_LIST = std::initializer_list<ONNX_NAMESPACE::OperatorSetVersion>;
const OPSET_VERSION_LIST opset_1{1};
const OPSET_VERSION_LIST opset_13_1{13, 1};
const OPSET_VERSION_LIST opset_13_9_1{13, 9, 1};
const OPSET_VERSION_LIST opset_13_11_1{13, 11, 1};
const OPSET_VERSION_LIST opset_13_9_6_1{13, 9, 6, 1};
const OPSET_VERSION_LIST opset_14_13_5_1{14, 13, 5, 1};
const OPSET_VERSION_LIST opset_14_13_7_6_1{14, 13, 7, 6, 1};
const OPSET_VERSION_LIST opset_13_12_10_7_6_1{13, 12, 10, 7, 6, 1};
const OPSET_VERSION_LIST opset_19_13_9_6_1{19, 13, 9, 6, 1};
const OPSET_VERSION_LIST opset_19_14_13_5_1{19, 14, 13, 5, 1};

/**
 * @brief Base class for all upstream operator passthrough actors.
 */
struct UpStreamOperatorActorBase {
};

/**
 * @brief Base class for all upstream operator info .
 */
struct UpstreamOperatorInfoBase {
  UpstreamOperatorInfoBase(Node* node, bool is_entry_node_ptr = false) : node_ptr(node) {
    if (is_entry_node_ptr) {
      entry_node_name = node_ptr->Name();
    }
  }

  Node* node_ptr;  // The node that triggers the optimization search.
  std::string entry_node_name;
};

/**
 * @brief Pass through configuration for specific operator.
 *
 * For each operator:
 * > `actor` will be used to perform the actual pass through, including both pre-check stage and post process
 *   stage.
 */
template <typename T>
struct OpPassThroughConfig {
  OpPassThroughConfig(std::shared_ptr<T> actor,
                      const OPSET_VERSION_LIST& opset_list)
      : actor(actor), opsets(opset_list) {
    // Compile-time check
    static_assert(std::is_base_of<UpStreamOperatorActorBase, T>::value,
                  "type parameter of this class must derive from UpStreamOperatorActorBase");
  }

  std::shared_ptr<T> actor;
  const OPSET_VERSION_LIST& opsets;
};

/**
 * @brief Insert a new node to the graph,
 *  1. taking dest_node.input[dest_input_index] as the input of the new node.
 *  2. remove connection of dest_node and it's dest_input_index-th input producer node.
 *  3. connect the new node and dest_node.
 *
 * Original graph:
 *         Node A
 *       /        \
 *  A-output-0    A-output-1
 *                  \         B-input-1
 *                   \       /
 *                     Node B
 *                       |
 *
 * dest_node = Node B
 * dest_input_index = 0
 * op_type = C
 *
 * After inserting the new node:
 *         Node A
 *       /        \
 *  A-output-0    A-output-1
 *                  \
 *                 Node C
 *                  /  \
 *         c-output-0  C-output-1     B-input-1
 *                         \       /
 *                          Node B
 *                             |
 * @param graph  Graph to insert the new node.
 * @param dest_node The node to insert the new node before.
 * @param dest_in_index The input index of the dest_node to insert the new node before.
 * @param new_node_output_index The output index of the new node to connect to the dest_node.
 * @param name The name of the new node.
 * @param op_type The op_type of the new node.
 * @param description The description of the new node.
 * @param input_args The input args of the new node. At least one of the input args should be the
 *   dest_node's dest_in_index-th input arg.
 * @param attributes The attributes of the new node.
 * @param domain The domain of the new node.
 * @param logger The logger.
 * @return
 */
Node* InsertIntermediateNodeOnDestInput(Graph& graph,
                                        Node& dest_node, int dest_in_index,
                                        int new_node_input_index,
                                        int new_node_output_index,
                                        const std::string& name, const std::string& op_type,
                                        const std::string& description,
                                        const InlinedVector<NodeArg*>& input_args,
                                        const InlinedVector<NodeArg*>& output_args,
                                        const onnxruntime::NodeAttributes& attributes,
                                        const std::string& domain,
                                        const logging::Logger& logger);

enum class DimCompare {
  Equal = 0,
  BroadCast = 1,  // e.g. dim value is 1.
  NotExist = 2,
  NotEqual = 3,
  DimCompareRetMax = 4,
};

/**
 * @brief Compare the target shape with the fully broadcasted output shape.
 *
 * @param full_broadcasted_shape Full broadcasted shape as a baseline to compare.
 * @param target_shape Shape to compare, can have a dim value be 1 for broad-cast-able dimension.
 * @return A bool indicate whether check successfully or not.
 *         A vector of type DimCompare. The size of the vector is the same as full_broadcasted_shape.
 *
 * Be noted: full_broadcasted_shape's length should be >= target_shape's length, otherwise return false.
 */
std::pair<bool, std::vector<DimCompare>> CompareInputShapeWithOutputShape(
    const ONNX_NAMESPACE::TensorShapeProto* full_broadcasted_shape,
    const ONNX_NAMESPACE::TensorShapeProto* target_shape);

/**
 * @brief Get opset version from the graph.
 */
int GetONNXOpSetVersion(const Graph& graph);

/**
 * @brief Create an initializer from given dims/value vector and name.
 *
 * @param dims A int vector as the shape of the created initializer. If we want to create a scalar initializer,
 *   we should pass an empty vector.
 * @param values A int vector containing the value buffer.
 */

NodeArg* CreateInitializerFromVector(Graph& graph,
                                     const InlinedVector<int64_t>& dims,
                                     const InlinedVector<int64_t>& values,
                                     const std::string& name);

/**
 * @brief Insert a pattern of 'Sub + NonZero + Squeeze' into the graph.
 *   This pattern is used to filter out the valid indices of the input tensor which is not padding idx.
 *   After inserting the pattern, the graph will be like:
 *     input_to_filter                   invalid_value
 *  (shape: [total_indices_count])
 *                         \         /
 *                             Sub
 *                              |
 *                           NonZero
 *                              |
 *                           Squeeze
 *                              |
 *                 output (shape: [valid_indices_count])
 *
 * @param graph The graph to insert the pattern.
 * @param input_to_filter The input tensor to filter.
 * @param invalid_value The invalid index value to remove.
 * @param execution_provider_type The execution provider type of the inserted nodes.
 */
NodeArg* InsertNodesForValidIndices(Graph& graph,
                                    NodeArg* input_to_filter,
                                    NodeArg* invalid_value,
                                    const std::string& execution_provider_type);

}  // namespace onnxruntime::optimizer::compute_optimizer
#endif
