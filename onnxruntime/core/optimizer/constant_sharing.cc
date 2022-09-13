// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef ENABLE_TRAINING

#include <limits>

#include "core/framework/op_kernel.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/constant_sharing.h"
#include "core/optimizer/utils.h"

namespace onnxruntime {

namespace {

// Supports limited data types.
InlinedVector<int32_t> supported_data_types{
    ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
    ONNX_NAMESPACE::TensorProto_DataType_FLOAT16,
    ONNX_NAMESPACE::TensorProto_DataType_DOUBLE,
    ONNX_NAMESPACE::TensorProto_DataType_INT32,
    ONNX_NAMESPACE::TensorProto_DataType_INT64,
};

bool IsSupportedDataType(int32_t data_type) {
  if (std::find(supported_data_types.begin(), supported_data_types.end(), data_type) == supported_data_types.end()) {
    return false;
  }

  return true;
}

bool IsSingleValueShape(const ONNX_NAMESPACE::TensorShapeProto* input_shape) {
  if (input_shape == nullptr) {
    return false;
  }

  size_t dim_size = static_cast<size_t>(input_shape->dim_size());
  if (dim_size == 0 ||
      (dim_size == 1 && utils::HasDimValue(input_shape->dim(0)) && input_shape->dim(0).dim_value() == 1)) {
    return true;
  }

  return false;
}

static constexpr char SHARED_INITIALIZER_PREFIX[] = "ort_shared_scalar_";

std::string CreateSharedInitializerName(const std::string& pattern_key) {
  return SHARED_INITIALIZER_PREFIX + pattern_key;
}

bool IsSharedInitializer(const NodeArg* arg) {
  return arg && (arg->Name().rfind(SHARED_INITIALIZER_PREFIX, 0) == 0);
}

}  // namespace

/**
 * @brief Share initializer for those who hold same value of same type and rank.
 *
 * @param graph Target graph to iterate.
 * @param node Target node to check initializer input.
 * @param input_index Input index of target node.
 * @param type_type_value_plus_rank_to_shared_arg_map Accumulated map from type/value/rank to initializer.
 *  The key indiciates initializer's data type, value and rank.
 *  The value is the first initializer NodeArg* to be shared.
 * @param pattern_key A string constructed by data type, value, and rank. Used as a key in
 *    param type_type_value_plus_rank_to_shared_arg_map.
 * @param initializer The initializer representation of the constant.
 */
void ConstantSharing::ShareInitializer(Graph& graph, Node* node, int input_index,
                                       std::map<std::string, NodeArg*>&
                                           type_type_value_plus_rank_to_shared_arg_map,
                                       const std::string& pattern_key,
                                       onnxruntime::Initializer& initializer) const {
  const NodeArg* input_def = node->InputDefs()[input_index];

  // If there is no such existing scalar pattern, add a new one.
  if (type_type_value_plus_rank_to_shared_arg_map.find(pattern_key) ==
      type_type_value_plus_rank_to_shared_arg_map.end()) {
    ONNX_NAMESPACE::TensorProto constant_tensor_proto_as_replacement;
    initializer.ToProto(constant_tensor_proto_as_replacement);
    constant_tensor_proto_as_replacement.set_name(graph.GenerateNodeArgName(CreateSharedInitializerName(pattern_key)));
    NodeArg& shared_scalar_initializer_node_arg = graph_utils::AddInitializer(graph,
                                                                              constant_tensor_proto_as_replacement);
    type_type_value_plus_rank_to_shared_arg_map[pattern_key] = &shared_scalar_initializer_node_arg;
  }

  // Replace the scalar reference using existing shared one.
  NodeArg* arg_used_to_replace = type_type_value_plus_rank_to_shared_arg_map[pattern_key];

  // Iterate all input defs to replace those that are equal to input_def,
  // Then it would be safe to remove the consumer node.
  for (int i = input_index; i < static_cast<int>(node->InputDefs().size()); ++i) {
    if (node->InputDefs()[i] == input_def) {
      graph_utils::ReplaceNodeInput(*node, i, *arg_used_to_replace);
    }
  }
  graph.RemoveConsumerNode(input_def->Name(), node);

  // Remove the initializer if no other consumer nodes.
  if (graph.GetConsumerNodes(input_def->Name()).size() == 0) {
    graph.RemoveInitializedTensor(input_def->Name());
  }

  // Add consumer ref count for shared scalar initializer.
  std::vector<const Node*> consumers = graph.GetConsumerNodes(arg_used_to_replace->Name());
  if (std::find(consumers.begin(), consumers.end(), node) == consumers.end()) {
    graph.AddConsumerNode(arg_used_to_replace->Name(), node);
  }
}

Status ConstantSharing::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  auto& order = graph_viewer.GetNodesInTopologicalOrder();
  std::map<std::string, NodeArg*> type_value_plus_rank_to_shared_arg_map;

  for (NodeIndex i : order) {
    auto* node = graph.GetNode(i);
    ORT_RETURN_IF_ERROR(Recurse(*node, modified, graph_level, logger));

    /**
     * Loop all inputs of Node node, find all constant initializers, if it's data type is int32_t/int64_t or
     * float/MLFloat16, and it contains single value, then we can share same initializer.
     */
    for (int input_index = 0; input_index < static_cast<int>(node->InputDefs().size()); ++input_index) {
      const NodeArg* input_def = node->InputDefs()[input_index];

      // Already handled, skip it (for example, two inputs share the same initializers at the beginning.).
      if (IsSharedInitializer(input_def)) {
        continue;
      }

      auto input_shape = input_def->Shape();
      if (input_shape == nullptr || !IsSingleValueShape(input_shape)) {
        continue;
      }

      // Ignore if not constant initializers.
      const ONNX_NAMESPACE::TensorProto* tensor_proto = graph.GetConstantInitializer(input_def->Name(), true);
      if (!tensor_proto || excluded_initializers_.find(input_def->Name()) != excluded_initializers_.end()) {
        continue;
      }

      int32_t data_type = tensor_proto->data_type();
      if (!IsSupportedDataType(data_type)) {
        continue;
      }

      onnxruntime::Initializer initializer{*tensor_proto, graph.ModelPath()};
      std::ostringstream pattern_key_oss;
      pattern_key_oss << data_type << "_";
      switch (data_type) {
        case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
          pattern_key_oss << *initializer.data<float>();
          break;
        case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
          pattern_key_oss << math::halfToFloat(initializer.data<MLFloat16>()->val);
          break;
        case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
          pattern_key_oss << *initializer.data<double>();
          break;
        case ONNX_NAMESPACE::TensorProto_DataType_INT32:
          pattern_key_oss << *initializer.data<int32_t>();
          break;
        case ONNX_NAMESPACE::TensorProto_DataType_INT64:
          pattern_key_oss << *initializer.data<int64_t>();
          break;
        default:
          ORT_THROW("Should not go here");
      }

      pattern_key_oss << "_" << input_def->Shape()->dim_size();
      ShareInitializer(graph, node, input_index, type_value_plus_rank_to_shared_arg_map, pattern_key_oss.str(),
                       initializer);

      modified = true;
    }
  }

  return Status::OK();
}
}  // namespace onnxruntime

#endif
