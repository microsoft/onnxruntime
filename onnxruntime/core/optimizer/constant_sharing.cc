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

/**
 * @brief Share initializer for those who hold same value in same type and shape.
 *
 * @tparam T The initializer value type, used to retrieved data values.
 * @param graph Target graph to iterate.
 * @param node Target node to check initializer input.
 * @param input_index Input index of target node.
 * @param type_value_plus_rank_to_shared_arg_map Accumulated map from value/rank to initializer.
 *  The key indiciate initializer's data type, value and rank.
 *  The value is the first initializer NodeArg* to be shared.
 * @param data_type Initializer data type.
 * @return true If the input initializer can be shared.
 * @return false If the input initializer CANNOT be shared.
 */
template <typename T>
bool ConstantSharing::ShareInitializer(Graph& graph, Node* node, int input_index,
                                       std::map<std::string, NodeArg*>&
                                           type_value_plus_rank_to_shared_arg_map,
                                       int32_t data_type) const {
  InlinedVector<T> values;
  const bool require_constant = true;
  const NodeArg* input_def = node->InputDefs()[input_index];
  if (!optimizer_utils::AppendTensorFromInitializer(graph, *input_def, values, require_constant) ||
      values.size() != 1) {
    return false;
  }

  int rank = input_def->Shape()->dim_size();
  std::ostringstream oss;
  oss << data_type << "_" << values[0] << "_" << rank;
  std::string p = oss.str();
  if (type_value_plus_rank_to_shared_arg_map.find(p) == type_value_plus_rank_to_shared_arg_map.end()) {
    type_value_plus_rank_to_shared_arg_map[p] = const_cast<NodeArg*>(input_def);
    return false;
  } else {
    graph_utils::ReplaceNodeInput(*node, input_index, *type_value_plus_rank_to_shared_arg_map[p]);
    graph.RemoveConsumerNode(input_def->Name(), node);
    if (graph.GetConsumerNodes(input_def->Name()).size() == 0) {
      graph.RemoveInitializedTensor(input_def->Name());
    }

    return true;
  }
}

Status ConstantSharing::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  auto& order = graph_viewer.GetNodesInTopologicalOrder();
  std::unordered_set<const NodeArg*> visited_node_args;
  std::map<std::string, NodeArg*> value_plus_rank_to_shared_arg_map;

  for (NodeIndex i : order) {
    auto* node = graph.GetNode(i);
    ORT_RETURN_IF_ERROR(Recurse(*node, modified, graph_level, logger));

    /**
     * Loop all inputs of Node node, find all constant initializers, if it's data type is int32_t/int64_t or
     * float/MLFloat16, and it contains single value, then we can share same initializer.
     */
    for (int input_index = 0; input_index < static_cast<int>(node->InputDefs().size()); ++input_index) {
      const NodeArg* input_def = node->InputDefs()[input_index];
      auto it = visited_node_args.find(input_def);
      if (it == visited_node_args.end()) {
        visited_node_args.insert(input_def);

        // Ignore if not constant initializers,
        const ONNX_NAMESPACE::TensorProto* tensor_proto = nullptr;
        if (!(tensor_proto = graph.GetConstantInitializer(input_def->Name(), true)) ||
            excluded_initializers_.find(input_def->Name()) != excluded_initializers_.end()) {
          continue;
        }

        onnxruntime::Initializer initializer{*tensor_proto, graph.ModelPath()};
        int32_t data_type = initializer.data_type();
        bool is_shared = false;
        if (data_type == utils::ToTensorProtoElementType<int32_t>() ||
            data_type == utils::ToTensorProtoElementType<int64_t>()) {
          is_shared = ShareInitializer<int64_t>(graph, node, input_index, value_plus_rank_to_shared_arg_map, data_type);
        } else if (data_type == utils::ToTensorProtoElementType<float>() ||
                   data_type == utils::ToTensorProtoElementType<MLFloat16>()) {
          is_shared = ShareInitializer<float>(graph, node, input_index, value_plus_rank_to_shared_arg_map, data_type);
        }
        modified = modified || is_shared;
      }
    }
  }

  return Status::OK();
}
}  // namespace onnxruntime

#endif
