// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <variant>

#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/constant_sharing.h"
#include "core/optimizer/utils.h"

namespace onnxruntime {

namespace {

// Supports limited data types.
static constexpr std::array supported_data_types{
    ONNX_NAMESPACE::TensorProto_DataType_FLOAT16,
    ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
    ONNX_NAMESPACE::TensorProto_DataType_DOUBLE,
    ONNX_NAMESPACE::TensorProto_DataType_INT32,
    ONNX_NAMESPACE::TensorProto_DataType_INT64,
};

bool IsSupportedDataType(int32_t data_type) {
  return std::find(supported_data_types.cbegin(), supported_data_types.cend(), data_type) !=
         supported_data_types.cend();
}

using SupportedTypeList = boost::mp11::mp_list<MLFloat16, float, double, int32_t, int64_t>;

// A threshold is defined here to restrict the graph transformation only applied to small tensors.
// Be note: having a bigger threshold means more overhead when we do the graph transformations.
// `8` is chosen to cover common constant use cases in some Reshape/Gather/Concat's inputs.
// TODO(pengwa): we can gradually increase this threshold if we see more benefits (memory-saving
// or more CSE optimizations triggered). Should be careful to cover test cases that assume initializer
// name did not change after transformation then.

bool IsAllowedToShare(const ONNX_NAMESPACE::TensorShapeProto* input_shape,
                      int64_t& num_elements) {
  if (input_shape == nullptr) return false;

  size_t dim_size = static_cast<size_t>(input_shape->dim_size());
  num_elements = 1;
  for (size_t i = 0; i < dim_size; ++i) {
    auto dim = input_shape->dim(static_cast<int>(i));
    if (!utils::HasDimValue(dim)) {
      return false;
    }

    int64_t dim_value = dim.dim_value();
    num_elements *= dim_value;
    if (num_elements > ConstantSharing::TENSOR_ELEM_COUNT_THRESHOLD) {
      return false;
    }
  }

  if (num_elements > 0 && num_elements <= ConstantSharing::TENSOR_ELEM_COUNT_THRESHOLD) {
    return true;
  }

  return false;
}

// Return true when initializer node arg is consumed by any node containing sub graphs;
// Otherwise, return false.
bool PrepareInputPortsToReplace(Graph& graph, const NodeArg* origin_initializer_node_arg,
                                InlinedHashMap<const Node*, InlinedVector<int>>& consumer_node_to_input_ports_map) {
  std::vector<const Node*> consumers = graph.GetConsumerNodes(origin_initializer_node_arg->Name());

  for (const Node* const_node : consumers) {
    // If usage is from subgraph, skip it now, can be extended to support if there is a need.
    for (int i = 0; i < static_cast<int>(const_node->ImplicitInputDefs().size()); ++i) {
      if (const_node->ImplicitInputDefs()[i] == origin_initializer_node_arg) {
        return true /* found subgraph usage */;
      }
    }

    // Iterate all input defs to replace those that are equal to origin_initializer_node_arg,
    // Then it would be safe to remove the consumer node afterward.
    for (int i = 0; i < static_cast<int>(const_node->InputDefs().size()); ++i) {
      if (const_node->InputDefs()[i] == origin_initializer_node_arg) {
        consumer_node_to_input_ports_map[const_node].push_back(i);
      }
    }
  }

  return false /* found subgraph usage */;
}

// Replace all consumer nodes to use shared initializers.
void ReplaceInputsToUseSharedInitializer(Graph& graph,
                                         const InlinedHashMap<const Node*, InlinedVector<int>>&
                                             consumer_node_to_input_ports_map,
                                         const NodeArg* origin_initializer_node_arg,
                                         NodeArg* shared_initializer_node_arg) {
  for (auto it = consumer_node_to_input_ports_map.begin(), end = consumer_node_to_input_ports_map.end();
       it != end; ++it) {
    Node* node = graph.GetNode(it->first->Index());
    // Iterate all input defs to replace those that are equal to origin_initializer_node_arg,
    // Then it would be safe to remove the consumer node.
    for (int input_index : it->second) {
      graph_utils::ReplaceNodeInput(*node, input_index, *shared_initializer_node_arg);
    }
    graph.RemoveConsumerNode(origin_initializer_node_arg->Name(), node);

    // Add consumer ref count for shared scalar initializer. The consumer set ignores duplicates.
    graph.AddConsumerNode(shared_initializer_node_arg->Name(), node);
  }

  // Remove the initializer if no other consumer nodes.
  if (graph.GetConsumerNodes(origin_initializer_node_arg->Name()).size() == 0) {
    graph.RemoveInitializedTensor(origin_initializer_node_arg->Name());
  }
}

// Return a key that is equal for two initializers exactly when their data types, shapes, and bytes are equal.
// The key is "<data type>_<dim>_..._<dim>:" followed by the raw bytes.
std::string MakeValueKey(const Initializer& initializer) {
  std::string key = std::to_string(initializer.data_type());
  for (int64_t dim : initializer.dims()) {
    key += '_';
    key += std::to_string(dim);
  }
  key += ':';
  const auto bytes = initializer.DataAsByteSpan();
  key.append(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  return key;
}

}  // namespace

Status ConstantSharing::ApplyImpl(Graph& graph, bool& modified, int /*graph_level*/,
                                  const logging::Logger& logger) const {
  int shared_count = 0;
  // Maps the key of an initializer value (see `MakeValueKey`) to the first initializer with that value.
  InlinedHashMap<std::string, NodeArg*> value_key_to_shared_arg_map;
  const InitializedTensorSet& initialized_tensor_set = graph.GetAllInitializedTensors();
  InlinedVector<std::string> original_initializer_names;
  original_initializer_names.reserve(initialized_tensor_set.size());
  for (const auto& entry : initialized_tensor_set) {
    // Ignore if the initializer exists in graph output,
    // or not a constant initializer (implicitly excludes the graph input).
    if (!graph_utils::IsConstantInitializer(graph, entry.first) ||
        graph.IsOutput(graph.GetNodeArg(entry.first)) ||
        excluded_initializers_.find(entry.first) != excluded_initializers_.end()) {
      continue;
    }
    original_initializer_names.push_back(entry.first);
  }

  for (const auto& initializer_name : original_initializer_names) {
    NodeArg* origin_initializer_node_arg = graph.GetNodeArg(initializer_name);
    int64_t num_elements = 1;
    if (origin_initializer_node_arg == nullptr ||
        !IsAllowedToShare(origin_initializer_node_arg->Shape(), num_elements)) {
      continue;
    }

    // Ignore if not constant initializers.
    const ONNX_NAMESPACE::TensorProto* tensor_proto = graph.GetConstantInitializer(
        origin_initializer_node_arg->Name(), true);
    if (!tensor_proto || !IsSupportedDataType(tensor_proto->data_type())) {
      continue;
    }
    // A map used to collect those consumers who have inputs use origin_initializer_node_arg.
    // > The key is consumer Node pointer.
    // > The value is a list of indices for the consumer Nodes' input (that used origin_initializer_node_arg).
    InlinedHashMap<const Node*, InlinedVector<int>> consumer_node_to_input_ports_map;
    bool found_subgraph_usage = PrepareInputPortsToReplace(graph, origin_initializer_node_arg,
                                                           consumer_node_to_input_ports_map);
    if (found_subgraph_usage || consumer_node_to_input_ports_map.size() == 0) {
      continue;
    }
    const Initializer initializer{*tensor_proto, graph.ModelPath()};
    auto [it, inserted] = value_key_to_shared_arg_map.try_emplace(MakeValueKey(initializer),
                                                                  origin_initializer_node_arg);
    if (!inserted) {
      shared_count += 1;
      ReplaceInputsToUseSharedInitializer(graph, consumer_node_to_input_ports_map, origin_initializer_node_arg,
                                          it->second);
      modified = true;
    }
  }
  if (shared_count > 0) {
    LOGS(logger, INFO) << "Total shared scalar initializer count: " << shared_count;
  }
  return Status::OK();
}

}  // namespace onnxruntime
