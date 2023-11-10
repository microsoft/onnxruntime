// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <variant>

#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/constant_sharing.h"
#include "core/optimizer/utils.h"

namespace onnxruntime {

namespace constantsharing_internal  {

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
// TODO(pengwa): we can gradually increase this threshold if we see more benefits (memory saving
// or more CSE optimizations triggered). Should be careful to cover test cases that assume initializer
// name did not change after transformation then.
static constexpr char SHARED_INITIALIZER_PREFIX[] = "ortshared_";

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
    // Then it would be safe to remove the consumer node afterwards.
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

    //  Add consumer ref count for shared scalar initializer.
    std::vector<const Node*> consumers = graph.GetConsumerNodes(shared_initializer_node_arg->Name());
    if (std::find(consumers.begin(), consumers.end(), node) == consumers.end()) {
      graph.AddConsumerNode(shared_initializer_node_arg->Name(), node);
    }
  }

  // Remove the initializer if no other consumer nodes.
  if (graph.GetConsumerNodes(origin_initializer_node_arg->Name()).size() == 0) {
    graph.RemoveInitializedTensor(origin_initializer_node_arg->Name());
  }
}

/**
 * @brief Initializer value representation, which is used to store and compare initializer values.
 *
 * Two instances of InitializerValue are equal when:
 * 1. data type match.
 * 2. data rank match.
 * 3. shape match.
 * 4. value exactly match.
 */
struct InitializerValue {
  InitializerValue(const ONNX_NAMESPACE::TensorProto* tensor_proto, Graph& graph)
      : initializer{*tensor_proto, graph.ModelPath()} {
  }

  bool operator==(const InitializerValue& other) const {
    if (initializer.data_type() == other.initializer.data_type() &&      // data type
        initializer.dims().size() == other.initializer.dims().size() &&  // rank
        SpanEq(initializer.dims(), other.initializer.dims())) {          // shape
      return SpanEq(initializer.DataAsByteSpan(), other.initializer.DataAsByteSpan());
    }

    return false;
  }

  bool operator!=(const InitializerValue& other) const {
    return !(*this == other);
  }

  Initializer initializer;
};

/**
 * @brief Get value unique id from constant store.
 *
 * If the value parsed from initializer exists in constant store, then return the index in the container;
 * Otherwise, insert the value into container, return the last index.
 */
size_t GetOrAddValueInConstantStore(
    std::unique_ptr<InitializerValue> initializer,
    InlinedHashMap<std::string, InlinedVector<std::unique_ptr<InitializerValue>>>& const_value_store,
    const std::string& data_store_key) {
  auto IsInitializerValueEqual = [&initializer](const std::unique_ptr<InitializerValue>& v) -> bool {
    return *v == *initializer;
  };

  auto& data_store = const_value_store[data_store_key];
  auto it = std::find_if(data_store.begin(), data_store.end(), IsInitializerValueEqual);
  if (it == data_store.end()) {
    data_store.emplace_back(std::move(initializer));
    return data_store.size() - 1;
  }
  return it - data_store.begin();
}

}  // namespace constantsharing_internal

Status ConstantSharing::ApplyImpl(Graph& graph, bool& modified, int /*graph_level*/,
                                  const logging::Logger& logger) const {
  using namespace constantsharing_internal;
  int shared_count = 0;
  // Accumulated map from type/value/rank to initializer:
  // > The key is a string representation of initializer's data type, value and rank.
  // > The value is newly created initializer NodeArg* to be shared.
  InlinedHashMap<std::string, NodeArg*> pattern_key_to_shared_arg_map;
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

  // Avoid using the scalar value directly in pattern_key because the value for example INT_MAX can be super big
  // and it will be hard to read. Instead, a constant value store is maintained, then the value index is used as the
  // value unique id when construct pattern key.
  InlinedHashMap<std::string, InlinedVector<std::unique_ptr<InitializerValue>>> const_value_store;
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
    const std::string data_store_key = MakeString(tensor_proto->data_type(),
                                                  "_", origin_initializer_node_arg->Shape()->dim_size(),
                                                  "_", num_elements);

    std::unique_ptr<InitializerValue> init_value = std::make_unique<InitializerValue>(tensor_proto, graph);
    // The constant value store contains multiple buckets, indexed by data_store_key.
    // For each initializer, we will check which bucket it belongs to,
    // then add the value into the bucket if it does not exits; or get the index within the bucket if it already exists.
    size_t value_id = GetOrAddValueInConstantStore(std::move(init_value), const_value_store, data_store_key);

    // Construct a string by data type, value, and rank. Used as a key in pattern_key_to_shared_arg_map.
    const std::string pattern_key = MakeString(SHARED_INITIALIZER_PREFIX, data_store_key, "_", value_id);

    // If there is no such existing scalar pattern, add a new one.
    if (pattern_key_to_shared_arg_map.find(pattern_key) == pattern_key_to_shared_arg_map.end()) {
      // Do a copy and rename the TensorProto.
      ONNX_NAMESPACE::TensorProto constant_tensor_proto_as_replacement(*tensor_proto);
      constant_tensor_proto_as_replacement.set_name(graph.GenerateNodeArgName(pattern_key));
      NodeArg& shared_scalar_initializer_node_arg = graph_utils::AddInitializer(graph,
                                                                                constant_tensor_proto_as_replacement);
      pattern_key_to_shared_arg_map[pattern_key] = &shared_scalar_initializer_node_arg;
    } else {
      shared_count += 1;
    }

    ReplaceInputsToUseSharedInitializer(graph, consumer_node_to_input_ports_map, origin_initializer_node_arg,
                                        pattern_key_to_shared_arg_map[pattern_key]);

    modified = true;
  }
  if (shared_count > 0) {
    LOGS(logger, INFO) << "Total shared scalar initializer count: " << shared_count;
  }
  return Status::OK();
}

}  // namespace onnxruntime
