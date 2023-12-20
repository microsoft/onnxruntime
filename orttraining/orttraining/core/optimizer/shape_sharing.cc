// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <variant>
#include "orttraining/core/optimizer/shape_sharing.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"

namespace onnxruntime {

namespace {

static constexpr char SHARED_SHAPE_OUT_PREFIX[] = "ortshared_shape_";

// Return true when initializer node arg is consumed by any node containing sub graphs;
// Otherwise, return false.
bool PrepareInputPortsToReplace(Graph& graph, const NodeArg* shape_node_out_arg,
                                InlinedHashMap<const Node*, InlinedVector<int>>& consumer_node_to_input_ports_map) {
  std::vector<const Node*> consumers = graph.GetConsumerNodes(shape_node_out_arg->Name());

  for (const Node* const_node : consumers) {
    // If usage is from subgraph, skip it now, can be extended to support if there is a need.
    for (int i = 0; i < static_cast<int>(const_node->ImplicitInputDefs().size()); ++i) {
      if (const_node->ImplicitInputDefs()[i] == shape_node_out_arg) {
        return true /* found subgraph usage */;
      }
    }

    // Iterate all input defs to replace those that are equal to shape_node_out_arg,
    // Then it would be safe to remove the consumer node afterwards.
    for (int i = 0; i < static_cast<int>(const_node->InputDefs().size()); ++i) {
      if (const_node->InputDefs()[i] == shape_node_out_arg) {
        consumer_node_to_input_ports_map[const_node].push_back(i);
      }
    }
  }

  return false /* found subgraph usage */;
}

// Replace all consumer nodes to use shared initializers.
void ReplaceInputsToUseSharedShapeNode(Graph& graph,
                                       const InlinedHashMap<const Node*, InlinedVector<int>>&
                                           consumer_node_to_input_ports_map,
                                       const NodeArg* shape_node_out_arg,
                                       NodeArg* shared_shape_node_out_arg) {
  for (auto it = consumer_node_to_input_ports_map.begin(), end = consumer_node_to_input_ports_map.end();
       it != end; ++it) {
    Node* node = graph.GetNode(it->first->Index());
    // Iterate all input defs to replace those that are equal to shape_node_out_arg,
    // Then it would be safe to remove the consumer node.
    for (int input_index : it->second) {
      graph_utils::ReplaceNodeInput(*node, input_index, *shared_shape_node_out_arg);
    }
    graph.RemoveConsumerNode(shape_node_out_arg->Name(), node);

    //  Add consumer ref count for shared scalar initializer.
    std::vector<const Node*> consumers = graph.GetConsumerNodes(shared_shape_node_out_arg->Name());
    if (std::find(consumers.begin(), consumers.end(), node) == consumers.end()) {
      graph.AddConsumerNode(shared_shape_node_out_arg->Name(), node);
    }
  }

  // Remove the Shape node if no other consumer nodes.
  if (graph.GetConsumerNodes(shape_node_out_arg->Name()).size() == 0) {
    const Node* const_shape_node = graph.GetProducerNode(shape_node_out_arg->Name());
    Node* shape_node = graph.GetNode(const_shape_node->Index());
    ORT_ENFORCE(shape_node, "Shape node should not be null: ", shape_node_out_arg->Name());
    graph_utils::RemoveNodeOutputEdges(graph, *shape_node);
    graph.RemoveNode(shape_node->Index());
  }
}

/**
 * @brief Shape value representation, which is used to store and compare Shape node output values.
 *
 * Two instances of ShapeValue are equal when:
 * 1. rank match.
 * 2. all dimensions are exactly match.
 */
struct ShapeValue {
  ShapeValue(const InlinedVector<std::variant<int64_t, std::string>>& dimensions) : dims{dimensions} {
  }

  std::string NormalizeAsString() const {
    std::ostringstream oss;
    oss << dims.size();
    for (const auto& dim : dims) {
      oss << "_";
      if (std::holds_alternative<int64_t>(dim)) {
        oss << std::get<int64_t>(dim);
      } else {
        oss << std::get<std::string>(dim);
      }
    }

    return oss.str();
  }

 private:
  InlinedVector<std::variant<int64_t, std::string>> dims;
};

static void GetShapeValue(const ONNX_NAMESPACE::TensorShapeProto* shape_proto_pointer,
                          InlinedVector<std::variant<int64_t, std::string>>& dims) {
  ORT_ENFORCE(shape_proto_pointer != nullptr, "Shape proto should not be nullptr");
  const size_t dim_size = static_cast<size_t>(shape_proto_pointer->dim_size());
  dims.reserve(dim_size);
  for (size_t i = 0; i < dim_size; ++i) {
    auto dim = shape_proto_pointer->dim(static_cast<int>(i));
    if (utils::HasDimValue(dim)) {
      int64_t dim_value = dim.dim_value();
      dims.push_back(dim_value);
    } else if (utils::HasDimParam(dim)) {
      dims.push_back(dim.dim_param());
    } else {
      ORT_THROW("Shape proto has invalid dimension.");
    }
  }
}

/**
 * @brief Get value unique id from shape store.
 *
 * If the value parsed from shape value exists in shape store, then return the index in the container;
 * Otherwise, insert the value into store, return the last index.
 */
std::tuple<bool, std::string> GetOrAddValueInShapeStore(
    std::unique_ptr<ShapeValue> shape_value,
    InlinedHashMap<std::string, std::unique_ptr<ShapeValue>>& shape_value_store) {
  const std::string data_store_key = shape_value->NormalizeAsString();
  auto it = shape_value_store.find(data_store_key);
  if (it == shape_value_store.end()) {
    shape_value_store.insert({data_store_key, std::move(shape_value)});
    return std::tuple<bool, std::string>{true, data_store_key};
  }

  return std::tuple<bool, std::string>{false, data_store_key};
}

}  // namespace

Status ShapeSharing::ApplyImpl(Graph& graph, bool& modified, int /*graph_level*/,
                               const logging::Logger& logger) const {
  int shared_count = 0;
  // > The key is a unique string representation of shape value.
  // > The value is the newly created Shape node's output NodeArg* to be shared.
  InlinedHashMap<std::string, NodeArg*> pattern_key_to_shared_arg_map;

  InlinedVector<Node*> candidate_shape_nodes_having_input_shape_proto;
  for (auto& node : graph.Nodes()) {
    if (node.OpType() == "Shape") {
      const NodeArg* input_node_arg = node.InputDefs()[0];
      if (input_node_arg->Shape() != nullptr) {
        candidate_shape_nodes_having_input_shape_proto.push_back(&node);
      }
    }
  }

  // Avoid using the scalar value directly in pattern_key because the value for example INT_MAX can be super big
  // and it will be hard to read. Instead, a constant value store is maintained, then the value index is used as the
  // value unique id when construct pattern key.
  InlinedHashMap<std::string, std::unique_ptr<ShapeValue>> shape_value_store;
  for (const auto& shape_node : candidate_shape_nodes_having_input_shape_proto) {
    NodeArg* shape_node_out_arg = graph.GetNodeArg(shape_node->OutputDefs()[0]->Name());
    InlinedVector<std::variant<int64_t, std::string>> dims;
    GetShapeValue(shape_node->InputDefs()[0]->Shape(), dims);

    if (graph_utils::IsSupportedOptypeVersionAndDomain(*shape_node, "Shape", {15, 19})) {
      int64_t start = 0;
      int64_t end = static_cast<int64_t>(dims.size());  // end is exclusive
      // Opset-15 Shape supports slicing using a 'start' and 'end' attribute
      const auto& shape_attributes = shape_node->GetAttributes();
      for (const auto& attr : shape_attributes) {
        if (attr.first == "start") {
          start = attr.second.i();
        } else if (attr.first == "end") {
          end = attr.second.i();
        }
      }

      // Remove the dimensions that are not used in the slice.
      InlinedVector<std::variant<int64_t, std::string>> updated_dims;
      for (int64_t i = start; i < end; ++i) {
        updated_dims.push_back(dims[i]);
      }

      dims = updated_dims;
    }

    // A map used to collect those consumers who have inputs use shape_node_out_arg.
    // > The key is the consumer Node pointer.
    // > The value is a list of indices for the consumer Nodes' input (that used shape_node_out_arg).
    InlinedHashMap<const Node*, InlinedVector<int>> consumer_node_to_input_ports_map;
    bool found_subgraph_usage = PrepareInputPortsToReplace(graph, shape_node_out_arg,
                                                           consumer_node_to_input_ports_map);
    if (found_subgraph_usage || consumer_node_to_input_ports_map.size() == 0) {
      continue;
    }

    std::unique_ptr<ShapeValue> shape_value = std::make_unique<ShapeValue>(dims);

    // The shape value store contains multiple buckets, indexed by data_store_key.
    // For each initializer, we will check which bucket it belongs to,
    // then add the value into the bucket if it does not exits; or get the index within the bucket if it already exists.
    bool inserted = false;
    std::string data_store_key;
    std::tie(inserted, data_store_key) = GetOrAddValueInShapeStore(std::move(shape_value), shape_value_store);

    // If there is no such existing scalar pattern, add a new one.
    if (pattern_key_to_shared_arg_map.find(data_store_key) == pattern_key_to_shared_arg_map.end()) {
      // Clone the shape node and rename the output NodeArg.
      NodeArg* new_shape_node_out_arg = &graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(SHARED_SHAPE_OUT_PREFIX),
                                                                  &(*shape_node_out_arg->TypeAsProto()));
      Node& cloned_shape_node = graph.AddNode(graph.GenerateNodeName(SHARED_SHAPE_OUT_PREFIX),
                                              "Shape",
                                              "Shape node shared by multiple nodes",
                                              shape_node->MutableInputDefs(),
                                              {new_shape_node_out_arg},
                                              &shape_node->GetAttributes(),
                                              kOnnxDomain);
      cloned_shape_node.SetExecutionProviderType(shape_node->GetExecutionProviderType());
      pattern_key_to_shared_arg_map[data_store_key] = new_shape_node_out_arg;
    } else {
      shared_count += 1;
    }

    ReplaceInputsToUseSharedShapeNode(graph, consumer_node_to_input_ports_map, shape_node_out_arg,
                                      pattern_key_to_shared_arg_map[data_store_key]);

    modified = true;
  }
  if (shared_count > 0) {
    LOGS(logger, INFO) << "Total shared shape node count: " << shared_count;
  }
  return Status::OK();
}

}  // namespace onnxruntime
