// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/graph_proto_serializer.h"

namespace onnxruntime {

void GraphViewerToProto(const GraphViewer& graph_view,
                        ONNX_NAMESPACE::GraphProto& graph_proto,
                        bool include_initializer,
                        bool include_outer_scope_args,
                        ExecutionOrder order) {
  graph_proto.set_name(graph_view.Name());
  graph_proto.set_doc_string(graph_view.Description());

  for (const auto* input_arg : graph_view.GetInputsIncludingInitializers()) {
    *(graph_proto.mutable_input()->Add()) = input_arg->ToProto();
  }

  for (const auto* output_arg : graph_view.GetOutputs()) {
    *(graph_proto.mutable_output()->Add()) = output_arg->ToProto();
  }

  std::unordered_set<const onnxruntime::NodeArg*> value_info_ = graph_view.GetValueInfo();

  // Reserve memory for the vector to avoid reallocations
  std::vector<const NodeArg*> value_info_sorted;
  value_info_sorted.reserve(value_info_.size());

  value_info_sorted.assign(value_info_.begin(), value_info_.end());
  auto sort_predicate = [](const NodeArg* v1, const NodeArg* v2) {
    return v1->Name() < v2->Name();
  };

  // This ensures consistent ordering of value_info entries in the output graph
  std::sort(value_info_sorted.begin(), value_info_sorted.end(), sort_predicate);

  for (const auto* value_info : value_info_sorted) {
    *(graph_proto.mutable_value_info()->Add()) = value_info->ToProto();
  }

  if (include_outer_scope_args) {
    // add the NodeArg info for outer scope NodeArgs so we capture the type information
    for (const auto& name : graph_view.GetOuterScopeNodeArgNames()) {
      auto* node_arg = graph_view.GetNodeArg(name);
      ORT_ENFORCE(node_arg, "Outer scope node arg name '" + name + "'was added but does not exist. ");
      *(graph_proto.mutable_value_info()->Add()) = node_arg->ToProto();
    }
  }

  // Nodes must be sorted in Topological Order in the GraphProto per ONNX spec.
  for (auto& node_idx : graph_view.GetNodesInTopologicalOrder(order)) {
    const gsl::not_null<ONNX_NAMESPACE::NodeProto*> node_proto{graph_proto.add_node()};
    const gsl::not_null<const Node*> p_node{graph_view.GetNode(node_idx)};
    // we need to update any GraphProto attributes for subgraphs so that any changes made by things
    // such as the optimizers are captured. otherwise we can end up saving an invalid graph.
    p_node->ToProto(*node_proto, /* update_subgraphs */ true);
  }

  if (include_initializer) {
    std::unordered_set<std::string> current_scope_initializer_set;

    auto& initializers = graph_view.GetAllInitializedTensors();

    // Sort initializers to maintain consistency in model proto created across inference requests
    std::vector<std::string> const_inits;
    for (auto& it : initializers) {
      const_inits.push_back(it.first);
    }
    std::sort(const_inits.begin(), const_inits.end());

    for (auto& it : const_inits) {
      auto* p_initializer = graph_proto.add_initializer();
      *p_initializer = *(initializers.at(it));
      current_scope_initializer_set.insert(it);
    }

    // handle outer scope value which is a constant initializer
    if (include_outer_scope_args) {
      for (auto& node_idx : graph_view.GetNodesInTopologicalOrder(order)) {
        const auto& node = graph_view.GetNode(node_idx);
        for (const auto& input : node->InputDefs()) {
          if (current_scope_initializer_set.find(input->Name()) != current_scope_initializer_set.end()) {
            continue;
          }
          if (graph_view.IsConstantInitializer(input->Name(), true)) {
            auto* p_initializer = graph_proto.add_initializer();
            *p_initializer = *(graph_view.GetConstantInitializer(input->Name(), true));
            current_scope_initializer_set.insert(input->Name());
          }
        }
      }
    }
  }
}

}  // namespace onnxruntime
