// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/graph_proto_serializer.h"

namespace onnxruntime {

void GraphViewerToProto(const GraphViewer& graph_view, 
                        ONNX_NAMESPACE::GraphProto& graph_proto, 
                        bool include_initializer,
                        bool include_outer_scope_args) {
  graph_proto.set_name(graph_view.Name());
  graph_proto.set_doc_string(graph_view.Description());

  for (const auto* input_arg : graph_view.GetInputsIncludingInitializers()) {
    *(graph_proto.mutable_input()->Add()) = input_arg->ToProto();
  }

  for (const auto* output_arg : graph_view.GetOutputs()) {
    *(graph_proto.mutable_output()->Add()) = output_arg->ToProto();
  }

  for (const auto* value_info : graph_view.GetValueInfo()) {
    *(graph_proto.mutable_value_info()->Add()) = value_info->ToProto();
  }

  if (include_outer_scope_args){
    // add the NodeArg info for outer scope NodeArgs so we capture the type information
    for (const auto& name : graph_view.GetOuterScopeNodeArgNames()) {
      auto* node_arg = graph_view.GetNodeArg(name);
      ORT_ENFORCE(node_arg, "Outer scope node arg name '" + name + "'was added but does not exist. ");
      *(graph_proto.mutable_value_info()->Add()) = node_arg->ToProto();
    }
  }
  
  // Nodes must be sorted in Topological Order in the GraphProto per ONNX spec.
  for (auto& node_idx : graph_view.GetNodesInTopologicalOrder()) {
    const gsl::not_null<ONNX_NAMESPACE::NodeProto*> node_proto{graph_proto.add_node()};
    const gsl::not_null<const Node*> p_node{graph_view.GetNode(node_idx)};
    // we need to update any GraphProto attributes for subgraphs so that any changes made by things
    // such as the optimizers are captured. otherwise we can end up saving an invalid graph.
    p_node->ToProto(*node_proto, /* update_subgraphs */ true);
  }

  if (include_initializer) {
    auto& initializers = graph_view.GetAllInitializedTensors();
    for (auto& it : initializers) {
      auto* p_initializer = graph_proto.add_initializer();
      *p_initializer = *(it.second);
    }
  }
}

}