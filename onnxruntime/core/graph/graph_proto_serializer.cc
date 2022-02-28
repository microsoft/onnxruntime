#include "core/graph/graph_proto_serializer.h"

namespace onnxruntime {

void GraphProtoSerializer::ToProto(ONNX_NAMESPACE::GraphProto& graph_proto, bool include_initializer) {
  graph_proto.set_name(graph_viewer_->Name());
  graph_proto.set_doc_string(graph_viewer_->Description());

  for (const auto* input_arg : graph_viewer_->GetInputsIncludingInitializers()) {
    *(graph_proto.mutable_input()->Add()) = input_arg->ToProto();
  }

  for (const auto* output_arg : graph_viewer_->GetOutputs()) {
    *(graph_proto.mutable_output()->Add()) = output_arg->ToProto();
  }

  for (const auto* value_info : graph_viewer_->GetValueInfo()) {
    *(graph_proto.mutable_value_info()->Add()) = value_info->ToProto();
  }

  // add the NodeArg info for outer scope NodeArgs so we capture the type information
  for (const auto& name : graph_viewer_->GetOuterScopeNodeArgNames()) {
    auto* node_arg = graph_viewer_->GetNodeArg(name);
    ORT_ENFORCE(node_arg, "Outer scope node arg name '" + name + "'was added but does not exist. ");
    *(graph_proto.mutable_value_info()->Add()) = node_arg->ToProto();
  }

  // Nodes must be sorted in Topological Order in the GraphProto per ONNX spec.
  for (auto& node_idx : graph_viewer_->GetNodesInTopologicalOrder()) {
    const gsl::not_null<ONNX_NAMESPACE::NodeProto*> node_proto{graph_proto.add_node()};
    const gsl::not_null<const Node*> p_node{graph_viewer_->GetNode(node_idx)};
    // we need to update any GraphProto attributes for subgraphs so that any changes made by things
    // such as the optimizers are captured. otherwise we can end up saving an invalid graph.
    p_node->ToProto(*node_proto, /* update_subgraphs */ true);
  }

  if (include_initializer) {
    auto& initializers = graph_viewer_->GetAllInitializedTensors();
    for (auto& it : initializers) {
      auto* p_initializer = graph_proto.add_initializer();
      *p_initializer = *(it.second);
    }
  }
}

}