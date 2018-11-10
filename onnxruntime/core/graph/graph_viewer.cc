// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef _WIN32
// disable some warnings from protobuf to pass Windows build
#pragma warning(disable : 4244)
#endif

#include "core/graph/graph.h"

namespace onnxruntime {
GraphViewer::GraphViewer(const Graph& graph) {
  graph_ = &graph;
  // TODO: this will be refactored.
  // The topological order will be done in <*this> graph viewer.
  const std::vector<NodeIndex>* nodes;
  ONNXRUNTIME_ENFORCE(graph_->GetNodesInTopologicalOrder(nodes).IsOK());
  for (auto index : *nodes) {
    nodes_in_topological_order_.push_back(index);
  }

  for (auto& node : graph_->Nodes()) {
    if (node.InputEdgesBegin() == node.InputEdgesEnd()) {
      root_nodes_.push_back(node.Index());
    }
  }
}

// Graph name.
const std::string& GraphViewer::Name() const noexcept {
  return graph_->Name();
}

const std::string& GraphViewer::Description() const noexcept {
  return graph_->Description();
}

bool GraphViewer::GetInitializedTensor(const std::string& tensor_name, const ONNX_NAMESPACE::TensorProto*& value) const {
  return graph_->GetInitializedTensor(tensor_name, value);
}

// Graph inputs excluding initializers.
const std::vector<const NodeArg*>& GraphViewer::GetInputs() const noexcept {
  return graph_->GetInputs();
}
// Graph inputs including initializers. Contains no nullptr values.
// This will match the number and order of inputs from the GraphProto.
const std::vector<const NodeArg*>& GraphViewer::GetInputsIncludingInitializers() const noexcept {
  return graph_->GetInputsIncludingInitializers();
}

// Graph outputs. Should have no nullptr values.
const std::vector<const NodeArg*>& GraphViewer::GetOutputs() const noexcept {
  return graph_->GetOutputs();
}

// Get graph value infos.
const std::vector<const NodeArg*>& GraphViewer::GetValueInfo() const noexcept {
  return graph_->GetValueInfo();
}

// Get const Node given specific node index. May return nullptr if node as been freed.
const Node* GraphViewer::GetNode(NodeIndex node_index) const {
  return graph_->GetNode(node_index);
}

const GraphNodes& GraphViewer::Nodes() const noexcept {
  return graph_->Nodes();
}

int GraphViewer::NumberOfNodes() const noexcept {
  return graph_->NumberOfNodes();
}

int GraphViewer::MaxNodeIndex() const noexcept {
  return graph_->MaxNodeIndex();
}

const std::vector<NodeIndex>& GraphViewer::GetNodesInTopologicalOrder() const {
  return nodes_in_topological_order_;
}

const std::vector<NodeIndex>& GraphViewer::GetRootNodes() const {
  return root_nodes_;
}
}  // namespace onnxruntime
