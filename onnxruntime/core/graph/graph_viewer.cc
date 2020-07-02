// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef _WIN32
// disable some warnings from protobuf to pass Windows build
#pragma warning(disable : 4244)
#endif

#include "core/graph/graph_viewer.h"

#include "core/graph/graph_utils.h"

namespace onnxruntime {

bool NodeCompare::operator()(const Node* n1, const Node* n2) const {
  return n1->Index() < n2->Index();
}

GraphViewer::GraphViewer(const Graph& graph) {
  graph_ = &graph;
  std::vector<const Node*> leaf_nodes;
  for (auto& node : graph_->Nodes()) {
    if (node.OutputNodesBegin() == node.OutputNodesEnd()) {
      // This is a leaf node (without any output node).
      leaf_nodes.push_back(&node);
    }
  }
  graph.ReverseDFSFrom(
      leaf_nodes,
      nullptr,
      [this](const Node* n) {
        nodes_in_topological_order_.push_back(n->Index());
      },
      NodeCompare());

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

bool GraphViewer::CanOverrideInitializer() const noexcept {
  return graph_->CanOverrideInitializer();
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

const InitializedTensorSet& GraphViewer::GetAllInitializedTensors() const noexcept {
  return graph_->GetAllInitializedTensors();
}

const NodeArg* GraphViewer::GetNodeArg(const std::string& name) const {
  return graph_->GetNodeArg(name);
}

bool GraphViewer::IsSubgraph() const {
  return graph_->IsSubgraph();
}

bool GraphViewer::IsConstantInitializer(const std::string& name, bool check_outer_scope) const {
  return graph_utils::IsConstantInitializer(*graph_, name, check_outer_scope);
}

}  // namespace onnxruntime
