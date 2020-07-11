// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef _WIN32
// disable some warnings from protobuf to pass Windows build
#pragma warning(disable : 4244)
#endif

#include "core/graph/graph_viewer.h"

#include "core/graph/graph_utils.h"

#include <queue>

namespace onnxruntime {

bool NodeCompare::operator()(const Node* n1, const Node* n2) const {
  return n1->Index() < n2->Index();
}

struct PriorityNodeCompare {
  bool operator()(const Node* n1, const Node* n2) const {
    // if (n1->Priority() > n2->Priority()) {
    //   return true;
    // }
    // return n1->Index() < n2->Index();
    return n1->Priority() > n2->Priority();
  }
};

GraphViewer::GraphViewer(const Graph& graph) {
  graph_ = &graph;
  std::vector<const Node*> leaf_nodes;
  for (auto& node : graph_->Nodes()) {
    if (node.OutputNodesBegin() == node.OutputNodesEnd()) {
      // This is a leaf node (without any output node).
      leaf_nodes.push_back(&node);
    }
  }

  // Reverse the order of input vector, such that forward nodes are ReverseDFS first
  // This results in an execution order that forward nodes is always ran before the backward and recompute nodes
  std::reverse(leaf_nodes.begin(), leaf_nodes.end());

  // for (const Node* n : leaf_nodes) {
  //   std::cout << "Graph View leaf nodes: " << n->Name() << "\n";
  // }

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

const std::vector<NodeIndex> GraphViewer::GetNodesInTopologicalOrderWithPriority() const {
  std::unordered_map<NodeIndex, size_t> in_degree;
  std::priority_queue<const Node*, std::vector<const Node*>, PriorityNodeCompare> to_visit;
  std::vector<NodeIndex> topo_order;

  for (auto& node : graph_->Nodes()) {
    size_t input_edge_count = node.GetInputEdgesCount();
    //in_degree.insert(std::make_pair<NodeIndex, size_t>(node.Index(), input_edge_count));
    in_degree.insert({node.Index(), input_edge_count});

    // std::cout << "init: " << node.Name() << "indgree" << input_edge_count << "\n";

    if (input_edge_count == 0) {
      to_visit.push(&node);
      // std::cout << "To Visit: " << node.Name() << "\n";
    }
  }

  while (!to_visit.empty()) {
    const Node* current = to_visit.top();
    to_visit.pop();

    if (!current) continue;

    for (auto node_it = current->OutputNodesBegin(); node_it != current->OutputNodesEnd(); ++node_it) {
      in_degree[node_it->Index()]--;

      if (in_degree[node_it->Index()] == 0) {
        to_visit.push(&*node_it);
      }
    }

    topo_order.push_back(current->Index());

    //std::cout << "topo_order: " << current->Name() << " pri: " << current->Priority() << "\n";
  }

  if (graph_->NumberOfNodes() == static_cast<int>(topo_order.size())) {
    std::cout << "All Nodes included in the topo sort \n";
  } else {
    std::cout << "Graph has a cycle\n";
  }

  return topo_order;
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
