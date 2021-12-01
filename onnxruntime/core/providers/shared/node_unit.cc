// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "node_unit.h"
#include "core/optimizer/qdq_transformer/selectors_actions/qdq_selectors.h"

namespace onnxruntime {

class NodeUnit : public INodeUnit {
 public:
  NodeUnit(const Node& node)
      : node_(node),
        all_nodes_{&node} {}

  const ConstPointerContainer<std::vector<NodeArg*>> InputDefs() const noexcept override {
    return node_.InputDefs();
  }

  const ConstPointerContainer<std::vector<NodeArg*>> OutputDefs() const noexcept override {
    return node_.OutputDefs();
  }

  const std::string& OpType() const noexcept override { return node_.OpType(); }
  int SinceVersion() const noexcept override { return node_.SinceVersion(); }
  const std::string& Domain() const noexcept override { return node_.Domain(); }
  const Path& ModelPath() const noexcept override { return node_.ModelPath(); }
  const std::string& Name() const noexcept override { return node_.Name(); }

  const Node& GetNode() const noexcept override { return node_; }

  // size_t GetInputEdgesCount() const noexcept override { return node_.GetInputEdgesCount(); }
  NodeIndex Index() const noexcept override { return node_.Index(); }

  ProviderType GetExecutionProviderType() const noexcept override { return node_.GetExecutionProviderType(); }

  // Node::NodeConstIterator OutputNodesBegin() const noexcept override { return node_.OutputNodesBegin(); }

  // Node::NodeConstIterator OutputNodesEnd() const noexcept override { return node_.OutputNodesEnd(); }

  const std::vector<const Node*> GetAllNodes() const noexcept override { return all_nodes_; }

  INodeUnit::Type UnitType() const noexcept override { return INodeUnit::Type::Node; }

 private:
  const Node& node_;
  std::vector<const Node*> all_nodes_;
};

class QDQNodeUnit : public INodeUnit {
 public:
  QDQNodeUnit(const GraphViewer& graph_viewer, const QDQ::NodeGroup& qdq_group);

  const ConstPointerContainer<std::vector<NodeArg*>> InputDefs() const noexcept override {
    return ConstPointerContainer<std::vector<NodeArg*>>(input_defs_);
  }

  const ConstPointerContainer<std::vector<NodeArg*>> OutputDefs() const noexcept override {
    return ConstPointerContainer<std::vector<NodeArg*>>(output_defs_);
  }

  const std::string& OpType() const noexcept override { return node_.OpType(); }
  int SinceVersion() const noexcept override { return node_.SinceVersion(); }
  const std::string& Domain() const noexcept override { return node_.Domain(); }
  const Path& ModelPath() const noexcept override { return node_.ModelPath(); }
  const std::string& Name() const noexcept override { return node_.Name(); }

  const Node& GetNode() const noexcept override { return node_; }
  NodeIndex Index() const noexcept override { return node_.Index(); }
  ProviderType GetExecutionProviderType() const noexcept override { return node_.GetExecutionProviderType(); }

  const std::vector<const Node*> GetAllNodes() const noexcept override { return all_nodes_; }

  INodeUnit::Type UnitType() const noexcept override { return INodeUnit::Type::QDQ; }

 private:
  void init();
  const GraphViewer& graph_viewer_;
  const QDQ::NodeGroup qdq_group_;
  const Node& node_;
  std::vector<NodeArg*> input_defs_;
  std::vector<NodeArg*> output_defs_;
  std::vector<const Node*> all_nodes_;
};

// QUESTION/TODO, do we want to embed the graph_viewer into the QDQNodeGroup?
QDQNodeUnit::QDQNodeUnit(const GraphViewer& graph_viewer, const QDQ::NodeGroup& qdq_group)
    : graph_viewer_(graph_viewer),
      qdq_group_(qdq_group),
      node_(*graph_viewer_.GetNode(qdq_group_.target_node)) {
  init();
}

void QDQNodeUnit::init() {
  for (auto node_index : qdq_group_.dq_nodes) {
    all_nodes_.push_back(graph_viewer_.GetNode(node_index));
  }

  for (auto node_index : qdq_group_.q_nodes) {
    auto& node = const_cast<Node&>(*graph_viewer_.GetNode(node_index));
    for (auto* def : node.MutableOutputDefs()) {
      output_defs_.push_back(def);
    }
    all_nodes_.push_back(&node);
  }

  all_nodes_.push_back(&node_);

  auto get_input = [&](std::vector<NodeArg*>& defs, NodeIndex node_index) {
    // This is a bit hacky, but seems there is not other way to get a non-const NodeArg* from a const Node
    auto& node = const_cast<Node&>(*graph_viewer_.GetNode(node_index));
    defs.push_back(node.MutableInputDefs()[0]);
  };

  auto get_scale_zp = [&](std::vector<NodeArg*>& defs, NodeIndex node_index) {
    // This is a bit hacky, but seems there is not other way to get a non-const NodeArg* from a const Node
    auto& node = const_cast<Node&>(*graph_viewer_.GetNode(node_index));
    defs.push_back(node.MutableInputDefs()[1]);
    defs.push_back(node.MutableInputDefs()[2]);
  };

  auto get_all_input = [&](std::vector<NodeArg*>& defs, NodeIndex node_index) {
    // This is a bit hacky, but seems there is not other way to get a non-const NodeArg* from a const Node
    get_input(defs, node_index);
    get_scale_zp(defs, node_index);
  };

  // Conv only, other may also need special handling
  if (node_.OpType() == "Conv") {
    get_all_input(input_defs_, qdq_group_.dq_nodes[0]);
    get_all_input(input_defs_, qdq_group_.dq_nodes[1]);
    get_scale_zp(input_defs_, qdq_group_.q_nodes[0]);
    get_input(input_defs_, qdq_group_.dq_nodes[2]);
  }
}

const std::unique_ptr<INodeUnit> CreateNodeUnit(const Node& node) {
  return std::make_unique<NodeUnit>(node);
}

const std::unique_ptr<INodeUnit> CreateQDQNodeUnit(const GraphViewer& graph_viewer, const QDQ::NodeGroup& qdq_group) {
  return std::make_unique<QDQNodeUnit>(graph_viewer, qdq_group);
}

}  // namespace onnxruntime