// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/graph.h"
#include "core/providers/nnapi/nnapi_builtin/builders/node_unit.h"
#include "core/optimizer/selectors_actions/helpers.h"

namespace onnxruntime {

class NodeUnit : public INodeUnit {
 public:
  NodeUnit(const Node& node) : node_(node) {}

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

  const Node& GetNode() const noexcept override {
    return node_;
  }

 private:
  const Node& node_;
};

class QDQNodeUnit : public INodeUnit {
 public:
  QDQNodeUnit(const GraphViewer& graph_viewer, const QDQNodeGroup& qdq_group);

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

 private:
  void init();
  const GraphViewer& graph_viewer_;
  const QDQNodeGroup qdq_group_;
  const Node& node_;
  std::vector<NodeArg*> input_defs_;
  std::vector<NodeArg*> output_defs_;
};

// QUESTION/TODO, do we want to embed the graph_viewer into the QDQNodeGroup?
QDQNodeUnit::QDQNodeUnit(const GraphViewer& graph_viewer, const QDQNodeGroup& qdq_group)
    : graph_viewer_(graph_viewer),
      qdq_group_(qdq_group),
      node_(*graph_viewer_.GetNode(qdq_group_.core_node)) {
  init();
}

void QDQNodeUnit::init() {
  for (auto node_index : qdq_group_.dq_nodes) {
    // This is a bit hacky, but seems there is not other way to get a non-const NodeArg* from a const Node
    auto& node = const_cast<Node&>(*graph_viewer_.GetNode(node_index));
    for (auto* def : node.MutableInputDefs()) {
      input_defs_.push_back(def);
    }
  }
  for (auto node_index : qdq_group_.q_nodes) {
    auto& node = const_cast<Node&>(*graph_viewer_.GetNode(node_index));
    for (auto* def : node.MutableOutputDefs()) {
      output_defs_.push_back(def);
    }
  }
}

const std::unique_ptr<INodeUnit> CreateNodeUnit(const Node& node) {
  return std::make_unique<NodeUnit>(node);
}

}  // namespace onnxruntime
