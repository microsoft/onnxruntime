// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <deque>
#include <vector>
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/qdq_transformer.h"
#include "core/optimizer/utils.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {

class QDQOperatorTransformer {
 public:
  QDQOperatorTransformer(Node& node, Graph& graph) : node_(node), graph_(graph) {}
  virtual ~QDQOperatorTransformer() {}
  virtual bool Transform(const std::vector<const Node*>& parents, const std::vector<const Node*>& children) = 0;

 protected:
  Node& node_;
  Graph& graph_;
};

class QDQConvTransformer : public QDQOperatorTransformer {
 public:
  QDQConvTransformer(Node& node, Graph& graph) : QDQOperatorTransformer(node, graph) {}

  bool Transform(const std::vector<const Node*>& parents, const std::vector<const Node*>& children) override {
    std::vector<NodeArg*> input_defs(graph_.GetNode(parents[0]->Index())->MutableInputDefs());
    Node* weight = graph_.GetNode(parents[1]->Index());
    input_defs.insert(input_defs.end(), weight->MutableInputDefs().begin(), weight->MutableInputDefs().end());

    if (children.size() == 1) {
      Node* q = graph_.GetNode(children[0]->Index());
      input_defs.push_back(q->MutableInputDefs()[1]);
      input_defs.push_back(q->MutableInputDefs()[2]);
      if (parents.size() == 3) {
        input_defs.push_back(graph_.GetNode(parents[2]->Index())->MutableInputDefs()[0]);
      }

      Node& qlinear_conv_node = graph_.AddNode(node_.Name(),
                                               "QLinearConv",
                                               node_.Description(),
                                               input_defs,
                                               q->MutableOutputDefs(),
                                               &node_.GetAttributes(),
                                               kOnnxDomain);
      qlinear_conv_node.SetExecutionProviderType(kCpuExecutionProvider);
    } else {
      if (parents.size() == 3) {
        input_defs.push_back(graph_.GetNode(parents[2]->Index())->MutableInputDefs()[0]);
      }

      Node& qlinear_conv_node = graph_.AddNode(node_.Name(),
                                               "ConvIntegerToFloat",
                                               node_.Description(),
                                               input_defs,
                                               node_.MutableOutputDefs(),
                                               &node_.GetAttributes(),
                                               kMSDomain);
      qlinear_conv_node.SetExecutionProviderType(kCpuExecutionProvider);
    }
    return true;
  }
};

class QDQAddTransformer : public QDQOperatorTransformer {
 public:
  QDQAddTransformer(Node& node, Graph& graph) : QDQOperatorTransformer(node, graph) {}

  bool Transform(const std::vector<const Node*>& parents, const std::vector<const Node*>& children) override {
    if (children.size() != 1) {
      return false;
    }
    std::vector<NodeArg*> input_defs(graph_.GetNode(parents[0]->Index())->MutableInputDefs());
    Node* b = graph_.GetNode(parents[1]->Index());
    input_defs.insert(input_defs.end(), b->MutableInputDefs().begin(), b->MutableInputDefs().end());

    Node* q = graph_.GetNode(children[0]->Index());
    input_defs.push_back(q->MutableInputDefs()[1]);
    input_defs.push_back(q->MutableInputDefs()[2]);

    Node& qlinear_conv_node = graph_.AddNode(node_.Name(),
                                             "QLinearAdd",
                                             node_.Description(),
                                             input_defs,
                                             q->MutableOutputDefs(),
                                             &node_.GetAttributes(),
                                             kMSDomain);
    qlinear_conv_node.SetExecutionProviderType(kCpuExecutionProvider);
    return true;
  }
};

class QDQMatMulTransformer : public QDQOperatorTransformer {
 public:
  QDQMatMulTransformer(Node& node, Graph& graph) : QDQOperatorTransformer(node, graph) {}

  bool Transform(const std::vector<const Node*>& parents, const std::vector<const Node*>& children) override {
    if (children.size() != 1) {
      return false;
    }
    std::vector<NodeArg*> input_defs(graph_.GetNode(parents[0]->Index())->MutableInputDefs());
    Node* b = graph_.GetNode(parents[1]->Index());
    input_defs.insert(input_defs.end(), b->MutableInputDefs().begin(), b->MutableInputDefs().end());

    Node* q = graph_.GetNode(children[0]->Index());
    input_defs.push_back(q->MutableInputDefs()[1]);
    input_defs.push_back(q->MutableInputDefs()[2]);

    Node& qlinear_conv_node = graph_.AddNode(node_.Name(),
                                             "QLinearMatMul",
                                             node_.Description(),
                                             input_defs,
                                             q->MutableOutputDefs(),
                                             &node_.GetAttributes(),
                                             kOnnxDomain);
    qlinear_conv_node.SetExecutionProviderType(kCpuExecutionProvider);
    return true;
  }
};

class QDQTransformerImpl {
 public:
  QDQTransformerImpl(Graph& graph) noexcept : graph_(graph) {}

  void Transform(Node& node) {
    // extract DequantizeLinear from parents and QuantizeLinear in children
    std::vector<const Node*> parents = graph_utils::FindParentsByType(node, "DequantizeLinear");

    if (parents.size() == 0 || parents.size() != node.GetInputEdgesCount()) {
      return;
    }

    std::vector<const Node*> children = graph_utils::FindChildrenByType(node, "QuantizeLinear");

    // track dq output edges count
    for (auto parent_node : parents) {
      if (!dq_output_edges_count_.count(parent_node)) {
        dq_output_edges_count_[parent_node] = parent_node->GetOutputEdgesCount();
      }
    }

    std::unique_ptr<QDQOperatorTransformer> op_trans = CreateQDQOperatorTransformer(node);

    if (op_trans) {
      if (op_trans->Transform(parents, children)) {
        for (auto parent_node : parents) {
          dq_output_edges_count_[parent_node]--;

          UpdateNodesToRemove(parents);
          UpdateNodesToRemove(children);
          nodes_to_remove_.insert(node.Index());
        }
      }
    }
  }
  void Finalize(bool& modified) {
    for (auto node_idx : nodes_to_remove_) {
      graph_utils::RemoveNodeOutputEdges(graph_, *graph_.GetNode(node_idx));
      graph_.RemoveNode(node_idx);
    }
    modified = true;
  }

 private:
  std::unique_ptr<QDQOperatorTransformer> CreateQDQOperatorTransformer(Node& node) {
    if (node.OpType() == "Conv") {
      return std::make_unique<QDQConvTransformer>(node, graph_);
    } else if (node.OpType() == "Add") {
      return std::make_unique<QDQAddTransformer>(node, graph_);
    } else if (node.OpType() == "MatMul") {
      return std::make_unique<QDQMatMulTransformer>(node, graph_);
    }
    return std::unique_ptr<QDQOperatorTransformer>();
  }

  void UpdateNodesToRemove(const std::vector<const Node*>& nodes) {
    for (auto node : nodes) {
      if (dq_output_edges_count_[node] == 0 && !nodes_to_remove_.count(node->Index())) {
        nodes_to_remove_.insert(node->Index());
      }
    }
  }

  Graph& graph_;

  std::unordered_map<const Node*, size_t> dq_output_edges_count_;

  // Stores a queue of nodes to be removed after walking through the graph.
  std::set<NodeIndex> nodes_to_remove_;
};

Status QDQTransformer::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  QDQTransformerImpl impl(graph);
  GraphViewer graph_viewer(graph);

  for (auto index : graph_viewer.GetNodesInTopologicalOrder()) {
    auto& node = *graph.GetNode(index);
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));
    if (node.GetExecutionProviderType() == kCpuExecutionProvider) {
      impl.Transform(node);
    }
  }
  impl.Finalize(modified);
  return Status::OK();
}

}  // namespace onnxruntime
