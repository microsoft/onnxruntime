// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <deque>
#include <vector>
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/qdq_transformer/qdq_transformer.h"
#include "core/optimizer/qdq_transformer/qdq_op_transformer.h"
#include "core/optimizer/qdq_transformer/registry.h"
#include "core/optimizer/utils.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {

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

    std::unique_ptr<QDQOperatorTransformer> op_trans = QDQRegistry::CreateQDQTransformer(node, graph_);

    if (op_trans && op_trans->Transform(parents, children)) {
      for (auto parent_node : parents) {
        dq_output_edges_count_[parent_node]--;
      }

      UpdateNodesToRemove(parents);
      UpdateNodesToRemove(children);
      if (!op_trans->KeepNode()) {
        nodes_to_remove_.insert(node.Index());
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
  void UpdateNodesToRemove(const std::vector<const Node*>& nodes) {
    for (auto node : nodes) {
      if (dq_output_edges_count_[node] == 0 && !nodes_to_remove_.count(node->Index())) {
        nodes_to_remove_.insert(node->Index());
      }
    }
  }

  Graph& graph_;

  std::unordered_map<const Node*, size_t> dq_output_edges_count_;

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
