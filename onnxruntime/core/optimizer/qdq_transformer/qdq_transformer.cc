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
    std::vector<const Node*> dq_nodes = graph_utils::FindParentsByType(node, DQOPTypeName);
    std::vector<const Node*> q_nodes = graph_utils::FindChildrenByType(node, QOPTypeName);

    // track DequantizeLinear output edges count
    for (auto dq_node : dq_nodes) {
      if (!dq_output_edges_count_.count(dq_node)) {
        dq_output_edges_count_[dq_node] = dq_node->GetOutputEdgesCount();
      }
    }

    std::unique_ptr<QDQOperatorTransformer> op_trans = QDQRegistry::CreateQDQTransformer(node, graph_);
    if (op_trans && op_trans->Transform(dq_nodes, q_nodes)) {
      for (auto dq_node : dq_nodes) {
        dq_output_edges_count_[dq_node]--;
      }

      UpdateNodesToRemove(dq_nodes);
      UpdateNodesToRemove(q_nodes);
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
      if (nodes_to_remove_.count(node->Index())) {
        continue;
      }

      auto it = dq_output_edges_count_.find(node);
      // Add to nodes_to_remove_ directly if it is QuantizeLinear
      if (it == dq_output_edges_count_.end()) {
        nodes_to_remove_.insert(node->Index());
      } else if (it->second == 0 &&                                     // node has no edges
                 graph_.GetNodeOutputsInGraphOutputs(*node).empty()) {  // node outputs are not graph outputs
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
