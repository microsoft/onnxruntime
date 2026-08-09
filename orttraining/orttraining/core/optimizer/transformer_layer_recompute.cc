// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/optimizer/transformer_layer_recompute.h"
#include "orttraining/core/optimizer/dropout_recompute.h"
#include "orttraining/core/graph/recompute_graph_utils.h"
#include "core/common/common.h"

#include <deque>

namespace onnxruntime {
Status TransformerLayerRecompute::IdentifyTransformerLayerEdges(
    const Graph& graph,
    std::vector<std::pair<const NodeArg*, const NodeArg*>>& start_end_edges,
    const logging::Logger& logger) const {
  const InlinedHashSet<std::string_view> gelu_ops{"Gelu", "BiasGelu", "FastGelu"};
  const InlinedHashSet<std::string_view> dropout_ops{"Dropout", "BiasDropout"};
  const InlinedHashSet<std::string_view> layernorm_ops{"LayerNormalization", "SkipLayerNormalization"};

  std::vector<const NodeArg*> layer_start_edges, layer_end_edges;
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  for (auto node_index : node_topology_list) {
    auto& node = *graph.GetNode(node_index);

    // Look for start of a transformer layer
    if ((layernorm_ops.find(node.OpType()) != layernorm_ops.end() ||
         dropout_ops.find(node.OpType()) != dropout_ops.end()) &&
        node.GetOutputEdgesCount() == 4) {
      layer_start_edges.push_back(node.OutputDefs()[0]);
    }

    // Look for end of a transformer layer
    if (gelu_ops.find(node.OpType()) != gelu_ops.end()) {
      auto next_node = node.OutputNodesBegin();

      while (next_node->OutputNodesBegin() != next_node->OutputNodesEnd() &&
             dropout_ops.find(next_node->OpType()) == dropout_ops.end()) {
        next_node = next_node->OutputNodesBegin();
      }

      while (next_node->OutputNodesBegin() != next_node->OutputNodesEnd() &&
             layernorm_ops.find(next_node->OpType()) == layernorm_ops.end()) {
        next_node = next_node->OutputNodesBegin();
      }

      if (layernorm_ops.find(next_node->OpType()) != layernorm_ops.end()) {
        layer_end_edges.push_back(next_node->OutputDefs()[0]);
      }
    }
  }

  ORT_RETURN_IF_NOT(layer_start_edges.size() == layer_end_edges.size(),
                    "Number of start and end edges doesn't match!, #start=", layer_start_edges.size(),
                    ", #end=", layer_end_edges.size());

  start_end_edges.clear();

  LOGS(logger, INFO) << "Found " << layer_start_edges.size() << " transformer layers.";
  for (size_t i = 0; i < layer_start_edges.size(); ++i) {
    start_end_edges.push_back({layer_start_edges[i], layer_end_edges[i]});
    LOGS(logger, INFO) << "Start edge: " << layer_start_edges[i]->Name() << " End edge: " << layer_end_edges[i]->Name();
  }

  return Status::OK();
}

namespace {

typedef std::set<const Node*, NodeCompare> NodeSet;

NodeSet BFSFrom(const std::vector<const Node*>& start_nodes, bool reverse) {
  NodeSet visited(start_nodes.begin(), start_nodes.end());
  std::deque<const Node*> queue(start_nodes.begin(), start_nodes.end());
  while (!queue.empty()) {
    const Node* n = queue.front();
    queue.pop_front();

    auto begin = reverse ? n->InputNodesBegin() : n->OutputNodesBegin();
    auto end = reverse ? n->InputNodesEnd() : n->OutputNodesEnd();

    for (auto node_it = begin; node_it != end; ++node_it) {
      const Node& node = *node_it;
      if (visited.find(&node) == visited.end()) {
        queue.push_back(&node);
        visited.insert(&node);
      }
    }
  }
  return visited;
}
}  // namespace

std::vector<const Node*> TransformerLayerRecompute::NodesBetweenEdges(const Graph& graph, const NodeArg* start, const NodeArg* end) const {
  // Forward BFS from the start node
  std::vector<const Node*> start_nodes = graph.GetConsumerNodes(start->Name());
  NodeSet fw_visited = BFSFrom(start_nodes, /*reverse*/ false);

  // Reverse BFS from the end node
  const Node* end_node = graph.GetProducerNode(end->Name());
  NodeSet bw_visited = BFSFrom({end_node}, /*reverse*/ true);

  // Join fw_visited and bw_visited
  std::vector<const Node*> intersect_nodes;
  std::set_intersection(fw_visited.begin(), fw_visited.end(),
                        bw_visited.begin(), bw_visited.end(),
                        std::back_inserter(intersect_nodes), NodeCompare());

  return intersect_nodes;
}

void TransformerLayerRecompute::InsertRecomputeNodes(Graph& graph, const std::vector<const Node*>& nodes, int priority) const {
  auto initializers = graph.GetAllInitializedTensors();

  for (const Node* n : nodes) {
    Node* node = graph.GetNode(n->Index());

    // recomputed Dropout need to produce the same output as original dropout
    // currently reusing original dropout's mask to achieve this
    if (node->OpType() == "Dropout") {
      const NodeArg* input = node->InputDefs()[0];
      const Node* p_node = graph.GetProducerNode(input->Name());

      bool use_original_input =
          initializers.find(input->Name()) != initializers.end() ||
          std::find(nodes.begin(), nodes.end(), p_node) == nodes.end();

      Node& recompute_node = InsertDropoutRecompute(graph, *node, use_original_input);
      recompute_node.SetPriority(priority);
      continue;
    }

    // prepare inputs for recompute node
    std::vector<NodeArg*> recomputed_inputs;
    for (NodeArg* input : node->MutableInputDefs()) {
      const Node* p_node = graph.GetProducerNode(input->Name());

      // do not duplicate initializers in recompute subgraph
      if (initializers.find(input->Name()) != initializers.end() ||
          std::find(nodes.begin(), nodes.end(), p_node) == nodes.end()) {
        recomputed_inputs.push_back(input);
      } else {
        auto& recomputed_input = graph.GetOrCreateNodeArg(graph_utils::RecomputeName(input->Name()),
                                                          input->TypeAsProto());
        recomputed_inputs.push_back(&recomputed_input);
      }
    }

    // prepare ouputs for recompute node
    std::vector<NodeArg*> recomputed_outputs;
    for (NodeArg* output : node->MutableOutputDefs()) {
      auto& recomputed_output = graph.GetOrCreateNodeArg(graph_utils::RecomputeName(output->Name()),
                                                         output->TypeAsProto());
      recomputed_outputs.push_back(&recomputed_output);
    }

    Node& recompute_node = graph.AddNode(node->Name() + "_recompute",
                                         node->OpType(),
                                         "Recompute of " + node->Name(),
                                         recomputed_inputs,
                                         recomputed_outputs,
                                         &node->GetAttributes(),
                                         node->Domain());
    recompute_node.SetPriority(priority);
  }
  return;
}

Status TransformerLayerRecompute::ApplyImpl(Graph& graph, bool& modified, int /*graph_level*/, const logging::Logger& logger) const {
  std::vector<std::pair<const NodeArg*, const NodeArg*>> start_end_edges;

  Status s = IdentifyTransformerLayerEdges(graph, start_end_edges, logger);
  if (!s.IsOK()) {
    modified = false;
    return Status::OK();
  }

  // by default, apply recompute expect for the last transformer layer
  // otherwise, take user specified 'number_recompute_layers_'

  int n_layers;
  const int n_layers_limit = static_cast<int>(start_end_edges.size() - 1);
  if (number_recompute_layers_ > n_layers_limit) {
    LOGS(logger, WARNING) << "User specified number_recompute_layers " << number_recompute_layers_
                          << " is larger than limit " << n_layers_limit << "."
                          << "number_recompute_layers is now cliped to limit.";
    n_layers = n_layers_limit;
  } else if (number_recompute_layers_ > 0) {
    n_layers = number_recompute_layers_;
  } else {
    LOGS(logger, INFO) << "number_recompute_layers is not set by user, using default " << n_layers_limit << ".";
    n_layers = n_layers_limit;
  }

  // latter recompute layers have higher execution priorty
  for (int i = 0; i < n_layers; ++i) {
    std::vector<const Node*> nodes = NodesBetweenEdges(graph, start_end_edges[i].first, start_end_edges[i].second);
    InsertRecomputeNodes(graph, nodes, static_cast<int>(start_end_edges.size() - i));
  }

  modified = true;
  return Status::OK();
}

}  // namespace onnxruntime
