#include "orttraining/core/graph/graph_parser.h"

#include <queue>

using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace training {

bool PatternGraph::node_equal_properties(const Node* g, const Node* p, const Graph& target,
                                         const Graph& pattern) {
  if (!g && !p)
    return true;
  if (!g && p || !p && g)
    return false;
  if (check_optype && strcmp(g->OpType().c_str(), p->OpType().c_str()) != 0)
    return false;
  if (check_domain && domain_map.count(p->Name()) && domain_map[p->Name()].size() && !domain_map[p->Name()].count(g->Domain()))
    return false;
  if (check_version && version_map.count(p->Name()) && version_map[p->Name()].size() && !version_map[p->Name()].count(g->SinceVersion()))
    return false;
  for (auto& func : customized_constriants) {
    if (!func(g, p, target, pattern)) return false;
  }

  return true;
}

// let's assume that the graph and pattern do not have a circle.
bool PatternGraph::find_match(const Node* g, const Node* p, std::unordered_set<const Node*>& graph_path, std::unordered_set<const Node*>& pattern_path,
                              std::vector<PNN>& matched, const Graph& target, const Graph& pattern) {
  PARSER_MSG(std::cout << "jump in. g: " << g->Name() << "  p: " << p->Name() << std::endl;)
  // Ensure that the two nodes have same properties
  if (!node_equal_properties(g, p, target, pattern)) {
    PARSER_MSG(std::cout << "return not matched properties." << std::endl;)
    return false;
  }

  // These two are used to record the path in the recursion to avoid repeated visit of a node.
  graph_path.insert(g);
  pattern_path.insert(p);

  // A container to temporarily save matched results. If the total match succeed, it will be fully inserted into ```matched```.
  // If not, it will be dropped.
  std::vector<PNN> matches_if_success;

  // Different from the two set above, these two are used to record the visited nodes in current process instead of nodes in recursion.
  std::unordered_set<const Node*> visited_graph_nodes;
  std::unordered_set<const Node*> visited_pattern_nodes;

  // verify inputs
  for (auto pin_iter = p->InputNodesBegin(); pin_iter != p->InputNodesEnd(); ++pin_iter) {
    const Node& cur = *pin_iter;
    if (visited_pattern_nodes.count(&cur) || pattern_path.count(&cur))
      continue;
    bool has_matched_branch = false;
    for (auto gin_iter = g->InputNodesBegin(); gin_iter != g->InputNodesEnd(); ++gin_iter) {
      const Node& tar = *gin_iter;
      if (!graph_path.count(&tar) && !visited_graph_nodes.count(&tar) && find_match(&tar, &cur, graph_path, pattern_path, matched, target, pattern)) {
        has_matched_branch = true;
        matches_if_success.push_back({tar.Index(), &cur});
        visited_graph_nodes.insert(&tar);
        visited_pattern_nodes.insert(&cur);
        break;
      }
    }
    if (!has_matched_branch) {
      PARSER_MSG(std::cout << "return false." << std::endl;)
      pattern_path.erase(p);
      graph_path.erase(g);
      return false;
    }
  }
  // verify outputs
  for (auto pout_iter = p->OutputNodesBegin(); pout_iter != p->OutputNodesEnd(); ++pout_iter) {
    const Node& cur = *pout_iter;
    if (visited_pattern_nodes.count(&cur) || pattern_path.count(&cur))
      continue;
    bool has_matched_branch = false;
    for (auto gout_iter = g->OutputNodesBegin(); gout_iter != g->OutputNodesEnd(); ++gout_iter) {
      const Node& tar = *gout_iter;
      if (!pattern_path.count(&cur) && !graph_path.count(&tar) && !visited_graph_nodes.count(&tar) && find_match(&tar, &cur, graph_path, pattern_path, matched, target, pattern)) {
        has_matched_branch = true;
        matches_if_success.push_back({tar.Index(), &cur});
        visited_graph_nodes.insert(&tar);
        visited_pattern_nodes.insert(&cur);
        break;
      }
      graph_path.erase(&tar);
    }
    if (!has_matched_branch) {
      PARSER_MSG(std::cout << "return false." << std::endl;)
      pattern_path.erase(p);
      graph_path.erase(g);
      return false;
    }
  }

  matched.insert(matched.end(), matches_if_success.begin(), matches_if_success.end());

  PARSER_MSG(std::cout << "return true." << std::endl;)
  return true;
}

Status PatternGraph::TryMatch(Graph& target_graph, std::vector<PNN>& res,
                              const std::unordered_set<std::string>* p_initializer_names_to_preserve) {
  ORT_RETURN_IF_ERROR(to_graph(model, p_initializer_names_to_preserve));
  auto& pattern_graph = model.MainGraph();

  GraphViewer graph_viewer(target_graph);

  GraphViewer pattern_viewer(pattern_graph);
  const auto& graph_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  const auto& pattern_topology_list = pattern_viewer.GetNodesInTopologicalOrder();

  Node* pattern_root = nullptr;  // Not real root, only a root specified by user
  if (pattern_topology_list.size() && root_node.empty()) {
    pattern_root = pattern_graph.GetNode(0);
  }
  for (auto node_index : pattern_topology_list) {
    auto* node = pattern_graph.GetNode(node_index);
    if (strcmp(node->Name().c_str(), root_node.c_str()) == 0) {
      pattern_root = node;
      break;
    }
  }
  if (!pattern_root) {
    return Status(common::ONNXRUNTIME, common::FAIL, "Pattern root was not found.");
  }

  for (auto node_index : graph_topology_list) {
    auto* node = target_graph.GetNode(node_index);
    res.clear();
    std::unordered_set<const Node*> graph_path, pattern_path;
    if (find_match(node, pattern_root, graph_path, pattern_path, res, target_graph, pattern_graph)) {
      res.push_back({node_index, pattern_root});
      return Status::OK();
    }
  }

  return Status(common::ONNXRUNTIME, common::FAIL, "No match for the target graph.");
}

Status PatternGraph::TryReplace(Graph& graph, const NodeDef& alternative, std::vector<std::pair<std::string, int>> fusion_inputs,
                                std::vector<std::pair<std::string, int>> fusion_outputs,
                                const std::unordered_set<std::string>* p_initializer_names_to_preserve) {
  std::vector<PNN> match_results;
  ORT_RETURN_IF_ERROR(TryMatch(graph, match_results, p_initializer_names_to_preserve));
  InlinedVector<std::reference_wrapper<Node>> matched_nodes;
  for (auto iter = match_results.rbegin(); iter != match_results.rend(); iter++) {
    auto node = graph.GetNode(iter->first);
    matched_nodes.push_back(*node);
  }

  auto add_node_args_with_name = [&match_results](Graph& graph, std::string name, int idx, std::vector<NodeArg*>& args) {
    GraphViewer viewer(graph);
    // const auto& node_topology_list = viewer.GetNodesInTopologicalOrder();
    for (auto [node_index, pattern_node] : match_results) {
      if (pattern_node->Name() == name) {
        auto target_node = graph.GetNode(node_index);
        args.push_back(target_node->MutableInputDefs()[idx]);
        return Status::OK();
      }
    }
    return Status(common::ONNXRUNTIME, common::FAIL);
  };

  std::vector<NodeArg*> input_args, output_args;
  for (auto item : fusion_inputs) {
    ORT_RETURN_IF_ERROR(add_node_args_with_name(graph, item.first, item.second, input_args));
  }
  for (auto item : fusion_outputs) {
    ORT_RETURN_IF_ERROR(add_node_args_with_name(graph, item.first, item.second, output_args));
  }
  Node& replace_node = graph.AddNode(alternative.name, alternative.op_type, "",
                                     input_args, output_args, {}, alternative.domain.c_str());
  for (auto& attr : alternative.attributes) {
    replace_node.AddAttribute(attr.first, attr.second);
  }
  graph_utils::FinalizeNodeFusion(graph, matched_nodes, replace_node);
  return Status::OK();
}

}  // namespace training
}  // namespace onnxruntime