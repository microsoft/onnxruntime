#include "orttraining/core/graph/graph_parser/pattern_graph.h"

#include <queue>

using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace training {

Status PatternGraph::TryMatch(Graph& target_graph, std::vector<PNN>& res) {
  Graph& pattern_graph = GetGraph();
  GraphViewer graph_viewer(target_graph);
  GraphViewer pattern_viewer(pattern_graph);
  const auto& graph_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  const auto& pattern_topology_list = pattern_viewer.GetNodesInTopologicalOrder();

  Node* pattern_root = nullptr;  // Not real root, only a root specified by user
  if (pattern_topology_list.size() && root_node().empty()) {
    pattern_root = pattern_graph.GetNode(0);
  }
  for (auto node_index : pattern_topology_list) {
    auto* node = pattern_graph.GetNode(node_index);
    if (strcmp(node->Name().c_str(), root_node().c_str()) == 0) {
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
    std::unordered_map<const Node*, const Node*> path_map;
    if (FindMatchRecursively(node, pattern_root, graph_path, pattern_path, path_map, res, target_graph)) {
      res.push_back({node_index, pattern_root});
      return Status::OK();
    }
  }

  return Status(common::ONNXRUNTIME, common::FAIL, "No match for the target graph.");
}

// let's assume that the graph and pattern do not have a circle.
bool PatternGraph::FindMatchRecursively(const Node* g, const Node* p,
                                        std::unordered_set<const Node*>& graph_path, std::unordered_set<const Node*>& pattern_path,
                                        std::unordered_map<const Node*, const Node*>& path_map,
                                        std::vector<PNN>& matched, const Graph& target) {
  PARSER_LOG << "jump in. g: " << g->Name() << "  p: " << p->Name();

  const std::string pnode_name = p->Name();
  const PNode* pnode = name_pnode_mapping_[pnode_name];
  auto func = custom_constraints_.count(pnode_name) > 0 ? custom_constraints_[pnode_name].get() : default_compare_func_.get();
  // Ensure that the two nodes have same properties
  if (!(*func)(g, pnode, target, *this)) {
    PARSER_LOG << "return not matched properties.";
    return false;
  }

  // These two are used to record the path in the recursion to avoid repeated visit of a node.
  graph_path.insert(g);
  pattern_path.insert(p);
  path_map[p] = g;

  auto look_ahead = [&path_map](const Node* cur_gnode, const Node* next_pnode, bool verify_input) {
    auto next_gnode = path_map[next_pnode];
    if (verify_input) {
      for (auto iter = cur_gnode->InputNodesBegin(); iter != cur_gnode->InputNodesEnd(); ++iter) {
        if (&(*iter) == next_gnode) return true;
      }
    } else {
      for (auto iter = cur_gnode->OutputNodesBegin(); iter != cur_gnode->OutputNodesEnd(); ++iter) {
        if (&(*iter) == next_gnode) return true;
      }
    }
    return false;
  };

  // A container to temporarily save matched results. If the total match succeed, it will be fully inserted into ```matched```.
  // If not, it will be dropped.
  std::vector<PNN> matches_if_success;

  // Different from the two set above, these two are used to record the visited nodes in current process instead of nodes in recursion.
  std::unordered_set<const Node*> visited_graph_nodes;
  std::unordered_set<const Node*> visited_pattern_nodes;

  // verify inputs
  for (auto pin_iter = p->InputNodesBegin(); pin_iter != p->InputNodesEnd(); ++pin_iter) {
    const Node& cur = *pin_iter;
    if (visited_pattern_nodes.count(&cur))
      continue;
    if (pattern_path.count(&cur)) {
      if (look_ahead(g, &cur, true)) {
        continue;
      }
      pattern_path.erase(p);
      graph_path.erase(g);
      path_map.erase(p);
      return false;
    }
    bool has_matched_branch = false;
    for (auto gin_iter = g->InputNodesBegin(); gin_iter != g->InputNodesEnd(); ++gin_iter) {
      const Node& tar = *gin_iter;
      if (!graph_path.count(&tar) && !visited_graph_nodes.count(&tar) && FindMatchRecursively(&tar, &cur, graph_path, pattern_path, path_map, matched, target)) {
        has_matched_branch = true;
        matches_if_success.push_back({tar.Index(), &cur});
        visited_graph_nodes.insert(&tar);
        visited_pattern_nodes.insert(&cur);
        break;
      }
    }
    if (!has_matched_branch) {
      PARSER_LOG << "return false.";
      pattern_path.erase(p);
      graph_path.erase(g);
      path_map.erase(p);
      return false;
    }
  }
  // verify outputs
  for (auto pout_iter = p->OutputNodesBegin(); pout_iter != p->OutputNodesEnd(); ++pout_iter) {
    const Node& cur = *pout_iter;
    if (visited_pattern_nodes.count(&cur))
      continue;
    if (pattern_path.count(&cur)) {
      if (look_ahead(g, &cur, false)) {
        continue;
      }
      pattern_path.erase(p);
      graph_path.erase(g);
      path_map.erase(p);
      return false;
    }
    bool has_matched_branch = false;
    for (auto gout_iter = g->OutputNodesBegin(); gout_iter != g->OutputNodesEnd(); ++gout_iter) {
      const Node& tar = *gout_iter;
      if (!graph_path.count(&tar) && !visited_graph_nodes.count(&tar) && FindMatchRecursively(&tar, &cur, graph_path, pattern_path, path_map, matched, target)) {
        has_matched_branch = true;
        matches_if_success.push_back({tar.Index(), &cur});
        visited_graph_nodes.insert(&tar);
        visited_pattern_nodes.insert(&cur);
        break;
      }
      graph_path.erase(&tar);
    }
    if (!has_matched_branch) {
      PARSER_LOG << "return false.";
      pattern_path.erase(p);
      graph_path.erase(g);
      path_map.erase(p);
      return false;
    }
  }

  matched.insert(matched.end(), matches_if_success.begin(), matches_if_success.end());

  PARSER_LOG << "return true.";
  return true;
}

}  // namespace training
}  // namespace onnxruntime