#include "orttraining/core/graph/graph_parser/pattern_graph.h"

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
  if (check_optype && strcmp(g->OpType().c_str(), p->OpType().c_str()) != 0) {
    PARSER_MSG(std::cout << "OpType mismatch, "
                         << "g is: " << g->OpType() << ", p is: " << p->OpType() << std::endl;)
    return false;
  }
  if (check_domain && domain_map.count(p->Name()) && domain_map[p->Name()].size() && !domain_map[p->Name()].count(g->Domain())) {
    PARSER_MSG(std::cout << "Domain mismatch, "
                         << "g is: " << g->Domain() << ", p is: " << p->Domain() << std::endl;)
    return false;
  }
  if (check_version && version_map.count(p->Name()) && version_map[p->Name()].size() && !version_map[p->Name()].count(g->SinceVersion())) {
    PARSER_MSG(std::cout << "Version mismatch, "
                         << "g is: " << g->SinceVersion() << ", p is: " << p->SinceVersion() << std::endl;)
    return false;
  }
  for (auto& func : customized_constriants) {
    if (!func(g, p, target, pattern)) return false;
  }

  return true;
}

// let's assume that the graph and pattern do not have a circle.
bool PatternGraph::find_match(const Node* g, const Node* p, std::unordered_set<const Node*>& graph_path,
                              std::unordered_set<const Node*>& pattern_path, std::unordered_map<const Node*, const Node*>& path_map,
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
      if (!graph_path.count(&tar) && !visited_graph_nodes.count(&tar) && find_match(&tar, &cur, graph_path, pattern_path, path_map, matched, target, pattern)) {
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
      if (!graph_path.count(&tar) && !visited_graph_nodes.count(&tar) && find_match(&tar, &cur, graph_path, pattern_path, path_map, matched, target, pattern)) {
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
      path_map.erase(p);
      return false;
    }
  }

  matched.insert(matched.end(), matches_if_success.begin(), matches_if_success.end());

  PARSER_MSG(std::cout << "return true." << std::endl;)
  return true;
}

}  // namespace training
}  // namespace onnxruntime