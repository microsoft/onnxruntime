// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/graph/pattern_graph/pattern_graph.h"

using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace training {

Node* PatternMatchResult::GetNodeByName(std::string const& node_name) const {
  auto res = matched_node_groups_.find(node_name);
  ORT_ENFORCE(res != matched_node_groups_.end(),
              "No target node has cooresponding name %s in pattern graph", node_name);
  return res->second.matched_node;
}

NodeArg* PatternMatchResult::GetInputByName(std::string const& arg_name) const {
  auto res = matched_input_groups_.find(arg_name);
  ORT_ENFORCE(res != matched_input_groups_.end(),
              "No args has cooresponding name %s in pattern graph", arg_name);
  return res->second.matched_input_arg;
}

Status PatternMatchResult::GetNodesWithCondition(
    InlinedVector<std::reference_wrapper<Node>>& filtered_nodes,
    std::function<bool(std::string const&, MatchedNodeGroup&)> filter_func) {
  for (auto iter = matched_node_groups_.begin(); iter != matched_node_groups_.end(); ++iter) {
    if (filter_func(iter->first, iter->second)) {
      filtered_nodes.push_back(*GetNodeByName(iter->first));
    }
  }

  return Status::OK();
}

PatternGraph::PatternGraph(
    const std::vector<PGraphInput>& pgraph_inputs,
    const std::vector<PGraphNode>& pgraph_nodes,
    const std::string& pattern_graph_name)
    : pattern_graph_name_(pattern_graph_name),
      pgraph_inputs_(pgraph_inputs),
      pgraph_nodes_(pgraph_nodes) {
  // Figure out what domain to version should be set for the model.
  std::unordered_map<std::string, int> domain_to_version = {{kOnnxDomain, 7}};
  for (size_t i = 0; i < pgraph_nodes.size(); i++) {
    name_to_pnode_mapping_[pgraph_nodes[i].GetNodeName()] = &(pgraph_nodes_[i]);
    const std::unordered_map<std::string, std::vector<int>>& node_domain_version_mapping = pgraph_nodes[i].GetDomainVersionMap();
    for (auto it = node_domain_version_mapping.begin(); it != node_domain_version_mapping.end(); ++it) {
      int max_version = *std::max_element(it->second.begin(), it->second.end());
      if (domain_to_version.count(it->first)) {
        domain_to_version[it->first] = std::max(domain_to_version[it->first], max_version);
      } else {
        domain_to_version[it->first] = max_version;
      }
    }
  }

  for (size_t i = 0; i < pgraph_inputs.size(); i++) {
    name_to_parg_mapping_[pgraph_inputs[i].GetArgName()] = &(pgraph_inputs_[i]);
  }

  std::vector<ONNX_NAMESPACE::FunctionProto> model_specific_functions;
  ort_model_ptr_ = std::make_unique<Model>(
      "PatternModel", false /*is_onnx_domain_only*/, ModelMetaData(), PathString(),
      IOnnxRuntimeOpSchemaRegistryList(), domain_to_version,
      model_specific_functions, logging::LoggingManager::DefaultLogger());

  ORT_ENFORCE(ToGraphInternal().IsOK());

  auto& graph = GetGraph();
  GraphViewer viewer(graph);
  for (auto node_idx : viewer.GetNodesInTopologicalOrder()) {
    auto node = graph.GetNode(node_idx);
    name_to_patten_node_mapping_[node->Name()] = node;
  }

  default_node_compare_func_ = std::make_unique<DefaultNodeCompareFunc>(false, false);
  default_arg_compare_func_ = std::make_unique<DefaultArgCompareFunc>();
}

PatternGraph& PatternGraph::SetCustomConstraint(std::unique_ptr<NodeCompareFunc> func, std::string node_name) {
  if (node_name.empty()) {
    default_node_compare_func_ = std::move(func);
    custom_node_constraints_.clear();
  } else {
    custom_node_constraints_[node_name] = std::move(func);
  }
  return *this;
}

PatternGraph& PatternGraph::SetCustomConstraint(std::unique_ptr<ArgCompareFunc> func, std::string arg_name) {
  if (arg_name.empty()) {
    default_arg_compare_func_ = std::move(func);
    custom_arg_constraints_.clear();
  } else {
    custom_arg_constraints_[arg_name] = std::move(func);
  }
  return *this;
}

Status PatternGraph::TryMatch(
    Graph& target_graph, PatternMatchResult& match_result, const std::string& node_to_start) {
  Graph& pattern_graph = GetGraph();
  GraphViewer graph_viewer(target_graph);
  GraphViewer pattern_viewer(pattern_graph);
  const auto& graph_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  const auto& pattern_topology_list = pattern_viewer.GetNodesInTopologicalOrder();

  if (graph_topology_list.size() == 0 || pattern_topology_list.size() == 0)
    return Status(common::ONNXRUNTIME, common::FAIL, "The target graph or pattern graph is empty.");

  // 1. Choose the entry node to do matching.

  // Choosing different node (in pattern graph) to start implies different search performance.
  // for simplicity, we currently choose the first node defined in pattern graph if not specified by default.
  Node* pattern_node_ptr = nullptr;
  if (node_to_start.empty()) {
    pattern_node_ptr = pattern_graph.GetNode(0);
  } else {
    for (auto node_index : pattern_topology_list) {
      auto* node = pattern_graph.GetNode(node_index);
      if (strcmp(node->Name().c_str(), node_to_start.c_str()) == 0) {
        pattern_node_ptr = node;
        break;
      }
    }
  }
  if (!pattern_node_ptr) {
    return Status(common::ONNXRUNTIME, common::FAIL, "Pattern root was not found.");
  }

  // 2. For each node in target graph, tried to find a matching recursively, returning the first match if there is.
  for (auto node_index : graph_topology_list) {
    auto* target_node_ptr = target_graph.GetNode(node_index);

    PARSER_LOG << "Pattern graph [" << pattern_graph_name_ << "] start over matching with a new entry node called [" << target_node_ptr->Name() << ", " << target_node_ptr->OutputDefs()[0]->Name() << "]";

    // Clear up previous matching results.
    match_result.Clear();

    // Define Node* sets to record the visited nodes in target graph and pattern graph respectively,
    // used to avoid duplicated match.
    std::unordered_set<const Node*> target_graph_matched_path, pattern_graph_matched_path;
    std::unordered_map<const Node*, const Node*> matched_path_pattern_node_to_target_node_map;

    if (FindMatchRecursively(
            target_graph, target_node_ptr, pattern_node_ptr,
            target_graph_matched_path, pattern_graph_matched_path,
            matched_path_pattern_node_to_target_node_map,
            match_result)) {
      // If a full match happens using current node as starting node, append the node pairs into match results.
      std::unordered_map<std::string, MatchedNodeGroup> matches;
      matches.emplace(pattern_node_ptr->Name(),
                      MatchedNodeGroup(
                          target_graph.GetNode(node_index),
                          GetPGraphNodeFromPatternNodeInputName(pattern_node_ptr->Name())));
      match_result.AppendToNodeGroups(matches);
      return Status::OK();
    }
  }

  return Status(common::ONNXRUNTIME, common::FAIL, "No match for the target graph.");
}

Status PatternGraph::ToGraphInternal() {
  /** Todo: p_initializer_names_to_preserve should contains all initializer names in the pattern graphs.
   * to avoid constant folding by any opportunaties.
   */
  const std::unordered_set<std::string>* p_initializer_names_to_preserve = nullptr;
  GraphAugmenter::GraphDefs graph_defs;

  std::unordered_map<std::string, ArgDef> arg_name_to_arg_def_mapping;
  for (size_t i = 0; i < pgraph_inputs_.size(); ++i) {
    // Use the first allowed data type when building graph.
    TypeProto* type_proto = graph_defs.CreateTypeProto();

    type_proto->mutable_tensor_type()->set_elem_type(pgraph_inputs_[i].allowed_data_types_[0]);
    auto shape = type_proto->mutable_tensor_type()->mutable_shape();
    if (pgraph_inputs_[i].CanBeAnyRank()) {
      // Can use any data shapes, so we use {"dim_0"} here.
      shape->add_dim()->set_dim_param(pgraph_inputs_[i].GetArgName() + "dim_0");
    } else {
      // Use the first allowed rank when building graph.
      for (auto i = 0; i < pgraph_inputs_[i].allowed_ranks_[0]; ++i) {
        shape->add_dim()->set_dim_param(pgraph_inputs_[i].GetArgName() + "dim_" + std::to_string(i));
      }
    }
    auto arg_def = ArgDef(pgraph_inputs_[i].GetArgName(), type_proto);

    arg_name_to_arg_def_mapping[pgraph_inputs_[i].GetArgName()] = arg_def;
  }

  std::vector<NodeDef> node_defs;  // node definitions
  for (size_t i = 0; i < pgraph_nodes_.size(); ++i) {
    std::string domain = "";
    int version = 9;
    if (!pgraph_nodes_[i].domain_version_maps_.empty()) {
      auto first_pair = pgraph_nodes_[i].domain_version_maps_.begin();
      domain = first_pair->first;
      version = *first_pair->second.begin();
    }

    std::vector<ArgDef> input_args(pgraph_nodes_[i].input_args_names_.size());
    std::vector<ArgDef> output_args(pgraph_nodes_[i].output_args_names_.size());
    for (size_t j = 0; j < pgraph_nodes_[i].input_args_names_.size(); j++) {
      // For graph input args, we don't need explicitly pass the ArgDef created above.
      // Passing the name in enough, during graph argument, it will parse the corresponding
      // TypeProto.
      std::string temporary_name = pgraph_nodes_[i].input_args_names_[j];
      if (arg_name_to_arg_def_mapping.count(temporary_name))
        input_args[j] = arg_name_to_arg_def_mapping[temporary_name];
      else
        input_args[j] = ArgDef(temporary_name);
    }

    for (size_t j = 0; j < pgraph_nodes_[i].output_args_names_.size(); j++) {
      output_args[j] = ArgDef(pgraph_nodes_[i].output_args_names_[j]);
    }

    auto node_def = NodeDef(OpDef(pgraph_nodes_[i].op_type_, domain, version),
                            input_args, output_args, pgraph_nodes_[i].attributes,
                            pgraph_nodes_[i].GetNodeName());
    node_defs.push_back(node_def);
  }

  graph_defs.AddNodeDefs(node_defs);
  Status status = GraphAugmenter::AugmentGraph(ort_model_ptr_->MainGraph(), graph_defs, p_initializer_names_to_preserve);
  ORT_ENFORCE(status.IsOK(), status.ErrorMessage());
  return Status::OK();
}

bool PatternGraph::ExactlyMatchGraphInputArgs(
    const Graph& target_graph,
    PatternGraph& pattern_graph,
    ConstPointerContainer<std::vector<NodeArg*>>& p_args,
    ConstPointerContainer<std::vector<NodeArg*>>& t_args) {
  if (p_args.size() != t_args.size()) return false;

  bool is_match = true;
  for (size_t i = 0; i < p_args.size(); ++i) {
    auto p_arg = p_args[i];
    auto t_arg = t_args[i];

    PARSER_LOG << "Try to find match for arg " << p_arg->Name();

    auto p_arg_name = p_arg->Name();
    const PGraphInput* p_arg_define = GetPGraphInputFromPatternNodeInputName(p_arg_name);
    if (p_arg_define) {
      // If this input arg is pattern graph input.
      auto func = GetArgConstraint(p_arg_name);
      if (!(*func)(target_graph, t_arg, pattern_graph, p_arg_define)) {
        is_match = false;
        break;
      }
    } else {
      // Otherwise, we do not check.
    }
  }

  return is_match;
}

bool PatternGraph::FuzzyMatchGraphInputArgs(
    const Graph& target_graph,
    PatternGraph& pattern_graph,
    ConstPointerContainer<std::vector<NodeArg*>>& p_args,
    ConstPointerContainer<std::vector<NodeArg*>>& t_args,
    size_t p_arg_idx,
    std::unordered_set<const NodeArg*>& visited) {
  if (p_arg_idx >= p_args.size()) return true;

  auto p_arg = p_args[p_arg_idx];

  PARSER_LOG << "Try to find fuzzy match for arg " << p_arg->Name();

  auto p_arg_name = p_arg->Name();
  const PGraphInput* p_arg_define = GetPGraphInputFromPatternNodeInputName(p_arg_name);
  if (!p_arg_define) {
    // If the input arg is not pattern graph input, skip current arg checks.
    return FuzzyMatchGraphInputArgs(
        target_graph, pattern_graph,
        p_args, t_args,
        p_arg_idx + 1, visited);
  }

  auto func = GetArgConstraint(p_arg_name);
  for (auto t_arg : t_args) {
    if (!visited.count(t_arg)) {
      visited.insert(t_arg);
      if ((*func)(target_graph, t_arg, pattern_graph, p_arg_define) &&
          FuzzyMatchGraphInputArgs(target_graph, pattern_graph, p_args, t_args, p_arg_idx + 1, visited)) {
        return true;
      } else {
        visited.erase(t_arg);
        continue;
      }
    }
  }
  return false;
}

bool PatternGraph::MatchNodeAndNodeInputArgs(Graph& target_graph, const Node* target_node, const Node* pattern_node) {
  // 1. Compare node properties using NodeComparisionFunc functor.
  //    Load customized comparision function if specified, otherwise, use default one.
  const std::string pnode_name = pattern_node->Name();
  const PGraphNode* pnode = name_to_pnode_mapping_[pnode_name];
  auto func = GetNodeConstraint(pnode_name);
  if (!(*func)(target_graph, target_node, *this, pnode)) {
    PARSER_LOG << "stop matching due to node properties mismatch.";
    return false;
  }

  // 2. Compare node's input args using ArgComparisionFunc functor.
  //    Be noted, we only do the compare when the input arg is pattern graph input, otherwise, we will just skip.
  //    Call different node args matching paths according to nodes' ignore node arg order attribute.
  //    Load customized comparision function if specified, otherwise, use default one.
  auto p_args = pattern_node->InputDefs();
  auto t_args = target_node->InputDefs();
  if (pnode->IgnoreNodeArgOrder()) {
    std::unordered_set<const NodeArg*> visited_args;
    if (!FuzzyMatchGraphInputArgs(target_graph, *this, p_args, t_args, 0, visited_args)) {
      PARSER_LOG << "stop matching due to node input defs mis-fuzzymatch.";
      return false;
    }
  } else {
    if (!ExactlyMatchGraphInputArgs(target_graph, *this, p_args, t_args)) {
      PARSER_LOG << "stop matching due to node input defs mismatch.";
      return false;
    }
  }
  return true;
}

bool PatternGraph::FindMatchRecursively(
    Graph& target_graph, const Node* target_node, const Node* pattern_node,
    std::unordered_set<const Node*>& target_graph_matched_path,
    std::unordered_set<const Node*>& pattern_graph_matched_path,
    std::unordered_map<const Node*, const Node*>& matched_path_pattern_node_to_target_node_map,
    PatternMatchResult& matched) {
  PARSER_LOG << "matching node for target: [" << target_node->Name() << ", " << target_node->OutputDefs()[0]->Name() << "]  pattern : [" << pattern_node->Name() << "]";

  if (!MatchNodeAndNodeInputArgs(target_graph, target_node, pattern_node)) {
    return false;
  }

  // Record nodes into matched paths in the recursion to avoid repeated visit of a node.
  target_graph_matched_path.insert(target_node);
  pattern_graph_matched_path.insert(pattern_node);

  // The map is just a mapping from nodes in "pattern_graph_matched_path" to nodes in "target_graph_matched_path", which is used for "look_ahead".
  matched_path_pattern_node_to_target_node_map[pattern_node] = target_node;

  auto look_ahead = [&matched_path_pattern_node_to_target_node_map](const Node* cur_gnode, const Node* next_pnode, bool verify_input) {
    auto next_gnode = matched_path_pattern_node_to_target_node_map[next_pnode];
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

  // A container to temporarily save matched results. If the total match succeed, it will be fully inserted into matched.
  // If not, it will be dropped.
  std::unordered_map<std::string, MatchedNodeGroup> matches_if_success;

  // Different from the two set above, these two are used to record the visited nodes in current process instead of nodes in recursion.
  std::unordered_set<const Node*> visited_graph_nodes;
  std::unordered_set<const Node*> visited_pattern_nodes;

  // TODO: constraint the order of the input nodes?
  // Iterate all input nodes of pattern_node, try to find a full match for each input node.
  for (auto pin_iter = pattern_node->InputNodesBegin(); pin_iter != pattern_node->InputNodesEnd(); ++pin_iter) {
    const Node& pattern_node_prior = *pin_iter;
    if (visited_pattern_nodes.count(&pattern_node_prior))
      continue;

    if (pattern_graph_matched_path.count(&pattern_node_prior)) {
      // look ahead when encounter a visited node
      if (look_ahead(target_node, &pattern_node_prior, true)) {
        continue;
      }
      pattern_graph_matched_path.erase(pattern_node);
      target_graph_matched_path.erase(target_node);
      matched_path_pattern_node_to_target_node_map.erase(pattern_node);
      return false;
    }

    bool has_matched_branch = false;

    // Iterate all input nodes of target_node to find a match for pattern_node_prior.
    for (auto gin_iter = target_node->InputNodesBegin(); gin_iter != target_node->InputNodesEnd(); ++gin_iter) {
      const Node& tar = *gin_iter;
      if (!target_graph_matched_path.count(&tar) && !visited_graph_nodes.count(&tar) &&
          FindMatchRecursively(target_graph, &tar, &pattern_node_prior,
                               target_graph_matched_path, pattern_graph_matched_path,
                               matched_path_pattern_node_to_target_node_map, matched)) {
        has_matched_branch = true;
        matches_if_success.emplace(pattern_node_prior.Name(), MatchedNodeGroup(target_graph.GetNode(tar.Index()), name_to_pnode_mapping_[pattern_node_prior.Name()]));
        visited_graph_nodes.insert(&tar);
        visited_pattern_nodes.insert(&pattern_node_prior);
        break;
      }
    }
    // If we cannot find match for every input nodes of p, we need to restore the scene
    if (!has_matched_branch) {
      PARSER_LOG << "return false because of no match input node match for " << pattern_node_prior.Name();
      pattern_graph_matched_path.erase(pattern_node);
      target_graph_matched_path.erase(target_node);
      matched_path_pattern_node_to_target_node_map.erase(pattern_node);
      return false;
    }
  }

  // try to find a full match for all the input nodes of the current node p in pattern graph, which is like a mirror symmetry of inputs matching.
  for (auto pout_iter = pattern_node->OutputNodesBegin(); pout_iter != pattern_node->OutputNodesEnd(); ++pout_iter) {
    const Node& cur = *pout_iter;
    if (visited_pattern_nodes.count(&cur))
      continue;
    if (pattern_graph_matched_path.count(&cur)) {
      if (look_ahead(target_node, &cur, false)) {
        continue;
      }
      pattern_graph_matched_path.erase(pattern_node);
      target_graph_matched_path.erase(target_node);
      matched_path_pattern_node_to_target_node_map.erase(pattern_node);
      return false;
    }
    bool has_matched_branch = false;
    for (auto gout_iter = target_node->OutputNodesBegin(); gout_iter != target_node->OutputNodesEnd(); ++gout_iter) {
      const Node& tar = *gout_iter;
      if (!target_graph_matched_path.count(&tar) && !visited_graph_nodes.count(&tar) &&
          FindMatchRecursively(target_graph, &tar, &cur, target_graph_matched_path, pattern_graph_matched_path, matched_path_pattern_node_to_target_node_map, matched)) {
        has_matched_branch = true;
        matches_if_success.emplace(cur.Name(), MatchedNodeGroup(target_graph.GetNode(tar.Index()), name_to_pnode_mapping_[cur.Name()]));
        visited_graph_nodes.insert(&tar);
        visited_pattern_nodes.insert(&cur);
        break;
      }
      target_graph_matched_path.erase(&tar);
    }
    if (!has_matched_branch) {
      PARSER_LOG << "return false because of no match output node match for " << cur.Name();
      pattern_graph_matched_path.erase(pattern_node);
      target_graph_matched_path.erase(target_node);
      matched_path_pattern_node_to_target_node_map.erase(pattern_node);
      return false;
    }
  }

  // Reaching here means it has been matched successfully, the nodes will be collected into result.
  matched.AppendToNodeGroups(matches_if_success);

  PARSER_LOG << "return true.";
  return true;
}

bool DefaultNodeCompareFunc::operator()(
    const Graph&, const Node* target_node,
    const PatternGraph& pattern_graph,
    const PGraphNode* pattern_node) const {
  if (!target_node && !pattern_node)
    return true;
  if (!target_node && pattern_node || !pattern_node && target_node)
    return false;
  if (!skip_op_type_ && !pattern_node->MatchesOpType(target_node->OpType())) {
    PARSER_LOG << "OpType mismatch, "
               << "target_node is: " << target_node->OpType();
    return false;
  }
  if (!skip_domain_and_version_ && !pattern_node->MatchesDomainVersion(target_node->Domain(), target_node->SinceVersion())) {
    PARSER_LOG << "Domain or Version mismatch, "
               << "target_node's domain is: " << target_node->Domain()
               << "target_node's version is: " << target_node->SinceVersion();
    return false;
  }

  if (pattern_node->output_edges_count_ == 0 &&
      target_node->GetOutputEdgesCount() != pattern_graph.GetPatternGraphNode(pattern_node->node_name_)->GetOutputEdgesCount()) {
    PARSER_LOG << "Output edges count mismatch, "
               << "target_node is: " << target_node->SinceVersion();
    return false;
  } else if (pattern_node->output_edges_count_ > 0 &&
             target_node->GetOutputEdgesCount() != static_cast<size_t>(pattern_node->output_edges_count_)) {
    PARSER_LOG << "Output edges count mismatch, "
               << "target_node is: " << target_node->SinceVersion();
    return false;
  }
  return true;
}

}  // namespace training
}  // namespace onnxruntime
