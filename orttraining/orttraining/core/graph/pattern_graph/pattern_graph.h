// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "orttraining/core/graph/graph_augmenter.h"
#include "orttraining/core/graph/pattern_graph/pattern_node.h"

#include "core/graph/contrib_ops/onnx_function_util.h"
#include "core/graph/graph_utils.h"
#include "core/graph/model.h"

#define PARSER_LOG \
  LOGS_DEFAULT(WARNING)

namespace onnxruntime {
namespace training {

// First: graph node index. Second: corresponding pattern node.
typedef std::pair<onnxruntime::NodeIndex, const onnxruntime::Node*> PatternMatchPair;

struct PatternMatchResult {
  friend class PatternGraph;

 public:
  explicit PatternMatchResult(Graph& target_graph) : target_graph_(target_graph) {}

  Node* GetTargetNodeWithName(std::string name) const {
    auto res = mapping_.find(name);
    ORT_ENFORCE(res != mapping_.end(), "No target node has cooresponding name %s in pattern graph", name);
    return target_graph_.GetNode(res->second.first);
  }

  NodeIndex GetTargetNodeIdxWithName(std::string name) const {
    auto res = mapping_.find(name);
    ORT_ENFORCE(res != mapping_.end(), "No target node has cooresponding name %s in pattern graph", name);
    return res->second.first;
  }

  PatternMatchPair GetMatchPairWithName(std::string name) const {
    auto res = mapping_.find(name);
    ORT_ENFORCE(res != mapping_.end(), "No node with name %s in pattern graph", name);
    return res->second;
  }

  const onnxruntime::Node* GetPatternNodeWithName(std::string name) const {
    auto res = mapping_.find(name);
    ORT_ENFORCE(res != mapping_.end(), "No node with name %s in pattern graph", name);
    return res->second.second;
  }

  std::map<std::string, PatternMatchPair> GetMatchMap() const {
    return mapping_;
  }

  Graph& GetTargetGraph() const {
    return target_graph_;
  }

 private:
  void InsertMatchMappings(std::map<std::string, PatternMatchPair> results) {
    mapping_.insert(results.begin(), results.end());
  }

  void AddMatchMapping(std::string pattern_node_name, PatternMatchPair result) {
    mapping_[pattern_node_name] = result;
  }

  void Clear() {
    mapping_.clear();
  }

  Graph& target_graph_;
  std::map<std::string, PatternMatchPair> mapping_;
};

struct PatternGraph {
 public:
  PatternGraph(const std::vector<PGraphInput>& constants,
               const std::vector<PGraphNode>& pnodes,
               const std::string& name = "")
      : name_(name),
        pargs(constants),
        pnodes(pnodes) {
    ort_model_ptr_ = std::make_unique<Model>("PatternModel", false, logging::LoggingManager::DefaultLogger());
    for (size_t i = 0; i < pnodes.size(); i++) {
      nodes.push_back(pnodes[i].GetNodeDef());
      name_pnode_mapping_[pnodes[i].GetNodeName()] = &(this->pnodes[i]);
    }
    for (size_t i = 0; i < constants.size(); i++) {
      nodes.push_back(constants[i].GetNodeDef());
      name_parg_mapping_[pargs[i].GetArgName()] = &(this->pargs[i]);
    }

    ORT_ENFORCE(ToGraphInternal().IsOK());
    auto& graph = GetGraph();
    GraphViewer viewer(graph);
    for (auto node_idx : viewer.GetNodesInTopologicalOrder()) {
      auto node = graph.GetNode(node_idx);
      name_patten_graph_node_mapping_[node->Name()] = node;
    }

    default_node_compare_func_ = std::make_unique<DefaultNodeCompareFunc>(false, false, false);
    default_arg_compare_func_ = std::make_unique<DefaultArgCompareFunc>();
  }

  const std::string& Name() const {
    return name_;
  }

  PatternGraph& SetCustomConstraint(std::unique_ptr<NodeCompareFunc> func, std::string node_name = "") {
    if (node_name.empty()) {
      default_node_compare_func_ = std::move(func);
      custom_node_constraints_.clear();
    } else {
      custom_node_constraints_[node_name] = std::move(func);
    }
    return *this;
  }

  PatternGraph& SetCustomConstraint(std::unique_ptr<ArgCompareFunc> func, std::string arg_name = "") {
    if (arg_name.empty()) {
      default_arg_compare_func_ = std::move(func);
      custom_arg_constraints_.clear();
    } else {
      custom_arg_constraints_[arg_name] = std::move(func);
    }
    return *this;
  }

  Graph& GetGraph() {
    return ort_model_ptr_->MainGraph();
  }

  const Node* GetPatternGraphNode(const std::string& name) const {
    auto res = name_patten_graph_node_mapping_.find(name);
    ORT_ENFORCE(res != name_patten_graph_node_mapping_.end(), "No pattern node named %s", name);
    return res->second;
  }

  Status TryMatch(Graph& target_graph, PatternMatchResult& res, const std::string& root_node = "");

 private:
  Status ToGraphInternal() {
    // Todo: p_initializer_names_to_preserve should contains all names in the pattern graphs.
    const std::unordered_set<std::string>* p_initializer_names_to_preserve = nullptr;
    GraphAugmenter::GraphDefs graph_defs;
    graph_defs.AddNodeDefs(nodes);
    auto status = GraphAugmenter::AugmentGraph(ort_model_ptr_->MainGraph(), graph_defs, p_initializer_names_to_preserve);
    ORT_ENFORCE(status.IsOK());
    // ORT_ENFORCE(GraphAugmenter::AugmentGraph(ort_model_ptr_->MainGraph(), graph_defs, p_initializer_names_to_preserve).IsOK());

    return Status::OK();
  }

  bool FindMatchRecursively(const Node* g, const Node* p,
                            std::unordered_set<const Node*>& graph_path, std::unordered_set<const Node*>& pattern_path,
                            std::unordered_map<const Node*, const Node*>& path_map,
                            PatternMatchResult& matched, const Graph& target);

  std::string name_;  // name of the graph

  std::vector<PGraphInput> pargs;  // arg definitions
  std::vector<NodeDef> nodes;      // node definitions
  std::vector<PGraphNode> pnodes;  // node definitions
  std::unique_ptr<Model> ort_model_ptr_;

  std::unordered_map<std::string, const Node*> name_patten_graph_node_mapping_;
  std::unordered_map<std::string, const PGraphNode*> name_pnode_mapping_;
  std::unordered_map<std::string, const PGraphInput*> name_parg_mapping_;

  std::unique_ptr<NodeCompareFunc> default_node_compare_func_;
  std::unordered_map<std::string, std::unique_ptr<NodeCompareFunc>> custom_node_constraints_;
  std::unique_ptr<ArgCompareFunc> default_arg_compare_func_;
  std::unordered_map<std::string, std::unique_ptr<ArgCompareFunc>> custom_arg_constraints_;
};

}  // namespace training
}  // namespace onnxruntime
