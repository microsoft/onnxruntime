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

/**
 * @brief Matched Node Groups.
 *
 *  Each group contains
 *  > matched node (of type Node*) in target graph.
 *  > graph node in pattern graph..
 */
struct MatchedNodeGroup {
 public:
  explicit MatchedNodeGroup(Node* target_node, const PGraphNode* pattern_node)
      : matched_node(target_node),
        pattern_node(pattern_node) {
  }
  Node* matched_node;
  const PGraphNode* pattern_node;
};

/**
 * @brief Matched Input Args Groups.
 *
 *  Each group contains
 *  > matched input arg (of type NodeArg*) in target graph.
 *  > graph input arg in pattern graph.
 */
struct MatchedInputGroup {
  NodeArg* matched_input_arg;
  const PGraphInput* pattern_input;
};

/**
 * @brief Match result class.
 *
 */
struct PatternMatchResult {
  friend class PatternGraph;

 public:
  Node* GetNodeByName(std::string const& node_name) const;

  NodeArg* GetInputByName(std::string const& arg_name) const;

  Status GetNodesWithCondition(InlinedVector<std::reference_wrapper<Node>>& filtered_nodes,
                               std::function<bool(std::string const&, MatchedNodeGroup&)> filter_func);

 private:
  void AppendToNodeGroups(std::unordered_map<std::string, MatchedNodeGroup> results) {
    matched_node_groups_.insert(results.begin(), results.end());
  }

  void AppendToInputGroups(std::unordered_map<std::string, MatchedInputGroup> results) {
    matched_input_groups_.insert(results.begin(), results.end());
  }

  void Clear() {
    matched_input_groups_.clear();
    matched_node_groups_.clear();
  }

  std::unordered_map<std::string, MatchedInputGroup> matched_input_groups_;
  std::unordered_map<std::string, MatchedNodeGroup> matched_node_groups_;
};

struct PatternGraph;

/**
 * @brief Pattern graph input comparison functor base class.
 * @param target_graph the Graph we target to do match on.
 * @param target_node_arg the node arg we are comparing on the target Graph.
 * @param pattern_graph the pattern graph.
 * @param pattern_node_arg the node arg in the pattern graph we are comparing with target node arg.
 */
struct ArgCompareFunc {
  virtual bool operator()(const Graph& target_graph, const NodeArg* target_node_arg,
                          const PatternGraph& pattern_graph, const PGraphInput* pattern_node_arg) const = 0;
};

/**
 * @brief Default pattern graph input comparison functor.
 */
class DefaultArgCompareFunc : public ArgCompareFunc {
 public:
  bool operator()(const Graph& target_graph, const NodeArg* target_node_arg,
                  const PatternGraph&, const PGraphInput* pattern_node_arg) const override {
    return pattern_node_arg->IsDangling() || pattern_node_arg->MatchesDataType(target_graph, *target_node_arg);
  }
};

/**
 * @brief Pattern graph node comparison functor base class.
 * @param target_graph the Graph we target to do match on.
 * @param target_node the Node we are comparing on the target Graph.
 * @param pattern_graph the pattern graph.
 * @param pattern_node the node in the pattern graph we are comparing with target node.
 */
struct NodeCompareFunc {
  virtual bool operator()(const Graph& target_graph, const Node* target_node,
                          const PatternGraph& pattern_graph, const PGraphNode* pattern_node) const = 0;
};

/**
 * @brief Default pattern graph node comparison functor.
 *
 * @param skip_op_type If true, skip the op type check.
 * @param skip_domain_and_version If True, skip the domain and version check.
 */
struct DefaultNodeCompareFunc : public NodeCompareFunc {
 public:
  DefaultNodeCompareFunc(bool skip_op_type,
                         bool skip_domain_and_version)
      : skip_op_type_(skip_op_type),
        skip_domain_and_version_(skip_domain_and_version) {}

  bool operator()(const Graph& target_graph, const Node* target_node,
                  const PatternGraph& pattern_graph, const PGraphNode* pattern_node) const override;

 private:
  bool skip_op_type_, skip_domain_and_version_;
};

struct PatternGraph {
 public:
  PatternGraph(const std::vector<PGraphInput>& pgraph_inputs,
               const std::vector<PGraphNode>& pgraph_nodes,
               const std::string& pattern_graph_name = "")
      : pattern_graph_name_(pattern_graph_name),
        pgraph_inputs_(pgraph_inputs),
        pgraph_nodes_(pgraph_nodes) {
    ort_model_ptr_ = std::make_unique<Model>("PatternModel", false, logging::LoggingManager::DefaultLogger());
    for (size_t i = 0; i < pgraph_nodes.size(); i++) {
      name_pnode_mapping_[pgraph_nodes[i].GetNodeName()] = &(pgraph_nodes_[i]);
    }
    for (size_t i = 0; i < pgraph_inputs.size(); i++) {
      name_parg_mapping_[pgraph_inputs[i].GetArgName()] = &(pgraph_inputs_[i]);
    }

    ORT_ENFORCE(ToGraphInternal().IsOK());
    auto& graph = GetGraph();
    GraphViewer viewer(graph);
    for (auto node_idx : viewer.GetNodesInTopologicalOrder()) {
      auto node = graph.GetNode(node_idx);
      name_patten_graph_node_mapping_[node->Name()] = node;
    }

    default_node_compare_func_ = std::make_unique<DefaultNodeCompareFunc>(false, false);
    default_arg_compare_func_ = std::make_unique<DefaultArgCompareFunc>();
  }

  const std::string& Name() const {
    return pattern_graph_name_;
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
  Status GetNodeDefs() {
    for (size_t i = 0; i < pgraph_nodes_.size(); i++) {
      node_defs_.push_back(pgraph_nodes_[i].GetNodeDef());
    }
    for (size_t i = 0; i < pgraph_inputs_.size(); i++) {
      node_defs_.push_back(pgraph_inputs_[i].GetNodeDef());
    }
    return Status::OK();
  }

  Status ToGraphInternal() {
    ORT_ENFORCE(GetNodeDefs().IsOK());
    /** Todo: p_initializer_names_to_preserve should contains all initializer names in the pattern graphs.
     * to avoid constant folding by any opportunaties.
     */
    const std::unordered_set<std::string>* p_initializer_names_to_preserve = nullptr;
    GraphAugmenter::GraphDefs graph_defs;
    graph_defs.AddNodeDefs(node_defs_);
    auto status = GraphAugmenter::AugmentGraph(ort_model_ptr_->MainGraph(), graph_defs, p_initializer_names_to_preserve);
    ORT_ENFORCE(status.IsOK());
    return Status::OK();
  }

  bool FindMatchRecursively(const Node* g, const Node* p,
                            std::unordered_set<const Node*>& graph_path, std::unordered_set<const Node*>& pattern_path,
                            std::unordered_map<const Node*, const Node*>& path_map,
                            PatternMatchResult& matched, Graph& target);

  std::string pattern_graph_name_;  // name of the graph

  std::vector<PGraphInput> pgraph_inputs_;  // arg definitions
  std::vector<PGraphNode> pgraph_nodes_;    // node definitions
  std::unordered_map<std::string, const PGraphNode*> name_pnode_mapping_;
  std::unordered_map<std::string, const PGraphInput*> name_parg_mapping_;

  std::vector<NodeDef> node_defs_;  // node definitions
  std::unique_ptr<Model> ort_model_ptr_;
  std::unordered_map<std::string, const Node*> name_patten_graph_node_mapping_;

  std::unique_ptr<NodeCompareFunc> default_node_compare_func_;
  std::unordered_map<std::string, std::unique_ptr<NodeCompareFunc>> custom_node_constraints_;
  std::unique_ptr<ArgCompareFunc> default_arg_compare_func_;
  std::unordered_map<std::string, std::unique_ptr<ArgCompareFunc>> custom_arg_constraints_;
};

}  // namespace training
}  // namespace onnxruntime
