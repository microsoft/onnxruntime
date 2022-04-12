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
 * @brief Matched Node Group.
 *
 *  Each group contains
 *  > matched node (of type Node*) in target graph.
 *  > graph node in pattern graph.
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
 * @brief Matched Input Args Group.
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

/**
 * @brief PatternGraph class.
 * Class representing a pattern to match in target Graph.
 *
 * Under the hood, this PatternGraph will be transformed into ORT Graph structure
 * leveraging GraphAugmenter.
 */
struct PatternGraph {
  friend struct DefaultNodeCompareFunc;

 public:
  /**
   * @brief Construct a new PatternGraph object.
   *
   * @param pgraph_inputs graph inputs (node args) informations given to do input match.
   * @param pgraph_nodes graph nodes informations given to do node match.
   * @param pattern_graph_name [optional] graph name for this pattern.
   */
  PatternGraph(const std::vector<PGraphInput>& pgraph_inputs,
               const std::vector<PGraphNode>& pgraph_nodes,
               const std::string& pattern_graph_name = "");

  /**
   * @brief Get the Graph object created with given graph inputs/nodes descriptions.
   */
  Graph& GetGraph() {
    return ort_model_ptr_->MainGraph();
  }

  /**
   * @brief Set the CustomConstraint by node name.
   *
   * This is used to define customized constraint on PatternGraph's nodes.
   * @param func unique pointer of an constraint instance inheriting from NodeCompareFunc.
   * @param node_name node name that applied this constraint.
   *    If not specified, all nodes will use the func for node comparison.
   * @return PatternGraph&
   */
  PatternGraph& SetCustomConstraint(std::unique_ptr<NodeCompareFunc> func, std::string node_name = "");

  /**
   * @brief Set the CustomConstraint by node arg name.
   *
   * This is used to define customized constraint on PatternGraph's inputs (e.g. node args).
   * @param func unique pointer of an constraint instance inheriting from ArgCompareFunc.
   * @param arg_name graph input name that applied this constraint.
   *    If not specified, all graph inputs will use the func for arg comparison.
   * @return PatternGraph&
   */
  PatternGraph& SetCustomConstraint(std::unique_ptr<ArgCompareFunc> func, std::string arg_name = "");

  /**
   * @brief Core pattern match function.
   *
   * @param target_graph The target graph we do pattern match on.
   * @param res data structure storing the matched results.
   * @param root_node [optional] an starting entry point to do matching. If not specified, default using
   *    the first node defined in PatternGraph. Using this could help make the graph matching faster
   *    if a good starting_node is chosen.
   * @return Status
   */
  Status TryMatch(Graph& target_graph, PatternMatchResult& res, const std::string& root_node = "");

 private:
  /**
   * @brief Get the NodeDefs from given node descriptions.
   * Be noted, all graph inputs (e.g. pgraph_inputs) are defined as Constant nodes.
   */
  Status GetNodeDefs();

  /**
   * @brief Create ORT Graph object using GraphAugmenter.
   */
  Status ToGraphInternal();

  /**
   * @brief Search in the graph for a match with two given nodes as start.
   *
   * This function searches in the graph for a match with two given nodes as start.
   * @param target_graph is the target graph which should be passed by the user.
   * @param target_node is the node we want to start from in target graph.
   * @param pattern_node is that of pattern graph.
   * @param target_graph_matched_path  a set to record the visited nodes in target graph, used to avoid duplicated match.
   * @param pattern_graph_matched_path a set to record the visited nodes in pattern graph, used to avoid duplicated match.
   * @param matched_path_pattern_node_to_target_node_map is a mapping from nodes in graph_path and its corresponding matched node in pattern graph.
   *    It's used to look ahead when the match encounters visited nodes to avoid wrong match.
   *    We give a simple example below to indicate the condition and effect of "look ahead".
   *    C <----------- B
   *    |              ^
   *    v              |
   *    D ---> E1 <--- A
   *    |
   *    v
   *    E2
   *    |
   *    v
   *    F
   *    We start search from A, then B, C and D and all these nodes find a match.
   *    Then we want to find match for E1 and E2, which are all the same except the position.
   *    If we do not look ahead, we find E1 reach the end of recursion (A has been visited) and return true.
   *    Then We may actually match E1 with a node which should match E2.
   *    So we need to look ahead to make sure that the visited node (A) E1 finds is exactly the matched nodes
   *    for that of target graph. In this way, we can make sure that E1 and E2 are correctly matched.
   * @param matched_result
   *
   * The main idea of the algorithm is that if we regard target node (of target graph) and pattern node (of pattern graph) as a match,
   * they must satisfy two requiments.
   * > Target node and pattern node have same properties, including optype, version, domain and so on.
   * > All the neighbor nodes of pattern node could find a match among neighbor nodes of target node.
   *
   * @return true when g and p matches.
   * @return false when g and p does not match.
   */
  bool FindMatchRecursively(Graph& target_graph,
                            const Node* g, const Node* p,
                            std::unordered_set<const Node*>& target_graph_matched_path,
                            std::unordered_set<const Node*>& pattern_graph_matched_path,
                            std::unordered_map<const Node*, const Node*>& matched_path_pattern_node_to_target_node_map,
                            PatternMatchResult& matched_result);

  const Node* GetPatternGraphNode(const std::string& node_name) const {
    auto res = name_to_patten_node_mapping_.find(node_name);
    ORT_ENFORCE(res != name_to_patten_node_mapping_.end(), "No pattern node named %s", node_name);
    return res->second;
  }

  ArgCompareFunc* GetCustomArgConstraint(std::string const& arg_name) {
    return custom_arg_constraints_.count(arg_name) > 0 ? custom_arg_constraints_.find(arg_name)->second.get() : default_arg_compare_func_.get();
  }

  NodeCompareFunc* GetCustomNodeConstraint(std::string const& node_name) {
    return custom_node_constraints_.count(node_name) > 0 ? custom_node_constraints_.find(node_name)->second.get() : default_node_compare_func_.get();
  }

  bool MatchNodeAndNodeInputArgs(Graph& target_graph, const Node* target_node, const Node* pattern_node);

  /**
   * @brief Strictly node args comparision, considering the order of node arg.
   *
   * @param target_graph
   * @param pattern_graph
   * @param name_to_pargs_mapping
   * @param p_args
   * @param t_args
   * @param arg_constraints
   * @param arg_default_constraint
   * @return true
   * @return false
   */
  bool ExactlyMatchNodeArgs(
      const Graph& target_graph,
      PatternGraph& pattern_graph,
      ConstPointerContainer<std::vector<NodeArg*>>& p_args,
      ConstPointerContainer<std::vector<NodeArg*>>& t_args);

  /**
   * Try to find a match for all the args of a node. We just use brute force here to iterate all the possible cases.
   * Let's assume that args of the node in pattern graph are a1, a2, ..., ai, ... an and that of target graph is b1, b2, ..., bj, ... bn
   * Then we keep the order of {ai} fixed and try to match it with {bj}.
   * We first choose b1, then choose an arg in {bj | j != 1, j <= n} as next arg and the recurse it.
   * "visited" is used to record visited args to avoid duplicated conditions.
   */
  /**
   *
   */
  bool FuzzyMatchNodeArgs(
      const Graph& target_graph,
      PatternGraph& pattern_graph,
      ConstPointerContainer<std::vector<NodeArg*>>& p_args,
      ConstPointerContainer<std::vector<NodeArg*>>& t_args,
      size_t p_arg_idx,
      std::unordered_set<const NodeArg*>& visited);

  const PGraphInput* GetPGraphInputFromPatternNodeInputName(std::string const& p_arg_name) {
    auto find_defined_arg = name_to_parg_mapping_.find(p_arg_name);
    if (find_defined_arg != name_to_parg_mapping_.end())
      return find_defined_arg->second;
    return nullptr;
  }

  const PGraphNode* GetPGraphNodeFromPatternNodeInputName(std::string const& p_node_name) {
    auto find_defined_pnode = name_to_pnode_mapping_.find(p_node_name);
    if (find_defined_pnode != name_to_pnode_mapping_.end())
      return find_defined_pnode->second;

    return nullptr;
  }

  std::string pattern_graph_name_;  // name of the graph

  std::vector<PGraphInput> pgraph_inputs_;  // arg definitions
  std::vector<PGraphNode> pgraph_nodes_;    // node definitions
  std::unordered_map<std::string, const PGraphNode*> name_to_pnode_mapping_;
  std::unordered_map<std::string, const PGraphInput*> name_to_parg_mapping_;

  std::vector<NodeDef> node_defs_;  // node definitions
  std::unique_ptr<Model> ort_model_ptr_;
  std::unordered_map<std::string, const Node*> name_to_patten_node_mapping_;

  std::unique_ptr<NodeCompareFunc> default_node_compare_func_;
  std::unordered_map<std::string, std::unique_ptr<NodeCompareFunc>> custom_node_constraints_;
  std::unique_ptr<ArgCompareFunc> default_arg_compare_func_;
  std::unordered_map<std::string, std::unique_ptr<ArgCompareFunc>> custom_arg_constraints_;
};

}  // namespace training
}  // namespace onnxruntime
