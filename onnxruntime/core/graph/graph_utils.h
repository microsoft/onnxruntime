// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_MINIMAL_BUILD)
#include "onnx/defs/schema.h"
#else
#include "onnx/defs/data_type_utils.h"
#endif
#include "onnx/onnx_pb.h"
#include "onnx/onnx-operators_pb.h"

#include "core/common/inlined_containers.h"
#include "core/graph/graph.h"

namespace onnxruntime {
namespace graph_utils {

/** Checks if the operator's type, version, and domain of the given node match the given values.
 * @remarks Use kOnnxDomain and not kOnnxDomainAlias for ONNX operators.
 */
bool IsSupportedOptypeVersionAndDomain(const Node& node,
                                       std::string_view op_type,
                                       std::initializer_list<ONNX_NAMESPACE::OperatorSetVersion> versions,
                                       std::string_view domain = kOnnxDomain);

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

/** Returns the attribute of a Node with a given name. */
const ONNX_NAMESPACE::AttributeProto* GetNodeAttribute(const Node& node, const std::string& attr_name);

/** Add a new initializer to 'graph'.
Checks that new_initializer does not already exist in 'graph' before adding it.
@returns The NodeArg for the new initializer.
@remarks No matching graph input is created, so the initializer will be constant.
*/
NodeArg& AddInitializer(Graph& graph, const ONNX_NAMESPACE::TensorProto& new_initializer);

/** Gets the index of an output arg with the specified output arg name. */
int GetNodeOutputIndexFromOutputName(const Node& node, const std::string& output_name);

/** Find node parents by op types.
    @returns The matched parents are sorted by destination argument index of their corresponding edge.
**/
std::vector<const Node*> FindParentsByType(const Node& node, const std::string& parent_type);

/** Find node children by op types.
    @returns The matched children are sorted by source argument index of their corresponding edge.
**/
std::vector<const Node*> FindChildrenByType(const Node& node, const std::string& child_type);

/** Gets the name of the incoming NodeArg with the specified index for the given node. */
const std::string& GetNodeInputName(const Node& node, int index);

/** Gets the name of the outgoing NodeArg with the specified index for the given node. */
const std::string& GetNodeOutputName(const Node& node, int index);

/** Find the input edge of a node for a specified input index.
@returns nullptr when not found.
*/
const Node::EdgeEnd* GetInputEdge(const Node& node, int arg_index);

/** Removes all output edges from the given Node of the Graph.
    This should probably be elevated to the Graph API eventually. */
size_t RemoveNodeOutputEdges(Graph& graph, Node& node);

/** Removes output edges from the specific output_idx for the given Node of the Graph. */
size_t RemoveNodeOutputEdges(Graph& graph, Node& node, int output_idx);

/** returns the initializer's TensorProto if 'name' is an initializer, is constant and
cannot be overridden at runtime. If the initializer is not found or is not constant, a nullptr is returned.
@param check_outer_scope If true and the graph is a subgraph, check ancestor graph/s for 'name' if not found in 'graph'.
*/
const ONNX_NAMESPACE::TensorProto* GetConstantInitializer(const Graph& graph, const std::string& name,
                                                          bool check_outer_scope = true);

// A class helps handle graph edges
struct GraphEdge {
  NodeIndex src_node;
  NodeIndex dst_node;
  int src_arg_index;
  int dst_arg_index;
  std::string arg_name;

  GraphEdge(NodeIndex src_node, NodeIndex dst_node, int src_arg_index, int dst_arg_index, const std::string& arg_name);

  // Constructs a GraphEdge given a node, an edge_end, and a boolean for the edge direction.
  static GraphEdge CreateGraphEdge(const Node& node, const Node::EdgeEnd& edge_end, bool is_input_edge);

  /** Returns a vector of the input GraphEdges of a node. */
  static std::vector<GraphEdge> GetNodeInputEdges(const Node& node);

  /** Returns a vector of the output GraphEdges of a node. */
  static std::vector<GraphEdge> GetNodeOutputEdges(const Node& node);

  /** Returns a vector of output GraphEdges of a node for the provided output index. */
  static std::vector<GraphEdge> GetNodeOutputEdges(const Node& node, size_t index);

  /** Removes a set of GraphEdges from the graph. */
  static void RemoveGraphEdges(Graph& graph, const std::vector<GraphEdge>& edges);
};

#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

#if !defined(ORT_MINIMAL_BUILD)
/** Checks if the node has the same operator since version as the given one. */
bool MatchesOpSinceVersion(const Node& node, std::initializer_list<ONNX_NAMESPACE::OperatorSetVersion> versions);
bool MatchesOpSinceVersion(const Node& node, gsl::span<const ONNX_NAMESPACE::OperatorSetVersion> versions);

/** Checks if the node has the same op set domain as the given one. */
bool MatchesOpSetDomain(const Node& node, std::string_view domain);

/** Returns true if the execution provider assigned to current node is present in the compatible providers list
    or if the compatible_providers list is empty. */
bool IsSupportedProvider(const Node& node,
                         const InlinedHashSet<std::string_view>& compatible_providers);

/** Checks if the output at the specified index is input to downstream Nodes. */
bool IsOutputUsed(const Node& node, int index);

/** Returns true if the graph has the given input.*/
bool IsGraphInput(const Graph& graph, const NodeArg* input);

/** returns true if 'name' is an initializer in 'graph', or an ancestor graph if check_outer_scope is true.
@param check_outer_scope If true and 'graph' is a subgraph, check ancestor graph/s for 'name' if not found in 'graph'.
*/
bool IsInitializer(const Graph& graph, const std::string& name, bool check_outer_scope);

/** returns true if 'name' is an initializer, and is constant and cannot be overridden at runtime.
@param check_outer_scope If true and 'graph' is a subgraph, check ancestor graph/s for 'name' if not found in 'graph'.
*/
bool IsConstantInitializer(const Graph& graph, const std::string& name, bool check_outer_scope = true);

/** Checks if the given NodeArg is constant, i.e., it appears in the graph's initializers but not in its inputs. */
bool NodeArgIsConstant(const Graph& graph, const NodeArg& node_arg);

/** Checks if the given node has only constant inputs (initializers) and no input is in excluded_initializers.
If so returns them in constant_inputs as they may come from outer scope. */
bool AllNodeInputsAreConstant(const Graph& graph, const Node& node, InitializedTensorSet& constant_inputs,
                              const InlinedHashSet<std::string>& excluded_initializers = {});

/** Gets the index of an input arg with the specified input arg name. */
int GetNodeInputIndexFromInputName(const Node& node, const std::string& input_name);

// replacement for InlinedVector
namespace onnx_repeated_values {
template <typename T>
inline InlinedVector<T> RetrieveValues(const ONNX_NAMESPACE::AttributeProto& attr);

template <>
inline InlinedVector<int64_t> RetrieveValues(const ONNX_NAMESPACE::AttributeProto& attr) {
  return {attr.ints().begin(), attr.ints().end()};
}

template <>
inline InlinedVector<std::string> RetrieveValues(const ONNX_NAMESPACE::AttributeProto& attr) {
  return {attr.strings().begin(), attr.strings().end()};
}

template <>
inline InlinedVector<float> RetrieveValues(const ONNX_NAMESPACE::AttributeProto& attr) {
  return {attr.floats().begin(), attr.floats().end()};
}
}  // namespace onnx_repeated_values

/** Retrieves the values for a repeated attribute of a node and place them to the values vector. */
template <typename T>
bool GetRepeatedNodeAttributeValues(const Node& node,
                                    const std::string& attr_name,
                                    InlinedVector<T>& values) {
  const auto* attr = graph_utils::GetNodeAttribute(node, attr_name);
  if (attr) {
    values = onnx_repeated_values::RetrieveValues<T>(*attr);
    return true;
  }
  return false;
}

/** Find the first child of the specified op type. */
const Node* FirstChildByType(const Node& node, const std::string& child_type);

/** Find the first parent of the specified op type. */
const Node* FirstParentByType(const Node& node, const std::string& parent_type);

/** Tests if we can remove a node and merge its input edge (if any) with its output edges.
Conditions:
 Input rules:
 - the node has one input edge
   - it may have multiple other inputs coming from graph inputs or initializers
 - or the node has no input edges, and a single input
   - the input will be coming from a graph input or initializer

 Output rules:
 - Only one of the node's outputs is used by downstream operators
   - multiple edges for the single used output are allowed
 - The node does not produce a graph output
   - the node removal will result in that output name not being produced

 Subgraph rules:
 - Removing the node won't break a subgraph that consumes the node's output
*/
bool CanRemoveNode(const Graph& graph, const Node& node, const logging::Logger& logger);

/** Removes the given Node from the Graph.
See CanRemoveNode for the conditions that must be satisfied in order to remove the node.

If the node has one input edge, merge the input edge with any output edges.
If the node has no input edges it has a single input, so update any output edges to use the input as their source.

After output edges are updated, remove the node.
*/
bool RemoveNode(Graph& graph, Node& node);

/** Tests if we can remove a node and replace its output with an initializer.
Conditions:
 - Only one of the node's outputs is used by downstream operators or as a graph output
   - multiple edges for the single used output are allowed
 - If the node produces a graph output the initializer_name must be the same as the node's output name
   - otherwise the required graph output will not be produced
 - Removing the node won't break a subgraph that consumes the node's output
*/
bool CanReplaceNodeWithInitializer(const Graph& graph, const Node& node, const std::string& initializer_name,
                                   const logging::Logger& logger);

/** Remove a node and replace its output with the provided NodeArg for an initializer.
See CanReplaceNodeWithInitializer for the conditions that must be satisfied in order to remove the node.*/
bool ReplaceNodeWithInitializer(Graph& graph, Node& node, NodeArg& replacement);

/** Replaces the input to nodes that are downstream from 'node', which was being provided by an output of 'node',
    with an output from a different node. Moves the output edges from 'node' for 'output_idx' to the replacement node.
@param replacement The node providing the replacement output.
@param replacement_output_idx The index of the output from 'replacement' to use.

e.g. Node A produces outputs A1 and A2.
     Node B consumes A2 (edge between A and B for A2) and produces B1.
     Node C consumes B1 (edge between B and C for B1).

     If Node B was determined to not be needed, you would call ReplaceDownstreamNodeInput(graph, B, 0, A, 1)
     to replace B1 (output index 0 for node B) with A2 (output index 1 for node A) as input to the downstream node C.
     The edge that existed between B and C for B1 will be removed, and replaced with an edge between A and C for A2.
*/
void ReplaceDownstreamNodeInput(Graph& graph, Node& node, int output_idx, Node& replacement, int replacement_output_idx);

/** Replace the input to a node with a NodeArg.
@remarks The replacement only updates the node's input definition and does not create any edges,
         as typically this function is used to replace an input with an initializer or graph input
         (there is no edge between an initializer or graph input and a Node).
*/
void ReplaceNodeInput(Node& target, int target_input_idx, NodeArg& new_input);

/** Add an input to a node with a NodeArg for an initializer or graph input.
@remarks target_input_idx must be the next input slot.
           e.g. if a Node has 2 inputs, AddNodeInput can only add input 3 and not 4.
         There is no edge between an initializer or graph input and a Node, so the replacement only updates the
         node's input definition and does not create any new edges.
*/
void AddNodeInput(Node& target, int target_input_idx, NodeArg& new_input);

/** Finalize the fusion of second_node into first_node.
    The output definitions and edges from the second_node are moved to first_node. second_node is deleted.
    e.g. Conv + Add fusion fuses the 'Add' into the Conv.
*/
void FinalizeNodeFusion(Graph& graph, Node& first_node, Node& second_node);

/** Finalize the fusion of two or more nodes which are being replaced with two or more nodes.
    The first and last entries in 'nodes' are assumed to be the first and last nodes in a chain of nodes being fused.

    Conceptually multiple nodes are being combined, and post-fusion will produce output/s with the same names
    as the last node in 'nodes', and be connected to the same downstream nodes.

    The input edges to the first node in 'nodes' will be moved to replacement_node_start. No other input edges are moved.
    The output definitions and edges from the last node in 'nodes' will be moved to replacement_node_end.
    All nodes in 'nodes' will be removed.
*/
void FinalizeNodeFusion(Graph& graph,
                        gsl::span<const std::reference_wrapper<Node>> nodes,
                        Node& replacement_node_start,
                        Node& replacement_node_end);

inline void FinalizeNodeFusion(Graph& graph,
                               std::initializer_list<std::reference_wrapper<Node>> nodes,
                               Node& replacement_node_start,
                               Node& replacement_node_end) {
  FinalizeNodeFusion(graph, gsl::make_span(nodes), replacement_node_start, replacement_node_end);
}

/** Finalize the fusion of two or more nodes which are being replaced with a single node.
    The first and last entries in 'nodes' are assumed to be the first and last nodes in a chain of nodes being fused.

    Conceptually multiple nodes are being combined into one, and post-fusion will produce output/s with the same names
    as the last node in 'nodes', and be connected to the same downstream nodes.

    The input edges to the first node in 'nodes' will be moved to replacement_node. No other input edges are moved.
    The output definitions and edges from the last node in 'nodes' will be moved to replacement_node.
    All nodes in 'nodes' will be removed.
*/
inline void FinalizeNodeFusion(Graph& graph, gsl::span<const std::reference_wrapper<Node>> nodes, Node& replacement_node) {
  FinalizeNodeFusion(graph, nodes, replacement_node, replacement_node);
}

inline void FinalizeNodeFusion(Graph& graph, std::initializer_list<std::reference_wrapper<Node>> nodes, Node& replacement_node) {
  FinalizeNodeFusion(graph, gsl::make_span(nodes.begin(), nodes.end()), replacement_node, replacement_node);
}

/** Find the source node of an input edge for a specified input index.
@returns nullptr when not found.
*/
const Node* GetInputNode(const Node& node, int arg_index);

/** Expected edge end information for matching input or output edge.
    For input edge, the node in the edge end refers to the source node, otherwise the destination node.
*/
struct EdgeEndToMatch {
  // Source arg index of edge.
  int src_arg_index;

  // Destination arg index of edge.
  int dst_arg_index;

  // Expected operator type of the node in the edge end.
  std::string op_type;

  // Expected version of the operator of node in the edge end.
  InlinedVector<ONNX_NAMESPACE::OperatorSetVersion> versions;

  // Expected domain of the operator of node in the edge end.
  std::string domain;
};

/** Find a path that matches the specified edges.
    A path is a sequence of adjacent edges, and the result is returned as a list of EdgeEnd items.
@param node is the current node to start matching.
@param is_input_edge is a flag to indicate  whether the edges are input or output edges.
@param edges_to_match has information of a sequence of adjacent edges in the path to be matched one by one.
@param result stores edges that are found.
@returns false when one edge has multiple candidates, or not all edges are found.
@remarks matching an EdgeEndToMatch might get multiple candidates in output edges.
    When such case is encountered, this function will return false. This is by design to reduce complexity.
    Here is an example graph:
                  Add
                 /    \
               Mul     Mul
                 \    /
                  Sub
    For example, you want to match path from top to bottom: Add-->Mul-->Sub.
    When matching the first edge Add-->Mul, the algorithm found two matches.
    Then it returns false, and output a warning log entry.

    It is recommended to match path from bottom to top direction to avoid such issue.
    It is because each node input (dst_arg_index) only accepts one input edge.
*/
bool FindPath(const Node& node, bool is_input_edge, gsl::span<const EdgeEndToMatch> edges_to_match, std::vector<const Node::EdgeEnd*>& result, const logging::Logger& logger);

inline bool FindPath(const Node& node, bool is_input_edge, std::initializer_list<EdgeEndToMatch> edges_to_match, std::vector<const Node::EdgeEnd*>& result, const logging::Logger& logger) {
  return FindPath(node, is_input_edge, gsl::make_span(edges_to_match), result, logger);
}

/** Same as FindPath above, but return the references of matched Node
 */
bool FindPath(Graph& graph, const Node& node, bool is_input_edge, gsl::span<const EdgeEndToMatch> edges_to_match, std::vector<std::reference_wrapper<Node>>& result, const logging::Logger& logger);

inline bool FindPath(Graph& graph, const Node& node, bool is_input_edge, std::initializer_list<EdgeEndToMatch> edges_to_match, std::vector<std::reference_wrapper<Node>>& result, const logging::Logger& logger) {
  return FindPath(graph, node, is_input_edge, gsl::make_span(edges_to_match), result, logger);
}

/**
 * Remove nodes with only one output edge using bottom-up bfs traversal.
 * @param node: The node to start with.
 * @returns true if there is one or more node(s) removed by this function. Otherwise return false.
 */
bool RemoveNodesWithOneOutputBottomUp(Graph& graph, const Node& node);

/** Creates a mutable NodeArg owned by the graph with mirrored base_arg's TypeProto and name
 * @param base_arg The NodeArg the newly created NodeArg is mirrored based off.
 * @returns NodeArg reference that contains the same TypeProto info as base_arg with generated different names.
 */
NodeArg& CreateNodeArg(Graph& graph, const NodeArg& base_arg);

#endif  // !defined(ORT_MINIMAL_BUILD)

}  // namespace graph_utils
}  // namespace onnxruntime
