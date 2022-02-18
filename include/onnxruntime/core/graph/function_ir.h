// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

#ifdef _WIN32
#pragma warning(push)
// disable some warnings from protobuf to pass Windows build
#pragma warning(disable : 4244)
#endif

#if !defined(ORT_MINIMAL_BUILD)
#include "onnx/defs/schema.h"
#else
#include "onnx/defs/data_type_utils.h"
#endif
#include "onnx/onnx_pb.h"
#include "onnx/onnx-operators_pb.h"

#ifdef _WIN32
#pragma warning(pop)
#endif

#include "gsl/gsl"

#include "core/common/common.h"
#include "core/common/const_pointer_container.h"
#include "core/common/path.h"
#include "core/common/status.h"
#include "core/common/logging/logging.h"
#include "core/graph/basic_types.h"
#include "core/graph/constants.h"
#include "core/graph/node_arg.h"
#include "core/graph/node.h"
#include "core/graph/graph_nodes.h"

namespace onnxruntime {
struct IndexedSubGraph;
    /**
@class FunctionIR
A pure IR which represent the computation function.
It doesn't contains any materialization like GraphProto / Intializers / ....
*/
class FunctionIR {
 public:
  /** Gets the Graph inputs excluding initializers.
  These are the required inputs to the Graph as the initializers can be optionally overridden via graph inputs.
  @remarks Contains no nullptr values. */
  const std::vector<const NodeArg*>& GetInputs() const noexcept { return graph_inputs_; }

  /** Gets the Graph outputs.
  @remarks Contains no nullptr values.*/
  const std::vector<const NodeArg*>& GetOutputs() const noexcept { return graph_outputs_; }

  bool IsOutput(const NodeArg* node_arg) const noexcept {
    return std::find(graph_outputs_.begin(), graph_outputs_.end(), node_arg) != graph_outputs_.end();
  }

  /** Returns true if one or more of the Node outputs are Graph outputs.
  @remarks Cheaper than calling GetNodeOutputsInGraphOutputs.
  */
  bool NodeProducesGraphOutput(const Node& node) const {
    auto end_outputs = graph_outputs_.cend();
    for (auto output_def : node.OutputDefs()) {
      if (std::find(graph_outputs_.cbegin(), end_outputs, output_def) != end_outputs) {
        return true;
      }
    }
    return false;
  }

  /** Returns a vector with the indexes of the outputs of the given Node that are also Graph outputs. */
  std::vector<int> GetNodeOutputsInGraphOutputs(const Node& node) const {
    int output_idx = 0;
    std::vector<int> indexes;
    for (auto output_def : node.OutputDefs()) {
      if (std::find(GetOutputs().cbegin(), GetOutputs().cend(), output_def) != GetOutputs().cend()) {
        indexes.push_back(output_idx);
      }

      ++output_idx;
    }

    return indexes;
  }

  /** Gets the NodeArgs that represent value_info instances in the Graph.
  These are the values that are neither Graph inputs nor outputs.
  @remarks Contains no nullptr values. */
  const std::unordered_set<const NodeArg*>& GetValueInfo() const noexcept { return value_info_; }

#if !defined(ORT_MINIMAL_BUILD)
  void AddValueInfo(const NodeArg* new_value_info);
#endif

  /** Gets the Node with the specified node index.
  @returns Node instance if found. nullptr if node_index is invalid or node has been freed.
  */
  const Node* GetNode(NodeIndex node_index) const { return NodeAtIndexImpl(node_index); }

  /** Gets the mutable Node with the specified node index.
  @returns Mutable Node instance if found. nullptr if node_index is invalid or node has been freed.
  */
  Node* GetNode(NodeIndex node_index) { return NodeAtIndexImpl(node_index); }

  /** Get a GraphNodes instance that provides mutable access to all valid Nodes in the Graph. */
  GraphNodes& Nodes() noexcept { return iterable_nodes_; }

  /** Get a GraphNodes instance that provides const access to all valid Nodes in the Graph. */
  const GraphNodes& Nodes() const noexcept { return iterable_nodes_; }

  /** Get a ConstGraphNodes instance that provides access to a filtered set of valid Nodes in the Graph.
  @remarks We can't use GraphNodes as that would provide mutable access to the nodes by default, and we can't prevent
           that by returning a const instance of GraphNodes as we're creating a new instance here due to the filter
           being something we don't control (i.e. we have to return a new instance so it can't be const).
  */
  ConstGraphNodes FilteredNodes(GraphNodes::NodeFilterFunc&& filter_func) const noexcept {
    return ConstGraphNodes(nodes_, std::move(filter_func));
  }

  /** Gets the maximum NodeIndex value used in the Graph.
  WARNING: This actually returns the max index value used + 1.
  */
  int MaxNodeIndex() const noexcept { return static_cast<int>(nodes_.size()); }  //assume the casting won't overflow

  /** Gets the number of valid Nodes in the Graph.
  @remarks This may be smaller than MaxNodeIndex(), as Nodes may be removed during optimization.
  */
  int NumberOfNodes() const noexcept { return num_of_nodes_; }

  /** Gets the mutable NodeArg with the provided name.
  @returns Pointer to NodeArg if found, nullptr if not. */
  NodeArg* GetNodeArg(const std::string& name) {
    auto iter = node_args_.find(name);
    if (iter != node_args_.end()) {
      return iter->second.get();
    }
    return nullptr;
  }

  const std::unordered_set<std::string>& GetOuterScopeNodeArgNames() const {
    return outer_scope_node_arg_names_;
  }

  bool IsProducedInCurrentGraph(const std::string& arg) const {
    return node_arg_to_producer_node_.find(arg) != node_arg_to_producer_node_.cend();
  }

  /** Gets the const NodeArg with the provided name.
  @returns Pointer to const NodeArg if found, nullptr if not. */
  const NodeArg* GetNodeArg(const std::string& name) const {
    return const_cast<FunctionIR*>(this)->GetNodeArg(name);
  }

  /** Gets a mutable NodeArg by name. Creates a new NodeArg that is owned by this Graph if not found.
  @param name The NodeArg name.
  @param[in] p_arg_type Optional TypeProto to use if the NodeArg needs to be created.
  @returns NodeArg reference.
  */
  NodeArg& GetOrCreateNodeArg(const std::string& name, const ONNX_NAMESPACE::TypeProto* p_arg_type) {
    auto iter = node_args_.find(name);
    if (iter != node_args_.end()) {
      return *(iter->second);
    }
    auto result = node_args_.insert(std::make_pair(name, std::make_unique<NodeArg>(name, p_arg_type)));
    return *(result.first->second);
  }

#if !defined(ORT_MINIMAL_BUILD)
  /** Copy a Node and add it to this Graph.
  @param other Node to copy
  @returns Reference to the Node that was created and added to this Graph.
  @remarks Do not call AddNode and Remove Node concurrently as they are not thread-safe.
  */
  Node& AddNode(const Node& other);
#endif

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
  /** Add a Node to this Graph.
  @param name The Node name. Must be unique in this Graph.
  @param op_type The operator type. e.g. ONNX operator name.
  @param description Arbitrary description of the Node.
  @param input_args The explicit inputs to this Node.
  @param output_args The outputs from this Node.
  @param attributes Optional NodeAttributes to add.
  @param domain The domain for the op_type.
  @returns Reference to the new Node.
  @remarks Do not call AddNode and Remove Node concurrently as they are not thread-safe.
  */
  Node& AddNode(const std::string& name,
                const std::string& op_type,
                const std::string& description,
                const std::vector<NodeArg*>& input_args,
                const std::vector<NodeArg*>& output_args,
                const NodeAttributes* attributes = nullptr,
                const std::string& domain = "");

  /** Remove a Node from this Graph and free it.
  The output edges of this specified node MUST have been removed before removing the node.
  The input edges of this specified node is removed while removing the node. The process of
  removing a node from a graph should be,
  1. Remove out edges of this specified node.
  2. Remove this specified node.
  3. Add new input edges connected with all out nodes.
  @returns true if the node_index was valid
  @remarks Do not call AddNode and Remove Node concurrently as they are not thread-safe.
  */
  bool RemoveNode(NodeIndex node_index);

  /** Add an edge between two Nodes.
  @param src_node_index NodeIndex of source Node that is providing output to the destination Node.
  @param dst_node_index NodeIndex of destination Node that is receiving input from the source Node.
  @param src_arg_index node arg index of source node.
  @param dst_arg_index node arg index of destination node.
  */
  void AddEdge(NodeIndex src_node_index, NodeIndex dst_node_index, int src_arg_index, int dst_arg_index);

  /** Remove an edge between two Nodes.
  @param src_node_index NodeIndex of source Node to remove an output edge from.
  @param dst_node_index NodeIndex of destination Node to remove an input edge from.
  @param src_arg_index node arg index of source node.
  @param dst_arg_index node arg index of destination node.
  */
  void RemoveEdge(NodeIndex src_node_index, NodeIndex dst_node_index, int src_arg_index, int dst_arg_index);
#endif

#if !defined(ORT_MINIMAL_BUILD)
  /**
  Add a control edge between two Nodes in this Graph.
  The source Node does not produce output that is directly consumed by the destination Node, however the
  destination Node must execute after the source node. The control edge allows this ordering to occur.
  */
  bool AddControlEdge(NodeIndex src_node_index, NodeIndex dst_node_index);
#endif  // !defined(ORT_MINIMAL_BUILD)

  
  /** Performs a reverse depth-first search (DFS) traversal from a set of nodes, via their inputs,
  up to their source node/s.
  @param from NodeIndex values for a set of Nodes to traverse from.
  @param enter Visit function that will be invoked on a node when it is visited but its parents haven't been.
  @param leave Visit function invoked on the node after its parents have all been visited.
  @param comp Comparison function to stabilize the traversal order by making Node ordering deterministic.
  */
  void ReverseDFSFrom(const std::vector<NodeIndex>& from,
                      const std::function<void(const Node*)>& enter,
                      const std::function<void(const Node*)>& leave,
                      const std::function<bool(const Node*, const Node*)>& comp = {}) const;

  /** Performs a reverse depth-first search (DFS) traversal from a set of nodes, via their inputs,
  up to their source node/s.
  @param from Set of Nodes to traverse from.
  @param enter Visit function that will be invoked on a node when it is visited but its parents haven't been.
  @param leave Visit function invoked on the node after its parents have all been visited.
  @param comp Comparison function to stabilize the traversal order by making Node ordering deterministic.
  */
  void ReverseDFSFrom(const std::vector<const Node*>& from,
                      const std::function<void(const Node*)>& enter,
                      const std::function<void(const Node*)>& leave,
                      const std::function<bool(const Node*, const Node*)>& comp = {}) const;

  /** Performs a reverse depth-first search (DFS) traversal from a set of nodes, via their inputs,
  up to their source node/s.
  @param from Set of Nodes to traverse from.
  @param enter Visit function that will be invoked on a node when it is visited but its parents haven't been.
  @param leave Visit function invoked on the node after its parents have all been visited.
  @param stop Stop traversal from node n to input node p if stop(n, p) is true.
  @param comp Comparison function to stabilize the traversal order by making Node ordering deterministic.
  */
  void ReverseDFSFrom(const std::vector<const Node*>& from,
                      const std::function<void(const Node*)>& enter,
                      const std::function<void(const Node*)>& leave,
                      const std::function<bool(const Node*, const Node*)>& comp,
                      const std::function<bool(const Node*, const Node*)>& stop) const;

#if !defined(ORT_MINIMAL_BUILD)
  /** Performs topological sort with Kahn's algorithm on the graph/s.
  @param enter Visit function that will be invoked on a node when it is visited.
  @param comp Comparison function to stabilize the traversal order by making Node ordering deterministic.
  */
  void KahnsTopologicalSort(const std::function<void(const Node*)>& enter,
                            const std::function<bool(const Node*, const Node*)>& comp) const;

#endif

//TODO!!!!!
//Re-implement this part
//#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
//  /**
//  Create a single Node that will be the result of the a fusion of multiple nodes in this Graph.
//  @param sub_graph A IndexSubGraph instance with details of the nodes to fuse.
//  @param fused_node_name The name for the new Node.
//  @returns Node with fused subgraph.
//  @remarks As a new Graph instance for the fused nodes is not created, a GraphViewer can be constructed with the
//           IndexedSubGraph information to provide a view of the subgraph. The original nodes are left in place
//           while this is in use.
//           Call FinalizeFuseSubGraph to remove them once the fused replacement node is fully created.
//  */
//  Node& BeginFuseSubGraph(const IndexedSubGraph& sub_graph, const std::string& fused_node_name);
//
//  /**
//  If we have BeginFuseSubGraph, but somehow hit errors, such as Compile of an EP failed on thesub_graph.
//  We can call CancelFuseSubGraph to undo the changes of BeginFuseSubGraph
//  @param fused_node The fused node and it's function body to be removed from the graph
//  */
//  void CancelFuseSubGraph(const Node& fused_node);
//
//  void FinalizeFuseSubGraph(const IndexedSubGraph& sub_graph, Node& fused_node);
//#endif

#if !defined(ORT_MINIMAL_BUILD)
  //TODO!!!!
  //Re-impelemnt this part
  /**
  Create a single Function based Node that is the result of the a fusion of multiple nodes in this Graph.
  A new Graph instance will be created for the fused nodes.
  @param sub_graph A IndexSubGraph instance with details of the nodes to fuse. Ownership is transferred to the new Node
  @param fused_node_name The name for the new Node.
  @returns Function based Node with fused subgraph. The Node body will contain a Function instance.
  */
  /*Node& FuseSubGraph(const IndexedSubGraph& sub_graph, const std::string& fused_node_name);*/

  /**
  Directly insert the nodes in the function Node provided into this Graph.
  @param node Node with Node::Type of Node::Type::Fused
  @returns Status indicating success or providing an error message.
  */
  /*Status InlineFunction(Node& node);*/

  /** Initialize function body for the given node */
  /*void InitFunctionBodyForNode(Node& node);*/

  /** Gets Model local functions from the root/parent graph.*/
  /*const std::unordered_map<std::string, const ONNX_NAMESPACE::FunctionProto*>& GetModelLocalFunctions() const;*/

  /** Mark a NodeArg name as coming from the outer scope when programmatically constructing a Graph that will
  be used as a GraphProto attribute in another Node..
  e.g. when creating a Graph instance that will be used as a subgraph in a control flow operator, it is necessary to
  define placeholder NodeArgs for outer scope values. This prevents these values from becoming explicit graph inputs
  when the Graph is resolved.
  */
  void AddOuterScopeNodeArg(const std::string& name) {
    ORT_IGNORE_RETURN_VALUE(outer_scope_node_arg_names_.insert(name));
  }

  /** Explicitly set graph inputs.
  @param inputs NodeArgs that represent complete graph inputs which need to be explicitly ordered.
  @remarks Note that the input order matters for subgraphs.
  */
  void SetInputs(const std::vector<const NodeArg*>& inputs);

  /** Explicitly set graph outputs.
  @param outputs NodeArgs that represent complete graph outputs which need to be explicitly ordered.
  @remarks Note that the output order matters for subgraphs.
  */
  void SetOutputs(const std::vector<const NodeArg*>& outputs);

  void SetValueInfo(const std::unordered_set<const NodeArg*>& value_infos);

  // explicitly remove a value from outputs
  void RemoveFromOutputs(const NodeArg* output) {
    graph_outputs_.erase(std::remove(graph_outputs_.begin(), graph_outputs_.end(), output),
                         graph_outputs_.end());
  }

#endif  // !defined(ORT_MINIMAL_BUILD)

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_ENABLE_RUNTIME_OPTIMIZATION_IN_MINIMAL_BUILD)
  /** Sets the type of a NodeArg, replacing existing type/shape if any */
  void SetNodeArgType(NodeArg& arg, const ONNX_NAMESPACE::TypeProto& type_proto);

  const Node* GetProducerNode(const std::string& node_arg_name) const {
    return GetProducerNodeImpl(*this, node_arg_name);
  }

  Node* GetMutableProducerNode(const std::string& node_arg_name) {
    return GetProducerNodeImpl(*this, node_arg_name);
  }

  void UpdateProducerNode(const std::string& node_arg_name, NodeIndex node_index) {
    auto iter = node_arg_to_producer_node_.find(node_arg_name);

    if (iter != node_arg_to_producer_node_.end()) {
      iter->second = node_index;
    } else {
      node_arg_to_producer_node_[node_arg_name] = node_index;
    }
  }

  std::vector<const Node*> GetConsumerNodes(const std::string& node_arg_name) const {
    return GetConsumerNodesImpl(*this, node_arg_name);
  }

  // Without removing the existing consumers, add a consumer to the give node arg name.
  void AddConsumerNode(const std::string& node_arg_name, Node* consumer) {
    node_arg_to_consumer_nodes_[node_arg_name].insert(consumer->Index());
  }

  // Remove a consumer from the set
  void RemoveConsumerNode(const std::string& node_arg_name, Node* consumer) {
    node_arg_to_consumer_nodes_[node_arg_name].erase(consumer->Index());
  }
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_ENABLE_RUNTIME_OPTIMIZATION_IN_MINIMAL_BUILD)

#if !defined(ORT_MINIMAL_BUILD)
  std::vector<Node*> GetMutableConsumerNodes(const std::string& node_arg_name) {
    return GetConsumerNodesImpl(*this, node_arg_name);
  }

  void UpdateConsumerNodes(const std::string& node_arg_name, const std::vector<Node*>& nodes) {
    auto iter = node_arg_to_consumer_nodes_.find(node_arg_name);
    if (iter != node_arg_to_consumer_nodes_.end()) {
      node_arg_to_consumer_nodes_.erase(node_arg_name);
    }
    for (Node* node : nodes) {
      node_arg_to_consumer_nodes_[node_arg_name].insert(node->Index());
    }
  }
#endif  // !defined(ORT_MINIMAL_BUILD)

  virtual ~FunctionIR();
  
  FunctionIR(Graph* graph) : graph_(graph) {}

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(FunctionIR);

  friend class Graph;
  friend class GraphContext;

  
  // Add node with specified <node_proto>.
  Node& AddNode(const ONNX_NAMESPACE::NodeProto& node_proto,
                const ArgNameToTypeMap& name_to_type);

#if !defined(ORT_MINIMAL_BUILD)

  std::vector<NodeArg*> CreateNodeArgs(const google::protobuf::RepeatedPtrField<std::string>& names,
                                       const ArgNameToTypeMap& name_to_type_map);

#endif  // !defined(ORT_MINIMAL_BUILD)

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_ENABLE_RUNTIME_OPTIMIZATION_IN_MINIMAL_BUILD)
  Status PopulateNodeArgToProducerConsumerLookupsFromNodes();

  template <typename TInstance>
  static auto GetConsumerNodesImpl(
      TInstance& instance, const std::string& node_arg_name) -> std::vector<decltype(instance.GetNode(0))> {
    std::vector<decltype(instance.GetNode(0))> results;
    auto iter = instance.node_arg_to_consumer_nodes_.find(node_arg_name);
    if (iter != instance.node_arg_to_consumer_nodes_.end()) {
      results.reserve(iter->second.size());
      for (auto node_index : iter->second) {
        results.push_back(instance.GetNode(node_index));
      }
    }
    return results;
  }

  template <typename TInstance>
  static auto GetProducerNodeImpl(
      TInstance& instance, const std::string& node_arg_name) -> decltype(instance.GetNode(0)) {
    auto iter = instance.node_arg_to_producer_node_.find(node_arg_name);
    if (iter != instance.node_arg_to_producer_node_.end()) {
      auto node_index = iter->second;
      return instance.GetNode(node_index);
    }
    return nullptr;
  }
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_ENABLE_RUNTIME_OPTIMIZATION_IN_MINIMAL_BUILD)

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

  Status MoveToTarget(const IndexedSubGraph& sub_graph, FunctionIR* target);

  //TODO!!!!
  //Do we need this?
  gsl::not_null<Node*> AllocateNode();

  // Release the node.
  // @returns false if node_index was invalid.
  bool ReleaseNode(NodeIndex node_index);

  //TODO!!!
  //reimplement this
  //Node& CreateFusedSubGraphNode(const IndexedSubGraph& sub_graph, const std::string& fused_node_name);
#endif

  Node* NodeAtIndexImpl(NodeIndex node_index) const {
    // if we are trying to access a node that doesn't exist there's (most
    // likely) either a logic issue or a graph consistency/correctness issue.
    // use ORT_ENFORCE to prove that or uncover scenarios where we actually
    // expect attempts to retrieve a non-existent node.
    ORT_ENFORCE(node_index < nodes_.size(), "Validating no unexpected access using an invalid node_index. Got:",
                node_index, " Max:", nodes_.size());
    return nodes_[node_index].get();
  }

  // The graph this function belong to
  Graph* graph_;

  // Graph nodes.
  // Element in <nodes_> may be nullptr due to graph optimization.
  std::vector<std::unique_ptr<Node>> nodes_;

  // Wrapper of Graph nodes to provide iteration services that hide nullptr entries
  GraphNodes iterable_nodes_{nodes_};

  // Number of nodes.
  // Normally this is smaller than the size of <m_nodes>, as some
  // elements in <m_nodes> may be removed when doing graph optimization,
  // or some elements may be merged, etc.
  int num_of_nodes_ = 0;

  std::vector<const NodeArg*> graph_inputs_;

  // Graph outputs.
  std::vector<const NodeArg*> graph_outputs_;
  
  // Graph value_info.
  std::unordered_set<const NodeArg*> value_info_;

  // All node args owned by <*this> graph. Key is node arg name.
  std::unordered_map<std::string, std::unique_ptr<NodeArg>> node_args_;

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_ENABLE_RUNTIME_OPTIMIZATION_IN_MINIMAL_BUILD)
  // node arg to its producer node
  std::unordered_map<std::string, NodeIndex> node_arg_to_producer_node_;

  // node arg to its consumer nodes
  std::unordered_map<std::string, std::unordered_set<NodeIndex>> node_arg_to_consumer_nodes_;
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_ENABLE_RUNTIME_OPTIMIZATION_IN_MINIMAL_BUILD)
  // TODO!!!
  // Do we need this???
  // NodeArgs that come from outer scope. Used when building a graph so that
  // these don't get recorded as graph inputs in the GraphProto.
  std::unordered_set<std::string> outer_scope_node_arg_names_;
};

#if !defined(ORT_MINIMAL_BUILD)
// Print NodeArg as
//  name : type
// For example,
//  "110": tensor(float)
std::ostream& operator<<(std::ostream& out, const NodeArg& node_arg);
// Print Node as,
//  (operator's name, operator's type, domain, version) : (input0, input1, ...) -> (output0, output1, ...)
// For example,
//  ("Add_14", Add, "", 7) : ("110": tensor(float),"109": tensor(float),) -> ("111": tensor(float),)
std::ostream& operator<<(std::ostream& out, const Node& node);
#endif

}  // namespace onnxruntime
