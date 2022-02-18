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
#include "core/graph/graph_nodes.h"
#include "core/graph/node_arg.h"
#include "core/graph/function_ir.h"
#include "core/graph/graph_context.h"

namespace flatbuffers {
class FlatBufferBuilder;
template <typename T>
struct Offset;
}  // namespace flatbuffers

namespace onnxruntime {
class Graph;
struct IndexedSubGraph;
class Model;
class OpSignature;

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_ENABLE_RUNTIME_OPTIMIZATION_IN_MINIMAL_BUILD)
class RuntimeOptimizationRecordContainer;
#endif

namespace fbs {
struct Graph;
struct Node;
struct NodeEdge;
}  // namespace fbs

/**
@class Graph
The Graph representation containing the graph inputs and outputs, the Node instances,
and the edges connecting the nodes.
*/
class Graph {
 public:
  /** Gets the Graph name. */
  const std::string& Name() const noexcept;

  /** Gets the Graph description. */
  const std::string& Description() const noexcept;

  /** Gets the path of the owning model, if any. */
  const Path& ModelPath() const;

  /** Returns true if this is a subgraph or false if it is a high-level graph. */
  bool IsSubgraph() const { return parent_graph_ != nullptr; }

  /** Returns the parent graph if this is a subgraph */
  const Graph* ParentGraph() const { return parent_graph_; }

  /** Returns the mutable parent graph if this is a subgraph */
  Graph* MutableParentGraph() { return parent_graph_; }

#if !defined(ORT_MINIMAL_BUILD)
  /** Sets the Graph name. */
  void SetName(const std::string& name);

  /** Gets the Graph description. */
  void SetDescription(const std::string& description);

  /** Replaces the initializer tensor with the same name as the given initializer tensor.
  The replacement initializer tensor must have the same type and shape as the existing initializer tensor.

  Note: This currently has linear time complexity. There is room for improvement but it would likely require changes to
  how initializer tensors are stored and tracked.
  */
  common::Status ReplaceInitializedTensor(const ONNX_NAMESPACE::TensorProto& new_initializer);
#endif  // !defined(ORT_MINIMAL_BUILD)

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
  /** Add an initializer tensor to the Graph. */
  void AddInitializedTensor(const ONNX_NAMESPACE::TensorProto& tensor_proto);
#endif

  /** Remove the initializer tensor with the provided name from the Graph. */
  void RemoveInitializedTensor(const std::string& tensor_name);

  /** Check if a given name is an initializer tensor's name in this graph. */
  bool IsInitializedTensor(const std::string& name) const;

#if !defined(DISABLE_SPARSE_TENSORS)
  /** Check if a given name is a sparse initializer's name in the model
   * we currently convert sparse_initializer field in the model into dense Tensor instances.
   * However, we sometimes want to check if this initializer was stored as sparse in the model.
   */
  bool IsSparseInitializer(const std::string& name) const;
#endif

  /** Gets an initializer tensor with the provided name.
  @param[out] value Set to the TensorProto* if the initializer is found, or nullptr if not.
  @returns True if found.
  */
  bool GetInitializedTensor(const std::string& tensor_name, const ONNX_NAMESPACE::TensorProto*& value) const;

  /** Gets all the initializer tensors in this Graph. */
  const InitializedTensorSet& GetAllInitializedTensors() const noexcept;

  /** Removes all initializer tensors from this Graph and releases the memory they were using. */
  void CleanAllInitializedTensors() noexcept;

  /** Returns true if an initializer value can be overridden by a graph input with the same name. */
  bool CanOverrideInitializer() const noexcept { return ir_version_ >= 4; }

  /** returns the initializer's TensorProto if 'name' is an initializer, is constant and
  cannot be overridden at runtime. If the initializer is not found or is not constant, a nullptr is returned.
  @param check_outer_scope If true and the graph is a subgraph,
         check ancestor graph/s for 'name' if not found in 'graph'.
  @remarks check_outer_scope of true is not supported in a minimal build
  */
  const ONNX_NAMESPACE::TensorProto* GetConstantInitializer(const std::string& name, bool check_outer_scope) const;

  /** returns the initializer's TensorProto if 'name' is an initializer (both constant and overridable).
  If the initializer is not found, a nullptr is returned.
  @param check_outer_scope If true and the graph is a subgraph,
         check ancestor graph/s for 'name' if not found in 'graph'.
  @remarks check_outer_scope of true is not supported in a minimal build
  */
  const ONNX_NAMESPACE::TensorProto* GetInitializer(const std::string& name, bool check_outer_scope) const;

  /** Gets the Graph inputs excluding initializers.
  These are the required inputs to the Graph as the initializers can be optionally overridden via graph inputs.
  @remarks Contains no nullptr values. */
  const std::vector<const NodeArg*>& GetInputs() const noexcept { return graph_inputs_excluding_initializers_; }

  /** Gets the Graph inputs including initializers.
  This is the full set of inputs, in the same order as defined in the GraphProto.
  @remarks Contains no nullptr values. */
  const std::vector<const NodeArg*>& GetInputsIncludingInitializers() const noexcept {
    return graph_context_.GetMainFunction().GetInputs();
  }

  /** Return true if "node_arg" is a input or an initializer. Otherwise, returns false. */
  bool IsInputsIncludingInitializers(const NodeArg* node_arg) const noexcept {
    auto& inputs_include_initializers = graph_context_.GetMainFunction().GetInputs();
    return std::find(inputs_include_initializers.begin(),
                     inputs_include_initializers.end(), node_arg) != inputs_include_initializers.end();
  }

  /** Gets the Graph inputs that are initializers
  These are overridable initializers. This is a difference between
  graph_inputs_including_initializers_ and graph_inputs_excluding_initializers_
  @remarks Contains no nullptr values. */
  const std::vector<const NodeArg*>& GetOverridableInitializers() const {
    return graph_overridable_initializers_;
  }

  /** Gets the Graph outputs.
  @remarks Contains no nullptr values.*/
  const std::vector<const NodeArg*>& GetOutputs() const noexcept { return graph_context_.GetMainFunction().GetOutputs(); }

  bool IsOutput(const NodeArg* node_arg) const noexcept {
    return graph_context_.GetMainFunction().IsOutput(node_arg);
  }

  /** Returns true if one or more of the Node outputs are Graph outputs.
  @remarks Cheaper than calling GetNodeOutputsInGraphOutputs.
  */
  bool NodeProducesGraphOutput(const Node& node) const {
    return graph_context_.GetMainFunction().NodeProducesGraphOutput(node);
  }

  /** Returns a vector with the indexes of the outputs of the given Node that are also Graph outputs. */
  std::vector<int> GetNodeOutputsInGraphOutputs(const Node& node) const {
    return graph_context_.GetMainFunction().GetNodeOutputsInGraphOutputs(node);
  }

  /** Gets the NodeArgs that represent value_info instances in the Graph.
  These are the values that are neither Graph inputs nor outputs.
  @remarks Contains no nullptr values. */
  const std::unordered_set<const NodeArg*>& GetValueInfo() const noexcept { return graph_context_.GetMainFunction().GetValueInfo(); }

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
  GraphNodes& Nodes() noexcept { return graph_context_.GetMutableMainFunction()->Nodes(); }

  /** Get a GraphNodes instance that provides const access to all valid Nodes in the Graph. */
  const GraphNodes& Nodes() const noexcept { return graph_context_.GetMainFunction().Nodes(); }

  /** Get a ConstGraphNodes instance that provides access to a filtered set of valid Nodes in the Graph.
  @remarks We can't use GraphNodes as that would provide mutable access to the nodes by default, and we can't prevent
           that by returning a const instance of GraphNodes as we're creating a new instance here due to the filter
           being something we don't control (i.e. we have to return a new instance so it can't be const).
  */
  ConstGraphNodes FilteredNodes(GraphNodes::NodeFilterFunc&& filter_func) const noexcept {
    return graph_context_.GetMainFunction().FilteredNodes(std::move(filter_func));
  }

  /** Gets the maximum NodeIndex value used in the Graph.
  WARNING: This actually returns the max index value used + 1.
  */
  int MaxNodeIndex() const noexcept { return graph_context_.GetMainFunction().MaxNodeIndex(); }  //assume the casting won't overflow

  /** Gets the number of valid Nodes in the Graph.
  @remarks This may be smaller than MaxNodeIndex(), as Nodes may be removed during optimization.
  */
  int NumberOfNodes() const noexcept { return graph_context_.GetMainFunction().NumberOfNodes(); }

  /** Gets the mutable NodeArg with the provided name.
  @returns Pointer to NodeArg if found, nullptr if not. */
  NodeArg* GetNodeArg(const std::string& name) {
    return graph_context_.GetMutableMainFunction()->GetNodeArg(name);
  }

  /** Gets the const NodeArg with the provided name.
  @returns Pointer to const NodeArg if found, nullptr if not. */
  const NodeArg* GetNodeArg(const std::string& name) const {
    return const_cast<Graph*>(this)->GetNodeArg(name);
  }

  // search this and up through any parent_graph_ instance for a NodeArg
  NodeArg* GetNodeArgIncludingParentGraphs(const std::string& node_arg_name);

  /** Gets a mutable NodeArg by name. Creates a new NodeArg that is owned by this Graph if not found.
  @param name The NodeArg name.
  @param[in] p_arg_type Optional TypeProto to use if the NodeArg needs to be created.
  @returns NodeArg reference.
  */
  NodeArg& GetOrCreateNodeArg(const std::string& name, const ONNX_NAMESPACE::TypeProto* p_arg_type) {
    return graph_context_.GetMutableMainFunction()->GetOrCreateNodeArg(name, p_arg_type);
  }

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_ENABLE_RUNTIME_OPTIMIZATION_IN_MINIMAL_BUILD)
  /** Generate a unique name in this Graph for a NodeArg */
  std::string GenerateNodeArgName(const std::string& base_name);

  /** Generate a unique name in this Graph for a Node */
  std::string GenerateNodeName(const std::string& base_name);
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_ENABLE_RUNTIME_OPTIMIZATION_IN_MINIMAL_BUILD)

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

  /** Mark the Graph as needing Resolve() to be called.
  This should be done after modifying any aspect of the Graph that changes the Nodes or relationships between them. */
  Graph& SetGraphResolveNeeded() noexcept {
    graph_resolve_needed_ = true;
    return *this;
  }

  /** Gets flag indicating whether Graph::Resolve needs to be called before using the Graph. */
  bool GraphResolveNeeded() const noexcept {
    return graph_resolve_needed_;
  }

  /** Sets flag that Graph::graph_proto_ needs to be updated to reflect changes in the Graph. */
  Graph& SetGraphProtoSyncNeeded() noexcept {
    graph_proto_sync_needed_ = true;
    return *this;
  }

  /** Gets flag indicating whether Graph::graph_proto_ needs to be synchronized with this Graph instance. */
  bool GraphProtoSyncNeeded() const noexcept {
    return graph_proto_sync_needed_;
  }

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

  /** Gets the map of operator domains to their opset versions. */
  const std::unordered_map<std::string, int>& DomainToVersionMap() const noexcept {
    return domain_to_version_;
  }

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
  //TODO! re-implement this 
  ///**
  //Create a single Node that will be the result of the a fusion of multiple nodes in this Graph.
  //@param sub_graph A IndexSubGraph instance with details of the nodes to fuse.
  //@param fused_node_name The name for the new Node.
  //@returns Node with fused subgraph.
  //@remarks As a new Graph instance for the fused nodes is not created, a GraphViewer can be constructed with the
  //         IndexedSubGraph information to provide a view of the subgraph. The original nodes are left in place
  //         while this is in use.
  //         Call FinalizeFuseSubGraph to remove them once the fused replacement node is fully created.
  //*/
  //Node& BeginFuseSubGraph(const IndexedSubGraph& sub_graph, const std::string& fused_node_name);

  ///**
  //If we have BeginFuseSubGraph, but somehow hit errors, such as Compile of an EP failed on thesub_graph.
  //We can call CancelFuseSubGraph to undo the changes of BeginFuseSubGraph
  //@param fused_node The fused node and it's function body to be removed from the graph
  //*/
  //void CancelFuseSubGraph(const Node& fused_node);

  //void FinalizeFuseSubGraph(const IndexedSubGraph& sub_graph, Node& fused_node);
#endif

#if !defined(ORT_MINIMAL_BUILD)
  /** Gets the GraphProto representation of this Graph. */
  const ONNX_NAMESPACE::GraphProto& ToGraphProto();
  ONNX_NAMESPACE::GraphProto ToGraphProto() const;

  /** Gets the GraphProto representation of this Graph
  @params external_file_name name of the binary file to use for initializers
  @param initializer_size_threshold initializers larger or equal to this threshold (in bytes) are saved
  in the external file. Initializer smaller than this threshold are included in the onnx file.
  @returns GraphProto serialization of the graph.
  */
  ONNX_NAMESPACE::GraphProto ToGraphProtoWithExternalInitializers(const std::string& external_file_name,
                                                                  size_t initializer_size_threshold) const;

  /** Gets the ISchemaRegistry instances being used with this Graph. */
  IOnnxRuntimeOpSchemaCollectionPtr GetSchemaRegistry() const;

  /**
  Looks up the op schema in the schema registry and sets it for the given node.
  @param node The node to update.
  @return Whether the node's op schema was set to a valid value.
  */
  bool SetOpSchemaFromRegistryForNode(Node& node);

  /**
  Create a single Function based Node that is the result of the a fusion of multiple nodes in this Graph.
  A new Graph instance will be created for the fused nodes.
  @param sub_graph A IndexSubGraph instance with details of the nodes to fuse. Ownership is transferred to the new Node
  @param fused_node_name The name for the new Node.
  @returns Function based Node with fused subgraph. The Node body will contain a Function instance.
  */
  Node& FuseSubGraph(const IndexedSubGraph& sub_graph, const std::string& fused_node_name);

  //TODO!!
  //re-implement this
  /**
  Directly insert the nodes in the function Node provided into this Graph.
  @param node Node with Node::Type of Node::Type::Fused
  @returns Status indicating success or providing an error message.
  */
  //Status InlineFunction(Node& node);

  //TODO!!
  //re-implement this
  /** Initialize function body for the given node */
  //void InitFunctionBodyForNode(Node& node);

  /** Gets Model local functions from the root/parent graph.*/
  //const std::unordered_map<std::string, const ONNX_NAMESPACE::FunctionProto*>& GetModelLocalFunctions() const;

  /** Mark a NodeArg name as coming from the outer scope when programmatically constructing a Graph that will
  be used as a GraphProto attribute in another Node..
  e.g. when creating a Graph instance that will be used as a subgraph in a control flow operator, it is necessary to
  define placeholder NodeArgs for outer scope values. This prevents these values from becoming explicit graph inputs
  when the Graph is resolved.
  */
  void AddOuterScopeNodeArg(const std::string& name) {
    graph_context_.GetMutableMainFunction()->AddOuterScopeNodeArg(name);
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

#endif  // !defined(ORT_MINIMAL_BUILD)

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_ENABLE_RUNTIME_OPTIMIZATION_IN_MINIMAL_BUILD)
  /** Sets the type of a NodeArg, replacing existing type/shape if any */
  void SetNodeArgType(NodeArg& arg, const ONNX_NAMESPACE::TypeProto& type_proto);

  const Node* GetProducerNode(const std::string& node_arg_name) const {
    return graph_context_.GetMainFunction().GetProducerNode(node_arg_name);
  }

  Node* GetMutableProducerNode(const std::string& node_arg_name) {
    return graph_context_.GetMutableMainFunction()->GetMutableProducerNode(node_arg_name);
  }

  void UpdateProducerNode(const std::string& node_arg_name, NodeIndex node_index) {
    return graph_context_.GetMutableMainFunction()->UpdateProducerNode(node_arg_name, node_index);
  }

  std::vector<const Node*> GetConsumerNodes(const std::string& node_arg_name) const {
    return graph_context_.GetMainFunction().GetConsumerNodes(node_arg_name);
  }

  // Without removing the existing consumers, add a consumer to the give node arg name.
  void AddConsumerNode(const std::string& node_arg_name, Node* consumer) {
    return graph_context_.GetMutableMainFunction()->AddConsumerNode(node_arg_name, consumer);
  }

  // Remove a consumer from the set
  void RemoveConsumerNode(const std::string& node_arg_name, Node* consumer) {
    return graph_context_.GetMutableMainFunction()->RemoveConsumerNode(node_arg_name, consumer);
  }
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_ENABLE_RUNTIME_OPTIMIZATION_IN_MINIMAL_BUILD)

#if !defined(ORT_MINIMAL_BUILD)
  std::vector<Node*> GetMutableConsumerNodes(const std::string& node_arg_name) {
    return graph_context_.GetMutableMainFunction()->GetMutableConsumerNodes(node_arg_name);
  }

  void UpdateConsumerNodes(const std::string& node_arg_name, const std::vector<Node*>& nodes) {
    return graph_context_.GetMutableMainFunction()->UpdateConsumerNodes(node_arg_name, nodes);
  }

  /** During constant folding it may become possible to infer the shape for a node.
      To avoid running a full Resolve allow an individual node to have the shape inferencing re-run.
  */
  Status UpdateShapeInference(Node& node);

  // Options to control Graph::Resolve.
  struct ResolveOptions {
    // Whether to override existing types with inferred types.
    bool override_types = false;
    // Names of initializers to keep even if unused (optional).
    const std::unordered_set<std::string>* initializer_names_to_preserve = nullptr;
    // Whether to set that no proto sync is required after resolving.
    // Useful for resolving right after loading from a GraphProto.
    bool no_proto_sync_required = false;
    // When set to true, graph resolve will be called for initialized function bodies as well. This is used
    // in case of nested model local functions.
    bool traverse_function_body = false;
  };

  /**
  Resolve this Graph to ensure it is completely valid, fully initialized, and able to be executed.
  1. Run through all validation rules.
    a. Node name and node output's names should be unique.
    b. Attribute match between node and op definition.
    c. Input/Output match between node and op definition.
    d. Graph is acyclic and sort nodes in topological order.
  2. Check & Setup inner nodes' dependency.
  3. Cleanup function definition lists.
  Note: the weights for training can't be cleaned during resolve.
  @returns common::Status with success or error information.
  */
  common::Status Resolve(const ResolveOptions& options);

  common::Status Resolve() {
    ResolveOptions default_options;
    return Resolve(default_options);
  }

  common::Status SaveToOrtFormat(flatbuffers::FlatBufferBuilder& builder,
                                 flatbuffers::Offset<onnxruntime::fbs::Graph>& fbs_graph) const;

#endif  // !defined(ORT_MINIMAL_BUILD)

  /** Returns the Node containing the GraphProto for this Graph instance if IsSubgraph is true */
  const Node* ParentNode() const { return parent_node_; }

  /** Returns true if the name is for a value that is coming from outer scope */
  bool IsOuterScopeValue(const std::string& name) const {
    if (!parent_node_) return false;
    const auto& implicit_input_defs = parent_node_->ImplicitInputDefs();
    return std::any_of(implicit_input_defs.cbegin(), implicit_input_defs.cend(),
                       [&name](const NodeArg* implicit_input) {
                         return implicit_input->Name() == name;
                       });
  }

#if !defined(ORT_MINIMAL_BUILD)
  /** Construct a Graph instance for a subgraph that is created from a GraphProto attribute in a Node.
  Inherits some properties from the parent graph.
  @param parent_graph The Graph containing the Node that has the GraphProto attribute.
  @param parent_node The Node that has the GraphProto attribute.
  @param subgraph_proto The GraphProto from the Node attribute.
  */
  Graph(Graph& parent_graph, const Node& parent_node, ONNX_NAMESPACE::GraphProto& subgraph_proto);
#endif

  virtual ~Graph();

  static common::Status LoadFromOrtFormat(
      const onnxruntime::fbs::Graph& fbs_graph, const Model& owning_model,
      const std::unordered_map<std::string, int>& domain_to_version,
#if !defined(ORT_MINIMAL_BUILD)
      IOnnxRuntimeOpSchemaCollectionPtr schema_registry,
#endif
      const logging::Logger& logger, std::unique_ptr<Graph>& graph);

  // deserialize a subgraph
  static Status LoadFromOrtFormat(const onnxruntime::fbs::Graph& fbs_graph,
                                  Graph& parent_graph, const Node& parent_node,
                                  const logging::Logger& logger, std::unique_ptr<Graph>& graph);

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_ENABLE_RUNTIME_OPTIMIZATION_IN_MINIMAL_BUILD)
  const RuntimeOptimizationRecordContainer& RuntimeOptimizations() const {
    return runtime_optimizations_;
  }

  RuntimeOptimizationRecordContainer& MutableRuntimeOptimizations() {
    return runtime_optimizations_;
  }

  // Stores information collected during the replay of loaded runtime optimizations
  struct RuntimeOptimizationReplayContext {
    std::unordered_map<NodeIndex, HashValue> produced_node_index_to_kernel_def_hash{};
    size_t num_replayed_optimizations{};
  };

  const RuntimeOptimizationReplayContext& RuntimeOptimizationReplayCtx() const {
    return runtime_optimization_replay_context_;
  }

  RuntimeOptimizationReplayContext& MutableRuntimeOptimizationReplayCtx() {
    return runtime_optimization_replay_context_;
  }
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_ENABLE_RUNTIME_OPTIMIZATION_IN_MINIMAL_BUILD)

  // This friendship relationship should only be used to call Graph::Graph and
  // Graph::LoadGraph All other access should be via the public API.
  friend class Model;

  Graph() = delete;

  // Create empty Graph instance to re-create from ORT format serialized data.
  Graph(const Model& owning_model,
        const std::unordered_map<std::string, int>& domain_to_version,
#if !defined(ORT_MINIMAL_BUILD)
        IOnnxRuntimeOpSchemaCollectionPtr schema_registry,
#endif
        Graph* parent_graph, const Node* parent_node,
        const logging::Logger& logger);

  // Populate Graph instance from ORT format serialized data.
  common::Status LoadFromOrtFormat(const onnxruntime::fbs::Graph& fbs_graph);

#if !defined(ORT_MINIMAL_BUILD)
  // Constructor: Given a <GraphProto> loaded from model file, construct
  // a <Graph> object. Used by Model to create a Graph instance.
  Graph(const Model& owning_model,
        ONNX_NAMESPACE::GraphProto* graph_proto,
        const std::unordered_map<std::string, int>& domain_to_version,
        Version ir_version,
        IOnnxRuntimeOpSchemaCollectionPtr schema_registry,
        const std::vector<const ONNX_NAMESPACE::FunctionProto*>& model_functions,
        const logging::Logger& logger);

  // internal use by the Graph class only
  Graph(const Model& owning_model,
        ONNX_NAMESPACE::GraphProto* graph_proto,
        const std::unordered_map<std::string, int>& domain_to_version,
        Version ir_version,
        IOnnxRuntimeOpSchemaCollectionPtr schema_registry,
        Graph* parent_graph,
        const Node* parent_node,
        const std::vector<const ONNX_NAMESPACE::FunctionProto*>& model_functions,
        const logging::Logger& logger);

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(Graph);
  void InitializeStateFromModelFileGraphProto(ONNX_NAMESPACE::GraphProto* graph_proto);

  // Add node with specified <node_proto>.
  Node& AddNode(const ONNX_NAMESPACE::NodeProto& node_proto,
                const ArgNameToTypeMap& name_to_type);

#endif

  Version IrVersion() const noexcept {
    return ir_version_;
  }

  Graph& GraphResolveNeeded(bool needed) noexcept {
    graph_resolve_needed_ = needed;
    return *this;
  }

  Graph& GraphProtoSyncNeeded(bool needed) noexcept {
    graph_proto_sync_needed_ = needed;
    return *this;
  }

  // During the Resolve of a Graph it is necessary to recursively descend into subgraphs (created from GraphProto
  // Node attributes in the Graph) if present.
  // The ResolveContext holds the collection of values for the current Graph instance, be it the main graph
  // or a subgraph, so that the various operations that are part of the Resolve can work iteratively or
  // recursively as needed.
  struct ResolveContext {
    ResolveContext() = default;

    std::unordered_map<std::string, std::pair<Node*, int>> output_args;
    std::unordered_set<std::string> inputs_and_initializers;
    std::unordered_set<std::string> outer_scope_node_args;
    std::unordered_map<std::string, NodeIndex> node_name_to_index;
    std::unordered_set<Node*> nodes_with_subgraphs;

    void Clear() {
      output_args.clear();
      inputs_and_initializers.clear();
      outer_scope_node_args.clear();
      node_name_to_index.clear();
      nodes_with_subgraphs.clear();
    }

   private:
    ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ResolveContext);
  };

  // Initialize all the graph inputs, initializers and outputs
  common::Status InitInputsInitializersOutputs();

  // Initialize overridable initializers container
  void ComputeOverridableInitializers();

#if !defined(ORT_MINIMAL_BUILD)
  // Build and verify node connection (edges).
  // Verify NodeArg name/type/shape matching correctly.
  common::Status BuildConnections(std::unordered_set<std::string>& outer_scope_node_args_consumed);

  common::Status VerifyNoDuplicateName();

  // Check whether <*this> graph is acyclic while performing a topological sort.
  // Depth-first going from bottom up through the graph and checking whether there are any back edges.
  // NodesInTopologicalOrder is updated with the nodes' indexes in topological
  // order if <Status> returned is "OK", otherwise it's undefined.
  common::Status PerformTopologicalSortAndCheckIsAcyclic();

  common::Status PerformTypeAndShapeInferencing(const ResolveOptions& options);

  // Recursively find all subgraphs including nested subgraphs
  void FindAllSubgraphs(std::vector<Graph*>& subgraphs);

  // Iterate this Graph instance and all subgraphs, calling the provided function for each.
  common::Status ForThisAndAllSubgraphs(const std::vector<Graph*>& subgraphs, std::function<Status(Graph&)> func);

  common::Status InferAndVerifyTypeMatch(Node& node, const ONNX_NAMESPACE::OpSchema& op, const ResolveOptions& options);

  // perform type and shape inferencing on the subgraph and Resolve to validate
  static common::Status InferAndVerifySubgraphTypes(const Node& node, Graph& subgraph,
                                                    const std::vector<const ONNX_NAMESPACE::TypeProto*>& input_types,
                                                    std::vector<const ONNX_NAMESPACE::TypeProto*>& output_types,
                                                    const Graph::ResolveOptions& options);

  // Apply type-inference and type-checking to all inputs and initializers:
  common::Status TypeCheckInputsAndInitializers();

  // Compute set of input and initializer names and checking for duplicate names
  common::Status VerifyInputAndInitializerNames();

  // Infer and set type information across <*this> graph if needed, and verify type/attribute
  // information matches between node and op.
  common::Status VerifyNodeAndOpMatch(const ResolveOptions& options);

  // Set graph inputs/outputs when resolving a graph..
  common::Status SetGraphInputsOutputs();

  // recursively accumulate and set the outer scope node args in the resolve context for all subgraphs
  // so they can be used to resolve outer scope dependencies when running BuildConnections for the subgraphs.
  common::Status SetOuterScopeNodeArgs(const std::unordered_set<std::string>& outer_scope_node_args);

  // Clear all unused initializers and NodeArgs
  void CleanUnusedInitializersAndNodeArgs(const std::unordered_set<std::string>* initializer_names_to_preserve = nullptr);

  std::vector<NodeArg*> CreateNodeArgs(const google::protobuf::RepeatedPtrField<std::string>& names,
                                       const ArgNameToTypeMap& name_to_type_map);

  void ToGraphProtoInternal(ONNX_NAMESPACE::GraphProto& graph_proto) const;

#endif  // !defined(ORT_MINIMAL_BUILD)

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_ENABLE_RUNTIME_OPTIMIZATION_IN_MINIMAL_BUILD)
  Status PopulateNodeArgToProducerConsumerLookupsFromNodes();
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_ENABLE_RUNTIME_OPTIMIZATION_IN_MINIMAL_BUILD)

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
  gsl::not_null<Node*> AllocateNode();

  // Release the node.
  // @returns false if node_index was invalid.
  bool ReleaseNode(NodeIndex node_index);

  //TODO!!!
  //Re-implement this
  //Node& CreateFusedSubGraphNode(const IndexedSubGraph& sub_graph, const std::string& fused_node_name);
#endif

  Node* NodeAtIndexImpl(NodeIndex node_index) const {
    // if we are trying to access a node that doesn't exist there's (most
    // likely) either a logic issue or a graph consistency/correctness issue.
    // use ORT_ENFORCE to prove that or uncover scenarios where we actually
    // expect attempts to retrieve a non-existent node.
    return graph_context_.GetMainFunction().NodeAtIndexImpl(node_index);
  }

  const Model& owning_model_;

  // TODO!!
  // unify it later
  // GraphProto that provides storage for the ONNX proto types deserialized from a flexbuffer/flatbuffer
  ONNX_NAMESPACE::GraphProto deserialized_proto_data_;

  // TODO!!
  // revisit it later
#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_ENABLE_RUNTIME_OPTIMIZATION_IN_MINIMAL_BUILD)
  // Runtime optimization storage.
  // Note: runtime_optimizations_ == *runtime_optimizations_ptr_ and must be initialized
  std::unique_ptr<RuntimeOptimizationRecordContainer> runtime_optimizations_ptr_;
  RuntimeOptimizationRecordContainer& runtime_optimizations_;

  RuntimeOptimizationReplayContext runtime_optimization_replay_context_;
#endif

#if !defined(ORT_MINIMAL_BUILD)
  IOnnxRuntimeOpSchemaCollectionPtr schema_registry_;
#endif

  // TODO!!
  // revisit this later
  // A flag indicates whether <*this> graph needs to be resolved.
  bool graph_resolve_needed_ = false;

  bool graph_proto_sync_needed_ = false;

  bool graph_inputs_manually_set_ = false;

  // Graph inputs excluding initializers.
  std::vector<const NodeArg*> graph_inputs_excluding_initializers_;

  // Overridable Initializers. The difference between graph_inputs_including_initializers_
  // and graph_inputs_excluding_initializers_
  std::vector<const NodeArg*> graph_overridable_initializers_;
  
  bool graph_outputs_manually_set_ = false;

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_ENABLE_RUNTIME_OPTIMIZATION_IN_MINIMAL_BUILD)
  int name_generator_ = 0;

  // Strings which have been used as node names.
  // New node name should not conflict with this set.
  std::unordered_set<std::string> generated_node_names_;

  // Strings which have been used as node_arg names.
  // New node_arg name should not conflict this this set.
  std::unordered_set<std::string> generated_node_arg_names_;
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_ENABLE_RUNTIME_OPTIMIZATION_IN_MINIMAL_BUILD)

  //TODO!!
  //this is only used for graph resolve. move it to GraphResolver
  // The topological order of node index used to do node and op match verification temporarily.
  std::vector<NodeIndex> nodes_in_topological_order_;

  const std::unordered_map<std::string, int> domain_to_version_;

  // Model IR version.
  Version ir_version_{ONNX_NAMESPACE::Version::IR_VERSION};

  // Is model using latest ONNX opset
  bool using_latest_onnx_opset_{false};

  ResolveContext resolve_context_;

  // the parent graph if this is a subgraph.
  Graph* parent_graph_;
  // the node containing the graph if parent_graph_ is not nullptr
  const Node* parent_node_;

  // number of times Resolve has run.
  int num_resolves_ = 0;

  const logging::Logger& logger_;

  // distinguishes between graph loaded from model file and graph created from scratch
  const bool is_loaded_from_model_file_;

  // Current graph context, which hold the storages (graph proto, functions, initializers)
  GraphContext graph_context_;
};

#if !defined(ORT_MINIMAL_BUILD)
// Print Graph as, for example,
// Inputs:
//    "Input": tensor(float)
// Nodes:
//    ("add0", Add, "", 7) : ("Input": tensor(float),"Bias": tensor(float),) -> ("add0_out": tensor(float),)
//    ("matmul", MatMul, "", 9) : ("add0_out": tensor(float),"matmul_weight": tensor(float),) -> ("matmul_out": tensor(float),)
//    ("add1", Add, "", 7) : ("matmul_out": tensor(float),"add_weight": tensor(float),) -> ("add1_out": tensor(float),)
//    ("reshape", Reshape, "", 5) : ("add1_out": tensor(float),"concat_out": tensor(int64),) -> ("Result": tensor(float),)
// Outputs:
//    "Result": tensor(float)
// Inputs' and outputs' format is described in document of NodeArg's operator<< above.
// Node format is described in Node's operator<< above.
std::ostream& operator<<(std::ostream& out, const Graph& graph);
#endif

}  // namespace onnxruntime
