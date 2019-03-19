// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

#include "core/common/common.h"
#include "core/common/const_pointer_container.h"
#include "core/common/status.h"
#include "core/graph/basic_types.h"
#include "core/graph/constants.h"
#include "core/graph/graph_nodes.h"
#include "core/graph/node_arg.h"
#include "core/graph/onnx_protobuf.h"
#include "core/graph/function.h"
#include "gsl/gsl_util"
#include "gsl/pointers"

namespace onnxruntime {
class Graph;
struct IndexedSubGraph;
class Node;
class OpSignature;

/**
@class Node
Class representing a node in the graph.
*/
class Node {
 public:
  /** Node types */
  enum class Type {
    Primitive = 0,  ///< The node refers to a primitive operator.
    Fused = 1,      ///< The node refers to a function.
  };

  ~Node() = default;

  /**
  @class EdgeEnd
  Class representing the end of an edge. It could be an input or output edge end of a node.
  For the node's input edge end, it's the source end, as the destination end is the node itself.
  For the node's output edge end, it's the destination end, as the source end is the node itself.
  */
  class EdgeEnd {
   public:
    /**
    Construct an EdgeEnd
    @param node The source node if this is an input edge to the current node,
    or the destination node if this is an output edge from the current node.
    @param src_arg_index The node arg index of source node of the edge.
    @param dst_arg_index The node arg index of destination node of the edge.
    */
    EdgeEnd(const Node& node, int src_arg_index, int dst_arg_index) noexcept;

    /** Construct a control edge.
    @param node The node the edge joins to the current node.
    */
    explicit EdgeEnd(const Node& node) noexcept;

    /** Gets the Node that this EdgeEnd refers to. */
    const Node& GetNode() const noexcept;

    /** Gets the source arg index.
    @returns the source arg index of <*this> edge.*/
    int GetSrcArgIndex() const;

    /** Gets the destination arg index.
    @returns the destination arg index of <*this> edge.*/
    int GetDstArgIndex() const;

   private:
    const Node* node_;
    const int src_arg_index_;
    const int dst_arg_index_;
  };

  /** Gets the Node's NodeIndex. */
  NodeIndex Index() const noexcept;

  /** Gets the Node's name. */
  const std::string& Name() const noexcept;

  /** Gets the Node's operator type. */
  const std::string& OpType() const noexcept;

  /** Gets the domain of the OperatorSet that specifies the operator returned by #OpType. */
  const std::string& Domain() const noexcept;

  /** Gets the Node's OpSchema.
  @remarks The graph containing this node must be resolved, otherwise nullptr will be returned. */
  const ONNX_NAMESPACE::OpSchema* Op() const noexcept;

  /** Gets the Node's Node::Type. */
  Node::Type NodeType() const noexcept;

  /** Gets the function body if the #NodeType is fused, or nullptr if not. */
  const Function* GetFunctionBody() const noexcept;

  /** Gets the node description. */
  const std::string& Description() const noexcept;

  /**
  Helper to iterate through the container returned by #InputDefs() or #OutputDefs() and call the provided function.
  @param node_args Collection of NodeArgs returned by #InputDefs() or #OutputDefs()
  @param func Function to call for each valid NodeArg in the node_args. The function is called with the NodeArg
              and the index number in the container.
  @returns common::Status with success or error information.
  @remarks Returns immediately on error.
  */
  static common::Status ForEachWithIndex(const ConstPointerContainer<std::vector<NodeArg*>>& node_args,
                                         std::function<common::Status(const NodeArg& arg, size_t index)> func) {
    for (size_t index = 0; index < node_args.size(); ++index) {
      auto arg = node_args[index];
      if (!arg->Exists())
        continue;
      ORT_RETURN_IF_ERROR(func(*arg, index));
    }
    return common::Status::OK();
  }

  /** Gets the Node's input definitions.
  @remarks requires ConstPointerContainer wrapper to apply const to the NodeArg pointers so access is read-only. */
  const ConstPointerContainer<std::vector<NodeArg*>> InputDefs() const noexcept {
    return ConstPointerContainer<std::vector<NodeArg*>>(definitions_.input_defs);
  }

  /** Gets a modifiable collection of the Node's input definitions. */
  std::vector<NodeArg*>& MutableInputDefs() noexcept {
    return definitions_.input_defs;
  }

  /** Gets a modifiable collection of the Node's output definitions. */
  std::vector<NodeArg*>& MutableOutputDefs() noexcept {
    return definitions_.output_defs;
  }

  /** Gets the count of arguments for each of the Node's explicit inputs. */
  const std::vector<int>& InputArgCount() const noexcept { return definitions_.input_arg_count; }

  /** Gets a modifiable count of arguments for each of the Node's explicit inputs.
  @todo This should be removed in favor of a method that updates the input args and the count.
        Currently these operations are separate which is not a good setup. */
  std::vector<int>& MutableInputArgsCount() { return definitions_.input_arg_count; }

  /** Gets the implicit inputs to this Node.
  If this Node contains a subgraph, these are the NodeArg's that are implicitly consumed by Nodes within that
  subgraph. e.g. If and Loop operators.*/
  const std::vector<NodeArg*>& ImplicitInputDefs() const noexcept {
    return definitions_.implicit_input_defs;
  }

  /** Gets the Node's output definitions.
  @remarks requires ConstPointerContainer wrapper to apply const to the NodeArg pointers so access is read-only. */
  const ConstPointerContainer<std::vector<NodeArg*>> OutputDefs() const noexcept {
    return ConstPointerContainer<std::vector<NodeArg*>>(definitions_.output_defs);
  }

  /** Struct to provide sorting between EdgeEnd instances based on NodeIndex first, and NodeArg::Name second. */
  struct EdgeEndCompare {
    bool operator()(const EdgeEnd& lhs, const EdgeEnd& rhs) const {
      if (lhs.GetNode().Index() == rhs.GetNode().Index()) {
        if (lhs.GetSrcArgIndex() == rhs.GetSrcArgIndex()) {
          return lhs.GetDstArgIndex() < rhs.GetDstArgIndex();
        }
        return lhs.GetSrcArgIndex() < rhs.GetSrcArgIndex();
      }
      return lhs.GetNode().Index() < rhs.GetNode().Index();
    }
  };

  using EdgeSet = std::set<EdgeEnd, EdgeEndCompare>;
  using EdgeConstIterator = EdgeSet::const_iterator;

  /**
  @class NodeConstIterator
  Class to provide const access to Node instances iterated via an EdgeConstIterator. */
  class NodeConstIterator {
   public:
    NodeConstIterator(EdgeConstIterator p_iter);

    bool operator==(const NodeConstIterator& p_other) const;

    bool operator!=(const NodeConstIterator& p_other) const;

    void operator++();
    void operator--();

    const Node& operator*();

   private:
    EdgeConstIterator m_iter;
  };

  // Functions defined to traverse a Graph as below.

  /** Gets an iterator to the beginning of the input nodes to this Node. */
  NodeConstIterator InputNodesBegin() const noexcept { return NodeConstIterator(relationships_.input_edges.cbegin()); };
  /** Gets an iterator to the end of the input nodes to this Node. */
  NodeConstIterator InputNodesEnd() const noexcept { return NodeConstIterator(relationships_.input_edges.cend()); }

  /** Gets an iterator to the beginning of the output nodes from this Node. */
  NodeConstIterator OutputNodesBegin() const noexcept { return NodeConstIterator(relationships_.output_edges.cbegin()); }
  /** Gets an iterator to the end of the output nodes from this Node. */
  NodeConstIterator OutputNodesEnd() const noexcept { return NodeConstIterator(relationships_.output_edges.cend()); }

  /** Gets an iterator to the beginning of the input edges to this Node.
  @remarks There are no nullptr entries in this collection. */
  EdgeConstIterator InputEdgesBegin() const noexcept { return relationships_.input_edges.cbegin(); }

  /** Gets an iterator to the end of the input edges to this Node. */
  EdgeConstIterator InputEdgesEnd() const noexcept { return relationships_.input_edges.cend(); }

  /** Gets an iterator to the beginning of the output edges from this Node.
  @remarks There are no nullptr entries in this collection. */
  EdgeConstIterator OutputEdgesBegin() const noexcept { return relationships_.output_edges.cbegin(); }

  /** Gets an iterator to the end of the output edges from this Node. */
  EdgeConstIterator OutputEdgesEnd() const noexcept { return relationships_.output_edges.cend(); }

  /** Gets the Node's control inputs. */
  const std::set<std::string>& ControlInputs() const noexcept { return relationships_.control_inputs; }

  /** Gets the number of input edges to this Node */
  size_t GetInputEdgesCount() const noexcept { return relationships_.input_edges.size(); }

  /** Gets the number of output edges from this Node */
  size_t GetOutputEdgesCount() const noexcept { return relationships_.output_edges.size(); }

  /** Add an attribute to this Node with specified attribute name and value. */
  void AddAttribute(const std::string& attr_name, const ONNX_NAMESPACE::AttributeProto& value);

#define ADD_ATTR_INTERFACES(TypeName)                                     \
  void AddAttribute(const std::string& attr_name, const TypeName& value); \
  void AddAttribute(const std::string& attr_name,                         \
                    const std::vector<TypeName>& values);

  ADD_ATTR_INTERFACES(int64_t)
  ADD_ATTR_INTERFACES(float)
  ADD_ATTR_INTERFACES(std::string)
  ADD_ATTR_INTERFACES(ONNX_NAMESPACE::TensorProto)
  ADD_ATTR_INTERFACES(ONNX_NAMESPACE::GraphProto)

  /** Remove the specified attribute from this Node */
  bool ClearAttribute(const std::string& attr_name);

  /** Gets the Node's attributes. */
  const NodeAttributes& GetAttributes() const noexcept;

  /** Gets the Graph instance that is instantiated from a GraphProto attribute during Graph::Resolve.
  @param attr_name Attribute name for the GraphProto attribute.
  @returns nullptr if the Graph instance has not been instantiated or attribute does not contain a GraphProto.
  */
  const Graph* GetGraphAttribute(const std::string& attr_name) const;

  /** Gets the mutable Graph instance that is instantiated from a GraphProto attribute during Graph::Resolve.
  @param attr_name Attribute name for the GraphProto attribute.
  @returns nullptr if the Graph instance has not been instantiated or attribute does not contain a GraphProto.
  */
  Graph* GetMutableGraphAttribute(const std::string& attr_name);

  /** Gets a map of attribute name to the mutable Graph instances for all subgraphs of the Node.
  @returns Map of the attribute name that defines the subgraph to the subgraph's Graph instance.
           nullptr if the Node has no subgraphs.
  */
  const std::unordered_map<std::string, gsl::not_null<Graph*>>& GetAttributeNameToMutableSubgraphMap() {
    return attr_to_subgraph_map_;
  }

  /** Gets the execution ProviderType that this node will be executed by. */
  ProviderType GetExecutionProviderType() const noexcept;

  /** Sets the execution ProviderType that this Node will be executed by. */
  void SetExecutionProviderType(ProviderType execution_provider_type);

  /** Gets the NodeProto representation of this Node. */
  void ToProto(ONNX_NAMESPACE::NodeProto& proto) const;

  /** Call the provided function for all explicit inputs, implicit inputs, and outputs of this Node.
      If the NodeArg is an explicit or implicit input, is_input will be true when func is called.
      @param include_missing_optional_defs Include NodeArgs that are optional and were not provided
                                           i.e. NodeArg::Exists() == false.
      */
  void ForEachDef(std::function<void(const onnxruntime::NodeArg&, bool is_input)> func,
                  bool include_missing_optional_defs = false) const;

  /** Replaces any matching definitions in the Node's explicit inputs or explicit outputs.
  @param replacements Map of current NodeArg to replacement NodeArg.
  */
  void ReplaceDefs(const std::map<const onnxruntime::NodeArg*, onnxruntime::NodeArg*>& replacements);

  /**
  @class Definitions
  The input and output definitions for this Node.
  */
  class Definitions {
   public:
    Definitions() noexcept = default;

    /** The Node's explicit input definitions. */
    std::vector<NodeArg*> input_defs;

    /**
    The number of inputs for each argument of the operator or function which this node refers.
    @remarks For example, #input_defs has 10 elements (inputs), and #input_arg_count is {4, 6}.
    This means that 4 elements (inputs) of input_defs map to the first argument of the operator or function, and
    the other 6 map to the second argument.
    */
    std::vector<int> input_arg_count;

    /** The Node's output definitions. */
    std::vector<NodeArg*> output_defs;

    /** The Node's implicit input definitions if the Node contains one or more subgraphs
    (i.e. GraphProto attributes) and the subgraph/s implicitly consume these values.
    @remarks For example, a subgraph in an 'If' node gets all its input values via this mechanism rather than
    there being explicit inputs to the 'If' node that are passed to the subgraph.
    They are pseudo-inputs to this Node as it has an implicit dependency on them. */
    std::vector<NodeArg*> implicit_input_defs;

   private:
    ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(Definitions);
  };

  /**
  @class Relationships
  Defines the relationships between this Node and other Nodes in the Graph.
  */
  class Relationships {
   public:
    Relationships() = default;

    void Clear() noexcept {
      input_edges.clear();
      output_edges.clear();
      control_inputs.clear();
    }

    /** The edges for Nodes that provide inputs to this Node. */
    EdgeSet input_edges;

    /** The edges for Nodes that receive outputs from this Node. */
    EdgeSet output_edges;

    /** The Node names of the control inputs to this Node. */
    std::set<std::string> control_inputs;

   private:
    ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(Relationships);
  };

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(Node);

  // NOTE: This friendship relationship should ONLY be used for calling methods of the Node class and not accessing
  // the data members directly, so that the Node can maintain its internal invariants.
  friend class Graph;

  Node(NodeIndex index, Graph& graph) : index_(index), graph_(&graph) {}

  void Init(const std::string& name,
            const std::string& op_type,
            const std::string& description,
            const std::vector<NodeArg*>& input_args,
            const std::vector<NodeArg*>& output_args,
            const NodeAttributes* attributes,
            const std::string& domain);

  // create a Graph instance for an attribute that contains a GraphProto
  void CreateSubgraph(const std::string& attr_name);

  // internal only method to allow selected classes to directly alter the input/output definitions and arg counts
  Definitions& MutableDefinitions() noexcept;

  // internal only method to allow selected classes to directly alter the links between nodes.
  Relationships& MutableRelationships() noexcept;

  const std::vector<std::unique_ptr<Graph>>& MutableSubgraphs() noexcept { return subgraphs_; }

  const Definitions& GetDefinitions() const noexcept { return definitions_; }
  const Relationships& GetRelationships() const noexcept { return relationships_; }

  void SetNodeType(Node::Type node_type) noexcept;

  void SetFunctionBody(const Function& func);

  // validate and update the input arg count
  common::Status UpdateInputArgCount();

  // Node index. Default to impossible value rather than 0.
  NodeIndex index_ = std::numeric_limits<NodeIndex>::max();

  // Node name.
  std::string name_;

  // Node operator type.
  std::string op_type_;

  // OperatorSet domain of op_type_.
  std::string domain_;

  // OperatorSchema that <*this> node refers to.
  const ONNX_NAMESPACE::OpSchema* op_ = nullptr;
  Node::Type node_type_ = Node::Type::Primitive;

  // The function body is owned by graph_
  const Function* func_body_ = nullptr;

  // Node doc string.
  std::string description_;

  // input/output defs and arg count
  Definitions definitions_;

  // Relationships between this node and others in the graph
  Relationships relationships_;

  // Device.
  std::string execution_provider_type_;

  // Map from attribute name to attribute.
  // This allows attribute adding and removing.
  NodeAttributes attributes_;

  // Graph that contains this Node
  Graph* graph_;

  // Map of attribute name to the Graph instance created from the GraphProto attribute
  std::unordered_map<std::string, gsl::not_null<Graph*>> attr_to_subgraph_map_;

  // Graph instances for subgraphs that are owned by this Node
  std::vector<std::unique_ptr<Graph>> subgraphs_;
};

/**
@class Graph
The Graph representation containing the graph inputs and outputs, the Node instances,
and the edges connecting the nodes.
*/
class Graph {
 public:
  /**
  Resolve this Graph to ensure it is completely valid, fully initialized, and able to be executed.
  1. Run through all validation rules.
      a. Node name and node output's names should be unique.
      b. Attribute match between node and op definition.
      c. Input/Output match between node and op definition.
      d. Graph is acyclic and sort nodes in topological order.
  2. Check & Setup inner nodes' dependency.
  3. Cleanup function definition lists.
  @returns common::Status with success or error information.
  */
  common::Status Resolve();

  /** Gets the Graph name. */
  const std::string& Name() const noexcept;
  /** Sets the Graph name. */
  void SetName(const std::string& name);

  /** Gets the Graph description. */
  const std::string& Description() const noexcept;
  /** Gets the Graph description. */
  void SetDescription(const std::string& description);

  /** Add an initializer tensor to the Graph. */
  void AddInitializedTensor(const ONNX_NAMESPACE::TensorProto& tensor_proto);

  /** Remove the initializer tensor with the provided name from the Graph. */
  void RemoveInitializedTensor(const std::string& tensor_name);

  /** Gets an initializer tensor with the provided name.
  @param[out] value Set to the TensorProto* if the initializer is found, or nullptr if not.
  @returns True if found.
  */
  bool GetInitializedTensor(const std::string& tensor_name, const ONNX_NAMESPACE::TensorProto*& value) const;

  /** Gets all the initializer tensors in this Graph. */
  const InitializedTensorSet& GetAllInitializedTensors() const noexcept;

  /** Removes all initializer tensors from this Graph and releases the memory they were using. */
  void CleanAllInitializedTensors() noexcept;

  /** Gets the Graph inputs excluding initializers.
  These are the required inputs to the Graph as the initializers can be optionally overridden via graph inputs.
  @remarks Contains no nullptr values. */
  const std::vector<const NodeArg*>& GetInputs() const noexcept { return graph_inputs_excluding_initializers_; }

  /** Gets the Graph inputs including initializers.
  This is the full set of inputs, in the same order as defined in the GraphProto.
  @remarks Contains no nullptr values. */
  const std::vector<const NodeArg*>& GetInputsIncludingInitializers() const noexcept {
    return graph_inputs_including_initializers_;
  }

  /** Gets the Graph outputs.
  @remarks Contains no nullptr values.*/
  const std::vector<const NodeArg*>& GetOutputs() const noexcept { return graph_outputs_; }

  /** Returns true if a Node output is a Graph output. */
  bool IsNodeOutputsInGraphOutputs(const Node& node) {
    for (auto output_def : node.OutputDefs()) {
      if (std::find(GetOutputs().cbegin(), GetOutputs().cend(), output_def) != GetOutputs().cend()) {
        return true;
      }
    }
    return false;
  }

  /** Gets the NodeArgs that represent value_info instances in the Graph.
  These are the values that are neither Graph inputs nor outputs.
  @remarks Contains no nullptr values. */
  const std::vector<const NodeArg*>& GetValueInfo() const noexcept;

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

  /** Gets the maximum NodeIndex value used in the Graph. */
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

  /** Gets the const NodeArg with the provided name.
  @returns Pointer to const NodeArg if found, nullptr if not. */
  const NodeArg* GetNodeArg(const std::string& name) const {
    return const_cast<Graph*>(this)->GetNodeArg(name);
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

  /** Generate a unique name.in this Graph for a NodeArg */
  std::string GenerateNodeArgName(const std::string& base_name);

  /** Generate a unique name.in this Graph for a Node */
  std::string GenerateNodeName(const std::string& base_name);

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

  /** Copy a Node and add it to this Graph.
  @param other Node to copy
  @returns Reference to the Node that was created and added to this Graph.
  @remarks Do not call AddNode and Remove Node concurrently as they are not thread-safe.
  */
  Node& AddNode(const Node& other);

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

  /**
  Add a control edge between two Nodes in this Graph.
  The source Node does not produce output that is directly consumed by the destination Node, however the
  destination Node must execute after the source node. The control edge allows this ordering to occur.
  */
  bool AddControlEdge(NodeIndex src_node_index, NodeIndex dst_node_index);

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

  /** Gets the map of operator domains to their opset versions. */
  const std::unordered_map<std::string, int>& DomainToVersionMap() const noexcept {
    return domain_to_version_;
  }

  /** Gets the GraphProto representation of this Graph. */
  const ONNX_NAMESPACE::GraphProto& ToGraphProto();

  /** Gets the ISchemaRegistry instances being used with this Graph. */
  IOnnxRuntimeOpSchemaCollectionPtr GetSchemaRegistry() const;

  /**
  Create a single Node that is the result of the a fusion of multiple nodes in this Graph.
  @param sub_graph A IndexSubGraph instance with details of the nodes to fuse.
  @param fused_node_name The name for the new Node.
  @returns Node with fused subgraph.
  */
  Node& FuseSubGraph(std::unique_ptr<IndexedSubGraph> sub_graph, const std::string& fused_node_name);

  /**
  Directly insert the nodes in the function Node provided into this Graph.
  @param node Node with Node::Type of Node::Type::Fused
  @returns Status indicating success or providing an error message.
  */
  Status InlineFunction(Node& node);

  /** Mark a NodeArg name as coming from the outer scope when programmatically constructing a Graph that will
  be used as a GraphProto attribute in another Node..
  e.g. when creating a Graph instance that will be used as a subgraph in a control flow operator, it is necessary to
  define placeholder NodeArgs for outer scope values. This prevents these values from becoming explicit graph inputs
  when the Graph is resolved.
  */
  void AddOuterScopeNodeArg(const std::string& name) {
    ORT_IGNORE_RETURN_VALUE(outer_scope_node_arg_names_.insert(name));
  }

  /** When programmatically constructing a Graph, explicitly set the order to use for graph inputs when the graph is
  resolved.
  This will determine the graph input order when the Graph is converted to a GraphProto by Graph::ToGraphProto.
  @param inputs NodeArgs that represent graph inputs which need to be explicitly ordered.
  Any graph inputs not in this list will be appended to the ordered graph input list, in the order that they were first
  used by Nodes (i.e. the order of Node creation implicitly determines the ordering).
  @remarks If the Graph was loaded from a GraphProto this has no effect.*/
  void SetInputOrder(const std::vector<const NodeArg*> inputs) {
    graph_input_order_ = inputs;
  }

  /** When programmatically constructing a Graph, explicitly set the order to use for graph outputs when the graph is
  resolved.
  This will determine the graph output order when the Graph is converted to a GraphProto by Graph::ToGraphProto.
  @param outputs NodeArgs that represent graph outputs which need to be explicitly ordered.
  Any graph outputs not in this list will be appended to the ordered graph output list, in the order that they were first
  produced by Nodes (i.e. the order of Node creation implicitly determines the ordering).
  @remarks If the Graph was loaded from a GraphProto this has no effect.*/
  void SetOutputOrder(const std::vector<const NodeArg*> outputs) {
    graph_output_order_ = outputs;
  }

  /** Construct a Graph instance for a subgraph that is created from a GraphProto attribute in a Node.
  Inherits some properties from the parent graph.
  @param parent_graph The Graph containing the Node which has a GraphProto attribute.
  @param subgraph_proto The GraphProto from the Node attribute.
  */
  Graph(Graph& parent_graph, ONNX_NAMESPACE::GraphProto& subgraph_proto);

  virtual ~Graph();

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(Graph);

  // This friendship relationship should only be used to call Graph::Graph and
  // Graph::LoadGraph All other access should be via the public API.
  friend class Model;

  Graph() = delete;

  // Constructor: Given a <GraphProto> loaded from model file, construct
  // a <Graph> object. Used by Model to create a Graph instance.
  Graph(ONNX_NAMESPACE::GraphProto* graph_proto,
        const std::unordered_map<std::string, int>& domain_to_version,
        Version ir_version,
        IOnnxRuntimeOpSchemaCollectionPtr schema_registry,
        const std::unordered_map<std::string, const ONNX_NAMESPACE::FunctionProto*>& model_functions = {});

  // internal use by the Graph class only
  Graph(ONNX_NAMESPACE::GraphProto* graph_proto,
        const std::unordered_map<std::string, int>& domain_to_version,
        Version ir_version,
        IOnnxRuntimeOpSchemaCollectionPtr schema_registry,
        Graph* parent_graph,
        const std::unordered_map<std::string, const ONNX_NAMESPACE::FunctionProto*>& model_functions = {});

  // Add node with specified <node_proto>.
  Node& AddNode(const ONNX_NAMESPACE::NodeProto& node_proto,
                const ArgNameToTypeMap& name_to_type);

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

  // search this and up through any parent_graph_ instance for a NodeArg
  NodeArg* GetNodeArgIncludingParentGraphs(const std::string& node_arg_name);

  // Initialize all the graph inputs, initializers and outputs
  common::Status InitInputsInitializersOutputs();

  // recursively accumulate and set the outer scope node args in the resolve context for all subgraphs
  // so they can be used to resolve outer scope dependencies when running BuildConnections for the subgraphs.
  common::Status SetOuterScopeNodeArgs(const std::unordered_set<std::string>& outer_scope_node_args);

  // Build and verify node connection (edges).
  // Verify NodeArg name/type/shape matching correctly.
  common::Status BuildConnections(std::vector<std::string>& outer_scope_node_args_consumed);

  common::Status VerifyNoDuplicateName();

  // Check whether <*this> graph is acyclic while performing a topological sort.
  // Depth-first going from bottom up through the graph and checking whether there are any back edges.
  // NodesInTopologicalOrder is updated with the nodes' indexes in topological
  // order if <Status> returned is "OK", otherwise it's undefined.
  common::Status PerformTopologicalSortAndCheckIsAcyclic();

  common::Status PerformTypeAndShapeInferencing();

  enum class Type {
    // A main graph.
    Main = 1,
    // A sub graph (function).
    Sub = 2,
  };

  common::Status Resolve(bool no_proto_sync_required);

  // Recursively find all subgraphs including nested subgraphs
  void FindAllSubgraphs(std::vector<Graph*>& subgraphs);

  // Iterate this Graph instance and all subgraphs, calling the provided function for each.
  common::Status ForThisAndAllSubgraphs(const std::vector<Graph*>& subgraphs, std::function<Status(Graph&)> func);

  common::Status InferAndVerifyTypeMatch(Node& node, const ONNX_NAMESPACE::OpSchema& op);

  // perform type and shape inferencing on the subgraph and Resolve to validate
  static common::Status InferAndVerifySubgraphTypes(const Node& node, Graph& subgraph,
                                                    const std::vector<const ONNX_NAMESPACE::TypeProto*>& input_types,
                                                    std::vector<const ONNX_NAMESPACE::TypeProto*>& output_types);

  // Apply type-inference and type-checking to all inputs and initializers:
  common::Status TypeCheckInputsAndInitializers();

  // Compute set of input and initializer names and checking for duplicate names
  common::Status VerifyInputAndInitializerNames();

  // Infer and set type information across <*this> graph if needed, and verify type/attribute
  // information matches between node and op.
  common::Status VerifyNodeAndOpMatch();

  // Set graph inputs/outputs when resolving a graph..
  common::Status SetGraphInputsOutputs();

  // Sync graph inputs/outputs when serializing to proto.
  void SyncGraphInputsOutputs();

  // Clear all unused initializers
  void CleanUnusedInitializers();

  gsl::not_null<Node*> AllocateNode();

  // Release the node.
  // @returns false if node_index was invalid.
  bool ReleaseNode(NodeIndex node_index);

  Node* NodeAtIndexImpl(NodeIndex node_index) const {
    // if we are trying to access a node that doesn't exist there's (most
    // likely) either a logic issue or a graph consistency/correctness issue.
    // use ORT_ENFORCE to prove that or uncover scenarios where we actually
    // expect attempts to retrieve a non-existent node.
    ORT_ENFORCE(node_index < nodes_.size(), "Validating no unexpected access using an invalid node_index.");
    return nodes_[node_index].get();
  }

  std::vector<NodeArg*> CreateNodeArgs(const google::protobuf::RepeatedPtrField<std::string>& names,
                                       const ArgNameToTypeMap& name_to_type_map);

  bool IsSubgraph() const { return parent_graph_ != nullptr; }

  void AddFunction(const ONNX_NAMESPACE::FunctionProto* func_proto);

  // GraphProto to store name, version, initializer.
  // When serializing <*this> Graph to a GraphProto, the nodes and
  // functions in <Graph> will also be fed into <graph_proto_> so that
  // it's consistent with <*this> graph.
  // This pointer is owned by parent model.
  ONNX_NAMESPACE::GraphProto* graph_proto_;

  InitializedTensorSet name_to_initial_tensor_;
  std::vector<int> removed_initializer_indexes_;

  Type graph_type_ = Type::Main;

  IOnnxRuntimeOpSchemaCollectionPtr schema_registry_;

  std::vector<std::unique_ptr<onnxruntime::Function>> function_container_;

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

  // A flag indicates whether <*this> graph needs to be resolved.
  bool graph_resolve_needed_ = false;

  bool graph_proto_sync_needed_ = false;

  // The topological order of node index used to do node and op match verification temporarily.
  std::vector<NodeIndex> nodes_in_topological_order_;

  // Full list of graph inputs. Matches number and order of inputs in the GraphProto.
  std::vector<const NodeArg*> graph_inputs_including_initializers_;

  // Graph inputs excluding initializers.
  std::vector<const NodeArg*> graph_inputs_excluding_initializers_;

  // Graph outputs.
  std::vector<const NodeArg*> graph_outputs_;

  // Graph value_info.
  std::vector<const NodeArg*> value_info_;

  // All node args owned by <*this> graph. Key is node arg name.
  std::unordered_map<std::string, std::unique_ptr<NodeArg>> node_args_;

  const std::unordered_map<std::string, int> domain_to_version_;

  std::unordered_map<std::string, const ONNX_NAMESPACE::FunctionProto*> model_functions_;

  // Model IR version.
  Version ir_version_{};

  int name_generator_ = 0;

  ResolveContext resolve_context_;

  // the parent graph if this is a subgraph.
  Graph* parent_graph_;

  // NodeArgs that come from outer scope. Used when building a graph so that
  // these don't get recorded as graph inputs in the GraphProto.
  std::unordered_set<std::string> outer_scope_node_arg_names_;

  // Explicit graph input order to be used when constructing a Graph manually.
  std::vector<const NodeArg*> graph_input_order_;

  // Explicit graph output order to be used when constructing a Graph manually.
  std::vector<const NodeArg*> graph_output_order_;
};

}  // namespace onnxruntime
