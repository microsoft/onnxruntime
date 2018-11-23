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

// A node representation class.
class Node {
 public:
  // Node types.
  enum class Type {
    // A node refers to a primitive operator.
    Primitive = 0,
    // A node refers to a function.
    Fused = 1,
  };

  ~Node() = default;

  // An edge end. It could be input or output edge end of a node.
  // For node's input edge end, it's the source end, as the destination
  // end is the node itself.
  // For node's output edge end, it's the destination end, as the source
  // end is the node itself.
  class EdgeEnd {
   public:
    // Constructor.
    // An EdgeEnd contains a Node and NodeArg.
    EdgeEnd(const Node& node, const NodeArg& node_arg) noexcept;
    // A control edge, which does not have NodeArg.
    EdgeEnd(const Node& node) noexcept;

    // Get the <Node*> that this edge end refers to.
    const Node& GetNode() const noexcept;

    // Get the <NodeArg*> that this edge end refers to.
    const NodeArg* GetNodeArg() const noexcept;

   private:
    const Node* node_;
    const NodeArg* node_arg_;
  };

  // Get node index.
  NodeIndex Index() const noexcept;

  // Get node name.
  const std::string& Name() const noexcept;

  // Get node operator type.
  const std::string& OpType() const noexcept;

  // Get the domain of the OperatorSet that specifies the operator named by <op_type_>.
  const std::string& Domain() const noexcept;

  // Get the OperatorSchema this node refers to. ValidateOpType() must have been called previously.
  // May be null in the future.
  const ONNX_NAMESPACE::OpSchema* Op() const noexcept;
  Node::Type NodeType() const noexcept;
  // Get function body if the node type is fused.
  // The function body is owned by <*this> node's parent graph.
  const Function* GetFunctionBody() const noexcept;

  // Get node description.
  const std::string& Description() const noexcept;

  // Iterate through Input/OutputDefs() with index, note the loop early terminates with error.
  static common::Status ForEachWithIndex(
      const ConstPointerContainer<std::vector<NodeArg*>>& nodeArgVec,
      std::function<common::Status(const NodeArg& arg, size_t index)> func) {
    for (size_t index = 0; index < nodeArgVec.size(); ++index) {
      auto arg = nodeArgVec[index];
      if (!arg->Exists())
        continue;
      ONNXRUNTIME_RETURN_IF_ERROR(func(*arg, index));
    }
    return common::Status::OK();
  }

  // read only access. requires special wrapper to apply const to the NodeArg
  const ConstPointerContainer<std::vector<NodeArg*>> InputDefs() const noexcept {
    return ConstPointerContainer<std::vector<NodeArg*>>(definitions_.input_defs);
  }

  const std::vector<int>& InputArgCount() const noexcept { return definitions_.input_arg_count; }

  std::vector<int>& MutableInputArgsCount() { return MutableDefinitions().input_arg_count; }

  // If this Node contains a subgraph, the NodeArg's that are implicitly consumed by Nodes within that subgraph.
  const std::vector<const NodeArg*>& ImplicitInputDefs() const noexcept {
    return definitions_.implicit_input_defs;
  }

  // read only access. requires special wrapper to apply const to the NodeArg
  const ConstPointerContainer<std::vector<NodeArg*>> OutputDefs() const noexcept {
    return ConstPointerContainer<std::vector<NodeArg*>>(definitions_.output_defs);
  }

  std::vector<NodeArg*>& MutableInputDefs() noexcept {
    return MutableDefinitions().input_defs;
  }

  struct EdgeEndCompare {
    bool operator()(const EdgeEnd& lhs, const EdgeEnd& rhs) const {
      if (lhs.GetNode().Index() == rhs.GetNode().Index()) {
        auto lhs_arg = lhs.GetNodeArg();
        auto rhs_arg = rhs.GetNodeArg();
        std::string lhs_arg_name = lhs_arg == nullptr ? "" : lhs_arg->Name();
        std::string rhs_arg_name = rhs_arg == nullptr ? "" : rhs_arg->Name();
        return lhs_arg_name.compare(rhs_arg_name) < 0;
      }
      return lhs.GetNode().Index() < rhs.GetNode().Index();
    }
  };

  using EdgeSet = std::set<EdgeEnd, EdgeEndCompare>;
  using EdgeConstIterator = EdgeSet::const_iterator;
  class NodeConstIterator {
   public:
    NodeConstIterator(EdgeConstIterator p_iter);

    bool operator==(const NodeConstIterator& p_other) const;

    bool operator!=(const NodeConstIterator& p_other) const;

    void operator++();
    void operator--();

    const Node* operator*();

   private:
    EdgeConstIterator m_iter;
  };

  // Functions defined to traverse a Graph as below.
  // Read all input nodes of <*this>.
  // Beginning of input nodes. Iterator should have no nullptr values.
  NodeConstIterator InputNodesBegin() const noexcept { return NodeConstIterator(relationships_.input_edges.cbegin()); };
  // End of input nodes.
  NodeConstIterator InputNodesEnd() const noexcept { return NodeConstIterator(relationships_.input_edges.cend()); }

  // Beginning of output nodes. Iterator should have no nullptr values.
  NodeConstIterator OutputNodesBegin() const noexcept { return NodeConstIterator(relationships_.output_edges.cbegin()); }
  // End of output nodes.
  NodeConstIterator OutputNodesEnd() const noexcept { return NodeConstIterator(relationships_.output_edges.cend()); }

  // Beginning of input edge. Iterator should have no nullptr values.
  EdgeConstIterator InputEdgesBegin() const noexcept { return relationships_.input_edges.cbegin(); }

  // End of input nodes.
  EdgeConstIterator InputEdgesEnd() const noexcept { return relationships_.input_edges.cend(); }

  // Beginning of output edge. Iterator should have no nullptr values.
  EdgeConstIterator OutputEdgesBegin() const noexcept { return relationships_.output_edges.cbegin(); }

  // End of output nodes.
  EdgeConstIterator OutputEdgesEnd() const noexcept { return relationships_.output_edges.cend(); }

  const std::set<std::string>& ControlInputs() const noexcept { return relationships_.control_inputs; }

  size_t GetInputEdgesCount() const noexcept { return relationships_.input_edges.size(); }

  size_t GetOutputEdgesCount() const noexcept { return relationships_.output_edges.size(); }

  // Add a node attribute with specified attribute name and value.
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

  // Clear specified node attribute.
  bool ClearAttribute(const std::string& attr_name);

  // Get node attributes.
  const NodeAttributes& GetAttributes() const noexcept;

  // get the Graph instance that is instantiated from a GraphProto attribute
  // when the main Graph is resolved. returns nullptr if the Graph instance
  // has not been instantiated.
  const Graph* GetGraphAttribute(const std::string& attr_name) const;

  // get the Graph instance that is instantiated from a GraphProto attribute
  // when the main Graph is resolved. returns nullptr if the Graph instance
  // has not been instantiated.
  Graph* GetMutableGraphAttribute(const std::string& attr_name);

  // Indicates on which we will run this node in runtime.
  // Executor will decide which device that this node will run against
  // and set it properly.
  // TODO: may change the return value type to be an ENUM.
  ProviderType GetExecutionProviderType() const noexcept;
  void SetExecutionProviderType(ProviderType execution_provider_type);

  // Get the corresponding <NodeProto>.
  void ToProto(ONNX_NAMESPACE::NodeProto& proto) const;

  // iterate through all input/output defs
  void ForEachDef(std::function<void(const onnxruntime::NodeArg*, bool is_input)> func) const;

  // iterate through all input defs
  void ForEachInputDef(std::function<void(const onnxruntime::NodeArg*)> func) const;

  // iterate through all output defs
  void ForEachOutputDef(std::function<void(const onnxruntime::NodeArg*)> func) const;

  // Replaces defs
  void ReplaceDefs(const std::map<const onnxruntime::NodeArg*, onnxruntime::NodeArg*>& replacements);

  // Node definitions. Really a struct but we want to prevent accidental copies.
  class Definitions {
   public:
    Definitions() noexcept = default;

    // Node inputs' definition.
    std::vector<NodeArg*> input_defs;

    // The number of inputs for each argument of the operator or function which
    // this node refers.
    // For example, <input_defs_> has 10 elements (inputs), and
    // <input_arg_count_> is {4, 6}. This means that 4 elements (inputs) of
    // <input_defs_> map to the first argument of the operator or function, and
    // the other 6 map to the second argument.
    std::vector<int> input_arg_count;

    // Node outputs' definition.
    std::vector<NodeArg*> output_defs;

    // For a Node that contains a subgraph, NodeArg instances that are consumed by Nodes in a subgraph.
    // e.g. the subgraph in an 'If' node gets all its input values via this mechanism
    //      rather than explicit inputs.
    // They are pseudo-inputs to this Node as it has an implicit dependency on them.
    std::vector<const NodeArg*> implicit_input_defs;

   private:
    ONNXRUNTIME_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(Definitions);
  };

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 26439)
#endif
  class Relationships {
   public:
    Relationships() = default;

    void Clear() noexcept {
      input_edges.clear();
      output_edges.clear();
      control_inputs.clear();
    }

    // Node input edges.
    EdgeSet input_edges;
    // Node output edges.
    EdgeSet output_edges;

    // Control input nodes' names.
    std::set<std::string> control_inputs;

   private:
    ONNXRUNTIME_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(Relationships);
  };

 private:
  ONNXRUNTIME_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(Node);

  // NOTE: These friendship relationships should ONLY be used for calling the
  // following methods so that the Node can maintain its internal invariants as
  // well as possible. Node::Node Node::Init Node::MutableDefinitions
  // Node::MutableRelationships
  // Node::ValdiateVersion
  // All other calls should be made through the public Node interface.
  // Friend classes should NOT be directly accessing any member variables.
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

  // internal only method to allow selected classes to directly alter
  // the input/output definitions and arg counts
  Definitions& MutableDefinitions() noexcept;

  // internal only method to allow selected classes to directly alter
  // the links between nodes.
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

  // OperatorSet domain of <op_type_).
  std::string domain_;

  // OperatorSchema that <*this> node refers to.
  const ONNX_NAMESPACE::OpSchema* op_ = nullptr;
  Node::Type node_type_ = Node::Type::Primitive;
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

  std::unordered_map<std::string, Graph*> attr_to_subgraph_map_;
  std::vector<std::unique_ptr<Graph>> subgraphs_;
};

#ifdef _MSC_VER
#pragma warning(pop)
#endif

class Graph {
 public:
  // Resolve <*this> graph to ensure it's in a good shape with full
  // functionality.
  // 1. Run through all validation rules.
  //    a. Node name and node output's names should be unique.
  //    b. Attribute match between node and op definition.
  //    c. Input/Output match between node and op definition.
  //    d. Graph is acyclic and sort nodes in topological order.
  // 2. Check & Setup inner nodes' dependency.
  // 3. Cleanup function definition lists.
  // Returns resolving status.
  common::Status Resolve();

  // Getter and Setter for graph name.
  const std::string& Name() const noexcept;
  void SetName(const std::string& name);

  const std::string& Description() const noexcept;
  void SetDescription(const std::string& description);

  // Add/Remove/Get initial tensors for some graph inputs.
  void AddInitializedTensor(const ONNX_NAMESPACE::TensorProto& tensor_proto);
  void RemoveInitializedTensor(const std::string& tensor_name);
  bool GetInitializedTensor(const std::string& tensor_name, const ONNX_NAMESPACE::TensorProto*& value) const;
  const InitializedTensorSet& GetAllInitializedTensors() const noexcept;
  void CleanAllInitializedTensors() noexcept;

  // Graph inputs excluding initializers. Contains no nullptr values.
  const std::vector<const NodeArg*>& GetInputs() const noexcept { return graph_inputs_excluding_initializers_; }

  // Graph inputs including initializers. Contains no nullptr values.
  // This will match the number and order of inputs from the GraphProto.
  const std::vector<const NodeArg*>& GetInputsIncludingInitializers() const noexcept {
    return graph_inputs_including_initializers_;
  }

  // Graph outputs. Should have no nullptr values.
  const std::vector<const NodeArg*>& GetOutputs() const noexcept { return graph_outputs_; }

  bool IsNodeOutputsInGraphOutputs(const Node& node) {
    for (auto output_def : node.OutputDefs()) {
      if (std::find(GetOutputs().cbegin(), GetOutputs().cend(), output_def) != GetOutputs().cend()) {
        return true;
      }
    }
    return false;
  }
  // Get graph value infos.
  const std::vector<const NodeArg*>& GetValueInfo() const noexcept;

  // Get const Node given specific node index. May return nullptr if node as been freed.
  const Node* GetNode(NodeIndex node_index) const { return NodeAtIndexImpl(node_index); }

  // Mutable node at index. May return nullptr if node has been freed.
  Node* GetNode(NodeIndex node_index) { return NodeAtIndexImpl(node_index); }

  GraphNodes& Nodes() noexcept { return iterable_nodes_; }

  const GraphNodes& Nodes() const noexcept { return iterable_nodes_; }

  // Max NodeIndex in the Graph
  int MaxNodeIndex() const noexcept { return gsl::narrow_cast<int>(nodes_.size()); }

  // Number of nodes in the <Graph>.
  // This is smaller than MaxNodeIndex(), since there may be nodes
  // removed during optimization.
  int NumberOfNodes() const noexcept { return num_of_nodes_; }

  NodeArg* GetNodeArg(const std::string& name) {
    auto iter = node_args_.find(name);
    if (iter != node_args_.end()) {
      return iter->second.get();
    }
    return nullptr;
  }

  const NodeArg* GetNodeArg(const std::string& name) const {
    auto iter = node_args_.find(name);
    if (iter != node_args_.end()) {
      return iter->second.get();
    }
    return nullptr;
  }

  // Get NodeArg by name, or create NodeArg owned by the graph if not found
  NodeArg& GetOrCreateNodeArg(const std::string& name, const ONNX_NAMESPACE::TypeProto* p_arg_type) {
    auto iter = node_args_.find(name);
    if (iter != node_args_.end()) {
      return *(iter->second);
    }

    auto result = node_args_.insert(std::make_pair(name, std::make_unique<NodeArg>(name, p_arg_type)));
    return *(result.first->second);
  }

  // create a unique name for NodeArg
  std::string GenerateNodeArgName(const std::string& base_name);

  // create a unique name for Node
  std::string GenerateNodeName(const std::string& base_name);

  // Add node to <*this> graph.
  Node* AddNode(const std::string& name,
                const std::string& op_type,
                const std::string& description,
                const std::vector<NodeArg*>& input_args,
                const std::vector<NodeArg*>& output_args,
                const NodeAttributes* attributes = nullptr,
                const std::string& domain = "");

  // Copy node and add to graph.
  // @param other Node to copy
  // @param returns Pointer to node that was created and inserted.
  Node* AddNode(const Node& other);

  // Remove node and free it.
  bool RemoveNode(NodeIndex node_index);

  // Add|Remove an edge.
  void AddEdge(NodeIndex src_node_index, NodeIndex dst_node_index, const NodeArg& node_arg);
  void RemoveEdge(NodeIndex src_node_index, NodeIndex dst_node_index, const NodeArg& node_arg);

  // Add control edge into <*this> graph.
  // The <dst_node_index> node does not consume any data output by
  // <src_node_index>, but it's designed to be executed behind.
  bool AddControlEdge(NodeIndex src_node_index, NodeIndex dst_node_index);

  // Mark Graph as needing Resolve() to be called
  Graph& SetGraphResolveNeeded() noexcept {
    graph_resolve_needed_ = true;
    return *this;
  }

  bool GraphResolveNeeded() const noexcept {
    return graph_resolve_needed_;
  }

  Graph& SetGraphProtoSyncNeeded() noexcept {
    graph_proto_sync_needed_ = true;
    return *this;
  }

  bool GraphProtoSyncNeeded() const noexcept {
    return graph_proto_sync_needed_;
  }

  // Performs reverse DFS traversal from a set of nodes in 'from' up to
  // the SOURCE node. 'enter' is a visit function that will be invoked
  // on a node when it is visited but its parents haven't been. 'leave'
  // is the visit function invoked on the node after its parents have
  // all been visited. 'comp' is used to stable the traversal order.
  void ReverseDFSFrom(const std::vector<NodeIndex>& from,
                      const std::function<void(const Node*)>& enter,
                      const std::function<void(const Node*)>& leave,
                      const std::function<bool(const Node*, const Node*)>& comp = {}) const;

  void ReverseDFSFrom(const std::vector<const Node*>& from,
                      const std::function<void(const Node*)>& enter,
                      const std::function<void(const Node*)>& leave,
                      const std::function<bool(const Node*, const Node*)>& comp = {}) const;

  const std::unordered_map<std::string, int>& DomainToVersionMap() const noexcept {
    return domain_to_version_;
  }

  // Serialize the <Graph> into <GraphProto>.
  const ONNX_NAMESPACE::GraphProto& ToGraphProto();

  IOnnxRuntimeOpSchemaCollectionPtr GetSchemaRegistry() const;

  Node* FuseSubGraph(std::unique_ptr<::onnxruntime::IndexedSubGraph> sub_graph, const std::string& fused_node_name);

  Status InlineFunction(Node& node);

  // when creating a subgraph, record that a NodeArg will come from the outer scope.
  // This prevents it from being added to the graph inputs.
  void AddOuterScopeNodeArg(const std::string& name) {
    ONNXRUNTIME_IGNORE_RETURN_VALUE(outer_scope_node_arg_names_.insert(name));
  }

  // when constructing a Graph, explicitly set the input order to be used.
  // If the Graph is loaded from a GraphProto this has no effect.
  void SetInputOrder(const std::vector<const NodeArg*> inputs) {
    graph_input_order_ = inputs;
  }

  // when constructing a Graph, explicitly set the input order to be used.
  // If the Graph is loaded from a GraphProto this has no effect.
  void SetOutputOrder(const std::vector<const NodeArg*> outputs) {
    graph_output_order_ = outputs;
  }

  // Construct a Graph instance for a subgraph. Inherits some properties from the parent graph.
  Graph(Graph& parent_graph, ONNX_NAMESPACE::GraphProto& subgraph_proto);

  virtual ~Graph();

 private:
  ONNXRUNTIME_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(Graph);

  // This friendship relationship should only be used to call Graph::Graph and
  // Graph::LoadGraph All other access should be via the public API.
  friend class Model;

  Graph() = delete;

  // Constructor: Given a <GraphProto> loaded from model file, construct
  // a <Graph> object.
  Graph(ONNX_NAMESPACE::GraphProto* graph_proto,
        const std::unordered_map<std::string, int>& domain_to_version,
        Version ir_version,
        IOnnxRuntimeOpSchemaCollectionPtr schema_registry,
        const std::unordered_map<std::string, const ONNX_NAMESPACE::FunctionProto*>& model_functions = {});

  // internal use only
  Graph(ONNX_NAMESPACE::GraphProto* graph_proto,
        const std::unordered_map<std::string, int>& domain_to_version,
        Version ir_version,
        IOnnxRuntimeOpSchemaCollectionPtr schema_registry,
        Graph* parent_graph,
        const std::unordered_map<std::string, const ONNX_NAMESPACE::FunctionProto*>& model_functions = {});

  // Add node with specified <node_proto>.
  Node* AddNode(const ONNX_NAMESPACE::NodeProto& node_proto,
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

  // During the Resolve of a Graph it is necessary to recursively descend into subgraphs if present.
  // The ResolveContext holds the collection of values for the current Graph instance, be it the main graph
  // or a subgraph, so that the various operations that are part of the Resolve can work iteratively or
  // recursively as needed.
  struct ResolveContext {
    ResolveContext() = default;

    std::unordered_map<std::string, Node*> output_args;
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
    ONNXRUNTIME_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ResolveContext);
  };

  // search this and up through any parent_graph_ instance for a NodeArg
  const NodeArg* GetNodeArgIncludingParentGraphs(const std::string& node_arg_name) const;

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
    // use ONNXRUNTIME_ENFORCE to prove that or uncover scenarios where we actually
    // expect attempts to retrieve a non-existent node.
    ONNXRUNTIME_ENFORCE(node_index < nodes_.size(), "Validating no unexpected access using an invalid node_index.");
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
