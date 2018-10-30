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
#include "core/graph/onnx_protobuf.h"
#include "gsl/gsl_util"
#include "gsl/pointers"

#include "core/framework/func_api.h"

namespace onnxruntime {
class Function;
}  // namespace onnxruntime

namespace onnxruntime {

class Graph;
class GraphBase;
class Node;
class OpSignature;

// Node argument definition, for both input and output,
// including arg name, arg type (contains both type and shape).
//
// Design Question: in my (Ke's) opinion, shape should not be part of type.
// We may align the protobuf design with our operator registry interface,
// which has type specified for each operator, but no shape. Well, shape
// should be inferred with a separate shape inference function given
// input shapes, or input tensor data sometimes.
// With shape as part of type (current protobuf design),
// 1) we'll have to split the "TypeProto" into type and shape in this internal
// representation interface so that it could be easily used when doing type
// inference and matching with operator registry.
// 2) SetType should be always called before SetShape, otherwise, SetShape()
// will fail. Because shape is located in a TypeProto.
// Thoughts?
//
class NodeArg {
 public:
  // Constructor by specifying node arg name and type&shape which is
  // optional. This is called when loading a <Graph> from <GraphProto>
  // normally.
  NodeArg(const std::string& name,
          const ONNX_NAMESPACE::TypeProto* p_arg_type);

  NodeArg(NodeArg&& other) = default;

  // Get node arg name.
  const std::string& Name() const noexcept;

  // Get node arg type.
  ONNX_NAMESPACE::DataType Type() const noexcept;
  const ONNX_NAMESPACE::TypeProto* TypeAsProto() const noexcept;

  // Get node arg shape.
  // Return null pointer if there's no shape specified.
  const ONNX_NAMESPACE::TensorShapeProto* Shape() const;

  // Set node arg shape.
  // Shape could only be set after setting type since shape information
  // now is part of TypeProto.
  void SetShape(const ONNX_NAMESPACE::TensorShapeProto& shape);

  // Get node arg info proto.
  const NodeArgInfo& ToProto() const noexcept { return node_arg_info_; }

  // Indicates whether <*this> node arg exists or not.
  // Optional inputs are allowed in ONNX. Empty arg name represents
  // a non-existing input argument.
  bool Exists() const noexcept;

 private:
  ONNXRUNTIME_DISALLOW_COPY_AND_ASSIGNMENT(NodeArg);
  friend class Graph;

  void SetType(ONNX_NAMESPACE::DataType p_type);
  void SetType(const ONNX_NAMESPACE::TypeProto& type_proto);

  NodeArg& operator=(NodeArg&& other) = delete;

  // Node arg PType.
  ONNX_NAMESPACE::DataType type_;

  // Node arg name, type and shape.
  NodeArgInfo node_arg_info_;

  // Flag indicates whether <*this> node arg exists or not.
  bool exists_;
};

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
    // TODO: Can these values ever be null?
    EdgeEnd(const Node& node, const NodeArg& node_arg) noexcept;

    // Get the <Node*> that this edge end refers to.
    const Node& GetNode() const noexcept;

    // Get the <NodeArg*> that this edge end refers to.
    const NodeArg& GetNodeArg() const noexcept;

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

  // read only access. requires special wrapper to apply const to the NodeArg
  const ConstPointerContainer<std::vector<NodeArg*>> OutputDefs() const noexcept {
    return ConstPointerContainer<std::vector<NodeArg*>>(definitions_.output_defs);
  }

  std::vector<NodeArg*>& MutableInputDefs() noexcept {
    return MutableDefinitions().input_defs;
  }

  struct IndexCompare {
    bool operator()(const Node* lhs, const Node* rhs) {
      return lhs->Index() < rhs->Index();
    }
  };
  typedef std::set<const Node*, IndexCompare> NodeSet;
  using NodeConstIterator = NodeSet::const_iterator;
  using EdgeConstIterator = std::set<EdgeEnd*>::const_iterator;

  // Functions defined to traverse a Graph as below.
  // Read all input nodes of <*this>.
  // Beginning of input nodes. Iterator should have no nullptr values.
  NodeConstIterator InputNodesBegin() const noexcept { return relationships_.input_nodes.cbegin(); };
  // End of input nodes.
  NodeConstIterator InputNodesEnd() const noexcept { return relationships_.input_nodes.cend(); }

  // Beginning of output nodes. Iterator should have no nullptr values.
  NodeConstIterator OutputNodesBegin() const noexcept { return relationships_.output_nodes.cbegin(); }
  // End of output nodes.
  NodeConstIterator OutputNodesEnd() const noexcept { return relationships_.output_nodes.cend(); }

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
      input_nodes.clear();
      output_nodes.clear();
      control_inputs.clear();
    }

    // Node input edges.
    std::set<EdgeEnd*> input_edges;
    // Node output edges.
    std::set<EdgeEnd*> output_edges;

    // Node input nodes, besides input nodes mentioned in <inputs_> above,
    // it also contains all control input nodes;
    NodeSet input_nodes;
    // Control input nodes' names.
    std::set<std::string> control_inputs;
    // Node's output nodes.
    NodeSet output_nodes;

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
  friend class GraphBase;
  friend class Graph;

  Node(NodeIndex index, GraphBase& graph) : index_(index), graph_(&graph){}

  void Init(const std::string& name,
            const std::string& op_type,
            const std::string& description,
            const std::vector<NodeArg*>& input_args,
            const std::vector<NodeArg*>& output_args,
            const NodeAttributes* attributes,
            const std::string& domain);

  // internal only method to allow selected classes to directly alter
  // the input/output definitions and arg counts
  Definitions& MutableDefinitions() noexcept;

  // internal only method to allow selected classes to directly alter
  // the links between nodes.
  Relationships& MutableRelationships() noexcept;

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

  GraphBase* graph_;
};
#ifdef _MSC_VER
#pragma warning(pop)
#endif

// TODO: Graph base class.
// It should cover the common things between function and graph.
// Move these common things from Graph to GraphBase.
// 1. Graph does not have attributes, while function has.
// 2. Graph does have initializers, while function does not.
// 3. Graph does have value_info, while function does not.
class GraphBase {
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
  virtual common::Status Resolve() = 0;

  // Getter and Setter for graph name.
  virtual const std::string& Name() const noexcept = 0;
  virtual void SetName(const std::string& name) = 0;

  virtual const std::string& Description() const noexcept = 0;
  virtual void SetDescription(const std::string& description) = 0;

  // Graph inputs. Should have no nullptr values.
  const std::vector<const NodeArg*>& GetInputs() const noexcept { return graph_inputs_; }

  // Graph outputs. Should have no nullptr values.
  const std::vector<const NodeArg*>& GetOutputs() const noexcept { return graph_outputs_; }

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

  // Add control edge into <*this> graph.
  // The <dst_node_index> node does not consume any data output by
  // <src_node_index>, but it's designed to be executed behind.
  bool AddControlEdge(NodeIndex src_node_index, NodeIndex dst_node_index);

  bool IsSourceNode(NodeIndex index) const noexcept;
  bool IsSinkNode(NodeIndex index) const noexcept;

  bool IsSourceNode(const Node& node) const noexcept {
    return source_node_index_ == node.Index();
  }

  bool IsSinkNode(const Node& node) const noexcept {
    return sink_node_index_ == node.Index();
  }

  const Node* SourceNode() const;
  const Node* SinkNode() const;

  common::Status GetNodesInTopologicalOrder(/*out*/ const std::vector<NodeIndex>*& pp_nodes) const;

  // Mark Graph as needing Resolve() to be called
  GraphBase& SetGraphResolveNeeded() noexcept {
    graph_resolve_needed_ = true;
    return *this;
  }

  bool GraphResolveNeeded() const noexcept {
    return graph_resolve_needed_;
  }

  GraphBase& SetGraphProtoSyncNeeded() noexcept {
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

  virtual ~GraphBase() = default;

 protected:
  GraphBase() = default;
  GraphBase(bool graph_resolve_needed,
            bool graph_proto_sync_needed,
            const std::unordered_map<std::string, int>& domain_to_version,
            Version ir_version)
      : graph_resolve_needed_(graph_resolve_needed),
        graph_proto_sync_needed_(graph_proto_sync_needed),
        domain_to_version_(domain_to_version),
        ir_version_(ir_version) {}

  // Add source/sink nodes to <*this> graph.
  void AddSourceSinkNodes();

  // Add node with specified <node_proto>.
  Node* AddNode(const ONNX_NAMESPACE::NodeProto& node_proto,
                const ArgNameToTypeMap& name_to_type);

  NodeIndex SourceNodeIndex() const noexcept { return source_node_index_; }

  NodeIndex SinkNodeIndex() const noexcept { return sink_node_index_; }

  // The topological order of node index as last set by Resolve()
  const std::vector<NodeIndex>& NodesInTopologicalOrder() const noexcept {
    return nodes_in_topological_order_;
  }

  std::vector<NodeIndex>& MutableNodesInTopologicalOrder() noexcept {
    return nodes_in_topological_order_;
  }

  // Mutable graph inputs.
  std::vector<const NodeArg*>& MutableInputs() noexcept {
    return graph_inputs_;
  }

  // Mutable graph outputs.
  std::vector<const NodeArg*>& MutableOutputs() noexcept {
    return graph_outputs_;
  }

  const std::unordered_map<std::string, int>& DomainToVersionMap() const noexcept {
    return domain_to_version_;
  }

  Version IrVersion() const noexcept {
    return ir_version_;
  }

  GraphBase& GraphResolveNeeded(bool needed) noexcept {
    graph_resolve_needed_ = needed;
    return *this;
  }

  GraphBase& GraphProtoSyncNeeded(bool needed) noexcept {
    graph_proto_sync_needed_ = needed;
    return *this;
  }

  // Build and verify node connection (edges).
  // Verify NodeArg name/type/shape matching correctly.
  common::Status BuildConnections(
      const std::unordered_map<std::string, Node*>& output_args,
      const std::unordered_map<std::string, NodeIndex>& node_name_to_index);

  common::Status VerifyNoDuplicateName(
      /*in*/ const std::unordered_set<std::string>& inputs_and_initializers,
      /*out*/ std::unordered_map<std::string, Node*>& output_args,
      /*out*/ std::unordered_map<std::string, NodeIndex>& node_name_to_index);

  // Check whether <*this> graph is acyclic while performing a topological sort.
  // Depth-first going from bottom up through the graph and checking whether there are any back edges.
  // NodesInTopologicalOrder is updated with the nodes' indexes in topological
  // order if <Status> returned is "OK", otherwise it's undefined.
  common::Status PerformTopologicalSortAndCheckIsAcyclic();

  // Apply shape/type inference to a single node. This is a wrapper for
  // invoking ONNX-defined shape+type inference for a single node.
  // Returns the inferred shape+type for every output of the node in
  // output parameter inferredShapes.
  common::Status InferOutputTypesAndShapes(onnxruntime::Node& node,
                                           /*out*/ std::vector<ONNX_NAMESPACE::TypeProto>& inferred_shapes);

 private:
  // need custom versions to handle the unique_ptr's in nodes_
  ONNXRUNTIME_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GraphBase);

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

 protected:
  // default to impossible value and not 0
  NodeIndex source_node_index_ = std::numeric_limits<NodeIndex>::max();
  NodeIndex sink_node_index_ = std::numeric_limits<NodeIndex>::max();

 private:
  // A flag indicates whether <*this> graph needs to be resolved.
  bool graph_resolve_needed_ = false;

  bool graph_proto_sync_needed_ = false;

  // The topological order of node index.
  std::vector<NodeIndex> nodes_in_topological_order_;

  // Graph inputs.
  std::vector<const NodeArg*> graph_inputs_;

  // Graph outputs.
  std::vector<const NodeArg*> graph_outputs_;

  // All node args owned by <*this> graph. Key is node arg name.
  std::unordered_map<std::string, std::unique_ptr<NodeArg>> node_args_;

  // Node::EdgeEnd instances that we own
  std::vector<std::unique_ptr<Node::EdgeEnd>> owned_edges_;

  const std::unordered_map<std::string, int> domain_to_version_;

  // Model IR version.
  Version ir_version_{};

  int name_generator_ = 0;
};

}  // namespace onnxruntime
