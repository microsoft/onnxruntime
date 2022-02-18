// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once


#include "core/common/status.h"
#include "core/common/logging/logging.h"

#include "core/common/common.h"
//#include "core/common/path.h"

#include "core/common/const_pointer_container.h"

#include "core/graph/basic_types.h"
#include "core/graph/node_arg.h"


#if !defined(ORT_MINIMAL_BUILD)
#include "onnx/defs/schema.h"
#else
#include "onnx/defs/data_type_utils.h"
#endif
#include "onnx/onnx_pb.h"
#include "onnx/onnx-operators_pb.h"

namespace flatbuffers {
class FlatBufferBuilder;
template <typename T>
struct Offset;
}  // namespace flatbuffers

namespace onnxruntime {
class Graph;

namespace fbs {
struct Graph;
struct Node;
struct NodeEdge;
}  // namespace fbs

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
    const Node& GetNode() const noexcept { return *node_; }

    /** Gets the source arg index.
    @returns the source arg index of <*this> edge.*/
    int GetSrcArgIndex() const { return src_arg_index_; }

    /** Gets the destination arg index.
    @returns the destination arg index of <*this> edge.*/
    int GetDstArgIndex() const { return dst_arg_index_; }

   private:
    const Node* node_;
    const int src_arg_index_;
    const int dst_arg_index_;
  };

  /** Gets the Node's NodeIndex. */
  NodeIndex Index() const noexcept { return index_; }

  /** Gets the Node's name. */
  const std::string& Name() const noexcept { return name_; }

  /** Gets the Node's operator type. */
  const std::string& OpType() const noexcept { return op_type_; }

  /** Gets the domain of the OperatorSet that specifies the operator returned by #OpType. */
  const std::string& Domain() const noexcept { return domain_; }

  // TODO!!!!
  // Why node ndeed model path?
  /** Gets the path of the owning model if any. */
  /*const Path& ModelPath() const noexcept;*/

  /** Gets the Node's execution priority.
  @remarks Lower value means higher priority  */
  int Priority() const noexcept { return priority_; };

  /** Sets the execution priority of a node.
  @remarks Lower value means higher priority  */
  void SetPriority(int priority) noexcept;

  /** Gets the node description. */
  const std::string& Description() const noexcept { return description_; }

  /** Gets the Node's Node::Type. */
  Node::Type NodeType() const noexcept { return node_type_; }

  /** Gets the opset version that the Node's operator was first defined in.
  @returns Opset version. If -1 the Node's operator has not been set.
  @remarks Prefer over Op()->SinceVersion() as Op() is disabled in a minimal build
  */
  int SinceVersion() const noexcept { return since_version_; }

#if !defined(ORT_MINIMAL_BUILD)
  //TODO!!!!!
  //Should we move this out of node?
  /** Gets the Node's OpSchema.
  @remarks The graph containing this node must be resolved, otherwise nullptr will be returned. */
  const ONNX_NAMESPACE::OpSchema* Op() const noexcept { return op_; }

  //TODO!!!!
  //Re-implement Function
  /**
  Gets the function body if applicable otherwise nullptr
  @param try_init_func_body If not already initialized, initialize the function body
  (This is not applicable for primitive operators.)
  Function body can be initialized in 3 cases :
  1. For nodes of type "Fused"
  2. For nodes which are defined as functions in the spec (example: DynamicQuantizeLinear)
  3. For nodes which reference a model local function. These functions are defined in the model itself and
  do not have any schema associated with them.
  For all other cases this will always return nullptr.
  Nodes of type "Fused" are created during partitioning and the function body
  initialization for such nodes also happens during node creation. Therefore,
  initialization of function body will happen via this method only in cases 2 and 3 mentioned above.
  */
  //Function* GetMutableFunctionBody(bool try_init_func_body = true);

  /** Gets the function body if applicable otherwise nullptr. */
  //const Function* GetFunctionBody() const noexcept { return func_body_; }

#endif

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

  /** Gets the count of arguments for each of the Node's explicit inputs. */
  const std::vector<int>& InputArgCount() const noexcept { return definitions_.input_arg_count; }

  /** Gets the Node's input definitions.
  @remarks requires ConstPointerContainer wrapper to apply const to the NodeArg pointers so access is read-only. */
  ConstPointerContainer<std::vector<NodeArg*>> InputDefs() const noexcept {
    return ConstPointerContainer<std::vector<NodeArg*>>(definitions_.input_defs);
  }

  /** Gets the implicit inputs to this Node.
  If this Node contains a subgraph, these are the NodeArg's that are implicitly consumed by Nodes within that
  subgraph. e.g. If and Loop operators.*/
  ConstPointerContainer<std::vector<NodeArg*>> ImplicitInputDefs() const noexcept {
    return ConstPointerContainer<std::vector<NodeArg*>>(definitions_.implicit_input_defs);
  }

  /** Gets the Node's output definitions.
  @remarks requires ConstPointerContainer wrapper to apply const to the NodeArg pointers so access is read-only. */
  ConstPointerContainer<std::vector<NodeArg*>> OutputDefs() const noexcept {
    return ConstPointerContainer<std::vector<NodeArg*>>(definitions_.output_defs);
  }

#if !defined(ORT_MINIMAL_BUILD)
  /**
  Helper to iterate through the container returned by #MutableInputDefs() or #MutableOutputDefs() and call the provided function.
  @param node_args Collection of NodeArgs returned by #MutableInputDefs() or #MutableOutputDefs()
  @param func Function to call for each valid NodeArg in the node_args. The function is called with the NodeArg
              and the index number in the container.
  @returns common::Status with success or error information.
  @remarks Returns immediately on error.
  */
  static common::Status ForEachMutableWithIndex(std::vector<NodeArg*>& node_args,
                                                std::function<common::Status(NodeArg& arg, size_t index)> func) {
    for (size_t index = 0; index < node_args.size(); ++index) {
      auto arg = node_args[index];
      if (!arg->Exists())
        continue;
      ORT_RETURN_IF_ERROR(func(*arg, index));
    }
    return common::Status::OK();
  }

  /** Gets a modifiable collection of the Node's implicit input definitions. */
  std::vector<NodeArg*>& MutableImplicitInputDefs() noexcept {
    return definitions_.implicit_input_defs;
  }
#endif  // !defined(ORT_MINIMAL_BUILD)

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
  /** Gets a modifiable count of arguments for each of the Node's explicit inputs.
  @todo This should be removed in favor of a method that updates the input args and the count.
        Currently these operations are separate which is not a good setup. */
  std::vector<int>& MutableInputArgsCount() { return definitions_.input_arg_count; }

  /** Gets a modifiable collection of the Node's input definitions. */
  std::vector<NodeArg*>& MutableInputDefs() noexcept {
    return definitions_.input_defs;
  }

  /** Gets a modifiable collection of the Node's output definitions. */
  std::vector<NodeArg*>& MutableOutputDefs() noexcept {
    return definitions_.output_defs;
  }
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

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

    const Node& operator*() const;
    const Node* operator->() const;

   private:
    EdgeConstIterator m_iter;
  };

  // Functions defined to traverse a Graph as below.

  /** Gets an iterator to the beginning of the input nodes to this Node. */
  NodeConstIterator InputNodesBegin() const noexcept { return NodeConstIterator(relationships_.input_edges.cbegin()); };
  /** Gets an iterator to the end of the input nodes to this Node. */
  NodeConstIterator InputNodesEnd() const noexcept { return NodeConstIterator(relationships_.input_edges.cend()); }

  /** Gets an iterator to the beginning of the output nodes from this Node. */
  NodeConstIterator OutputNodesBegin() const noexcept {
    return NodeConstIterator(relationships_.output_edges.cbegin());
  }

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
#if !defined(DISABLE_SPARSE_TENSORS)
  ADD_ATTR_INTERFACES(ONNX_NAMESPACE::SparseTensorProto)
#endif
  ADD_ATTR_INTERFACES(ONNX_NAMESPACE::TypeProto)

  /** Gets the Node's attributes. */
  const NodeAttributes& GetAttributes() const noexcept { return attributes_; }

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_ENABLE_RUNTIME_OPTIMIZATION_IN_MINIMAL_BUILD)
  /** Remove the specified attribute from this Node */
  bool ClearAttribute(const std::string& attr_name);
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_ENABLE_RUNTIME_OPTIMIZATION_IN_MINIMAL_BUILD)

#if !defined(ORT_MINIMAL_BUILD)
  /** Gets the Node's mutable attributes. */
  NodeAttributes& GetMutableAttributes() noexcept { return attributes_; }

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
#endif  // !defined(ORT_MINIMAL_BUILD)

  /** Checks if the Node contains at least one subgraph (this is the case for control flow operators, such as If, Scan, Loop).
  @returns true if the Node contains a subgraph.
  */
  bool ContainsSubgraph() const {
    return !attr_to_subgraph_map_.empty();
  }

  /** Get the const subgraphs from a node.
  @remarks Creates a new vector so calling ContainsSubgraphs first is preferred. */
  std::vector<gsl::not_null<const Graph*>> GetSubgraphs() const;

  /** Gets a map of attribute name to the mutable Graph instances for all subgraphs of the Node.
  @returns Map of the attribute name that defines the subgraph to the subgraph's Graph instance.
           nullptr if the Node has no subgraphs.
  */
  const std::unordered_map<std::string, gsl::not_null<Graph*>>& GetAttributeNameToMutableSubgraphMap() {
    return attr_to_subgraph_map_;
  }

  /** Gets a map of attribute name to the const Graph instances for all subgraphs of the Node.
  @returns Map of the attribute name that defines the subgraph to the subgraph's Graph instance.
           nullptr if the Node has no subgraphs.
  */
  std::unordered_map<std::string, gsl::not_null<const Graph*>> GetAttributeNameToSubgraphMap() const;

  /** Gets the execution ProviderType that this node will be executed by. */
  ProviderType GetExecutionProviderType() const noexcept { return execution_provider_type_; }

  //TODO:!!!!
  // should this be move out of node?
  /** Sets the execution ProviderType that this Node will be executed by. */
  void SetExecutionProviderType(ProviderType execution_provider_type) {
    execution_provider_type_ = execution_provider_type;
  }

  //TODO!!!!!!!
  //re-implment function
  /** Sets initialized function body for node. This is called right after function body initialization for a node.
   * or during function inlining when a nested function is encountered.
   */
  //void SetFunctionBody(Function& func);

  /** Call the provided function for all explicit inputs, implicit inputs, and outputs of this Node.
      If the NodeArg is an explicit or implicit input, is_input will be true when func is called.
      @param include_missing_optional_defs Include NodeArgs that are optional and were not provided
                                           i.e. NodeArg::Exists() == false.
      */
  void ForEachDef(std::function<void(const onnxruntime::NodeArg&, bool is_input)> func,
                  bool include_missing_optional_defs = false) const;

#if !defined(ORT_MINIMAL_BUILD)
  /** Replaces any matching definitions in the Node's explicit inputs or explicit outputs.
  @param replacements Map of current NodeArg to replacement NodeArg.
  */
  void ReplaceDefs(const std::map<const onnxruntime::NodeArg*, onnxruntime::NodeArg*>& replacements);

  //TODO!!!!!
  //Move it to GraphProtoSerializer
  /** Gets the NodeProto representation of this Node.
  @param update_subgraphs Update the GraphProto values for any subgraphs in the returned NodeProto.
                          If graph optimization has been run this is most likely required
                          to ensure the complete Graph is valid.
  */
  void ToProto(ONNX_NAMESPACE::NodeProto& proto, bool update_subgraphs = false) const;

  //TODO!!!!!!!!!!!!!!
  //Move it to ORTFormatSaver
  Status SaveToOrtFormat(flatbuffers::FlatBufferBuilder& builder,
                         flatbuffers::Offset<onnxruntime::fbs::Node>& fbs_node) const;

  flatbuffers::Offset<onnxruntime::fbs::NodeEdge>
  SaveEdgesToOrtFormat(flatbuffers::FlatBufferBuilder& builder) const;

#endif

  //TODO!!!!!
  //Move it to ORTFormatLoder
  static Status LoadFromOrtFormat(const onnxruntime::fbs::Node& fbs_node, Graph& graph,
                                  const logging::Logger& logger, std::unique_ptr<Node>& node);

  Status LoadFromOrtFormat(const onnxruntime::fbs::Node& fbs_node, const logging::Logger& logger);
  Status LoadEdgesFromOrtFormat(const onnxruntime::fbs::NodeEdge& fbs_node_edgs, const Graph& graph);

  /**
  @class Definitions
  The input and output definitions for this Node.
  */
  class Definitions {
   public:
    Definitions() = default;

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

  // NOTE: This friendship relationship should ONLY be used for calling methods of the Node class and not accessing
  // the data members directly, so that the Node can maintain its internal invariants.
  //TODO!!!!!
  //Refine the API
  friend class Graph;
  friend class FunctionIR;
  Node(NodeIndex index, Graph& graph) : index_(index), graph_(&graph) {}

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(Node);

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
  void Init(const std::string& name,
            const std::string& op_type,
            const std::string& description,
            const std::vector<NodeArg*>& input_args,
            const std::vector<NodeArg*>& output_args,
            const NodeAttributes* attributes,
            const std::string& domain);

  // internal only method to allow selected classes to directly alter the input/output definitions and arg counts
  Definitions& MutableDefinitions() noexcept;

  // internal only method to allow selected classes to directly alter the links between nodes.
  Relationships& MutableRelationships() noexcept;

  void SetNodeType(Node::Type node_type) noexcept { node_type_ = node_type; }
#endif

  // create a Graph instance for an attribute that contains a GraphProto
  void CreateSubgraph(const std::string& attr_name);

  const std::vector<std::unique_ptr<Graph>>& MutableSubgraphs() noexcept { return subgraphs_; }

  // validate and update the input arg count
  common::Status UpdateInputArgCount();

  const Definitions& GetDefinitions() const noexcept { return definitions_; }
  const Relationships& GetRelationships() const noexcept { return relationships_; }

  // Node index. Default to impossible value rather than 0.
  NodeIndex index_ = std::numeric_limits<NodeIndex>::max();

  // Node name.
  std::string name_;

  // Node operator type.
  std::string op_type_;

  // OperatorSet domain of op_type_.
  std::string domain_;

#if !defined(ORT_MINIMAL_BUILD)
  // OperatorSchema that <*this> node refers to.
  const ONNX_NAMESPACE::OpSchema* op_ = nullptr;
#endif

  // Execution priority, lower value for higher priority
  int priority_ = 0;

  // set from op_->SinceVersion() or via deserialization when OpSchema is not available
  int since_version_ = -1;

  Node::Type node_type_ = Node::Type::Primitive;

  // The function body is owned by graph_
  // TODO: re-implement function
  // Function* func_body_ = nullptr;

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
}  // namespace onnxruntime
