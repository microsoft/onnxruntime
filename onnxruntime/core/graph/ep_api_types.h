// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <gsl/gsl>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/common/inlined_containers.h"
#include "core/framework/allocator.h"
#include "core/graph/basic_types.h"
#include "core/graph/abi_graph_types.h"
#include "core/graph/graph_viewer.h"

namespace onnxruntime {
struct EpGraph;

/// <summary>
/// Concrete implementation of OrtValueInfo used in the OrtEpApi.
/// </summary>
struct EpValueInfo : public OrtValueInfo {
 public:
  enum Flags {
    kFlagNone = 0,
    kIsRequiredGraphInput = 1 << 0,
    kIsOptionalGraphInput = 1 << 1,
    kIsGraphOutput = 1 << 2,
    kIsConstantInitializer = 1 << 3,
    kIsOuterScope = 1 << 4,
  };

  EpValueInfo(const EpGraph* graph, const std::string& name, std::unique_ptr<OrtTypeInfo>&& type_info,
              size_t flags);

  // Defines ToExternal() and ToInternal() functions to convert between OrtValueInfo and EpValueInfo.
  DEFINE_ORT_GRAPH_IR_TO_EXTERNAL_INTERNAL_FUNCS(OrtValueInfo, EpValueInfo, OrtGraphIrApi::kEpApi)

  //
  // Publicly accessible overrides defined by OrtValueInfo.
  //

  // Returns the value's name in the graph.
  const std::string& GetName() const override { return name_; }

  // Returns the value's type information, which includes both type and shape.
  const OrtTypeInfo* GetTypeInfo() const override { return type_info_.get(); }

  // Gets the information (OrtNode and output index) about the node that produces this value.
  Status GetProducerInfo(OrtValueInfo::ProducerInfo& producer_info) const override;

  // Gets information (OrtNode and input index) about the consumer nodes that use this value as an input.
  // An OrtNode instance may appear multiple times if it uses the value as an input more than once (e.g., Mul(x, x)).
  // The input index is set to -1 if the consumer node uses the value as an "implicit input".
  Status GetConsumerInfos(std::vector<OrtValueInfo::ConsumerInfo>& consumer_infos) const override;

  // Gets the number of ConsumerInfo instances that will be returned by GetConsumerInfos.
  Status GetNumConsumerInfos(size_t& num_consumers) const override;

  // Gets the initializer OrtValue associated with this OrtValueInfo. Returns nullptr if this does not
  // represent an initializer (either constant or non-constant).
  Status GetInitializerValue(const OrtValue*& value) const override;

  // Gets external initializer information (file path, file offset, byte size) if this is an external initializer.
  // Otherwise, sets the output parameter `ext_info` to nullptr (not an error).
  Status GetExternalInitializerInfo(std::unique_ptr<onnxruntime::ExternalDataInfo>& ext_info) const override;

  // Check if this value is a required graph input.
  Status IsRequiredGraphInput(bool& is_required_graph_input) const override;

  // Check if this value is an optional graph input.
  Status IsOptionalGraphInput(bool& is_optional_graph_input) const override;

  // Check if this value is a graph output.
  Status IsGraphOutput(bool& is_graph_output) const override;

  // Check if this value is a constant initializer.
  Status IsConstantInitializer(bool& is_const_initializer) const override;

  // Check if this value is defined in an outer scope (i.e., an outer graph).
  Status IsFromOuterScope(bool& is_outer_scope) const override;

  //
  // Helper functions used when working directly with an EpValueInfo.
  //

  // Helper to set a flag.
  void SetFlag(EpValueInfo::Flags flag) { flags_ |= flag; }

  // Helper to check if a flag is set.
  bool IsFlagSet(EpValueInfo::Flags flag) const { return flags_ & flag; }

 private:
  // Back pointer to parent graph. If not null, enables retrieval of consumer and producer nodes.
  // Is null if the EpValueInfo was created without an owning EpGraph
  // (e.g., OrtValueInfo instances created for fused nodes in OrtEp::Compile()).
  const EpGraph* graph_ = nullptr;
  std::string name_;
  std::unique_ptr<OrtTypeInfo> type_info_;
  size_t flags_ = 0;
};

/// <summary>
/// Concrete implementation of OrtNode used in the OrtEpApi.
/// </summary>
struct EpNode : public OrtNode {
 private:
  struct PrivateTag {};  // Used to prevent use of public constructor (use static EpNode::Create())
                         // Need to make the constructor public for std::make_unique().

  struct SubgraphState {
    SubgraphState() = default;
    SubgraphState(SubgraphState&& other) = default;
    std::string attribute_name;
    std::unique_ptr<GraphViewer> subgraph_viewer;  // The graph_viewer wrapped by EpGraph below.
    std::unique_ptr<EpGraph> ep_subgraph;
  };

 public:
  EpNode(const EpGraph* ep_graph, const Node& node, PrivateTag);

  /// <summary>
  /// Creates an instance of EpNode, which wraps an onnxruntime::Node.
  /// </summary>
  /// <param name="node">The actual node to wrap.</param>
  /// <param name="ep_graph">Optional pointer to the parent graph. Set this to a valid graph to be able to get
  ///                        neighboring nodes from this node's input and output OrtValueInfo instances.</param>
  /// <param name="value_infos">Cache of all OrtValueInfo instances in the graph. Can be set to an empty
  ///                           std::unordered_map if creating a node without a graph.</param>
  /// <param name="result">The new EpNode instance.</param>
  /// <returns>A Status indicating success or an error.</returns>
  static Status Create(const Node& node, const EpGraph* ep_graph,
                       std::unordered_map<std::string, std::unique_ptr<EpValueInfo>>& value_infos,
                       /*out*/ std::unique_ptr<EpNode>& result);

  // Defines ToExternal() and ToInternal() functions to convert between OrtNode and EpNode.
  DEFINE_ORT_GRAPH_IR_TO_EXTERNAL_INTERNAL_FUNCS(OrtNode, EpNode, OrtGraphIrApi::kEpApi)

  //
  // Publicly accessible overrides defined by OrtNode.
  //

  // Returns the node's ID (i.e., NodeIndex).
  size_t GetId() const override;

  // Returns the node's name.
  const std::string& GetName() const override;

  // Returns the node's operator type (e.g., "Conv").
  const std::string& GetOpType() const override;

  // Returns the node's domain name.
  const std::string& GetDomain() const override;

  // Gets the opset version in which this node's operator was first defined.
  Status GetSinceVersion(int& since_version) const override;

  // Get the number of node inputs.
  size_t GetNumInputs() const override;

  // Gets the node's inputs.
  Status GetInputs(gsl::span<const OrtValueInfo*> inputs) const override;

  // Get the number of node outputs.
  size_t GetNumOutputs() const override;

  // Gets the node's outputs.
  Status GetOutputs(gsl::span<const OrtValueInfo*> outputs) const override;

  // Gets the number of implicit inputs.
  Status GetNumImplicitInputs(size_t& num_implicit_inputs) const override;

  // Gets the node's implicit inputs.
  Status GetImplicitInputs(gsl::span<const OrtValueInfo*> inputs) const override;

  // Get the number of node attributes.
  size_t GetNumAttributes() const override;

  // Gets the node's attributes.
  Status GetAttributes(gsl::span<const OrtOpAttr*> attrs) const override;

  Status GetTensorAttributeAsOrtValue(const OrtOpAttr* attribute,
                                      const OrtValue*& attr_tensor) override;

  // Gets the number of subgraphs contained by this node.
  Status GetNumSubgraphs(size_t& num_subgraphs) const override;

  // Gets the subgraphs contained by this node.
  Status GetSubgraphs(gsl::span<const OrtGraph*> subgraphs,
                      const char** opt_attribute_names) const override;

  // Gets this node's parent graph, which is the graph that directly contains this node.
  Status GetGraph(const OrtGraph*& parent_graph) const override;

  //
  // Helper functions used when working directly with an EpNode.
  //

  // Returns the internal onnxruntime::Node& that this OrtNode wraps.
  const Node& GetInternalNode() const { return node_; }

  // Helper that returns this node's inputs as a span of EpValueInfo pointers.
  gsl::span<const EpValueInfo* const> GetInputsSpan() const;

  // Helper that returns this node's implicit inputs as a span of EpValueInfo pointers.
  gsl::span<const EpValueInfo* const> GetImplicitInputsSpan() const;

  // Helper that returns this node's outputs as a span of EpValueInfo pointers.
  gsl::span<const EpValueInfo* const> GetOutputsSpan() const;

  // Helper that gets the node's attributes by name.
  const OrtOpAttr* GetAttribute(const std::string& name) const;

  // Helper that gets the execution provider name that this node is assigned to run on.
  const std::string& GetEpName() const;

  // Helper to get the unique name for the 'TENSOR' attribute. Returns empty string if
  // attribute is not 'TENSOR' type.
  const std::string GetUniqueTensorAttributeName(const OrtOpAttr* attr) const;

 private:
  // Back pointer to containing graph. Useful when traversing through nested subgraphs.
  // Will be nullptr if the EpNode was created without an owning graph.
  // (e.g., OrtNode instances created for fused nodes in OrtEp::Compile()).
  const EpGraph* ep_graph_ = nullptr;
  const Node& node_;

  InlinedVector<EpValueInfo*> inputs_;
  InlinedVector<EpValueInfo*> outputs_;

  std::unordered_map<std::string, std::unique_ptr<ONNX_NAMESPACE::AttributeProto>> attributes_map_;
  std::vector<OrtOpAttr*> attributes_;
  std::unordered_map<std::string, std::unique_ptr<OrtValue>> tensor_attribute_values_;  // The 'TENSOR' Attribute as an OrtValue

  std::vector<EpValueInfo*> implicit_inputs_;
  std::vector<SubgraphState> subgraphs_;
};

/// <summary>
/// Concrete implementation of OrtGraph used in the OrtEpApi.
/// </summary>
struct EpGraph : public OrtGraph {
 private:
  struct PrivateTag {};  // Used to prevent use of public constructor (use static EpGraph::Create())
                         // Need to make the constructor public for std::make_unique().

  // Class that maps a NodeIndex to an EpNode* using a std::vector.
  // This is used a lot and should be more efficient than using an unordered_map.
  struct IndexToEpNodeMap {
   public:
    IndexToEpNodeMap() = default;
    IndexToEpNodeMap(IndexToEpNodeMap&& other) = default;
    IndexToEpNodeMap& operator=(IndexToEpNodeMap&& other) = default;
    void Resize(NodeIndex min_node_index, NodeIndex max_node_index);
    EpNode* GetEpNode(NodeIndex node_index) const;
    void SetEpNode(NodeIndex node_index, EpNode* ep_node);

   private:
    NodeIndex min_node_index_ = 0;
    std::vector<EpNode*> nodes_;
  };

 public:
  EpGraph(const GraphViewer& graph_viewer, PrivateTag);
  EpGraph(std::unique_ptr<GraphViewer> graph_viewer,
          std::unique_ptr<IndexedSubGraph> indexed_sub_graph,
          PrivateTag);

  /// <summary>
  /// Creates an instance of EpGraph, which wraps a GraphViewer.
  /// This call is used when creating an EpGraph from a GraphViewer instance. The GraphViewer instance is not onwed by this EpGraph.
  /// </summary>
  /// <param name="graph_viewer"></param>
  /// <param name="result"></param>
  /// <returns></returns>
  static Status Create(const GraphViewer& graph_viewer, /*out*/ std::unique_ptr<EpGraph>& result);

  /// <summary>
  /// Creates an instance of EpGraph, which wraps a GraphViewer.
  /// This call is used when creating an EpGraph from a subset of nodes in another EpGraph.
  /// In this case, due to the implementation of OrtApis::Graph_GetGraphView, the new EpGraph instance
  /// must take ownership of both the GraphViewer and IndexedSubGraph.
  /// </summary>
  /// <param name="graph_viewer"></param>
  /// <param name="result"></param>
  /// <returns></returns>
  static Status Create(std::unique_ptr<GraphViewer> graph_viewer,
                       std::unique_ptr<IndexedSubGraph> indexed_sub_graph,
                       /*out*/ std::unique_ptr<EpGraph>& result);

  // Defines ToExternal() and ToInternal() functions to convert between OrtGraph and EpGraph.
  DEFINE_ORT_GRAPH_IR_TO_EXTERNAL_INTERNAL_FUNCS(OrtGraph, EpGraph, OrtGraphIrApi::kEpApi)

  //
  // Publicly accessible overrides defined by OrtGraph.
  //

  // Returns the graph's name.
  const std::string& GetName() const override;

  // Returns the model path.
  const ORTCHAR_T* GetModelPath() const override;

  // Returns the model's ONNX IR version.
  int64_t GetOnnxIRVersion() const override;

  // Gets the number of operator sets that the graph's model uses.
  Status GetNumOperatorSets(size_t& num_operator_sets) const override;

  // Gets the operator sets that the graph's model uses. An operator set is uniquely identified by a
  // (domain, opset version) pair.
  Status GetOperatorSets(gsl::span<const char*> domains,
                         gsl::span<int64_t> opset_versions) const override;

  // Get the number of graph inputs, including initializers that are listed as graph inputs.
  size_t GetNumInputs() const override;

  // Gets the graph's inputs as OrtValueInfo instances.
  // Includes initializers that are graph inputs.
  Status GetInputs(gsl::span<const OrtValueInfo*> inputs) const override;

  // Get the number of graph outputs.
  size_t GetNumOutputs() const override;

  // Gets the graph's outputs as OrtValueInfo instances.
  Status GetOutputs(gsl::span<const OrtValueInfo*> outputs) const override;

  // Get the number of graph initializers, including both constant and non-constant initializers.
  size_t GetNumInitializers() const override;

  // Gets the graph's initializers as OrtValueInfo instances.
  // Includes both constant initializers and non-constant initializers (aka optional graph inputs).
  Status GetInitializers(gsl::span<const OrtValueInfo*> initializers) const override;

  // Get the number of nodes in the graph.
  size_t GetNumNodes() const override;

  // Gets the graph's nodes. The nodes are sorted in a default "reverse DFS" topological order.
  Status GetNodes(gsl::span<const OrtNode*> nodes) const override;

  // Gets the graph's parent node or nullptr if this is not a nested subgraph.
  Status GetParentNode(const OrtNode*& parent_node) const override;

  //
  // Helper functions used when working directly with an EpGraph.
  //

  // Sets this graph's parent node.
  void SetParentNode(const EpNode* node);

  // Returns the onnxruntime::GraphViewer& wrapped by this OrtGraph.
  const GraphViewer& GetGraphViewer() const;

  // Returns the EpNode with the given ID (i.e., a NodeIndex).
  // Returns nullptr if this graph does not directly contain a node with the given ID.
  const EpNode* GetNode(NodeIndex node_index) const;

  // Returns the OrtValue for an OrtValueInfo that represents an initializer.
  // Considers both constant and non-constant initializers.
  // Supports initializers defined in an outer scope as long as that initializer is used
  // within this graph.
  Status GetInitializerValue(std::string_view name, const OrtValue*& value) const;

 private:
  /// <summary>
  /// The real implementation of creating an EpGraph instance.
  /// Please use one of the above 'Create' functions that internally call this function, and avoid calling this function directly.
  /// </summary>
  /// <param name="ep_graph"></param>
  /// <param name="graph_viewer"></param>
  /// <param name="result"></param>
  /// <returns></returns>
  static Status CreateImpl(std::unique_ptr<EpGraph> ep_graph, const GraphViewer& graph_viewer, /*out*/ std::unique_ptr<EpGraph>& result);

  const GraphViewer& graph_viewer_;
  const EpNode* parent_node_ = nullptr;

  std::unique_ptr<GraphViewer> owned_graph_viewer_ = nullptr;
  std::unique_ptr<IndexedSubGraph> owned_indexed_sub_graph_ = nullptr;

  std::vector<std::unique_ptr<EpNode>> nodes_;
  IndexToEpNodeMap index_to_ep_node_;

  std::unordered_map<std::string, std::unique_ptr<EpValueInfo>> value_infos_;  // All value infos in the graph

  std::vector<EpValueInfo*> initializer_value_infos_;
  std::unordered_map<std::string_view, std::unique_ptr<OrtValue>> initializer_values_;
  std::unordered_map<std::string_view, std::unique_ptr<OrtValue>> outer_scope_initializer_values_;

  InlinedVector<EpValueInfo*> inputs_;
  InlinedVector<EpValueInfo*> outputs_;
};
}  // namespace onnxruntime
