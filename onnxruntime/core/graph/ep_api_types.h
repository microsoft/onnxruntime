// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <gsl/gsl>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/common/inlined_containers.h"
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
    kIsGraphInput = 1 << 0,
    kIsGraphOutput = 1 << 1,
    kIsInitializer = 1 << 2,
    kIsOuterScope = 1 << 3,
  };

  EpValueInfo(const EpGraph* graph, const std::string& name, std::unique_ptr<OrtTypeInfo>&& type_info,
              size_t flags)
      : OrtValueInfo(OrtGraphIrApi::kEpApi),
        graph(graph),
        name(name),
        type_info(std::move(type_info)),
        flags(flags) {}

  // Defines ToExternal() and ToInternal() functions to convert between OrtValueInfo and EpValueInfo.
  DEFINE_ORT_GRAPH_IR_TO_EXTERNAL_INTERNAL_FUNCS(OrtValueInfo, EpValueInfo, OrtGraphIrApi::kEpApi)

  const std::string& Name() const override { return name; }
  const OrtTypeInfo* TypeInfo() const override { return type_info.get(); }
  Status GetProducerInfo(OrtValueInfo::ProducerInfo& producer_info) const override;
  Status GetConsumerInfos(std::vector<OrtValueInfo::ConsumerInfo>& consumer_infos) const override;
  Status GetNumConsumerInfos(size_t& num_consumers) const override;
  bool IsGraphInput() const override {
    return IsFlagSet(kIsGraphInput);
  }
  bool IsGraphOutput() const override {
    return IsFlagSet(kIsGraphOutput);
  }
  bool IsInitializer() const override {
    return IsFlagSet(kIsInitializer);
  }
  bool IsFromOuterScope() const override {
    return IsFlagSet(kIsOuterScope);
  }

  void SetFlag(EpValueInfo::Flags flag) { flags |= flag; }
  bool IsFlagSet(EpValueInfo::Flags flag) const { return flags & flag; }

  // Back pointer to parent graph. If not null, enables retrieval of consumer and producer nodes.
  // Is null if the EpValueInfo was created without an owning EpGraph
  // (e.g., OrtValueInfo instances created for fused nodes in OrtEp::Compile()).
  const EpGraph* graph = nullptr;
  std::string name;
  std::unique_ptr<OrtTypeInfo> type_info;
  size_t flags = 0;
};

struct SubgraphState {
  SubgraphState() = default;
  SubgraphState(SubgraphState&& other) = default;
  std::unique_ptr<GraphViewer> subgraph_viewer;  // The graph_viewer wrapped by EpGraph below.
  std::unique_ptr<EpGraph> ep_subgraph;
};

/// <summary>
/// Concrete implementation of OrtNode used in the OrtEpApi.
/// </summary>
struct EpNode : public OrtNode {
 private:
  struct PrivateTag {};  // Used to prevent use of public constructor (use static EpNode::Create())
                         // Need to make the constructor public for std::make_unique().

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
  /// <returns>An EpNode instance.</returns>
  static std::unique_ptr<EpNode> Create(const Node& node, const EpGraph* ep_graph,
                                        std::unordered_map<std::string, std::unique_ptr<EpValueInfo>>& value_infos);

  // Defines ToExternal() and ToInternal() functions to convert between OrtNode and EpNode.
  DEFINE_ORT_GRAPH_IR_TO_EXTERNAL_INTERNAL_FUNCS(OrtNode, EpNode, OrtGraphIrApi::kEpApi)

  size_t Id() const override;
  const std::string& Name() const override;
  const std::string& OpType() const override;
  const std::string& Domain() const override;
  Status GetSinceVersion(int& since_version) const override;
  size_t NumInputs() const override { return inputs.size(); }
  size_t NumOutputs() const override { return outputs.size(); }
  Status GetInputs(InlinedVector<const OrtValueInfo*>& inputs) const override;
  Status GetOutputs(InlinedVector<const OrtValueInfo*>& outputs) const override;
  Status GetNumImplicitInputs(size_t& num_implicit_inputs) const override;
  Status GetImplicitInputs(InlinedVector<const OrtValueInfo*>& inputs) const override;
  Status GetNumSubgraphs(size_t& num_subgraphs) const override;
  Status GetSubgraphs(InlinedVector<const OrtGraph*>& subgraphs) const override;
  Status GetParentGraph(const OrtGraph*& parent_graph) const override;

  // Back pointer to containing graph. Useful when traversing through nested subgraphs.
  // Will be nullptr if the EpNode was created without an owning graph.
  // (e.g., OrtNode instances created for fused nodes in OrtEp::Compile()).
  const EpGraph* ep_graph = nullptr;
  const Node& node;
  InlinedVector<EpValueInfo*> inputs;
  InlinedVector<EpValueInfo*> outputs;

  // Storing data related to implicit inputs and subgraphs in std::vector instead of InlinedVector
  // because sizeof(InlinedVector) > sizeof(std::vector) and most nodes will *NOT* have this data.
  std::vector<EpValueInfo*> implicit_inputs;
  std::vector<SubgraphState> subgraphs;
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
  EpGraph(const GraphViewer& graph_viewer, const EpNode* parent_node, PrivateTag);

  static std::unique_ptr<EpGraph> Create(const GraphViewer& graph_viewer, const EpNode* parent_ep_node = nullptr);

  // Defines ToExternal() and ToInternal() functions to convert between OrtGraph and EpGraph.
  DEFINE_ORT_GRAPH_IR_TO_EXTERNAL_INTERNAL_FUNCS(OrtGraph, EpGraph, OrtGraphIrApi::kEpApi)

  const std::string& Name() const override;
  int64_t OnnxIRVersion() const override;
  size_t NumInputs() const override;
  size_t NumOutputs() const override;
  Status GetInputs(InlinedVector<const OrtValueInfo*>& inputs) const override;
  Status GetOutputs(InlinedVector<const OrtValueInfo*>& outputs) const override;
  size_t NumNodes() const override;
  std::vector<const OrtNode*> GetNodes(int order) const override;
  Status GetParentNode(const OrtNode*& parent_node) const override;

  const GraphViewer& graph_viewer;
  const EpNode* parent_node = nullptr;
  std::vector<std::unique_ptr<EpNode>> nodes;
  IndexToEpNodeMap index_to_ep_node;

  std::unordered_map<std::string, std::unique_ptr<EpValueInfo>> value_infos;
  InlinedVector<EpValueInfo*> inputs;
  InlinedVector<EpValueInfo*> outputs;
};
}  // namespace onnxruntime
