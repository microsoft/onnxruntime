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

namespace onnxruntime {
class Node;
class GraphViewer;
struct EpGraph;

/// <summary>
/// Concrete implementation of OrtValueInfo used in the OrtEpApi.
/// </summary>
struct EpValueInfo : public OrtValueInfo {
 public:
  EpValueInfo(const EpGraph* graph, const std::string& name, std::unique_ptr<OrtTypeInfo>&& type_info)
      : OrtValueInfo(OrtGraphIrApi::kEpApi), graph(graph), name(name), type_info(std::move(type_info)) {}

  // Defines ToExternal() and ToInternal() functions to convert between OrtValueInfo and EpValueInfo.
  DEFINE_ORT_GRAPH_IR_TO_EXTERNAL_INTERNAL_FUNCS(OrtValueInfo, EpValueInfo, OrtGraphIrApi::kEpApi)

  const std::string& Name() const override { return name; }
  const OrtTypeInfo* TypeInfo() const override { return type_info.get(); }
  Status GetProducerInfo(OrtValueInfo::ProducerInfo& producer_info) const override;
  Status GetUses(std::vector<OrtValueInfo::UseInfo>& uses) const override;
  Status GetNumUses(size_t& num_consumers) const override;

  // Back pointer to parent graph. If not null, enables retrieval of consumer and producer nodes.
  // Is null if the EpValueInfo was created without an owning EpGraph
  // (e.g., OrtValueInfo instances created for fused nodes in OrtEp::Compile()).
  const EpGraph* graph;
  std::string name;
  std::unique_ptr<OrtTypeInfo> type_info;
};

/// <summary>
/// Concrete implementation of OrtNode used in the OrtEpApi.
/// </summary>
struct EpNode : public OrtNode {
 public:
  EpNode(const Node& node, InlinedVector<EpValueInfo*>&& inputs, InlinedVector<EpValueInfo*>&& outputs)
      : OrtNode(OrtGraphIrApi::kEpApi), node(node), inputs(std::move(inputs)), outputs(std::move(outputs)) {}

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

  const std::string& Name() const override;
  const std::string& OpType() const override;
  const std::string& Domain() const override;
  size_t NumInputs() const override { return inputs.size(); }
  size_t NumOutputs() const override { return outputs.size(); }
  Status GetInputs(InlinedVector<const OrtValueInfo*>& inputs) const override;
  Status GetOutputs(InlinedVector<const OrtValueInfo*>& outputs) const override;

  const Node& node;
  InlinedVector<EpValueInfo*> inputs;
  InlinedVector<EpValueInfo*> outputs;
};

/// <summary>
/// Concrete implementation of OrtGraph used in the OrtEpApi.
/// </summary>
struct EpGraph : public OrtGraph {
 private:
  struct PrivateTag {};  // Used to prevent use of public constructor (use static EpGraph::Create())
                         // Need to make the constructor public for std::make_unique().

 public:
  EpGraph(const GraphViewer& graph_viewer, PrivateTag) : OrtGraph(OrtGraphIrApi::kEpApi), graph_viewer(graph_viewer) {}

  static std::unique_ptr<EpGraph> Create(const GraphViewer& graph_viewer);

  // Defines ToExternal() and ToInternal() functions to convert between OrtGraph and EpGraph.
  DEFINE_ORT_GRAPH_IR_TO_EXTERNAL_INTERNAL_FUNCS(OrtGraph, EpGraph, OrtGraphIrApi::kEpApi)

  const std::string& Name() const override;
  size_t NumInputs() const override;
  size_t NumOutputs() const override;
  size_t NumNodes() const override;
  std::vector<const OrtNode*> GetNodes(int order) const override;

  const GraphViewer& graph_viewer;
  std::vector<std::unique_ptr<EpNode>> nodes;
  std::unordered_map<NodeIndex, EpNode*> index_to_node;

  std::unordered_map<std::string, std::unique_ptr<EpValueInfo>> value_infos;
  InlinedVector<EpValueInfo*> inputs;
  InlinedVector<EpValueInfo*> outputs;
};
}  // namespace onnxruntime
