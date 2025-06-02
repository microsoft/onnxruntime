// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <gsl/gsl>
#include <memory>
#include <unordered_map>
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
  EpValueInfo(const EpGraph& graph, const std::string& name, std::unique_ptr<OrtTypeInfo>&& type_info)
      : OrtValueInfo(OrtGraphIrApi::kEpApi), graph(graph), name(name), type_info(std::move(type_info)) {}

  // Defines ToExternal() and ToInternal() functions to convert between OrtValueInfo and EpValueInfo.
  DEFINE_ORT_GRAPH_IR_TO_EXTERNAL_INTERNAL_FUNCS(OrtValueInfo, EpValueInfo, OrtGraphIrApi::kEpApi)

  const std::string& Name() const override { return name; }
  const OrtTypeInfo* TypeInfo() const override { return type_info.get(); }
  Status GetProducerInfo(OrtValueInfo::ProducerInfo& producer_info) const override;
  Status GetConsumerInfos(std::vector<ConsumerInfo>& consumer_infos) const override;
  Status GetNumConsumers(size_t& num_consumers) const override;

  const EpGraph& graph;  // Back pointer to graph to be able to get consumers/producer
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

  // Defines ToExternal() and ToInternal() functions to convert between OrtNode and EpNode.
  DEFINE_ORT_GRAPH_IR_TO_EXTERNAL_INTERNAL_FUNCS(OrtNode, EpNode, OrtGraphIrApi::kEpApi)

  const std::string& Name() const override;
  const std::string& OpType() const override;
  const std::string& Domain() const override;
  size_t GetNumInputs() const override { return inputs.size(); }
  size_t GetNumOutputs() const override { return outputs.size(); }
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
  size_t NumberOfNodes() const override;
  std::vector<const OrtNode*> GetNodes(int order) const override;

  const GraphViewer& graph_viewer;
  std::vector<std::unique_ptr<EpNode>> nodes;
  std::unordered_map<NodeIndex, EpNode*> index_to_node;

  std::unordered_map<std::string, std::unique_ptr<EpValueInfo>> value_infos;
  InlinedVector<EpValueInfo*> inputs;
  InlinedVector<EpValueInfo*> outputs;
};
}  // namespace onnxruntime
