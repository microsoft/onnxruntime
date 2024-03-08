// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

#include "utils.h"

#include <iostream>
#include <string>
#include <vector>

#include <core/graph/graph_viewer.h>
#include <core/providers/common.h>

#include "core/optimizer/qdq_transformer/selectors_actions/qdq_selectors.h"
#include "core/optimizer/qdq_transformer/selectors_actions/shared/utils.h"

namespace onnxruntime {
namespace QDQ {

void Selectors::RegisterSelector(const OpVersionsAndSelector::OpVersionsMap& ops_and_versions_in,
                                 std::unique_ptr<NodeGroupSelector> selector_in) {
  auto entry = std::make_unique<OpVersionsAndSelector>(
      ops_and_versions_in,
      std::move(selector_in));

  ORT_IGNORE_RETURN_VALUE(selectors_set_.insert(std::move(entry)));
}

/* static methods to return different operator's OpVersionMap */

// These are operators that do not change the data and therefore the input DQ and
// output Q have the same scale and zero_point.
static const OpVersionsAndSelector::OpVersionsMap GetMiscOpVersionsMap() {
  return {{"Gather", {}},
          {"Reshape", {}},
          {"Expand", {}},
          {"Flatten", {}},
          {"Transpose", {}},
          {"MaxPool", {12}},
          {"Resize", {}},
          {"Squeeze", {}},
          {"Unsqueeze", {}},
          {"Tile", {}}};
}

// These produce int64 indices output, which can't be quantized, so there's no downstream Q node.
static const OpVersionsAndSelector::OpVersionsMap GetDropDQOpVersionsMap() {
  return {{"ArgMax", {}},
          {"ArgMin", {}}};
}

static const OpVersionsAndSelector::OpVersionsMap GetUnaryOpVersionsMap() {
  return {{"AveragePool", {}},
          {"GlobalAveragePool", {}},
          {"GlobalMaxPool", {}},
          {"LeakyRelu", {}},
          {"ReduceMean", {}},
          {"ReduceMin", {}},
          {"ReduceMax", {}},
          {"ReduceProd", {}},
          {"ReduceSum", {}},
          {"Relu", {}},
          {"Gelu", {}},
          {"Elu", {}},
          {"HardSwish", {}},
          {"Sigmoid", {}},
          {"Slice", {}},
          {"LogSoftmax", {}},
          {"Softmax", {}},
          {"Sqrt", {}},
          {"Atan", {}},
          {"Asin", {}},
          {"Sin", {}},
          {"Cos", {}},
          {"Sign", {}},
          {"Tanh", {}},
          {"Exp", {}},
          {"Log", {}},
          {"LRN", {}},
          {"Ceil", {}},
          {"Floor", {}},
          {"Round", {}},
          {"Abs", {}},
          {"Neg", {}},
          {"DepthToSpace", {}},
          {"SpaceToDepth", {}},
          {"Clip", {}},
          {"LpNormalization", {}}};
}
static const OpVersionsAndSelector::OpVersionsMap GetBinaryOpVersionsMap() {
  return {{"Add", {}},
          {"Div", {}},
          {"Mul", {}},
          {"Pow", {}},
          {"Sub", {}},
          {"PRelu", {}},
          {"GridSample", {}}};
}
static const OpVersionsAndSelector::OpVersionsMap GetVariadicOpVersionsMap() {
  return {{"Concat", {}},
          {"Max", {}},
          {"Min", {}}};
}
static const OpVersionsAndSelector::OpVersionsMap GetSplitOpVersionsMap() {
  return {{"Split", {}}};
}
static const OpVersionsAndSelector::OpVersionsMap GetConvOpVersionsMap() {
  return {{"Conv", {}}};
}
static const OpVersionsAndSelector::OpVersionsMap GetConvTransposeOpVersionsMap() {
  return {{"ConvTranspose", {}}};
}
static const OpVersionsAndSelector::OpVersionsMap GetMatMulOpVersionsMap() {
  return {{"MatMul", {}}};
}
static const OpVersionsAndSelector::OpVersionsMap GetGemmOpVersionsMap() {
  return {{"Gemm", {}}};
}
static const OpVersionsAndSelector::OpVersionsMap GetInstanceAndLayerNormalizationOpVersionsMap() {
  return {{"InstanceNormalization", {}},
          {"LayerNormalization", {}}};
}
static const OpVersionsAndSelector::OpVersionsMap GetBatchNormalizationOpVersionsMap() {
  return {{"BatchNormalization", {}}};
}
static const OpVersionsAndSelector::OpVersionsMap GetLogicalComparisonOpVersionsMap() {
  return {{"Equal", {}},
          {"Greater", {}},
          {"GreaterOrEqual", {}},
          {"Less", {}},
          {"LessOrEqual", {}}};
}
static const OpVersionsAndSelector::OpVersionsMap GetWhereOpVersionsMap() {
  return {{"Where", {}}};
}
static const OpVersionsAndSelector::OpVersionsMap GetPadOpVersionsMap() {
  return {{"Pad", {}}};
}

static const OpVersionsAndSelector::OpVersionsMap GetTopKOpVersionsMap() {
  return {{"TopK", {}}};
}

/* Selector rules registration related */
void RegisterMiscSelectors(Selectors& qdq_selectors) {
  /* register selectors for miscellaneous ops */
  std::unique_ptr<NodeGroupSelector> selector = std::make_unique<DropQDQNodeGroupSelector>();
  qdq_selectors.RegisterSelector(GetMiscOpVersionsMap(),
                                 std::move(selector));
}

void RegisterDropDQSelectors(Selectors& qdq_selectors) {
  /* register selectors for ops that have a sigle DQ -> node */
  std::unique_ptr<NodeGroupSelector> selector = std::make_unique<DropDQNodeGroupSelector>();
  qdq_selectors.RegisterSelector(GetDropDQOpVersionsMap(),
                                 std::move(selector));
}

void RegisterUnarySelectors(Selectors& qdq_selectors) {
  /* regsiter selectors for unary ops */
  std::unique_ptr<NodeGroupSelector> selector = std::make_unique<UnaryNodeGroupSelector>();
  qdq_selectors.RegisterSelector(GetUnaryOpVersionsMap(),
                                 std::move(selector));
}

void RegisterBinarySelectors(Selectors& qdq_selectors) {
  /* register selectors for binary ops */
  std::unique_ptr<NodeGroupSelector> selector = std::make_unique<BinaryNodeGroupSelector>();
  qdq_selectors.RegisterSelector(GetBinaryOpVersionsMap(),
                                 std::move(selector));
}

void RegisterVariadicSelectors(Selectors& qdq_selectors) {
  /* register selectors for variadic ops */
  std::unique_ptr<NodeGroupSelector> selector = std::make_unique<VariadicNodeGroupSelector>();
  qdq_selectors.RegisterSelector(GetVariadicOpVersionsMap(),
                                 std::move(selector));
}

void RegisterSplitSelector(Selectors& qdq_selectors) {
  /* register selectors for Split op */
  std::unique_ptr<NodeGroupSelector> selector = std::make_unique<SplitNodeGroupSelector>();
  qdq_selectors.RegisterSelector(GetSplitOpVersionsMap(),
                                 std::move(selector));
}

void RegisterConvSelector(Selectors& qdq_selectors) {
  /* register selector for conv op */
  std::unique_ptr<NodeGroupSelector> selector = std::make_unique<ConvNodeGroupSelector>();
  qdq_selectors.RegisterSelector(GetConvOpVersionsMap(),
                                 std::move(selector));
}

void RegisterConvTransposeSelector(Selectors& qdq_selectors) {
  // register selector for ConvTranspose op
  // it shares selector with Conv op, they have the same input/output def.
  std::unique_ptr<NodeGroupSelector> selector = std::make_unique<ConvNodeGroupSelector>();
  qdq_selectors.RegisterSelector(GetConvTransposeOpVersionsMap(),
                                 std::move(selector));
}

void RegisterMatMulSelector(Selectors& qdq_selectors) {
  /* register selector for matmul op */
  std::unique_ptr<NodeGroupSelector> selector = std::make_unique<MatMulNodeGroupSelector>();
  qdq_selectors.RegisterSelector(GetMatMulOpVersionsMap(),
                                 std::move(selector));
}

void RegisterGemmSelector(Selectors& qdq_selectors) {
  /* register selector for gemm op */
  std::unique_ptr<NodeGroupSelector> selector = std::make_unique<GemmNodeGroupSelector>();
  qdq_selectors.RegisterSelector(GetGemmOpVersionsMap(),
                                 std::move(selector));
}

void RegisterInstanceAndLayerNormalizationSelector(Selectors& qdq_selectors) {
  /* register selector for InstanceNormalization op */
  std::unique_ptr<NodeGroupSelector> selector = std::make_unique<InstanceAndLayerNormalizationNodeGroupSelector>();
  qdq_selectors.RegisterSelector(GetInstanceAndLayerNormalizationOpVersionsMap(),
                                 std::move(selector));
}

void RegisterBatchNormalizationSelector(Selectors& qdq_selectors) {
  /* register selector for BatchNormalization op */
  std::unique_ptr<NodeGroupSelector> selector = std::make_unique<BatchNormalizationNodeGroupSelector>();
  qdq_selectors.RegisterSelector(GetBatchNormalizationOpVersionsMap(),
                                 std::move(selector));
}

void RegisterLogicalComparisonSelectors(Selectors& qdq_selectors) {
  /* register selectors for logical comparison ops */
  std::unique_ptr<NodeGroupSelector> selector = std::make_unique<LogicalComparisonNodeGroupSelector>();
  qdq_selectors.RegisterSelector(GetLogicalComparisonOpVersionsMap(),
                                 std::move(selector));
}

void RegisterWhereSelectors(Selectors& qdq_selectors) {
  /* register selectors for Where ops */
  std::unique_ptr<NodeGroupSelector> selector = std::make_unique<WhereNodeGroupSelector>();
  qdq_selectors.RegisterSelector(GetWhereOpVersionsMap(),
                                 std::move(selector));
}

void RegisterPadSelectors(Selectors& qdq_selectors) {
  /* register selectors for Pad ops */
  std::unique_ptr<NodeGroupSelector> selector = std::make_unique<PadNodeGroupSelector>();
  qdq_selectors.RegisterSelector(GetPadOpVersionsMap(),
                                 std::move(selector));
}

void RegisterTopKSelector(Selectors& qdq_selectors) {
  /* register selector for TopK op */
  std::unique_ptr<NodeGroupSelector> selector = std::make_unique<TopKNodeGroupSelector>();
  qdq_selectors.RegisterSelector(GetTopKOpVersionsMap(),
                                 std::move(selector));
}

void SelectorManager::CreateSelectors() {
  RegisterMiscSelectors(qdq_selectors_);
  RegisterDropDQSelectors(qdq_selectors_);
  RegisterUnarySelectors(qdq_selectors_);
  RegisterBinarySelectors(qdq_selectors_);
  RegisterVariadicSelectors(qdq_selectors_);
  RegisterSplitSelector(qdq_selectors_);
  RegisterConvSelector(qdq_selectors_);
  RegisterConvTransposeSelector(qdq_selectors_);
  RegisterMatMulSelector(qdq_selectors_);
  RegisterGemmSelector(qdq_selectors_);
  RegisterInstanceAndLayerNormalizationSelector(qdq_selectors_);
  RegisterBatchNormalizationSelector(qdq_selectors_);
  RegisterLogicalComparisonSelectors(qdq_selectors_);
  RegisterWhereSelectors(qdq_selectors_);
  RegisterPadSelectors(qdq_selectors_);
  RegisterTopKSelector(qdq_selectors_);
}

void SelectorManager::InitializeSelectorsMap() {
  for (const auto& entry : qdq_selectors_.SelectorsSet()) {
    for (const auto& op_info : entry->op_versions_map) {
      bool inserted = op_type_to_selectors_map_.insert({op_info.first, &*entry}).second;
      ORT_ENFORCE(inserted, "Multiple entries for operator is not supported. OpType=", op_info.first);
    }
  }
}

SelectorManager::SelectorManager() {
  CreateSelectors();
  InitializeSelectorsMap();
}

std::vector<NodeGroup> SelectorManager::GetQDQSelections(const GraphViewer& graph_viewer) const {
  std::vector<NodeGroup> qdq_selections;
  for (auto index : graph_viewer.GetNodesInTopologicalOrder()) {
    const auto* node = graph_viewer.GetNode(index);
    // post layout transformation all the layout sensitive nodes are converted to domain
    // kMSInternalNHWCDomain. Therefore need to allow this domain as well.
    // Allow kMSDomain for contrib op like Gelu
    if (node->Domain() != kOnnxDomain && node->Domain() != kMSInternalNHWCDomain && node->Domain() != kMSDomain) {
      continue;
    }

    auto op_rule = op_type_to_selectors_map_.find(node->OpType());
    if (op_rule == op_type_to_selectors_map_.cend()) {
      continue;
    }

    const auto& op_versions_and_selector = *op_rule->second;

    // check the supported versions if specified
    const auto& versions = op_versions_and_selector.op_versions_map.find(node->OpType())->second;
    if (!versions.empty()) {
      if (std::find(versions.cbegin(), versions.cend(), node->SinceVersion()) == versions.cend()) {
        LOGS_DEFAULT(VERBOSE) << "Op version is not supported for" << node->OpType();
        continue;
      }
    }

    const auto qdq_node_group_selection = op_versions_and_selector.selector->GetQDQSelection(graph_viewer, *node);
    if (qdq_node_group_selection.has_value()) {
      const auto& qdq_group = *qdq_node_group_selection;
      qdq_selections.push_back(qdq_group);
    }
  }

  return qdq_selections;
}

std::pair<std::vector<std::unique_ptr<NodeUnit>>, std::unordered_map<const Node*, const NodeUnit*>>
GetAllNodeUnits(const GraphViewer& graph_viewer) {
  std::vector<std::unique_ptr<NodeUnit>> node_unit_holder;
  std::unordered_map<const Node*, const NodeUnit*> node_unit_map;

  const auto add_node_unit_to_map = [&](const std::vector<NodeIndex>& node_indices, const NodeUnit* node_unit) {
    for (const auto& node_idx : node_indices) {
      const auto* node = graph_viewer.GetNode(node_idx);
      node_unit_map.insert({node, node_unit});
    }
  };

  // Get QDQ NodeUnits first
  QDQ::SelectorManager selector_mgr;
  const auto qdq_selections = selector_mgr.GetQDQSelections(graph_viewer);

  for (const auto& qdq_selection : qdq_selections) {
    auto qdq_unit = std::make_unique<NodeUnit>(graph_viewer, qdq_selection);

    // Fill the node to node_unit map for all nodes in the QDQ Group
    add_node_unit_to_map(qdq_selection.dq_nodes, qdq_unit.get());
    add_node_unit_to_map(qdq_selection.q_nodes, qdq_unit.get());
    add_node_unit_to_map({qdq_selection.target_node}, qdq_unit.get());

    node_unit_holder.push_back(std::move(qdq_unit));
  }

  // Get the left over SingleNode NodeUnits
  const auto& node_indices = graph_viewer.GetNodesInTopologicalOrder();
  for (const auto node_idx : node_indices) {
    const auto* node(graph_viewer.GetNode(node_idx));

    // This is already part of a QDQ NodeUnit
    if (node_unit_map.find(node) != node_unit_map.cend())
      continue;

    auto node_unit = std::make_unique<NodeUnit>(*node);
    node_unit_map[node] = node_unit.get();
    node_unit_holder.push_back(std::move(node_unit));
  }

  return std::make_pair(std::move(node_unit_holder), std::move(node_unit_map));
}

}  // namespace QDQ
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
