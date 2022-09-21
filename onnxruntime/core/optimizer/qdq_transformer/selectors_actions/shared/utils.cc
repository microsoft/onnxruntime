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
static const OpVersionsAndSelector::OpVersionsMap GetMiscOpVersionsMap() {
  return {{"Gather", {}},
          {"Reshape", {}},
          {"Transpose", {}},
          {"MaxPool", {12}},
          {"Resize", {}},
          {"Squeeze", {}},
          {"Unsqueeze", {}}};
}

static const OpVersionsAndSelector::OpVersionsMap GetUnaryOpVersionsMap() {
  return {{"AveragePool", {}},
          {"Softmax", {}},
          {"LeakyRelu", {}}};
}
static const OpVersionsAndSelector::OpVersionsMap GetBinaryOpVersionsMap() {
  return {{"Add", {}},
          {"Mul", {}}};
}
static const OpVersionsAndSelector::OpVersionsMap GetVariadicOpVersionsMap() {
  return {{"Concat", {}}};
}
static const OpVersionsAndSelector::OpVersionsMap GetConvOpVersionsMap() {
  return {{"Conv", {}}};
}
static const OpVersionsAndSelector::OpVersionsMap GetMatMulOpVersionsMap() {
  return {{"MatMul", {}}};
}
static const OpVersionsAndSelector::OpVersionsMap GetGemmOpVersionsMap() {
  return {{"Gemm", {}}};
}

/* Selector rules registration related */
void RegisterMiscSelectors(Selectors& qdq_selectors) {
  /* register selectors for miscellaneous ops */
  std::unique_ptr<NodeGroupSelector> selector = std::make_unique<DropQDQNodeGroupSelector>();
  qdq_selectors.RegisterSelector(GetMiscOpVersionsMap(),
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

void RegisterConvSelector(Selectors& qdq_selectors) {
  /* register selector for conv op */
  std::unique_ptr<NodeGroupSelector> selector = std::make_unique<ConvNodeGroupSelector>();
  qdq_selectors.RegisterSelector(GetConvOpVersionsMap(),
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

void SelectorManager::CreateSelectors() {
  RegisterMiscSelectors(qdq_selectors_);
  RegisterUnarySelectors(qdq_selectors_);
  RegisterBinarySelectors(qdq_selectors_);
  RegisterVariadicSelectors(qdq_selectors_);
  RegisterConvSelector(qdq_selectors_);
  RegisterMatMulSelector(qdq_selectors_);
  RegisterGemmSelector(qdq_selectors_);
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
    if (node->Domain() != kOnnxDomain && node->Domain() != kMSInternalNHWCDomain) {
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

}  // namespace QDQ
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
