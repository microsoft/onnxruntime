// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>
#include <string>
#include <vector>
#include <optional>

#include <core/graph/graph_viewer.h>
#include <core/providers/common.h>

#include "core/optimizer/qdq_transformer/selectors_actions/qdq_selector_action_transformer.h"
#include "core/optimizer/qdq_transformer/selectors_actions/qdq_selectors.h"

#include "utils.h"

namespace onnxruntime {
namespace QDQ {

using std::string;
using std::vector;

using OpVersionsMap = std::unordered_map<std::string, std::vector<ONNX_NAMESPACE::OperatorSetVersion>>;

void Selectors::RegisterSelector(const Selector::OpVersionsMap& ops_and_versions_in,
                                 std::unique_ptr<QDQ::BaseSelector> selector_in) {
  auto entry = std::make_unique<Selector>(
      ops_and_versions_in,
      std::move(selector_in));

  ORT_IGNORE_RETURN_VALUE(selectors_set_.insert(std::move(entry)));
}

/* Selector Rules Related */
void RegisterMiscSelectors(Selectors& qdq_selectors) {
  /* register selectors for miscellaneous ops */
  std::unique_ptr<BaseSelector> selector(new QDQ::DropDQDNodesSelector());
  qdq_selectors.RegisterSelector(Selector::OpVersionsMap{{"Gather", {}},
                                                         {"Reshape", {}},
                                                         {"Transpose", {}},
                                                         {"MaxPool", {12}},
                                                         {"Resize", {}}},
                                 std::move(selector));
}

void RegisterUnarySelectors(Selectors& qdq_selectors) {
  /* regsiter selectors for unary ops */
  std::unique_ptr<BaseSelector> selector(new QDQ::UnarySelector());
  qdq_selectors.RegisterSelector(Selector::OpVersionsMap{{"AveragePool", {}}},
                                 std::move(selector));
}

void RegisterBinarySelectors(Selectors& qdq_selectors) {
  /* register selectors for binary ops */
  std::unique_ptr<BaseSelector> selector(new QDQ::BinarySelector());
  qdq_selectors.RegisterSelector(Selector::OpVersionsMap{{"Add", {}},
                                                         {"Mul", {}}},
                                 std::move(selector));
}

void RegisterVariadicSelectors(Selectors& qdq_selectors) {
  /* register selectors for variadic ops */
  std::unique_ptr<BaseSelector> selector(new QDQ::VariadicSelector());
  qdq_selectors.RegisterSelector(Selector::OpVersionsMap{{"Concat", {}}},
                                 std::move(selector));
}

void RegisterConvSelector(Selectors& qdq_selectors) {
  /* register selector for conv op */
  std::unique_ptr<BaseSelector> selector(new QDQ::ConvSelector());
  qdq_selectors.RegisterSelector(Selector::OpVersionsMap{{"Conv", {}}},
                                 std::move(selector));
}

void RegisterMatMulSelector(Selectors& qdq_selectors) {
  /* register selector for matmul op */
  std::unique_ptr<BaseSelector> selector(new QDQ::MatMulSelector());
  qdq_selectors.RegisterSelector(Selector::OpVersionsMap{{"MatMul", {}}},
                                 std::move(selector));
}

Selectors CreateSelectors() {
  Selectors qdq_selectors;

  RegisterMiscSelectors(qdq_selectors);
  RegisterUnarySelectors(qdq_selectors);
  RegisterBinarySelectors(qdq_selectors);
  RegisterVariadicSelectors(qdq_selectors);
  RegisterConvSelector(qdq_selectors);
  RegisterMatMulSelector(qdq_selectors);

  return qdq_selectors;
}

void IntializeSelectorsMap(Selectors selectors) {
  for (const auto& entry : selectors.SelectorsSet()) {
    for (const auto& op_info : entry->op_versions_map) {
      bool inserted = op_type_to_selectors_map.insert({op_info.first, &*entry}).second;
      ORT_ENFORCE(inserted, "Multiple entries for operator is not supported. OpType=", op_info.first);
    }
  }
}

std::unique_ptr<BaseSelector> GetQDQSelector(const Node& node) {
  if (node.Domain() != kOnnxDomain) {
    return nullptr;
  }

  auto op_rule = op_type_to_selectors_map.find(node.OpType());
  if (op_rule == op_type_to_selectors_map.cend()) {
    return nullptr;
  }

  const auto& selector = *op_rule->second;

  // check the supported versions if specified
  const auto& versions = selector.op_versions_map.find(node.OpType())->second;
  if (!versions.empty()) {
    if (std::find(versions.cbegin(), versions.cend(), node.SinceVersion()) == versions.cend()) {
      LOGS_DEFAULT(VERBOSE) << "Op version is not supported for" << node.OpType();
      return nullptr;
    }
  }

  return std::move(selector.selector);
}

std::vector<std::unique_ptr<BaseSelector>> GetQDQSelectors(const GraphViewer& graph_viewer) {
  std::vector<std::unique_ptr<BaseSelector>> qdq_selectors;
  for (auto index : graph_viewer.GetNodesInTopologicalOrder()) {
    const auto* node = graph_viewer.GetNode(index);
    auto qdq_selector = GetQDQSelector(*node);
    qdq_selectors.push_back(std::move(qdq_selector));
  }
  return qdq_selectors;
}

}  // namespace QDQ
}  // namespace onnxruntime