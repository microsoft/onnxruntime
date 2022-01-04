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

void SelectorManager::CreateSelectors() {
  RegisterMiscSelectors(qdq_selectors_);
  RegisterUnarySelectors(qdq_selectors_);
  RegisterBinarySelectors(qdq_selectors_);
  RegisterVariadicSelectors(qdq_selectors_);
  RegisterConvSelector(qdq_selectors_);
  RegisterMatMulSelector(qdq_selectors_);
}

void SelectorManager::InitializeSelectorsMap() {
  for (const auto& entry : qdq_selectors_.SelectorsSet()) {
    for (const auto& op_info : entry->op_versions_map) {
      bool inserted = op_type_to_selectors_map_.insert({op_info.first, &*entry}).second;
      ORT_ENFORCE(inserted, "Multiple entries for operator is not supported. OpType=", op_info.first);
    }
  }
}

const std::unordered_map<const Node*, std::unique_ptr<BaseSelector>> SelectorManager::GetQDQSelectors(const GraphViewer& graph_viewer) const {
  std::unordered_map<const Node*, std::unique_ptr<BaseSelector>> qdq_selectors;
  for (auto index : graph_viewer.GetNodesInTopologicalOrder()) {
    const auto* node = graph_viewer.GetNode(index);
    if (node->Domain() != kOnnxDomain) {
      break;
    }

    auto op_rule = op_type_to_selectors_map_.find(node->OpType());
    if (op_rule == op_type_to_selectors_map_.cend()) {
      if (node->OpType() != "DequantizeLinear" && node->OpType() != "QuantizeLinear") {
        break;
      }
    }

    std::string op_type = "";

    if (node->OpType() != "DequantizeLinear" && node->OpType() != "QuantizeLinear") {
      const auto& selector = *op_rule->second;

      // check the supported versions if specified
      const auto& versions = selector.op_versions_map.find(node->OpType())->second;
      if (!versions.empty()) {
        if (std::find(versions.cbegin(), versions.cend(), node->SinceVersion()) == versions.cend()) {
          LOGS_DEFAULT(VERBOSE) << "Op version is not supported for" << node->OpType();
          break;
        }
      }

      op_type = op_rule->first;
    }

    // identify the op_type and add corresponding selector into the result
    if (op_type == "AveragePool") {
      qdq_selectors.emplace(node, std::make_unique<QDQ::UnarySelector>());
    } else if (op_type == "Add" || op_type == "Mul") {
      qdq_selectors.emplace(node, std::make_unique<QDQ::BinarySelector>());
    } else if (op_type == "Concat") {
      qdq_selectors.emplace(node, std::make_unique<QDQ::VariadicSelector>());
    } else if (op_type == "Conv") {
      qdq_selectors.emplace(node, std::make_unique<QDQ::ConvSelector>());
    } else if (op_type == "MatMul") {
      qdq_selectors.emplace(node, std::make_unique<QDQ::MatMulSelector>());
    } else if (op_type == "Gather" || op_type == "Reshape" || op_type == "Transpose" || op_type == "MaxPool" || op_type == "Resize") {
      qdq_selectors.emplace(node, std::make_unique<QDQ::DropDQDNodesSelector>());
    } else {
      LOGS_DEFAULT(VERBOSE) << "Selector type is not supported.";
    }
  }

  return qdq_selectors;
}

}  // namespace QDQ
}  // namespace onnxruntime