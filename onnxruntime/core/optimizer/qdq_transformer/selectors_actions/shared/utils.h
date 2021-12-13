// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include "core/graph/basic_types.h"
#include "core/providers/nnapi/nnapi_builtin/nnapi_lib/NeuralNetworksTypes.h"
#include "../qdq_selectors.h"
#include "../../../selectors_actions/helpers.h"

namespace onnxruntime {

class GraphViewer;
class Node;

namespace QDQ {

struct Selector {
  using OpVersionsMap = std::unordered_map<std::string, std::vector<ONNX_NAMESPACE::OperatorSetVersion>>;

  Selector(const OpVersionsMap& ops_and_versions_in,
           std::unique_ptr<QDQ::BaseSelector> selector_in)
      : op_versions_map{ops_and_versions_in},
        selector{std::move(selector_in)} {}

  OpVersionsMap op_versions_map;
  std::unique_ptr<QDQ::BaseSelector> selector;

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(Selector);
};

class Selectors {
 public:
  Selectors() = default;

  Selectors(Selectors&& rhs) noexcept
      : selectors_set_{std::move(rhs.selectors_set_)} {}

  void RegisterSelector(const Selector::OpVersionsMap& ops_and_versions_in,
                        std::unique_ptr<QDQ::BaseSelector> selector_in);

  const std::unordered_set<std::unique_ptr<Selector>>& SelectorsSet() const {
    return selectors_set_;
  }

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(Selectors);

  std::unordered_set<std::unique_ptr<Selector>> selectors_set_;
};

std::unordered_map<std::string, const Selector*> op_type_to_selectors_map;

Selectors CreateSelectors();

void InitializeSelectorsMap(Selectors selectors);

std::unique_ptr<QDQ::BaseSelector> GetQDQSelector(const Node& node);

std::vector<std::unique_ptr<QDQ::BaseSelector>> GetQDQSelectors(const GraphViewer& graph_viewer);

}  // namespace QDQ
}  // namespace onnxruntime