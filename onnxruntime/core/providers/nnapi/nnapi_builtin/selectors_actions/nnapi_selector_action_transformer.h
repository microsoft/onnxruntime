// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <core/graph/graph.h>
#include "core/providers/nnapi/nnapi_builtin/selectors_actions/nnapi_qdq_selector_helper.h"

namespace onnxruntime {

class Graph;
class Node;

struct NNAPIQDQNodeSelector {
  // Select one or more qdq nodes structure for NNAPI EP to determine support capabilities
  // TODO: Select() takes in a graph may cause issues. (graph_viewer?)
  virtual bool Select(const Graph& graph, const Node& node, std::unique_ptr<ConstNodesToOptimize>& selection) const = 0;
  virtual ~NNAPIQDQNodeSelector() = default;

 protected:
  NNAPIQDQNodeSelector() = default;
};

struct NNAPIQDQSelectorAndAction {
  using OpVersionsMap = std::unordered_map<std::string, std::vector<ONNX_NAMESPACE::OperatorSetVersion>>;

  // ctor so we can use make_unique to construct this class
  NNAPIQDQSelectorAndAction(const std::string& name_in,
                            const OpVersionsMap& ops_and_versions_in,
                            std::unique_ptr<NNAPIQDQNodeSelector> selector_in)
      : name{name_in},
        ops_and_versions{ops_and_versions_in},
        nnapi_selector{std::move(selector_in)} {}

  const std::string name;
  OpVersionsMap ops_and_versions;
  std::unique_ptr<NNAPIQDQNodeSelector> nnapi_selector;

  // can't copy/assign our unique_ptr members
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(NNAPIQDQSelectorAndAction);
};

class NNAPIQDQSelectorsAndActions {
 public:
  NNAPIQDQSelectorsAndActions() = default;

  NNAPIQDQSelectorsAndActions(NNAPIQDQSelectorsAndActions&& nnapi_sat) noexcept
      : nnapi_selectors_and_actions_map_{std::move(nnapi_sat.nnapi_selectors_and_actions_map_)} {}

  void RegisterSelector(const std::string& name,
                        const NNAPIQDQSelectorAndAction::OpVersionsMap& ops_and_versions_in,
                        std::unique_ptr<NNAPIQDQNodeSelector> selector_in);

  const std::unordered_map<std::string, std::unique_ptr<NNAPIQDQSelectorAndAction>>& NNAPIQDQSelectorsAndActionsMap() const {
    return nnapi_selectors_and_actions_map_;
  }

 private:
  std::unordered_map<std::string, std::unique_ptr<NNAPIQDQSelectorAndAction>> nnapi_selectors_and_actions_map_;
};

class NNAPISelectorActionTransformer {
 public:
  NNAPISelectorActionTransformer() = default;

  NNAPISelectorActionTransformer(const std::string& name, NNAPIQDQSelectorsAndActions&& nnapi_qdq_selectors_and_actions);

  const std::string name_;
  NNAPIQDQSelectorsAndActions nnapi_qdq_selectors_and_actions_;

  std::unique_ptr<ConstNodesToOptimize> Match(const Graph& graph, const Node& node) const;

  std::unordered_map<std::string, const NNAPIQDQSelectorAndAction*> op_type_to_nnapi_qdq_sat_;
};

}  // namespace onnxruntime