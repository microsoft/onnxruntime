// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nnapi/nnapi_builtin/selectors_actions/nnapi_qdq_selector_action_transformer.h"
#include "core/providers/nnapi/nnapi_builtin/selectors_actions/nnapi_qdq_selectors.h"

namespace onnxruntime {

using NTO = onnxruntime::ConstNodesToOptimize;

inline void UnaryOpQDQRules(NNAPIQDQSelectorsAndActions& nnapi_qdq_selectors_and_actions) {
  // 3 nodes. DQ, target, Q
  // Replace with internal QLinear version of operator. Delete all original nodes.
  const std::string action_name{"1DQ"};

  std::unique_ptr<NNAPIQDQNodeSelector> selector(new NNAPIQDQ::UnarySelector());
  nnapi_qdq_selectors_and_actions.RegisterSelector(action_name,
                                                   NNAPIQDQSelectorAndAction::OpVersionsMap{{"AveragePool", {}}},
                                                   std::move(selector));
}

inline void BinaryOpQDQRules(NNAPIQDQSelectorsAndActions& nnapi_qdq_selectors_and_actions) {
  // 4 nodes. 2 x DQ for inputs, target, Q
  // Replace with internal QLinear version of operator. Delete all original nodes.
  const std::string action_name{"2DQ"};

  std::unique_ptr<NNAPIQDQNodeSelector> selector(new NNAPIQDQ::BinarySelector());
  nnapi_qdq_selectors_and_actions.RegisterSelector(action_name,
                                                   NNAPIQDQSelectorAndAction::OpVersionsMap{{"Add", {}},
                                                                                            {"Mul", {}}},
                                                   std::move(selector));
}

inline void VariadicOpQDQRules(NNAPIQDQSelectorsAndActions& nnapi_qdq_selectors_and_actions) {
  // 0=variadic DQ nodes 2=target, 3=Q
  // Replace with QLinear version of operator. Delete all original nodes.
  const std::string action_name{"*DQ"};

  std::unique_ptr<NNAPIQDQNodeSelector> selector(new NNAPIQDQ::VariadicSelector());

  nnapi_qdq_selectors_and_actions.RegisterSelector(action_name,
                                                   NNAPIQDQSelectorAndAction::OpVersionsMap{{"Concat", {}}},
                                                   std::move(selector));
}

inline void ConvQDQRules(NNAPIQDQSelectorsAndActions& nnapi_qdq_selectors_and_actions) {
  // 4 or 5 Nodes. 0=DQ X, 1=DQ W, 2=DQ B (optional), 3=Conv, 4=Q
  // Handle the DQ input for the Bias being optional.
  // Replace Conv with QLinearConv
  // Delete all original nodes
  const std::string action_name{"Conv"};

  std::unique_ptr<NNAPIQDQNodeSelector> selector(new NNAPIQDQ::ConvSelector());

  nnapi_qdq_selectors_and_actions.RegisterSelector(action_name,
                                                   NNAPIQDQSelectorAndAction::OpVersionsMap{{"Conv", {}}},
                                                   std::move(selector));
}

inline void MatMulQDQRules(NNAPIQDQSelectorsAndActions& nnapi_qdq_selectors_and_actions) {
  // 3 or 4 nodes. 2 x DQ for inputs, target, optional Q
  // Replace with QLinearMatMul if Q found, or MatMulIntegerToFloat if not.
  // Delete all original nodes.
  const std::string action_name{"MatMul"};

  //std::unique_ptr<Action> action(new QDQ::MatMulReplaceWithQLinear());

  std::unique_ptr<NNAPIQDQNodeSelector> selector(new NNAPIQDQ::MatMulSelector());
  nnapi_qdq_selectors_and_actions.RegisterSelector(action_name,
                                                   NNAPIQDQSelectorAndAction::OpVersionsMap{{"MatMul", {}}},
                                                   std::move(selector));
}

inline NNAPIQDQSelectorsAndActions CreateNNAPISelectorsAndActions() {
  NNAPIQDQSelectorsAndActions nnapi_qdq_selectors_and_actions;

  UnaryOpQDQRules(nnapi_qdq_selectors_and_actions);
  BinaryOpQDQRules(nnapi_qdq_selectors_and_actions);
  VariadicOpQDQRules(nnapi_qdq_selectors_and_actions);
  ConvQDQRules(nnapi_qdq_selectors_and_actions);
  MatMulQDQRules(nnapi_qdq_selectors_and_actions);

  return nnapi_qdq_selectors_and_actions;
}

inline NNAPIQDQSelectorActionTransformer::NNAPIQDQSelectorActionTransformer()
    : NNAPISelectorActionTransformer{
          "NNAPIQDQSelectorActionTransformer",
          CreateNNAPISelectorsAndActions()} {
}

}  // namespace onnxruntime