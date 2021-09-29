// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nnapi/nnapi_builtin/selectors_actions/nnapi_qdq_selector_action_transformer.h"
#include "core/providers/nnapi/nnapi_builtin/selectors_actions/nnapi_qdq_actions.h"
#include "core/providers/nnapi/nnapi_builtin/selectors_actions/nnapi_qdq_selectors.h"

namespace onnxruntime {

using NTO = onnxruntime::NodesToOptimize;

// create rules for ops that don't change the data
void DropQDQNodesRules(NNAPIQDQSelectorsAndActions& nnapi_qdq_selectors_and_actions) {
  // 3 nodes. DQ, target, Q. Merge into target and remove DQ and Q.
  const std::string action_name{"drop"};
  NTO::NodeLocation dq{NTO::NodeType::kInput, 0};
  NTO::NodeLocation q{NTO::NodeType::kOutput, 0};

  // Move DQ input 0 to target input 0.
  // Move Q output 0 to target output 0.j
  std::vector<NodeAndMoveInfo> moves{
      MoveToSlot(dq, ArgType::kInput, 0, ArgType::kInput, 0),
      MoveToSlot(q, ArgType::kOutput, 0, ArgType::kOutput, 0)};

  std::unique_ptr<NNAPIQDQNodeSelector> selector(new QDQ::DropDQDNodesSelector());
  nnapi_qdq_selectors_and_actions.RegisterSelector(action_name,
                                                   NNAPIQDQSelectorAndAction::OpVersionsMap{{"Gather", {}},
                                                                                            {"Reshape", {}},
                                                                                            {"Transpose", {}},
                                                                                            {"MaxPool", {12}}},
                                                   std::move(selector));
}

void UnaryOpQDQRules(NNAPIQDQSelectorsAndActions& nnapi_qdq_selectors_and_actions) {
  // 3 nodes. DQ, target, Q
  // Replace with internal QLinear version of operator. Delete all original nodes.
  const std::string action_name{"1DQ"};

  std::unique_ptr<NNAPIQDQNodeSelector> selector(new QDQ::UnarySelector());
  nnapi_qdq_selectors_and_actions.RegisterSelector(action_name,
                                                   NNAPIQDQSelectorAndAction::OpVersionsMap{{"AveragePool", {}}},
                                                   std::move(selector));
}

void BinaryOpQDQRules(NNAPIQDQSelectorsAndActions& nnapi_qdq_selectors_and_actions) {
  // 4 nodes. 2 x DQ for inputs, target, Q
  // Replace with internal QLinear version of operator. Delete all original nodes.
  const std::string action_name{"2DQ"};

  std::unique_ptr<NNAPIQDQNodeSelector> selector(new QDQ::BinarySelector());
  nnapi_qdq_selectors_and_actions.RegisterSelector(action_name,
                                                   NNAPIQDQSelectorAndAction::OpVersionsMap{{"Add", {}},
                                                                                            {"Mul", {}}},
                                                   std::move(selector));
}

void VariadicOpQDQRules(NNAPIQDQSelectorsAndActions& nnapi_qdq_selectors_and_actions) {
  // 0=variadic DQ nodes 2=target, 3=Q
  // Replace with QLinear version of operator. Delete all original nodes.
  const std::string action_name{"*DQ"};

  std::unique_ptr<NNAPIQDQNodeSelector> selector(new QDQ::VariadicSelector());

  nnapi_qdq_selectors_and_actions.RegisterSelector(action_name,
                                                   NNAPIQDQSelectorAndAction::OpVersionsMap{{"Concat", {}}},
                                                   std::move(selector));
}

void ConvQDQRules(NNAPIQDQSelectorsAndActions& nnapi_qdq_selectors_and_actions) {
  // 4 or 5 Nodes. 0=DQ X, 1=DQ W, 2=DQ B (optional), 3=Conv, 4=Q
  // Handle the DQ input for the Bias being optional.
  // Replace Conv with QLinearConv
  // Delete all original nodes
  const std::string action_name{"Conv"};

  std::unique_ptr<NNAPIQDQNodeSelector> selector(new QDQ::ConvSelector());

  nnapi_qdq_selectors_and_actions.RegisterSelector(action_name,
                                                   NNAPIQDQSelectorAndAction::OpVersionsMap{{"Conv", {}}},
                                                   std::move(selector));
}

void MatMulQDQRules(NNAPIQDQSelectorsAndActions& nnapi_qdq_selectors_and_actions) {
  // 3 or 4 nodes. 2 x DQ for inputs, target, optional Q
  // Replace with QLinearMatMul if Q found, or MatMulIntegerToFloat if not.
  // Delete all original nodes.
  const std::string action_name{"MatMul"};

  //std::unique_ptr<Action> action(new QDQ::MatMulReplaceWithQLinear());

  std::unique_ptr<NNAPIQDQNodeSelector> selector(new QDQ::MatMulSelector());
  nnapi_qdq_selectors_and_actions.RegisterSelector(action_name,
                                                   NNAPIQDQSelectorAndAction::OpVersionsMap{{"MatMul", {}}},
                                                   std::move(selector));
}

NNAPIQDQSelectorsAndActions CreateNNAPISelectorsAndActions() {
  NNAPIQDQSelectorsAndActions nnapi_qdq_selectors_and_actions;

  DropQDQNodesRules(nnapi_qdq_selectors_and_actions);
  UnaryOpQDQRules(nnapi_qdq_selectors_and_actions);
  BinaryOpQDQRules(nnapi_qdq_selectors_and_actions);
  VariadicOpQDQRules(nnapi_qdq_selectors_and_actions);
  ConvQDQRules(nnapi_qdq_selectors_and_actions);
  MatMulQDQRules(nnapi_qdq_selectors_and_actions);

  return nnapi_qdq_selectors_and_actions;
}

NNAPIQDQSelectorActionTransformer::NNAPIQDQSelectorActionTransformer()
    : NNAPISelectorActionTransformer{
          "NNAPIQDQSelectorActionTransformer",
          CreateNNAPISelectorsAndActions()} {
}

}  // namespace onnxruntime