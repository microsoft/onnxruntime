// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/qdq_transformer/qdq_actions.h"
#if !defined(ORT_MINIMAL_BUILD)
#include "core/optimizer/qdq_transformer/qdq_selector_action_transformer.h"
#include "core/optimizer/qdq_transformer/qdq_selectors.h"
#endif

namespace onnxruntime {

namespace {

using NTO = onnxruntime::NodesToOptimize;

// create rules for ops that don't change the data
void DropQDQNodesRules(SelectorsAndActions& qdq_selectors_and_actions) {
  // 3 nodes. DQ, target, Q. Merge into target and remove DQ and Q.
  const std::string action_name{"drop"};
  NTO::NodeLocation dq{NTO::NodeType::kInput, 0};
  NTO::NodeLocation q{NTO::NodeType::kOutput, 0};

  // Move DQ input 0 to target input 0.
  // Move Q output 0 to target output 0.
  std::vector<NodeAndMoveInfo> moves{
      MoveToSlot(dq, ArgType::kInput, 0, ArgType::kInput, 0),
      MoveToSlot(q, ArgType::kOutput, 0, ArgType::kOutput, 0)};

  std::unique_ptr<Action> action(new MergeIntoTarget(std::move(moves)));

#if !defined(ORT_MINIMAL_BUILD)
  std::unique_ptr<NodeSelector> selector(new QDQ::DropDQDNodesSelector());
  qdq_selectors_and_actions.RegisterSelectorAndAction(action_name,
                                                      SelectorAndAction::OpVersionsMap{{"Gather", {}},
                                                                                       {"Reshape", {}},
                                                                                       {"Transpose", {}},
                                                                                       {"MaxPool", {12}}},
                                                      std::move(selector),
                                                      std::move(action));
#else
  qdq_selectors_and_actions.RegisterAction(action_name, std::move(action));
#endif
}

void UnaryOpQDQRules(SelectorsAndActions& qdq_selectors_and_actions) {
  // 3 nodes. DQ, target, Q
  // Replace with internal QLinear version of operator. Delete all original nodes.
  const std::string action_name{"1DQ"};
  std::unique_ptr<Action> action(new QDQ::UnaryReplaceWithQLinear(kMSDomain));

#if !defined(ORT_MINIMAL_BUILD)
  std::unique_ptr<NodeSelector> selector(new QDQ::UnarySelector());
  qdq_selectors_and_actions.RegisterSelectorAndAction(action_name,
                                                      SelectorAndAction::OpVersionsMap{{"AveragePool", {}}},
                                                      std::move(selector),
                                                      std::move(action));
#else
  qdq_selectors_and_actions.RegisterAction(action_name, std::move(action));
#endif
}

void BinaryOpQDQRules(SelectorsAndActions& qdq_selectors_and_actions) {
  // 4 nodes. 2 x DQ for inputs, target, Q
  // Replace with internal QLinear version of operator. Delete all original nodes.
  const std::string action_name{"2DQ"};
  std::unique_ptr<Action> action(new QDQ::BinaryReplaceWithQLinear(kMSDomain));

#if !defined(ORT_MINIMAL_BUILD)
  std::unique_ptr<NodeSelector> selector(new QDQ::BinarySelector());
  qdq_selectors_and_actions.RegisterSelectorAndAction(action_name,
                                                      SelectorAndAction::OpVersionsMap{{"Add", {}},
                                                                                       {"Mul", {}}},
                                                      std::move(selector),
                                                      std::move(action));

#else
  qdq_selectors_and_actions.RegisterAction(action_name, std::move(action));
#endif
}

void VariadicOpQDQRules(SelectorsAndActions& qdq_selectors_and_actions) {
  // 0=variadic DQ nodes 2=target, 3=Q
  // Replace with QLinear version of operator. Delete all original nodes.
  const std::string action_name{"*DQ"};
  std::unique_ptr<Action> action(new QDQ::VariadicReplaceWithQLinear(kMSDomain));

#if !defined(ORT_MINIMAL_BUILD)
  std::unique_ptr<NodeSelector> selector(new QDQ::VariadicSelector());

  qdq_selectors_and_actions.RegisterSelectorAndAction(action_name,
                                                      SelectorAndAction::OpVersionsMap{{"Concat", {}}},
                                                      std::move(selector),
                                                      std::move(action));

#else
  qdq_selectors_and_actions.RegisterAction(action_name, std::move(action));
#endif
}

void ConvQDQRules(SelectorsAndActions& qdq_selectors_and_actions) {
  // 4 or 5 Nodes. 0=DQ X, 1=DQ W, 2=DQ B (optional), 3=Conv, 4=Q
  // Handle the DQ input for the Bias being optional.
  // Replace Conv with QLinearConv
  // Delete all original nodes
  const std::string action_name{"Conv"};
  std::unique_ptr<Action> action(new QDQ::ConvReplaceWithQLinear());

#if !defined(ORT_MINIMAL_BUILD)
  std::unique_ptr<NodeSelector> selector(new QDQ::ConvSelector());

  qdq_selectors_and_actions.RegisterSelectorAndAction(action_name,
                                                      SelectorAndAction::OpVersionsMap{{"Conv", {}}},
                                                      std::move(selector),
                                                      std::move(action));

#else
  qdq_selectors_and_actions.RegisterAction(action_name, std::move(action));
#endif
}

void MatMulQDQRules(SelectorsAndActions& qdq_selectors_and_actions) {
  // 3 or 4 nodes. 2 x DQ for inputs, target, optional Q
  // Replace with QLinearMatMul if Q found, or MatMulIntegerToFloat if not.
  // Delete all original nodes.
  const std::string action_name{"MatMul"};

  std::unique_ptr<Action> action(new QDQ::MatMulReplaceWithQLinear());

#if !defined(ORT_MINIMAL_BUILD)
  std::unique_ptr<NodeSelector> selector(new QDQ::MatMulSelector());
  qdq_selectors_and_actions.RegisterSelectorAndAction(action_name,
                                                      SelectorAndAction::OpVersionsMap{{"MatMul", {}}},
                                                      std::move(selector),
                                                      std::move(action));

#else
  qdq_selectors_and_actions.RegisterAction(action_name, std::move(action));
#endif
}

SelectorsAndActions CreateSelectorsAndActions() {
  SelectorsAndActions qdq_selectors_and_actions;

  DropQDQNodesRules(qdq_selectors_and_actions);
  UnaryOpQDQRules(qdq_selectors_and_actions);
  BinaryOpQDQRules(qdq_selectors_and_actions);
  VariadicOpQDQRules(qdq_selectors_and_actions);
  ConvQDQRules(qdq_selectors_and_actions);
  MatMulQDQRules(qdq_selectors_and_actions);

  return qdq_selectors_and_actions;
}

}  // namespace

QDQSelectorActionTransformer::QDQSelectorActionTransformer()
    : SelectorActionTransformer{
          "QDQSelectorActionTransformer",
          CreateSelectorsAndActions()} {
}

}  // namespace onnxruntime
