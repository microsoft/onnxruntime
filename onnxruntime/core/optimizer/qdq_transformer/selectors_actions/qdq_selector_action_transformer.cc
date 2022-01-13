// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/qdq_transformer/selectors_actions/qdq_selector_action_transformer.h"
#include "core/mlas/inc/mlas.h"

#include "core/optimizer/qdq_transformer/selectors_actions/qdq_actions.h"
#if !defined(ORT_MINIMAL_BUILD)
#include "core/optimizer/qdq_transformer/selectors_actions/qdq_selectors.h"
#endif

namespace onnxruntime {

namespace {

using NTO = onnxruntime::NodesToOptimize;

// create rules for ops that don't change the data
void DropQDQNodesRules(SelectorActionRegistry& qdq_selector_action_registry) {
  // 3 nodes. DQ, target, Q. Merge into target and remove DQ and Q.
  const std::string action_name{"drop"};
  NTO::NodeLocation dq{NTO::NodeType::kInput, 0};
  NTO::NodeLocation q{NTO::NodeType::kOutput, 0};

  // Move DQ input 0 to target input 0.
  // Move Q output 0 to target output 0.
  std::vector<NodeAndMoveInfo> moves{
      MoveToSlot(dq, ArgType::kInput, 0, ArgType::kInput, 0),
      MoveToSlot(q, ArgType::kOutput, 0, ArgType::kOutput, 0)};

  std::unique_ptr<Action> action = std::make_unique<MergeIntoTarget>(std::move(moves));

#if !defined(ORT_MINIMAL_BUILD)
  std::unique_ptr<NodeSelector> selector = std::make_unique<QDQ::DropQDQNodesSelector>();
  qdq_selector_action_registry.RegisterSelectorAndAction(action_name,
                                                         {{"Gather", {}},
                                                          {"Reshape", {}},
                                                          {"Transpose", {}},
                                                          {"MaxPool", {12}},
                                                          {"Resize", {}}},
                                                         std::move(selector),
                                                         std::move(action));
#else
  qdq_selector_action_registry.RegisterAction(action_name, std::move(action));
#endif
}

void UnaryOpQDQRules(SelectorActionRegistry& qdq_selector_action_registry) {
  // 3 nodes. DQ, target, Q
  // Replace with internal QLinear version of operator. Delete all original nodes.
  const std::string action_name{"1DQ"};
  std::unique_ptr<Action> action = std::make_unique<QDQ::UnaryReplaceWithQLinear>(kMSDomain);

#if !defined(ORT_MINIMAL_BUILD)
  std::unique_ptr<NodeSelector> selector = std::make_unique<QDQ::UnarySelector>();
  qdq_selector_action_registry.RegisterSelectorAndAction(action_name,
                                                         {{"AveragePool", {}},
                                                          {"LeakyRelu", {}}},
                                                         std::move(selector),
                                                         std::move(action));
#else
  qdq_selector_action_registry.RegisterAction(action_name, std::move(action));
#endif
}

void BinaryOpQDQRules(SelectorActionRegistry& qdq_selector_action_registry) {
  // 4 nodes. 2 x DQ for inputs, target, Q
  // Replace with internal QLinear version of operator. Delete all original nodes.
  const std::string action_name{"2DQ"};
  std::unique_ptr<Action> action = std::make_unique<QDQ::BinaryReplaceWithQLinear>(kMSDomain);

#if !defined(ORT_MINIMAL_BUILD)
  std::unique_ptr<NodeSelector> selector = std::make_unique<QDQ::BinarySelector>();
  qdq_selector_action_registry.RegisterSelectorAndAction(action_name,
                                                         {{"Add", {}},
                                                          {"Mul", {}}},
                                                         std::move(selector),
                                                         std::move(action));

#else
  qdq_selector_action_registry.RegisterAction(action_name, std::move(action));
#endif
}

void VariadicOpQDQRules(SelectorActionRegistry& qdq_selector_action_registry) {
  // 0=variadic DQ nodes 2=target, 3=Q
  // Replace with QLinear version of operator. Delete all original nodes.
  const std::string action_name{"*DQ"};
  std::unique_ptr<Action> action = std::make_unique<QDQ::VariadicReplaceWithQLinear>(kMSDomain);

#if !defined(ORT_MINIMAL_BUILD)
  std::unique_ptr<NodeSelector> selector = std::make_unique<QDQ::VariadicSelector>();

  qdq_selector_action_registry.RegisterSelectorAndAction(action_name,
                                                         {{"Concat", {}}},
                                                         std::move(selector),
                                                         std::move(action));

#else
  qdq_selector_action_registry.RegisterAction(action_name, std::move(action));
#endif
}

void ConvQDQRules(SelectorActionRegistry& qdq_selector_action_registry, bool is_int8_allowed = false) {
  // 4 or 5 Nodes. 0=DQ X, 1=DQ W, 2=DQ B (optional), 3=Conv, 4=Q
  // Handle the DQ input for the Bias being optional.
  // Replace Conv with QLinearConv
  // Delete all original nodes
  const std::string action_name{"Conv"};
  std::unique_ptr<Action> action = std::make_unique<QDQ::ConvReplaceWithQLinear>();

#if !defined(ORT_MINIMAL_BUILD)
  std::unique_ptr<NodeSelector> selector = std::make_unique<QDQ::ConvSelector>(is_int8_allowed);

  qdq_selector_action_registry.RegisterSelectorAndAction(action_name,
                                                         {{"Conv", {}}},
                                                         std::move(selector),
                                                         std::move(action));

#else
  ORT_UNUSED_PARAMETER(is_int8_allowed);
  qdq_selector_action_registry.RegisterAction(action_name, std::move(action));
#endif
}

void MatMulQDQRules(SelectorActionRegistry& qdq_selector_action_registry, bool is_int8_allowed = false) {
  // 3 or 4 nodes. 2 x DQ for inputs, target, optional Q
  // Replace with QLinearMatMul if Q found, or MatMulIntegerToFloat if not.
  // Delete all original nodes.
  const std::string action_name{"MatMul"};

  std::unique_ptr<Action> action = std::make_unique<QDQ::MatMulReplaceWithQLinear>();

#if !defined(ORT_MINIMAL_BUILD)
  std::unique_ptr<NodeSelector> selector = std::make_unique<QDQ::MatMulSelector>(is_int8_allowed);
  qdq_selector_action_registry.RegisterSelectorAndAction(action_name,
                                                         {{"MatMul", {}}},
                                                         std::move(selector),
                                                         std::move(action));

#else
  ORT_UNUSED_PARAMETER(is_int8_allowed);
  qdq_selector_action_registry.RegisterAction(action_name, std::move(action));
#endif
}

SelectorActionRegistry CreateSelectorActionRegistry(bool is_int8_allowed) {
  SelectorActionRegistry qdq_selector_action_registry;

  DropQDQNodesRules(qdq_selector_action_registry);
  UnaryOpQDQRules(qdq_selector_action_registry);
  BinaryOpQDQRules(qdq_selector_action_registry);
  VariadicOpQDQRules(qdq_selector_action_registry);
  ConvQDQRules(qdq_selector_action_registry, is_int8_allowed);
  MatMulQDQRules(qdq_selector_action_registry, is_int8_allowed);

  return qdq_selector_action_registry;
}

}  // namespace

QDQSelectorActionTransformer::QDQSelectorActionTransformer(const SatApplyContextVariant& apply_context)
    : SelectorActionTransformer{
          "QDQSelectorActionTransformer",
          CreateSelectorActionRegistry(QDQIsInt8Allowed()),
          apply_context} {
}

}  // namespace onnxruntime
