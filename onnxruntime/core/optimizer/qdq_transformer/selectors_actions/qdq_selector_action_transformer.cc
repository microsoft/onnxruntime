// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/qdq_transformer/selectors_actions/qdq_selector_action_transformer.h"
#include <memory>
#include "core/mlas/inc/mlas.h"

#include "core/optimizer/qdq_transformer/selectors_actions/qdq_actions.h"
#if !defined(ORT_MINIMAL_BUILD)
#include "core/optimizer/qdq_transformer/selectors_actions/qdq_selectors.h"
#endif

namespace onnxruntime {

namespace {

using NTO = onnxruntime::NodesToOptimize;

void SplitQDQRules(SelectorActionRegistry& qdq_selector_action_registry) {
  const std::string action_name{"dropSplitQDQ"};
  std::unique_ptr<Action> action = std::make_unique<QDQ::SplitReplaceWithQuant>();
#if !defined(ORT_MINIMAL_BUILD)
  std::unique_ptr<NodeSelector> selector = std::make_unique<QDQ::SplitSelector>(true /*req_equal_quant_params*/);
  qdq_selector_action_registry.RegisterSelectorAndAction(action_name,
                                                         {{"Split", {}}},
                                                         std::move(selector),
                                                         std::move(action));
#else
  qdq_selector_action_registry.RegisterAction(action_name, std::move(action));
#endif
}

// create rules for ops that don't change the data
void DropQDQNodesRules(SelectorActionRegistry& qdq_selector_action_registry) {
  // 3 nodes. DQ, target, Q. Merge into target and remove DQ and Q.
  const std::string drop_action_name{"drop"};
  const std::string drop_action_no_int16_name{"drop_no_int16_support"};
  NTO::NodeLocation dq{NTO::NodeType::kInput, 0};
  NTO::NodeLocation q{NTO::NodeType::kOutput, 0};

  // Move DQ input 0 to target input 0.
  // Move Q output 0 to target output 0.
  std::vector<NodeAndMoveInfo> moves{
      MoveToSlot(dq, ArgType::kInput, 0, ArgType::kInput, 0),
      MoveToSlot(q, ArgType::kOutput, 0, ArgType::kOutput, 0)};

  std::unique_ptr<Action> drop_action_no_int16 = std::make_unique<MergeIntoTarget>(
      std::vector<NodeAndMoveInfo>(moves));  // Copy before std::move(moves)
  std::unique_ptr<Action> drop_action = std::make_unique<MergeIntoTarget>(std::move(moves));

#if !defined(ORT_MINIMAL_BUILD)
  // Use a separate selector + action that disallows 16-bit types for MaxPool and Resize.
  // int16 MaxPool is not supported by the ONNX specification.
  // int16 Resize is not supported by the ORT implementation (although allowed by ONNX).
  std::unique_ptr<NodeSelector> selector_disallow_16bit = std::make_unique<QDQ::DropQDQNodesSelector>(false);
  qdq_selector_action_registry.RegisterSelectorAndAction(drop_action_no_int16_name,
                                                         {{"MaxPool", {12}},
                                                          {"Resize", {}}},
                                                         std::move(selector_disallow_16bit),
                                                         std::move(drop_action_no_int16));

  std::unique_ptr<NodeSelector> selector = std::make_unique<QDQ::DropQDQNodesSelector>(true);
  qdq_selector_action_registry.RegisterSelectorAndAction(drop_action_name,
                                                         {{"Gather", {}},
                                                          {"Reshape", {}},
                                                          {"Transpose", {}},
                                                          {"Squeeze", {}},
                                                          {"Unsqueeze", {}}},
                                                         std::move(selector),
                                                         std::move(drop_action));
#else
  qdq_selector_action_registry.RegisterAction(drop_action_no_int16_name, std::move(drop_action_no_int16));
  qdq_selector_action_registry.RegisterAction(drop_action_name, std::move(drop_action));
#endif
}

// create rules for ops that don't change the data
void DropDQNodesRules(SelectorActionRegistry& qdq_selector_action_registry) {
  // 2 nodes. DQ, target. Merge into target and remove DQ.
  const std::string action_name{"dropDQ"};
  NTO::NodeLocation dq{NTO::NodeType::kInput, 0};

  // Move DQ input 0 to target input 0.
  std::vector<NodeAndMoveInfo> moves{
      MoveToSlot(dq, ArgType::kInput, 0, ArgType::kInput, 0)};

  std::unique_ptr<Action> action = std::make_unique<MergeIntoTarget>(std::move(moves));

#if !defined(ORT_MINIMAL_BUILD)
  // TODO: Enable 16-bit types in selector when ArgMax supports 16-bit integer input tensors.
  std::unique_ptr<NodeSelector> selector = std::make_unique<QDQ::DropDQNodesSelector>();
  qdq_selector_action_registry.RegisterSelectorAndAction(action_name,
                                                         {{"ArgMax", {}}},
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
  std::vector<const char*> providers = {kCpuExecutionProvider};
  std::unique_ptr<NodeSelector> selector = std::make_unique<QDQ::UnarySelector>(providers);
  qdq_selector_action_registry.RegisterSelectorAndAction(action_name,
                                                         {{"AveragePool", {}},
                                                          {"LeakyRelu", {}},
                                                          {"GlobalAveragePool", {}},
                                                          {"Sigmoid", {}},
                                                          {"Softmax", {}}},
                                                         std::move(selector),
                                                         std::move(action));
#else
  qdq_selector_action_registry.RegisterAction(action_name, std::move(action));
#endif
}

void BinaryOpQDQRules(SelectorActionRegistry& qdq_selector_action_registry) {
  // 4 nodes. 2 x DQ for inputs, target, Q
  // Replace with internal QLinear version of operator. Delete all original nodes.
  {
    const std::string action_name{"2DQ"};
    std::unique_ptr<Action> action = std::make_unique<QDQ::BinaryReplaceWithQLinear>(kMSDomain);

#if !defined(ORT_MINIMAL_BUILD)
    // TODO: Enable 16-bit types in selector when binary QLinear* ops support 16-bit.
    std::vector<const char*> providers = {kCpuExecutionProvider};
    std::unique_ptr<NodeSelector> selector = std::make_unique<QDQ::BinarySelector>(providers);
    qdq_selector_action_registry.RegisterSelectorAndAction(action_name,
                                                           {{"Add", {}},
                                                            {"Mul", {}}},
                                                           std::move(selector),
                                                           std::move(action));

#else
    qdq_selector_action_registry.RegisterAction(action_name, std::move(action));
#endif
  }

#ifdef USE_DML
  {
    const std::string action_name{"2DQ_DML"};
    std::unique_ptr<Action> action = std::make_unique<QDQ::BinaryReplaceWithQLinear>(kMSDomain);

#if !defined(ORT_MINIMAL_BUILD)
    std::vector<const char*> providers = {kDmlExecutionProvider};
    std::unique_ptr<NodeSelector> selector = std::make_unique<QDQ::BinarySelector>(providers);

    qdq_selector_action_registry.RegisterSelectorAndAction(action_name,
                                                           {{"Add", {}}},
                                                           std::move(selector),
                                                           std::move(action));

#else
#error "ORT_MINIMAL_BUILD and USE_DML are not expected simultaneously. This would require RegisterAction to be called here."
#endif
  }
#endif
}

void VariadicOpQDQRules(SelectorActionRegistry& qdq_selector_action_registry) {
  // 0=variadic DQ nodes 2=target, 3=Q
  // Replace with QLinear version of operator. Delete all original nodes.
  const std::string action_name{"*DQ"};
  std::unique_ptr<Action> action = std::make_unique<QDQ::VariadicReplaceWithQLinear>(kMSDomain);

#if !defined(ORT_MINIMAL_BUILD)
  // TODO: Enable 16-bit types in selector when QLinearConcat supports 16-bit.
  std::unique_ptr<NodeSelector> selector = std::make_unique<QDQ::InputVariadicSelector>();

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
  // TODO: Enable 16-bit types in selector when QLinearConv supports 16-bit.
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
  // TODO: Enable 16-bit types in selector when QLinearMatMul and MatMulInteger support 16-bit.
  std::vector<const char*> providers = {kCpuExecutionProvider};
  std::unique_ptr<NodeSelector> selector = std::make_unique<QDQ::MatMulSelector>(providers, is_int8_allowed);
  qdq_selector_action_registry.RegisterSelectorAndAction(action_name,
                                                         {{"MatMul", {}}},
                                                         std::move(selector),
                                                         std::move(action));

#else
  ORT_UNUSED_PARAMETER(is_int8_allowed);
  qdq_selector_action_registry.RegisterAction(action_name, std::move(action));
#endif
}

void GemmQDQRules(SelectorActionRegistry& qdq_selector_action_registry) {
  // 3 to 5 nodes. 0=DQ A, 1=DQ B, 2=DQ C(optional), 3=Gemm, 4=Q Y(optional)
  // Replace with QGemm
  // Delete all original nodes.
  const std::string action_name{"Gemm"};

  std::unique_ptr<Action> action = std::make_unique<QDQ::GemmReplaceWithQuant>();

#if !defined(ORT_MINIMAL_BUILD)
  std::vector<const char*> providers = {kCpuExecutionProvider};
  std::unique_ptr<NodeSelector> selector = std::make_unique<QDQ::GemmSelector>(providers);
  qdq_selector_action_registry.RegisterSelectorAndAction(action_name,
                                                         {{"Gemm", {}}},
                                                         std::move(selector),
                                                         std::move(action));

#else
  qdq_selector_action_registry.RegisterAction(action_name, std::move(action));
#endif
}

void WhereQDQRules(SelectorActionRegistry& qdq_selector_action_registry) {
  // 3 nodes.  2 x DQ for inputs and 1X Q for output
  // Compare to other BinaryOperators (Add, Mul), Where also have a special case that it has boolean input
  // Where Replace with QLinearWhere
  // Delete all original nodes.
  const std::string action_name{"Where"};
  std::unique_ptr<Action> action = std::make_unique<QDQ::WhereReplaceWithQLinear>();

#if !defined(ORT_MINIMAL_BUILD)

  std::vector<const char*> providers = {kCpuExecutionProvider};
  std::unique_ptr<NodeSelector> selector = std::make_unique<QDQ::WhereSelector>(providers);
  qdq_selector_action_registry.RegisterSelectorAndAction(action_name,
                                                         {{"Where", {}}},
                                                         std::move(selector),
                                                         std::move(action));

#else
  qdq_selector_action_registry.RegisterAction(action_name, std::move(action));
#endif
}

SelectorActionRegistry CreateSelectorActionRegistry(bool is_int8_allowed) {
  SelectorActionRegistry qdq_selector_action_registry;
  SplitQDQRules(qdq_selector_action_registry);
  DropQDQNodesRules(qdq_selector_action_registry);
  DropDQNodesRules(qdq_selector_action_registry);
  UnaryOpQDQRules(qdq_selector_action_registry);
  BinaryOpQDQRules(qdq_selector_action_registry);
  VariadicOpQDQRules(qdq_selector_action_registry);
  ConvQDQRules(qdq_selector_action_registry, is_int8_allowed);
  MatMulQDQRules(qdq_selector_action_registry, is_int8_allowed);
  GemmQDQRules(qdq_selector_action_registry);
  WhereQDQRules(qdq_selector_action_registry);

  return qdq_selector_action_registry;
}

}  // namespace

QDQSelectorActionTransformer::QDQSelectorActionTransformer(
    bool is_int8_allowed, const SatApplyContextVariant& apply_context)
    : SelectorActionTransformer{
          "QDQSelectorActionTransformer",
          CreateSelectorActionRegistry(is_int8_allowed),
          apply_context,
          // this transformer is only compatible with the CPU and DML EP
          {kCpuExecutionProvider, kDmlExecutionProvider}} {
}

}  // namespace onnxruntime
