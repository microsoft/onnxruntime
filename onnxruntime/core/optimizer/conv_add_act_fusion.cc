// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <deque>
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/conv_add_act_fusion.h"
#include "core/mlas/inc/mlas.h"
#include "core/graph/node_attr_utils.h"
#include "core/optimizer/utils.h"
#include "core/optimizer/conv_activation_action_base.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {
namespace {

namespace selectors {
bool HasElementDataType(const NodeArg& node_arg, int32_t data_type) {
  if (!node_arg.Exists()) {
    return false;
  }

  const auto* type_proto = node_arg.TypeAsProto();
  if (!type_proto) {
    return false;
  }

  int32_t actual_data_type;
  if (!utils::TryGetElementDataType(*type_proto, actual_data_type)) {
    return false;
  }

  return data_type == actual_data_type;
}

const Node* GetLoneConsumerNode(const GraphViewer& graph_viewer, const Node& node) {
  if (!optimizer_utils::CheckOutputEdges(graph_viewer.GetGraph(), node, 1)) {
    return nullptr;
  }
  return &*node.OutputNodesBegin();
}

class ConvAddActivationSelector : public NodeSelector {
 public:
  ConvAddActivationSelector() = default;
  std::optional<NodesToOptimizeIndices> Select(const GraphViewer& graph_viewer, const Node& node) const override {
    const std::string_view node_ep = node.GetExecutionProviderType();
#ifdef MLAS_F16VEC_INTRINSICS_SUPPORTED
    if (node_ep != kCpuExecutionProvider ||
        (!HasElementDataType(*node.InputDefs()[0], ONNX_NAMESPACE::TensorProto_DataType_FLOAT) &&
         !HasElementDataType(*node.InputDefs()[0], ONNX_NAMESPACE::TensorProto_DataType_FLOAT16))) {
      return std::nullopt;
    }
#else
    if (node_ep != kCpuExecutionProvider ||
        !HasElementDataType(*node.InputDefs()[0], ONNX_NAMESPACE::TensorProto_DataType_FLOAT)) {
      return std::nullopt;
    }
#endif  // MLAS_F16VEC_INTRINSICS_SUPPORTED
    // we can't assign `conv_node` as the producer-node, even it is, because we have to make sure
    // 1. Its type is 'conv', 2. it has to satisfy the other requirements,like shape, please refer to SelectConvProducer for more info
    const Node* conv_node = nullptr;
    const auto* add_node = GetLoneConsumerNode(graph_viewer, node);
    if (add_node == nullptr) {
      return std::nullopt;
    }
    // Let's support addition first, leave any-element-wise-op fusion in the future.
    // what we want to here is that:
    // 1 find the Add node, 2 find it's producer node and make sure it's a conv node
    // 3 find the next node and check if it's a activation node, if yes, we will fuse conv+add+activation or conv+add
    //
    if (graph_utils::IsSupportedOptypeVersionAndDomain(*add_node, "Add", {7, 13, 14})) {
      conv_node = SelectProducerConv(*add_node);
    }
    if (conv_node == nullptr) {
      return std::nullopt;
    }
    // GetLoneConsumerNode will ensure outputedge_count is 1
    const auto* act_node = GetLoneConsumerNode(graph_viewer, *add_node);
    // even the next node is not a activation node, it's also fine.
    if (act_node == nullptr) {
      // we can't fuse add-activation when add_node has multiple consumer nodes
      act_node = nullptr;
    } else if (SelectActivation(graph_viewer, *act_node)) {
      // this branch is deliberately empty as we want to keep 'act_node' as remains.
    } else {
      act_node = nullptr;
    }

    NodesToOptimizeIndicesBuilder builder{};
    builder.target_node = conv_node->Index();
    builder.output_nodes = {add_node->Index()};
    if (act_node != nullptr) {
      builder.output_nodes.push_back(act_node->Index());
    }
    return builder.Build();
  }

  static bool SelectActivation(const GraphViewer& graph_viewer, const Node& activation_node) {
    auto is_supported_cpu_ep_activation = [&graph_viewer](const Node& activation_node) {
      if (graph_utils::IsSupportedOptypeVersionAndDomain(activation_node, "Relu", {6, 13, 14}) ||
          graph_utils::IsSupportedOptypeVersionAndDomain(activation_node, "Sigmoid", {6, 13}) ||
          graph_utils::IsSupportedOptypeVersionAndDomain(activation_node, "Tanh", {6, 13}) ||
          graph_utils::IsSupportedOptypeVersionAndDomain(activation_node, "LeakyRelu", {6, 16})) {
        return true;
      }

      if (graph_utils::IsSupportedOptypeVersionAndDomain(activation_node, "Clip", {6, 11, 12, 13})) {
        float min, max;
        if (!optimizer_utils::GetClipConstantMinMax(graph_viewer.GetGraph(), activation_node, min, max)) {
          return false;
        }
        return true;
      }

      if (graph_utils::IsSupportedOptypeVersionAndDomain(activation_node, "HardSigmoid", {6})) {
        return true;
      }
      return false;
    };
    return is_supported_cpu_ep_activation(activation_node);
  }

  const Node* SelectProducerConv(const Node& node) const {
    InlinedVector<const Node*> inputs_node;
    constexpr int32_t kTensorDims = 4;  // NCHW
    const auto& input_defs = node.InputDefs();

    for (auto producer_node_ptr = node.InputNodesBegin(); producer_node_ptr != node.InputNodesEnd(); ++producer_node_ptr) {
      const Node* producer_node = dynamic_cast<const Node*>(&(*producer_node_ptr));
      inputs_node.push_back(producer_node);
    }
    size_t input_defs_count = input_defs.size();
    if (input_defs_count != 2 || inputs_node.size() > input_defs_count) {
      return nullptr;
    }
    // Test if all of inputs have an equal shape.
    auto* input_0_shape = input_defs[0]->Shape();
    // Check if ONNX shape inferencing has computed a precise dimension value.
    if ((input_0_shape == nullptr) || (input_0_shape->dim_size() != kTensorDims)) {
      return nullptr;
    }
    for (int i = 0; i < kTensorDims; i++) {
      auto& input_0_dim = input_0_shape->dim(i);
      // even though zero-dim is valid, but we don't support here
      if (!utils::HasDimValue(input_0_dim) || (input_0_dim.dim_value() == 0)) {
        if (!utils::HasDimParam(input_0_dim)) {
          return nullptr;
        }
      }
    }
    // we can't fuse them if shape is not matched, it will happens when broadcast-Add
    for (size_t n = 1; n < input_defs_count; n++) {
      auto* input_n_shape = input_defs[n]->Shape();
      if (input_n_shape == nullptr || (input_n_shape->dim_size() != kTensorDims)) {
        return nullptr;
      }
      for (int i = 0; i < kTensorDims; i++) {
        auto& input_0_dim = input_0_shape->dim(i);
        auto& input_n_dim = input_n_shape->dim(i);
        if (!utils::HasDimValue(input_n_dim) || (input_0_dim.dim_value() != input_n_dim.dim_value())) {
          if (!utils::HasDimParam(input_0_dim) || !utils::HasDimParam(input_n_dim) || (input_0_dim.dim_param() != input_n_dim.dim_param())) {
            return nullptr;
          }
        }
      }
    }

    // If one of the inputs to the Add node is a convolution, then
    // attempt to fuse the addition into the convolution itself.
    for (size_t n = 0; (n < inputs_node.size()) && inputs_node[n]; n++) {
      const auto& producer_input_defs = inputs_node[n]->InputDefs();
      const auto& producer_input_args_count = inputs_node[n]->InputArgCount();
      size_t pre_input_defs_count = producer_input_defs.size();
      // Check if this is a single use convolution that hasn't already
      // been fused with another Add/Sum node. The Add/Sum can also only be
      // fused if the convolution isn't itself fused with an activation.
      if ((inputs_node[n]->OpType() == "Conv") && (pre_input_defs_count < 4) &&
          (producer_input_args_count.size() < 4) &&
          (graph_utils::GetNodeAttribute(*inputs_node[n], "activation") == nullptr) &&
          (inputs_node[n]->GetOutputEdgesCount() == 1)) {
        if (pre_input_defs_count < 3) {  // The optional bias parameter is empty so set to an empty string.
          // TODO, add a new null arguments for bias
          continue;
        }
        return inputs_node[n];
      }
      if (inputs_node[n]->OpType() == "NhwcFusedConv" && (pre_input_defs_count < 4) &&
          (producer_input_args_count.size() < 5) &&
          (graph_utils::GetNodeAttribute(*inputs_node[n], "activation") == nullptr) &&
          (inputs_node[n]->GetOutputEdgesCount() == 1)) {
        if (pre_input_defs_count < 3) {  // The optional bias parameter is empty so set to an empty string.
          // TODO, add a new null arguments for bias
          continue;
        }
        return inputs_node[n];
      }
    }
    return nullptr;
  }
};

}  // namespace selectors

namespace actions {
using NTO = NodesToOptimize;

class FuseConvAddActivationAction : public FusedConvActivationActionBase {
 private:
  NodeAttributes ExtraAttributes(const RuntimeState& state) const override {
    NodeAttributes extra_fused_conv_attributes;

    const auto* activation = state.selected_nodes.Output(state.selected_nodes.num_outputs - 1);
    if (state.selected_nodes.num_outputs == 1 || activation->OpType() == "Add") {
      // activation node is the last node in conv+add+activation fusion pattern, while conv+add is also possible
      return extra_fused_conv_attributes;
    }
    ORT_ENFORCE(activation != nullptr, "Expected activation node.");

    const auto& activation_op_type = activation->OpType();
    utils::SetNodeAttribute(utils::MakeAttribute("activation", activation_op_type), extra_fused_conv_attributes);

    InlinedVector<float> activation_params;
    if (activation_op_type == "LeakyRelu") {
      activation_params.push_back(graph_utils::GetNodeAttribute(*activation, "alpha")->f());
    } else if (activation_op_type == "Clip") {
      float min, max;
      ORT_ENFORCE(optimizer_utils::GetClipConstantMinMax(state.graph, *activation, min, max),
                  "Failed to get Clip min/max constants.");
      activation_params.push_back(min);
      activation_params.push_back(max);
    } else if (activation_op_type == "HardSigmoid") {
      auto* alpha_attr = graph_utils::GetNodeAttribute(*activation, "alpha");
      auto* beta_attr = graph_utils::GetNodeAttribute(*activation, "beta");
      float alpha = (alpha_attr == nullptr ? 0.2f : alpha_attr->f());
      float beta = (beta_attr == nullptr ? 0.5f : beta_attr->f());
      activation_params.push_back(alpha);
      activation_params.push_back(beta);
    }

    if (!activation_params.empty()) {
      utils::SetNodeAttribute(utils::MakeAttribute("activation_params", activation_params),
                              extra_fused_conv_attributes);
    }

    return extra_fused_conv_attributes;
  }

  std::vector<NodeAndMoveInfo> ValueMoves(const RuntimeState& state) const override {
    const auto& conv = state.selected_nodes.Target();
    ORT_ENFORCE(conv.GetOutputEdgesCount() == 1 && conv.OutputNodesBegin()->OpType() == "Add",
                "Expected Conv then Add.");

    const auto add_input_idx = 1 - conv.OutputEdgesBegin()->GetDstArgIndex();

    const auto conv_location = NTO::NodeLocation{NTO::NodeType::kTarget, 0};
    const auto add_location = NTO::NodeLocation{NTO::NodeType::kOutput, 0};
    const auto activation_location = NTO::NodeLocation{NTO::NodeType::kOutput, 1};
    // Conv+add+activation
    if (state.selected_nodes.num_outputs == 2) {
      return {
          MoveAll(conv_location, ArgType::kInput),                                       // move all inputs from conv
          MoveAndAppend(add_location, ArgType::kInput, add_input_idx, ArgType::kInput),  // append add input
          MoveAll(activation_location, ArgType::kOutput),                                // move all outputs from relu
      };
    } else {
      // Conv+Add only
      return {
          MoveAll(conv_location, ArgType::kInput),                                       // move all inputs from conv
          MoveAndAppend(add_location, ArgType::kInput, add_input_idx, ArgType::kInput),  // append add input
          MoveAll(add_location, ArgType::kOutput),                                       // move all outputs from relu
      };
    }
  }
};
}  // namespace actions

void RegisterConvAddActivationFusionRules(SelectorActionRegistry& registry) {
  auto action = std::make_unique<actions::FuseConvAddActivationAction>();
  auto selector = std::make_unique<selectors::ConvAddActivationSelector>();
  const std::string kMSDomainNhwcFusedConv = std::string(kMSDomain) + ":NhwcConv";

  registry.RegisterSelectorAndAction("ConvAddAct",
                                     {{"Conv", {1, 11}},
                                      {kMSDomainNhwcFusedConv, {1}}},
                                     std::move(selector), std::move(action));
}

SelectorActionRegistry CreateSelectorActionRegistry() {
  SelectorActionRegistry registry{};
  RegisterConvAddActivationFusionRules(registry);
  return registry;
}

}  // namespace
ConvAddActivationFusion::ConvAddActivationFusion(const InlinedHashSet<std::string_view>& compatible_execution_providers,
                                                 const SatApplyContextVariant& apply_context)
    : SelectorActionTransformer{
          "ConvAddActivationFusion", CreateSelectorActionRegistry(), apply_context, compatible_execution_providers} {
}
}  // namespace onnxruntime
