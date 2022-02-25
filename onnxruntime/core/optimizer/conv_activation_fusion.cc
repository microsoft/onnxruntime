// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/conv_activation_fusion.h"

#include <string_view>

#include "core/framework/tensorprotoutils.h"
#include "core/framework/inlined_containers.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"
#include "core/optimizer/selectors_actions/actions.h"

namespace onnxruntime {

namespace {

namespace selectors {
const Node* GetLoneConsumerNode(const GraphViewer& graph_viewer, const Node& node) {
  if (graph_viewer.NodeProducesGraphOutput(node) || node.GetOutputEdgesCount() != 1) {
    return nullptr;
  }
  return &*node.OutputNodesBegin();
}

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

class ConvActivation : public NodeSelector {
 public:
  ConvActivation() = default;

  std::optional<NodesToOptimizeIndices> Select(const GraphViewer& graph_viewer, const Node& node) const override {
    const std::string_view node_ep = node.GetExecutionProviderType();
    const auto* next_node = GetLoneConsumerNode(graph_viewer, node);
    if (!next_node ||
        next_node->GetExecutionProviderType() != node_ep) {
      return std::nullopt;
    }

    auto is_supported_non_cuda_ep_activation = [&graph_viewer, &node](const Node& activation_node) {
      if (graph_utils::IsSupportedOptypeVersionAndDomain(activation_node, "Relu", {6, 13, 14}) ||
          graph_utils::IsSupportedOptypeVersionAndDomain(activation_node, "Sigmoid", {6, 13}) ||
          graph_utils::IsSupportedOptypeVersionAndDomain(activation_node, "Tanh", {6, 13}) ||
          graph_utils::IsSupportedOptypeVersionAndDomain(activation_node, "LeakyRelu", {6})) {
        return true;
      }

      if (graph_utils::IsSupportedOptypeVersionAndDomain(activation_node, "Clip", {6, 11, 12, 13})) {
        float min, max;
        if (!optimizer_utils::GetClipConstantMinMax(graph_viewer, node, min, max)) {
          return false;
        }
        return true;
      }

      return false;
    };

    // check EP type and activation
    if (node_ep == kCudaExecutionProvider) {
      if (!HasElementDataType(*node.InputDefs()[0], ONNX_NAMESPACE::TensorProto_DataType_FLOAT)) {
        return std::nullopt;
      }

      if (!graph_utils::IsSupportedOptypeVersionAndDomain(*next_node, "Relu", {6, 13, 14})) {
        return std::nullopt;
      }
    } else if (node_ep.empty() || node_ep == kCpuExecutionProvider) {
      if (!is_supported_non_cuda_ep_activation(*next_node) &&
          !graph_utils::IsSupportedOptypeVersionAndDomain(*next_node, "HardSigmoid", {6})) {
        return std::nullopt;
      }
    } else {
      if (!is_supported_non_cuda_ep_activation(*next_node)) {
        return std::nullopt;
      }
    }

    NodesToOptimizeIndicesBuilder builder{};
    builder.target_node = node.Index();
    builder.output_nodes.push_back(next_node->Index());
    return builder.Build();
  }
};

class ConvAddRelu : public NodeSelector {
 public:
  ConvAddRelu() = default;

  std::optional<NodesToOptimizeIndices> Select(const GraphViewer& graph_viewer, const Node& node) const override {
    const std::string_view node_ep = node.GetExecutionProviderType();
    // only for CUDA EP
    if (node_ep != kCudaExecutionProvider) {
      return std::nullopt;
    }

    const auto* add_node = GetLoneConsumerNode(graph_viewer, node);
    if (!add_node ||
        !graph_utils::IsSupportedOptypeVersionAndDomain(*add_node, "Add", {6, 7, 13, 14}) ||
        add_node->GetExecutionProviderType() != node_ep) {
      return std::nullopt;
    }

    const auto* relu_node = GetLoneConsumerNode(graph_viewer, *add_node);
    if (!relu_node ||
        !graph_utils::IsSupportedOptypeVersionAndDomain(*relu_node, "Relu", {6, 13, 14}) ||
        relu_node->GetExecutionProviderType() != node_ep) {
      return std::nullopt;
    }

    NodesToOptimizeIndicesBuilder builder{};
    builder.target_node = node.Index();
    builder.output_nodes = {add_node->Index(),
                            relu_node->Index()};
    return builder.Build();
  }
};

}  // namespace selectors

namespace actions {
// TODO refactor to lift common logic from Graph::AddAttribute()
void SetStringAttribute(std::string name, std::string value, NodeAttributes& attributes) {
  ONNX_NAMESPACE::AttributeProto a{};
  a.set_name(name);
  a.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_STRING);
  a.set_s(std::move(value));
  attributes.insert_or_assign(std::move(name), std::move(a));
};

void SetFloatsAttribute(std::string name, gsl::span<float> value, NodeAttributes& attributes) {
  ONNX_NAMESPACE::AttributeProto a{};
  a.set_name(name);
  a.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_FLOATS);
  a.mutable_floats()->Assign(value.begin(), value.end());
  attributes.insert_or_assign(std::move(name), std::move(a));
};

using NTO = NodesToOptimize;

class FuseConvActivation : public ReplaceWithNew {
 public:
  FuseConvActivation()
      : ReplaceWithNew{kMSDomain, "FusedConv", CreateValueMoves()} {
  }

 private:
  NodeAttributes Attributes(const RuntimeState& state) const override {
    const auto* activation = state.selected_nodes.Output(0);
    ORT_ENFORCE(activation != nullptr, "Expected activation node.");

    const auto& activation_op_type = activation->OpType();
    NodeAttributes fused_conv_attributes{};
    SetStringAttribute("activation", activation_op_type, fused_conv_attributes);

    std::vector<float> activation_params;
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

    SetFloatsAttribute("activation_params", activation_params, fused_conv_attributes);

    return fused_conv_attributes;
  }

  static std::vector<NodeAndMoveInfo> CreateValueMoves() {
    const NTO::NodeLocation conv{NTO::NodeType::kTarget, 0};
    const NTO::NodeLocation activation{NTO::NodeType::kOutput, 0};

    return {
        MoveAll(conv, ArgType::kInput),         // move all inputs from conv
        MoveAll(activation, ArgType::kOutput),  // move all outputs from activation
    };
  }
};

class FuseConvAddRelu : public ReplaceWithNew {
 public:
  FuseConvAddRelu()
      : ReplaceWithNew{
            kMSDomain, "FusedConv", {}} {
  }

 private:
  NodeAttributes Attributes(const RuntimeState&) const override {
    NodeAttributes fused_conv_attributes{};
    SetStringAttribute("activation", "Relu", fused_conv_attributes);
    return fused_conv_attributes;
  }

  std::vector<NodeAndMoveInfo> ValueMoves(const RuntimeState& state) const override {
    const auto& conv = state.selected_nodes.Target();

    ORT_ENFORCE(conv.GetOutputEdgesCount() == 1 && conv.OutputNodesBegin()->OpType() == "Add",
                "Expected Conv then Add.");
    const auto add_input_idx = 1 - conv.OutputEdgesBegin()->GetDstArgIndex();

    const auto conv_location = NTO::NodeLocation{NTO::NodeType::kTarget, 0};
    const auto add_location = NTO::NodeLocation{NTO::NodeType::kOutput, 0};
    const auto relu_location = NTO::NodeLocation{NTO::NodeType::kOutput, 1};

    return {
        MoveAll(conv_location, ArgType::kInput),                                       // move all inputs from conv
        MoveAndAppend(add_location, ArgType::kInput, add_input_idx, ArgType::kInput),  // append add input
        MoveAll(relu_location, ArgType::kOutput),                                      // move all outputs from relu
    };
  }
};
}  // namespace actions

void RegisterConvActivationFusionRules(SelectorActionRegistry& registry) {
  const auto name = "ConvAct";
  auto action = std::make_unique<actions::FuseConvActivation>();
  auto selector = std::make_unique<selectors::ConvActivation>();
  registry.RegisterSelectorAndAction(name, {{"Conv", {1, 11}}},
                                     std::move(selector), std::move(action));
}

void RegisterConvAddReluFusionRules(SelectorActionRegistry& registry) {
  const auto name = "ConvAddRelu";
  auto action = std::make_unique<actions::FuseConvAddRelu>();
  auto selector = std::make_unique<selectors::ConvAddRelu>();
  registry.RegisterSelectorAndAction(name, {{"Conv", {1, 11}}},
                                     std::move(selector), std::move(action));
}

SelectorActionRegistry CreateSelectorActionRegistry() {
  SelectorActionRegistry registry{};
  RegisterConvActivationFusionRules(registry);
  return registry;
}

}  // namespace

ConvActivationFusion2::ConvActivationFusion2(const std::unordered_set<std::string>& compatible_execution_providers,
                                             const SatApplyContextVariant& apply_context)
    : SelectorActionTransformer{
          "ConvActivationFusion", CreateSelectorActionRegistry(), apply_context, compatible_execution_providers} {
}

Status ConvActivationFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& order = graph_viewer.GetNodesInTopologicalOrder();

  for (auto index : order) {
    auto* node = graph.GetNode(index);
    // check that node hasn't already been removed
    if (!node)
      continue;

    ORT_RETURN_IF_ERROR(Recurse(*node, modified, graph_level, logger));

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(*node, "Conv", {1, 11}) ||
        !graph_utils::IsSupportedProvider(*node, GetCompatibleExecutionProviders()) ||
        node->GetOutputEdgesCount() != 1) {
      continue;
    }

    const auto& next_node = *(node->OutputNodesBegin());

    if (next_node.GetExecutionProviderType() != node->GetExecutionProviderType()) {
      continue;
    }

    if (graph.NodeProducesGraphOutput(*node)) {
      continue;
    }

    if (node->GetExecutionProviderType() == onnxruntime::kCudaExecutionProvider) {
      if (node->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type() !=
          ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
        continue;
      }
      if (graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "Relu", {6, 13, 14})) {
        Node& conv_node = *node;
        Node& act_node = *graph.GetNode(next_node.Index());
        auto node_name = graph.GenerateNodeName(conv_node.Name() + "_" + act_node.Name());
        Node& fused_conv = graph.AddNode(node_name,
                                         "FusedConv",
                                         node_name,
                                         conv_node.MutableInputDefs(),
                                         {},
                                         &conv_node.GetAttributes(),
                                         onnxruntime::kMSDomain);
        fused_conv.SetExecutionProviderType(conv_node.GetExecutionProviderType());
        fused_conv.AddAttribute("activation", "Relu");
        graph_utils::FinalizeNodeFusion(graph, {conv_node, act_node}, fused_conv);
        modified = true;
        //////// this is the code to copy!
      } else if (graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "Add", {6, 7, 13, 14})) {
        if (next_node.GetOutputEdgesCount() != 1) {
          continue;
        }
        const auto& last_node = *(next_node.OutputNodesBegin());
        if (last_node.GetExecutionProviderType() != node->GetExecutionProviderType()) {
          continue;
        }
        if (graph_utils::IsSupportedOptypeVersionAndDomain(last_node, "Relu", {6, 13, 14})) {
          Node& conv_node = *node;
          Node& add_node = *graph.GetNode(next_node.Index());
          Node& act_node = *graph.GetNode(last_node.Index());
          auto conv_inputs = conv_node.MutableInputDefs();
          auto conv_outputs = conv_node.MutableOutputDefs();
          auto add_inputs = add_node.MutableInputDefs();
          int32_t dependent = 0, independent = 0;
          for (auto add_input : add_inputs) {
            if (add_input->Name() == conv_outputs[0]->Name()) {
              dependent++;
            } else {
              conv_inputs.push_back(add_input);
              independent++;
            }
          }
          if (dependent != 1 || independent != 1) {
            continue;
          }
          auto node_name = graph.GenerateNodeName(conv_node.Name() + "_" +
                                                  add_node.Name() + "_" +
                                                  act_node.Name());
          Node& fused_conv = graph.AddNode(node_name,
                                           "FusedConv",
                                           node_name,
                                           conv_inputs,
                                           {}, &conv_node.GetAttributes(),
                                           onnxruntime::kMSDomain);
          fused_conv.SetExecutionProviderType(conv_node.GetExecutionProviderType());
          fused_conv.AddAttribute("activation", "Relu");
          graph_utils::FinalizeNodeFusion(graph, {conv_node, add_node, act_node}, fused_conv);
          modified = true;
        }
      }
    } else {
      // Test if this is an activation that can be fused and also extract the
      // activation's parameters.
      InlinedVector<float> activation_params;
      if (!graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "Relu", {6, 13, 14}) &&
          !graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "Sigmoid", {6, 13}) &&
          !graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "Tanh", {6, 13})) {
        if (graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "LeakyRelu", {6})) {
          activation_params.push_back(graph_utils::GetNodeAttribute(next_node, "alpha")->f());
        } else if (graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "Clip", {6, 11, 12, 13})) {
          float min, max;
          if (optimizer_utils::GetClipConstantMinMax(graph, next_node, min, max)) {
            activation_params.push_back(min);
            activation_params.push_back(max);
          } else {
            continue;
          }
        } else if ((node->GetExecutionProviderType().empty() || node->GetExecutionProviderType() == onnxruntime::kCpuExecutionProvider) &&
                   graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "HardSigmoid", {6})) {
          auto* alpha_attr = graph_utils::GetNodeAttribute(next_node, "alpha");
          auto* beta_attr = graph_utils::GetNodeAttribute(next_node, "beta");
          float alpha = (alpha_attr == nullptr ? 0.2f : alpha_attr->f());
          float beta = (beta_attr == nullptr ? 0.5f : beta_attr->f());
          activation_params.push_back(alpha);
          activation_params.push_back(beta);
        } else {
          continue;
        }
      }

      Node& conv_node = *node;
      Node& act_node = *graph.GetNode(next_node.Index());

      Node& fused_conv = graph.AddNode(graph.GenerateNodeName("fused " + conv_node.Name()), "FusedConv",
                                       "fused Conv " + conv_node.Name() + "with activation " + act_node.OpType(),
                                       conv_node.MutableInputDefs(),
                                       {},
                                       &conv_node.GetAttributes(),
                                       "com.microsoft");

      // Assign provider to this new node. Provider should be same as the provider for old node.
      fused_conv.SetExecutionProviderType(conv_node.GetExecutionProviderType());

      // Add attributes to specify the activation type and parameters.
      fused_conv.AddAttribute("activation", next_node.OpType());
      if (!activation_params.empty()) {
        fused_conv.AddAttribute("activation_params", activation_params);
      }

      // move output definitions and edges from act_node to fused_conv. delete conv_node and act_node.
      graph_utils::FinalizeNodeFusion(graph, {conv_node, act_node}, fused_conv);

      modified = true;
    }
  }

  return Status::OK();
}
}  // namespace onnxruntime
