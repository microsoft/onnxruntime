// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/matmul_nbits_fusion.h"

#include "core/common/common.h"
#include "core/optimizer/selectors_actions/actions.h"

#if !defined(ORT_MINIMAL_BUILD)
#include "core/graph/graph_utils.h"
#include "core/optimizer/utils.h"
#include "core/framework/tensorprotoutils.h"
#endif

namespace onnxruntime {

namespace {

#if !defined(ORT_MINIMAL_BUILD)

namespace selectors {

class BiasFusion : public NodeSelector {
 public:
  std::optional<NodesToOptimizeIndices> Select(const GraphViewer& graph_viewer,
                                               const Node& node) const override {
    // Prepacked weights force a kernel/layout selected at export time. Some prepacked
    // implementations do not support MatMulNBits bias input 5, so keep the Add separate.
    if (const auto* weight_prepacked = graph_utils::GetNodeAttribute(node, "weight_prepacked");
        weight_prepacked != nullptr && weight_prepacked->i() != 0) {
      return std::nullopt;
    }

    // check if MatMulNBits node already has a bias input
    if (const auto input_defs = node.InputDefs();
        input_defs.size() > 5 && input_defs[5]->Exists()) {
      return std::nullopt;
    }

    if (!optimizer_utils::CheckOutputEdges(graph_viewer.GetGraph(), node, 1)) {
      return std::nullopt;
    }

    const auto edge_to_next_node = node.OutputEdgesBegin();
    const auto& next_node = edge_to_next_node->GetNode();

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "Add", {7, 13, 14})) {
      return std::nullopt;
    }

    if (node.GetExecutionProviderType() != next_node.GetExecutionProviderType()) {
      return std::nullopt;
    }

    // check shape of other Add input
    // at this time, we only support adding a bias with shape [N]

    const auto bias_index = edge_to_next_node->GetDstArgIndex() == 0 ? 1 : 0;
    const NodeArg& bias_arg = *next_node.InputDefs()[bias_index];

    const auto* bias_shape = bias_arg.Shape();
    if (bias_shape == nullptr) {
      return std::nullopt;
    }

    const int64_t N = graph_utils::GetNodeAttribute(node, "N")->i();

    if (bias_shape->dim_size() != 1 ||
        !utils::HasDimValue(bias_shape->dim(0)) ||
        bias_shape->dim(0).dim_value() != N) {
      return std::nullopt;
    }

    NodesToOptimizeIndicesBuilder builder{};
    builder.target_node = node.Index();
    builder.output_nodes = {next_node.Index()};
    return builder.Build();
  }
};

}  // namespace selectors

#endif  // !defined(ORT_MINIMAL_BUILD)

namespace actions {

using NTO = NodesToOptimize;

struct BiasFusion : MergeIntoTarget {
 private:
  std::vector<NodeAndMoveInfo> ValueMoves(const RuntimeState& runtime_state) const override {
    const Node& target = runtime_state.selected_nodes.Target();
    ORT_ENFORCE(target.GetOutputEdgesCount() == 1);
    const auto edge_to_next_node = target.OutputEdgesBegin();
    const auto bias_index = edge_to_next_node->GetDstArgIndex() == 0 ? 1 : 0;

    NTO::NodeLocation add_location{NTO::NodeType::kOutput, 0};

    std::vector<NodeAndMoveInfo> value_moves{
        MoveToSlot(add_location, ArgType::kInput, bias_index, ArgType::kInput, 5),  // move bias input from Add
        MoveToSlot(add_location, ArgType::kOutput, 0, ArgType::kOutput, 0),         // move output from Add
    };

    return value_moves;
  }
};

}  // namespace actions

void BiasFusionRule(SelectorActionRegistry& registry) {
  constexpr const char* name = "FuseBias";

  auto action = std::make_unique<actions::BiasFusion>();

#if !defined(ORT_MINIMAL_BUILD)

  auto selector = std::make_unique<selectors::BiasFusion>();

  registry.RegisterSelectorAndAction(name,
                                     {{SelectorActionRegistry::OpVersionsMapKey("MatMulNBits", kMSDomain), {}}},
                                     std::move(selector),
                                     std::move(action));

#else

  registry.RegisterAction(name, std::move(action));

#endif
}

}  // namespace

SelectorActionRegistry MatMulNBitsFusion::CreateSelectorActionRegistry() const {
  SelectorActionRegistry registry{};

  BiasFusionRule(registry);

  return registry;
}

MatMulNBitsFusion::MatMulNBitsFusion(const InlinedHashSet<std::string_view>& compatible_eps,
                                     const SatApplyContextVariant& apply_context)
    : SelectorActionTransformer{"MatMulNBitsFusion",
                                CreateSelectorActionRegistry(),
                                apply_context,
                                compatible_eps} {
}

#if !defined(ORT_MINIMAL_BUILD)
namespace {

bool IsSingleUseIntermediate(const Graph& graph, const Node& node) {
  return node.GetOutputEdgesCount() == 1 && !graph.NodeProducesGraphOutput(node);
}

bool IsEmptyLoraDefault(const Graph& graph, const NodeArg* input,
                        int64_t first, int64_t second, int32_t data_type) {
  if (!graph_utils::IsGraphInput(graph, input)) {
    return false;
  }
  const auto* tensor = graph.GetInitializer(input->Name(), true);
  return tensor && tensor->data_type() == data_type &&
         tensor->dims_size() == 2 && tensor->dims(0) == first &&
         tensor->dims(1) == second && tensor->external_data_size() == 0 &&
         tensor->data_location() != ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL &&
         tensor->raw_data().empty() && tensor->float_data_size() == 0 &&
         tensor->int32_data_size() == 0;
}

struct QuantizedLoraWeight {
  Node* dequantize{};
  Node* cast{};
  NodeArg* weights{};
  NodeArg* scales{};
};

bool MatchQuantizedLoraWeight(Graph& graph, NodeArg* value,
                              const Node& root, int32_t compute_type,
                              QuantizedLoraWeight& result) {
  Node* node = graph.GetMutableProducerNode(value->Name());
  if (compute_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
    if (!node || !graph_utils::IsSupportedOptypeVersionAndDomain(*node, "Cast", {9, 13, 19, 21}) ||
        !IsSingleUseIntermediate(graph, *node) || node->InputDefs().size() != 1 ||
        node->GetExecutionProviderType() != root.GetExecutionProviderType()) {
      return false;
    }
    const auto* to = graph_utils::GetNodeAttribute(*node, "to");
    if (!to || to->i() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
      return false;
    }
    result.cast = node;
    node = graph.GetMutableProducerNode(node->InputDefs()[0]->Name());
  }
  if (!node ||
      !graph_utils::IsSupportedOptypeVersionAndDomain(*node, "DequantizeLinear", {21, 23, 24}) ||
      !IsSingleUseIntermediate(graph, *node) ||
      node->GetExecutionProviderType() != root.GetExecutionProviderType() ||
      node->InputDefs().size() < 2 || node->InputDefs().size() > 3 ||
      (node->InputDefs().size() == 3 && node->InputDefs()[2]->Exists())) {
    return false;
  }
  const auto* axis = graph_utils::GetNodeAttribute(*node, "axis");
  const auto* block = graph_utils::GetNodeAttribute(*node, "block_size");
  const auto* output_dtype = graph_utils::GetNodeAttribute(*node, "output_dtype");
  if (!axis || axis->i() != 0 || !block || block->i() != 32 ||
      (output_dtype && output_dtype->i() != 0 &&
       output_dtype->i() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT)) {
    return false;
  }
  result.dequantize = node;
  result.weights = node->MutableInputDefs()[0];
  result.scales = node->MutableInputDefs()[1];
  return true;
}

bool FuseLoraUpdate(Graph& graph, Node& root, NodeArg* base_result, Node* lora_a,
                    NodeArg* lora_b_weight, Node* lora_b_node = nullptr) {
  Node* base = graph.GetMutableProducerNode(base_result->Name());
  if (!base || !lora_a || !lora_b_weight ||
      !graph_utils::IsSupportedOptypeVersionAndDomain(*lora_a, "MatMul", {1, 9, 13}) ||
      !IsSingleUseIntermediate(graph, *base) || !IsSingleUseIntermediate(graph, *lora_a) ||
      lora_a->InputDefs().size() != 2 ||
      base->GetExecutionProviderType() != root.GetExecutionProviderType() ||
      lora_a->GetExecutionProviderType() != root.GetExecutionProviderType() ||
      (lora_b_node && (!IsSingleUseIntermediate(graph, *lora_b_node) ||
                       lora_b_node->GetExecutionProviderType() != root.GetExecutionProviderType()))) {
    return false;
  }
  const auto* type = lora_a->InputDefs()[0]->TypeAsProto();
  if (!type || !type->has_tensor_type()) {
    return false;
  }
  const int32_t data_type = type->tensor_type().elem_type();
  QuantizedLoraWeight a, b;
  if ((data_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT &&
       (data_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT16 ||
        root.GetExecutionProviderType() != kWebGpuExecutionProvider)) ||
      !MatchQuantizedLoraWeight(graph, lora_a->MutableInputDefs()[1], root, data_type, a) ||
      !MatchQuantizedLoraWeight(graph, lora_b_weight, root, data_type, b)) {
    return false;
  }
  const auto* a_default = graph.GetInitializer(a.weights->Name(), true);
  const auto* b_default = graph.GetInitializer(b.weights->Name(), true);
  if (!a_default || !b_default || a_default->dims_size() != 2 || b_default->dims_size() != 2) {
    return false;
  }
  const int64_t K = a_default->dims(0);
  const int64_t N = b_default->dims(1);
  if (K <= 0 || N <= 0 ||
      !IsEmptyLoraDefault(graph, a.weights, K, 0, ONNX_NAMESPACE::TensorProto_DataType_INT8) ||
      !IsEmptyLoraDefault(graph, b.weights, 0, N, ONNX_NAMESPACE::TensorProto_DataType_INT8) ||
      !IsEmptyLoraDefault(graph, a.scales, (K - 1) / 32 + 1, 0, ONNX_NAMESPACE::TensorProto_DataType_FLOAT) ||
      !IsEmptyLoraDefault(graph, b.scales, 0, N, ONNX_NAMESPACE::TensorProto_DataType_FLOAT)) {
    return false;
  }
  const auto* input_shape = lora_a->InputDefs()[0]->Shape();
  const auto* base_shape = base_result->Shape();
  if (!input_shape || !base_shape || input_shape->dim_size() == 0 ||
      input_shape->dim_size() != base_shape->dim_size() ||
      !base_shape->dim(base_shape->dim_size() - 1).has_dim_value() ||
      base_shape->dim(base_shape->dim_size() - 1).dim_value() != N) {
    return false;
  }
  // LoraMulAdd does not implement Add/Gemm's broadcasted base input.
  for (int axis = 0; axis + 1 < input_shape->dim_size(); ++axis) {
    const auto& input_dim = input_shape->dim(axis);
    const auto& base_dim = base_shape->dim(axis);
    if (!((input_dim.has_dim_value() && base_dim.has_dim_value() &&
           input_dim.dim_value() == base_dim.dim_value()) ||
          (input_dim.has_dim_param() && base_dim.has_dim_param() &&
           !input_dim.dim_param().empty() && input_dim.dim_param() == base_dim.dim_param()))) {
      return false;
    }
  }
  std::vector<NodeArg*> inputs{base_result, lora_a->MutableInputDefs()[0],
                               a.weights, b.weights, a.scales, b.scales};
  Node& fused = graph.AddNode(
      graph.GenerateNodeName("LoraMulAdd"), "LoraMulAdd",
      "optional quantized low-rank update of an existing base result",
      inputs, root.MutableOutputDefs(), nullptr, kMSDomain);
  fused.SetExecutionProviderType(root.GetExecutionProviderType());
  std::vector<std::reference_wrapper<Node>> nodes{*a.dequantize, *b.dequantize, *lora_a};
  if (a.cast) nodes.push_back(*a.cast);
  if (b.cast) nodes.push_back(*b.cast);
  if (lora_b_node) {
    nodes.push_back(*lora_b_node);
  }
  nodes.push_back(root);
  graph_utils::FinalizeNodeFusion(graph, nodes, fused);
  return true;
}

}  // namespace

Status LoraMulAddFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                   const logging::Logger& logger) const {
  GraphViewer viewer(graph);
  for (auto index : viewer.GetNodesInTopologicalOrder()) {
    Node* root = graph.GetNode(index);
    if (!root) {
      continue;
    }
    ORT_RETURN_IF_ERROR(Recurse(*root, modified, graph_level, logger));
    if (!graph_utils::IsSupportedProvider(*root, GetCompatibleExecutionProviders())) {
      continue;
    }
    if (graph_utils::IsSupportedOptypeVersionAndDomain(*root, "Gemm", {7, 9, 11, 13}) &&
        root->InputDefs().size() == 3 && root->InputDefs()[2]->Exists()) {
      const auto* alpha = graph_utils::GetNodeAttribute(*root, "alpha");
      const auto* beta = graph_utils::GetNodeAttribute(*root, "beta");
      const auto* trans_a = graph_utils::GetNodeAttribute(*root, "transA");
      const auto* trans_b = graph_utils::GetNodeAttribute(*root, "transB");
      if ((!alpha || alpha->f() == 1.0f) && (!beta || beta->f() == 1.0f) &&
          (!trans_a || trans_a->i() == 0) && (!trans_b || trans_b->i() == 0)) {
        if (FuseLoraUpdate(graph, *root,
                           root->MutableInputDefs()[2],
                           graph.GetMutableProducerNode(root->InputDefs()[0]->Name()),
                           root->MutableInputDefs()[1])) {
          modified = true;
        }
      }
      continue;
    }
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(*root, "Add", {7, 13, 14}) ||
        root->InputDefs().size() != 2) {
      continue;
    }
    for (size_t base_index = 0; base_index < 2; ++base_index) {
      Node* lora_b = graph.GetMutableProducerNode(root->InputDefs()[1 - base_index]->Name());
      if (!lora_b ||
          !graph_utils::IsSupportedOptypeVersionAndDomain(*lora_b, "MatMul", {1, 9, 13}) ||
          lora_b->InputDefs().size() != 2) {
        continue;
      }
      Node* lora_a = graph.GetMutableProducerNode(lora_b->InputDefs()[0]->Name());
      if (FuseLoraUpdate(graph, *root, root->MutableInputDefs()[base_index], lora_a,
                         lora_b->MutableInputDefs()[1], lora_b)) {
        modified = true;
        break;
      }
    }
  }
  return Status::OK();
}
#endif

}  // namespace onnxruntime
