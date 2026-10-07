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
         tensor->raw_data().empty() && tensor->float_data_size() == 0 &&
         tensor->int32_data_size() == 0;
}

bool FuseLoraUpdate(Graph& graph, Node& root, Node* base, Node* lora_a,
                    NodeArg* lora_b_weight, Node* lora_b_node = nullptr) {
  if (!base || !lora_a || !lora_b_weight ||
      !graph_utils::IsSupportedOptypeVersionAndDomain(*base, "MatMulNBits", {1}, kMSDomain) ||
      !graph_utils::IsSupportedOptypeVersionAndDomain(*lora_a, "MatMul", {1, 9, 13}) ||
      !IsSingleUseIntermediate(graph, *base) || !IsSingleUseIntermediate(graph, *lora_a) ||
      base->InputDefs().size() > 6 || lora_a->InputDefs().size() != 2 ||
      base->InputDefs()[0] != lora_a->InputDefs()[0] ||
      base->GetExecutionProviderType() != root.GetExecutionProviderType() ||
      lora_a->GetExecutionProviderType() != root.GetExecutionProviderType() ||
      (lora_b_node && (!IsSingleUseIntermediate(graph, *lora_b_node) ||
                       lora_b_node->GetExecutionProviderType() != root.GetExecutionProviderType()))) {
    return false;
  }
  const auto* type = base->InputDefs()[0]->TypeAsProto();
  const auto* k = graph_utils::GetNodeAttribute(*base, "K");
  const auto* n = graph_utils::GetNodeAttribute(*base, "N");
  const auto* prepacked = graph_utils::GetNodeAttribute(*base, "weight_prepacked");
  if (!type || !type->has_tensor_type() ||
      !k || !n || k->i() <= 0 || n->i() <= 0 || (prepacked && prepacked->i() != 0)) {
    return false;
  }
  const int32_t data_type = type->tensor_type().elem_type();
  if ((data_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT &&
       (data_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT16 ||
        root.GetExecutionProviderType() != kWebGpuExecutionProvider)) ||
      !IsEmptyLoraDefault(graph, lora_a->InputDefs()[1], k->i(), 0, data_type) ||
      !IsEmptyLoraDefault(graph, lora_b_weight, 0, n->i(), data_type)) {
    return false;
  }
  auto inputs = base->MutableInputDefs();
  while (inputs.size() < 6) {
    inputs.push_back(&graph.GetOrCreateNodeArg("", nullptr));
  }
  inputs.push_back(lora_a->MutableInputDefs()[1]);
  inputs.push_back(lora_b_weight);
  const auto attributes = base->GetAttributes();
  Node& fused = graph.AddNode(
      graph.GenerateNodeName("MatMulNBitsLora"), "MatMulNBitsLora",
      "quantized projection with a runtime-selectable low-rank update",
      inputs, root.MutableOutputDefs(), &attributes, kMSDomain);
  fused.SetExecutionProviderType(root.GetExecutionProviderType());
  if (lora_b_node) {
    graph_utils::FinalizeNodeFusion(graph, {*base, *lora_a, *lora_b_node, root}, fused);
  } else {
    graph_utils::FinalizeNodeFusion(graph, {*base, *lora_a, root}, fused);
  }
  return true;
}

}  // namespace

Status MatMulNBitsLoraFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level,
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
                           graph.GetMutableProducerNode(root->InputDefs()[2]->Name()),
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
      Node* base = graph.GetMutableProducerNode(root->InputDefs()[base_index]->Name());
      Node* lora_b = graph.GetMutableProducerNode(root->InputDefs()[1 - base_index]->Name());
      if (!lora_b ||
          !graph_utils::IsSupportedOptypeVersionAndDomain(*lora_b, "MatMul", {1, 9, 13}) ||
          lora_b->InputDefs().size() != 2) {
        continue;
      }
      Node* lora_a = graph.GetMutableProducerNode(lora_b->InputDefs()[0]->Name());
      if (FuseLoraUpdate(graph, *root, base, lora_a, lora_b->MutableInputDefs()[1], lora_b)) {
        modified = true;
        break;
      }
    }
  }
  return Status::OK();
}
#endif

}  // namespace onnxruntime
