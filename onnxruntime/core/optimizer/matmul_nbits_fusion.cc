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

}  // namespace onnxruntime
