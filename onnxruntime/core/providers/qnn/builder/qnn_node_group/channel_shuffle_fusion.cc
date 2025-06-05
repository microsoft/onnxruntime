// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/qnn_node_group/channel_shuffle_fusion.h"

#include <gsl/gsl>
#include <optional>
#include <utility>
#include <string>
#include <array>
#include <memory>
#include <unordered_map>
#include <vector>

#include "core/common/inlined_containers.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_node_group/utils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"

namespace onnxruntime {
namespace qnn {
namespace {

constexpr char kAttrTransposePerm[] = "perm";
constexpr char kOpChannelShuffle[] = "ChannelShuffle";
constexpr char kOpTranspose[] = "Transpose";
constexpr char kOpReshape[] = "Reshape";

using MapNodeToNodeUnit = std::unordered_map<const Node*, const NodeUnit*>;
using MapNodeUnitToGroup = std::unordered_map<const NodeUnit*, const IQnnNodeGroup*>;

std::optional<std::vector<int64_t>> GetTransposePerm(const NodeUnit& transpose) {
  if (transpose.OpType() != kOpTranspose) {
    return std::nullopt;
  }
  NodeAttrHelper helper(transpose.GetNode());
  return helper.Get(kAttrTransposePerm, std::vector<int64_t>());
}

std::vector<int64_t> InvertTransposePerm(gsl::span<const int64_t> perm) {
  const size_t perm_size = perm.size();
  std::vector<int64_t> perm_inverse(perm_size);
  for (size_t i = 0; i < perm_size; ++i) {
    size_t j = gsl::narrow_cast<size_t>(perm[i]);
    perm_inverse[j] = gsl::narrow_cast<int64_t>(i);
  }
  return perm_inverse;
}

bool IsCancelingTransposePermPair(
    std::optional<gsl::span<const int64_t>> perm1,
    std::optional<gsl::span<const int64_t>> perm2) {
  if (!perm1.has_value() || !perm2.has_value()) {
    return false;
  }
  if (perm1->size() != perm2->size()) {
    return false;
  }
  std::vector<int64_t> perm1_inverted_vector = InvertTransposePerm(*perm1);
  auto perm1_inverted = gsl::make_span<const int64_t>(
      perm1_inverted_vector.data(), perm1_inverted_vector.size());
  if (perm1_inverted != perm2.value()) {
    return false;
  }
  return true;
}

/// @brief Match pattern: Transpose -> ChannelShuffle (Reshape -> Transpose -> Reshape) -> Transpose
/// E.g.,: T(perm=[0, 2, 1, 3]) -> R(N, G, C/G, H, W) -> T(perm=[0, 1, 3, 2, 4]) -> R(N, C, H, W) -> T(perm=[0, 2, 1, 3])
/// @param graph_viewer QNN graph viewer.
/// @param transpose_head The first transpose node starting the pattern.
/// @param node_to_node_unit Maps a Node to a NodeUnit.
/// @param node_unit_to_qnn_node_group Maps a NodeUnit to a IQnnNodeGroup.
/// @return The matched pattern as an array of NodeUnits if found, otherwise std::nullopt.
/// @note This is ChannelShuffle with transpose wraps commonly seen ORT post partitioning.
std::optional<std::array<const NodeUnit*, 5>> MatchChannelShufflePattern(
    const GraphViewer& graph_viewer,
    const NodeUnit* transpose_head,
    const MapNodeToNodeUnit& node_to_node_unit,
    const MapNodeUnitToGroup& node_unit_to_qnn_node_group) {
  // Helper function to get a single child of a specific type
  auto GetChildOfType = [&](const NodeUnit& node, std::string_view expect_type) -> const NodeUnit* {
    const std::array<std::string_view, 1> child_op_types{expect_type};
    const NodeUnit* child = GetOnlyChildOfType(
        graph_viewer, node, child_op_types, node_to_node_unit, node_unit_to_qnn_node_group);
    if (child == nullptr) {
      return nullptr;
    }
    if (child->OpType() != expect_type) {
      return nullptr;
    }
    if (child->UnitType() != NodeUnit::Type::SingleNode) {
      return nullptr;
    }
    return child;
  };

  if (transpose_head->OpType() != kOpTranspose) {
    return std::nullopt;
  }
  if (transpose_head->UnitType() != NodeUnit::Type::SingleNode) {
    return std::nullopt;
  }
  const NodeUnit* reshape1 = GetChildOfType(*transpose_head, kOpReshape);
  if (reshape1 == nullptr) {
    return std::nullopt;
  }
  const NodeUnit* transpose = GetChildOfType(*reshape1, kOpTranspose);
  if (transpose == nullptr) {
    return std::nullopt;
  }
  const NodeUnit* reshape2 = GetChildOfType(*transpose, kOpReshape);
  if (reshape2 == nullptr) {
    return std::nullopt;
  }
  const NodeUnit* transpose_tail = GetChildOfType(*reshape2, kOpTranspose);
  if (transpose_tail == nullptr) {
    return std::nullopt;
  }
  return std::array<const NodeUnit*, 5>{transpose_head, reshape1, transpose, reshape2, transpose_tail};
}

/// @brief Create or validate the QNN node of type ChannelShuffle.
/// @param qnn_model_wrapper QNN model wrapper
/// @param node_units The node units containing the nodes in pattern
/// @param validate Whether to validate the QNN node
/// @return Status
Status CreateOrValidateOnQnn(
    QnnModelWrapper* qnn_model_wrapper,
    gsl::span<const NodeUnit* const> node_units,
    bool validate) {
  const NodeUnit* transpose_head = node_units[0];
  const NodeUnit* transpose_tail = node_units[4];
  const NodeUnitIODef& cs_input_def = transpose_head->Inputs()[0];
  const NodeUnitIODef& cs_output_def = transpose_tail->Outputs()[0];

  std::vector<std::string> param_tensor_names;
  std::vector<Qnn_Param_t> param_tensors;
  {
    auto transpose_head_proto = transpose_head->GetNode().InputDefs()[0]->Shape();
    ORT_RETURN_IF_NOT(transpose_head_proto != nullptr, "Failed to get input shape proto.");
    TensorShape transpose_head_input_shape = utils::GetTensorProtoShape(*transpose_head_proto);
    const uint32_t channel_axis = static_cast<uint32_t>(transpose_head_input_shape.NumDimensions() - 1);
    Qnn_Scalar_t axis_scalar = QNN_SCALAR_INIT;
    axis_scalar.dataType = QNN_DATATYPE_UINT_32;
    axis_scalar.uint32Value = channel_axis;
    QnnParamWrapper param_wrapper(transpose_tail->Index(),
                                  transpose_tail->Name(),
                                  QNN_OP_CHANNEL_SHUFFLE_PARAM_AXIS,
                                  axis_scalar);
    ORT_RETURN_IF_NOT(qnn_model_wrapper->AddParamWrapper(std::move(param_wrapper)), "Failed to add param");
    param_tensor_names.push_back(param_wrapper.GetParamTensorName());
    param_tensors.push_back(param_wrapper.GetQnnParam());
  }
  {
    // Extract channel dimension from transpose (from channel last -> first)
    const NodeUnit* reshape1 = node_units[1];
    auto reshape1_proto = reshape1->GetNode().OutputDefs()[0]->Shape();
    ORT_RETURN_IF_NOT(reshape1_proto != nullptr, "Failed to get input shape proto.");
    TensorShape reshape1_output_shape = utils::GetTensorProtoShape(*reshape1_proto);
    Qnn_Scalar_t num_groups_scalar = QNN_SCALAR_INIT;
    num_groups_scalar.dataType = QNN_DATATYPE_UINT_32;
    num_groups_scalar.uint32Value = static_cast<uint32_t>(reshape1_output_shape.GetDims()[1]);
    QnnParamWrapper param_wrapper(transpose_tail->Index(),
                                  transpose_tail->Name(),
                                  QNN_OP_CHANNEL_SHUFFLE_PARAM_NUM_GROUPS,
                                  num_groups_scalar);
    ORT_RETURN_IF_NOT(qnn_model_wrapper->AddParamWrapper(std::move(param_wrapper)), "Failed to add param");
    param_tensor_names.push_back(param_wrapper.GetParamTensorName());
    param_tensors.push_back(param_wrapper.GetQnnParam());
  }

  QnnTensorWrapper channel_shuffle_input;
  QnnTensorWrapper channel_shuffle_output;
  ORT_RETURN_IF_ERROR(qnn_model_wrapper->MakeTensorWrapper(cs_input_def, channel_shuffle_input));
  ORT_RETURN_IF_ERROR(qnn_model_wrapper->MakeTensorWrapper(cs_output_def, channel_shuffle_output));

  // Note: Skipped QNN validation API due to its inconsistent behavior than creation API. Re-enable it when fixed.
  if (!validate) {
    ORT_RETURN_IF_NOT(qnn_model_wrapper->AddTensorWrapper(std::move(channel_shuffle_input)), "Failed to add input");
    ORT_RETURN_IF_NOT(qnn_model_wrapper->AddTensorWrapper(std::move(channel_shuffle_output)), "Failed to add output");
    ORT_RETURN_IF_NOT(qnn_model_wrapper->CreateQnnNode(transpose_tail->Name(),
                                                       QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                       QNN_OP_CHANNEL_SHUFFLE,
                                                       {cs_input_def.node_arg.Name()},
                                                       {cs_output_def.node_arg.Name()},
                                                       std::move(param_tensor_names),
                                                       validate),
                      "Failed to add fused " + std::string(kOpChannelShuffle) + " node.");
  }

  return Status::OK();
}

}  // namespace

std::unique_ptr<IQnnNodeGroup> ChannelShuffleFusion::TryFusion(
    QnnModelWrapper& qnn_model_wrapper,
    const NodeUnit& transpose_head,
    const MapNodeToNodeUnit& node_to_node_unit,
    const MapNodeUnitToGroup& node_unit_to_qnn_node_group,
    [[maybe_unused]] const logging::Logger& logger) {
  const GraphViewer& graph_viewer = qnn_model_wrapper.GetGraphViewer();
  std::optional<std::array<const NodeUnit*, 5>> pattern = MatchChannelShufflePattern(
      graph_viewer, &transpose_head, node_to_node_unit, node_unit_to_qnn_node_group);
  if (!pattern.has_value()) {
    return nullptr;
  }
  const NodeUnit* reshape1 = pattern->at(1);
  const NodeUnit* transpose = pattern->at(2);
  const NodeUnit* reshape2 = pattern->at(3);

  // Input shape to reshape1 must equal output shape of reshape2; and has rank > 2
  auto reshape1_input0_proto = reshape1->GetNode().InputDefs()[0]->Shape();
  auto reshape2_output_proto = reshape2->GetNode().OutputDefs()[0]->Shape();
  if (reshape1_input0_proto == nullptr || reshape2_output_proto == nullptr) {
    return nullptr;
  }
  TensorShape reshape1_input0_shape = utils::GetTensorProtoShape(*reshape1_input0_proto);
  TensorShape reshape2_output_shape = utils::GetTensorProtoShape(*reshape2_output_proto);
  if (reshape1_input0_shape.NumDimensions() != reshape2_output_shape.NumDimensions()) {
    return nullptr;
  }
  gsl::span<const int64_t> reshape1_input0_dims = reshape1_input0_shape.GetDims();
  gsl::span<const int64_t> reshape2_output_dims = reshape2_output_shape.GetDims();
  if (reshape1_input0_dims != reshape2_output_dims) {
    return nullptr;
  }

  // Intermediate shape must be 1 rank higher than input shape
  auto reshape1_output_proto = reshape1->GetNode().OutputDefs()[0]->Shape();
  if (reshape1_output_proto == nullptr) {
    return nullptr;
  }
  TensorShape reshape1_output_shape = utils::GetTensorProtoShape(*reshape1_output_proto);

  // Intermediate shape must split channels in groups only
  gsl::span<const int64_t> reshape1_output_dims = reshape1_output_shape.GetDims();
  if (reshape1_input0_dims[0] != reshape1_output_dims[0]) {
    return nullptr;
  }
  if (reshape1_output_dims.size() < 3) {
    return nullptr;
  }
  if (reshape1_input0_dims[1] != (reshape1_output_dims[1] * reshape1_output_dims[2])) {
    return nullptr;
  }
  if (reshape1_output_dims.size() != reshape1_input0_dims.size() + 1) {
    return nullptr;
  }
  size_t remaining_dims = reshape1_input0_dims.size() - 2;
  if (reshape1_output_dims.size() < remaining_dims + 3) {
    return nullptr;
  }
  for (size_t i = 0; i < remaining_dims; ++i) {
    if (reshape1_input0_dims[i + 2] != reshape1_output_dims[i + 3]) {
      return nullptr;
    }
  }

  // Intermediate transpose must only permute channels
  std::optional<std::vector<int64_t>> perm = GetTransposePerm(*transpose);
  if (!perm.has_value()) {
    return nullptr;
  }
  std::vector<int64_t> perm_to_check = perm.value();
  std::swap(perm_to_check[1], perm_to_check[2]);
  std::vector<int64_t> perm_expected(perm_to_check.size());
  for (size_t i = 0; i < perm_expected.size(); ++i) {
    perm_expected[i] = static_cast<int64_t>(i);
  }
  if (perm_to_check != perm_expected) {
    return nullptr;
  }

  // Check if the first and last transpose is a canceling transpose pair
  const NodeUnit* transpose_tail = pattern->at(4);
  std::optional<std::vector<int64_t>> perm_head = GetTransposePerm(transpose_head);
  if (!perm_head.has_value()) {
    return nullptr;
  }
  std::optional<std::vector<int64_t>> perm_tail = GetTransposePerm(*transpose_tail);
  if (!perm_tail.has_value()) {
    return nullptr;
  }
  if (!IsCancelingTransposePermPair(perm_head, perm_tail)) {
    return nullptr;
  }

  if (CreateOrValidateOnQnn(&qnn_model_wrapper, pattern.value(), /*validate=*/true) != Status::OK()) {
    return nullptr;
  }
  return std::make_unique<ChannelShuffleFusion>(pattern.value());
}

gsl::span<const NodeUnit* const> ChannelShuffleFusion::GetNodeUnits() const {
  return gsl::span<const NodeUnit* const>{node_units_.data(), node_units_.size()};
}

Status ChannelShuffleFusion::IsSupported(
    QnnModelWrapper& qnn_model_wrapper, [[maybe_unused]] const logging::Logger& logger) const {
  return CreateOrValidateOnQnn(&qnn_model_wrapper, GetNodeUnits(), /*validate=*/true);
}

Status ChannelShuffleFusion::AddToModelBuilder(
    QnnModelWrapper& qnn_model_wrapper, [[maybe_unused]] const logging::Logger& logger) const {
  return CreateOrValidateOnQnn(&qnn_model_wrapper, GetNodeUnits(), /*validate=*/false);
}

}  // namespace qnn
}  // namespace onnxruntime
