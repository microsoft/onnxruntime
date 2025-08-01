#include "core/providers/qnn-abi/builder/qnn_node_group/channel_shuffle_fusion.h"

#include <gsl/gsl>
#include <optional>
#include <utility>
#include <string>
#include <array>
#include <memory>
#include <unordered_map>
#include <vector>

#include "core/common/inlined_containers.h"
#include "core/providers/qnn-abi/ort_api.h"
#include "core/providers/qnn-abi/builder/qnn_utils.h"
#include "core/providers/qnn-abi/builder/op_builder_factory.h"
#include "core/providers/qnn-abi/builder/qnn_node_group/utils.h"
#include "core/providers/qnn-abi/builder/qnn_model_wrapper.h"
#include "core/providers/qnn-abi/builder/opbuilder/base_op_builder.h"

namespace onnxruntime {
namespace qnn {
namespace {

constexpr char kAttrTransposePerm[] = "perm";
constexpr char kOpChannelShuffle[] = "ChannelShuffle";
constexpr char kOpTranspose[] = "Transpose";
constexpr char kOpReshape[] = "Reshape";

using MapNodeToNodeUnit = std::unordered_map<const OrtNode*, const OrtNodeUnit*>;
using MapNodeUnitToGroup = std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>;

std::optional<std::vector<int64_t>> GetTransposePerm(const OrtNodeUnit& transpose, const OrtApi& ort_api) {
  if (transpose.OpType() != kOpTranspose) {
    return std::nullopt;
  }
  OrtNodeAttrHelper helper(ort_api, transpose);
  std::vector<int64_t> perm;
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
/// @param qnn_model_wrapper QNN model wrapper.
/// @param transpose_head The first transpose node starting the pattern.
/// @param node_to_node_unit Maps a Node to a NodeUnit.
/// @param node_unit_to_qnn_node_group Maps a NodeUnit to a IQnnNodeGroup.
/// @return The matched pattern as an array of NodeUnits if found, otherwise std::nullopt.
/// @note This is ChannelShuffle with transpose wraps commonly seen ORT post partitioning.
std::optional<std::array<const OrtNodeUnit*, 5>> MatchChannelShufflePattern(
    const QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit* transpose_head,
    const MapNodeToNodeUnit& node_to_node_unit,
    const MapNodeUnitToGroup& node_unit_to_qnn_node_group) {
  // Helper function to get a single child of a specific type
  auto GetChildOfType = [&](const OrtNodeUnit& node, std::string_view expect_type) -> const OrtNodeUnit* {
    const std::array<std::string_view, 1> child_op_types{expect_type};
    const OrtNodeUnit* child = GetOnlyChildOfType(qnn_model_wrapper, node, child_op_types,
                                                  node_to_node_unit, node_unit_to_qnn_node_group);
    if (child == nullptr) {
      return nullptr;
    }
    if (child->OpType() != expect_type) {
      return nullptr;
    }
    if (child->UnitType() != OrtNodeUnit::Type::SingleNode) {
      return nullptr;
    }
    return child;
  };

  if (transpose_head->OpType() != kOpTranspose) {
    return std::nullopt;
  }
  if (transpose_head->UnitType() != OrtNodeUnit::Type::SingleNode) {
    return std::nullopt;
  }
  const OrtNodeUnit* reshape1 = GetChildOfType(*transpose_head, kOpReshape);
  if (reshape1 == nullptr) {
    return std::nullopt;
  }
  const OrtNodeUnit* transpose = GetChildOfType(*reshape1, kOpTranspose);
  if (transpose == nullptr) {
    return std::nullopt;
  }
  const OrtNodeUnit* reshape2 = GetChildOfType(*transpose, kOpReshape);
  if (reshape2 == nullptr) {
    return std::nullopt;
  }
  const OrtNodeUnit* transpose_tail = GetChildOfType(*reshape2, kOpTranspose);
  if (transpose_tail == nullptr) {
    return std::nullopt;
  }
  return std::array<const OrtNodeUnit*, 5>{transpose_head, reshape1, transpose, reshape2, transpose_tail};
}

/// @brief Create or validate the QNN node of type ChannelShuffle.
/// @param qnn_model_wrapper QNN model wrapper
/// @param node_units The node units containing the nodes in pattern
/// @param validate Whether to validate the QNN node
/// @return Status
Status CreateOrValidateOnQnn(
    QnnModelWrapper& qnn_model_wrapper,
    gsl::span<const OrtNodeUnit* const> node_units,
    bool validate) {
  const OrtNodeUnit* transpose_head = node_units[0];
  const OrtNodeUnit* transpose_tail = node_units[4];
  const OrtNodeUnitIODef& cs_input_def = transpose_head->Inputs()[0];
  const OrtNodeUnitIODef& cs_output_def = transpose_tail->Outputs()[0];

  std::vector<std::string> param_tensor_names;
  std::vector<Qnn_Param_t> param_tensors;
  const OrtApi& ort_api = qnn_model_wrapper.GetOrtApi();

  // Get input shape to determine channel axis
  {
    // Get input shape information
    size_t num_transpose_head_inputs = 0;
    ort_api.Node_GetNumInputs(&transpose_head->GetNode(), &num_transpose_head_inputs);
    std::vector<const OrtValueInfo*> transpose_head_inputs(num_transpose_head_inputs);
    ort_api.Node_GetInputs(&transpose_head->GetNode(), transpose_head_inputs.data(), transpose_head_inputs.size());
    const OrtValueInfo* transpose_head_input_info = transpose_head_inputs[0];
    const OrtTypeInfo* transpose_head_input_type_info = transpose_head_input_info->GetTypeInfo();
    const OrtTensorTypeAndShapeInfo* transpose_head_input_tensor_info = nullptr;
    ort_api.CastTypeInfoToTensorInfo(transpose_head_input_type_info, &transpose_head_input_tensor_info);

    // Get dimensions count
    size_t transpose_head_input_dims_count = 0;
    ort_api.GetDimensionsCount(transpose_head_input_tensor_info, &transpose_head_input_dims_count);

    // Set channel axis parameter
    const uint32_t channel_axis = static_cast<uint32_t>(transpose_head_input_dims_count - 1);
    Qnn_Scalar_t axis_scalar = QNN_SCALAR_INIT;
    axis_scalar.dataType = QNN_DATATYPE_UINT_32;
    axis_scalar.uint32Value = channel_axis;
    QnnParamWrapper param_wrapper(transpose_tail->GetNode().GetId(),
                                  transpose_tail->Name(),
                                  QNN_OP_CHANNEL_SHUFFLE_PARAM_AXIS,
                                  axis_scalar);
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(param_wrapper)), "Failed to add axis param");
    param_tensor_names.push_back(param_wrapper.GetParamTensorName());
  }

  // Extract number of groups from reshape1 output shape
  {
    // Get reshape1 output shape
    const OrtNodeUnit* reshape1 = node_units[1];
    size_t num_reshape1_outputs = 0;
    ort_api.Node_GetNumOutputs(&reshape1->GetNode(), &num_reshape1_outputs);
    std::vector<const OrtValueInfo*> reshape1_outputs(num_reshape1_outputs);
    ort_api.Node_GetOutputs(&reshape1->GetNode(), reshape1_outputs.data(), reshape1_outputs.size());
    const OrtValueInfo* reshape1_output_info = reshape1_outputs[0];
    const OrtTypeInfo* reshape1_output_type_info = reshape1_output_info->GetTypeInfo();
    const OrtTensorTypeAndShapeInfo* reshape1_output_tensor_info = nullptr;
    ort_api.CastTypeInfoToTensorInfo(reshape1_output_type_info, &reshape1_output_tensor_info);

    // Get dimensions
    size_t reshape1_output_dims_count = 0;
    ort_api.GetDimensionsCount(reshape1_output_tensor_info, &reshape1_output_dims_count);
    std::vector<int64_t> reshape1_output_dims(reshape1_output_dims_count);
    ort_api.GetDimensions(reshape1_output_tensor_info, reshape1_output_dims.data(), reshape1_output_dims_count);

    // Set number of groups parameter
    Qnn_Scalar_t num_groups_scalar = QNN_SCALAR_INIT;
    num_groups_scalar.dataType = QNN_DATATYPE_UINT_32;
    num_groups_scalar.uint32Value = static_cast<uint32_t>(reshape1_output_dims[1]);
    QnnParamWrapper param_wrapper(transpose_tail->GetNode().GetId(),
                                  transpose_tail->Name(),
                                  QNN_OP_CHANNEL_SHUFFLE_PARAM_NUM_GROUPS,
                                  num_groups_scalar);
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(param_wrapper)), "Failed to add num_groups param");
    param_tensor_names.push_back(param_wrapper.GetParamTensorName());
  }

  // Create tensor wrappers for input and output
  QnnTensorWrapper channel_shuffle_input;
  QnnTensorWrapper channel_shuffle_output;
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(cs_input_def, channel_shuffle_input));
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(cs_output_def, channel_shuffle_output));

  if (validate) {
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.ValidateQnnNode(transpose_tail->Name(),
                                                          QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                          QNN_OP_CHANNEL_SHUFFLE,
                                                          {channel_shuffle_input.GetQnnTensor()},
                                                          {channel_shuffle_output.GetQnnTensor()},
                                                          {}));
  } else {
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(channel_shuffle_input)), "Failed to add input");
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(channel_shuffle_output)), "Failed to add output");
    ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(transpose_tail->Name(),
                                                      QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                      QNN_OP_CHANNEL_SHUFFLE,
                                                      {cs_input_def.name},
                                                      {cs_output_def.name},
                                                      std::move(param_tensor_names),
                                                      validate),
                      "Failed to add fused " + std::string(kOpChannelShuffle) + " node.");
  }

  return Status::OK();
}

}  // namespace

std::unique_ptr<IQnnNodeGroup> ChannelShuffleFusion::TryFusion(
    QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& transpose_head,
    const MapNodeToNodeUnit& node_to_node_unit,
    const MapNodeUnitToGroup& node_unit_to_qnn_node_group,
    [[maybe_unused]] const logging::Logger& logger) {
  std::optional<std::array<const OrtNodeUnit*, 5>> pattern = MatchChannelShufflePattern(
      qnn_model_wrapper, &transpose_head, node_to_node_unit, node_unit_to_qnn_node_group);
  if (!pattern.has_value()) {
    return nullptr;
  }

  const OrtNodeUnit* reshape1 = pattern->at(1);
  const OrtNodeUnit* transpose = pattern->at(2);
  const OrtNodeUnit* reshape2 = pattern->at(3);
  const OrtNodeUnit* transpose_tail = pattern->at(4);
  const OrtApi& ort_api = qnn_model_wrapper.GetOrtApi();

  // Input shape to reshape1 must equal output shape of reshape2; and has rank > 2
  // Get reshape1 input shape
  size_t num_reshape1_inputs = 0;
  ort_api.Node_GetNumInputs(&reshape1->GetNode(), &num_reshape1_inputs);
  std::vector<const OrtValueInfo*> reshape1_inputs(num_reshape1_inputs);
  ort_api.Node_GetInputs(&reshape1->GetNode(), reshape1_inputs.data(), reshape1_inputs.size());
  const OrtValueInfo* reshape1_input_info = reshape1_inputs[0];
  const OrtTypeInfo* reshape1_input_type_info = reshape1_input_info->GetTypeInfo();
  const OrtTensorTypeAndShapeInfo* reshape1_input_tensor_info = nullptr;
  ort_api.CastTypeInfoToTensorInfo(reshape1_input_type_info, &reshape1_input_tensor_info);

  // Get reshape2 output shape
  size_t num_reshape2_outputs = 0;
  ort_api.Node_GetNumOutputs(&reshape2->GetNode(), &num_reshape2_outputs);
  std::vector<const OrtValueInfo*> reshape2_outputs(num_reshape2_outputs);
  ort_api.Node_GetOutputs(&reshape2->GetNode(), reshape2_outputs.data(), reshape2_outputs.size());
  const OrtValueInfo* reshape2_output_info = reshape2_outputs[0];
  const OrtTypeInfo* reshape2_output_type_info = reshape2_output_info->GetTypeInfo();
  const OrtTensorTypeAndShapeInfo* reshape2_output_tensor_info = nullptr;
  ort_api.CastTypeInfoToTensorInfo(reshape2_output_type_info, &reshape2_output_tensor_info);

  // Compare dimensions
  size_t reshape1_input_dims_count = 0;
  size_t reshape2_output_dims_count = 0;
  ort_api.GetDimensionsCount(reshape1_input_tensor_info, &reshape1_input_dims_count);
  ort_api.GetDimensionsCount(reshape2_output_tensor_info, &reshape2_output_dims_count);

  if (reshape1_input_dims_count != reshape2_output_dims_count) {
    return nullptr;
  }

  std::vector<int64_t> reshape1_input_dims(reshape1_input_dims_count);
  std::vector<int64_t> reshape2_output_dims(reshape2_output_dims_count);
  ort_api.GetDimensions(reshape1_input_tensor_info, reshape1_input_dims.data(), reshape1_input_dims_count);
  ort_api.GetDimensions(reshape2_output_tensor_info, reshape2_output_dims.data(), reshape2_output_dims_count);

  // Check if reshape1 input dims equal reshape2 output dims
  if (!std::equal(reshape1_input_dims.begin(), reshape1_input_dims.end(), reshape2_output_dims.begin())) {
    return nullptr;
  }

  // Get reshape1 output shape
  size_t num_reshape1_outputs = 0;
  ort_api.Node_GetNumOutputs(&reshape1->GetNode(), &num_reshape1_outputs);
  std::vector<const OrtValueInfo*> reshape1_outputs(num_reshape1_outputs);
  ort_api.Node_GetOutputs(&reshape1->GetNode(), reshape1_outputs.data(), reshape1_outputs.size());
  const OrtValueInfo* reshape1_output_info = reshape1_outputs[0];
  const OrtTypeInfo* reshape1_output_type_info = reshape1_output_info->GetTypeInfo();
  const OrtTensorTypeAndShapeInfo* reshape1_output_tensor_info = nullptr;
  ort_api.CastTypeInfoToTensorInfo(reshape1_output_type_info, &reshape1_output_tensor_info);

  size_t reshape1_output_dims_count = 0;
  ort_api.GetDimensionsCount(reshape1_output_tensor_info, &reshape1_output_dims_count);
  std::vector<int64_t> reshape1_output_dims(reshape1_output_dims_count);
  ort_api.GetDimensions(reshape1_output_tensor_info, reshape1_output_dims.data(), reshape1_output_dims_count);

  // Intermediate shape must split channels in groups only
  if (reshape1_input_dims[0] != reshape1_output_dims[0]) {
    return nullptr;
  }

  if (reshape1_output_dims_count < 3) {
    return nullptr;
  }

  if (reshape1_input_dims[1] != (reshape1_output_dims[1] * reshape1_output_dims[2])) {
    return nullptr;
  }

  if (reshape1_output_dims_count != reshape1_input_dims_count + 1) {
    return nullptr;
  }

  size_t remaining_dims = reshape1_input_dims_count - 2;
  if (reshape1_output_dims_count < remaining_dims + 3) {
    return nullptr;
  }

  for (size_t i = 0; i < remaining_dims; ++i) {
    if (reshape1_input_dims[i + 2] != reshape1_output_dims[i + 3]) {
      return nullptr;
    }
  }

  // Intermediate transpose must only permute channels
  std::optional<std::vector<int64_t>> perm = GetTransposePerm(*transpose, ort_api);
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
  std::optional<std::vector<int64_t>> perm_head = GetTransposePerm(transpose_head, ort_api);
  if (!perm_head.has_value()) {
    return nullptr;
  }

  std::optional<std::vector<int64_t>> perm_tail = GetTransposePerm(*transpose_tail, ort_api);
  if (!perm_tail.has_value()) {
    return nullptr;
  }

  auto perm_head_span = gsl::make_span<const int64_t>(perm_head.value().data(), perm_head.value().size());
  auto perm_tail_span = gsl::make_span<const int64_t>(perm_tail.value().data(), perm_tail.value().size());

  if (!IsCancelingTransposePermPair(perm_head_span, perm_tail_span)) {
    return nullptr;
  }

  if (CreateOrValidateOnQnn(qnn_model_wrapper, pattern.value(), /*validate=*/true) != Status::OK()) {
    return nullptr;
  }
  return std::make_unique<ChannelShuffleFusion>(pattern.value());
}

gsl::span<const OrtNodeUnit* const> ChannelShuffleFusion::GetNodeUnits() const {
  return gsl::span<const OrtNodeUnit* const>{node_units_.data(), node_units_.size()};
}

Status ChannelShuffleFusion::IsSupported(
    QnnModelWrapper& qnn_model_wrapper, [[maybe_unused]] const logging::Logger& logger) const {
  return CreateOrValidateOnQnn(qnn_model_wrapper, GetNodeUnits(), /*validate=*/true);
}

Status ChannelShuffleFusion::AddToModelBuilder(
    QnnModelWrapper& qnn_model_wrapper, [[maybe_unused]] const logging::Logger& logger) const {
  return CreateOrValidateOnQnn(qnn_model_wrapper, GetNodeUnits(), /*validate=*/false);
}

}  // namespace qnn
}  // namespace onnxruntime
