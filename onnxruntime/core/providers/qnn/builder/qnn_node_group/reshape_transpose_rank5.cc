// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/qnn_node_group/reshape_transpose_rank5.h"

#include <gsl/gsl>
#include <optional>
#include <utility>
#include <string>
#include <array>
#include <memory>
#include <unordered_map>
#include <vector>
#include <sstream>

#include "core/common/inlined_containers.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_node_group/utils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/common/safeint.h"

namespace onnxruntime {
namespace qnn {
namespace {

constexpr size_t kRank6 = 6;
constexpr size_t kRank5 = 5;
constexpr const char* kOpTypeReshape = "Reshape";
constexpr const char* kOpTypeTranspose = "Transpose";
constexpr const char* kAttrTransposePerm = "perm";

using MapNodeToNodeUnit = std::unordered_map<const Node*, const NodeUnit*>;
using MapNodeUnitToGroup = std::unordered_map<const NodeUnit*, const IQnnNodeGroup*>;

/// @brief Get the shape of a tensor from its NodeArg
std::optional<TensorShape> GetTensorShape(const NodeArg* node_arg) {
  if (node_arg == nullptr) {
    return std::nullopt;
  }
  auto shape_proto = node_arg->Shape();
  if (shape_proto == nullptr) {
    return std::nullopt;
  }
  return utils::GetTensorProtoShape(*shape_proto);
}

/// @brief Get child NodeUnit of specified type, allowing QDQ-wrapped nodes
const NodeUnit* GetChildNodeUnit(
    const GraphViewer& graph_viewer,
    const NodeUnit& parent_node_unit,
    const std::string& child_op_type,
    const MapNodeToNodeUnit& node_to_node_unit,
    const MapNodeUnitToGroup& node_unit_to_qnn_node_group,
    const logging::Logger& logger) {
  const Node& parent_node = parent_node_unit.GetNode();

  ORT_UNUSED_PARAMETER(logger);
  // For QDQ NodeUnits, we need to look at the Q node's output, not the target node's output
  const Node* search_node = &parent_node;
  if (parent_node_unit.UnitType() == NodeUnit::Type::QDQGroup) {
    const auto& q_nodes = parent_node_unit.GetQNodes();
    if (!q_nodes.empty()) {
      search_node = q_nodes[0];  // Use first Q node
    }
  }

  // Search node must have a single child (1 output edge) and must not produce a graph output
  if (search_node->GetOutputEdgesCount() != 1 || graph_viewer.NodeProducesGraphOutput(*search_node)) {
    return nullptr;
  }

  // Get the child node from the search node's output edge
  const Node* potential_child = &search_node->OutputEdgesBegin()->GetNode();
  if (graph_viewer.GetNode(potential_child->Index()) == nullptr) {
    return nullptr;
  }

  // If the child is a DequantizeLinear, skip it and look at its child (the target op of the next QDQ group)
  if (potential_child->OpType() == "DequantizeLinear") {
    if (potential_child->GetOutputEdgesCount() != 1) {
      return nullptr;
    }
    potential_child = &potential_child->OutputEdgesBegin()->GetNode();
    if (graph_viewer.GetNode(potential_child->Index()) == nullptr) {
      return nullptr;
    }
  }

  // Check if this node matches the target type
  if (potential_child->OpType() != child_op_type) {
    return nullptr;
  }

  // Get the NodeUnit for the child
  const auto child_node_unit_it = node_to_node_unit.find(potential_child);
  if (child_node_unit_it == node_to_node_unit.end()) {
    return nullptr;
  }

  const NodeUnit* child_node_unit = child_node_unit_it->second;

  // Check if child node has already been handled
  if (node_unit_to_qnn_node_group.count(child_node_unit) != 0) {
    return nullptr;
  }

  return child_node_unit;
}

/// @brief Match the pattern: Reshape -> Transpose -> Reshape with rank-6 intermediate tensors
std::optional<std::array<const NodeUnit*, 3>> MatchRank6ToRank5Pattern(
    const GraphViewer& graph_viewer,
    const NodeUnit* reshape1,
    const MapNodeToNodeUnit& node_to_node_unit,
    const MapNodeUnitToGroup& node_unit_to_qnn_node_group,
    const logging::Logger& logger) {
  LOGS(logger, VERBOSE) << "[Rank6ToRank5] MatchPattern: Checking node " << reshape1->Name()
                        << " OpType=" << reshape1->OpType()
                        << " UnitType=" << static_cast<int>(reshape1->UnitType());

  // Validate first Reshape in pattern - allow both SingleNode and QDQGroup
  if (reshape1->OpType() != kOpTypeReshape) {
    LOGS(logger, VERBOSE) << "[Rank6ToRank5] First node in pattern is not a Reshape op";
    return std::nullopt;
  }

  // Get Transpose child (middle node in pattern) - allow both SingleNode and QDQGroup
  const NodeUnit* transpose = GetChildNodeUnit(
      graph_viewer, *reshape1, kOpTypeTranspose, node_to_node_unit, node_unit_to_qnn_node_group, logger);
  if (transpose == nullptr) {
    LOGS(logger, VERBOSE) << "[Rank6ToRank5] Transpose (middle node in pattern) not found after first Reshape";
    return std::nullopt;
  }

  LOGS(logger, VERBOSE) << "[Rank6ToRank5] Found Transpose (middle node): " << transpose->Name();

  // Get second Reshape child (last node in pattern) - allow both SingleNode and QDQGroup
  const NodeUnit* reshape2 = GetChildNodeUnit(
      graph_viewer, *transpose, kOpTypeReshape, node_to_node_unit, node_unit_to_qnn_node_group, logger);
  if (reshape2 == nullptr) {
    LOGS(logger, VERBOSE) << "[Rank6ToRank5] Second Reshape (last node in pattern) not found after Transpose";
    return std::nullopt;
  }

  LOGS(logger, VERBOSE) << "[Rank6ToRank5] Found second Reshape (last node): " << reshape2->Name();
  LOGS(logger, INFO) << "[Rank6ToRank5] Pattern matched: Reshape -> Transpose -> Reshape";

  return std::array<const NodeUnit*, 3>{reshape1, transpose, reshape2};
}

/// @brief Validate the pattern conditions and find the unit dimension index
std::optional<size_t> ValidatePatternConditions(
    const NodeUnit* reshape1,
    const NodeUnit* transpose,
    const NodeUnit* reshape2,
    const QnnModelWrapper& qnn_model_wrapper,
    const logging::Logger& logger) {
  // Check if reshape shape inputs are constants
  const NodeArg* reshape1_shape_input = reshape1->GetNode().InputDefs()[1];
  const NodeArg* reshape2_shape_input = reshape2->GetNode().InputDefs()[1];

  if (!qnn_model_wrapper.IsConstantInput(reshape1_shape_input->Name())) {
    LOGS(logger, VERBOSE) << "[Rank6ToRank5] ValidateConditions: Reshape1 shape input is not constant";
    return std::nullopt;
  }

  if (!qnn_model_wrapper.IsConstantInput(reshape2_shape_input->Name())) {
    LOGS(logger, VERBOSE) << "[Rank6ToRank5] ValidateConditions: Reshape2 shape input is not constant";
    return std::nullopt;
  }

  // Get tensor shapes
  auto t0_shape = GetTensorShape(reshape1->GetNode().InputDefs()[0]);
  auto t1_shape = GetTensorShape(reshape1->GetNode().OutputDefs()[0]);
  auto t2_shape = GetTensorShape(transpose->GetNode().OutputDefs()[0]);
  auto t3_shape = GetTensorShape(reshape2->GetNode().OutputDefs()[0]);

  if (!t0_shape.has_value() || !t1_shape.has_value() ||
      !t2_shape.has_value() || !t3_shape.has_value()) {
    LOGS(logger, VERBOSE) << "[Rank6ToRank5] ValidateConditions: Failed to get tensor shapes";
    return std::nullopt;
  }

  auto t1_dims = t1_shape->GetDims();
  auto t2_dims = t2_shape->GetDims();

  // Condition 1: Rank(t1) == Rank(t2) == 6
  if (t1_shape->NumDimensions() != kRank6 || t2_shape->NumDimensions() != kRank6) {
    LOGS(logger, VERBOSE) << "[Rank6ToRank5] ValidateConditions: Condition 1 failed - not rank-6: t1_rank="
                          << t1_shape->NumDimensions() << " t2_rank=" << t2_shape->NumDimensions();
    return std::nullopt;
  }

  if (t1_dims.empty() || t2_dims.empty()) {
    LOGS(logger, VERBOSE) << "[Rank6ToRank5] ValidateConditions: Empty dims";
    return std::nullopt;
  }

  // Condition 2: Find a dimension with value 1 that exists at the same index in both t1 and t2
  std::optional<size_t> unit_dim_index;
  for (size_t i = 0; i < kRank6; ++i) {
    if (t1_dims[i] == 1 && t2_dims[i] == 1) {
      unit_dim_index = i;
      break;
    }
  }

  if (!unit_dim_index.has_value()) {
    LOGS(logger, VERBOSE) << "[Rank6ToRank5] ValidateConditions: No common unit dimension found in t1 and t2";
    return std::nullopt;
  }

  // Condition 3: Transpose must leave the unit dimension in place
  NodeAttrHelper transpose_helper(transpose->GetNode());
  std::vector<int64_t> perm = transpose_helper.Get(kAttrTransposePerm, std::vector<int64_t>{});
  if (perm.size() != kRank6) {
    LOGS(logger, VERBOSE) << "[Rank6ToRank5] ValidateConditions: Invalid permutation size: " << perm.size();
    return std::nullopt;
  }

  if (perm[unit_dim_index.value()] != static_cast<int64_t>(unit_dim_index.value())) {
    LOGS(logger, VERBOSE) << "[Rank6ToRank5] ValidateConditions: Transpose moves unit dimension from index "
                          << unit_dim_index.value() << " to " << perm[unit_dim_index.value()];
    return std::nullopt;
  }

  LOGS(logger, INFO) << "[Rank6ToRank5] ValidateConditions: All conditions passed! Unit dimension at index "
                     << unit_dim_index.value();
  return unit_dim_index;
}

/// @brief Create or validate the QNN nodes with rank-5 tensors
Status CreateOrValidateOnQnn(
    QnnModelWrapper* qnn_model_wrapper,
    gsl::span<const NodeUnit* const> node_units,
    size_t unit_dim_index,
    bool validate,
    const logging::Logger& logger) {
  LOGS(logger, VERBOSE) << "[Rank6ToRank5] CreateOrValidateOnQnn: validate=" << validate
                        << " unit_dim_index=" << unit_dim_index;

  const NodeUnit* reshape1 = node_units[0];
  const NodeUnit* transpose = node_units[1];
  const NodeUnit* reshape2 = node_units[2];

  // Get input and output definitions
  const NodeUnitIODef& reshape1_input = reshape1->Inputs()[0];
  const NodeUnitIODef& reshape2_output = reshape2->Outputs()[0];

  // Get original shapes
  auto t1_shape = GetTensorShape(reshape1->GetNode().OutputDefs()[0]);
  auto t2_shape = GetTensorShape(transpose->GetNode().OutputDefs()[0]);

  if (!t1_shape.has_value() || !t2_shape.has_value()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to get intermediate tensor shapes");
  }

  auto t1_dims = t1_shape->GetDims();
  auto t2_dims = t2_shape->GetDims();

  // Create rank-5 shape for t1 (remove unit dimension at unit_dim_index)
  std::vector<uint32_t> t1_rank5_dims;
  t1_rank5_dims.reserve(kRank5);
  for (size_t i = 0; i < t1_dims.size(); ++i) {
    if (i != unit_dim_index) {
      t1_rank5_dims.push_back(static_cast<uint32_t>(t1_dims[i]));
    }
  }

  // Create rank-5 shape for t2 (remove unit dimension at unit_dim_index)
  std::vector<uint32_t> t2_rank5_dims;
  t2_rank5_dims.reserve(kRank5);
  for (size_t i = 0; i < t2_dims.size(); ++i) {
    if (i != unit_dim_index) {
      t2_rank5_dims.push_back(static_cast<uint32_t>(t2_dims[i]));
    }
  }

  // Get transpose permutation and adjust for rank-5
  NodeAttrHelper transpose_helper(transpose->GetNode());
  std::vector<int64_t> perm = transpose_helper.Get(kAttrTransposePerm, std::vector<int64_t>{});
  if (perm.size() != kRank6) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Expected rank-6 permutation, got rank-", perm.size());
  }

  // Remove unit dimension and adjust indices
  std::vector<uint32_t> perm_rank5;
  perm_rank5.reserve(kRank5);
  for (size_t i = 0; i < perm.size(); ++i) {
    if (i != unit_dim_index) {
      int64_t perm_val = perm[i];
      // Adjust index: if perm_val > unit_dim_index, subtract 1
      if (perm_val > static_cast<int64_t>(unit_dim_index)) {
        perm_val--;
      }
      perm_rank5.push_back(static_cast<uint32_t>(perm_val));
    }
  }

  // Use original tensor names from ONNX
  const std::string& t1_name = reshape1->GetNode().OutputDefs()[0]->Name();
  const std::string& t2_name = transpose->GetNode().OutputDefs()[0]->Name();

  // Get data type from the NodeUnit's output (handles both quantized and float types)
  const NodeUnitIODef& reshape1_output = reshape1->Outputs()[0];
  Qnn_DataType_t data_type;
  ORT_RETURN_IF_ERROR(utils::GetQnnDataType(reshape1_output.quant_param.has_value(),
                                            reshape1_output.node_arg.TypeAsProto(),
                                            data_type));

  // Get input shape for first Reshape
  std::vector<uint32_t> reshape1_input_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper->GetOnnxShape(reshape1_input.node_arg, reshape1_input_shape),
                    "Failed to get first Reshape input shape");

  // Get quantization params for first Reshape input
  QnnQuantParamsWrapper quant_param;
  ORT_RETURN_IF_ERROR(quant_param.Init(*qnn_model_wrapper, reshape1_input));

  // Create Reshape1 with rank-5 output using AddReshapeNode
  ORT_RETURN_IF_ERROR(qnn_model_wrapper->AddReshapeNode(
      reshape1_input.node_arg.Name(),
      t1_name,
      reshape1_input_shape,
      t1_rank5_dims,
      data_type,
      quant_param,
      validate,
      false,  // is_for_input
      false   // is_for_output
      ));

  // Create Transpose with rank-5 input/output
  {
    // Get quantization params for transpose output
    const NodeUnitIODef& transpose_output = transpose->Outputs()[0];
    QnnQuantParamsWrapper transpose_quant_param;
    ORT_RETURN_IF_ERROR(transpose_quant_param.Init(*qnn_model_wrapper, transpose_output));

    // Check if output tensor already exists
    if (!qnn_model_wrapper->IsQnnTensorWrapperExist(t2_name)) {
      // Create rank-5 output tensor for transpose with proper quantization params
      QnnTensorWrapper t2_tensor(t2_name, QNN_TENSOR_TYPE_NATIVE, data_type, std::move(transpose_quant_param),
                                 std::vector<uint32_t>(t2_rank5_dims));
      ORT_RETURN_IF_NOT(qnn_model_wrapper->AddTensorWrapper(std::move(t2_tensor)), "Failed to add transpose output");
    }

    // Create perm parameter
    std::vector<uint32_t> perm_shape = {static_cast<uint32_t>(perm_rank5.size())};
    QnnParamWrapper perm_param(transpose->Index(), transpose->Name(), QNN_OP_TRANSPOSE_PARAM_PERM,
                               std::move(perm_shape), std::move(perm_rank5));
    std::vector<std::string> param_tensor_names = {perm_param.GetParamTensorName()};
    ORT_RETURN_IF_NOT(qnn_model_wrapper->AddParamWrapper(std::move(perm_param)), "Failed to add perm param");

    std::vector<std::string> transpose_input_names = {t1_name};
    std::vector<std::string> transpose_output_names = {t2_name};

    ORT_RETURN_IF_NOT(qnn_model_wrapper->CreateQnnNode(
                          utils::GetUniqueName(*transpose),
                          QNN_OP_PACKAGE_NAME_QTI_AISW,
                          QNN_OP_TRANSPOSE,
                          std::move(transpose_input_names),
                          std::move(transpose_output_names),
                          std::move(param_tensor_names),
                          validate),
                      "Failed to create rank-5 Transpose node");
  }

  // Get output shape for reshape2
  std::vector<uint32_t> reshape2_output_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper->GetOnnxShape(reshape2_output.node_arg, reshape2_output_shape),
                    "Failed to get reshape2 output shape");

  // Get quantization params for reshape2
  QnnQuantParamsWrapper quant_param2;
  ORT_RETURN_IF_ERROR(quant_param2.Init(*qnn_model_wrapper, reshape2_output));

  // Get data type from the NodeUnit's output (handles both quantized and float types)
  ORT_RETURN_IF_ERROR(utils::GetQnnDataType(reshape2_output.quant_param.has_value(),
                                            reshape2_output.node_arg.TypeAsProto(),
                                            data_type));

  // Create Reshape2 with rank-5 input using AddReshapeNode
  ORT_RETURN_IF_ERROR(qnn_model_wrapper->AddReshapeNode(
      t2_name,
      reshape2_output.node_arg.Name(),
      t2_rank5_dims,
      reshape2_output_shape,
      data_type,
      quant_param2,
      validate,
      false,  // is_for_input
      false   // is_for_output
      ));

  return Status::OK();
}

}  // namespace

std::unique_ptr<IQnnNodeGroup> Rank6ToRank5Fusion::TryFusion(
    QnnModelWrapper& qnn_model_wrapper,
    const NodeUnit& reshape1_node_unit,
    const MapNodeToNodeUnit& node_to_node_unit,
    const MapNodeUnitToGroup& node_unit_to_qnn_node_group,
    const logging::Logger& logger) {
  LOGS(logger, VERBOSE) << "[Rank6ToRank5] TryFusion called for node: " << reshape1_node_unit.Name()
                        << " OpType: " << reshape1_node_unit.OpType();

  const GraphViewer& graph_viewer = qnn_model_wrapper.GetGraphViewer();

  // Match the pattern
  std::optional<std::array<const NodeUnit*, 3>> pattern = MatchRank6ToRank5Pattern(
      graph_viewer, &reshape1_node_unit, node_to_node_unit, node_unit_to_qnn_node_group, logger);

  if (!pattern.has_value()) {
    LOGS(logger, VERBOSE) << "[Rank6ToRank5] Pattern match failed for node: " << reshape1_node_unit.Name();
    return nullptr;
  }

  const NodeUnit* reshape1 = pattern->at(0);
  const NodeUnit* transpose = pattern->at(1);
  const NodeUnit* reshape2 = pattern->at(2);

  // Validate pattern conditions and get unit dimension index
  auto unit_dim_index = ValidatePatternConditions(reshape1, transpose, reshape2, qnn_model_wrapper, logger);
  if (!unit_dim_index.has_value()) {
    LOGS(logger, VERBOSE) << "[Rank6ToRank5] Pattern condition validation failed";
    return nullptr;
  }

  // Validate on QNN
  if (CreateOrValidateOnQnn(&qnn_model_wrapper, pattern.value(), unit_dim_index.value(), /*validate=*/true, logger) != Status::OK()) {
    LOGS(logger, VERBOSE) << "[Rank6ToRank5] QNN validation failed";
    return nullptr;
  }

  LOGS(logger, INFO) << "[Rank6ToRank5] Fusion successful! Creating Rank6ToRank5Fusion node group";
  return std::make_unique<Rank6ToRank5Fusion>(pattern.value(), unit_dim_index.value());
}

gsl::span<const NodeUnit* const> Rank6ToRank5Fusion::GetNodeUnits() const {
  return gsl::span<const NodeUnit* const>{node_units_.data(), node_units_.size()};
}

Status Rank6ToRank5Fusion::IsSupported(
    QnnModelWrapper& qnn_model_wrapper, const logging::Logger& logger) const {
  return CreateOrValidateOnQnn(&qnn_model_wrapper, GetNodeUnits(), unit_dim_index_, /*validate=*/true, logger);
}

Status Rank6ToRank5Fusion::AddToModelBuilder(
    QnnModelWrapper& qnn_model_wrapper, const logging::Logger& logger) const {
  return CreateOrValidateOnQnn(&qnn_model_wrapper, GetNodeUnits(), unit_dim_index_, /*validate=*/false, logger);
}

}  // namespace qnn
}  // namespace onnxruntime
