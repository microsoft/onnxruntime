// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn-abi/builder/qnn_node_group/scale_softmax_fusion.h"

#include <gsl/gsl>
#include <optional>
#include <utility>
#include <string>
#include <array>
#include <memory>
#include <unordered_map>
#include <vector>

#include "core/providers/qnn-abi/builder/qnn_utils.h"
#include "core/providers/qnn-abi/builder/op_builder_factory.h"
#include "core/providers/qnn-abi/builder/qnn_node_group/utils.h"
#include "core/providers/qnn-abi/builder/qnn_model_wrapper.h"
#include "core/providers/qnn-abi/builder/opbuilder/base_op_builder.h"

namespace onnxruntime {
namespace qnn {
namespace {

constexpr char kOpMul[] = "Mul";
constexpr char kOpSoftmax[] = "Softmax";

/// @brief Get the index of the scalar input in the mul node
/// @param mul Multiply node unit
/// @return The index of the scalar input (0 or 1) if found, otherwise std::nullopt
std::optional<size_t> GetMulScalarInputIndex(const NodeUnit* mul) {
  const NodeArg* mul_y = mul->GetNode().InputDefs()[1];
  const NodeArg* mul_x = mul->GetNode().InputDefs()[0];
  auto y_shape_proto = mul_y->Shape();
  auto x_shape_proto = mul_x->Shape();
  bool is_y_scalar = false;
  if (y_shape_proto != nullptr) {
    auto y_shape = utils::GetTensorProtoShape(*y_shape_proto);
    is_y_scalar = y_shape.NumDimensions() == 0;
  }
  bool is_x_scalar = false;
  if (x_shape_proto != nullptr) {
    auto x_shape = utils::GetTensorProtoShape(*x_shape_proto);
    is_x_scalar = x_shape.NumDimensions() == 0;
  }
  if (is_y_scalar) {
    return 1U;
  } else if (is_x_scalar) {
    return 0U;
  }
  return std::nullopt;
}

/// @brief Get the axis for softmax
/// @param mul Multiply node unit
/// @param softmax Softmax node unit
/// @return The axis for softmax
std::optional<uint32_t> GetPositiveSoftmaxAxis(const NodeUnit* mul, const NodeUnit* softmax) {
  NodeAttrHelper softmax_attr_helper(softmax->GetNode());
  std::optional<int64_t> param_axis = softmax_attr_helper.GetInt64(QNN_OP_SOFTMAX_PARAM_AXIS);
  if (!param_axis.has_value()) {
    return std::nullopt;
  }
  int64_t axis_value = param_axis.value();
  if (axis_value < 0) {
    size_t input_scale_index = GetMulScalarInputIndex(mul).value();
    size_t input_other_index = 1U - input_scale_index;
    int rank = mul->GetNode().InputDefs()[input_other_index]->Shape()->dim_size();
    axis_value += static_cast<int64_t>(rank);
  }
  return static_cast<uint32_t>(axis_value);
}

/// @brief Identify scalar input from mul node if present
/// @param mul Multiply node unit
/// @return The scalar input float value if found, otherwise std::nullopt
std::optional<float> ExtractScalarValueFromMul(const GraphViewer& graph_viewer, const NodeUnit* mul) {
  std::optional<size_t> input_scale_index = GetMulScalarInputIndex(mul);
  if (!input_scale_index.has_value()) {
    return std::nullopt;
  }
  const NodeArg* scalar_arg = mul->GetNode().InputDefs()[input_scale_index.value()];
  if (!graph_viewer.IsConstantInitializer(scalar_arg->Name(), true)) {
    return std::nullopt;
  }
  const auto* scalar_tensor = graph_viewer.GetConstantInitializer(scalar_arg->Name());
  if (!scalar_tensor) {
    return std::nullopt;
  }
  if (scalar_tensor->data_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    return std::nullopt;
  }
  const auto& raw_data = scalar_tensor->raw_data();
  if (raw_data.size() != sizeof(float) || reinterpret_cast<uintptr_t>(raw_data.data()) % alignof(float) != 0) {
    return std::nullopt;
  }
  return *reinterpret_cast<const float*>(raw_data.data());
}

/// @brief Create or validate the QNN node
/// @param qnn_model_wrapper QNN model wrapper
/// @param node_units The node units containing the softmax and mul nodes
/// @param validate Whether to validate the QNN node
/// @return Status
Status CreateOrValidateOnQnn(
    QnnModelWrapper* qnn_model_wrapper,
    gsl::span<const NodeUnit* const> node_units,
    bool validate) {
  const NodeUnit* mul = node_units[0];
  const NodeUnit* softmax = node_units[1];
  ORT_RETURN_IF_NOT(mul->OpType() == kOpMul,
                    "Expected scale node to be of type Mul, got ", mul->OpType());
  ORT_RETURN_IF_NOT(softmax->OpType() == kOpSoftmax,
                    "Expected softmax node to be of type Softmax, got ", softmax->OpType());
  size_t input_scale_index = GetMulScalarInputIndex(mul).value();
  size_t input_other_index = 1U - input_scale_index;
  const NodeUnitIODef& mul_input_other = mul->Inputs()[input_other_index];
  const NodeUnitIODef& softmax_output = softmax->Outputs()[0];

  std::vector<std::string> param_tensor_names;
  {  // axis
    std::optional<uint32_t> axis = GetPositiveSoftmaxAxis(mul, softmax);
    if (axis.has_value()) {
      Qnn_Scalar_t axis_scalar = QNN_SCALAR_INIT;
      axis_scalar.dataType = QNN_DATATYPE_UINT_32;
      axis_scalar.uint32Value = axis.value();
      QnnParamWrapper param_wrapper(softmax->Index(),
                                    softmax->Name(),
                                    QNN_OP_SOFTMAX_PARAM_AXIS,
                                    axis_scalar);
      ORT_RETURN_IF_NOT(qnn_model_wrapper->AddParamWrapper(std::move(param_wrapper)), "Failed to add param");
      param_tensor_names.push_back(param_wrapper.GetParamTensorName());
    }
  }
  {  // beta
    NodeAttrHelper softmax_attr_helper(softmax->GetNode());
    std::optional<float> beta = softmax_attr_helper.GetFloat(QNN_OP_SOFTMAX_PARAM_BETA);
    float scale = ExtractScalarValueFromMul(qnn_model_wrapper->GetGraphViewer(), mul).value_or(1.0f);
    Qnn_Scalar_t beta_scalar = QNN_SCALAR_INIT;
    beta_scalar.dataType = QNN_DATATYPE_FLOAT_32;
    beta_scalar.floatValue = scale * beta.value_or(1.0f);
    QnnParamWrapper param_wrapper(softmax->Index(),
                                  softmax->Name(),
                                  QNN_OP_SOFTMAX_PARAM_BETA,
                                  beta_scalar);
    ORT_RETURN_IF_NOT(qnn_model_wrapper->AddParamWrapper(std::move(param_wrapper)), "Failed to add param");
    param_tensor_names.push_back(param_wrapper.GetParamTensorName());
  }

  QnnTensorWrapper fused_softmax_input;
  QnnTensorWrapper fused_softmax_output;
  ORT_RETURN_IF_ERROR(qnn_model_wrapper->MakeTensorWrapper(mul_input_other, fused_softmax_input));
  ORT_RETURN_IF_ERROR(qnn_model_wrapper->MakeTensorWrapper(softmax_output, fused_softmax_output));

  if (validate) {
    ORT_RETURN_IF_ERROR(qnn_model_wrapper->ValidateQnnNode(softmax->Name(),
                                                           QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                           QNN_OP_SOFTMAX,
                                                           {fused_softmax_input.GetQnnTensor()},
                                                           {fused_softmax_output.GetQnnTensor()},
                                                           {}));
  } else {
    ORT_RETURN_IF_NOT(qnn_model_wrapper->AddTensorWrapper(std::move(fused_softmax_input)), "Failed to add input");
    ORT_RETURN_IF_NOT(qnn_model_wrapper->AddTensorWrapper(std::move(fused_softmax_output)), "Failed to add output");
    ORT_RETURN_IF_NOT(qnn_model_wrapper->CreateQnnNode(softmax->Name(),
                                                       QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                       QNN_OP_SOFTMAX,
                                                       {mul_input_other.node_arg.Name()},
                                                       {softmax_output.node_arg.Name()},
                                                       std::move(param_tensor_names),
                                                       validate),
                      "Failed to add fused " + std::string(kOpSoftmax) + " node.");
  }
  return Status::OK();
}

}  // namespace

std::unique_ptr<IQnnNodeGroup> ScaleSoftmaxFusion::TryFusion(
    QnnModelWrapper& qnn_model_wrapper,
    const NodeUnit& mul_node_unit,
    const std::unordered_map<const Node*, const NodeUnit*>& node_to_node_unit,
    const std::unordered_map<const NodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
    [[maybe_unused]] const logging::Logger& logger) {
  if (mul_node_unit.OpType() != kOpMul || mul_node_unit.UnitType() != NodeUnit::Type::SingleNode) {
    return nullptr;
  }
  // Check if the mul node has a scalar input that can fold into the softmax's beta
  const GraphViewer& graph_viewer = qnn_model_wrapper.GetGraphViewer();
  std::optional<float> scalar = ExtractScalarValueFromMul(graph_viewer, &mul_node_unit);
  if (!scalar.has_value()) {
    return nullptr;
  }

  // Mul node must have a single Softmax node as child
  const std::array<std::string_view, 1> child_op_types{kOpSoftmax};
  const NodeUnit* softmax = GetOnlyChildOfType(graph_viewer, mul_node_unit, child_op_types,
                                               node_to_node_unit, node_unit_to_qnn_node_group);
  if (softmax == nullptr) {
    return nullptr;
  }

  std::array<const NodeUnit*, 2> node_unit_array{&mul_node_unit, softmax};
  auto node_units = gsl::make_span<const NodeUnit*>(node_unit_array.data(), 2);
  if (CreateOrValidateOnQnn(&qnn_model_wrapper, node_units, /*validate=*/true) != Status::OK()) {
    return nullptr;
  }
  return std::make_unique<ScaleSoftmaxFusion>(node_units);
}

gsl::span<const NodeUnit* const> ScaleSoftmaxFusion::GetNodeUnits() const {
  return gsl::span<const NodeUnit* const>{node_units_.data(), node_units_.size()};
}

Status ScaleSoftmaxFusion::IsSupported(
    QnnModelWrapper& qnn_model_wrapper, [[maybe_unused]] const logging::Logger& logger) const {
  return CreateOrValidateOnQnn(&qnn_model_wrapper, GetNodeUnits(), /*validate=*/true);
}

Status ScaleSoftmaxFusion::AddToModelBuilder(
    QnnModelWrapper& qnn_model_wrapper, [[maybe_unused]] const logging::Logger& logger) const {
  return CreateOrValidateOnQnn(&qnn_model_wrapper, GetNodeUnits(), /*validate=*/false);
}

}  // namespace qnn
}  // namespace onnxruntime
