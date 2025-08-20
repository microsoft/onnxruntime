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

#include "core/providers/qnn-abi/ort_api.h"
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

// Forward declarations.
#define ValidateOnQnn(qnn_model_wrapper, mul_node_unit, softmax_node_unit) \
  CreateOrValidateOnQnn((qnn_model_wrapper), (mul_node_unit), (softmax_node_unit), true)
#define CreateOnQnn(qnn_model_wrapper, mul_node_unit, softmax_node_unit) \
  CreateOrValidateOnQnn((qnn_model_wrapper), (mul_node_unit), (softmax_node_unit), false)
static Status CreateOrValidateOnQnn(QnnModelWrapper& qnn_model_wrapper, const OrtNodeUnit& mul_node_unit,
                                    const OrtNodeUnit& softmax_node_unit, bool validate);

/// @brief Get the index of the scalar input in the mul node
/// @param mul Multiply node unit
/// @param ort_api ORT API interface
/// @return The index of the scalar input (0 or 1) if found, otherwise std::nullopt
std::optional<size_t> GetMulScalarInputIndex(const OrtNodeUnit& mul, const OrtApi& ort_api) {
  // Get inputs of mul node
  size_t num_inputs = 0;
  QNN_RETURN_IF_STATUS_NOT_OK(ort_api.Node_GetNumInputs(&mul.GetNode(), &num_inputs), ort_api, std::nullopt);
  std::vector<const OrtValueInfo*> inputs(num_inputs);
  QNN_RETURN_IF_STATUS_NOT_OK(ort_api.Node_GetInputs(&mul.GetNode(), inputs.data(), inputs.size()), ort_api,
                              std::nullopt);

  const OrtValueInfo* mul_x = inputs[0];
  const OrtValueInfo* mul_y = inputs[1];

  // Get type info for inputs
  const OrtTypeInfo* x_type_info = mul_x->GetTypeInfo();
  const OrtTypeInfo* y_type_info = mul_y->GetTypeInfo();

  // Cast to tensor info
  const OrtTensorTypeAndShapeInfo* x_tensor_info = nullptr;
  const OrtTensorTypeAndShapeInfo* y_tensor_info = nullptr;
  QNN_RETURN_IF_STATUS_NOT_OK(ort_api.CastTypeInfoToTensorInfo(x_type_info, &x_tensor_info), ort_api, std::nullopt);
  QNN_RETURN_IF_STATUS_NOT_OK(ort_api.CastTypeInfoToTensorInfo(y_type_info, &y_tensor_info), ort_api, std::nullopt);

  // Get dimensions count
  size_t x_dims_count = 0;
  size_t y_dims_count = 0;
  QNN_RETURN_IF_STATUS_NOT_OK(ort_api.GetDimensionsCount(x_tensor_info, &x_dims_count), ort_api, std::nullopt);
  QNN_RETURN_IF_STATUS_NOT_OK(ort_api.GetDimensionsCount(y_tensor_info, &y_dims_count), ort_api, std::nullopt);

  bool is_x_scalar = (x_dims_count == 0);
  bool is_y_scalar = (y_dims_count == 0);

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
/// @param ort_api ORT API interface
/// @return The axis for softmax
std::optional<uint32_t> GetPositiveSoftmaxAxis(const OrtNodeUnit& mul, const OrtNodeUnit& softmax, const OrtApi& ort_api) {
  OrtNodeAttrHelper softmax_attr_helper(ort_api, softmax);
  int64_t axis_value = softmax_attr_helper.Get("axis", (int64_t)-1);

  if (axis_value < 0) {
    // Get the scalar input index
    std::optional<size_t> input_scale_index = GetMulScalarInputIndex(mul, ort_api);
    if (!input_scale_index.has_value()) {
      return std::nullopt;
    }

    size_t input_other_index = 1U - input_scale_index.value();

    // Get the other input's shape
    size_t num_inputs = 0;
    QNN_RETURN_IF_STATUS_NOT_OK(ort_api.Node_GetNumInputs(&mul.GetNode(), &num_inputs), ort_api, std::nullopt);
    std::vector<const OrtValueInfo*> inputs(num_inputs);
    QNN_RETURN_IF_STATUS_NOT_OK(ort_api.Node_GetInputs(&mul.GetNode(), inputs.data(), inputs.size()), ort_api,
                                std::nullopt);

    const OrtValueInfo* other_input = inputs[input_other_index];
    const OrtTypeInfo* type_info = other_input->GetTypeInfo();
    const OrtTensorTypeAndShapeInfo* tensor_info = nullptr;
    QNN_RETURN_IF_STATUS_NOT_OK(ort_api.CastTypeInfoToTensorInfo(type_info, &tensor_info), ort_api, std::nullopt);

    size_t dims_count = 0;
    QNN_RETURN_IF_STATUS_NOT_OK(ort_api.GetDimensionsCount(tensor_info, &dims_count), ort_api, std::nullopt);

    axis_value += static_cast<int64_t>(dims_count);
  }

  return static_cast<uint32_t>(axis_value);
}

/// @brief Identify scalar input from mul node if present
/// @param qnn_model_wrapper QNN model wrapper
/// @param mul Multiply node unit
/// @param ort_api ORT API interface
/// @return The scalar input float value if found, otherwise std::nullopt
std::optional<float> ExtractScalarValueFromMul(const QnnModelWrapper& qnn_model_wrapper, const OrtNodeUnit& mul, const OrtApi& ort_api) {
  std::optional<size_t> input_scale_index = GetMulScalarInputIndex(mul, ort_api);
  if (!input_scale_index.has_value()) {
    return std::nullopt;
  }

  // Get inputs of mul node
  size_t num_inputs = 0;
  QNN_RETURN_IF_STATUS_NOT_OK(ort_api.Node_GetNumInputs(&mul.GetNode(), &num_inputs), ort_api, std::nullopt);
  std::vector<const OrtValueInfo*> inputs(num_inputs);
  QNN_RETURN_IF_STATUS_NOT_OK(ort_api.Node_GetInputs(&mul.GetNode(), inputs.data(), inputs.size()), ort_api,
                              std::nullopt);

  const OrtValueInfo* scalar_input = inputs[input_scale_index.value()];

  // Get the name of the scalar input
  const char* scalar_name = nullptr;
  QNN_RETURN_IF_STATUS_NOT_OK(ort_api.GetValueInfoName(scalar_input, &scalar_name), ort_api, std::nullopt);

  // Check if it's a constant initializer
  if (!qnn_model_wrapper.IsConstantInput(scalar_name)) {
    return std::nullopt;
  }

  // Get the constant tensor
  const OrtValueInfo* scalar_tensor = qnn_model_wrapper.GetConstantTensor(scalar_name);
  if (!scalar_tensor) {
    return std::nullopt;
  }

  // Get the value
  const OrtValue* value = nullptr;
  Status ort_status = scalar_tensor->GetInitializerValue(value);
  if (!ort_status.IsOK() || value == nullptr) {
    return std::nullopt;
  }

  // Check if it's a float tensor
  ONNXTensorElementDataType element_type;
  OrtTensorTypeAndShapeInfo* tensor_info = nullptr;
  QNN_RETURN_IF_STATUS_NOT_OK(ort_api.GetTensorTypeAndShape(value, &tensor_info), ort_api, std::nullopt);
  if (OrtStatus* status = ort_api.GetTensorElementType(tensor_info, &element_type)) {
    ort_api.ReleaseTensorTypeAndShapeInfo(tensor_info);
    QNN_RETURN_IF_STATUS_NOT_OK(status, ort_api, std::nullopt);
  }
  ort_api.ReleaseTensorTypeAndShapeInfo(tensor_info);
  if (element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return std::nullopt;
  }

  // Get the raw data
  const void* raw_data = nullptr;
  QNN_RETURN_IF_STATUS_NOT_OK(ort_api.GetTensorData(value, &raw_data), ort_api, std::nullopt);

  // Return the float value
  return *static_cast<const float*>(raw_data);
}

/// @brief Create or validate the QNN node
/// @param qnn_model_wrapper QNN model wrapper
/// @param node_units The node units containing the softmax and mul nodes
/// @param validate Whether to validate the QNN node
/// @return Status
Status CreateOrValidateOnQnn(
    QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& mul_node_unit,
    const OrtNodeUnit& softmax_node_unit,
    bool validate) {
  assert(mul_node_unit.OpType() == kOpMul && softmax_node_unit.OpType() == kOpSoftmax);
  const OrtApi& ort_api = qnn_model_wrapper.GetOrtApi();

  std::optional<size_t> input_scale_index = GetMulScalarInputIndex(mul_node_unit, ort_api);
  if (!input_scale_index.has_value()) {
    return Status(common::ONNXRUNTIME, common::FAIL, "Failed to get scalar input index");
  }

  size_t input_other_index = 1U - input_scale_index.value();
  const OrtNodeUnitIODef& mul_input_other = mul_node_unit.Inputs()[input_other_index];
  const OrtNodeUnitIODef& softmax_output = softmax_node_unit.Outputs()[0];

  std::vector<std::string> param_tensor_names;
  {  // axis
    std::optional<uint32_t> axis = GetPositiveSoftmaxAxis(mul_node_unit, softmax_node_unit, ort_api);
    if (axis.has_value()) {
      Qnn_Scalar_t axis_scalar = QNN_SCALAR_INIT;
      axis_scalar.dataType = QNN_DATATYPE_UINT_32;
      axis_scalar.uint32Value = axis.value();
      QnnParamWrapper param_wrapper(softmax_node_unit.GetNode().GetId(),
                                    softmax_node_unit.Name(),
                                    QNN_OP_SOFTMAX_PARAM_AXIS,
                                    axis_scalar);
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(param_wrapper)), "Failed to add axis param");
      param_tensor_names.push_back(param_wrapper.GetParamTensorName());
    }
  }
  {  // beta
    OrtNodeAttrHelper softmax_attr_helper(ort_api, softmax_node_unit);
    float beta = softmax_attr_helper.Get("beta", 1.0f);
    float scale = ExtractScalarValueFromMul(qnn_model_wrapper, mul_node_unit, ort_api).value_or(1.0f);
    Qnn_Scalar_t beta_scalar = QNN_SCALAR_INIT;
    beta_scalar.dataType = QNN_DATATYPE_FLOAT_32;
    beta_scalar.floatValue = scale * beta;
    QnnParamWrapper param_wrapper(softmax_node_unit.GetNode().GetId(),
                                  softmax_node_unit.Name(),
                                  QNN_OP_SOFTMAX_PARAM_BETA,
                                  beta_scalar);
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(param_wrapper)), "Failed to add beta param");
    param_tensor_names.push_back(param_wrapper.GetParamTensorName());
  }

  QnnTensorWrapper fused_softmax_input;
  QnnTensorWrapper fused_softmax_output;
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(mul_input_other, fused_softmax_input));
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(softmax_output, fused_softmax_output));

  if (validate) {
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.ValidateQnnNode(softmax_node_unit.Name(),
                                                          QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                          QNN_OP_SOFTMAX,
                                                          {fused_softmax_input.GetQnnTensor()},
                                                          {fused_softmax_output.GetQnnTensor()},
                                                          {}));
  } else {
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(fused_softmax_input)), "Failed to add input");
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(fused_softmax_output)), "Failed to add output");
    ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(softmax_node_unit.Name(),
                                                      QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                      QNN_OP_SOFTMAX,
                                                      {mul_input_other.name},
                                                      {softmax_output.name},
                                                      std::move(param_tensor_names),
                                                      validate),
                      "Failed to add fused " + std::string(kOpSoftmax) + " node.");
  }

  return Status::OK();
}

}  // namespace

std::unique_ptr<IQnnNodeGroup> ScaleSoftmaxFusion::TryFusion(
    QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& mul_node_unit,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
    const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
    [[maybe_unused]] const logging::Logger& logger) {
  if (mul_node_unit.OpType() != kOpMul || mul_node_unit.UnitType() != OrtNodeUnit::Type::SingleNode) {
    return nullptr;
  }

  // Get the OrtApi interface
  const OrtApi& ort_api = qnn_model_wrapper.GetOrtApi();

  // Check if the mul node has a scalar input that can fold into the softmax's beta
  std::optional<float> scalar = ExtractScalarValueFromMul(qnn_model_wrapper, mul_node_unit, ort_api);
  if (!scalar.has_value()) {
    return nullptr;
  }

  // Mul node must have a single Softmax node as child
  const std::array<std::string_view, 1> child_op_types{kOpSoftmax};
  const OrtNodeUnit* softmax = GetOnlyChildOfType(qnn_model_wrapper, mul_node_unit, child_op_types,
                                                  node_to_node_unit, node_unit_to_qnn_node_group);
  if (softmax == nullptr) {
    return nullptr;
  }

  if (Status status = ValidateOnQnn(qnn_model_wrapper, mul_node_unit, *softmax);
      !status.IsOK()) {
    return nullptr;
  }

  return std::make_unique<ScaleSoftmaxFusion>(mul_node_unit, *softmax);
}

gsl::span<const OrtNodeUnit* const> ScaleSoftmaxFusion::GetNodeUnits() const {
  return node_units_;
}

Status ScaleSoftmaxFusion::IsSupported(
    QnnModelWrapper& qnn_model_wrapper, [[maybe_unused]] const logging::Logger& logger) const {
  return ValidateOnQnn(qnn_model_wrapper, *node_units_[0], *node_units_[1]);
}

Status ScaleSoftmaxFusion::AddToModelBuilder(
    QnnModelWrapper& qnn_model_wrapper, [[maybe_unused]] const logging::Logger& logger) const {
  return CreateOnQnn(qnn_model_wrapper, *node_units_[0], *node_units_[1]);
}

}  // namespace qnn
}  // namespace onnxruntime
