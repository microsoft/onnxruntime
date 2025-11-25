// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn-abi/builder/qnn_node_group/dq_q_fusion.h"

#include <algorithm>
#include <cassert>
#include <gsl/gsl>
#include <limits>
#include <optional>
#include <utility>

#include "core/providers/qnn-abi/builder/op_builder_factory.h"
#include "core/providers/qnn-abi/builder/qnn_model_wrapper.h"
#include "core/providers/qnn-abi/builder/qnn_node_group/utils.h"
#include "core/providers/qnn-abi/builder/qnn_utils.h"
#include "core/providers/qnn-abi/ort_api.h"

namespace onnxruntime {
namespace qnn {

// Forward declarations.
#define ValidateOnQnn(qnn_model_wrapper, dq_node_unit, q_node_unit) \
  CreateOrValidateOnQnn((qnn_model_wrapper), (dq_node_unit), (q_node_unit), true)
#define CreateOnQnn(qnn_model_wrapper, dq_node_unit, q_node_unit) \
  CreateOrValidateOnQnn((qnn_model_wrapper), (dq_node_unit), (q_node_unit), false)
static Ort::Status CreateOrValidateOnQnn(QnnModelWrapper& qnn_model_wrapper, const OrtNodeUnit& dq_node_unit,
                                         const OrtNodeUnit& q_node_unit, bool validate);
static bool IsDQQConversion(const QnnModelWrapper& qnn_model_wrapper, const OrtNode& dq_node, const OrtNode& q_node);

std::unique_ptr<IQnnNodeGroup> DQQFusion::TryFusion(
    QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& dq_node_unit,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
    const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
    const Ort::Logger& logger) {
  ORT_UNUSED_PARAMETER(logger);
  // Expect that this function is called with a standalone DQ.
  if (dq_node_unit.OpType() != DEQUANTIZE_LINEAR || dq_node_unit.UnitType() != OrtNodeUnit::Type::SingleNode) {
    return nullptr;
  }

  const OrtNode& dq_node = dq_node_unit.GetNode();

  // DQ must have a single Q child (1 output edge) and must not produce a graph output.
  const std::array<std::string_view, 1> child_types = {QUANTIZE_LINEAR};
  const OrtNodeUnit* q_node_unit = GetOnlyChildOfType(qnn_model_wrapper, dq_node_unit, child_types,
                                                      node_to_node_unit, node_unit_to_qnn_node_group);

  if (q_node_unit == nullptr) {
    return nullptr;
  }

  // DQ and Q must have equal scale type and different zp type.
  if (!IsDQQConversion(qnn_model_wrapper, dq_node, q_node_unit->GetNode())) {
    return nullptr;
  }

  if (Ort::Status status = ValidateOnQnn(qnn_model_wrapper, dq_node_unit, *q_node_unit);
      !status.IsOK()) {
    return nullptr;
  }

  return std::make_unique<DQQFusion>(dq_node_unit, *q_node_unit);
}

DQQFusion::DQQFusion(const OrtNodeUnit& dq_node_unit, const OrtNodeUnit& q_node_unit)
    : node_units_{&dq_node_unit, &q_node_unit} {
}

Ort::Status DQQFusion::IsSupported(QnnModelWrapper& qmw, const Ort::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);
  return ValidateOnQnn(qmw, *node_units_[0], *node_units_[1]);
}

Ort::Status DQQFusion::AddToModelBuilder(QnnModelWrapper& qmw, const Ort::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);
  return CreateOnQnn(qmw, *node_units_[0], *node_units_[1]);
}

gsl::span<const OrtNodeUnit* const> DQQFusion::GetNodeUnits() const {
  return node_units_;
}

const OrtNodeUnit* DQQFusion::GetTargetNodeUnit() const {
  return node_units_[0];
}

static Ort::Status CreateOrValidateOnQnn(QnnModelWrapper& qnn_model_wrapper,
                                         const OrtNodeUnit& dq_node_unit,
                                         const OrtNodeUnit& q_node_unit,
                                         bool validate) {
  assert(dq_node_unit.OpType() == DEQUANTIZE_LINEAR && q_node_unit.OpType() == QUANTIZE_LINEAR);
  const auto& node_name = utils::GetUniqueName(dq_node_unit);
  const OrtNodeUnitIODef& input_def = dq_node_unit.Inputs()[0];
  const OrtNodeUnitIODef& output_def = q_node_unit.Outputs()[0];

  QnnTensorWrapper input_tensor;
  QnnTensorWrapper output_tensor;

  RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(input_def, input_tensor));
  RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(output_def, output_tensor));

  if (validate) {
    RETURN_IF_ERROR(qnn_model_wrapper.ValidateQnnNode(node_name,
                                                      QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                      QNN_OP_CONVERT,
                                                      {input_tensor.GetQnnTensor()},
                                                      {output_tensor.GetQnnTensor()},
                                                      {}));
  } else {
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensor)), "Failed to add input");
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensor)), "Failed to add output");
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(node_name,
                                                  QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                  QNN_OP_CONVERT,
                                                  {input_def.name},
                                                  {output_def.name},
                                                  {},
                                                  validate),
                  "Failed to add fused Convert node.");
  }

  return Ort::Status();
  ;
}

static bool IsDQQConversion(const QnnModelWrapper& qnn_model_wrapper, const OrtNode& dq_node, const OrtNode& q_node) {
  const OrtApi& ort_api = qnn_model_wrapper.GetOrtApi();

  // Get DQ inputs
  size_t dq_inputs_count = 0;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetNumInputs(&dq_node, &dq_inputs_count), ort_api, false);
  std::vector<const OrtValueInfo*> dq_inputs(dq_inputs_count);
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetInputs(&dq_node, dq_inputs.data(), dq_inputs.size()), ort_api, false);

  // Get Q inputs
  size_t q_inputs_count = 0;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetNumInputs(&q_node, &q_inputs_count), ort_api, false);
  std::vector<const OrtValueInfo*> q_inputs(q_inputs_count);
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetInputs(&q_node, q_inputs.data(), q_inputs.size()), ort_api, false);

  auto is_scalar_shape = [&ort_api](const OrtValueInfo* value_info) -> bool {
    const OrtTypeInfo* type_info = nullptr;
    RETURN_DEFAULT_IF_API_FAIL(ort_api.GetValueInfoTypeInfo(value_info, &type_info), ort_api, false);

    const OrtTensorTypeAndShapeInfo* tensor_info = nullptr;
    RETURN_DEFAULT_IF_API_FAIL(ort_api.CastTypeInfoToTensorInfo(type_info, &tensor_info), ort_api, false);

    size_t dims_count = 0;
    RETURN_DEFAULT_IF_API_FAIL(ort_api.GetDimensionsCount(tensor_info, &dims_count), ort_api, false);

    if (dims_count == 0) {
      return true;
    }

    if (dims_count == 1) {
      int64_t dim_value = 0;
      RETURN_DEFAULT_IF_API_FAIL(ort_api.GetDimensions(tensor_info, &dim_value, 1), ort_api, false);
      return dim_value == 1;
    }

    return false;
  };

  // Q/DQ contains optional input is not supported
  // non-scalar Q/DQ scale and zero point needs are not supported
  if (dq_inputs_count != QDQ_MAX_NUM_INPUTS || q_inputs_count != QDQ_MAX_NUM_INPUTS) {
    return false;
  }

  const OrtValueInfo* dq_scale = dq_inputs[QDQ_SCALE_INPUT_IDX];
  const OrtValueInfo* dq_zero_point = dq_inputs[QDQ_ZERO_POINT_INPUT_IDX];
  const OrtValueInfo* q_scale = q_inputs[QDQ_SCALE_INPUT_IDX];
  const OrtValueInfo* q_zero_point = q_inputs[QDQ_ZERO_POINT_INPUT_IDX];

  if (!is_scalar_shape(dq_scale) || !is_scalar_shape(dq_zero_point) ||
      !is_scalar_shape(q_scale) || !is_scalar_shape(q_zero_point)) {
    return false;
  }

  // Check if the inputs are constant initializers
  const OrtValue* dq_scale_value = nullptr;
  const OrtValue* dq_zero_point_value = nullptr;
  const OrtValue* q_scale_value = nullptr;
  const OrtValue* q_zero_point_value = nullptr;

  OrtStatus* status = ort_api.ValueInfo_GetInitializerValue(dq_scale, &dq_scale_value);
  if (status != nullptr) {
    ort_api.ReleaseStatus(status);
    return false;
  }

  status = ort_api.ValueInfo_GetInitializerValue(dq_zero_point, &dq_zero_point_value);
  if (status != nullptr) {
    ort_api.ReleaseStatus(status);
    return false;
  }

  status = ort_api.ValueInfo_GetInitializerValue(q_scale, &q_scale_value);
  if (status != nullptr) {
    ort_api.ReleaseStatus(status);
    return false;
  }

  status = ort_api.ValueInfo_GetInitializerValue(q_zero_point, &q_zero_point_value);
  if (status != nullptr) {
    ort_api.ReleaseStatus(status);
    return false;
  }

  // Get the data types
  const OrtTypeInfo* dq_scale_type_info = nullptr;
  const OrtTypeInfo* q_scale_type_info = nullptr;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.GetValueInfoTypeInfo(dq_scale, &dq_scale_type_info), ort_api, false);
  RETURN_DEFAULT_IF_API_FAIL(ort_api.GetValueInfoTypeInfo(q_scale, &q_scale_type_info), ort_api, false);

  const OrtTensorTypeAndShapeInfo* dq_scale_tensor_info = nullptr;
  const OrtTensorTypeAndShapeInfo* q_scale_tensor_info = nullptr;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.CastTypeInfoToTensorInfo(dq_scale_type_info, &dq_scale_tensor_info),
                             ort_api,
                             false);
  RETURN_DEFAULT_IF_API_FAIL(ort_api.CastTypeInfoToTensorInfo(q_scale_type_info, &q_scale_tensor_info),
                             ort_api,
                             false);

  ONNXTensorElementDataType dq_scale_data_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  ONNXTensorElementDataType q_scale_data_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.GetTensorElementType(dq_scale_tensor_info, &dq_scale_data_type),
                             ort_api,
                             false);
  RETURN_DEFAULT_IF_API_FAIL(ort_api.GetTensorElementType(q_scale_tensor_info, &q_scale_data_type),
                             ort_api,
                             false);

  // For scale, ensure that the Q/DQ have same scale type.
  return (dq_scale_data_type == q_scale_data_type);
}

}  // namespace qnn
}  // namespace onnxruntime
