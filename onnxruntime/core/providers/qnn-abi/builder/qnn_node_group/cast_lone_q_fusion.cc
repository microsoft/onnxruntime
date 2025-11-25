// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn-abi/builder/qnn_node_group/cast_lone_q_fusion.h"

#include "core/providers/qnn-abi/builder/qnn_model_wrapper.h"
#include "core/providers/qnn-abi/builder/qnn_node_group/utils.h"
#include "core/providers/qnn-abi/ort_api.h"

namespace onnxruntime {
namespace qnn {

constexpr char kOpCast[] = "Cast";
constexpr char kOpConvert[] = "Convert";

Ort::Status CreateOrValidateOnQnn(QnnModelWrapper* qnn_model_wrapper,
                                  gsl::span<const OrtNodeUnit* const> node_units,
                                  [[maybe_unused]] const Ort::Logger& logger,
                                  bool validate) {
  const OrtNodeUnit* cast = node_units[0];
  const OrtNodeUnit* quantize_linear = node_units[1];

  // ProcessInputs
  const auto& input_name = cast->Inputs()[0].name;
  if (!qnn_model_wrapper->IsQnnTensorWrapperExist(input_name)) {
    TensorInfo cast_node_input_info = {};
    RETURN_IF_ERROR(qnn_model_wrapper->GetTensorInfo(cast->Inputs()[0], cast_node_input_info));
    QnnTensorWrapper input_tensor_wrapper;
    RETURN_IF_ERROR(qnn_model_wrapper->MakeTensorWrapper(cast_node_input_info, input_name, input_tensor_wrapper));
    RETURN_IF_NOT(qnn_model_wrapper->AddTensorWrapper(std::move(input_tensor_wrapper)),
                  "Failed to add input tensor for QNN Convert node.");
  }
  // ProcessAttributesAndOutputs
  const auto& output_name = quantize_linear->Outputs()[0].name;
  TensorInfo q_node_output_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper->GetTensorInfo(quantize_linear->Outputs()[0], q_node_output_info));
  QnnTensorWrapper output_tensor_wrapper;
  RETURN_IF_ERROR(qnn_model_wrapper->MakeTensorWrapper(q_node_output_info, output_name, output_tensor_wrapper));
  RETURN_IF_NOT(qnn_model_wrapper->AddTensorWrapper(std::move(output_tensor_wrapper)),
                "Failed to add output tensor for QNN Convert node.");
  RETURN_IF_NOT(qnn_model_wrapper->CreateQnnNode(cast->Name() + "_ort_qnn_ep_convert",
                                                 QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                 QNN_OP_CONVERT,
                                                 {input_name},
                                                 {output_name},
                                                 {},
                                                 validate),
                ("Failed to add fused " + std::string(kOpConvert) + " node.").c_str());

  return Ort::Status();
}

std::unique_ptr<IQnnNodeGroup> CastLoneQFusion::TryFusion(
    QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& cast_node_unit,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
    const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
    [[maybe_unused]] const Ort::Logger& logger) {
  if (cast_node_unit.OpType() != kOpCast || cast_node_unit.UnitType() != OrtNodeUnit::Type::SingleNode) {
    return nullptr;
  }

  // Transform the pattern Non-DQ Node -> Cast -> Q into Non-DQ Node -> Convert
  const std::array<std::string_view, 1> child_op_types{QUANTIZE_LINEAR};
  const OrtNodeUnit* quantize_linear = GetOnlyChildOfType(qnn_model_wrapper,
                                                          cast_node_unit,
                                                          child_op_types,
                                                          node_to_node_unit,
                                                          node_unit_to_qnn_node_group);
  const std::array<std::string_view, 1> parent_op_types{DEQUANTIZE_LINEAR};
  const OrtNodeUnit* dequantize_linear = GetParentOfType(qnn_model_wrapper,
                                                         cast_node_unit,
                                                         parent_op_types,
                                                         node_to_node_unit,
                                                         node_unit_to_qnn_node_group);

  if (quantize_linear == nullptr || dequantize_linear != nullptr) {
    return nullptr;
  }

  // Skip Constant cast
  if (qnn_model_wrapper.IsConstantInput(cast_node_unit.Inputs()[0].name)) {
    return nullptr;
  }
  std::array<const OrtNodeUnit*, 2> node_unit_array{&cast_node_unit, quantize_linear};
  auto node_units = gsl::make_span<const OrtNodeUnit*>(node_unit_array.data(), 2);

  if (!CreateOrValidateOnQnn(&qnn_model_wrapper, node_units, logger, /*validate=*/true).IsOK()) {
    return nullptr;
  }
  return std::make_unique<CastLoneQFusion>(node_units);
}

gsl::span<const OrtNodeUnit* const> CastLoneQFusion::GetNodeUnits() const {
  return gsl::span<const OrtNodeUnit* const>{node_units_.data(), node_units_.size()};
}

Ort::Status CastLoneQFusion::IsSupported(
    QnnModelWrapper& qnn_model_wrapper, [[maybe_unused]] const Ort::Logger& logger) const {
  return CreateOrValidateOnQnn(&qnn_model_wrapper, GetNodeUnits(), logger, true);
}

Ort::Status CastLoneQFusion::AddToModelBuilder(
    QnnModelWrapper& qnn_model_wrapper, [[maybe_unused]] const Ort::Logger& logger) const {
  return CreateOrValidateOnQnn(&qnn_model_wrapper, GetNodeUnits(), logger, false);
}

}  // namespace qnn
}  // namespace onnxruntime
