// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/qnn_node_group/cast_lone_q_fusion.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_node_group/utils.h"

namespace onnxruntime {
namespace qnn {

constexpr char kOpCast[] = "Cast";
constexpr char kOpConvert[] = "Convert";

Status CreateOrValidateOnQnn(
    QnnModelWrapper* qnn_model_wrapper,
    gsl::span<const NodeUnit* const> node_units,
    [[maybe_unused]] const logging::Logger& logger,
    bool validate) {
  const NodeUnit* cast = node_units[0];
  const NodeUnit* quantizeLinear = node_units[1];

  // ProcessInputs
  const auto& input_name = cast->Inputs()[0].node_arg.Name();
  if (!qnn_model_wrapper->IsQnnTensorWrapperExist(input_name)) {
    TensorInfo cast_node_input_info = {};
    ORT_RETURN_IF_ERROR(qnn_model_wrapper->GetTensorInfo(cast->Inputs()[0], cast_node_input_info));
    QnnTensorWrapper input_tensor_wrapper;
    ORT_RETURN_IF_ERROR(qnn_model_wrapper->MakeTensorWrapper(cast_node_input_info, input_name, input_tensor_wrapper));
    ORT_RETURN_IF_NOT(qnn_model_wrapper->AddTensorWrapper(std::move(input_tensor_wrapper)),
                      "Failed to add input tensor for QNN Convert node.");
  }
  // ProcessAttributesAndOutputs
  const auto& output_name = quantizeLinear->Outputs()[0].node_arg.Name();
  TensorInfo q_node_output_info = {};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper->GetTensorInfo(quantizeLinear->Outputs()[0], q_node_output_info));
  QnnTensorWrapper output_tensor_wrapper;
  ORT_RETURN_IF_ERROR(qnn_model_wrapper->MakeTensorWrapper(q_node_output_info, output_name, output_tensor_wrapper));
  ORT_RETURN_IF_NOT(qnn_model_wrapper->AddTensorWrapper(std::move(output_tensor_wrapper)),
                    "Failed to add output tensor for QNN Convert node.");
  ORT_RETURN_IF_NOT(qnn_model_wrapper->CreateQnnNode(cast->Name() + "_ort_qnn_ep_convert",
                                                     QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                     QNN_OP_CONVERT,
                                                     {input_name},
                                                     {output_name},
                                                     {},
                                                     validate),
                    "Failed to add fused " + std::string(kOpConvert) + " node.");

  return Status::OK();
}

std::unique_ptr<IQnnNodeGroup> CastLoneQFusion::TryFusion(
    QnnModelWrapper& qnn_model_wrapper,
    const NodeUnit& cast_node_unit,
    const std::unordered_map<const Node*, const NodeUnit*>& node_to_node_unit,
    const std::unordered_map<const NodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
    [[maybe_unused]] const logging::Logger& logger) {
  if (cast_node_unit.OpType() != kOpCast || cast_node_unit.UnitType() != NodeUnit::Type::SingleNode) {
    return nullptr;
  }

  // Transform the pattern Non-DQ Node -> Cast -> Q into Non-DQ Node -> Convert
  const GraphViewer& graph_viewer = qnn_model_wrapper.GetGraphViewer();
  const std::array<std::string_view, 1> child_op_types{QUANTIZE_LINEAR};
  const NodeUnit* quantizeLinear = GetOnlyChildOfType(
      graph_viewer, cast_node_unit, child_op_types,
      node_to_node_unit, node_unit_to_qnn_node_group);
  const std::array<std::string_view, 1> parent_op_types{DEQUANTIZE_LINEAR};
  const NodeUnit* dequantizeLinear = GetParentOfType(
      graph_viewer, cast_node_unit, parent_op_types,
      node_to_node_unit, node_unit_to_qnn_node_group);

  if (quantizeLinear == nullptr || dequantizeLinear != nullptr) {
    return nullptr;
  }

  // Skip Constant cast
  if (qnn_model_wrapper.IsConstantInput(cast_node_unit.Inputs()[0].node_arg.Name())) {
    return nullptr;
  }
  std::array<const NodeUnit*, 2> node_unit_array{&cast_node_unit, quantizeLinear};
  auto node_units = gsl::make_span<const NodeUnit*>(node_unit_array.data(), 2);

  if (CreateOrValidateOnQnn(&qnn_model_wrapper, node_units, logger, /*validate=*/true) != Status::OK()) {
    return nullptr;
  }
  return std::make_unique<CastLoneQFusion>(node_units);
}

gsl::span<const NodeUnit* const> CastLoneQFusion::GetNodeUnits() const {
  return gsl::span<const NodeUnit* const>{node_units_.data(), node_units_.size()};
}

Status CastLoneQFusion::IsSupported(
    QnnModelWrapper& qnn_model_wrapper, [[maybe_unused]] const logging::Logger& logger) const {
  return CreateOrValidateOnQnn(&qnn_model_wrapper, GetNodeUnits(), logger, true);
}

Status CastLoneQFusion::AddToModelBuilder(
    QnnModelWrapper& qnn_model_wrapper, [[maybe_unused]] const logging::Logger& logger) const {
  return CreateOrValidateOnQnn(&qnn_model_wrapper, GetNodeUnits(), logger, false);
}

}  // namespace qnn
}  // namespace onnxruntime