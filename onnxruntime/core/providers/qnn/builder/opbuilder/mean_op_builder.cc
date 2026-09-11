// Copyright (c) Qualcomm. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <set>
#include <string>
#include <vector>

#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

class MeanOpBuilder : public BaseOpBuilder {
 public:
  MeanOpBuilder() : BaseOpBuilder("MeanOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(MeanOpBuilder);

 protected:
  Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit,
                                     std::vector<std::string>&& input_names, const logging::Logger& logger,
                                     bool do_op_validation) const override ORT_MUST_USE_RESULT;
};

Status MeanOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit,
                                                  std::vector<std::string>&& input_names, const logging::Logger& logger,
                                                  bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(logger);
  ORT_UNUSED_PARAMETER(do_op_validation);

  const auto& inputs = node_unit.Inputs();
  const auto& output = node_unit.Outputs()[0];

  if (inputs.size() < 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Mean op requires at least two inputs.");
  }

  // Combine Add Operations together
  std::string sum_output = input_names[0];
  TensorInfo input_info = {};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[0], input_info));

  for (size_t i = 1; i < input_names.size(); ++i) {
    // Get output shape
    std::vector<uint32_t> output_shape;
    ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(output.node_arg, output_shape), "Failed to get output shape.");
    std::vector<uint8_t> unpackage_data(sizeof(float));

    const std::string add_output = utils::GetUniqueName(sum_output, "_add" + std::to_string(i));
    QnnTensorWrapper add_tensor(add_output, QNN_TENSOR_TYPE_NATIVE, input_info.qnn_data_type,
                                QnnQuantParamsWrapper(), std::move(output_shape));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(add_tensor)),
                      "Failed to add Add tensor wrapper.");
    ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::GetUniqueName(node_unit, QNN_OP_ELEMENT_WISE_ADD),
                                                      QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                      QNN_OP_ELEMENT_WISE_ADD,
                                                      {sum_output, input_names[i]},
                                                      {add_output},
                                                      {},
                                                      do_op_validation),
                      "Create Qnn Node for Add Op Failed");

    sum_output = add_output;
  }

  // Number of inputs to divide with
  float divisor = static_cast<float>(inputs.size());
  std::vector<uint32_t> scalar_shape = {1};
  std::vector<uint8_t> divisor_data(sizeof(float));
  memcpy(divisor_data.data(), &divisor, sizeof(float));

  const std::string divisor_name = utils::GetUniqueName(sum_output, "_divisor");

  QnnTensorWrapper divisor_tensor(divisor_name, QNN_TENSOR_TYPE_STATIC, input_info.qnn_data_type,
                                  QnnQuantParamsWrapper(), std::move(scalar_shape), std::move(divisor_data));

  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(divisor_tensor)), "AddTensorWrapper Failed");

  // Final step - Division
  const std::string output_name = output.node_arg.Name();
  std::vector<uint32_t> output_shape;
  TensorInfo output_info = {};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(output, output_info));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(output.node_arg, output_shape), "Failed to get output shape.");
  Qnn_TensorType_t output_tensor_type = qnn_model_wrapper.IsGraphOutput(output.node_arg.Name()) ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
  QnnTensorWrapper output_tensor(output_name, output_tensor_type, output_info.qnn_data_type,
                                 output_info.quant_param.Copy(), std::move(output_shape));

  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensor)),
                    "Failed to add output tensor wrapper.");
  std::vector<std::string> div_inputs = {sum_output, divisor_name};
  ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::GetUniqueName(node_unit, QNN_OP_ELEMENT_WISE_DIVIDE),
                                                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                    QNN_OP_ELEMENT_WISE_DIVIDE,
                                                    {sum_output, divisor_name},
                                                    {output_name},
                                                    {},
                                                    do_op_validation),
                    "Failed to create Mean_Div node.");

  return Status::OK();
}

void CreateMeanOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<MeanOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
