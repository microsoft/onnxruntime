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

class ReciprocalOpBuilder : public BaseOpBuilder {
 public:
  ReciprocalOpBuilder() : BaseOpBuilder("ReciprocalOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ReciprocalOpBuilder);

 protected:
  Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit,
                                     std::vector<std::string>&& input_names, const logging::Logger& logger,
                                     bool do_op_validation) const override ORT_MUST_USE_RESULT;
};

Status ReciprocalOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                        const NodeUnit& node_unit,
                                                        std::vector<std::string>&& input_names,
                                                        const logging::Logger& logger,
                                                        bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(logger);
  ORT_UNUSED_PARAMETER(do_op_validation);

  const auto& input = node_unit.Inputs()[0];
  const auto& output = node_unit.Outputs()[0];

  TensorInfo reciprocal_input_info = {};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(input, reciprocal_input_info));

  // Create constant tensor with value 1.0
  float one_value = 1.0f;
  std::vector<uint32_t> scalar_shape = {1};
  std::vector<uint8_t> one_data(sizeof(float));
  memcpy(one_data.data(), &one_value, sizeof(float));

  const std::string one_tensor_name = input_names[0] + "_ort_qnn_ep_one";

  QnnTensorWrapper one_tensor(one_tensor_name, QNN_TENSOR_TYPE_STATIC, reciprocal_input_info.qnn_data_type,
                              reciprocal_input_info.quant_param.Copy(), std::move(scalar_shape), std::move(one_data));

  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(one_tensor)),
                    "Failed to add constant tensor wrapper for reciprocal numerator.");

  // Get output shape
  std::vector<uint32_t> output_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(output.node_arg, output_shape),
                    "Failed to get output shape.");

  const std::string output_name = output.node_arg.Name();

  TensorInfo reciprocal_output_info = {};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(output, reciprocal_output_info));
  Qnn_TensorType_t output_tensor_type = qnn_model_wrapper.IsGraphOutput(output_name)
                                            ? QNN_TENSOR_TYPE_APP_READ
                                            : QNN_TENSOR_TYPE_NATIVE;

  QnnTensorWrapper output_tensor(output_name, output_tensor_type, reciprocal_output_info.qnn_data_type,
                                 reciprocal_output_info.quant_param.Copy(), std::move(output_shape));

  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensor)),
                    "Failed to add output tensor wrapper for reciprocal.");

  // Create Div node: 1 / input
  ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(output_name + "_div",
                                                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                    QNN_OP_ELEMENT_WISE_DIVIDE,
                                                    {one_tensor_name, input_names[0]},
                                                    {output_name},
                                                    {},
                                                    do_op_validation),
                    "Failed to create Reciprocal_Div node.");

  return Status::OK();
}

void CreateReciprocalOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<ReciprocalOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime