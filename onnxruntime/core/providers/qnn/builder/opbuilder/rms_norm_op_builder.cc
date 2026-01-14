// Copyright (c) Qualcomm. All rights reserved.
// Licensed under the MIT License.

#include <cassert>

#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"

namespace onnxruntime {
namespace qnn {

class RMSNormOpBuilder : public BaseOpBuilder {
 public:
  RMSNormOpBuilder() : BaseOpBuilder("RMSNormOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(RMSNormOpBuilder);

  Status IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                       const NodeUnit& node_unit,
                       const logging::Logger& logger) const override final ORT_MUST_USE_RESULT;

 protected:
  Status ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                       const NodeUnit& node_unit,
                       const logging::Logger& logger,
                       std::vector<std::string>& input_names,
                       bool do_op_validation) const override ORT_MUST_USE_RESULT;
  Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                     const NodeUnit& node_unit,
                                     std::vector<std::string>&& input_names,
                                     const logging::Logger& logger,
                                     bool do_op_validation) const override ORT_MUST_USE_RESULT;
};

Status RMSNormOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                       const NodeUnit& node_unit,
                                       const logging::Logger& logger) const {
  const auto& inputs = node_unit.Inputs();
  const auto& outputs = node_unit.Outputs();

  // Validate scale input is present
  constexpr size_t SCALE_IDX = 1;
  const bool has_scale_input = inputs.size() > SCALE_IDX && inputs[SCALE_IDX].node_arg.Exists();
  ORT_RETURN_IF_NOT(has_scale_input, "QNN EP requires scale input for RMSNorm operator");

  // Validate input and output rank constraints
  std::vector<uint32_t> input_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[0].node_arg, input_shape), "Cannot get shape of input 0");
  const size_t input_rank = input_shape.size();
  ORT_RETURN_IF(input_rank > 4, "QNN RMSNorm only supports input rank <= 4");

  std::vector<uint32_t> output_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(outputs[0].node_arg, output_shape), "Cannot get shape of output 0");
  const size_t output_rank = output_shape.size();
  ORT_RETURN_IF(output_rank > 4, "QNN RMSNorm only supports output rank <= 4");

  // Additional constraints for NPU backend
  bool is_npu_backend = IsNpuBackend(qnn_model_wrapper.GetQnnBackendType());
  if (is_npu_backend) {
    int32_t axis = -1;
    Qnn_Scalar_t axis_qnn_scalar = QNN_SCALAR_INIT;
    ORT_RETURN_IF_ERROR(ProcessAxisAttribute(qnn_model_wrapper, node_unit, axis_qnn_scalar, axis));
    ORT_RETURN_IF(static_cast<size_t>(axis) != input_rank - 1,
                  "QNN RMSNorm for NPU backend only supports axis with last input dimension");
  }

  return AddToModelBuilder(qnn_model_wrapper, node_unit, logger, true);
}

Status RMSNormOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                       const NodeUnit& node_unit,
                                       const logging::Logger& logger,
                                       std::vector<std::string>& input_names,
                                       bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(do_op_validation);

  const auto& inputs = node_unit.Inputs();
  constexpr size_t X_IDX = 0;
  constexpr size_t SCALE_IDX = 1;

  ORT_RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[X_IDX], logger, input_names));
  ORT_RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[SCALE_IDX], logger, input_names));

  // Create dummy beta tensor for NPU backend
  bool is_npu_backend = IsNpuBackend(qnn_model_wrapper.GetQnnBackendType());
  if (is_npu_backend) {
    TensorInfo scale_info = {};
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[SCALE_IDX], scale_info));

    std::vector<uint32_t> beta_shape = scale_info.shape;

    // Match beta datatype to scale for float types, use UFIXED_POINT_8 for INT types
    Qnn_DataType_t beta_data_type = QNN_DATATYPE_UFIXED_POINT_8;
    if (scale_info.qnn_data_type == QNN_DATATYPE_FLOAT_32 ||
        scale_info.qnn_data_type == QNN_DATATYPE_FLOAT_16) {
      beta_data_type = scale_info.qnn_data_type;
    }

    // Use appropriate quantization parameters for zero values
    QnnQuantParamsWrapper beta_quant_param;
    if (scale_info.quant_param.IsQuantized()) {
      float quant_scale = 1.0f;
      int32_t zero_point = 0;
      beta_quant_param = QnnQuantParamsWrapper(quant_scale, zero_point);
    }

    const size_t beta_size_in_bytes = utils::GetQnnTensorDataSizeInBytes(beta_shape, beta_data_type);
    std::vector<uint8_t> beta_data(beta_size_in_bytes, 0);
    const std::string beta_tensor_name = node_unit.Name() + "_beta_dummy";
    QnnTensorWrapper beta_tensor_wrapper(beta_tensor_name,
                                         QNN_TENSOR_TYPE_STATIC,
                                         beta_data_type,
                                         std::move(beta_quant_param),
                                         std::move(beta_shape),
                                         std::move(beta_data));

    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(beta_tensor_wrapper)),
                      "Failed to add dummy beta tensor for QNN RMSNorm node.");
    input_names.push_back(beta_tensor_name);
  }

  return Status::OK();
}

Status RMSNormOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                     const NodeUnit& node_unit,
                                                     std::vector<std::string>&& input_names,
                                                     const logging::Logger& logger,
                                                     bool do_op_validation) const {
  NodeAttrHelper node_helper(node_unit);
  std::vector<std::string> param_tensor_names;

  // Process epsilon attribute
  const float epsilon = node_helper.Get("epsilon", 1e-05f);
  Qnn_Scalar_t epsilon_param = QNN_SCALAR_INIT;
  epsilon_param.dataType = QNN_DATATYPE_FLOAT_32;
  epsilon_param.floatValue = epsilon;
  QnnParamWrapper epsilon_param_wrapper(node_unit.Index(),
                                        node_unit.Name(),
                                        QNN_OP_RMS_NORM_PARAM_EPSILON,
                                        epsilon_param);
  param_tensor_names.push_back(epsilon_param_wrapper.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(epsilon_param_wrapper));

  // Process axis attribute and create axes parameter
  std::vector<uint32_t> input_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(node_unit.Inputs()[0].node_arg, input_shape), "Cannot get shape of Input 0");
  const size_t input_rank = input_shape.size();
  int32_t axis = -1;
  Qnn_Scalar_t axis_qnn_scalar = QNN_SCALAR_INIT;
  ORT_RETURN_IF_ERROR(ProcessAxisAttribute(qnn_model_wrapper, node_unit, axis_qnn_scalar, axis));
  size_t axes_rank = input_rank - static_cast<size_t>(axis);
  std::vector<uint32_t> axes(axes_rank, 0);
  std::vector<uint32_t> axes_shape{SafeInt<uint32_t>(axes_rank)};
  axes[0] = static_cast<uint32_t>(axis);
  for (size_t i = 1; i < axes.size(); ++i) {
    axes[i] = axes[i - 1] + 1;
  }

  QnnParamWrapper axes_param(node_unit.Index(), node_unit.Name(), QNN_OP_RMS_NORM_PARAM_AXES,
                             std::move(axes_shape), std::move(axes));
  param_tensor_names.push_back(axes_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(axes_param));

  ORT_RETURN_IF_ERROR(ProcessOutputs(qnn_model_wrapper, node_unit,
                                     std::move(input_names),
                                     std::move(param_tensor_names),
                                     logger,
                                     do_op_validation,
                                     GetQnnOpType(node_unit.OpType())));
  return Status::OK();
}

void CreateRMSNormOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<RMSNormOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
