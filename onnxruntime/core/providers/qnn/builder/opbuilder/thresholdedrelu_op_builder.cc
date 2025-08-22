// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <unordered_set>
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/cpu/tensor/slice_helper.h"

namespace onnxruntime {
namespace qnn {
class ThresholdedReluOpBuilder : public BaseOpBuilder {
 public:
  ThresholdedReluOpBuilder() : BaseOpBuilder("ThresholdedReluOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ThresholdedReluOpBuilder);

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

 private:
  Status ExplictOpCheck(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit) const;
};

Status ThresholdedReluOpBuilder::ExplictOpCheck(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit) const {
  TensorInfo input_info = {};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Inputs()[0], input_info));

  // Greater op supporting input dtypes
  static const std::unordered_set<int> greater_op_support_dtypes = {
      QNN_DATATYPE_FLOAT_16,
      QNN_DATATYPE_FLOAT_32,
      QNN_DATATYPE_UFIXED_POINT_16,
      QNN_DATATYPE_SFIXED_POINT_16,
      QNN_DATATYPE_UFIXED_POINT_8,
      QNN_DATATYPE_SFIXED_POINT_8,
      QNN_DATATYPE_INT_32};

  if (greater_op_support_dtypes.count(input_info.qnn_data_type) == 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "ThresholdRelu input data type not supported.");
  }

  return Status::OK();
}

static Status SetAlphaByte(Qnn_DataType_t qnn_data_type,
                           std::vector<uint8_t>& alpha_bytes,
                           float alpha_value) {
  switch (qnn_data_type) {
    case QNN_DATATYPE_FLOAT_16: {
      MLFloat16 zero_fp16 = static_cast<MLFloat16>(alpha_value);
      uint16_t cast_value = *reinterpret_cast<uint16_t*>(&zero_fp16);
      alpha_bytes.resize(sizeof(uint16_t));
      std::memcpy(alpha_bytes.data(), &cast_value, sizeof(uint16_t));
      break;
    }
    case QNN_DATATYPE_FLOAT_32: {
      float cast_value = static_cast<float>(alpha_value);
      alpha_bytes.resize(sizeof(float));
      std::memcpy(alpha_bytes.data(), &cast_value, sizeof(float));
      break;
    }
    case QNN_DATATYPE_UFIXED_POINT_16: {
      uint16_t cast_value = static_cast<uint16_t>(alpha_value);
      alpha_bytes.resize(sizeof(uint16_t));
      std::memcpy(alpha_bytes.data(), &cast_value, sizeof(uint16_t));
      break;
    }
    case QNN_DATATYPE_SFIXED_POINT_16: {
      int16_t cast_value = static_cast<int16_t>(alpha_value);
      alpha_bytes.resize(sizeof(int16_t));
      std::memcpy(alpha_bytes.data(), &cast_value, sizeof(int16_t));
      break;
    }
    case QNN_DATATYPE_UFIXED_POINT_8: {
      uint8_t cast_value = static_cast<uint8_t>(alpha_value);
      alpha_bytes.resize(sizeof(uint8_t));
      std::memcpy(alpha_bytes.data(), &cast_value, sizeof(uint8_t));
      break;
    }
    case QNN_DATATYPE_SFIXED_POINT_8: {
      int8_t cast_value = static_cast<int8_t>(alpha_value);
      alpha_bytes.resize(sizeof(int8_t));
      std::memcpy(alpha_bytes.data(), &cast_value, sizeof(int8_t));
      break;
    }
    case QNN_DATATYPE_INT_32: {
      int32_t cast_value = static_cast<int32_t>(alpha_value);
      alpha_bytes.resize(sizeof(int32_t));
      std::memcpy(alpha_bytes.data(), &cast_value, sizeof(int32_t));
      break;
    }
    default: {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported QNN Data type for thresholdedrelu.");
    }
  }

  return Status::OK();
}

Status ThresholdedReluOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                               const NodeUnit& node_unit,
                                               const logging::Logger& logger,
                                               std::vector<std::string>& input_names,
                                               bool do_op_validation) const {
  if (do_op_validation) {
    ORT_RETURN_IF_ERROR(ExplictOpCheck(qnn_model_wrapper, node_unit));
  }
  NodeAttrHelper node_helper(node_unit);
  const auto& inputs = node_unit.Inputs();
  const auto input_count = GetInputCountQnnRequired(node_unit);
  for (size_t input_i = 0; input_i < input_count; ++input_i) {
    ORT_RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[input_i], logger, input_names));
  }

  return Status::OK();
}

Status ThresholdedReluOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                             const NodeUnit& node_unit,
                                                             std::vector<std::string>&& input_names,
                                                             const logging::Logger& logger,
                                                             bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(logger);
  TensorInfo input_info = {};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Inputs()[0], input_info));
  TensorInfo output_info = {};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Outputs()[0], output_info));

  NodeAttrHelper node_helper(node_unit);
  std::string& input_name = input_names[0];
  const std::string& org_output_name = node_unit.Outputs()[0].node_arg.Name();
  const bool is_graph_output = qnn_model_wrapper.IsGraphOutput(org_output_name);

  std::vector<uint32_t> output_shape = output_info.shape;
  Qnn_TensorType_t op_output_tensor_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;

  // Create alpha tensor.
  // QNN sub gives memory error, use input + (-alpha) as input - alpha's work around.
  float negtive_alpha = node_helper.Get("alpha", static_cast<float>(0)) * -1;
  std::vector<uint8_t> alpha_bytes;
  ORT_RETURN_IF_ERROR(SetAlphaByte(input_info.qnn_data_type, alpha_bytes, negtive_alpha));

  std::string negtive_alpha_tensor_name = utils::GetUniqueName(node_unit, "_alpha");
  QnnTensorWrapper negtive_alpha_tensorwrapper(negtive_alpha_tensor_name,
                                               QNN_TENSOR_TYPE_STATIC,
                                               input_info.qnn_data_type,
                                               QnnQuantParamsWrapper(),
                                               std::vector<uint32_t>({1}),
                                               std::move(alpha_bytes));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(negtive_alpha_tensorwrapper)), "Failed to add tensor.");

  // input -> add -> relu -> sign -> mul -> output
  //       --------------------------/
  // 1. Add
  std::string add_name = utils::GetUniqueName(node_unit, "_Add");
  std::string add_output_name = utils::GetUniqueName(node_unit, "_Add_output");
  QnnTensorWrapper add_output(add_output_name,
                              QNN_TENSOR_TYPE_NATIVE,
                              input_info.qnn_data_type,
                              QnnQuantParamsWrapper(),
                              std::vector<uint32_t>(output_shape));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(add_output)),
                    "Failed to add ThresholdRelu - Sub output tensor.");

  ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(add_name,
                                                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                    QNN_OP_ELEMENT_WISE_ADD,
                                                    {input_name, negtive_alpha_tensor_name},
                                                    {add_output_name},
                                                    {},
                                                    do_op_validation),
                    "Failed to add ThresholdRelu - Sub node.");

  // 2. Relu
  std::string relu_name = utils::GetUniqueName(node_unit, "_Relu");
  std::string relu_output_name = utils::GetUniqueName(node_unit, "_Relu_output");

  QnnTensorWrapper relu_output(relu_output_name,
                               QNN_TENSOR_TYPE_NATIVE,
                               input_info.qnn_data_type,
                               QnnQuantParamsWrapper(),
                               std::vector<uint32_t>(output_shape));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(relu_output)),
                    "Failed to add ThresholdRelu - Relu output tensor.");

  ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(relu_name,
                                                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                    QNN_OP_RELU,
                                                    {add_output_name},
                                                    {relu_output_name},
                                                    {},
                                                    do_op_validation),
                    "Failed to add ThresholdRelu - Relu node.");

  // 3. Sign
  std::string sign_name = utils::GetUniqueName(node_unit, "_Sign");
  std::string sign_output_name = utils::GetUniqueName(node_unit, "_Sign_output");
  QnnTensorWrapper sign_output(sign_output_name,
                               QNN_TENSOR_TYPE_NATIVE,
                               input_info.qnn_data_type,
                               QnnQuantParamsWrapper(),
                               std::vector<uint32_t>(output_shape));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(sign_output)),
                    "Failed to add ThresholdRelu - Sign output tensor.");

  ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(sign_name,
                                                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                    QNN_OP_ELEMENT_WISE_SIGN,
                                                    {relu_output_name},
                                                    {sign_output_name},
                                                    {},
                                                    do_op_validation),
                    "Failed to add ThresholdRelu - Sign node.");

  // 4. Mul
  std::string mul_name = utils::GetUniqueName(node_unit, "_Mul");
  QnnTensorWrapper mul_output(org_output_name,
                              op_output_tensor_type,
                              output_info.qnn_data_type,
                              output_info.quant_param.Copy(),
                              std::vector<uint32_t>(output_shape));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(mul_output)),
                    "Failed to add ThresholdRelu - Mul output tensor.");

  ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(mul_name,
                                                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                    QNN_OP_ELEMENT_WISE_MULTIPLY,
                                                    {input_name, sign_output_name},
                                                    {org_output_name},
                                                    {},
                                                    do_op_validation),
                    "Failed to add ThresholdRelu - Mul node.");

  return Status::OK();
}

void CreateThresholdedReluOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<ThresholdedReluOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
