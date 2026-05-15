// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

// FusedMatMul operator is decomposed into MatMul with optional transposition and alpha scaling.
class FusedMatMulOpBuilder : public BaseOpBuilder {
 public:
  FusedMatMulOpBuilder() : BaseOpBuilder("FusedMatMulOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(FusedMatMulOpBuilder);

 protected:
  Status ProcessInputs(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit, const logging::Logger& logger,
                       std::vector<std::string>& input_names, bool do_op_validation) const override ORT_MUST_USE_RESULT;

  Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit,
                                     std::vector<std::string>&& input_names, const logging::Logger& logger,
                                     bool do_op_validation) const override ORT_MUST_USE_RESULT;

 private:
  Status ProcessMatMulInputs(QnnModelWrapper& qnn_model_wrapper,
                             const NodeUnit& node_unit,
                             const logging::Logger& logger,
                             std::vector<std::string>& input_names) const ORT_MUST_USE_RESULT;

  Status GetFusedMatMulAttributes(const NodeUnit& node_unit,
                                  bool& transA,
                                  bool& transB,
                                  bool& transBatchA,
                                  bool& transBatchB,
                                  float& alpha) const ORT_MUST_USE_RESULT;

  Status ProcessPermAttribute(QnnModelWrapper& qnn_model_wrapper,
                              const NodeUnit& node_unit,
                              const std::vector<uint32_t>& perm,
                              std::vector<std::string>& param_tensor_names) const;

  void CreateBatchTransposePermVector(const std::vector<uint32_t>& input_shape, std::vector<uint32_t>& perm, bool trans_mat = false) const;

  Status HandleBatchTranspose(QnnModelWrapper& qnn_model_wrapper,
                              const NodeUnit& node_unit,
                              const TensorInfo& input_info,
                              const std::string& input_name,
                              std::string& transposed_name,
                              bool trans_mat,
                              bool do_op_validation) const;
};

Status FusedMatMulOpBuilder::GetFusedMatMulAttributes(const NodeUnit& node_unit,
                                                      bool& transA,
                                                      bool& transB,
                                                      bool& transBatchA,
                                                      bool& transBatchB,
                                                      float& alpha) const {
  NodeAttrHelper node_helper(node_unit);

  transA = node_helper.Get("transA", static_cast<int64_t>(0)) != 0;
  transB = node_helper.Get("transB", static_cast<int64_t>(0)) != 0;

  transBatchA = node_helper.Get("transBatchA", static_cast<int64_t>(0)) != 0;
  transBatchB = node_helper.Get("transBatchB", static_cast<int64_t>(0)) != 0;

  alpha = node_helper.Get("alpha", 1.0f);

  return Status::OK();
}

Status FusedMatMulOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit,
                                           const logging::Logger& logger, std::vector<std::string>& input_names,
                                           bool /*do_op_validation*/) const {
  const auto& inputs = node_unit.Inputs();

  if (inputs.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "FusedMatMul requires exactly 2 inputs, got ", inputs.size());
  }

  TensorInfo input_info_0{};
  TensorInfo input_info_1{};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[0], input_info_0));
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[1], input_info_1));

  ORT_RETURN_IF_ERROR(ProcessMatMulInputs(qnn_model_wrapper, node_unit, logger, input_names));

  return Status::OK();
}

Status FusedMatMulOpBuilder::ProcessMatMulInputs(QnnModelWrapper& qnn_model_wrapper,
                                                 const NodeUnit& node_unit,
                                                 const logging::Logger& logger,
                                                 std::vector<std::string>& input_names) const {
  const auto& inputs = node_unit.Inputs();

  // Process input A
  const std::string& input_a_name = inputs[0].node_arg.Name();
  if (qnn_model_wrapper.IsQnnTensorWrapperExist(input_a_name)) {
    LOGS(logger, VERBOSE) << "Tensor already added, skip it: " << input_a_name;
  } else {
    QnnTensorWrapper input_a_tensor;
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(inputs[0], input_a_tensor));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_a_tensor)), "Failed to add input A tensor.");
  }
  input_names.emplace_back(input_a_name);

  // Process input B
  const std::string& input_b_name = inputs[1].node_arg.Name();
  if (qnn_model_wrapper.IsQnnTensorWrapperExist(input_b_name)) {
    LOGS(logger, VERBOSE) << "Tensor already added, skip it: " << input_b_name;
  } else {
    QnnTensorWrapper input_b_tensor;
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(inputs[1], input_b_tensor));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_b_tensor)), "Failed to add input B tensor.");
  }
  input_names.emplace_back(input_b_name);

  return Status::OK();
}

Status FusedMatMulOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                         const NodeUnit& node_unit,
                                                         std::vector<std::string>&& input_names,
                                                         const logging::Logger& /*logger*/,
                                                         bool do_op_validation) const {
  bool transA = false;
  bool transB = false;
  bool transBatchA = false;
  bool transBatchB = false;
  float alpha = 1.0f;
  ORT_RETURN_IF_ERROR(GetFusedMatMulAttributes(node_unit, transA, transB, transBatchA, transBatchB, alpha));

  TensorInfo input_a_info{};
  TensorInfo input_b_info{};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Inputs()[0], input_a_info));
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Inputs()[1], input_b_info));

  std::vector<std::string> matmul_param_tensor_names;

  // Set transpose parameters for last two dimensions
  // Skip using transpose_in0 param when both transA and transBatchA are present
  // Only use transpose_in0 when transA is present and transBatchA is not present
  if (!(transA && transBatchA)) {
    Qnn_Scalar_t transpose_a_scalar = QNN_SCALAR_INIT;
    transpose_a_scalar.dataType = QNN_DATATYPE_BOOL_8;
    transpose_a_scalar.bool8Value = transA ? 1 : 0;
    QnnParamWrapper transpose_a_param(node_unit.Index(), node_unit.Name(),
                                      QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0, transpose_a_scalar);
    matmul_param_tensor_names.push_back(transpose_a_param.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(transpose_a_param));
  }

  // Skip using transpose_in1 param when both transB and transBatchB are present
  // Only use transpose_in1 when transB is present and transBatchB is not present
  if (!(transB && transBatchB)) {
    Qnn_Scalar_t transpose_b_scalar = QNN_SCALAR_INIT;
    transpose_b_scalar.dataType = QNN_DATATYPE_BOOL_8;
    transpose_b_scalar.bool8Value = transB ? 1 : 0;
    QnnParamWrapper transpose_b_param(node_unit.Index(), node_unit.Name(),
                                      QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1, transpose_b_scalar);
    matmul_param_tensor_names.push_back(transpose_b_param.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(transpose_b_param));
  }

  // QNN doesn't directly support batch dimension transposition in MatMul
  // We need to insert additional transpose operations before the MatMul if transBatchA or transBatchB is true
  std::string input_a_for_matmul = input_names[0];
  std::string input_b_for_matmul = input_names[1];

  if (transBatchA && input_a_info.shape.size() > 2) {
    std::string transposed_a_name;
    ORT_RETURN_IF_ERROR(HandleBatchTranspose(qnn_model_wrapper, node_unit, input_a_info,
                                             input_a_for_matmul, transposed_a_name, transA, do_op_validation));
    input_a_for_matmul = transposed_a_name;
  }

  if (transBatchB && input_b_info.shape.size() > 2) {
    std::string transposed_b_name;
    ORT_RETURN_IF_ERROR(HandleBatchTranspose(qnn_model_wrapper, node_unit, input_b_info,
                                             input_b_for_matmul, transposed_b_name, transB, do_op_validation));
    input_b_for_matmul = transposed_b_name;
  }

  const std::string& output_name = node_unit.Outputs()[0].node_arg.Name();
  TensorInfo output_info{};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Outputs()[0], output_info));

  if (alpha == 1.0f) {
    // When alpha is 1.0f, MatMul output is the final output
    Qnn_TensorType_t tensor_type = qnn_model_wrapper.IsGraphOutput(output_name) ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;

    QnnTensorWrapper output_tensor(output_name,
                                   tensor_type,
                                   output_info.qnn_data_type,
                                   output_info.quant_param.Copy(),
                                   std::vector<uint32_t>(output_info.shape));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensor)),
                      "Failed to add final output tensor.");

    ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                          utils::GetUniqueName(node_unit.Name() + "_matmul"),
                          QNN_OP_PACKAGE_NAME_QTI_AISW,
                          QNN_OP_MAT_MUL,
                          {input_a_for_matmul, input_b_for_matmul},
                          {output_name},
                          std::move(matmul_param_tensor_names),
                          do_op_validation),
                      "Failed to create MatMul node for FusedMatMul.");
  } else {
    // When alpha is not 1.0f, we need an intermediate tensor for MatMul output
    // and then apply alpha scaling
    std::string matmul_output_name = utils::GetUniqueName(node_unit.Name() + "_matmul_output");

    QnnTensorWrapper matmul_output_tensor(matmul_output_name,
                                          QNN_TENSOR_TYPE_NATIVE,
                                          output_info.qnn_data_type,
                                          QnnQuantParamsWrapper(),
                                          std::vector<uint32_t>(output_info.shape));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(matmul_output_tensor)),
                      "Failed to add MatMul output tensor.");

    Qnn_TensorType_t tensor_type = qnn_model_wrapper.IsGraphOutput(output_name) ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;

    QnnTensorWrapper output_tensor(output_name,
                                   tensor_type,
                                   output_info.qnn_data_type,
                                   output_info.quant_param.Copy(),
                                   std::vector<uint32_t>(output_info.shape));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensor)),
                      "Failed to add output tensor.");

    ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                          utils::GetUniqueName(node_unit.Name() + "_matmul"),
                          QNN_OP_PACKAGE_NAME_QTI_AISW,
                          QNN_OP_MAT_MUL,
                          {input_a_for_matmul, input_b_for_matmul},
                          {matmul_output_name},
                          std::move(matmul_param_tensor_names),
                          do_op_validation),
                      "Failed to create MatMul node for FusedMatMul.");

    std::string alpha_tensor_name = utils::GetUniqueName(node_unit.Name() + "_alpha");
    std::vector<uint32_t> alpha_shape{1};
    Qnn_DataType_t alpha_qnn_data_type = output_info.qnn_data_type;
    std::vector<uint8_t> alpha_data;

    // The alpha tensor data type should match the MatMul output data type for element-wise multiply
    if (alpha_qnn_data_type == QNN_DATATYPE_FLOAT_16) {
      alpha_data.resize(sizeof(MLFloat16));
      MLFloat16 alpha_fp16(alpha);
      memcpy(alpha_data.data(), &alpha_fp16.val, sizeof(MLFloat16));
    } else {
      alpha_data.resize(sizeof(float));
      memcpy(alpha_data.data(), &alpha, sizeof(float));
    }

    QnnTensorWrapper alpha_tensor_wrapper(alpha_tensor_name,
                                          QNN_TENSOR_TYPE_STATIC,
                                          alpha_qnn_data_type,
                                          QnnQuantParamsWrapper(),
                                          std::move(alpha_shape),
                                          std::move(alpha_data));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(alpha_tensor_wrapper)),
                      "Failed to add alpha tensor.");

    ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                          utils::GetUniqueName(node_unit.Name() + "_alpha_scale"),
                          QNN_OP_PACKAGE_NAME_QTI_AISW,
                          QNN_OP_ELEMENT_WISE_MULTIPLY,
                          {matmul_output_name, alpha_tensor_name},
                          {output_name},
                          {},
                          do_op_validation),
                      "Failed to create alpha scaling node for FusedMatMul.");
  }

  return Status::OK();
}

Status FusedMatMulOpBuilder::ProcessPermAttribute(QnnModelWrapper& qnn_model_wrapper,
                                                  const NodeUnit& node_unit,
                                                  const std::vector<uint32_t>& perm,
                                                  std::vector<std::string>& param_tensor_names) const {
  QnnParamWrapper transpose_param(node_unit.Index(), node_unit.Name(), QNN_OP_TRANSPOSE_PARAM_PERM,
                                  {static_cast<uint32_t>(perm.size())}, std::vector<uint32_t>(perm));
  param_tensor_names.push_back(transpose_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(transpose_param));

  return Status::OK();
}

void FusedMatMulOpBuilder::CreateBatchTransposePermVector(const std::vector<uint32_t>& input_shape,
                                                          std::vector<uint32_t>& perm,
                                                          bool trans_mat) const {
  const size_t shape_size = input_shape.size();

  perm.clear();
  perm.reserve(shape_size);

  // 1. Add batch dimensions (1 to shape_size-2)
  for (size_t i = 1; i < shape_size - 1; ++i) {
    perm.push_back(static_cast<uint32_t>(i));
  }

  // 2. Add the second-to-last dimension based on trans_mat
  perm.push_back(trans_mat ? static_cast<uint32_t>(shape_size - 1) : 0);

  // 3. Add the last dimension based on trans_mat
  perm.push_back(trans_mat ? 0 : static_cast<uint32_t>(shape_size - 1));
}

Status FusedMatMulOpBuilder::HandleBatchTranspose(QnnModelWrapper& qnn_model_wrapper,
                                                  const NodeUnit& node_unit,
                                                  const TensorInfo& input_info,
                                                  const std::string& input_name,
                                                  std::string& transposed_name,
                                                  bool trans_mat,
                                                  bool do_op_validation) const {
  transposed_name = utils::GetUniqueName(node_unit.Name() + "_transposed_" + input_name.substr(input_name.find_last_of('/') + 1));

  // Create perm vector for batch transpose
  std::vector<uint32_t> perm;
  CreateBatchTransposePermVector(input_info.shape, perm, trans_mat);

  std::vector<std::string> transpose_params;
  ORT_RETURN_IF_ERROR(ProcessPermAttribute(qnn_model_wrapper, node_unit, perm, transpose_params));

  // Calculate transposed shape directly using the permutation
  std::vector<uint32_t> transposed_shape(input_info.shape.size());
  for (size_t i = 0; i < perm.size(); ++i) {
    transposed_shape[i] = input_info.shape[perm[i]];
  }

  QnnTensorWrapper transposed_tensor(transposed_name,
                                     QNN_TENSOR_TYPE_NATIVE,
                                     input_info.qnn_data_type,
                                     input_info.quant_param.Copy(),
                                     std::move(transposed_shape));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(transposed_tensor)),
                    "Failed to add transposed tensor.");

  ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                        utils::GetUniqueName(node_unit.Name() + "_transpose_" + input_name.substr(input_name.find_last_of('/') + 1)),
                        QNN_OP_PACKAGE_NAME_QTI_AISW,
                        QNN_OP_TRANSPOSE,
                        {input_name},
                        {transposed_name},
                        std::move(transpose_params),
                        do_op_validation),
                    "Failed to create transpose node.");

  return Status::OK();
}

void CreateFusedMatMulOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<FusedMatMulOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
