// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/framework/tensorprotoutils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/common/safeint.h"
#include "core/util/qmath.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace qnn {

// Operator which only need to hanle node inputs & outputs, no attributes or no need to handle attributes
class SimpleOpBuilder : public BaseOpBuilder {
 public:
  SimpleOpBuilder() : BaseOpBuilder("SimpleOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(SimpleOpBuilder);

 protected:
  Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                     const NodeUnit& node_unit,
                                     std::vector<std::string>&& input_names,
                                     const logging::Logger& logger,
                                     bool is_quantized_model,
                                     bool do_op_validation) const override ORT_MUST_USE_RESULT;

 private:
  Status ExplictOpCheck(const QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit) const;
  Status ProcessPermAttribute(QnnModelWrapper& qnn_model_wrapper,
                              const NodeUnit& node_unit,
                              std::vector<std::string>& param_tensor_names) const;
  Status ProcessAlphaAttribute(QnnModelWrapper& qnn_model_wrapper,
                               const NodeUnit& node_unit,
                               std::vector<std::string>& param_tensor_names) const;
  Status ProcessAlphaAttributeAsInput(QnnModelWrapper& qnn_model_wrapper,
                                      const NodeUnit& node_unit,
                                      const std::string input_name,
                                      bool is_quantized_model) const;
  Status HandleSingleTransposeNode(QnnModelWrapper& qnn_model_wrapper,
                                   const NodeUnit& node_unit,
                                   std::vector<std::string>&& input_names,
                                   bool is_quantized_model) const;
};

Status SimpleOpBuilder::ExplictOpCheck(const QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit) const {
  // QNN Softmax only supports an axis value equal to input_rank - 1 (i.e., same as -1).
  if (node_unit.OpType() == "Softmax") {
    int32_t axis = node_unit.SinceVersion() < 13 ? 1 : -1;  // Default axis changed from 1 to -1 in opset 13.
    Qnn_Scalar_t axis_qnn_scalar = QNN_SCALAR_INIT;
    ORT_RETURN_IF_ERROR(ProcessAxisAttribute(qnn_model_wrapper, node_unit, axis_qnn_scalar, axis));
    std::vector<uint32_t> input_shape;
    ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(node_unit.Inputs()[0].node_arg, input_shape),
                      "QNN EP: Cannot get shape for Softmax input");
    ORT_RETURN_IF(axis != static_cast<int32_t>(input_shape.size() - 1),
                  "QNN Softmax only supports an `axis` attribute equal to input_rank-1 (or -1)");
  }

  return Status::OK();
}

Status SimpleOpBuilder::ProcessPermAttribute(QnnModelWrapper& qnn_model_wrapper,
                                             const NodeUnit& node_unit,
                                             std::vector<std::string>& param_tensor_names) const {
  auto inputs = node_unit.Inputs();
  std::vector<uint32_t> input_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[0].node_arg, input_shape), "Cannot get shape");
  // set default perm
  uint32_t rank = static_cast<uint32_t>(input_shape.size());
  std::vector<int64_t> transpose_perm(rank);
  for (uint32_t i = 0; i < rank; ++i) {
    transpose_perm[i] = rank - 1 - i;
  }

  NodeAttrHelper node_helper(node_unit);
  transpose_perm = node_helper.Get("perm", transpose_perm);
  auto perm_size = static_cast<uint32_t>(transpose_perm.size());
  std::vector<uint32_t> perm_shape{perm_size};
  std::vector<uint32_t> perm_data;
  perm_data.resize(perm_size);
  std::transform(transpose_perm.begin(), transpose_perm.end(), perm_data.begin(),
                 [](int64_t item) { return SafeInt<uint32_t>(item); });

  QnnParamWrapper transpose_param(node_unit.Index(), node_unit.Name(), QNN_OP_TRANSPOSE_PARAM_PERM,
                                  std::move(perm_shape), std::move(perm_data));
  param_tensor_names.push_back(transpose_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(transpose_param));

  return Status::OK();
}

Status SimpleOpBuilder::ProcessAlphaAttribute(QnnModelWrapper& qnn_model_wrapper,
                                              const NodeUnit& node_unit,
                                              std::vector<std::string>& param_tensor_names) const {
  NodeAttrHelper node_helper(node_unit);
  float alpha = node_helper.Get("alpha", 1.0f);
  Qnn_Scalar_t alpha_qnn_scalar = QNN_SCALAR_INIT;
  alpha_qnn_scalar.dataType = QNN_DATATYPE_FLOAT_32;
  alpha_qnn_scalar.floatValue = alpha;

  QnnParamWrapper alpha_param(node_unit.Index(), node_unit.Name(), QNN_OP_ELU_PARAM_ALPHA, alpha_qnn_scalar);
  param_tensor_names.push_back(alpha_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(alpha_param));

  return Status::OK();
}

Status SimpleOpBuilder::ProcessAlphaAttributeAsInput(QnnModelWrapper& qnn_model_wrapper,
                                                     const NodeUnit& node_unit,
                                                     const std::string input_name,
                                                     bool is_quantized_model) const {
  NodeAttrHelper node_helper(node_unit);
  Qnn_QuantizeParams_t quantize_param = QNN_QUANTIZE_PARAMS_INIT;
  Qnn_DataType_t qnn_data_type = QNN_DATATYPE_FLOAT_32;
  union {
    float alpha;
    uint8_t unpack[sizeof(float)];
  } tensor_data;
  tensor_data.alpha = node_helper.Get("alpha", 0.01f);
  std::vector<uint8_t> unpacked_data;
  if (is_quantized_model) {
    float scale;
    uint8_t zero_point;
    int64_t num_of_elements = 1;
    concurrency::ThreadPool* thread_pool = nullptr;
    GetQuantizationParameter(&tensor_data.alpha, num_of_elements, scale, zero_point, thread_pool);
    unpacked_data.resize(1);
    ParQuantizeLinearStd(&tensor_data.alpha, unpacked_data.data(), num_of_elements, scale, zero_point, thread_pool);
    utils::InitializeQuantizeParam(quantize_param, is_quantized_model, scale, static_cast<int32_t>(zero_point));
    qnn_data_type = QNN_DATATYPE_UFIXED_POINT_8;
  } else {
    unpacked_data.assign(tensor_data.unpack, tensor_data.unpack + sizeof(float));
  }
  std::vector<uint32_t> input_shape{1};
  Qnn_TensorType_t tensor_type = QNN_TENSOR_TYPE_STATIC;
  QnnTensorWrapper input_tensorwrapper(input_name, tensor_type, qnn_data_type, quantize_param,
                                       std::move(input_shape), std::move(unpacked_data));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)), "Failed to add tensor.");
  return Status::OK();
}

// Support Transpose single node in QDQ model since it just change the data layout
// Single node doesn't has any quantization parameters
// Input tensors are created by the previous node. Output tensors are created by the next node,
// unless the output is the graph's final output.
Status SimpleOpBuilder::HandleSingleTransposeNode(QnnModelWrapper& qnn_model_wrapper,
                                                  const NodeUnit& node_unit,
                                                  std::vector<std::string>&& input_names,
                                                  bool is_quantized_model) const {
  std::vector<std::string> param_tensor_names;
  ORT_RETURN_IF_ERROR(ProcessPermAttribute(qnn_model_wrapper, node_unit, param_tensor_names));
  const auto& outputs = node_unit.Outputs();
  ORT_ENFORCE(outputs.size() == 1, "QNN Transpose node must have a single output.");
  const auto& output = outputs[0];
  auto& output_name = output.node_arg.Name();

  const bool is_graph_output = qnn_model_wrapper.IsGraphOutput(output_name);

  // Need to add output to the QNN model wrapper if this Transpose node's output is also
  // the graph's output.
  if (is_graph_output) {
    const auto* type_proto = output.node_arg.TypeAsProto();
    Qnn_DataType_t qnn_data_type = QNN_DATATYPE_UNDEFINED;
    ORT_RETURN_IF_ERROR(utils::GetQnnDataType(is_quantized_model, type_proto, qnn_data_type));

    Qnn_QuantizeParams_t quantize_param = QNN_QUANTIZE_PARAMS_INIT;
    std::vector<uint32_t> output_shape;
    ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(output.node_arg, output_shape),
                      "Cannot get shape for QNN Transpose output");

    QnnTensorWrapper output_tensorwrapper(output_name,
                                          QNN_TENSOR_TYPE_APP_READ,
                                          qnn_data_type,
                                          quantize_param,
                                          std::move(output_shape));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensorwrapper)),
                      "Failed to add output tensor for QNN Transpose");
  }

  ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(GetNodeName(node_unit),
                                                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                    GetQnnOpType(node_unit.OpType()),
                                                    std::move(input_names),
                                                    {output_name},
                                                    std::move(param_tensor_names)),
                    "Failed to add node.");
  return Status::OK();
}

Status SimpleOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                    const NodeUnit& node_unit,
                                                    std::vector<std::string>&& input_names,
                                                    const logging::Logger& logger,
                                                    bool is_quantized_model,
                                                    bool do_op_validation) const {
  if (input_names.size() < 1) {
    return Status::OK();
  }

  const std::string& op_type = node_unit.OpType();

  if (do_op_validation) {
    ORT_RETURN_IF_ERROR(ExplictOpCheck(qnn_model_wrapper, node_unit));
  } else if (is_quantized_model && NodeUnit::Type::SingleNode == node_unit.UnitType() &&
             op_type == "Transpose") {
    LOGS(logger, VERBOSE) << "Add single Transpose node: " << node_unit.Name();
    return HandleSingleTransposeNode(qnn_model_wrapper, node_unit, std::move(input_names), is_quantized_model);
  }

  std::vector<std::string> param_tensor_names;
  // Add attribute
  if (op_type == "LogSoftmax" || op_type == "Softmax" || op_type == "Concat") {
    int32_t default_axis = ("Softmax" == op_type) ? -1 : 0;
    Qnn_Scalar_t axis_qnn_scalar = QNN_SCALAR_INIT;
    ORT_RETURN_IF_ERROR(ProcessAxisAttribute(qnn_model_wrapper, node_unit, axis_qnn_scalar, default_axis));
    QnnParamWrapper axis_param(node_unit.Index(), node_unit.Name(), QNN_OP_SOFTMAX_PARAM_AXIS, axis_qnn_scalar);
    param_tensor_names.push_back(axis_param.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(axis_param));
  }

  if (op_type == "MatMul") {
    Qnn_Scalar_t scalar_param = QNN_SCALAR_INIT;
    scalar_param.dataType = QNN_DATATYPE_BOOL_8;
    scalar_param.bool8Value = 0;
    QnnParamWrapper transpose_in0_param(node_unit.Index(), node_unit.Name(), QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0, scalar_param);
    param_tensor_names.push_back(transpose_in0_param.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(transpose_in0_param));

    QnnParamWrapper transpose_in1_param(node_unit.Index(), node_unit.Name(), QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1, scalar_param);
    param_tensor_names.push_back(transpose_in1_param.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(transpose_in1_param));
  }

  if (op_type == "Transpose") {
    ORT_RETURN_IF_ERROR(ProcessPermAttribute(qnn_model_wrapper, node_unit, param_tensor_names));
  }

  if (op_type == "LeakyRelu") {
    std::string input_name = "alpha";
    ORT_RETURN_IF_ERROR(ProcessAlphaAttributeAsInput(qnn_model_wrapper, node_unit, input_name, is_quantized_model));
    input_names.push_back(input_name);
  }

  if (op_type == "Elu") {
    ORT_RETURN_IF_ERROR(ProcessAlphaAttribute(qnn_model_wrapper, node_unit, param_tensor_names));
  }

  ORT_RETURN_IF_ERROR(ProcessOutputs(qnn_model_wrapper, node_unit,
                                     std::move(input_names),
                                     std::move(param_tensor_names),
                                     logger, is_quantized_model, do_op_validation, GetQnnOpType(op_type)));
  return Status::OK();
}

void CreateSimpleOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<SimpleOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
