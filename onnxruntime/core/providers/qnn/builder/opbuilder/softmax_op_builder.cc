// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/framework/tensorprotoutils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/common/safeint.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace qnn {

class SoftmaxOpBuilder : public BaseOpBuilder {
 public:
  SoftmaxOpBuilder() : BaseOpBuilder("SoftmaxOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(SoftmaxOpBuilder);

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

constexpr int32_t GetDefaultAxisAttribute(int opset_version) {
  // Default axis changed from 1 to -1 in opset 13.
  return opset_version < 13 ? 1 : -1;
}

Status SoftmaxOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                       const NodeUnit& node_unit,
                                       const logging::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);
  const int opset_version = node_unit.SinceVersion();

  // The QNN HTP backend only supports an `axis` attribute that refers to the last input dimension.
  // QNN EP is able to support arbitrary axis attributes by wrapping the QNN operator with transposes.
  // However, the exception is Softmax/LogSoftmax with opset < 13. For these older ONNX operators, only
  // axis == input_rank - 1 is supported.
  if (opset_version < 13) {
    const std::string& op_type = node_unit.OpType();

    int32_t axis = GetDefaultAxisAttribute(opset_version);
    Qnn_Scalar_t axis_qnn_scalar = QNN_SCALAR_INIT;
    ORT_RETURN_IF_ERROR(ProcessAxisAttribute(qnn_model_wrapper, node_unit, axis_qnn_scalar, axis));
    std::vector<uint32_t> input_shape;
    ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(node_unit.Inputs()[0].node_arg, input_shape),
                      "QNN EP: Cannot get shape for Softmax input");
    ORT_RETURN_IF(axis != static_cast<int32_t>(input_shape.size() - 1),
                  "QNN ", op_type.c_str(),
                  " only supports an `axis` attribute equal to input_rank-1 (or -1) for ONNX opset < 13");
  }

  return AddToModelBuilder(qnn_model_wrapper, node_unit, logger, true);
}

static std::vector<uint32_t> GetTransposePermToUseLastAxis(uint32_t input_rank, uint32_t axis) {
  assert(axis < input_rank);
  std::vector<uint32_t> transpose_perm;
  transpose_perm.reserve(input_rank);

  for (uint32_t dim = 0; dim < input_rank; dim++) {
    transpose_perm.push_back(dim);
  }

  // Swap axis dim with last dim.
  transpose_perm[axis] = input_rank - 1;
  transpose_perm[input_rank - 1] = axis;

  return transpose_perm;
}

Status SoftmaxOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                       const NodeUnit& node_unit,
                                       const logging::Logger& logger,
                                       std::vector<std::string>& input_names,
                                       bool do_op_validation) const {
  const bool is_npu_backend = IsNpuBackend(qnn_model_wrapper.GetQnnBackendType());
  const auto& inputs = node_unit.Inputs();
  assert(inputs.size() == 1);

  int32_t axis = GetDefaultAxisAttribute(node_unit.SinceVersion());
  Qnn_Scalar_t axis_qnn_scalar = QNN_SCALAR_INIT;
  ORT_RETURN_IF_ERROR(ProcessAxisAttribute(qnn_model_wrapper, node_unit, axis_qnn_scalar, axis));

  TensorInfo input_info = {};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[0], input_info));
  const size_t input_rank = input_info.shape.size();

  // If the axis attribute refers to the last dimension, then process the input as normal.
  if (!is_npu_backend || axis == static_cast<int32_t>(input_rank) - 1) {
    return ProcessInput(qnn_model_wrapper, inputs[0], logger, input_names);
  }

  //
  // The axis does **not** refer to the last input dimension. Must wrap transposes around the operator to be able to use
  // QNN's Softmax operator, which always uses an axis value that refers to the last dimension.
  //

  std::vector<uint32_t> transpose_perm = GetTransposePermToUseLastAxis(static_cast<uint32_t>(input_rank),
                                                                       static_cast<uint32_t>(axis));

  const std::string& input_name = inputs[0].node_arg.Name();
  std::string op_input_name = input_info.is_initializer ? input_name : input_name + "_ort_qnn_ep_transpose";
  input_names.push_back(op_input_name);

  std::vector<uint32_t> op_input_shape = input_info.shape;
  op_input_shape[input_rank - 1] = input_info.shape[axis];
  op_input_shape[axis] = input_info.shape[input_rank - 1];

  ORT_RETURN_IF(input_info.is_initializer, "QNN EP does not support (Log)Softmax with an initializer input, ",
                "which should be optimized away by the ORT optimizer");

  // Input is dynamic, so add transpose node before input.
  const bool is_graph_input = qnn_model_wrapper.IsGraphInput(input_name);

  ORT_RETURN_IF_ERROR(qnn_model_wrapper.AddTransposeNode(node_unit.Index(),
                                                         input_name,
                                                         op_input_name,
                                                         input_info.shape,
                                                         transpose_perm,
                                                         op_input_shape,
                                                         input_info.qnn_data_type,
                                                         input_info.quant_param,
                                                         do_op_validation,
                                                         is_graph_input));

  Qnn_TensorType_t tensor_type = GetInputTensorType(qnn_model_wrapper, op_input_name);
  QnnTensorWrapper input_tensorwrapper(op_input_name, tensor_type, input_info.qnn_data_type, input_info.quant_param,
                                       std::move(op_input_shape), {});
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)), "Failed to add tensor.");

  return Status::OK();
}

Status SoftmaxOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                     const NodeUnit& node_unit,
                                                     std::vector<std::string>&& input_names,
                                                     const logging::Logger& logger,
                                                     bool do_op_validation) const {
  const bool is_npu_backend = IsNpuBackend(qnn_model_wrapper.GetQnnBackendType());
  const std::string& op_type = node_unit.OpType();
  const auto& outputs = node_unit.Outputs();
  assert(outputs.size() == 1);

  int32_t axis = GetDefaultAxisAttribute(node_unit.SinceVersion());
  Qnn_Scalar_t axis_qnn_scalar = QNN_SCALAR_INIT;
  ORT_RETURN_IF_ERROR(ProcessAxisAttribute(qnn_model_wrapper, node_unit, axis_qnn_scalar, axis));

  TensorInfo output_info = {};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(outputs[0], output_info));
  const size_t output_rank = output_info.shape.size();
  const bool axis_is_last_dim = static_cast<size_t>(axis) == output_rank - 1;

  // If axis refers to the last dimension, process outputs as usual.
  if (!is_npu_backend || axis_is_last_dim) {
    QnnParamWrapper axis_param(node_unit.Index(), node_unit.Name(), QNN_OP_SOFTMAX_PARAM_AXIS, axis_qnn_scalar);

    std::vector<std::string> param_tensor_names;
    param_tensor_names.push_back(axis_param.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(axis_param));

    return ProcessOutputs(qnn_model_wrapper, node_unit,
                          std::move(input_names),
                          std::move(param_tensor_names),
                          logger, do_op_validation, GetQnnOpType(op_type));
  }

  //
  // The axis **does** not refer to the last dimension. Must wrap the operator with Transposes to be able to use
  // QNN's Softmax operator, which only supports an axis that refers to the last dimension.
  //

  axis_qnn_scalar.uint32Value = static_cast<uint32_t>(output_rank - 1);  // NOTE: override axis.
  QnnParamWrapper axis_param(node_unit.Index(), node_unit.Name(), QNN_OP_SOFTMAX_PARAM_AXIS, axis_qnn_scalar);

  std::vector<std::string> param_tensor_names;
  param_tensor_names.push_back(axis_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(axis_param));

  const std::string& orig_output_name = outputs[0].node_arg.Name();
  std::string op_output_name = orig_output_name + "_ort_qnn_ep_transpose";

  std::vector<uint32_t> op_output_shape = output_info.shape;
  op_output_shape[output_rank - 1] = output_info.shape[axis];
  op_output_shape[axis] = output_info.shape[output_rank - 1];

  QnnTensorWrapper output_tensorwrapper(op_output_name, QNN_TENSOR_TYPE_NATIVE, output_info.qnn_data_type, output_info.quant_param,
                                        std::vector<uint32_t>(op_output_shape));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensorwrapper)), "Failed to add tensor.");
  ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(GetNodeName(node_unit),
                                                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                    GetQnnOpType(node_unit.OpType()),
                                                    std::move(input_names),
                                                    {op_output_name},
                                                    std::move(param_tensor_names)),
                    "Failed to add node.");

  const bool is_graph_output = qnn_model_wrapper.IsGraphOutput(orig_output_name);
  std::vector<uint32_t> transpose_perm = GetTransposePermToUseLastAxis(static_cast<uint32_t>(output_rank),
                                                                       static_cast<uint32_t>(axis));

  ORT_RETURN_IF_ERROR(qnn_model_wrapper.AddTransposeNode(node_unit.Index(),
                                                         op_output_name,
                                                         orig_output_name,
                                                         op_output_shape,
                                                         transpose_perm,
                                                         output_info.shape,
                                                         output_info.qnn_data_type,
                                                         output_info.quant_param,
                                                         do_op_validation,
                                                         false,
                                                         is_graph_output));

  return Status::OK();
}

void CreateSoftmaxOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<SoftmaxOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
