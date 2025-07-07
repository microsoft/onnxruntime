// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/providers/qnn-abi/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn-abi/builder/qnn_utils.h"
#include "core/providers/qnn-abi/builder/qnn_model_wrapper.h"
#include "core/providers/qnn-abi/builder/op_builder_factory.h"

namespace onnxruntime {
namespace qnn {

class SoftmaxOpBuilder : public BaseOpBuilder {
 public:
  SoftmaxOpBuilder() : BaseOpBuilder("SoftmaxOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(SoftmaxOpBuilder);

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

std::vector<uint32_t> FlattenShapeFromAxis(const std::vector<uint32_t>& input_shape, int32_t axis) {
  /*
  Return the shape with all dimensions multiplied onward from the specified axis. If axis is 0, the returned shape
  will include an additional batch of size 1 as the first dimension.
  */
  assert(axis >= 0 && static_cast<size_t>(axis) < input_shape.size());
  std::vector<uint32_t> output_shape(input_shape.begin(), input_shape.begin() + axis);

  if (axis == 0) {
    output_shape.push_back(1);  // Additional batch included
  }
  output_shape.push_back(
      std::accumulate(input_shape.begin() + axis, input_shape.end(), 1, std::multiplies<uint32_t>()));

  return output_shape;
}

Status SoftmaxOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                       const NodeUnit& node_unit,
                                       const logging::Logger& logger,
                                       std::vector<std::string>& input_names,
                                       bool do_op_validation) const {
  const bool is_npu_backend = IsNpuBackend(qnn_model_wrapper.GetQnnBackendType());
  const auto& inputs = node_unit.Inputs();
  const std::string& input_name = inputs[0].node_arg.Name();
  assert(inputs.size() == 1);

  const int opset_version = node_unit.SinceVersion();
  int32_t axis = GetDefaultAxisAttribute(opset_version);
  Qnn_Scalar_t axis_qnn_scalar = QNN_SCALAR_INIT;
  ORT_RETURN_IF_ERROR(ProcessAxisAttribute(qnn_model_wrapper, node_unit, axis_qnn_scalar, axis));

  TensorInfo input_info = {};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[0], input_info));
  size_t input_rank = input_info.shape.size();
  ORT_RETURN_IF(input_info.is_initializer, "QNN EP does not support (Log)Softmax with an initializer input, ",
                "which should be optimized away by the ORT optimizer");

  if (opset_version < 13) {
    /*
    For Onnx Softmax with opset < 13, its behavior is to flatten the input starting from the axis, and perform
    softmax operation along the axis dimension, then reshape back to the original input shape.
    QNN EP is able to support arbitrary axis attribute by wrapping reshapes around the operator.

    Here provides an example:
    Given an input with shape=(3, 4, 5) and axis=1. Its behavior is to reshape the input to (3, 20), perform softmax,
    and then reshape back to (3, 4, 5).

    When axis equals 0, the reshape output shape includes an additional batch of size 1 as the first dimension.
    Here provides an example:
    Given an input with shape=(3, 4, 5) and axis=0. Its behavior is to reshape the input to (1, 60), perform softmax,
    and then reshape back to (3, 4, 5).
    */
    std::string reshape_output_name = input_name + "_ort_qnn_ep_reshape";
    std::vector<uint32_t> reshape_output_shape = FlattenShapeFromAxis(input_info.shape, axis);

    // Input is dynamic, so add reshape node before input.
    const bool is_graph_input = qnn_model_wrapper.IsGraphInput(input_name);

    ORT_RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(input_name,
                                                         reshape_output_name,
                                                         input_info.shape,
                                                         reshape_output_shape,
                                                         input_info.qnn_data_type,
                                                         input_info.quant_param,
                                                         do_op_validation,
                                                         is_graph_input,
                                                         false));
    input_names.push_back(reshape_output_name);
  } else if (is_npu_backend && axis != static_cast<int32_t>(input_rank) - 1) {
    /*
    For Onnx Softmax with opset >= 13, the QNN HTP backend only supports the axis attribute that refers to the last
    input dimension.
    QNN EP is able to support arbitrary axis attribute by wrapping transposes around the operator.
    */
    std::string transpose_output_name = input_name + "_ort_qnn_ep_transpose";
    std::vector<uint32_t> transpose_perm;
    ORT_RETURN_IF_ERROR(utils::GetPermToLastAxis(static_cast<uint32_t>(axis),
                                                 static_cast<uint32_t>(input_rank),
                                                 transpose_perm));

    std::vector<uint32_t> transpose_output_shape = input_info.shape;
    transpose_output_shape[input_rank - 1] = input_info.shape[axis];
    transpose_output_shape[axis] = input_info.shape[input_rank - 1];

    // Input is dynamic, so add transpose node before input.
    const bool is_graph_input = qnn_model_wrapper.IsGraphInput(input_name);

    ORT_RETURN_IF_ERROR(qnn_model_wrapper.AddTransposeNode(node_unit.Index(),
                                                           input_name,
                                                           transpose_output_name,
                                                           input_info.shape,
                                                           transpose_perm,
                                                           transpose_output_shape,
                                                           input_info.qnn_data_type,
                                                           input_info.quant_param,
                                                           do_op_validation,
                                                           is_graph_input,
                                                           false));
    input_names.push_back(transpose_output_name);
  } else {
    // Process the input as normal.
    return ProcessInput(qnn_model_wrapper, inputs[0], logger, input_names);
  }

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
  const std::string& orig_output_name = outputs[0].node_arg.Name();
  assert(outputs.size() == 1);

  const int opset_version = node_unit.SinceVersion();
  int32_t axis = GetDefaultAxisAttribute(opset_version);
  Qnn_Scalar_t axis_qnn_scalar = QNN_SCALAR_INIT;
  ORT_RETURN_IF_ERROR(ProcessAxisAttribute(qnn_model_wrapper, node_unit, axis_qnn_scalar, axis));

  TensorInfo output_info = {};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(outputs[0], output_info));
  size_t output_rank = output_info.shape.size();

  if (opset_version < 13) {
    std::string reshape_input_name = orig_output_name + "_ort_qnn_ep_reshape";

    std::vector<uint32_t> reshape_input_shape = FlattenShapeFromAxis(output_info.shape, axis);
    if (axis == 0) {
      // Override axis due to the inserted batch=1 to the first dimension
      axis_qnn_scalar.uint32Value = 1;
    }

    QnnParamWrapper axis_param(node_unit.Index(), node_unit.Name(), QNN_OP_SOFTMAX_PARAM_AXIS, axis_qnn_scalar);
    std::vector<std::string> param_tensor_names;
    param_tensor_names.push_back(axis_param.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(axis_param));

    QnnTensorWrapper output_tensorwrapper(reshape_input_name, QNN_TENSOR_TYPE_NATIVE, output_info.qnn_data_type,
                                          output_info.quant_param.Copy(), std::vector<uint32_t>(reshape_input_shape));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensorwrapper)), "Failed to add tensor.");
    ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::GetNodeName(node_unit),
                                                      QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                      GetQnnOpType(node_unit.OpType()),
                                                      std::move(input_names),
                                                      {reshape_input_name},
                                                      std::move(param_tensor_names)),
                      "Failed to add node.");

    const bool is_graph_output = qnn_model_wrapper.IsGraphOutput(orig_output_name);
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(reshape_input_name,
                                                         orig_output_name,
                                                         reshape_input_shape,
                                                         output_info.shape,
                                                         output_info.qnn_data_type,
                                                         output_info.quant_param,
                                                         do_op_validation,
                                                         false,
                                                         is_graph_output));
  } else if (is_npu_backend && axis != static_cast<int32_t>(output_rank) - 1) {
    std::string transpose_input_name = orig_output_name + "_ort_qnn_ep_transpose";

    std::vector<uint32_t> transpose_input_shape = output_info.shape;
    transpose_input_shape[output_rank - 1] = output_info.shape[axis];
    transpose_input_shape[axis] = output_info.shape[output_rank - 1];

    // Override axis due to the actual shape after the inserted transpose node
    axis_qnn_scalar.uint32Value = static_cast<uint32_t>(output_rank) - 1;

    QnnParamWrapper axis_param(node_unit.Index(), node_unit.Name(), QNN_OP_SOFTMAX_PARAM_AXIS, axis_qnn_scalar);
    std::vector<std::string> param_tensor_names;
    param_tensor_names.push_back(axis_param.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(axis_param));

    QnnTensorWrapper output_tensorwrapper(transpose_input_name, QNN_TENSOR_TYPE_NATIVE, output_info.qnn_data_type,
                                          output_info.quant_param.Copy(), std::vector<uint32_t>(transpose_input_shape));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensorwrapper)), "Failed to add tensor.");
    ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::GetNodeName(node_unit),
                                                      QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                      GetQnnOpType(node_unit.OpType()),
                                                      std::move(input_names),
                                                      {transpose_input_name},
                                                      std::move(param_tensor_names)),
                      "Failed to add node.");

    const bool is_graph_output = qnn_model_wrapper.IsGraphOutput(orig_output_name);
    std::vector<uint32_t> transpose_perm;
    ORT_RETURN_IF_ERROR(utils::GetPermToLastAxis(static_cast<uint32_t>(axis),
                                                 static_cast<uint32_t>(output_rank),
                                                 transpose_perm));

    ORT_RETURN_IF_ERROR(qnn_model_wrapper.AddTransposeNode(node_unit.Index(),
                                                           transpose_input_name,
                                                           orig_output_name,
                                                           transpose_input_shape,
                                                           transpose_perm,
                                                           output_info.shape,
                                                           output_info.qnn_data_type,
                                                           output_info.quant_param,
                                                           do_op_validation,
                                                           false,
                                                           is_graph_output));
  } else {
    QnnParamWrapper axis_param(node_unit.Index(), node_unit.Name(), QNN_OP_SOFTMAX_PARAM_AXIS, axis_qnn_scalar);
    std::vector<std::string> param_tensor_names;
    param_tensor_names.push_back(axis_param.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(axis_param));

    return ProcessOutputs(qnn_model_wrapper, node_unit,
                          std::move(input_names),
                          std::move(param_tensor_names),
                          logger, do_op_validation, GetQnnOpType(op_type));
  }

  return Status::OK();
}

void CreateSoftmaxOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<SoftmaxOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
