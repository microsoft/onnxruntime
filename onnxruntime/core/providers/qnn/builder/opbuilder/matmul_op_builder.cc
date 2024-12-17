// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/safeint.h"
#include "core/providers/common.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/shared/utils/utils.h"

namespace onnxruntime {
namespace qnn {

/**
 * ONNX's MatMul supports 1D tensor as input on both size, but neither QNN's MatMul nor FullyConnected supports it.
 * So we need to add Reshape Ops if necessary.
 * In two cases, FullyConnected (input_1's shape is [n, k]) is used instead of MatMul without extra Transpose Op:
 * 1. input_1 is 2D initializer.
 * 2. input_1 is 1D tensor.
 */
class MatMulOpBuilder : public BaseOpBuilder {
 public:
  MatMulOpBuilder() : BaseOpBuilder("MatMulOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(MatMulOpBuilder);

 protected:
  Status ProcessInputs(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit, const logging::Logger& logger,
                       std::vector<std::string>& input_names, bool do_op_validation) const override ORT_MUST_USE_RESULT;

  Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit,
                                     std::vector<std::string>&& input_names, const logging::Logger& logger,
                                     bool do_op_validation) const override ORT_MUST_USE_RESULT;
};

namespace {

Status CheckInputs(const QnnModelWrapper& qnn_model_wrapper, const NodeUnitIODef& input_def_0,
                   const NodeUnitIODef& input_def_1, TensorInfo& input_info_0, TensorInfo& input_info_1,
                   bool& use_fully_connected) {
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(input_def_0, input_info_0));
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(input_def_1, input_info_1));

  // Use FullyConnected if 2nd input is 2D initializer or 1D tensor.
  // FullyConnected cannot pass the Op validation if keep_dims is true, so if input_0 is per-channel quantized tensor
  // with rank > 2, it's not easy to set the quantization parameters for the output reshaped 2D tensor.
  // In this case, we will not use FullyConnected.
  use_fully_connected =
      (input_info_1.shape.size() == 2 && input_info_1.is_initializer) || input_info_1.shape.size() == 1;
  use_fully_connected =
      use_fully_connected && !(input_info_0.quant_param.IsPerChannel() && input_info_0.shape.size() > 2);
  return Status::OK();
}

}  // namespace

Status MatMulOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit,
                                      const logging::Logger& logger, std::vector<std::string>& input_names,
                                      bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();
  TensorInfo input_info_0{};
  TensorInfo input_info_1{};
  bool use_fully_connected = false;
  ORT_RETURN_IF_ERROR(
      CheckInputs(qnn_model_wrapper, inputs[0], inputs[1], input_info_0, input_info_1, use_fully_connected));
  bool reshape_input_0 = input_info_0.shape.size() == 1;
  bool reshape_input_1 = input_info_1.shape.size() == 1;

  // Process input 0.
  const std::string& org_input_0_name = inputs[0].node_arg.Name();
  std::string input_0_name = org_input_0_name;
  if (reshape_input_0) {
    input_0_name = org_input_0_name + "_ort_qnn_ep_reshape";
    std::vector<uint32_t> shape_2d{1, input_info_0.shape[0]};
    QnnQuantParamsWrapper quant_param_2d = input_info_0.quant_param.Copy();
    ORT_RETURN_IF_ERROR(quant_param_2d.HandleUnsqueeze<uint32_t>(input_info_0.shape, shape_2d));

    // If input_0 is initializer, unpack it and add the tensor with new quantization parameter and shape.
    // Otherwise, add a Reshape node.
    if (input_info_0.is_initializer) {
      std::vector<uint8_t> unpacked_tensor;
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*input_info_0.initializer_tensor, unpacked_tensor));
      Qnn_TensorType_t tensor_type = qnn_model_wrapper.GetTensorType(org_input_0_name);
      QnnTensorWrapper input_tensorwrapper(input_0_name, tensor_type, input_info_0.qnn_data_type,
                                           std::move(quant_param_2d), std::move(shape_2d), std::move(unpacked_tensor));
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)), "Failed to add tensor.");
    } else {
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(org_input_0_name, input_0_name, input_info_0.shape, shape_2d,
                                                           input_info_0.qnn_data_type, input_info_0.quant_param,
                                                           quant_param_2d, do_op_validation,
                                                           qnn_model_wrapper.IsGraphInput(org_input_0_name), false));
    }
  } else {
    if (qnn_model_wrapper.IsQnnTensorWrapperExist(input_0_name)) {
      LOGS(logger, VERBOSE) << "Tensor already added, skip it: " << input_0_name;
    } else {
      QnnTensorWrapper input_0_tensor;
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(inputs[0], input_0_tensor));
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_0_tensor)), "Failed to add tensor.");
    }
  }
  input_names.emplace_back(input_0_name);

  // Process input 1.
  const std::string& org_input_1_name = inputs[1].node_arg.Name();
  std::string input_1_name = org_input_1_name;
  if (reshape_input_1 || use_fully_connected) {
    std::vector<uint32_t> shape_2d;
    QnnQuantParamsWrapper quant_param_2d = input_info_1.quant_param.Copy();
    if (reshape_input_1) {
      // Input is 1D tensor.
      input_1_name = org_input_1_name + "_ort_qnn_ep_reshape";
      if (use_fully_connected) {
        // FullyConnected requires input_1's shape to be [n, k].
        shape_2d = {1, input_info_1.shape[0]};
      } else {
        shape_2d = {input_info_1.shape[0], 1};
      }
      ORT_RETURN_IF_ERROR(quant_param_2d.HandleUnsqueeze<uint32_t>(input_info_1.shape, shape_2d));
    } else {
      input_1_name = org_input_1_name + "_ort_qnn_ep_transpose";
      shape_2d = {input_info_1.shape[1], input_info_1.shape[0]};
      ORT_RETURN_IF_ERROR(quant_param_2d.HandleTranspose<uint32_t>(std::vector<uint32_t>({1, 0})));
    }

    // If input_1 is initializer, unpack it and add the tensor with new quantization parameter and shape.
    // Otherwise, add a Reshape node.
    if (input_info_1.is_initializer) {
      std::vector<uint8_t> unpacked_tensor;
      if (use_fully_connected && !reshape_input_1) {
        // 2D initializer should be transposed to [n, k].
        ORT_RETURN_IF_ERROR(TwoDimensionTranspose(qnn_model_wrapper, input_info_1.shape,
                                                  *input_info_1.initializer_tensor, unpacked_tensor));
      } else {
        ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*input_info_1.initializer_tensor, unpacked_tensor));
      }

      Qnn_TensorType_t tensor_type = qnn_model_wrapper.GetTensorType(org_input_1_name);
      QnnTensorWrapper input_tensorwrapper(input_1_name, tensor_type, input_info_1.qnn_data_type,
                                           std::move(quant_param_2d), std::move(shape_2d), std::move(unpacked_tensor));
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)), "Failed to add tensor.");
    } else {
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(org_input_1_name, input_1_name, input_info_1.shape, shape_2d,
                                                           input_info_1.qnn_data_type, input_info_1.quant_param,
                                                           quant_param_2d, do_op_validation,
                                                           qnn_model_wrapper.IsGraphInput(org_input_1_name), false));
    }
  } else {
    if (qnn_model_wrapper.IsQnnTensorWrapperExist(input_1_name)) {
      LOGS(logger, VERBOSE) << "Tensor already added, skip it: " << input_1_name;
    } else {
      QnnTensorWrapper input_1_tensor;
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(inputs[1], input_1_tensor));
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_1_tensor)), "Failed to add tensor.");
    }
  }
  input_names.emplace_back(input_1_name);

  return Status::OK();
}

Status MatMulOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit,
                                                    std::vector<std::string>&& input_names,
                                                    const logging::Logger& /*logger*/, bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();
  TensorInfo input_info_0{};
  TensorInfo input_info_1{};
  bool use_fully_connected = false;
  ORT_RETURN_IF_ERROR(
      CheckInputs(qnn_model_wrapper, inputs[0], inputs[1], input_info_0, input_info_1, use_fully_connected));
  bool reshape_input_0 = input_info_0.shape.size() == 1;
  bool reshape_input_1 = input_info_1.shape.size() == 1;
  bool reshape_output = reshape_input_0 || reshape_input_1 || (use_fully_connected && input_info_0.shape.size() > 2);

  const std::string& org_output_name = node_unit.Outputs()[0].node_arg.Name();
  std::string op_output_name = org_output_name;
  TensorInfo output_info{};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Outputs()[0], output_info));
  std::vector<uint32_t> op_output_shape = output_info.shape;
  QnnQuantParamsWrapper op_output_quant_param = output_info.quant_param.Copy();
  if (reshape_output) {
    op_output_name = org_output_name + "_ort_qnn_ep_reshape";
    if (use_fully_connected && input_info_0.shape.size() > 2) {
      op_output_shape = {std::accumulate(input_info_0.shape.begin(), input_info_0.shape.end() - 1,
                                         static_cast<uint32_t>(1), std::multiplies<uint32_t>()),
                         reshape_input_1 ? 1 : input_info_1.shape.back()};
      ORT_ENFORCE(!op_output_quant_param.IsPerChannel());
    } else {
      // If both inputs are 1D tensors, the output shape is [1] instead of scalar. So if both inputs are 1D tensors,
      // we only need to add one "1" to the op_output_shape.
      if (reshape_input_1) {
        op_output_shape.emplace_back(1);
      } else if (reshape_input_0) {
        op_output_shape.insert(op_output_shape.end() - 1, 1);
      }
      ORT_RETURN_IF_ERROR(op_output_quant_param.HandleUnsqueeze<uint32_t>(output_info.shape, op_output_shape));
    }
  }

  const bool is_graph_output = qnn_model_wrapper.IsGraphOutput(org_output_name);
  const bool is_op_output_graph_output = is_graph_output && !reshape_output;
  Qnn_TensorType_t op_output_tensor_type =
      is_op_output_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
  QnnTensorWrapper op_output_tensor_wrapper(op_output_name, op_output_tensor_type, output_info.qnn_data_type,
                                            op_output_quant_param.Copy(), std::vector<uint32_t>(op_output_shape));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(op_output_tensor_wrapper)),
                    "Failed to add output tensor.");
  ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::GetNodeName(node_unit), QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                    use_fully_connected ? QNN_OP_FULLY_CONNECTED : QNN_OP_MAT_MUL,
                                                    std::move(input_names), {op_output_name}, {}, do_op_validation),
                    "Failed to add fused Matmul node.");

  if (reshape_output) {
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(
        op_output_name, org_output_name, op_output_shape, output_info.shape, output_info.qnn_data_type,
        op_output_quant_param, output_info.quant_param, do_op_validation, false, is_graph_output));
  }

  return Status::OK();
}

void CreateMatMulOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<MatMulOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
