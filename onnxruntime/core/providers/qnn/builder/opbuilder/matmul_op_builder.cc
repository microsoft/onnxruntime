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

// If MatMul's 2nd input is 2D initializer, we can transpose the 2nd input and use FullyConnected Op instead,
// which is faster and using MatMul Op.
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

Status MatMulOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit,
                                      const logging::Logger& logger, std::vector<std::string>& input_names,
                                      bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(do_op_validation);
  const auto& inputs = node_unit.Inputs();

  // Process input 0.
  const std::string& input_0_name = inputs[0].node_arg.Name();
  input_names.emplace_back(input_0_name);
  if (qnn_model_wrapper.IsQnnTensorWrapperExist(input_0_name)) {
    LOGS(logger, VERBOSE) << "Tensor already added, skip it: " << input_0_name;
  } else {
    QnnTensorWrapper input_0_tensor;
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(inputs[0], input_0_tensor));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_0_tensor)), "Failed to add tensor.");
  }

  // Process input 1.
  const std::string& input_1_name = inputs[1].node_arg.Name();
  input_names.emplace_back(input_1_name);
  bool is_initializer_input_1 = qnn_model_wrapper.IsInitializerInput(input_1_name);
  std::vector<uint32_t> input_1_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[1].node_arg, input_1_shape), "Cannot get shape");
  bool is_quantized_weight = inputs[1].quant_param.has_value();
  // Use FullyConnected if 2nd input is 2D initializer. Need to transpose this 2D initializer.
  // To make it simple, not doing this for quantized weight for now.
  if (!is_quantized_weight && is_initializer_input_1 && input_1_shape.size() == 2) {
    QnnQuantParamsWrapper quantize_param;
    ORT_RETURN_IF_ERROR(quantize_param.Init(qnn_model_wrapper, inputs[1]));
    Qnn_DataType_t qnn_data_type = QNN_DATATYPE_FLOAT_32;
    ORT_RETURN_IF_ERROR(utils::GetQnnDataType(is_quantized_weight, inputs[1].node_arg.TypeAsProto(), qnn_data_type));
    std::vector<uint8_t> unpacked_tensor;
    const auto& input_1_tensor = qnn_model_wrapper.GetInitializerTensors().at(input_1_name);
    ORT_RETURN_IF_ERROR(quantize_param.HandleTranspose<size_t>(std::vector<size_t>({1, 0})));
    ORT_RETURN_IF_ERROR(TwoDimensionTranspose(qnn_model_wrapper, input_1_shape, *input_1_tensor, unpacked_tensor));
    QnnTensorWrapper input_1_tensor_wrapper(input_1_name, QNN_TENSOR_TYPE_STATIC, qnn_data_type,
                                            std::move(quantize_param), std::move(input_1_shape),
                                            std::move(unpacked_tensor));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_1_tensor_wrapper)), "Failed to add tensor.");
  } else {
    if (qnn_model_wrapper.IsQnnTensorWrapperExist(input_1_name)) {
      LOGS(logger, VERBOSE) << "Tensor already added, skip it: " << input_1_name;
    } else {
      QnnTensorWrapper input_1_tensor;
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(inputs[1], input_1_tensor));
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_1_tensor)), "Failed to add tensor.");
    }
  }

  return Status::OK();
}

Status MatMulOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit,
                                                    std::vector<std::string>&& input_names,
                                                    const logging::Logger& logger, bool do_op_validation) const {
  const auto& input_1_node_arg = node_unit.Inputs()[1].node_arg;
  std::vector<uint32_t> input_1_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(input_1_node_arg, input_1_shape), "Cannot get shape");
  bool is_initializer_input_1 = qnn_model_wrapper.IsInitializerInput(input_1_node_arg.Name());
  bool is_quantized_weight = node_unit.Inputs()[1].quant_param.has_value();
  // Use FullyConnected if 2nd input is 2D initializer. Not to do this for quantized weight for now.
  if (!is_quantized_weight && is_initializer_input_1 && input_1_shape.size() == 2) {
    TensorInfo output_info = {};
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Outputs()[0], output_info));
    // Ideally if keep_dims of FullyConnected is true, we don't need to care about the shape. But currently it cannot
    // pass the Op validation (it says the expected value is 0), so we need to reshape the output to expected shape.
    bool need_reshape = output_info.shape.size() != 2;
    const std::string& org_output_name = node_unit.Outputs()[0].node_arg.Name();
    const bool is_graph_output = qnn_model_wrapper.IsGraphOutput(org_output_name);
    std::string op_output_name = need_reshape ? org_output_name + "_ort_qnn_ep_reshape" : org_output_name;
    const std::vector<uint32_t>& org_output_shape = output_info.shape;
    std::vector<uint32_t> op_output_shape =
        need_reshape ? std::vector<uint32_t>({std::accumulate(org_output_shape.begin(), org_output_shape.end() - 1,
                                                              static_cast<uint32_t>(1), std::multiplies<uint32_t>()),
                                              org_output_shape.back()})
                     : org_output_shape;
    Qnn_TensorType_t op_output_tensor_type =
        !need_reshape && is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
    QnnTensorWrapper output_tensor_wrapper(op_output_name, op_output_tensor_type, output_info.qnn_data_type,
                                           output_info.quant_param.Copy(), std::vector<uint32_t>(op_output_shape));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensor_wrapper)),
                      "Failed to add output tensor.");
    ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::GetNodeName(node_unit), QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                      QNN_OP_FULLY_CONNECTED, std::move(input_names), {op_output_name},
                                                      {}, do_op_validation),
                      "Failed to add fused Gemm node.");
    if (need_reshape) {
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(
          op_output_name, org_output_name, op_output_shape, org_output_shape, output_info.qnn_data_type,
          std::move(output_info.quant_param), do_op_validation, false, is_graph_output));
    }
  } else {
    ORT_RETURN_IF_ERROR(ProcessOutputs(qnn_model_wrapper, node_unit, std::move(input_names), {}, logger,
                                       do_op_validation, GetQnnOpType(node_unit.OpType())));
  }

  return Status::OK();
}

void CreateMatMulOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<MatMulOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
