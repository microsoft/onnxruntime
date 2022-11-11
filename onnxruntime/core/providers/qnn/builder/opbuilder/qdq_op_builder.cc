// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace qnn {

class QdqOpBuilder : public BaseOpBuilder {
 public:
  QdqOpBuilder() : BaseOpBuilder("QdqOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(QdqOpBuilder);

 protected:
  Status ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                       const NodeUnit& node_unit,
                       const logging::Logger& logger,
                       bool is_quantized_model,
                       std::vector<std::string>& input_names,
                       bool do_op_validation) const override ORT_MUST_USE_RESULT;

 private:
  Status AddQuantizeNodeOnModelInput(QnnModelWrapper& qnn_model_wrapper,
                                     const NodeUnit& node_unit,
                                     const logging::Logger& logger) const;
  Status AddDequantizeNodeOnModelOutput(QnnModelWrapper& qnn_model_wrapper,
                                        const NodeUnit& node_unit,
                                        const logging::Logger& logger) const;
};

Status QdqOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                   const NodeUnit& node_unit,
                                   const logging::Logger& logger,
                                   bool is_quantized_model,
                                   std::vector<std::string>& input_names,
                                   bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(input_names);
  ORT_UNUSED_PARAMETER(is_quantized_model);
  ORT_UNUSED_PARAMETER(do_op_validation);
  if (node_unit.OpType() == "QuantizeLinear") {
    return AddQuantizeNodeOnModelInput(qnn_model_wrapper, node_unit, logger);
  } else if (node_unit.OpType() == "DequantizeLinear") {
    return AddDequantizeNodeOnModelOutput(qnn_model_wrapper, node_unit, logger);
  }

  return Status::OK();
}

Status QdqOpBuilder::AddQuantizeNodeOnModelInput(QnnModelWrapper& qnn_model_wrapper,
                                                 const NodeUnit& node_unit,
                                                 const logging::Logger& logger) const {
  auto& input_name = node_unit.Inputs()[0].node_arg.Name();
  auto& output_name = node_unit.Outputs()[0].node_arg.Name();
  LOGS(logger, VERBOSE) << "AddQuantizeNodeOnModelInput: " << input_name << "->" << output_name;

  const auto* type_proto = node_unit.GetNode().InputDefs()[2]->TypeAsProto();
  int32_t onnx_data_type;
  Qnn_DataType_t qnn_data_type = QNN_DATATYPE_FLOAT_32;
  ORT_RETURN_IF_ERROR(GetQnnDataType(true, type_proto, onnx_data_type, qnn_data_type));

  std::vector<uint32_t> input_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(node_unit.Inputs()[0].node_arg, input_shape), "Cannot get shape");
  std::vector<uint32_t> output_shape = input_shape;

  float scale_value = 0.0f;
  int32_t offset_value = 0;
  const auto& scale_name = node_unit.GetNode().InputDefs()[1]->Name();
  ORT_RETURN_IF_NOT(qnn_model_wrapper.ProcessScale(scale_name, scale_value), "ProcessScale failed");
  const auto& zero_point_name = node_unit.GetNode().InputDefs()[2]->Name();
  ORT_RETURN_IF_NOT(qnn_model_wrapper.ProcessOffset(zero_point_name, offset_value), "ProcessOffset failed");

  Qnn_TensorType_t input_tensor_type = QNN_TENSOR_TYPE_APP_WRITE;
  Qnn_TensorDataFormat_t data_format = 0;
  Qnn_QuantizeParams_t quantize_params = QNN_QUANTIZE_PARAMS_INIT;
  QnnTensorWrapper input_tensorwrapper(input_name, input_tensor_type, data_format, QNN_DATATYPE_FLOAT_32, quantize_params,
                                       std::move(input_shape));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensor(input_name, std::move(input_tensorwrapper)), "Failed to add tensor.");
  std::vector<std::string> input_names{input_name};
  Qnn_TensorType_t output_tensor_type = QNN_TENSOR_TYPE_NATIVE;
  quantize_params.encodingDefinition = QNN_DEFINITION_DEFINED;
  quantize_params.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
  quantize_params.scaleOffsetEncoding.scale = scale_value;
  quantize_params.scaleOffsetEncoding.offset = offset_value;
  QnnTensorWrapper output_tensorwrapper(output_name, output_tensor_type, data_format, qnn_data_type, quantize_params,
                                        std::move(output_shape));

  std::vector<QnnTensorWrapper> output_tensors;
  output_tensors.emplace_back(std::move(output_tensorwrapper));
  const static std::string qnn_op_type = "Quantize";
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddNode(output_name,               // Node Name
                                               qnn_def::package_name,     // Package Name
                                               qnn_op_type,               // Qnn Node Type
                                               {},                        // Node Params
                                               input_names,               // Input Tensor Names
                                               std::move(output_tensors)  // Output Tensors
                                               ),
                    "Failed to add node.");

  return Status::OK();
}

Status QdqOpBuilder::AddDequantizeNodeOnModelOutput(QnnModelWrapper& qnn_model_wrapper,
                                                    const NodeUnit& node_unit,
                                                    const logging::Logger& logger) const {
  auto& input_name = node_unit.Inputs()[0].node_arg.Name();
  auto& output_name = node_unit.Outputs()[0].node_arg.Name();
  LOGS(logger, VERBOSE) << "AddDequantizeNodeOnModelOutput: " << input_name << "->" << output_name;

  // get type from zero_point
  const auto* type_proto = node_unit.GetNode().InputDefs()[2]->TypeAsProto();
  int32_t onnx_data_type;
  Qnn_DataType_t qnn_data_type = QNN_DATATYPE_FLOAT_32;
  ORT_RETURN_IF_ERROR(GetQnnDataType(true, type_proto, onnx_data_type, qnn_data_type));

  std::vector<uint32_t> output_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(node_unit.Outputs()[0].node_arg, output_shape),
                    "Cannot get shape");
  float scale_value = 0.0f;
  int32_t offset_value = 0;
  const auto& scale_name = node_unit.GetNode().InputDefs()[1]->Name();
  ORT_RETURN_IF_NOT(qnn_model_wrapper.ProcessScale(scale_name, scale_value), "ProcessScale failed");
  const auto& zero_point_name = node_unit.GetNode().InputDefs()[2]->Name();
  ORT_RETURN_IF_NOT(qnn_model_wrapper.ProcessOffset(zero_point_name, offset_value), "ProcessOffset failed");
  Qnn_TensorType_t input_tensor_type = QNN_TENSOR_TYPE_NATIVE;
  Qnn_QuantizeParams_t quantize_params = QNN_QUANTIZE_PARAMS_INIT;
  quantize_params.encodingDefinition = QNN_DEFINITION_DEFINED;
  quantize_params.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
  quantize_params.scaleOffsetEncoding.scale = scale_value;
  quantize_params.scaleOffsetEncoding.offset = offset_value;

  std::vector<uint32_t> input_shape = output_shape;
  QnnTensorWrapper input_tensorwrapper(input_name, input_tensor_type, 0, qnn_data_type, quantize_params,
                                       std::move(input_shape));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensor(input_name, std::move(input_tensorwrapper)), "Failed to add tensor.");
  bool is_graph_output = qnn_model_wrapper.IsGraphOutput(output_name);
  Qnn_TensorType_t output_tensor_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
  QnnTensorWrapper output_tensorwrapper(output_name, output_tensor_type, 0, QNN_DATATYPE_FLOAT_32,
                                        QNN_QUANTIZE_PARAMS_INIT, std::move(output_shape));
  const static std::string qnn_op_type = "Dequantize";

  std::vector<QnnTensorWrapper> output_tensors;
  output_tensors.emplace_back(std::move(output_tensorwrapper));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddNode(output_name,                 // Node Name
                                               qnn_def::package_name,       // Package Name
                                               qnn_op_type,                 // Qnn Node Type
                                               {},                          // Node Params
                                               {input_name},                // Input Tensor Names
                                               std::move(output_tensors)),  // Output Tensors
                    "Failed to add node.");

  return Status::OK();
}

void CreateQdqOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<QdqOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
