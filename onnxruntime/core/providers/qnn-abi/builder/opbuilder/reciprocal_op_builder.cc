// // Copyright (c) Microsoft Corporation. All rights reserved.
// // Licensed under the MIT License.

// #include "core/providers/qnn-abi/builder/opbuilder/base_op_builder.h"
// #include "core/providers/qnn-abi/builder/qnn_model_wrapper.h"
// #include "core/providers/qnn-abi/builder/op_builder_factory.h"
// #include "core/providers/qnn-abi/builder/qnn_utils.h"

// namespace onnxruntime {
// namespace qnn {

// class ReciprocalOpBuilder : public BaseOpBuilder {
//  public:
//   ReciprocalOpBuilder() : BaseOpBuilder("ReciprocalOpBuilder") {}
//   ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ReciprocalOpBuilder);

//   Status IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
//                        const NodeUnit& node_unit,
//                        const logging::Logger& logger) const override final ORT_MUST_USE_RESULT;

//  protected:
//   Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
//                                      const NodeUnit& node_unit,
//                                      std::vector<std::string>&& input_names,
//                                      const logging::Logger& logger,
//                                      bool do_op_validation) const override ORT_MUST_USE_RESULT;
// };

// Status ReciprocalOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
//                                           const NodeUnit& node_unit,
//                                           const logging::Logger& logger) const {
//   ORT_UNUSED_PARAMETER(logger);

//   const auto& inputs = node_unit.Inputs();
//   ORT_RETURN_IF_NOT(inputs.size() == 1, "Reciprocal operator must have exactly 1 input.");

//   const auto& outputs = node_unit.Outputs();
//   ORT_RETURN_IF_NOT(outputs.size() == 1, "Reciprocal operator must have exactly 1 output.");

//   // Check input type is float for CPU.
//   ORT_RETURN_IF_ERROR(DataTypeCheckForCpuBackend(qnn_model_wrapper, inputs[0].node_arg.Type()));

//   return Status::OK();
// }

// Status ReciprocalOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
//                                                         const NodeUnit& node_unit,
//                                                         std::vector<std::string>&& input_names,
//                                                         const logging::Logger& logger,
//                                                         bool do_op_validation) const {
//   ORT_UNUSED_PARAMETER(logger);

//   // Create a constant tensor for the divisor (1.0)
//   std::string divisor_name = node_unit.Name() + "_divisor";
//   std::vector<uint32_t> divisor_shape{1};
//   std::vector<uint8_t> divisor_data;

//   TensorInfo input_info{};
//   ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Inputs()[0], input_info));

//   QnnQuantParamsWrapper divisor_quant_param = input_info.quant_param.Copy();
//   Qnn_DataType_t divisor_qnn_data_type = input_info.qnn_data_type;

//   if (input_info.quant_param.IsQuantized()) {
//     // Create a quantized divisor tensor
//     double divisor_value = 1.0;
//     int quantized_divisor_value;
//     ORT_RETURN_IF_ERROR(utils::Quantize(divisor_value, divisor_quant_param.Get().scaleOffsetEncoding.scale,
//                                         divisor_quant_param.Get().scaleOffsetEncoding.offset,
//                                         divisor_qnn_data_type, quantized_divisor_value));
//     size_t element_size = qnn::utils::GetElementSizeByType(divisor_qnn_data_type);
//     divisor_data.resize(element_size);
//     std::memcpy(divisor_data.data(), &quantized_divisor_value, element_size);
//   } else {
//     // Create a float divisor tensor
//     divisor_data.resize(sizeof(float));
//     float one = 1.0f;
//     std::memcpy(divisor_data.data(), &one, sizeof(float));
//   }

//   QnnTensorWrapper divisor_tensorwrapper(divisor_name, QNN_TENSOR_TYPE_STATIC, divisor_qnn_data_type,
//                                          std::move(divisor_quant_param), std::move(divisor_shape), std::move(divisor_data));
//   ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(divisor_tensorwrapper)), "Failed to add divisor tensor.");

//   // Create the Div node
//   const auto& outputs = node_unit.Outputs();
//   const std::string& output_name = outputs[0].node_arg.Name();
//   bool is_graph_output = qnn_model_wrapper.IsGraphOutput(output_name);
//   Qnn_TensorType_t tensor_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
//   TensorInfo output_info{};
//   ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(outputs[0], output_info));
//   QnnTensorWrapper output_tensorwrapper(output_name, tensor_type, output_info.qnn_data_type,
//                                         output_info.quant_param.Copy(), std::move(output_info.shape));
//   ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensorwrapper)), "Failed to add output tensor.");

//   ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
//                         utils::GetNodeName(node_unit),
//                         QNN_OP_PACKAGE_NAME_QTI_AISW,
//                         QNN_OP_ELEMENT_WISE_DIVIDE,
//                         {divisor_name, input_names[0]},
//                         {output_name},
//                         {},
//                         do_op_validation),
//                     "Failed to create Div node.");

//   return Status::OK();
// }

// void CreateReciprocalOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
//   op_registrations.AddOpBuilder(op_type, std::make_unique<ReciprocalOpBuilder>());
// }

// }  // namespace qnn
// }  // namespace onnxruntime
