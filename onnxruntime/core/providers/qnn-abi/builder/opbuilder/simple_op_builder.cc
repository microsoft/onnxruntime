// // Copyright (c) Microsoft Corporation. All rights reserved.
// // Licensed under the MIT License.

// #include "core/providers/qnn-abi/builder/opbuilder/base_op_builder.h"
// #include "core/providers/qnn-abi/builder/qnn_model_wrapper.h"
// #include "core/providers/qnn-abi/builder/op_builder_factory.h"
// #include "core/providers/qnn-abi/builder/qnn_utils.h"

// namespace onnxruntime {
// namespace qnn {

// // Operator which only need to hanle node inputs & outputs, no attributes or no need to handle attributes
// class SimpleOpBuilder : public BaseOpBuilder {
//  public:
//   SimpleOpBuilder() : BaseOpBuilder("SimpleOpBuilder") {}
//   ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(SimpleOpBuilder);

// #if !BUILD_QNN_EP_STATIC_LIB
//   // ABI-compatible version of IsOpSupported for shared library builds
//   bool IsOpSupportedForABI(const OrtNode* ort_node,
//                           const OrtApi& ort_api,
//                           const OrtGraph* graph,
//                           const OrtLogger* logger,
//                           const OrtEpGraphSupportInfo* graph_support_info) const override;
// #endif

//  protected:
//   Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
//                                      const NodeUnit& node_unit,
//                                      std::vector<std::string>&& input_names,
//                                      const logging::Logger& logger,
//                                      bool do_op_validation) const override ORT_MUST_USE_RESULT;
//   Status OverrideOutputQuantParam(QnnModelWrapper& qnn_model_wrapper,
//                                   const NodeUnit& node_unit,
//                                   const logging::Logger& logger,
//                                   const std::vector<std::string>& input_names,
//                                   size_t output_index,
//                                   Qnn_DataType_t qnn_data_type,
//                                   QnnQuantParamsWrapper& quant_param) const override ORT_MUST_USE_RESULT;

//  private:
//   Status ExplicitOpCheck(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit) const;
//   Status ProcessSigmoidOrTanhOutput(QnnModelWrapper& qnn_model_wrapper,
//                                     const NodeUnit& node_unit,
//                                     std::vector<std::string>&& input_names,
//                                     std::vector<std::string>&& param_tensor_names,
//                                     const logging::Logger& logger,
//                                     bool do_op_validation) const ORT_MUST_USE_RESULT;

//   static constexpr std::array<std::string_view, 2> gridsample_supported_modes = {"bilinear", "nearest"};
//   static constexpr std::array<std::string_view, 3> gridsample_supported_padding_modes = {"zeros", "border", "reflection"};
//   static constexpr std::array<std::string_view, 3> scatternd_supported_reduction = {"none", "add", "mul"};
// };

// Status SimpleOpBuilder::ExplicitOpCheck(QnnModelWrapper& qnn_model_wrapper,
//                                         const NodeUnit& node_unit) const {
//   const std::string& op_type = node_unit.OpType();

//   if (op_type == "GridSample") {
//     NodeAttrHelper node_helper(node_unit);
//     std::string mode = node_helper.Get("mode", "linear");
//     ORT_RETURN_IF_NOT(utils::ArrayHasString(gridsample_supported_modes, mode), "GridSample does not support mode ",
//                       mode.c_str());
//     std::string padding_mode = node_helper.Get("padding_mode", "zeros");
//     ORT_RETURN_IF_NOT(utils::ArrayHasString(gridsample_supported_padding_modes, padding_mode), "GridSample does not support padding_mode ",
//                       padding_mode.c_str());
//   }

//   // To DO: Remove once QNN CPU supports ScatterND
//   const auto qnn_backend_type = qnn_model_wrapper.GetQnnBackendType();
//   if (op_type == "ScatterND") {
//     ORT_RETURN_IF_NOT(qnn_backend_type == QnnBackendType::HTP,
//                       "QNN EP only supports ScatterND op on HTP backend. Falling back to ORT CPU.");
//   }

//   // ONNX's Min, Max, and Sum operators accept a variable number of inputs (i.e., variadic).
//   // However, QNN's Min, Max, and Add operators must take in exactly two inputs.
//   if (op_type == "Min" || op_type == "Max") {
//     ORT_RETURN_IF_NOT(node_unit.Inputs().size() == 2,
//                       "QNN EP only supports ", op_type.c_str(), " operator with exactly 2 inputs.");
//   }

//   if (op_type == "Sum") {
//     size_t inputs_num = node_unit.Inputs().size();
//     ORT_RETURN_IF_NOT(inputs_num == 2,
//                       "QNN EP supports Sum operator with QNN_OP_ELEMENT_WISE_ADD, which takes exactly 2 inputs. Got ONNX's Sum operator with ",
//                       std::to_string(inputs_num).c_str(), " inputs.");
//   }

//   if (op_type == "DequantizeLinear") {
//     bool is_per_chan_quant = false;
//     int64_t quant_axis = 0;
//     ORT_RETURN_IF_ERROR(qnn_model_wrapper.IsPerChannelQuantized(node_unit.Inputs()[0], is_per_chan_quant, quant_axis));
//     ORT_RETURN_IF(is_per_chan_quant, "QNN EP does not support a standalone DQ op with per-channel quantization");

//     if (qnn_model_wrapper.GetModelSettings().offload_graph_io_quantization) {
//       ORT_RETURN_IF(qnn_model_wrapper.IsGraphOutput(node_unit.Outputs()[0].node_arg.Name()),
//                     "QNN EP is configured to not take DQ nodes that generate a graph output.");
//     }
//   }

//   if (op_type == "QuantizeLinear") {
//     bool is_per_chan_quant = false;
//     int64_t quant_axis = 0;
//     ORT_RETURN_IF_ERROR(qnn_model_wrapper.IsPerChannelQuantized(node_unit.Outputs()[0], is_per_chan_quant, quant_axis));
//     ORT_RETURN_IF(is_per_chan_quant, "QNN EP does not support a standalone Q op with per-channel quantization");

//     if (qnn_model_wrapper.GetModelSettings().offload_graph_io_quantization) {
//       ORT_RETURN_IF(qnn_model_wrapper.IsGraphInput(node_unit.Inputs()[0].node_arg.Name()),
//                     "QNN EP is configured to not take Q nodes that consume a graph input.");
//     }
//   }

//   // QNN ScatterND doesn't support MAX, MIN reduction
//   if (op_type == "ScatterND") {
//     NodeAttrHelper node_helper(node_unit);
//     std::string reduction = node_helper.Get("reduction", "none");
//     ORT_RETURN_IF_NOT(utils::ArrayHasString(scatternd_supported_reduction, reduction), "ScatterND does not support reduction ",
//                       reduction.c_str());
//   }

//   return Status::OK();
// }

// // Limit to float type for now
// Status ProcessNodeAttribute(QnnModelWrapper& qnn_model_wrapper,
//                             const NodeUnit& node_unit,
//                             const std::string& onnx_attr_key,
//                             const std::string& qnn_param_key,
//                             std::vector<std::string>& param_tensor_names,
//                             const float default_value = 1.0f) {
//   NodeAttrHelper node_helper(node_unit);
//   float attr_value = node_helper.Get(onnx_attr_key, default_value);
//   Qnn_Scalar_t attr_qnn_scalar = QNN_SCALAR_INIT;
//   attr_qnn_scalar.dataType = QNN_DATATYPE_FLOAT_32;
//   attr_qnn_scalar.floatValue = attr_value;

//   QnnParamWrapper alpha_param(node_unit.Index(), node_unit.Name(), qnn_param_key, attr_qnn_scalar);
//   param_tensor_names.push_back(alpha_param.GetParamTensorName());
//   qnn_model_wrapper.AddParamWrapper(std::move(alpha_param));

//   return Status::OK();
// }

// Status ProcessBlockSizeAttribute(QnnModelWrapper& qnn_model_wrapper,
//                                  const NodeUnit& node_unit,
//                                  std::vector<std::string>& param_tensor_names) {
//   NodeAttrHelper node_helper(node_unit);
//   uint32_t block_size = node_helper.Get("blocksize", static_cast<uint32_t>(0));
//   std::vector<uint32_t> block_size_shape{2};
//   std::vector<uint32_t> block_size_data(2, block_size);
//   QnnParamWrapper block_size_param(node_unit.Index(), node_unit.Name(), QNN_OP_DEPTH_TO_SPACE_PARAM_BLOCK_SIZE,
//                                    std::move(block_size_shape), std::move(block_size_data));
//   param_tensor_names.push_back(block_size_param.GetParamTensorName());
//   qnn_model_wrapper.AddParamWrapper(std::move(block_size_param));

//   return Status::OK();
// }

// Status ProcessModeAttribute(QnnModelWrapper& qnn_model_wrapper,
//                             const NodeUnit& node_unit,
//                             std::vector<std::string>& param_tensor_names) {
//   NodeAttrHelper node_helper(node_unit);
//   std::string mode = node_helper.Get("mode", "DCR");
//   Qnn_Scalar_t mode_qnn_scalar = QNN_SCALAR_INIT;
//   mode_qnn_scalar.dataType = QNN_DATATYPE_UINT_32;
//   if ("DCR" == mode) {
//     mode_qnn_scalar.uint32Value = QNN_OP_DEPTH_TO_SPACE_MODE_DCR;
//   } else if ("CRD" == mode) {
//     mode_qnn_scalar.uint32Value = QNN_OP_DEPTH_TO_SPACE_MODE_CRD;  // CRD mode
//   } else {
//     return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "DepthToSpace mode only support DCR & CRD.");
//   }

//   QnnParamWrapper mode_param(node_unit.Index(), node_unit.Name(), QNN_OP_DEPTH_TO_SPACE_PARAM_MODE, mode_qnn_scalar);
//   param_tensor_names.push_back(mode_param.GetParamTensorName());
//   qnn_model_wrapper.AddParamWrapper(std::move(mode_param));

//   return Status::OK();
// }

// // Process alpha attribute as input for Qnn LeakyRelu
// Status ProcessAlphaAttributeAsInput(QnnModelWrapper& qnn_model_wrapper,
//                                     const NodeUnit& node_unit,
//                                     const std::string input_name) {
//   NodeAttrHelper node_helper(node_unit);
//   QnnQuantParamsWrapper quantize_param;
//   Qnn_DataType_t qnn_data_type = QNN_DATATYPE_FLOAT_32;
//   union {
//     float alpha;
//     uint16_t alpha_fp16;
//     uint8_t unpack[sizeof(float)];
//   } tensor_data;
//   tensor_data.alpha = node_helper.Get("alpha", 0.01f);
//   std::vector<uint8_t> unpacked_data;
//   // Check LeakyRelu input 0 to see if it's quantized tensor
//   bool is_quantized_tensor = node_unit.Outputs()[0].quant_param.has_value();
//   if (is_quantized_tensor) {
//     qnn_data_type = QNN_DATATYPE_UFIXED_POINT_8;
//     std::array<float, 1> scales = {1.0f};
//     std::array<int32_t, 1> offsets = {0};
//     std::array<uint32_t, 1> shape = {1};
//     auto float_data = gsl::make_span<const float>(&tensor_data.alpha, 1);
//     ORT_RETURN_IF_ERROR(qnn::utils::GetDataQuantParams(float_data, shape, scales, offsets, qnn_data_type));

//     unpacked_data.resize(1);
//     ORT_RETURN_IF_ERROR(qnn::utils::QuantizeData(float_data, shape, scales, offsets, unpacked_data, qnn_data_type));
//     quantize_param = QnnQuantParamsWrapper(scales[0], static_cast<int32_t>(offsets[0]));
//   } else {
//     const auto& inputs = node_unit.Inputs();
//     TensorInfo input_info = {};
//     ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[0], input_info));
//     // QNN requires alpha is fp16 when input is fp16
//     if (input_info.qnn_data_type == QNN_DATATYPE_FLOAT_16) {
//       tensor_data.alpha_fp16 = MLFloat16(tensor_data.alpha).val;
//       qnn_data_type = QNN_DATATYPE_FLOAT_16;
//       unpacked_data.assign(tensor_data.unpack, tensor_data.unpack + sizeof(MLFloat16));
//     } else {
//       unpacked_data.assign(tensor_data.unpack, tensor_data.unpack + sizeof(float));
//     }
//   }
//   std::vector<uint32_t> input_shape{1};
//   Qnn_TensorType_t tensor_type = QNN_TENSOR_TYPE_STATIC;
//   QnnTensorWrapper input_tensorwrapper(input_name, tensor_type, qnn_data_type, std::move(quantize_param),
//                                        std::move(input_shape), std::move(unpacked_data));
//   ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)), "Failed to add tensor.");
//   return Status::OK();
// }

// Status ProcessGridSampleAttributes(QnnModelWrapper& qnn_model_wrapper,
//                                    const NodeUnit& node_unit,
//                                    std::vector<std::string>& param_tensor_names) {
//   NodeAttrHelper node_helper(node_unit);
//   int64_t align_corners = node_helper.Get("align_corners", static_cast<int64_t>(0));
//   Qnn_Scalar_t align_corners_qnn_scalar = QNN_SCALAR_INIT;
//   align_corners_qnn_scalar.dataType = QNN_DATATYPE_BOOL_8;
//   align_corners_qnn_scalar.bool8Value = static_cast<uint8_t>(align_corners == 0 ? 0 : 1);
//   QnnParamWrapper align_corners_param(node_unit.Index(), node_unit.Name(), QNN_OP_GRID_SAMPLE_PARAM_ALIGN_CORNERS, align_corners_qnn_scalar);
//   param_tensor_names.push_back(align_corners_param.GetParamTensorName());
//   qnn_model_wrapper.AddParamWrapper(std::move(align_corners_param));

//   std::string mode = node_helper.Get("mode", "linear");
//   Qnn_Scalar_t mode_qnn_scalar = QNN_SCALAR_INIT;
//   mode_qnn_scalar.dataType = QNN_DATATYPE_UINT_32;
//   if ("bilinear" == mode) {
//     mode_qnn_scalar.uint32Value = QNN_OP_GRID_SAMPLE_MODE_BILINEAR;
//   } else if ("nearest" == mode) {
//     mode_qnn_scalar.uint32Value = QNN_OP_GRID_SAMPLE_MODE_NEAREST;
//   } else {
//     return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "GridSample mode only support bilinear & nearest.");
//   }
//   QnnParamWrapper mode_param(node_unit.Index(), node_unit.Name(), QNN_OP_GRID_SAMPLE_PARAM_MODE, mode_qnn_scalar);
//   param_tensor_names.push_back(mode_param.GetParamTensorName());
//   qnn_model_wrapper.AddParamWrapper(std::move(mode_param));

//   std::string padding_mode = node_helper.Get("padding_mode", "zeros");
//   Qnn_Scalar_t padding_mode_qnn_scalar = QNN_SCALAR_INIT;
//   padding_mode_qnn_scalar.dataType = QNN_DATATYPE_UINT_32;
//   if ("zeros" == padding_mode) {
//     padding_mode_qnn_scalar.uint32Value = QNN_OP_GRID_SAMPLE_PADDING_MODE_ZEROS;
//   } else if ("border" == padding_mode) {
//     padding_mode_qnn_scalar.uint32Value = QNN_OP_GRID_SAMPLE_PADDING_MODE_BORDER;
//   } else if ("reflection" == padding_mode) {
//     padding_mode_qnn_scalar.uint32Value = QNN_OP_GRID_SAMPLE_PADDING_MODE_REFLECTION;
//   } else {
//     return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "GridSample padding_mode only support zeros, border & reflection.");
//   }
//   QnnParamWrapper padding_mode_param(node_unit.Index(), node_unit.Name(), QNN_OP_GRID_SAMPLE_PARAM_PADDING_MODE, padding_mode_qnn_scalar);
//   param_tensor_names.push_back(padding_mode_param.GetParamTensorName());
//   qnn_model_wrapper.AddParamWrapper(std::move(padding_mode_param));

//   return Status::OK();
// }

// // Process Reduction attribute of ScatterND op
// Status ProcessScatterNDReductionAttribute(QnnModelWrapper& qnn_model_wrapper,
//                                           const NodeUnit& node_unit,
//                                           std::vector<std::string>& param_tensor_names) {
//   NodeAttrHelper node_helper(node_unit);
//   std::string reduction = node_helper.Get("reduction", "none");
//   Qnn_Scalar_t reduction_qnn_scalar = QNN_SCALAR_INIT;
//   reduction_qnn_scalar.dataType = QNN_DATATYPE_UINT_32;
//   if ("none" == reduction) {
//     reduction_qnn_scalar.uint32Value = QNN_OP_SCATTER_ND_REDUCTION_NONE;
//   } else if ("add" == reduction) {
//     reduction_qnn_scalar.uint32Value = QNN_OP_SCATTER_ND_REDUCTION_ADD;
//   } else if ("mul" == reduction) {
//     reduction_qnn_scalar.uint32Value = QNN_OP_SCATTER_ND_REDUCTION_MUL;
//   } else {
//     return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "ScatterND support only reduction:{none, add, mul}.");
//   }
//   QnnParamWrapper reduction_param(node_unit.Index(), node_unit.Name(), QNN_OP_SCATTER_ND_PARAM_REDUCTION,
//                                   reduction_qnn_scalar);
//   param_tensor_names.push_back(reduction_param.GetParamTensorName());
//   qnn_model_wrapper.AddParamWrapper(std::move(reduction_param));

//   return Status::OK();
// }

// Status SimpleOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
//                                                     const NodeUnit& node_unit,
//                                                     std::vector<std::string>&& input_names,
//                                                     const logging::Logger& logger,
//                                                     bool do_op_validation) const {
//   if (input_names.size() < 1) {
//     return Status::OK();
//   }

//   const std::string& op_type = node_unit.OpType();

//   if (do_op_validation) {
//     ORT_RETURN_IF_ERROR(ExplicitOpCheck(qnn_model_wrapper, node_unit));
//     // Skip the op validation for DepthToSpace & SpaceToDepth if it's not NHWC data layout
//     if (node_unit.Domain() != kMSInternalNHWCDomain && (op_type == "DepthToSpace" || op_type == "SpaceToDepth" || op_type == "GridSample")) {
//       return Status::OK();
//     }

// #if QNN_API_VERSION_MAJOR >= 2 && QNN_API_VERSION_MINOR >= 21 && QNN_API_VERSION_MINOR <= 23
//     // Skip QNN validation for Tanh with uint16 (quantized) output.
//     // This gets around a Tanh QNN validation bug in QNN SDK 2.28.0 - 2.30.0.
//     // The QNN documentation states that the output scale and offset for ufixed_point_16 should be
//     // (1/32768) and -32768, respectively. However, the QNN validator incorrectly rejects these values.
//     if (op_type == "Tanh") {
//       TensorInfo output_info = {};
//       ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Outputs()[0], output_info));
//       if (output_info.qnn_data_type == QNN_DATATYPE_UFIXED_POINT_16) {
//         LOGS(logger, INFO) << "Skipping QNN validation for Tanh node '"
//                            << node_unit.Name() << "' with quantized unit16 output.";
//         return Status::OK();
//       }
//     }
// #endif
//   }

//   std::vector<std::string> param_tensor_names;
//   // Add attribute
//   if (op_type == "Concat") {
//     int32_t default_axis = 0;
//     Qnn_Scalar_t axis_qnn_scalar = QNN_SCALAR_INIT;
//     ORT_RETURN_IF_ERROR(ProcessAxisAttribute(qnn_model_wrapper, node_unit, axis_qnn_scalar, default_axis));
//     QnnParamWrapper axis_param(node_unit.Index(), node_unit.Name(), QNN_OP_SOFTMAX_PARAM_AXIS, axis_qnn_scalar);
//     param_tensor_names.push_back(axis_param.GetParamTensorName());
//     qnn_model_wrapper.AddParamWrapper(std::move(axis_param));
//   }

//   if (op_type == "LpNormalization") {
//     int32_t default_axis = -1;
//     Qnn_Scalar_t axis_qnn_scalar = QNN_SCALAR_INIT;
//     ORT_RETURN_IF_ERROR(ProcessAxisAttribute(qnn_model_wrapper, node_unit, axis_qnn_scalar, default_axis));
//     QnnParamWrapper axis_param(node_unit.Index(), node_unit.Name(), QNN_OP_L2_NORM_PARAM_AXIS, axis_qnn_scalar);
//     param_tensor_names.push_back(axis_param.GetParamTensorName());
//     qnn_model_wrapper.AddParamWrapper(std::move(axis_param));

//     NodeAttrHelper node_helper(node_unit);
//     int64_t norm_p_order = node_helper.Get("p", static_cast<int64_t>(2));
//     ORT_RETURN_IF(norm_p_order != 2, "QNN EP only supports LpNormalization with 'p' attribute equal to 2.");
//   }

//   if (op_type == "LeakyRelu") {
//     std::string input_name = "alpha";
//     ORT_RETURN_IF_ERROR(ProcessAlphaAttributeAsInput(qnn_model_wrapper, node_unit, input_name));
//     input_names.push_back(input_name);
//   }

//   if (op_type == "Elu") {
//     ORT_RETURN_IF_ERROR(ProcessNodeAttribute(qnn_model_wrapper, node_unit, "alpha",
//                                              QNN_OP_ELU_PARAM_ALPHA, param_tensor_names));
//   }

//   if (op_type == "HardSigmoid") {
//     int32_t onnx_data_type = 0;
//     ORT_RETURN_IF_ERROR(utils::GetOnnxTensorElemDataType(node_unit.Inputs()[0].node_arg, onnx_data_type));

//     ORT_RETURN_IF_ERROR(ProcessNodeAttribute(qnn_model_wrapper, node_unit, "alpha",
//                                              QNN_OP_ELEMENT_WISE_NEURON_PARAM_ALPHA,
//                                              param_tensor_names, 0.2f));
//     ORT_RETURN_IF_ERROR(ProcessNodeAttribute(qnn_model_wrapper, node_unit, "beta",
//                                              QNN_OP_ELEMENT_WISE_NEURON_PARAM_BETA,
//                                              param_tensor_names, 0.5f));
//     Qnn_Scalar_t neuron_operation = QNN_SCALAR_INIT;
//     neuron_operation.dataType = QNN_DATATYPE_UINT_32;
//     neuron_operation.uint32Value = QNN_OP_ELEMENT_WISE_NEURON_OPERATION_HARD_SIGMOID;

//     QnnParamWrapper operation_param(node_unit.Index(), node_unit.Name(),
//                                     QNN_OP_ELEMENT_WISE_NEURON_PARAM_OPERATION,
//                                     neuron_operation);
//     param_tensor_names.push_back(operation_param.GetParamTensorName());
//     qnn_model_wrapper.AddParamWrapper(std::move(operation_param));
//   }

//   if (op_type == "DepthToSpace") {
//     ORT_RETURN_IF_ERROR(ProcessBlockSizeAttribute(qnn_model_wrapper, node_unit, param_tensor_names));
//     ORT_RETURN_IF_ERROR(ProcessModeAttribute(qnn_model_wrapper, node_unit, param_tensor_names));
//   }

//   if (op_type == "SpaceToDepth") {
//     ORT_RETURN_IF_ERROR(ProcessBlockSizeAttribute(qnn_model_wrapper, node_unit, param_tensor_names));
//   }

//   if (op_type == "GridSample") {
//     ORT_RETURN_IF_ERROR(ProcessGridSampleAttributes(qnn_model_wrapper, node_unit, param_tensor_names));
//   }

//   if (op_type == "ScatterND") {
//     // Process reduction attribute
//     ORT_RETURN_IF_ERROR(ProcessScatterNDReductionAttribute(qnn_model_wrapper, node_unit, param_tensor_names));
//   }

//   return ProcessOutputs(qnn_model_wrapper, node_unit,
//                         std::move(input_names),
//                         std::move(param_tensor_names),
//                         logger, do_op_validation, GetQnnOpType(op_type));
// }

// /**
//  * Overrides offset and scale quantization parameters for operators (e.g., Sigmoid or Tanh) that require
//  * specific values. Returns true if the quantization parameters were overridden.
//  *
//  * \param op_type The ONNX operator type.
//  * \param qnn_data_type The QNN tensor data type.
//  * \param quant_params Output scale/offset parameter that may be overridden.
//  * \return True if the offset and scale were overridden.
//  */
// static bool OverrideQuantParams(const std::string& op_type, Qnn_DataType_t qnn_data_type,
//                                 Qnn_ScaleOffset_t& quant_params) {
//   const int32_t orig_offset = quant_params.offset;
//   const float orig_scale = quant_params.scale;

//   if (op_type == "Sigmoid") {
//     switch (qnn_data_type) {
//       case QNN_DATATYPE_UFIXED_POINT_16:
//         quant_params.offset = 0;
//         quant_params.scale = 1.0f / 65536.0f;
//         break;
//       case QNN_DATATYPE_SFIXED_POINT_16:
//         quant_params.offset = 0;
//         quant_params.scale = 1.0f / 32768.0f;
//         break;
//       default:
//         break;  // Do nothing.
//     }
//   }

//   if (op_type == "Tanh") {
//     switch (qnn_data_type) {
//       case QNN_DATATYPE_UFIXED_POINT_16:
//         quant_params.offset = -32768;
//         quant_params.scale = 1.0f / 32768.0f;
//         break;
//       case QNN_DATATYPE_SFIXED_POINT_16:
//         quant_params.offset = 0;
//         quant_params.scale = 1.0f / 32768.0f;
//         break;
//       default:
//         break;  // Do nothing.
//     }
//   }

//   return quant_params.offset != orig_offset || quant_params.scale != orig_scale;
// }

// Status SimpleOpBuilder::OverrideOutputQuantParam(QnnModelWrapper& qnn_model_wrapper,
//                                                  const NodeUnit& node_unit,
//                                                  const logging::Logger& logger,
//                                                  const std::vector<std::string>& input_names,
//                                                  size_t output_index,
//                                                  Qnn_DataType_t qnn_data_type,
//                                                  QnnQuantParamsWrapper& quant_param) const {
//   ORT_UNUSED_PARAMETER(input_names);
//   const std::string& op_type = node_unit.OpType();

//   // Override output quantization parameters for uint16 QDQ Sigmoid or Tanh.
//   // QNN requires 16-bit QDQ Sigmoid and Tanh to use specific output scale and zero-point values
//   // regardless of floating-point range.
//   if (op_type == "Sigmoid" || op_type == "Tanh") {
//     const auto& outputs = node_unit.Outputs();
//     ORT_RETURN_IF_NOT(output_index < outputs.size(),
//                       "Invalid output index in OverrideOutputQuantParam for op ", op_type.c_str());

//     const auto& output = node_unit.Outputs()[0];
//     const std::string& output_name = output.node_arg.Name();

//     if (quant_param.IsPerTensor(/*include_bw*/ false)) {
//       if (OverrideQuantParams(op_type, qnn_data_type, quant_param.Get().scaleOffsetEncoding)) {
//         const int32_t offset = quant_param.Get().scaleOffsetEncoding.offset;
//         const float scale = quant_param.Get().scaleOffsetEncoding.scale;

//         LOGS(logger, VERBOSE) << "QNN requires that 16-bit quantized " << op_type
//                               << " operators use offset/scale values "
//                               << "of <" << offset << ", " << scale
//                               << ">. QNN EP will override the original values for output " << output_name;
//         ORT_RETURN_IF(qnn_model_wrapper.IsQnnTensorWrapperExist(output_name),
//                       "QNN EP is unable to override output quantization parameters for ", op_type.c_str(),
//                       " operator. Node name: ", node_unit.Name().c_str(), ", output name: ", output_name.c_str());
//       }
//     }
//   }

//   return Status::OK();
// }

// #if !BUILD_QNN_EP_STATIC_LIB
// // ABI-compatible version of IsOpSupported for Simple operations
// bool SimpleOpBuilder::IsOpSupportedForABI(const OrtNode* ort_node,
//                                           const OrtApi& ort_api,
//                                           const OrtGraph* graph,
//                                           const OrtLogger* logger,
//                                           const OrtEpGraphSupportInfo* graph_support_info) const {
//   // Use the base class implementation which calls the exact IsOpSupported function
//   // This ensures we get the same validation logic as the regular IsOpSupported call
//   std::cout << "DEBUG: SimpleOpBuilder::IsOpSupportedForABI called." << std::endl;
//   return BaseOpBuilder::IsOpSupportedForABI(ort_node, ort_api, graph, logger, graph_support_info);
// }
// #endif

// void CreateSimpleOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
//   op_registrations.AddOpBuilder(op_type, std::make_unique<SimpleOpBuilder>());
// }

// }  // namespace qnn
// }  // namespace onnxruntime
