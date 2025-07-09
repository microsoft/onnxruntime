// // Copyright (c) Microsoft Corporation. All rights reserved.
// // Licensed under the MIT License.

// #include "core/providers/qnn-abi/builder/opbuilder/base_op_builder.h"
// #include "core/providers/qnn-abi/builder/qnn_utils.h"
// #include "core/providers/qnn-abi/builder/qnn_model_wrapper.h"
// #include "core/providers/qnn-abi/builder/op_builder_factory.h"

// namespace onnxruntime {
// namespace qnn {

// // ArgMax/ArgMin support limitations:
// //  - HTP only: max input rank is 4.
// //  - All backends: ONNX select_last_index attribute must be 0.
// class ArgMaxMinOpBuilder : public BaseOpBuilder {
//  public:
//   ArgMaxMinOpBuilder() : BaseOpBuilder("ArgMaxMinOpBuilder") {}

//  protected:
//   Qnn_DataType_t GetSupportedOutputDataType(size_t index,
//                                             Qnn_DataType_t qnn_data_type) const override ORT_MUST_USE_RESULT;

//   Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
//                                      const NodeUnit& node_unit,
//                                      std::vector<std::string>&& input_names,
//                                      const logging::Logger& logger,
//                                      bool do_op_validation) const override ORT_MUST_USE_RESULT;
// };

// Qnn_DataType_t ArgMaxMinOpBuilder::GetSupportedOutputDataType(size_t index, Qnn_DataType_t qnn_data_type) const {
//   // ONNX ArgMxx ops have int64 output, but QNN requires uint32 or int32.
//   // If this node produces a graph output, BaseOpBuilder::ProcessOutputs() adds a Cast node after the ArgMxx op.
//   // Otherwise, it just set the output type to unit32 or int32.
//   ORT_UNUSED_PARAMETER(index);
//   if (qnn_data_type == QNN_DATATYPE_INT_64) {
//     return QNN_DATATYPE_INT_32;
//   } else if (qnn_data_type == QNN_DATATYPE_UINT_64) {
//     return QNN_DATATYPE_UINT_32;
//   }

//   return qnn_data_type;
// }

// Status ArgMaxMinOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
//                                                        const NodeUnit& node_unit,
//                                                        std::vector<std::string>&& input_names,
//                                                        const logging::Logger& logger,
//                                                        bool do_op_validation) const {
//   std::vector<std::string> param_tensor_names;
//   int32_t default_axis_value = 0;
//   Qnn_Scalar_t axis_qnn_scalar = QNN_SCALAR_INIT;
//   ORT_RETURN_IF_ERROR(ProcessAxisAttribute(qnn_model_wrapper, node_unit, axis_qnn_scalar, default_axis_value));
//   QnnParamWrapper axis_param(node_unit.Index(), node_unit.Name(), QNN_OP_ARGMAX_PARAM_AXIS, axis_qnn_scalar);
//   param_tensor_names.push_back(axis_param.GetParamTensorName());
//   qnn_model_wrapper.AddParamWrapper(std::move(axis_param));

//   NodeAttrHelper node_helper(node_unit);
//   auto select_last_index = node_helper.Get("select_last_index", static_cast<int32_t>(0));
//   if (select_last_index != 0) {
//     return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN ArgMax/ArgMin only support select_last_index=0.");
//   }
//   auto onnx_keepdims = node_helper.Get("keepdims", static_cast<int32_t>(1));
//   Qnn_Scalar_t keep_dims_scalar = QNN_SCALAR_INIT;
//   keep_dims_scalar.dataType = QNN_DATATYPE_BOOL_8;
//   keep_dims_scalar.bool8Value = static_cast<uint8_t>(onnx_keepdims == 0 ? 0 : 1);
//   QnnParamWrapper keep_dims_param(node_unit.Index(), node_unit.Name(), QNN_OP_ARGMAX_PARAM_KEEP_DIMS, keep_dims_scalar);
//   param_tensor_names.push_back(keep_dims_param.GetParamTensorName());
//   qnn_model_wrapper.AddParamWrapper(std::move(keep_dims_param));

//   ORT_RETURN_IF_ERROR(ProcessOutputs(qnn_model_wrapper, node_unit,
//                                      std::move(input_names),
//                                      std::move(param_tensor_names),
//                                      logger, do_op_validation, GetQnnOpType(node_unit.OpType())));

//   return Status::OK();
// }

// void CreateArgMaxMinOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
//   op_registrations.AddOpBuilder(op_type, std::make_unique<ArgMaxMinOpBuilder>());
// }

// }  // namespace qnn
// }  // namespace onnxruntime
