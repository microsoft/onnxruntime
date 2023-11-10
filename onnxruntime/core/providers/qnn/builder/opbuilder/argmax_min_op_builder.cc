// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/framework/tensorprotoutils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace qnn {

// ArgMax/ArgMin support limitations:
//  - HTP only: cannot generate a graph output
//  - HTP only: max input rank is 4.
//  - All backends: ONNX select_last_index attribute must be 0.
class ArgMaxMinOpBuilder : public BaseOpBuilder {
 public:
  ArgMaxMinOpBuilder() : BaseOpBuilder("ArgMaxMinOpBuilder") {}

  Status IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                       const NodeUnit& node_unit,
                       const logging::Logger& logger,
                       bool is_quantized_model) const override ORT_MUST_USE_RESULT;

 protected:
  Qnn_DataType_t GetSupportedOutputDataType(size_t index,
                                            Qnn_DataType_t qnn_data_type) const override ORT_MUST_USE_RESULT;

  Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                     const NodeUnit& node_unit,
                                     std::vector<std::string>&& input_names,
                                     const logging::Logger& logger,
                                     bool is_quantized_model,
                                     bool do_op_validation) const override ORT_MUST_USE_RESULT;
};

Status ArgMaxMinOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                         const NodeUnit& node_unit,
                                         const logging::Logger& logger,
                                         bool is_quantized_model) const {
  // ONNX ArgMax/ArgMin ops output int64 indices, but the equivalent QNN ops output uint32 indices.
  // The QNN HTP backend does not generally support the int64 type, but QNN EP can just use the uint32 type
  // for ArgMax/ArgMin ops within the graph. However, if the ArgMin/ArgMax op **generates** a graph output,
  // then we cannot support it on the HTP backend.
  if (is_quantized_model) {
    const std::string& output_name = node_unit.Outputs()[0].node_arg.Name();
    ORT_RETURN_IF(qnn_model_wrapper.IsGraphOutput(output_name),
                  "QNN EP does not support ArgMin/ArgMax ops that generate a graph output.");
  }

  return AddToModelBuilder(qnn_model_wrapper, node_unit, logger, is_quantized_model, true);
}

Qnn_DataType_t ArgMaxMinOpBuilder::GetSupportedOutputDataType(size_t index, Qnn_DataType_t qnn_data_type) const {
  // ONNX ArgMxx ops have int64 output, but QNN requires uint32.
  // If this node produces a graph output, BaseOpBuilder::ProcessOutputs() adds a Cast node after the ArgMxx op.
  // Otherwise, it just set the output type to unit32. This only works for the QNN CPU backend, since the HTP backend
  // does not generally support int64.
  ORT_UNUSED_PARAMETER(index);
  ORT_UNUSED_PARAMETER(qnn_data_type);
  return QNN_DATATYPE_UINT_32;
}

Status ArgMaxMinOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                       const NodeUnit& node_unit,
                                                       std::vector<std::string>&& input_names,
                                                       const logging::Logger& logger,
                                                       bool is_quantized_model,
                                                       bool do_op_validation) const {
  std::vector<std::string> param_tensor_names;
  int32_t default_axis_value = 0;
  Qnn_Scalar_t axis_qnn_scalar = QNN_SCALAR_INIT;
  ORT_RETURN_IF_ERROR(ProcessAxisAttribute(qnn_model_wrapper, node_unit, axis_qnn_scalar, default_axis_value));
  QnnParamWrapper axis_param(node_unit.Index(), node_unit.Name(), qnn_def::axis, axis_qnn_scalar);
  param_tensor_names.push_back(axis_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(axis_param));

  NodeAttrHelper node_helper(node_unit);
  auto select_last_index = node_helper.Get("select_last_index", static_cast<int32_t>(0));
  if (select_last_index != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN ArgMax/ArgMin only support select_last_index=0.");
  }
  auto onnx_keepdims = node_helper.Get("keepdims", static_cast<int32_t>(1));
  Qnn_Scalar_t keep_dims_scalar = QNN_SCALAR_INIT;
  keep_dims_scalar.dataType = QNN_DATATYPE_BOOL_8;
  keep_dims_scalar.bool8Value = static_cast<uint8_t>(onnx_keepdims == 0 ? 0 : 1);
  QnnParamWrapper keep_dims_param(node_unit.Index(), node_unit.Name(), qnn_def::keep_dims, keep_dims_scalar);
  param_tensor_names.push_back(keep_dims_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(keep_dims_param));

  ORT_RETURN_IF_ERROR(ProcessOutputs(qnn_model_wrapper, node_unit,
                                     std::move(input_names),
                                     std::move(param_tensor_names),
                                     logger, is_quantized_model, do_op_validation, GetQnnOpType(node_unit.OpType())));

  return Status::OK();
}

void CreateArgMaxMinOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<ArgMaxMinOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
