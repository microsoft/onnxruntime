// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/builder/op_builder_factory.h"

namespace onnxruntime {
namespace qnn {
namespace {

Status GetOnnxAxis(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit, uint32_t& onnx_axis) {
  const auto& inputs = node_unit.Inputs();
  TensorInfo axis_input_info = {};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[1], axis_input_info));
  ORT_RETURN_IF_NOT(axis_input_info.is_initializer, "axis must be initializers");
  std::vector<uint8_t> axis_unpacked_tensor;
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*axis_input_info.initializer_tensor, axis_unpacked_tensor));
  ORT_RETURN_IF_NOT(1 == static_cast<uint32_t>(axis_unpacked_tensor.size() / sizeof(axis_input_info.qnn_data_type)),
                    "axis should be a single element");

  int32_t axis = 0;
  if (axis_input_info.qnn_data_type == QNN_DATATYPE_INT_64) {
    axis = static_cast<int32_t>(*reinterpret_cast<const int64_t*>(axis_unpacked_tensor.data()));
  } else {
    axis = static_cast<int32_t>(*reinterpret_cast<const int32_t*>(axis_unpacked_tensor.data()));
  }

  std::vector<uint32_t> input_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[0].node_arg, input_shape), "Cannot get shape");

  auto rank = static_cast<int32_t>(input_shape.size());
  if (axis < 0) {
    axis += rank;
  }

  ORT_RETURN_IF_NOT((axis >= 0 && axis < static_cast<int32_t>(input_shape.size())), "QNN requires axis range [0, rank-1].");

  onnx_axis = static_cast<uint32_t>(axis);

  return Status::OK();
}

}  // namespace

class CumSumOpBuilder : public BaseOpBuilder {
 public:
  CumSumOpBuilder() : BaseOpBuilder("CumSumOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(CumSumOpBuilder);

  Status IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                       const NodeUnit& node_unit,
                       const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

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

Status CumSumOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                      const NodeUnit& node_unit,
                                      const logging::Logger& logger) const {
  const auto& inputs = node_unit.Inputs();
  ORT_RETURN_IF_NOT(qnn_model_wrapper.IsConstantInput(inputs[1].node_arg.Name()),
                    "QNN CumSum needs axis as a param, hence input[1] must be a constant.");

  NodeAttrHelper node_helper(node_unit);
  int64_t exclusive = node_helper.Get("exclusive", static_cast<int64_t>(0));
  int64_t reverse = node_helper.Get("reverse", static_cast<int64_t>(0));

  // QNN HTP op validation passes for non-default values of attributes but fails in finalize.
  // Hence adding the checks here.
  ORT_RETURN_IF_NOT(exclusive == 0, "QNN only supports default value 0 for exclusive attribute");
  ORT_RETURN_IF_NOT(reverse == 0, "QNN only supports default value 0 for reverse attribute");

  return AddToModelBuilder(qnn_model_wrapper, node_unit, logger, true);
}

Status CumSumOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                      const NodeUnit& node_unit,
                                      const logging::Logger& logger,
                                      std::vector<std::string>& input_names,
                                      bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(do_op_validation);
  const auto& inputs = node_unit.Inputs();
  ORT_RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[0], logger, input_names));
  return Status::OK();
}

Status CumSumOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                    const NodeUnit& node_unit,
                                                    std::vector<std::string>&& input_names,
                                                    const logging::Logger& logger,
                                                    bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(do_op_validation);

  std::vector<std::string> param_tensor_names;

  // Add axis param
  Qnn_Scalar_t axis_qnn_scalar = QNN_SCALAR_INIT;
  uint32_t onnx_axis = 0;
  ORT_RETURN_IF_ERROR(GetOnnxAxis(qnn_model_wrapper, node_unit, onnx_axis));
  axis_qnn_scalar.dataType = QNN_DATATYPE_UINT_32;
  axis_qnn_scalar.uint32Value = onnx_axis;
  QnnParamWrapper axis_param(node_unit.Index(), node_unit.Name(), QNN_OP_CUMULATIVE_SUM_PARAM_AXIS, axis_qnn_scalar);
  param_tensor_names.push_back(axis_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(axis_param));

  // Add exclusive param
  NodeAttrHelper node_helper(node_unit);
  int64_t exclusive = node_helper.Get("exclusive", static_cast<int64_t>(0));
  Qnn_Scalar_t exclusive_qnn_scalar = QNN_SCALAR_INIT;
  exclusive_qnn_scalar.dataType = QNN_DATATYPE_BOOL_8;
  exclusive_qnn_scalar.bool8Value = static_cast<uint8_t>(exclusive == 0 ? 0 : 1);
  QnnParamWrapper exclusive_param(node_unit.Index(), node_unit.Name(), QNN_OP_CUMULATIVE_SUM_PARAM_EXCLUSIVE, exclusive_qnn_scalar);
  param_tensor_names.push_back(exclusive_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(exclusive_param));

  // Add reverse param
  int64_t reverse = node_helper.Get("reverse", static_cast<int64_t>(0));
  Qnn_Scalar_t reverse_qnn_scalar = QNN_SCALAR_INIT;
  reverse_qnn_scalar.dataType = QNN_DATATYPE_BOOL_8;
  reverse_qnn_scalar.bool8Value = static_cast<uint8_t>(reverse == 0 ? 0 : 1);
  QnnParamWrapper reverse_param(node_unit.Index(), node_unit.Name(), QNN_OP_CUMULATIVE_SUM_PARAM_REVERSE, reverse_qnn_scalar);
  param_tensor_names.push_back(reverse_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(reverse_param));

  return ProcessOutputs(qnn_model_wrapper, node_unit,
                        std::move(input_names),
                        std::move(param_tensor_names),
                        logger, do_op_validation, GetQnnOpType(node_unit.OpType()));
}

void CreateCumSumOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<CumSumOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
