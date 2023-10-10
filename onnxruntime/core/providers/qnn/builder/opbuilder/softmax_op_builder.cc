// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/framework/tensorprotoutils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/common/safeint.h"
#include "onnx/defs/data_type_utils.h"

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
  Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                     const NodeUnit& node_unit,
                                     std::vector<std::string>&& input_names,
                                     const logging::Logger& logger,
                                     bool do_op_validation) const override ORT_MUST_USE_RESULT;
};

static int32_t GetDefaultAxisAttribute(const std::string& op_type, int opset_version) {
  if (op_type == "Softmax" || op_type == "LogSoftmax") {
    // Default axis changed from 1 to -1 in opset 13.
    return opset_version < 13 ? 1 : -1;
  }

  return 0;
}

Status SoftmaxOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                       const NodeUnit& node_unit,
                                       const logging::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);
  const std::string& op_type = node_unit.OpType();

  int32_t axis = GetDefaultAxisAttribute(op_type, node_unit.SinceVersion());
  Qnn_Scalar_t axis_qnn_scalar = QNN_SCALAR_INIT;
  ORT_RETURN_IF_ERROR(ProcessAxisAttribute(qnn_model_wrapper, node_unit, axis_qnn_scalar, axis));
  std::vector<uint32_t> input_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(node_unit.Inputs()[0].node_arg, input_shape),
                    "QNN EP: Cannot get shape for Softmax input");
  ORT_RETURN_IF(axis != static_cast<int32_t>(input_shape.size() - 1),
                "QNN ", op_type.c_str(), " only supports an `axis` attribute equal to input_rank-1 (or -1)");

  return AddToModelBuilder(qnn_model_wrapper, node_unit, logger, true);
}

Status SoftmaxOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                     const NodeUnit& node_unit,
                                                     std::vector<std::string>&& input_names,
                                                     const logging::Logger& logger,
                                                     bool do_op_validation) const {
  const std::string& op_type = node_unit.OpType();

  int32_t default_axis = GetDefaultAxisAttribute(op_type, node_unit.SinceVersion());
  Qnn_Scalar_t axis_qnn_scalar = QNN_SCALAR_INIT;
  ORT_RETURN_IF_ERROR(ProcessAxisAttribute(qnn_model_wrapper, node_unit, axis_qnn_scalar, default_axis));
  QnnParamWrapper axis_param(node_unit.Index(), node_unit.Name(), QNN_OP_SOFTMAX_PARAM_AXIS, axis_qnn_scalar);

  std::vector<std::string> param_tensor_names;
  param_tensor_names.push_back(axis_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(axis_param));

  return ProcessOutputs(qnn_model_wrapper, node_unit,
                        std::move(input_names),
                        std::move(param_tensor_names),
                        logger, do_op_validation, GetQnnOpType(op_type));
}

void CreateSoftmaxOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<SoftmaxOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
