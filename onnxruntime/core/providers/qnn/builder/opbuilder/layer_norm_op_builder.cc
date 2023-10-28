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

class LayerNormOpBuilder : public BaseOpBuilder {
 public:
  LayerNormOpBuilder() : BaseOpBuilder("LayerNormOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(LayerNormOpBuilder);

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

Status LayerNormOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                         const NodeUnit& node_unit,
                                         const logging::Logger& logger) const {
  // Also check output type is float for CPU.
  const auto& outputs = node_unit.Outputs();
  ORT_RETURN_IF(outputs.size() > 1, "QNN LayerNorm only support 1 output.");

  // QNN Op validation can also do the same work, but the message is not so clear.
  // Explicit check and provide clear message here
  bool is_npu_backend = IsNpuBackend(qnn_model_wrapper.GetQnnBackendType());
  if (is_npu_backend) {
    std::vector<uint32_t> input_shape;
    const auto& inputs = node_unit.Inputs();
    ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[0].node_arg, input_shape), "Cannot get shape of input 0");
    const size_t input_rank = input_shape.size();
    int32_t default_axis = -1;
    Qnn_Scalar_t axis_qnn_scalar = QNN_SCALAR_INIT;
    ORT_RETURN_IF_ERROR(ProcessAxisAttribute(qnn_model_wrapper, node_unit, axis_qnn_scalar, default_axis));
    ORT_RETURN_IF(static_cast<size_t>(default_axis) != input_rank - 1, "QNN LayerNorm for HTP only support axis with last input dimension");
  }

  return AddToModelBuilder(qnn_model_wrapper, node_unit, logger, true);
}

Status LayerNormOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                       const NodeUnit& node_unit,
                                                       std::vector<std::string>&& input_names,
                                                       const logging::Logger& logger,
                                                       bool do_op_validation) const {
  NodeAttrHelper node_helper(node_unit);
  std::vector<std::string> param_tensor_names;

  const float epsilon = node_helper.Get("epsilon", 1e-05f);  // Default is 1e-05 according to ONNX spec.
  Qnn_Scalar_t epsilon_param = QNN_SCALAR_INIT;
  epsilon_param.dataType = QNN_DATATYPE_FLOAT_32;
  epsilon_param.floatValue = epsilon;
  QnnParamWrapper epsilon_param_wrapper(node_unit.Index(),
                                        node_unit.Name(),
                                        QNN_OP_LAYER_NORM_PARAM_EPSILON,
                                        epsilon_param);
  param_tensor_names.push_back(epsilon_param_wrapper.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(epsilon_param_wrapper));

  std::vector<uint32_t> input_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(node_unit.Inputs()[0].node_arg, input_shape), "Cannot get shape of input 0");
  const size_t input_rank = input_shape.size();
  int32_t default_axis = -1;
  Qnn_Scalar_t axis_qnn_scalar = QNN_SCALAR_INIT;
  ORT_RETURN_IF_ERROR(ProcessAxisAttribute(qnn_model_wrapper, node_unit, axis_qnn_scalar, default_axis));
  size_t axes_rank = input_rank - static_cast<size_t>(default_axis);
  std::vector<uint32_t> axes(axes_rank, 0);
  std::vector<uint32_t> axes_shape{SafeInt<uint32_t>(axes_rank)};
  axes[0] = static_cast<uint32_t>(default_axis);
  for (size_t i = 1; i < axes.size(); ++i) {
    axes[i] = axes[i - 1] + 1;
  }

  QnnParamWrapper axes_param(node_unit.Index(), node_unit.Name(), QNN_OP_LAYER_NORM_PARAM_AXES,
                             std::move(axes_shape), std::move(axes));
  param_tensor_names.push_back(axes_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(axes_param));

  ORT_RETURN_IF_ERROR(ProcessOutputs(qnn_model_wrapper, node_unit,
                                     std::move(input_names),
                                     std::move(param_tensor_names),
                                     logger, do_op_validation, GetQnnOpType(node_unit.OpType())));

  return Status::OK();
}

void CreateLayerNormOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<LayerNormOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
