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

class LRNOpBuilder : public BaseOpBuilder {
 public:
  LRNOpBuilder() : BaseOpBuilder("LRNOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(LRNOpBuilder);

  Status IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                       const NodeUnit& node_unit,
                       const logging::Logger& logger,
                       bool is_quantized_model) const override final ORT_MUST_USE_RESULT;

 protected:
  Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                     const NodeUnit& node_unit,
                                     std::vector<std::string>&& input_names,
                                     const logging::Logger& logger,
                                     bool is_quantized_model,
                                     bool do_op_validation) const override ORT_MUST_USE_RESULT;
};

// Instance normalization op is sensitive to data layout.
// The nodes from 1st call of GetCapability do not get layout transformer applied, so their shapes are still NCHW.
// The nodes from 2nd call of GetCapability get their layout transformed to NHWC.
// Therefore, we need to check the node domain to determine if the layout has been transformed.
Status LRNOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                            const NodeUnit& node_unit,
                                            const logging::Logger& logger,
                                            bool is_quantized_model) const {
  ORT_UNUSED_PARAMETER(logger);

  const auto float_elem_type = ONNX_NAMESPACE::Utils::DataTypeUtils::ToType("float");

  // Check input type is float for CPU.
  const auto& inputs = node_unit.Inputs();
  ONNX_NAMESPACE::DataType input_data_type = inputs[0].node_arg.Type();
  if (!is_quantized_model && input_data_type != float_elem_type) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN LRN data type " + *input_data_type + " is not supported in CPU backend.");
  }

  // Also check output type is float for CPU.
  const auto& outputs = node_unit.Outputs();
  ONNX_NAMESPACE::DataType output_data_type = outputs[0].node_arg.Type();
  if (!is_quantized_model && output_data_type != float_elem_type) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN LRN data type " + *output_data_type + " is not supported in CPU backend.");
  }

  std::vector<uint32_t> input_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[0].node_arg, input_shape), "Cannot get shape of input 0");
  const size_t input_rank = input_shape.size();

  if (input_rank <= 2 || input_rank > 4) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN LRN only supports input ranks of size 3 or 4.");
  }

  return Status::OK();
}

Status LRNOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                          const NodeUnit& node_unit,
                                                          std::vector<std::string>&& input_names,
                                                          const logging::Logger& logger,
                                                          bool is_quantized_model,
                                                          bool do_op_validation) const {
  /*
  NodeAttrHelper node_helper(node_unit);
  std::vector<std::string> param_tensor_names;

  const float epsilon = node_helper.Get("epsilon", 1e-05f);  // Default is 1e-05 according to ONNX spec.
  Qnn_Scalar_t epsilon_param = QNN_SCALAR_INIT;
  epsilon_param.dataType = QNN_DATATYPE_FLOAT_32;
  epsilon_param.floatValue = epsilon;
  QnnParamWrapper epsilon_param_wrapper(node_unit.Index(),
                                        node_unit.Name(),
                                        qnn_def::epsilon,
                                        epsilon_param);
  param_tensor_names.push_back(epsilon_param_wrapper.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(epsilon_param_wrapper));

  ORT_RETURN_IF_ERROR(ProcessOutputs(qnn_model_wrapper, node_unit,
                                     std::move(input_names),
                                     std::move(param_tensor_names),
                                     logger, is_quantized_model, do_op_validation));*/

  return Status::OK();
}

void CreateLRNOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<LRNOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
