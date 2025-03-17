// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/common/safeint.h"
#include "onnx/defs/data_type_utils.h"

#include "QnnOpDef.h"  // From QNN SDK: contains QNN constants (e.g., op names, param values).

namespace onnxruntime {
namespace qnn {

class LRNOpBuilder : public BaseOpBuilder {
 public:
  LRNOpBuilder() : BaseOpBuilder("LRNOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(LRNOpBuilder);

  Status IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                       const NodeUnit& node_unit,
                       const logging::Logger& logger) const override final ORT_MUST_USE_RESULT;

 protected:
  Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                     const NodeUnit& node_unit,
                                     std::vector<std::string>&& input_names,
                                     const logging::Logger& logger,
                                     bool do_op_validation) const override ORT_MUST_USE_RESULT;

 private:
  static const OnnxAttrInfo<float> onnx_alpha_attr;
  static const OnnxAttrInfo<float> onnx_beta_attr;
  static const OnnxAttrInfo<float> onnx_bias_attr;
  static const OnnxAttrInfo<int64_t> onnx_size_attr;
};

const OnnxAttrInfo<float> LRNOpBuilder::onnx_alpha_attr = {"alpha", 0.0001f};
const OnnxAttrInfo<float> LRNOpBuilder::onnx_beta_attr = {"beta", 0.75f};
const OnnxAttrInfo<float> LRNOpBuilder::onnx_bias_attr = {"bias", 1.0f};
const OnnxAttrInfo<int64_t> LRNOpBuilder::onnx_size_attr = {"size", 0};

// The LRN operator is layout sensitive. ONNX LRN has layout NCHW, but QNN requires layout NHWC.
// The nodes from 1st call of GetCapability do not get layout transformer applied, so their shapes are still NCHW.
// The nodes from 2nd call of GetCapability get their layout transformed to NHWC.
// Therefore, we need to check the node domain to determine if the layout has been transformed.
Status LRNOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                   const NodeUnit& node_unit,
                                   const logging::Logger& logger) const {
  if (node_unit.Domain() == kMSInternalNHWCDomain) {
    return AddToModelBuilder(qnn_model_wrapper, node_unit, logger, true);
  }

  const auto& inputs = node_unit.Inputs();
  const auto& outputs = node_unit.Outputs();

  ORT_RETURN_IF(inputs.size() != 1, "QNN EP: LRN operator must have 1 input.");
  ORT_RETURN_IF(outputs.size() != 1, "QNN EP: LRN operator must have 1 output.");

  const auto& input = inputs[0];
  const auto& output = outputs[0];
  // Check input type is float for CPU. Can't use Qnn Op validation API since it's before layout transformation
  ORT_RETURN_IF_ERROR(DataTypeCheckForCpuBackend(qnn_model_wrapper, inputs[0].node_arg.Type()));

  // Check that the input and output have the same shape.
  std::vector<uint32_t> input_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(input.node_arg, input_shape), "Cannot get shape of input 0");
  const size_t input_rank = input_shape.size();

  ORT_RETURN_IF(input_rank <= 2 || input_rank > 4, "QNN EP: LRN operator only supports input ranks of size 3 or 4.");

  std::vector<uint32_t> output_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(output.node_arg, output_shape), "Cannot get shape of output 0");

  ORT_RETURN_IF(output_shape != input_shape, "QNN EP: LRN operator's output must have the same shape as the input.");

  NodeAttrHelper node_helper(node_unit);

  // 'size' attribute must be odd and > 0.
  const int64_t onnx_size = GetOnnxAttr(node_helper, onnx_size_attr);
  ORT_RETURN_IF(onnx_size % 2 == 0, "QNN EP: LRN operator's size attribute must be odd.");

  // 'alpha' attribute must be > 0.0f.
  const float onnx_alpha = GetOnnxAttr(node_helper, onnx_alpha_attr);
  ORT_RETURN_IF(onnx_alpha <= 0.0f, "QNN EP: LRN operator's alpha attribute must be greater than zero.");

  // 'alpha' attribute must be > 0.0f.
  const float onnx_beta = GetOnnxAttr(node_helper, onnx_beta_attr);
  ORT_RETURN_IF(onnx_beta <= 0.0f, "QNN EP: LRN operator's beta attribute must be greater than zero.");

  return Status::OK();
}

Status LRNOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                 const NodeUnit& node_unit,
                                                 std::vector<std::string>&& input_names,
                                                 const logging::Logger& logger,
                                                 bool do_op_validation) const {
  std::vector<std::string> param_tensor_names;
  NodeAttrHelper node_helper(node_unit);

  const int64_t onnx_size = GetOnnxAttr(node_helper, onnx_size_attr);

  // Parameter 'radius'
  {
    Qnn_Scalar_t qnn_radius = QNN_SCALAR_INIT;
    qnn_radius.dataType = QNN_DATATYPE_INT_32;
    qnn_radius.int32Value = SafeInt<int32_t>((onnx_size - 1) / 2);  // Convert ONNX size into QNN radius.

    QnnParamWrapper qnn_param(node_unit.Index(), node_unit.Name(), QNN_OP_LRN_PARAM_RADIUS, qnn_radius);
    param_tensor_names.push_back(qnn_param.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(qnn_param));
  }

  // Parameter 'alpha'
  {
    float onnx_alpha = GetOnnxAttr(node_helper, onnx_alpha_attr);
    Qnn_Scalar_t qnn_alpha = QNN_SCALAR_INIT;
    qnn_alpha.dataType = QNN_DATATYPE_FLOAT_32;
    qnn_alpha.floatValue = onnx_alpha / static_cast<float>(onnx_size);  // QNN doesn't scale alpha by size.

    QnnParamWrapper qnn_param(node_unit.Index(), node_unit.Name(), QNN_OP_LRN_PARAM_ALPHA, qnn_alpha);
    param_tensor_names.push_back(qnn_param.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(qnn_param));
  }

  // Parameter 'beta'
  {
    Qnn_Scalar_t qnn_beta = QNN_SCALAR_INIT;
    qnn_beta.dataType = QNN_DATATYPE_FLOAT_32;
    qnn_beta.floatValue = GetOnnxAttr(node_helper, onnx_beta_attr);

    QnnParamWrapper qnn_param(node_unit.Index(), node_unit.Name(), QNN_OP_LRN_PARAM_BETA, qnn_beta);
    param_tensor_names.push_back(qnn_param.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(qnn_param));
  }

  // Parameter 'bias'
  {
    Qnn_Scalar_t qnn_bias = QNN_SCALAR_INIT;
    qnn_bias.dataType = QNN_DATATYPE_FLOAT_32;
    qnn_bias.floatValue = GetOnnxAttr(node_helper, onnx_bias_attr);

    QnnParamWrapper qnn_param(node_unit.Index(), node_unit.Name(), QNN_OP_LRN_PARAM_BIAS, qnn_bias);
    param_tensor_names.push_back(qnn_param.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(qnn_param));
  }

  // Parameter 'region'
  {
    Qnn_Scalar_t qnn_region = QNN_SCALAR_INIT;
    qnn_region.dataType = QNN_DATATYPE_UINT_32;
    qnn_region.uint32Value = QNN_OP_LRN_REGION_ACROSS_CHANNEL;  // ONNX's LRN only supports "across channel".

    QnnParamWrapper qnn_param(node_unit.Index(), node_unit.Name(), QNN_OP_LRN_PARAM_REGION, qnn_region);
    param_tensor_names.push_back(qnn_param.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(qnn_param));
  }

  return ProcessOutputs(qnn_model_wrapper, node_unit, std::move(input_names), std::move(param_tensor_names),
                        logger, do_op_validation, GetQnnOpType(node_unit.OpType()));
}

void CreateLRNOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<LRNOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
