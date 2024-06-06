// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstring>
#include <vector>
#include "core/framework/float16.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"

#include "QnnOpDef.h"
#include "QnnTypes.h"

namespace onnxruntime {
namespace qnn {

class HardSigmoidOpBuilder : public BaseOpBuilder {
 public:
  HardSigmoidOpBuilder() : BaseOpBuilder("HardSigmoidOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(HardSigmoidOpBuilder);

  Status IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                       const NodeUnit& node_unit,
                       const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

 protected:
  Status ProcessInputs(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit,
                       const logging::Logger& logger,
                       std::vector<std::string>& input_names,
                       bool do_op_validation = false) const override ORT_MUST_USE_RESULT;

  Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                     const NodeUnit& node_unit,
                                     std::vector<std::string>&& input_names,
                                     const logging::Logger& logger,
                                     bool do_op_validation) const override ORT_MUST_USE_RESULT;

 private:
  static const OnnxAttrInfo<float> onnx_alpha_attr;
  static const OnnxAttrInfo<float> onnx_beta_attr;
};

const OnnxAttrInfo<float> HardSigmoidOpBuilder::onnx_alpha_attr = {"alpha", 0.2f};
const OnnxAttrInfo<float> HardSigmoidOpBuilder::onnx_beta_attr = {"beta", 0.5};

// HardSigmoid is not natively supported by QNN. This builder must decompose HardSigmoid into
// HardSigmoid(X) = max(0, min(1, alpha*X + beta)). This is only valid for float (non-quantized) HardSigmoid ops
// because we don't compute internal quantization parameters (scale/zp) for any new nodes.
Status HardSigmoidOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                           const NodeUnit& node_unit,
                                           const logging::Logger& logger) const {
  ORT_RETURN_IF_NOT(node_unit.UnitType() == NodeUnit::Type::SingleNode,
                    "QNN EP does not support quantized (QDQ) HardSigmoid");

  const auto& inputs = node_unit.Inputs();
  ORT_RETURN_IF(inputs.size() != 1, "HardSigmoid operator must have 1 input.");
  const auto& input = inputs[0];

  int32_t onnx_data_type = 0;
  ORT_RETURN_IF_ERROR(utils::GetOnnxTensorElemDataType(input.node_arg, onnx_data_type));

  const bool is_float_type = (onnx_data_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) ||
                             (onnx_data_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
  ORT_RETURN_IF_NOT(is_float_type, "QNN EP only supports HardSigmoid with float/float16 inputs");

  return AddToModelBuilder(qnn_model_wrapper, node_unit, logger, true);
}

Status HardSigmoidOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                           const NodeUnit& node_unit,
                                           const logging::Logger& logger,
                                           std::vector<std::string>& input_names,
                                           bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(do_op_validation);
  const auto& inputs = node_unit.Inputs();

  return ProcessInput(qnn_model_wrapper, inputs[0], logger, input_names);
}

static Status GetFloatBytes(float f32_val, Qnn_DataType_t qnn_data_type, std::vector<uint8_t>& bytes) {
  switch (qnn_data_type) {
    case QNN_DATATYPE_FLOAT_32: {
      bytes.resize(sizeof(float));
      std::memcpy(bytes.data(), &f32_val, bytes.size());
      break;
    }
    case QNN_DATATYPE_FLOAT_16: {
      bytes.resize(sizeof(MLFloat16));
      const MLFloat16 f16_val(f32_val);
      std::memcpy(bytes.data(), &f16_val, bytes.size());
      break;
    }
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Qnn Data Type: ", qnn_data_type, " is not supported");
  }

  return Status::OK();
}

Status HardSigmoidOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                         const NodeUnit& node_unit,
                                                         std::vector<std::string>&& input_names,
                                                         const logging::Logger& logger,
                                                         bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(logger);
  const auto& onnx_node_name = utils::GetNodeName(node_unit);
  const auto& input = node_unit.Inputs()[0];
  const auto& output = node_unit.Outputs()[0];

  std::vector<uint32_t> input_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(input.node_arg, input_shape), "Cannot get shape of input 0");

  Qnn_DataType_t qnn_data_type = QNN_DATATYPE_FLOAT_32;
  ORT_RETURN_IF_ERROR(utils::GetQnnDataType(false /*is_quantized*/, input.node_arg.TypeAsProto(), qnn_data_type));

  NodeAttrHelper node_helper(node_unit);

  //
  // Create Mul node.
  //

  std::string alpha_input_name = MakeString("ort_qnn_ep_", onnx_node_name, "_HardSigmoid_Mul_alpha");
  std::vector<uint8_t> alpha_bytes;
  ORT_RETURN_IF_ERROR(GetFloatBytes(GetOnnxAttr(node_helper, onnx_alpha_attr), qnn_data_type, alpha_bytes));

  QnnTensorWrapper alpha_input(alpha_input_name,
                               QNN_TENSOR_TYPE_STATIC,
                               qnn_data_type,
                               QnnQuantParamsWrapper(),
                               {1},  // shape
                               std::move(alpha_bytes));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(alpha_input)), "Failed to add alpha input tensor.");

  std::string mul_output_name = MakeString("ort_qnn_ep_", onnx_node_name, "_HardSigmoid_Mul_output");
  std::string mul_node_name = MakeString("ort_qnn_ep_", onnx_node_name, "_HardSigmoid_Mul_node");
  QnnTensorWrapper mul_output(mul_output_name,
                              QNN_TENSOR_TYPE_NATIVE,
                              qnn_data_type,
                              QnnQuantParamsWrapper(),
                              std::vector<uint32_t>(input_shape));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(mul_output)), "Failed to add Mul output tensor.");
  ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(mul_node_name,
                                                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                    QNN_OP_ELEMENT_WISE_MULTIPLY,
                                                    {input_names[0], alpha_input_name},  // input names
                                                    {mul_output_name},                   // output names
                                                    {},
                                                    do_op_validation),
                    "Failed to add Mul node.");

  //
  // Create Add node.
  //

  std::string beta_input_name = MakeString("ort_qnn_ep_", onnx_node_name, "_HardSigmoid_Mul_beta");
  std::vector<uint8_t> beta_bytes;
  ORT_RETURN_IF_ERROR(GetFloatBytes(GetOnnxAttr(node_helper, onnx_beta_attr), qnn_data_type, beta_bytes));

  QnnTensorWrapper beta_input(beta_input_name,
                              QNN_TENSOR_TYPE_STATIC,
                              qnn_data_type,
                              QnnQuantParamsWrapper(),
                              {1},  // shape
                              std::move(beta_bytes));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(beta_input)), "Failed to add beta input tensor.");

  std::string add_output_name = MakeString("ort_qnn_ep_", onnx_node_name, "_HardSigmoid_Add_output");
  std::string add_node_name = MakeString("ort_qnn_ep_", onnx_node_name, "_HardSigmoid_Add_node");
  QnnTensorWrapper add_output(add_output_name,
                              QNN_TENSOR_TYPE_NATIVE,
                              qnn_data_type,
                              QnnQuantParamsWrapper(),
                              std::vector<uint32_t>(input_shape));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(add_output)), "Failed to add Add output tensor.");
  ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(add_node_name,
                                                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                    QNN_OP_ELEMENT_WISE_ADD,
                                                    {mul_output_name, beta_input_name},  // input names
                                                    {add_output_name},                   // output names
                                                    {},
                                                    do_op_validation),
                    "Failed to add Add node.");

  //
  // Create ReluMinMax node.
  //

  std::vector<std::string> param_tensor_names;

  // Parameter 'min_value'
  {
    Qnn_Scalar_t min_value = QNN_SCALAR_INIT;
    min_value.dataType = QNN_DATATYPE_FLOAT_32;
    min_value.floatValue = 0.0f;

    QnnParamWrapper qnn_param(node_unit.Index(), node_unit.Name(), QNN_OP_RELU_MIN_MAX_PARAM_MIN_VALUE, min_value);
    param_tensor_names.push_back(qnn_param.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(qnn_param));
  }

  // Parameter 'max_value'
  {
    Qnn_Scalar_t max_value = QNN_SCALAR_INIT;
    max_value.dataType = QNN_DATATYPE_FLOAT_32;
    max_value.floatValue = 1.0f;

    QnnParamWrapper qnn_param(node_unit.Index(), node_unit.Name(), QNN_OP_RELU_MIN_MAX_PARAM_MAX_VALUE, max_value);
    param_tensor_names.push_back(qnn_param.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(qnn_param));
  }

  const std::string& output_name = output.node_arg.Name();
  std::string relu_min_max_node_name = MakeString("ort_qnn_ep_", onnx_node_name, "_HardSigmoid_ReluMinMax_node");
  QnnTensorWrapper output_tensor(output_name,
                                 qnn_model_wrapper.GetTensorType(output_name),
                                 qnn_data_type,
                                 QnnQuantParamsWrapper(),
                                 std::vector<uint32_t>(input_shape));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensor)), "Failed to add output tensor.");
  ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(relu_min_max_node_name,
                                                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                    QNN_OP_RELU_MIN_MAX,
                                                    {add_output_name},  // input names
                                                    {output_name},      // output names
                                                    std::move(param_tensor_names),
                                                    do_op_validation),
                    "Failed to add ReluMinMax node.");

  return Status::OK();
}

void CreateHardSigmoidOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<HardSigmoidOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
